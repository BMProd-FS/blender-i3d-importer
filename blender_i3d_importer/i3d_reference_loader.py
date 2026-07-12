"""Load an external (referenced) i3d - e.g. a wheel/tire/rim - into the current
scene and optionally parent it under a node.

Foundation for wheel/tire loading. The Giants vehicle XML references tire, rim
and hub i3ds that are NOT part of the vehicle i3d. This imports one such file
via the normal import pipeline as a **standalone top-level import** (not nested
reference recursion, which is unstable and crashed earlier) and returns the
created top-level objects so the caller can place them.
"""

import itertools
import os

import bpy

from . import i3d_config_preview, importer

# Monotonic token so each referenced import's freshly built debug materials get
# a unique name. The importer rebuilds a debug material by REMOVING any existing
# datablock of the same name (recipe_loader.build_pbr_debug_material); importing
# the same tire/rim i3d twice would otherwise delete the first wheel's material
# and null its mesh slots. Renaming right after each import keeps them. Pairing
# for the Material-Switch operator is by IDProperties (material_id/import_uuid/
# kind), not name, so renaming is safe.
_REF_DEBUG_CTR = itertools.count(1)


# Sticky material mode of the scene: which kind of material ('debug' / 'export')
# newly imported meshes get attached to. Seeded by the .i3d import operator from
# its "Attach debug materials to mesh" option, then overwritten by every
# scene-wide material switch in the N-Panel. Parts loaded LATER (wheels, rims,
# referenced i3ds) follow it - without this they always came in with the import
# option's kind, so wheels loaded after a switch to debug stayed on their export
# material and no longer swapped along (white tires / weights).
MATERIAL_MODE_KEY = "_i3d_material_mode"


def get_material_mode(scene=None):
    """'debug' / 'export', or None when the scene has no choice recorded yet
    (older blend files -> fall back to the add-on preferences)."""
    sc = scene if scene is not None else bpy.context.scene
    mode = sc.get(MATERIAL_MODE_KEY) if sc is not None else None
    return mode if mode in ("debug", "export") else None


def set_material_mode(scene, kind):
    """Record the scene's material mode. Only scene-wide choices are recorded -
    a switch limited to the selection says nothing about what later imports
    should use."""
    if scene is not None and kind in ("debug", "export"):
        scene[MATERIAL_MODE_KEY] = kind


def apply_material_mode(objects, scene=None):
    """Force every i3d material slot of *objects* to the scene's material mode.

    The import option only decides which kind gets ATTACHED at import time. Wheel
    parts are additionally BAKED from the frozen pre-bake mesh datablock
    (_i3d_prebake_mesh), which preserves the material state of the original
    import - so a rim/connector baked after the user switched to debug came back
    with its export material (white rims / white twin connectors). Re-asserting
    the mode on the finished objects covers every route into a mesh slot,
    whether imported, duplicated or baked.

    Returns the number of swapped slots.
    """
    mode = get_material_mode(scene)
    if mode is None:
        return 0
    swapped = 0
    for o in objects:
        if o.type != "MESH" or not o.data:
            continue
        for slot in o.material_slots:
            cur = slot.material
            if cur is None or cur.get("_i3d_material_kind") not in ("debug", "export"):
                continue
            if cur.get("_i3d_material_kind") == mode:
                continue
            want = i3d_config_preview._material_pair(cur, mode)
            if want is not None and want is not cur:
                slot.material = want
                swapped += 1
    return swapped


def _prefs_import_settings():
    """Read the user's import defaults from the add-on preferences.

    A direct bpy.ops call bypasses the operator's invoke (which seeds defaults
    from prefs), so a sub-import would otherwise use the bare operator defaults
    (e.g. export materials instead of debug). Pass these explicitly so referenced
    parts match the vehicle import.

    The scene's sticky material mode (see MATERIAL_MODE_KEY) overrides the
    preference: it reflects what the user last chose for THIS scene, which is
    what a part loaded now has to match.
    """
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        settings = {
            "apply_axis_correction": prefs.apply_axis_correction_default,
            "auto_hide_invisible_shapes": prefs.auto_hide_invisible_shapes_default,
            "build_pbr_debug_materials": prefs.build_pbr_debug_materials_default,
            "attach_debug_materials_to_mesh": prefs.attach_debug_materials_to_mesh_default,
            "fs25_data_base": prefs.fs25_data_base or None,
        }
    except Exception:
        return {}
    mode = get_material_mode()
    if mode is not None:
        # debug mode needs debug materials to exist at all
        settings["attach_debug_materials_to_mesh"] = (
            mode == "debug" and bool(settings["build_pbr_debug_materials"]))
    return settings


def _pair_of(slot_mat):
    """(debug, export) pair for a mesh-slot material, or (None, None).

    The slot can hold EITHER kind, depending on the
    ATTACH_DEBUG_MATERIALS_TO_MESH preference - so never assume the attached
    material is the debug one. Resolving via the pair (id + import uuid) is what
    makes both passes below work in either mode; the previous
    ``"_pbr_debug" in m.name`` check on the slot material silently did nothing
    when the export material was attached, and recipe_loader then removed the
    unprotected debug material of an earlier import on the next name collision
    (front tires left with an export material that had no debug counterpart).
    """
    dbg = i3d_config_preview.debug_pair(slot_mat)
    if dbg is None:
        return None, None
    return dbg, i3d_config_preview._export_pair(slot_mat)


def _drop_unused(mats):
    """Remove materials that nothing references any more.

    The fake user (set on every material we build so it survives a save) is
    counted in ``users``, so it has to be cleared before the check - otherwise
    the duplicates pile up in the blend file. But it MUST be restored when the
    material turns out to be still referenced (a mesh datablock can hold a
    material without any object using it - e.g. the pristine pre-bake wheel
    meshes): a datablock left with users=0 AND no fake user is silently dropped
    on the next save/reload.
    """
    for m in mats:
        if m is None:
            continue
        had_fake = m.use_fake_user
        m.use_fake_user = False
        if m.users == 0:
            bpy.data.materials.remove(m)
        elif had_fake:
            m.use_fake_user = True


def _protect_debug_materials(new_objects):
    """Rename this import's debug materials so a later import of the same i3d
    cannot remove them (recipe_loader removes a same-named material and rebuilds
    it) - which would leave this import's export materials without a partner."""
    for o in new_objects:
        if o.type != "MESH":
            continue
        for slot_mat in o.data.materials:
            dbg, _exp = _pair_of(slot_mat)
            if dbg is not None and not dbg.get("_i3d_ref_renamed"):
                dbg["_i3d_ref_renamed"] = 1
                dbg.name = "%s__ref%d" % (dbg.name, next(_REF_DEBUG_CTR))


def _share_debug_materials(new_objects, registry, source_key=""):
    """Make freshly built materials SHARE one datablock across referenced
    imports. ``registry`` maps a base key to the kept DEBUG datablock; the kept
    export material is derived from it via the pair, so both kinds are shared in
    lockstep and the mesh slot keeps whichever kind it currently holds.

    The first import of a given material keeps its pair (the debug material is
    renamed so a later import's name-based rebuild can't remove it); every later
    import is pointed at the shared datablock and its fresh duplicates dropped.
    So all wheels using the same tire/rim material share it - one rim-colour
    change then colours every wheel, matching the game.

    The base key is ``<source i3d path>|<material name>``, NOT the material
    name alone: different rim i3ds all name their materials identically
    (rim001/rim006/rim009 each have a ``rim_inner_mat``) but carry DIFFERENT
    normal/detail textures - keying by name alone made rim006 render with
    rim009's normal map (Vestrum130 'UV verschoben', #14). Sharing is
    therefore per source file; repeat imports of the SAME file still share.
    """
    dups = set()
    for o in new_objects:
        if o.type != "MESH":
            continue
        md = o.data
        for i in range(len(md.materials)):
            slot_mat = md.materials[i]
            dbg, exp = _pair_of(slot_mat)
            if dbg is None:
                continue
            base = dbg.get("_i3d_debug_base") or "%s|%s" % (source_key, dbg.name)
            shared_dbg = registry.get(base)
            if shared_dbg is None:
                dbg["_i3d_debug_base"] = base
                # readable name (key may contain the source path - keep it in
                # the IDProperty only)
                if not dbg.name.endswith("__shared"):
                    dbg.name = dbg.name + "__shared"
                registry[base] = dbg
                continue
            if shared_dbg is dbg:
                continue
            # keep the kind the slot currently holds - swapping a debug material
            # into a slot that holds an export material (or vice versa) would
            # break the re-export / the debug view
            if slot_mat is dbg:
                want = shared_dbg
            else:
                want = i3d_config_preview._export_pair(shared_dbg)
            if want is None or want is slot_mat:
                continue
            md.materials[i] = want
            dups.add(dbg)
            if exp is not None:
                dups.add(exp)
    _drop_unused(dups)


def import_referenced_i3d(filepath, parent=None, tags=None, report=None,
                          settings=None, debug_registry=None):
    """Import *filepath* and return its new top-level objects.

    parent: if given (an object), the new top-level objects are parented to it,
        keeping their world transform (precise placement is the caller's job).
    tags: optional dict of custom properties to stamp on every new object.
    settings: optional dict of import_i3d kwargs; if None, the user's add-on
        preferences are used so the part matches the vehicle import.
    debug_registry: optional dict; when given, freshly built debug materials are
        SHARED across imports via this registry (base name -> datablock) instead
        of merely renamed. Pass the same dict for all parts that should share.
    Returns [] on failure (the importer error is reported, never raised).
    """
    if settings is None:
        settings = _prefs_import_settings()
    # Referenced parts (wheels/rims) must never move the user's viewport, and
    # must not re-point the Giants exporter's export file to their own filename
    # (that field should only follow the manual vehicle import).
    settings = dict(settings)
    settings["frame_view"] = False
    settings["configure_exporter"] = False
    before = set(bpy.data.objects)
    try:
        importer.import_i3d(filepath, report=report, **settings)
    except Exception as exc:
        if report is not None:
            report("WARNING", "Referenced i3d failed (%s): %r" % (filepath, exc))
        return []

    new = [o for o in bpy.data.objects if o not in before]
    if tags:
        for o in new:
            for k, v in tags.items():
                o[k] = v

    # Mark this import's collection so empty leftovers (after wheel/brand swaps)
    # can be purged. The importer stamps obj['_i3d_import_id'] = collection.name.
    for o in new:
        iid = o.get("_i3d_import_id")
        col = bpy.data.collections.get(iid) if iid else None
        if col is not None:
            col["_i3d_ref_import"] = 1
            break

    if debug_registry is not None:
        _share_debug_materials(new, debug_registry,
                               os.path.normcase(os.path.normpath(filepath)))
    else:
        _protect_debug_materials(new)

    roots = [o for o in new if o.parent is None or o.parent not in new]
    if parent is not None:
        for o in roots:
            world = o.matrix_world.copy()
            o.parent = parent
            o.matrix_world = world
    return roots
