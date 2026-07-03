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

from . import importer

# Monotonic token so each referenced import's freshly built debug materials get
# a unique name. The importer rebuilds a debug material by REMOVING any existing
# datablock of the same name (recipe_loader.build_pbr_debug_material); importing
# the same tire/rim i3d twice would otherwise delete the first wheel's material
# and null its mesh slots. Renaming right after each import keeps them. Pairing
# for the Material-Switch operator is by IDProperties (material_id/import_uuid/
# kind), not name, so renaming is safe.
_REF_DEBUG_CTR = itertools.count(1)


def _prefs_import_settings():
    """Read the user's import defaults from the add-on preferences.

    A direct bpy.ops call bypasses the operator's invoke (which seeds defaults
    from prefs), so a sub-import would otherwise use the bare operator defaults
    (e.g. export materials instead of debug). Pass these explicitly so referenced
    parts match the vehicle import.
    """
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        return {
            "apply_axis_correction": prefs.apply_axis_correction_default,
            "auto_hide_invisible_shapes": prefs.auto_hide_invisible_shapes_default,
            "build_pbr_debug_materials": prefs.build_pbr_debug_materials_default,
            "attach_debug_materials_to_mesh": prefs.attach_debug_materials_to_mesh_default,
            "fs25_data_base": prefs.fs25_data_base or None,
        }
    except Exception:
        return {}


def _protect_debug_materials(new_objects):
    """Rename freshly built ``*_pbr_debug*`` materials so a later import of the
    same i3d will not remove them (which would null this import's mesh slots)."""
    for o in new_objects:
        if o.type != "MESH":
            continue
        for m in o.data.materials:
            if m and "_pbr_debug" in m.name and not m.get("_i3d_ref_renamed"):
                m["_i3d_ref_renamed"] = 1
                m.name = "%s__ref%d" % (m.name, next(_REF_DEBUG_CTR))


def _share_debug_materials(new_objects, registry, source_key=""):
    """Make freshly built ``*_pbr_debug*`` materials SHARE one datablock across
    referenced imports. ``registry`` maps a base key to the kept datablock. The
    first import of a given material keeps it (renamed so a later import's
    name-based rebuild can't remove it); every later import of the same
    material is pointed at the shared datablock and its fresh duplicate
    dropped. So all wheels using the same tire/rim material share it - one
    rim-colour change then colours every wheel, matching the game.

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
            m = md.materials[i]
            if not m or "_pbr_debug" not in m.name:
                continue
            base = m.get("_i3d_debug_base") or "%s|%s" % (source_key, m.name)
            shared = registry.get(base)
            if shared is None:
                m["_i3d_debug_base"] = base
                m["_i3d_ref_shared"] = 1
                # readable name (key may contain the source path - keep it in
                # the IDProperty only)
                if not m.name.endswith("__shared"):
                    m.name = m.name + "__shared"
                registry[base] = m
            elif shared is not m:
                md.materials[i] = shared
                dups.add(m)
    for m in dups:
        if m.users == 0:
            bpy.data.materials.remove(m)


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
