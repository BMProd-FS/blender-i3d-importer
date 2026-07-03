"""Load and place FS25 wheels (tire + rim) onto a vehicle's drive nodes.

Reproduces the game's wheel-visual logic (dataS/scripts/vehicles/wheels/
WheelVisualPart.lua), verified against the source:

* Side handling is by NODE SELECTION, not geometry: the rim i3d ships pre-built
  left/right variants; the tire xml picks ``nodeLeft`` vs ``nodeRight`` (the
  ``outerRim`` uses the shared ``node``). No mirroring/negative scale.
* Rim size is ABSOLUTE scale: the rim mesh is normalized to 1 metre, so
  ``setScale(inchToM(width), inchToM(diam), inchToM(diam))`` from widthAndDiam.
* X position is ``offset * (isLeft and 1 or -1)``.

The tire i3d is dimension-specific, linked at local 0 with no geometric scale.

Performance: each UNIQUE tire/rim i3d is imported only once (importing reads the
shape binary and builds PBR debug materials - the slow part). The wheels are then
placed as linked-data duplicates of those templates, so a 4-wheel vehicle costs
3 imports (front tire, rear tire, rim) instead of 8.
"""

import math
import xml.etree.ElementTree as ET

import bpy
import mathutils

from . import i3d_reference_loader, i3d_wheel_resolver

INCH = 0.0254


def _descendants(o):
    out = []
    for c in o.children:
        out.append(c)
        out += _descendants(c)
    return out


def _wd_scale(width_diam):
    """"23 38" -> (0.5842, 0.9652) absolute Blender scale, or None."""
    if not width_diam:
        return None
    try:
        p = width_diam.split()
        return (float(p[0]) * INCH, float(p[1]) * INCH)
    except (ValueError, IndexError):
        return None


def _parse_scale(scale_str):
    """"0.6 1 1" -> (0.6, 1.0, 1.0), or None."""
    if not scale_str:
        return None
    try:
        v = [float(x) for x in scale_str.split()]
    except ValueError:
        return None
    while len(v) < 3:
        v.append(1.0)
    return (v[0], v[1], v[2])


def _link(obj, drive, x=0.0):
    """Seat *obj* on the drive node at local (x, 0, 0), no inherited offset."""
    obj.parent = drive
    obj.matrix_parent_inverse.identity()
    obj.location = (x, 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)


def _dup_subtree(root):
    """Linked-data copy of *root* + all descendants, preserving local transforms.
    Returns the new root (its parent is left as the source's; the caller re-links
    it). Cheap: mesh data is shared, not copied.
    """
    def rec(o):
        c = o.copy()  # shares o.data (linked duplicate)
        for coll in o.users_collection:
            coll.objects.link(c)
        for ch in o.children:
            nc = rec(ch)
            nc.parent = c
            nc.matrix_parent_inverse = ch.matrix_parent_inverse.copy()
            nc.location = ch.location.copy()
            nc.rotation_euler = ch.rotation_euler.copy()
            nc.scale = ch.scale.copy()
        return c
    return rec(root)


def _tag_tree(root, tag):
    for o in [root] + _descendants(root):
        for k, v in tag.items():
            o[k] = v


def _hide_lod_and_mud(root):
    """Hide the tire's far-LOD meshes (lod1, lod2, ...) and the mud overlay so
    only the close-range lod0 shows in the preview. They are flagged as
    GE-invisible so the N-Panel 'Invisible GE-objects' counter includes them and
    'Show All' brings them back before re-export."""
    for o in [root] + _descendants(root):
        nm = o.name.lower()
        is_far_lod = (":lod" in nm and not nm.split(":lod", 1)[1].startswith("0"))
        if is_far_lod or "mudmesh" in nm:
            o["_i3d_invisible_in_ge"] = True
            try:
                o.hide_set(True)
            except Exception:
                o.hide_viewport = True


def _drive_object_map(vehicle_xml_path, import_id):
    """driveNode i3dMapping id -> imported vehicle object (via _i3d_node_path)."""
    try:
        root = ET.parse(vehicle_xml_path).getroot()
    except (ET.ParseError, OSError):
        return {}
    id2path = {m.get("id"): m.get("node") for m in root.iter("i3dMapping")
               if m.get("id") and m.get("node")}
    path2obj = {o.get("_i3d_node_path"): o for o in bpy.data.objects
                if o.get("_i3d_import_id") == import_id}
    return {mid: path2obj[p] for mid, p in id2path.items() if p in path2obj}


def remove_wheels(import_id):
    """Delete all wheel objects previously loaded for *import_id*. Returns count."""
    victims = [o for o in bpy.data.objects
               if o.get("_i3d_wheel_import") == import_id]
    n = len(victims)
    for o in victims:
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass
    return n


def _restore_selection(objs, active):
    """Re-select *objs* and make *active* active, ignoring deleted references."""
    try:
        bpy.ops.object.select_all(action="DESELECT")
    except RuntimeError:
        pass
    for o in objs:
        try:
            o.select_set(True)
        except Exception:
            pass
    try:
        bpy.context.view_layer.objects.active = active
    except Exception:
        pass


def _purge_empty_ref_collections():
    """Remove now-empty collections left behind by referenced wheel/hub imports
    (each re-import creates a fresh collection; old ones empty out on swap)."""
    for c in list(bpy.data.collections):
        try:
            if c.get("_i3d_ref_import") and not c.all_objects:
                bpy.data.collections.remove(c)
        except Exception:
            pass


def _remove_subtree(root):
    for o in [root] + _descendants(root):
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass


def _apply_placement(obj, part):
    """(Re-)seat a placed part on its drive node: parent, local x, scale.
    Idempotent, so a kept part can be re-scaled/re-positioned without re-import."""
    obj.parent = part["drive"]
    obj.matrix_parent_inverse.identity()
    obj.location = (part["locx"], 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, part.get("rotz", 0.0))
    obj.scale = part["scale"] if part["scale"] else (1.0, 1.0, 1.0)
    # Rim parts carry their real width/diam so 'Prepare for Export' can write the
    # widthAndDiam shader parameter (the Giants rim shader dishes the rim from it).
    if part.get("rim_wd"):
        obj["_i3d_rim_wd"] = part["rim_wd"]


def _needed_parts(specs, hubs, drives):
    """Flatten resolved specs + hubs into placeable parts with a stable identity
    key ``drive|role|i3d|node|base_x``. Placement (locx/scale) is kept separate so
    a part that stays (same identity) can be re-scaled without re-import."""
    parts = []

    def key(dn, role, i3d, node, bx):
        return "%s|%s|%s|%s|%.3f" % (dn, role, i3d or "", node or "", bx)

    for s in specs:
        d = drives.get(s.drive_node)
        if d is None:
            continue
        direction = 1.0 if s.is_left else -1.0
        rs = _wd_scale(s.rim_width_diam)
        rs3 = (rs[0], rs[1], rs[1]) if rs else None
        bx = s.base_x
        if s.tire_i3d:
            # rimOffset is applied to the tire ("the wheel itself") as well as
            # the outer rim - only the inner rim stays put (Wheel.lua #rimOffset:
            # "Offset that is only applied to the outer rim and the wheel
            # itself, inner rim stays the same."; WheelVisual.lua adds
            # self.rimOffset to every part except innerRim/additional). Without
            # this the tire stays at the drive node while the rim alone shifts
            # by rimOffset, so they visibly separate.
            parts.append(dict(
                key=key(d.name, "tire", s.tire_i3d, "root", bx), role="tire",
                drive=d, i3d=s.tire_i3d, src_kind="root", root_index=0, node=None,
                locx=bx + s.rim_offset * direction, scale=None, hide_lod=True,
                rotz=(math.pi if s.tire_is_inverted else 0.0)))
        if s.outer_rim_i3d and s.outer_rim_node:
            # A raw <outerRim scale="..."/> override (game gives scale priority
            # over widthAndDiam in WheelVisualPart:setNode) is a real, exact
            # number that must survive export unchanged - so rim_wd is left
            # None for these, which excludes them from _prepare_rims_for_export's
            # widthAndDiam/mesh-splitting pass (see __init__.py) and their scale
            # is never reset back to 1.
            outer_raw_scale = _parse_scale(s.outer_rim_scale)
            parts.append(dict(
                key=key(d.name, "rim_outer", s.outer_rim_i3d, s.outer_rim_node, bx),
                role="rim_outer", drive=d, i3d=s.outer_rim_i3d, src_kind="node",
                root_index=0, node=s.outer_rim_node,
                locx=bx + s.rim_offset * direction,
                scale=(outer_raw_scale if outer_raw_scale else rs3), hide_lod=False,
                rim_wd=(None if outer_raw_scale else s.rim_width_diam),
                bake_wd=s.rim_width_diam, raw_scale=outer_raw_scale,
                rotz=(math.pi if s.outer_rim_is_inverted else 0.0)))
        inner = s.inner_rim_node_left if s.is_left else s.inner_rim_node_right
        inner_i3d = s.inner_rim_i3d or s.outer_rim_i3d
        if s.outer_rim_i3d and inner and not s.drop_inner_rim:
            # inner_rim_offset comes from a <wheel><innerRim offset=../> vehicle-
            # XML override (e.g. Puma NARROW_1500) - the tire xml's own default
            # innerRim has no offset of its own, so this is 0 unless overridden.
            # innerRim is excluded from the general rim_offset addition (see the
            # tire part above), it only ever gets this own offset.
            # inner_rim_i3d/inner_rim_width_diam let an <innerRim> override point
            # at a different file/size than the outer rim (falls back to the
            # outer rim's file/size when not overridden, same as before).
            inner_wd_str = s.inner_rim_width_diam or s.rim_width_diam
            irs = _wd_scale(inner_wd_str)
            irs3 = (irs[0], irs[1], irs[1]) if irs else None
            inner_raw_scale = _parse_scale(s.inner_rim_scale)
            parts.append(dict(
                key=key(d.name, "rim_inner", inner_i3d, inner, bx),
                role="rim_inner", drive=d, i3d=inner_i3d, src_kind="node",
                root_index=0, node=inner,
                locx=bx + s.inner_rim_offset * direction,
                scale=(inner_raw_scale if inner_raw_scale else irs3),
                hide_lod=False,
                rim_wd=(None if inner_raw_scale else inner_wd_str),
                bake_wd=inner_wd_str, raw_scale=inner_raw_scale,
                rotz=(math.pi if s.inner_rim_is_inverted else 0.0)))
        if s.add_i3d:
            anode = s.add_node_left if s.is_left else s.add_node_right
            if anode:
                parts.append(dict(
                    key=key(d.name, "weight", s.add_i3d, anode, bx), role="weight",
                    drive=d, i3d=s.add_i3d, src_kind="node", root_index=0,
                    node=anode, locx=bx + s.add_offset * direction,
                    scale=_parse_scale(s.add_scale), hide_lod=False))
        if s.connector_i3d:
            cnode = (s.connector_node_left if s.is_left
                     else s.connector_node_right)
            if cnode:
                # widthAndDiam.y for the bake: #diameter override, else the
                # additional wheel's rim diameter (outerRim widthAndDiam[1]).
                try:
                    diam_in = (s.connector_diameter
                               or float(s.rim_width_diam.split()[1]))
                except (AttributeError, ValueError, IndexError):
                    diam_in = 0.0
                parts.append(dict(
                    key=key(d.name, "connector", s.connector_i3d, cnode, bx),
                    role="connector", drive=d, i3d=s.connector_i3d,
                    src_kind="node", root_index=0, node=cnode, locx=0.0,
                    scale=_parse_scale(s.connector_scale), hide_lod=False,
                    is_connector=True,
                    conn_width_in=(s.phys_width / INCH if s.phys_width else 0.0),
                    conn_gap_in=s.connector_gap / INCH,
                    conn_diam_in=diam_in,
                    conn_hook_in=s.connector_hook_offset,
                    conn_spo_in=s.connector_start_pos_offset,
                    conn_epo_in=s.connector_end_pos_offset,
                    conn_simple=s.connector_simple,
                    conn_mode=s.connector_mode,
                    conn_start_in=s.connector_start_pos,
                    conn_end_in=s.connector_end_pos,
                    conn_uscale=s.connector_uniform_scale,
                    diam_scale=(rs[1] if rs else 1.0)))

    for h in hubs:
        d = drives.get(h.link_node)
        if d is None or not h.hub_i3d or not h.node:
            continue
        parts.append(dict(
            key="%s|hub|%s|%s|0.000" % (d.name, h.hub_i3d, h.node), role="hub",
            drive=d, i3d=h.hub_i3d, src_kind="node", root_index=0, node=h.node,
            locx=h.offset * (1.0 if h.is_left else -1.0),
            scale=_parse_scale(h.scale), hide_lod=False))
    return parts


def _registry_from_existing(import_id):
    """Debug-material share registry (base name -> material) from the wheels
    already loaded for *import_id*, so newly imported parts reuse the same
    datablocks instead of creating .001/.002 duplicates."""
    reg = {}
    for o in bpy.data.objects:
        if o.get("_i3d_wheel_import") != import_id:
            continue
        for ms in o.material_slots:
            m = ms.material
            if m and "_pbr_debug" in m.name:
                base = m.get("_i3d_debug_base") or m.name
                reg.setdefault(base, m)
    return reg


def _wheel_mat_prop(obj, prop):
    """Shader property of a wheel part's material. The mesh carries the DEBUG
    material, but the importer stores shader properties (customShaderVariation,
    customParameter_<name>) on the paired EXPORT material
    (_apply_material_custom_properties) - so check the debug material first,
    then its export pair (matched via _i3d_material_id + _i3d_import_uuid,
    same pairing as i3d_config_preview._export_pair)."""
    for ms in obj.material_slots:
        m = ms.material
        if not m:
            continue
        v = m.get(prop)
        if v:
            return v
        mid = m.get("_i3d_material_id")
        uuid = m.get("_i3d_import_uuid")
        if mid is None:
            continue
        for em in bpy.data.materials:
            if (em.get("_i3d_material_kind") == "export"
                    and em.get("_i3d_material_id") == mid
                    and em.get("_i3d_import_uuid") == uuid):
                v = em.get(prop)
                if v:
                    return v
                break
    return None


def _conn_statics(obj):
    """numberOfStatics (hook count) from the connector's i3d material custom
    parameter (4 vs 6 per dual001 node variant) - the angular raster the
    shader snaps the hook blocks to. Lives on the EXPORT material, hence
    _wheel_mat_prop. Default 4 = shader default."""
    v = _wheel_mat_prop(obj, "customParameter_numberOfStatics")
    if v:
        try:
            return max(1, int(float(str(v).split()[0])))
        except ValueError:
            pass
    return 4


def _bake_connector_mesh(obj, px, py, pz, pw, diam_m, statics):
    """Bake the game's RIM_DUAL vertex shader (vehicleShader.xml getRimPos)
    into *obj*'s mesh so the connector cage spans the twin gap without the
    runtime shader. Works in the shape's LOCAL i3d space (x = wheel axis,
    yz = radial plane; our importer keeps raw i3d vertex coordinates).

    Axial: control rings modelled at |x| = 1/2/3 m, one-hot masked by vertex
    colours R/G/B, move to 0.5*py / 0.5*py+pz / 0.5*py+pz+pw (metres, from
    connectorPos inches * 0.0254); unmasked verts (hook bars near x=0) get
    side*px. The formula is the shader's LITERALLY, including its quirks
    (sequential accumulation, mSide = sign(x)).

    Radial (NUMBER_OF_STATICS_AND_DIAM): the mesh is modelled at unit
    DIAMETER; every vert moves outward by (diam_m-1)*0.5 - a translation,
    not a scale, so ring/hook profiles keep their thickness. Hook verts
    (color.w < 0.5) move as rigid blocks along their angle snapped to the
    numberOfStatics raster (angle convention atan2(y, z), identical to the
    shader because the mesh keeps raw i3d coordinates).

    Idempotent via a bake signature on the mesh; shared mesh datablocks are
    single-user-copied before baking. Returns False when the mesh cannot be
    baked (no vertex colours / foreign bake signature) - the caller then
    falls back to the old span/centre approximation."""
    me = obj.data
    sig = "v2|%.5f|%.5f|%.5f|%.5f|%.5f|%d" % (px, py, pz, pw, diam_m, statics)
    if me.get("_i3d_conn_bake") == sig:
        return True
    old = None
    if me.get("_i3d_conn_bake") is not None:
        # Baked for OTHER params: happens when a kept object switches config
        # or a new connector is duplicated from an already-baked one (twin
        # front duped from twin rear). Restart from the pristine datablock -
        # giving up here left the approximation running on already-deformed
        # geometry (#14 round 2: connector 'seltsam in die Breite gezogen').
        pre = me.get("_i3d_prebake_mesh")
        orig = bpy.data.meshes.get(pre) if pre else None
        if orig is None:
            return False
        old = me
    else:
        orig = me
    if orig.color_attributes.get("Color") is None:
        return False
    # Keep the pristine datablock: the export must write UN-deformed geometry
    # (GE/game re-run the shader at runtime; the export prep restores any mesh
    # carrying _i3d_prebake_mesh). _i3d_prebake_kept marks it so 'None'
    # (unload wheels) can purge it despite the fake user.
    orig.use_fake_user = True
    orig["_i3d_prebake_kept"] = True
    obj.data = me = orig.copy()
    me["_i3d_prebake_mesh"] = orig.name
    if old is not None and old.users == 0:
        try:
            bpy.data.meshes.remove(old)
        except Exception:
            pass
    attr = me.color_attributes.get("Color")
    if attr is None or attr.domain != "CORNER":
        return False
    vcol = [None] * len(me.vertices)
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            if vcol[vi] is None:
                vcol[vi] = tuple(attr.data[li].color)
    step = (2.0 * math.pi) / statics
    for v in me.vertices:
        r, g, b, w = (vcol[v.index] or (0.0, 0.0, 0.0, 1.0))[:4]
        x, y, z = v.co
        side = 1.0 if x >= 0 else -1.0
        if w < 0.5:
            # Engine vertex space has y/z SWAPPED vs the shapes file: the
            # statics blocks are modelled at raster+half-step in atan2(y,z)
            # but exactly ON the numberOfStatics raster in atan2(z,y). The
            # snap must run in engine orientation, otherwise every block
            # straddles a raster boundary and gets torn apart tangentially
            # (visible as smear proportional to the radial move - the front
            # twin connector, #14 round 3).
            a = math.atan2(z, y)
            a = math.floor(a / step + 0.5) * step
            dy, dz = math.cos(a), math.sin(a)
        else:
            ln = math.hypot(y, z)
            dy, dz = (y / ln, z / ln) if ln > 1e-9 else (0.0, 0.0)
        ny = y + dy * (diam_m - 1.0) * 0.5
        nz = z + dz * (diam_m - 1.0) * 0.5
        nx = (1.0 - r) * (1.0 - g) * (1.0 - b) * (x + side * px)
        nx += r * (x + side * (0.5 * py - 1.0))
        nx += g * (x + side * (0.5 * py + pz - 2.0))
        nx += b * (x + side * (0.5 * py + pz + pw - 3.0))
        nx += (1.0 - (r + g + b)) * x
        v.co = (nx, ny, nz)
    me.update()
    me["_i3d_conn_bake"] = sig
    return True


def _wd_pair(wd_str):
    """"16 38" -> (16.0, 38.0) inches, or None."""
    if not wd_str:
        return None
    try:
        p = wd_str.split()
        return (float(p[0]), float(p[1]))
    except (ValueError, IndexError):
        return None


def _bake_rim_mesh(obj, width_in, diam_in, statics):
    """Bake the game's RIM vertex shader (vehicleShader.xml getRimPos,
    variations 'rim' / 'rim_numberOfStatics') into *obj*'s mesh. The game
    deforms every base-game rim procedurally from the widthAndDiam shader
    parameter; our previous (w,d,d) object scale is only the game's own Lua
    FALLBACK for meshes WITHOUT the parameter and distorts dish depth,
    profile thickness and thereby the texture look (#14).

    Axial: control rings modelled at x = +-0.5, masked by vertex colours
    R (+0.5 side) / G (-0.5 side), move to +-width/2; unmasked verts keep x.
    Radial 'rim': smooth offset (diam-1)*0.5 damped by (1-blue); then the
    hole term ``pos.yz * (blue * (z_param - 1))`` with the RUNTIME z_param =
    0 (WheelVisualPart:setNode always passes 0) collapses fully blue
    geometry to the axis - the game hides it; unbaked it shows as dark
    slivers on the rim face.
    Radial 'rim_numberOfStatics' (*statics* set): bolt/hook blocks
    (color.w < 0.5) move rigidly along the numberOfStatics raster, the rest
    smoothly; no blue/hole term (the shader skips it for this variation).

    The mesh is modelled at unit DIAMETER (radial term is an offset, not a
    scale). Keeps the pristine datablock for export. Idempotent per bake
    signature; returns False when the mesh cannot be baked (caller keeps the
    scale approximation)."""
    me = obj.data
    sig = "rim2|%.5f|%.5f|%s" % (width_in, diam_in, statics)
    if me.get("_i3d_rim_bake") == sig:
        return True
    old = None
    if me.get("_i3d_rim_bake") is not None:
        # Baked for ANOTHER size: happens whenever a kept/duplicated object
        # switches dimensions (config change; front/rear wheels sharing one
        # rim node). Restart from the pristine datablock - giving up here
        # left the (w,d,d) scale fallback running on already-baked geometry
        # (#14 round 2: every non-default Vestrum config had too-small,
        # too-narrow rims).
        pre = me.get("_i3d_prebake_mesh")
        orig = bpy.data.meshes.get(pre) if pre else None
        if orig is None:
            return False
        old = me
    else:
        orig = me
    orig.use_fake_user = True
    orig["_i3d_prebake_kept"] = True
    obj.data = me = orig.copy()
    me["_i3d_prebake_mesh"] = orig.name
    if old is not None and old.users == 0:
        try:
            bpy.data.meshes.remove(old)
        except Exception:
            pass
    # No vertex colours = GPU default (0,0,0,1): the radial offset still
    # applies in the game (e.g. rim009 has no colours at all), only the
    # x-control rings and the blue hole need actual masks.
    attr = me.color_attributes.get("Color")
    if attr is not None and attr.domain != "CORNER":
        attr = None
    vcol = [None] * len(me.vertices)
    if attr is not None:
        for poly in me.polygons:
            for li in poly.loop_indices:
                vi = me.loops[li].vertex_index
                if vcol[vi] is None:
                    vcol[vi] = tuple(attr.data[li].color)
    w_m = width_in * INCH
    d_m = diam_in * INCH
    step = (2.0 * math.pi) / statics if statics else 0.0
    for v in me.vertices:
        r, g, b, wa = (vcol[v.index] or (0.0, 0.0, 0.0, 1.0))[:4]
        x, y, z = v.co
        if statics and wa < 0.5:
            # atan2(z,y)/cos/sin: engine vertex space has y/z swapped vs the
            # shapes file - see the identical block in _bake_connector_mesh.
            a = math.atan2(z, y)
            a = math.floor(a / step + 0.5) * step
            ny = y + math.cos(a) * (d_m - 1.0) * 0.5
            nz = z + math.sin(a) * (d_m - 1.0) * 0.5
        else:
            ln = math.hypot(y, z)
            yn, zn = (y / ln, z / ln) if ln > 1e-9 else (0.0, 0.0)
            if statics:
                ny = y + yn * (d_m - 1.0) * 0.5
                nz = z + zn * (d_m - 1.0) * 0.5
            else:
                f = (d_m - 1.0) * 0.5 * (1.0 - b)
                # hole term, runtime z_param = 0: + pos * (b * (0 - 1))
                ny = y + yn * f - y * b
                nz = z + zn * f - z * b
        nx = r * (x + w_m * 0.5 - 0.5)
        nx += g * (x - w_m * 0.5 + 0.5)
        nx += (1.0 - (r + g)) * x
        v.co = (nx, ny, nz)
    me.update()
    me["_i3d_rim_bake"] = sig
    return True


def _bake_rim_part(obj, p):
    """Try the exact shader bake for a placed rim part; on success switch the
    object from the (w,d,d) scale approximation to the game's real transform:
    baked mesh + the raw XML scale (or 1). The game applies BOTH sequentially
    (WheelVisualPart:setNode: setScale, THEN the widthAndDiam shader
    parameter resolved through the baseConfig/tire-xml chain) - the earlier
    'raw scale replaces widthAndDiam' model was wrong (#14: Vestrum discs too
    big in front, too small on Broad). Objects whose materials carry no rim
    shader variation (e.g. mod rims) keep the scale approximation, exactly
    like the game's Lua fallback. Returns True if baked."""
    wd = _wd_pair(p.get("bake_wd"))
    if not wd:
        return False
    variation = _wheel_mat_prop(obj, "customShaderVariation")
    if variation not in ("rim", "rim_numberOfStatics"):
        # No rim shader on the material: the GAME's Lua fallback converts
        # widthAndDiam into a node scale and thereby OVERWRITES any raw XML
        # scale (WheelVisualPart:setNode calls setScale(scale) first, then
        # the widthAndDiam else-branch calls setScale again). Tigrecar's
        # rim013 (plain material, wd "10 14.1" + scale "0.5 1 1") rendered
        # its disc at ~1 m with the raw scale alone. Keep this scale on
        # export (the node scale IS the sizing - there is no shader to read
        # a widthAndDiam parameter, so none is written).
        obj.scale = (wd[0] * INCH, wd[1] * INCH, wd[1] * INCH)
        obj["_i3d_rim_keep_scale"] = True
        if "_i3d_rim_wd" in obj:
            del obj["_i3d_rim_wd"]
        return False
    statics = None
    if variation == "rim_numberOfStatics":
        v = _wheel_mat_prop(obj, "customParameter_numberOfStatics")
        try:
            statics = max(1, int(float(str(v).split()[0]))) if v else 4
        except ValueError:
            statics = 4
    if not _bake_rim_mesh(obj, wd[0], wd[1], statics):
        return False
    raw = p.get("raw_scale")
    obj.scale = raw if raw else (1.0, 1.0, 1.0)
    obj["_i3d_rim_wd"] = p["bake_wd"]
    obj["_i3d_rim_keep_scale"] = bool(raw)
    return True


def _bake_hubdual_mesh(obj, start_in, end_in, uscale):
    """Bake the HUB_DUAL vertex shader (vehicleShader.xml getHubDualPos) for
    usePosAndScale connectors (MT655: hubs/dual004.i3d): the START clamp
    (vertex-colour R, modelled around |x|=1) moves to startPos, the END clamp
    (G, modelled around |x|=4) to endPos, the whole mesh is scaled by
    uniformScale. Literal shader formula (sequential, mSide = sign(x)).
    Same pristine/signature machinery as the other bakes."""
    me = obj.data
    sig = "hubdual|%.5f|%.5f|%.5f" % (start_in, end_in, uscale)
    if me.get("_i3d_conn_bake") == sig:
        return True
    old = None
    if me.get("_i3d_conn_bake") is not None:
        pre = me.get("_i3d_prebake_mesh")
        orig = bpy.data.meshes.get(pre) if pre else None
        if orig is None:
            return False
        old = me
    else:
        orig = me
    if orig.color_attributes.get("Color") is None:
        return False
    orig.use_fake_user = True
    orig["_i3d_prebake_kept"] = True
    obj.data = me = orig.copy()
    me["_i3d_prebake_mesh"] = orig.name
    if old is not None and old.users == 0:
        try:
            bpy.data.meshes.remove(old)
        except Exception:
            pass
    attr = me.color_attributes.get("Color")
    if attr is None or attr.domain != "CORNER":
        return False
    vcol = [None] * len(me.vertices)
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            if vcol[vi] is None:
                vcol[vi] = tuple(attr.data[li].color)
    sx = start_in * INCH
    ex = end_in * INCH
    s = uscale
    for v in me.vertices:
        r, g = (vcol[v.index] or (0.0, 0.0, 0.0, 1.0))[:2]
        x, y, z = v.co
        side = 1.0 if x >= 0 else -1.0
        nx = x + r * side * (-1.0)
        nx *= s
        ny = y * s
        nz = z * s
        nx += g * side * (4.0 * (1.0 - s))
        nx += r * side * sx
        nx += g * side * (ex - 4.0)
        v.co = (nx, ny, nz)
    me.update()
    me["_i3d_conn_bake"] = sig
    return True


def _place_connector(conn, p):
    """Position + shape the twin connector exactly like the game: the node
    sits at the MAIN wheel's centre (WheelVisualPartConnector:setNode does
    localToLocal(connectedWheel.node, ...) - NOT midway between the wheels),
    the <connector> #offset is ignored on this code path, the XML scale
    applies, and the span comes from baking the RIM_DUAL shader with
    connectorPos = (0, addWheelWidth+startPosOffset, rimOffset+awOffset
    +endPosOffset, hookOffset or addWheelWidth) inches and widthAndDiam.y =
    #diameter or the additional wheel's rim diameter (Wheels.lua l.210 +
    WheelVisualPartConnector:setNode). Falls back to the old span/centre
    approximation for meshes without vertex colours or for the
    useWidthAndDiam/usePosAndScale variants."""
    conn.parent = p["drive"]
    conn.matrix_parent_inverse.identity()
    conn.rotation_euler = (0.0, 0.0, 0.0)
    conn.location = (0.0, 0.0, 0.0)
    conn.scale = p["scale"] if p["scale"] else (1.0, 1.0, 1.0)
    ok = False
    mode = p.get("conn_mode") or ("simple" if p.get("conn_simple", True)
                                  else "widthdiam")
    meshes = [o for o in [conn] + _descendants(conn) if o.type == "MESH"]
    if mode == "simple":
        wi = p.get("conn_width_in") or 0.0
        diam_m = (p.get("conn_diam_in") or 0.0) * INCH
        if wi > 0 and diam_m > 0 and meshes:
            py_in = wi + (p.get("conn_spo_in") or 0.0)
            pz_in = ((p.get("conn_gap_in") or 0.0)
                     + (p.get("conn_epo_in") or 0.0))
            pw_in = p.get("conn_hook_in") or wi
            ok = True
            for m in meshes:
                ok = _bake_connector_mesh(m, 0.0, py_in * INCH, pz_in * INCH,
                                          pw_in * INCH, diam_m,
                                          _conn_statics(m)) and ok
            if ok:
                # Runtime shader parameters for the export prep: the exported
                # connector is the PRISTINE mesh, and GE runs the rimDual
                # shader with the MATERIAL parameters - without these the
                # defaults (connectorPos "0 80 40 40") blow the cage up to a
                # 2 m drum (Vestrum re-export). Written into the export
                # materials by _prepare_rims_for_export.
                conn["_i3d_conn_shader"] = "rimDual"
                conn["_i3d_conn_connectorPos"] = (
                    "0 %.4f %.4f %.4f" % (py_in, pz_in, pw_in))
                conn["_i3d_conn_widthAndDiam"] = (
                    "40 %.4f 1" % (p.get("conn_diam_in") or 40.0))
    elif mode == "posscale":
        # usePosAndScale (HUB_DUAL shader): startPos/endPos in inches +
        # uniform scale; without both positions the shader keeps its
        # material defaults - then the approximation is safer.
        st = p.get("conn_start_in")
        en = p.get("conn_end_in")
        us = p.get("conn_uscale")
        if st is not None and en is not None and meshes:
            ok = True
            for m in meshes:
                ok = _bake_hubdual_mesh(m, st, en,
                                        us if us else 1.0) and ok
            if ok:
                conn["_i3d_conn_shader"] = "hubDual"
                conn["_i3d_conn_posAndScale"] = (
                    "%.4f %.4f %.4f" % (st, en, us if us else 1.0))
    if not ok:
        _place_connector_approx(conn, p["drive"], p["diam_scale"])


def _place_connector_approx(conn, drive, diam_scale):
    """Span + centre the twin connector cage (approximation, fallback for
    connectors the shader bake cannot handle). Scaled to the combined rim
    width, centred on the two tires' mid-point - both measured drive-local
    from the already-placed wheels."""
    bpy.context.view_layer.update()
    dinv = drive.matrix_world.inverted()

    def xext(roles):
        xs = []
        for o in bpy.data.objects:
            if (o.get("_i3d_wheel_of") != drive.name or o.type != "MESH"
                    or o.get("_i3d_wheel_role") not in roles):
                continue
            for c in o.bound_box:
                xs.append((dinv @ (o.matrix_world @ mathutils.Vector(c))).x)
        return (min(xs), max(xs)) if xs else None

    rim = xext({"rim_outer", "rim_inner"})
    tire = xext({"tire"})
    if not rim or not tire:
        return
    rim_width = rim[1] - rim[0]
    tire_center = (tire[0] + tire[1]) / 2.0
    raw_x = conn.dimensions.x / (conn.scale.x or 1.0)
    sx = rim_width / raw_x if raw_x > 1e-6 else 1.0
    conn.parent = drive
    conn.matrix_parent_inverse.identity()
    conn.rotation_euler = (0.0, 0.0, 0.0)
    conn.location = (0.0, 0.0, 0.0)
    conn.scale = (sx, diam_scale, diam_scale)
    bpy.context.view_layer.update()
    cx = [(dinv @ (conn.matrix_world @ mathutils.Vector(c))).x
          for c in conn.bound_box]
    conn.location.x += tire_center - (min(cx) + max(cx)) / 2.0


def load_all_wheels(vehicle_xml_path, data_dir, import_id, config_index=0,
                    report=None, replace=True, brand_index=0, dim_col=0):
    """Incrementally load one wheel configuration + size column: only parts that
    actually change are removed/imported; parts that stay are kept (and just re-
    scaled/positioned). Keeps config/brand switches fast and avoids duplicate
    datablocks. dim_col picks a size for multi-size configs. Returns the number of
    parts present afterwards.
    """
    prev_sel = list(bpy.context.selected_objects)
    prev_act = bpy.context.view_layer.objects.active

    specs = i3d_wheel_resolver.resolve_wheels(vehicle_xml_path, data_dir,
                                              config_index, dim_col, brand_index)
    hubs = i3d_wheel_resolver.resolve_hubs(vehicle_xml_path, data_dir)
    drives = _drive_object_map(vehicle_xml_path, import_id)

    needed = _needed_parts(specs, hubs, drives)
    needed_by_key = {p["key"]: p for p in needed}

    # Snapshot what is already placed. Keyed roots (+ their subtrees) are tracked;
    # any keyless wheel object is a legacy placement and gets cleared.
    wheel_objs = [o for o in bpy.data.objects
                  if o.get("_i3d_wheel_import") == import_id]
    present = {}
    keyed = set()
    for o in wheel_objs:
        k = o.get("_i3d_wheel_key")
        if k:
            present.setdefault(k, []).append(o)
            keyed.add(o)
            keyed.update(_descendants(o))
    for o in wheel_objs:
        if o not in keyed and o.get("_i3d_wheel_key") is None:
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass

    # REMOVE present parts no longer needed.
    for k in list(present.keys()):
        if k not in needed_by_key:
            for o in present.pop(k):
                _remove_subtree(o)

    # KEEP + update parts that remain needed (re-scale / re-position only).
    for p in needed:
        if p.get("is_connector"):
            continue
        for o in present.get(p["key"], []):
            _apply_placement(o, p)
            if p["hide_lod"]:
                _hide_lod_and_mud(o)

    # ADD parts that are needed but not present. Prefer duplicating an already
    # placed part with identical geometry (same i3d + node) over re-importing the
    # i3d: e.g. switching single->twin, the twin tires/rims/weights/hubs already
    # exist, so only the connector i3d is a real import. Geometry signature =
    # (i3d, src_kind, node, root_index); mesh data is shared by _dup_subtree.
    def _sig(p):
        return (p["i3d"], p["src_kind"], p.get("node"), p.get("root_index"))

    add = [p for p in needed if p["key"] not in present]
    if add:
        # Pool of dup sources from the KEEP parts (connectors excluded - they are
        # re-measured, not shared).
        source_pool = {}
        for p in needed:
            if p.get("is_connector"):
                continue
            objs = present.get(p["key"])
            if objs:
                source_pool.setdefault(_sig(p), objs[0])
        registry = _registry_from_existing(import_id)
        templates = {}
        tpl_objs = set()
        # Only import i3ds whose geometry is not already placed somewhere.
        for i3d in {p["i3d"] for p in add if _sig(p) not in source_pool}:
            roots = i3d_reference_loader.import_referenced_i3d(
                i3d, report=report, debug_registry=registry)
            rid = roots[0].get("_i3d_import_id") if roots else None
            by_path = {}
            for o in bpy.data.objects:
                if o.get("_i3d_import_id") == rid:
                    by_path[o.get("_i3d_node_path")] = o
                    tpl_objs.add(o)
            for r in roots:
                tpl_objs.add(r)
                tpl_objs.update(_descendants(r))
            templates[i3d] = {"roots": roots, "by_path": by_path}
        for p in add:
            sig = _sig(p)
            src = source_pool.get(sig)
            if src is None:
                tpl = templates.get(p["i3d"])
                if tpl:
                    if p["src_kind"] == "root":
                        src = (tpl["roots"][p["root_index"]]
                               if p["root_index"] < len(tpl["roots"]) else None)
                    else:
                        src = tpl["by_path"].get(p["node"])
            if src is None:
                if report:
                    report("WARNING", "Wheel part source missing: %s" % p["key"])
                continue
            c = _dup_subtree(src)
            _tag_tree(c, {"_i3d_wheel_of": p["drive"].name,
                          "_i3d_wheel_import": import_id,
                          "_i3d_wheel_role": p["role"]})
            c["_i3d_wheel_key"] = p["key"]
            # Later add parts with the same geometry can dup from this one, so a
            # brand-new i3d is imported at most once per load.
            source_pool.setdefault(sig, c)
            if p.get("is_connector"):
                continue  # positioned in the connector pass below
            _apply_placement(c, p)
            if p["hide_lod"]:
                _hide_lod_and_mud(c)
        for o in tpl_objs:
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass

    # Rim pass: exact shader bake (game deform) for every placed inner/outer
    # rim whose material carries a rim shader variation; replaces the (w,d,d)
    # scale approximation set by _apply_placement (kept as fallback for mod
    # rims without the shader parameter, like the game's own Lua fallback).
    rim_parts = [p for p in needed
                 if p["role"] in ("rim_inner", "rim_outer") and p.get("bake_wd")]
    if rim_parts:
        by_key_rim = {}
        for o in bpy.data.objects:
            if (o.get("_i3d_wheel_import") == import_id
                    and o.get("_i3d_wheel_role") in ("rim_inner", "rim_outer")):
                k = o.get("_i3d_wheel_key")
                if k:
                    by_key_rim.setdefault(k, []).append(o)
        for p in rim_parts:
            for o in by_key_rim.get(p["key"], []):
                _bake_rim_part(o, p)

    # Connector pass: now that every wheel of each drive is placed, span + centre
    # the twin connectors by measuring the actual rims/tires.
    conn_parts = [p for p in needed if p.get("is_connector")]
    if conn_parts:
        bpy.context.view_layer.update()
        by_key = {}
        for o in bpy.data.objects:
            if (o.get("_i3d_wheel_import") == import_id
                    and o.get("_i3d_wheel_role") == "connector"):
                k = o.get("_i3d_wheel_key")
                if k:
                    by_key.setdefault(k, o)
        for p in conn_parts:
            o = by_key.get(p["key"])
            if o is not None:
                _place_connector(o, p)

    bpy.context.view_layer.update()
    _purge_empty_ref_collections()
    _restore_selection(prev_sel, prev_act)
    return len(needed)
