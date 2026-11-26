import math
import os
import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

from f100_geom_primitives import COLOR_ID_SPECS, rgba, bm_apply_hard_surface_smoothing
from f100_geom_primitives import (
    MAT_STAND_STEEL, MAT_STAND_ORANGE, MAT_TRAILER_YELLOW,
    MAT_FASTENER, MAT_BRAID_HOSE, MAT_YELLOW_HARNESS,
    MAT_ADEL_CUSHION, MAT_CONNECTOR_38999, MAT_NOZ_ACT_AB, MAT_RUBBER_DARK,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _cap_even_ring_quads(bm, ring, mat_idx: int, flip: bool = False):

    n = len(ring)
    assert n >= 6 and (n % 2 == 0), f"Ring must have even >=6 verts, got {n}"
    quads = [(ring[0], ring[1], ring[2], ring[n - 1])]
    for i in range(2, n // 2 - 1):
        quads.append((ring[i], ring[i + 1], ring[n - i], ring[n - i + 1]))
    h = n // 2 - 1
    quads.append((ring[h], ring[h + 1], ring[h + 2], ring[n - h]))
    for q in quads:
        verts = tuple(reversed(q)) if flip else q
        try:
            f = bm.faces.new(verts)
            f.material_index = mat_idx
            f.smooth = False
        except ValueError:
            pass


def _ortho_basis(axis_z: Vector):
    z = axis_z.normalized()
    ref = Vector((1.0, 0.0, 0.0)) if abs(z.x) < 0.85 else Vector((0.0, 1.0, 0.0))
    x = ref.cross(z).normalized()
    y = z.cross(x).normalized()
    return x, y, z


def _add_quad_lathe(bm, origin: Vector, axis_z: Vector, profile, n_seg: int = 12,
                    mat_idx: int = 0, cap_start: bool = True, cap_end: bool = True,
                    knurl_amp: float = 0.0, knurl_z_range=None):

    bx, by, bz = _ortho_basis(axis_z)
    rings = []
    for zl, rad in profile:
        in_knurl = (knurl_amp > 0.0 and knurl_z_range is not None and knurl_z_range[0] <= zl <= knurl_z_range[1])
        ring = []
        for k in range(n_seg):
            th = 2.0 * math.pi * (k / float(n_seg))
            r_eff = rad + (knurl_amp if (in_knurl and (k % 2 == 0)) else (-knurl_amp if in_knurl else 0.0))
            p = origin + bz * zl + (bx * math.cos(th) + by * math.sin(th)) * r_eff
            ring.append(bm.verts.new(p))
        rings.append(ring)

    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
        for k in range(n_seg):
            kn = (k + 1) % n_seg
            try:
                f = bm.faces.new((r0[k], r0[kn], r1[kn], r1[k]))
                f.material_index = mat_idx
                f.smooth = True
            except ValueError:
                pass

    if cap_start:
        _cap_even_ring_quads(bm, rings[0], mat_idx, flip=True)
    if cap_end:
        _cap_even_ring_quads(bm, rings[-1], mat_idx, flip=False)
    return rings


def _add_quad_tube(bm, pts, radii, n_seg: int = 10, mat_idx: int = 0,
                   cap_start: bool = True, cap_end: bool = True, closed_loop: bool = False):

    m = len(pts)
    if m < 2:
        return []
    if isinstance(radii, (int, float)):
        r_list = [float(radii)] * m
    else:
        r_list = list(radii)


    tangents = []
    for i in range(m):
        if closed_loop:
            t = pts[(i + 1) % m] - pts[(i - 1) % m]
        elif i == 0:
            t = pts[1] - pts[0]
        elif i == m - 1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i + 1] - pts[i - 1]
        tangents.append(t.normalized() if t.length > 1e-8 else Vector((0, 1, 0)))

    u_cur, v_cur, _ = _ortho_basis(tangents[0])
    rings = []
    for i in range(m):
        ti = tangents[i]
        if i > 0:
            u_proj = u_cur - ti * u_cur.dot(ti)
            if u_proj.length > 1e-6:
                u_cur = u_proj.normalized()
            else:
                u_cur, _, _ = _ortho_basis(ti)
            v_cur = ti.cross(u_cur).normalized()
        ri = r_list[i]
        ring = []
        for k in range(n_seg):
            th = 2.0 * math.pi * (k / float(n_seg))
            p = pts[i] + (u_cur * math.cos(th) + v_cur * math.sin(th)) * ri
            ring.append(bm.verts.new(p))
        rings.append(ring)

    seg_count = m if closed_loop else (m - 1)
    for i in range(seg_count):
        r0 = rings[i]
        r1 = rings[(i + 1) % m]
        for k in range(n_seg):
            kn = (k + 1) % n_seg
            try:
                f = bm.faces.new((r0[k], r0[kn], r1[kn], r1[k]))
                f.material_index = mat_idx
                f.smooth = True
            except ValueError:
                pass

    if not closed_loop:
        if cap_start:
            _cap_even_ring_quads(bm, rings[0], mat_idx, flip=True)
        if cap_end:
            _cap_even_ring_quads(bm, rings[-1], mat_idx, flip=False)
    return rings


def _catmull_rom_sample(ctrl_pts, samples_per_seg: int = 8):

    n = len(ctrl_pts)
    if n < 2:
        return list(ctrl_pts)
    ext = [ctrl_pts[0] + (ctrl_pts[0] - ctrl_pts[1])] + list(ctrl_pts) + [ctrl_pts[-1] + (ctrl_pts[-1] - ctrl_pts[-2])]
    out = []
    for i in range(1, n):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        steps = samples_per_seg if i < n - 1 else samples_per_seg + 1
        for s in range(steps):
            t = s / float(samples_per_seg)
            t2 = t * t
            t3 = t2 * t
            q = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(q)
    return out


def _add_quad_box(bm, center: Vector, bx: Vector, by: Vector, bz: Vector,
                  hx: float, hy: float, hz: float, mat_idx: int = 0):

    corners = []
    for sx in (-1.0, +1.0):
        for sy in (-1.0, +1.0):
            for sz in (-1.0, +1.0):
                corners.append(bm.verts.new(center + bx * (sx * hx) + by * (sy * hy) + bz * (sz * hz)))

    faces = [
        (0, 2, 3, 1), (4, 5, 7, 6),
        (0, 1, 5, 4), (2, 6, 7, 3),
        (0, 4, 6, 2), (1, 3, 7, 5),
    ]
    for q in faces:
        try:
            f = bm.faces.new(tuple(corners[i] for i in q))
            f.material_index = mat_idx
            f.smooth = False
        except ValueError:
            pass


def _raycast_hull(bvh_list, origin: Vector, direction: Vector, max_dist: float = 0.35):

    best_d = 1e9
    best_hit = None
    best_nrm = None
    d_n = direction.normalized()
    for bvh in bvh_list:
        loc, nrm, _, dist = bvh.ray_cast(origin, d_n, max_dist)
        if loc is not None and dist < best_d:
            best_d = dist
            best_hit = loc
            best_nrm = nrm.normalized()
    return best_hit, best_nrm


def _build_bvh_for_objects(obj_names):
    bvhs = []
    for nm in obj_names:
        o = bpy.data.objects.get(nm)
        if o is None or o.type != 'MESH' or len(o.data.polygons) == 0:
            continue
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.transform(o.matrix_world)
        bvhs.append(BVHTree.FromBMesh(bm))
        bm.free()
    return bvhs


def _finalize_detail_mesh_obj(name: str, bm, mats: list, coll, shadow: bool = True,
                           smooth_angle: float = 36.0, recalc_normals: bool = True):

    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm_apply_hard_surface_smoothing(bm, angle_deg=smooth_angle)

    nq = sum(1 for f in bm.faces if len(f.verts) != 4)
    za = sum(1 for f in bm.faces if f.calc_area() < 1e-8)
    if nq != 0 or za != 0:
        bm.free()
        raise RuntimeError(f"Detail mesh {name} topology violation: NonQuads={nq}, ZeroArea={za}")

    old = bpy.data.objects.get(name)
    if old is not None:
        old_me = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if old_me and old_me.users == 0:
            bpy.data.meshes.remove(old_me)

    me = bpy.data.meshes.new(name)
    for m in mats:
        me.materials.append(m)
    bm.to_mesh(me)
    bm.free()
    me.update()

    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    obj.visible_shadow = shadow

    wn = obj.modifiers.new("WeightedNormal", 'WEIGHTED_NORMAL')
    wn.keep_sharp = True
    wn.weight = 50
    return obj


def _new_node(nt, bl_idname, loc=(0, 0), name=None):
    n = nt.nodes.new(bl_idname)
    n.location = loc
    if name:
        n.name = name
    return n


def _build_detail_metal(mat_name: str, parent_id: int, ng_global, ng_masks, ng_wrap,
                     *, col_a: int, col_b: int, crevice_col: int,
                     metallic: float = 0.96, r_min: float = 0.10, r_max: float = 0.24,
                     rough_floor: float = 0.08, f82_tint: int = 0xE4EAF0):

    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1000, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1000, -300))
    ctrl.node_tree = ng_global
    masks = _new_node(nt, "ShaderNodeGroup", (-740, 120))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.016
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00022
    masks.inputs["Micro_Scale"].default_value = 520.0

    mix_b = _new_node(nt, "ShaderNodeMix", (-440, 220)); mix_b.data_type = 'RGBA'
    mix_b.inputs[6].default_value = rgba(col_a)
    mix_b.inputs[7].default_value = rgba(col_b)
    nt.links.new(masks.outputs["Macro_Mottle"], mix_b.inputs[0])

    cr_w = _new_node(nt, "ShaderNodeMath", (-440, -60)); cr_w.operation = 'MULTIPLY'; cr_w.use_clamp = True
    nt.links.new(masks.outputs["Crevice_Mask"], cr_w.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], cr_w.inputs[1])
    cr_s = _new_node(nt, "ShaderNodeMath", (-280, -60)); cr_s.operation = 'MULTIPLY'; cr_s.use_clamp = True
    cr_s.inputs[1].default_value = 0.55
    nt.links.new(cr_w.outputs[0], cr_s.inputs[0])

    mix_c = _new_node(nt, "ShaderNodeMix", (-120, 220)); mix_c.data_type = 'RGBA'
    mix_c.inputs[7].default_value = rgba(crevice_col)
    nt.links.new(cr_s.outputs[0], mix_c.inputs[0])
    nt.links.new(mix_b.outputs[2], mix_c.inputs[6])

    r_m = _new_node(nt, "ShaderNodeMapRange", (-440, 60)); r_m.clamp = True
    r_m.inputs[1].default_value = 0.15; r_m.inputs[2].default_value = 0.85
    r_m.inputs[3].default_value = r_min; r_m.inputs[4].default_value = r_max
    nt.links.new(masks.outputs["Metal_Rough_Breakup"], r_m.inputs[0])

    r_fl = _new_node(nt, "ShaderNodeMath", (-120, 60), name="Metal_Roughness_Floor")
    r_fl.operation = 'MAXIMUM'; r_fl.use_clamp = True
    r_fl.inputs[1].default_value = rough_floor
    nt.links.new(r_m.outputs[0], r_fl.inputs[0])

    bmp = _new_node(nt, "ShaderNodeBump", (-120, -200))
    bmp.inputs["Strength"].default_value = 0.025
    bmp.inputs["Distance"].default_value = 0.00020
    nt.links.new(masks.outputs["Micro_Grain"], bmp.inputs["Height"])
    nt.links.new(masks.outputs["Bevel_Normal"], bmp.inputs["Normal"])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (160, 120))
    nt.links.new(mix_c.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = metallic
    nt.links.new(r_fl.outputs[0], bsdf.inputs["Roughness"])
    bsdf.inputs["Specular Tint"].default_value = rgba(f82_tint)
    bsdf.inputs["Anisotropic"].default_value = 0.0
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (460, 120))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    id_hex = COLOR_ID_SPECS[parent_id][1]
    wrap.inputs["ID_Color"].default_value = rgba(id_hex)
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (720, 120))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(id_hex)
    mat["f15_pbr_hex"] = int(col_a)
    mat.diffuse_color = rgba(col_a)
    mat.metallic = metallic
    mat.roughness = 0.5 * (r_min + r_max)
    return mat


def _build_detail_dielectric(mat_name: str, parent_id: int, ng_global, ng_masks, ng_wrap,
                          *, col_a: int, col_b: int, crevice_col: int,
                          r_min: float = 0.36, r_max: float = 0.54,
                          coat_w: float = 0.0, coat_r: float = 0.22,
                          weave_scale: float = 0.0, bump_str: float = 0.045,
                          spec_ior: float = 0.48):

    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1060, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1060, -300))
    ctrl.node_tree = ng_global
    masks = _new_node(nt, "ShaderNodeGroup", (-780, 120))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.014
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00022
    masks.inputs["Micro_Scale"].default_value = 460.0

    mix_b = _new_node(nt, "ShaderNodeMix", (-440, 220)); mix_b.data_type = 'RGBA'
    mix_b.inputs[6].default_value = rgba(col_a)
    mix_b.inputs[7].default_value = rgba(col_b)
    nt.links.new(masks.outputs["Macro_Mottle"], mix_b.inputs[0])

    cr_w = _new_node(nt, "ShaderNodeMath", (-440, -60)); cr_w.operation = 'MULTIPLY'; cr_w.use_clamp = True
    nt.links.new(masks.outputs["Crevice_Mask"], cr_w.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], cr_w.inputs[1])
    cr_s = _new_node(nt, "ShaderNodeMath", (-280, -60)); cr_s.operation = 'MULTIPLY'; cr_s.use_clamp = True
    cr_s.inputs[1].default_value = 0.58
    nt.links.new(cr_w.outputs[0], cr_s.inputs[0])

    mix_c = _new_node(nt, "ShaderNodeMix", (-120, 220)); mix_c.data_type = 'RGBA'
    mix_c.inputs[7].default_value = rgba(crevice_col)
    nt.links.new(cr_s.outputs[0], mix_c.inputs[0])
    nt.links.new(mix_b.outputs[2], mix_c.inputs[6])

    r_m = _new_node(nt, "ShaderNodeMapRange", (-440, 60)); r_m.clamp = True
    r_m.inputs[1].default_value = 0.15; r_m.inputs[2].default_value = 0.85
    r_m.inputs[3].default_value = r_min; r_m.inputs[4].default_value = r_max
    nt.links.new(masks.outputs["Photo_Smudge"], r_m.inputs[0])

    if weave_scale > 0.0:
        w1 = _new_node(nt, "ShaderNodeTexWave", (-640, -240))
        w1.wave_type = 'BANDS'; w1.bands_direction = 'DIAGONAL'
        w1.inputs["Scale"].default_value = weave_scale
        w1.inputs["Distortion"].default_value = 1.2
        nt.links.new(tc.outputs["Object"], w1.inputs["Vector"])
        mp2 = _new_node(nt, "ShaderNodeMapping", (-840, -380))
        mp2.inputs["Rotation"].default_value = (0.0, 0.0, math.pi * 0.5)
        nt.links.new(tc.outputs["Object"], mp2.inputs["Vector"])
        w2 = _new_node(nt, "ShaderNodeTexWave", (-640, -380))
        w2.wave_type = 'BANDS'; w2.bands_direction = 'DIAGONAL'
        w2.inputs["Scale"].default_value = weave_scale
        w2.inputs["Distortion"].default_value = 1.2
        nt.links.new(mp2.outputs["Vector"], w2.inputs["Vector"])
        w_add = _new_node(nt, "ShaderNodeMath", (-400, -300)); w_add.operation = 'MULTIPLY'
        nt.links.new(w1.outputs["Fac"], w_add.inputs[0])
        nt.links.new(w2.outputs["Fac"], w_add.inputs[1])
        h_src = w_add.outputs[0]
    else:
        h_src = masks.outputs["Micro_Grain"]

    bmp = _new_node(nt, "ShaderNodeBump", (-120, -220))
    bmp.inputs["Strength"].default_value = bump_str
    bmp.inputs["Distance"].default_value = 0.00030
    nt.links.new(h_src, bmp.inputs["Height"])
    nt.links.new(masks.outputs["Bevel_Normal"], bmp.inputs["Normal"])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (160, 120))
    nt.links.new(mix_c.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    nt.links.new(r_m.outputs[0], bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = spec_ior
    bsdf.inputs["Coat Weight"].default_value = coat_w
    bsdf.inputs["Coat Roughness"].default_value = coat_r
    if coat_w > 1e-4 and "Coat Normal" in bsdf.inputs:
        nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Coat Normal"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (460, 120))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    id_hex = COLOR_ID_SPECS[parent_id][1]
    wrap.inputs["ID_Color"].default_value = rgba(id_hex)
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (720, 120))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(id_hex)
    mat["f15_pbr_hex"] = int(col_a)
    mat.diffuse_color = rgba(col_a)
    mat.metallic = 0.0
    mat.roughness = 0.5 * (r_min + r_max)
    return mat


def _build_service_tag_card_mat(ng_global, ng_masks, ng_wrap, png_path: str):

    mat = bpy.data.materials.get("Mat_Service_DD1574_Card") or bpy.data.materials.new("Mat_Service_DD1574_Card")
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1080, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1080, -320))
    ctrl.node_tree = ng_global
    masks = _new_node(nt, "ShaderNodeGroup", (-800, -140))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.015
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00018

    img_name = "IMG_decal_dd1574_serviceable_tag"
    img = bpy.data.images.get(img_name)
    if img is None:
        img = bpy.data.images.load(png_path, check_existing=False)
        img.name = img_name
    else:
        img.filepath = png_path
        try:
            img.unpack(method='REMOVE')
        except Exception:
            pass
        img.reload()
    img.colorspace_settings.name = 'sRGB'
    try:
        img.pack()
    except Exception:
        pass

    tex = _new_node(nt, "ShaderNodeTexImage", (-760, 220))
    tex.image = img
    tex.extension = 'CLIP'
    nt.links.new(tc.outputs["UV"], tex.inputs["Vector"])


    w_m = _new_node(nt, "ShaderNodeMath", (-460, -20)); w_m.operation = 'MULTIPLY'; w_m.use_clamp = True
    nt.links.new(masks.outputs["Photo_Smudge"], w_m.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], w_m.inputs[1])
    w_s = _new_node(nt, "ShaderNodeMath", (-300, -20)); w_s.operation = 'MULTIPLY'; w_s.use_clamp = True
    w_s.inputs[1].default_value = 0.14
    nt.links.new(w_m.outputs[0], w_s.inputs[0])

    mix_c = _new_node(nt, "ShaderNodeMix", (-120, 220)); mix_c.data_type = 'RGBA'
    mix_c.inputs[7].default_value = rgba(0x6E5D34)
    nt.links.new(w_s.outputs[0], mix_c.inputs[0])
    nt.links.new(tex.outputs["Color"], mix_c.inputs[6])

    bmp = _new_node(nt, "ShaderNodeBump", (-120, -200))
    bmp.inputs["Strength"].default_value = 0.032
    bmp.inputs["Distance"].default_value = 0.00025
    nt.links.new(masks.outputs["Micro_Grain"], bmp.inputs["Height"])
    nt.links.new(masks.outputs["Bevel_Normal"], bmp.inputs["Normal"])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (160, 120))
    nt.links.new(mix_c.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.72
    bsdf.inputs["Specular IOR Level"].default_value = 0.34
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.035
        bsdf.inputs["Subsurface Scale"].default_value = 0.0015
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])


    tr = _new_node(nt, "ShaderNodeBsdfTransparent", (160, -180))
    m_al = _new_node(nt, "ShaderNodeMixShader", (420, 120))
    nt.links.new(tex.outputs["Alpha"], m_al.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], m_al.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], m_al.inputs[2])

    wrap = _new_node(nt, "ShaderNodeGroup", (660, 120))
    wrap.node_tree = ng_wrap
    nt.links.new(m_al.outputs["Shader"], wrap.inputs["PBR_Shader"])
    id_hex = COLOR_ID_SPECS[MAT_TRAILER_YELLOW][1]
    wrap.inputs["ID_Color"].default_value = rgba(id_hex)
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (900, 120))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(id_hex)
    mat["f15_pbr_hex"] = int(0xE0BE36)
    mat.diffuse_color = rgba(0xE0BE36)
    mat.metallic = 0.0
    mat.roughness = 0.72
    return mat


def _build_pvc_sleeve_mat(ng_global, ng_masks, ng_wrap):

    mat = bpy.data.materials.get("Mat_Service_PVC_Sleeve") or bpy.data.materials.new("Mat_Service_PVC_Sleeve")
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-960, 0))
    masks = _new_node(nt, "ShaderNodeGroup", (-680, -140))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])

    lw = _new_node(nt, "ShaderNodeLayerWeight", (-420, 220))
    lw.inputs["Blend"].default_value = 0.14
    op_map = _new_node(nt, "ShaderNodeMapRange", (-180, 220)); op_map.clamp = True
    op_map.inputs[1].default_value = 0.0; op_map.inputs[2].default_value = 1.0
    op_map.inputs[3].default_value = 0.055; op_map.inputs[4].default_value = 0.34
    nt.links.new(lw.outputs["Fresnel"], op_map.inputs[0])

    n_cr = _new_node(nt, "ShaderNodeTexNoise", (-680, 160))
    n_cr.inputs["Scale"].default_value = 28.0
    n_cr.inputs["Detail"].default_value = 4.0
    n_cr.inputs["Roughness"].default_value = 0.45
    nt.links.new(tc.outputs["Object"], n_cr.inputs["Vector"])

    bmp = _new_node(nt, "ShaderNodeBump", (-200, -120))
    bmp.inputs["Strength"].default_value = 0.018
    bmp.inputs["Distance"].default_value = 0.00035
    nt.links.new(n_cr.outputs["Factor"], bmp.inputs["Height"])
    nt.links.new(masks.outputs["Bevel_Normal"], bmp.inputs["Normal"])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (120, 120))
    bsdf.inputs["Base Color"].default_value = rgba(0xE8EEF4)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.08
    bsdf.inputs["IOR"].default_value = 1.52
    bsdf.inputs["Specular IOR Level"].default_value = 0.52
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])

    tr = _new_node(nt, "ShaderNodeBsdfTransparent", (120, -160))
    m_tr = _new_node(nt, "ShaderNodeMixShader", (400, 100))
    nt.links.new(op_map.outputs[0], m_tr.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], m_tr.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], m_tr.inputs[2])

    wrap = _new_node(nt, "ShaderNodeGroup", (640, 100))
    wrap.node_tree = ng_wrap
    nt.links.new(m_tr.outputs["Shader"], wrap.inputs["PBR_Shader"])
    id_hex = COLOR_ID_SPECS[MAT_TRAILER_YELLOW][1]
    wrap.inputs["ID_Color"].default_value = rgba(id_hex)
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (880, 100))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(id_hex)
    mat["f15_pbr_hex"] = int(0xDCE4EC)
    mat.diffuse_color = (0.86, 0.89, 0.92, 0.25)
    mat.metallic = 0.0
    mat.roughness = 0.16
    return mat


def _build_wheel_grease_zerks(coll_stand, m_chrome, m_redcap):

    bm = bmesh.new()

    wheels = [
        (+1.0, -3.8800, True),
        (+1.0, -1.4950, False),
        (-1.0, -3.8800, False),
        (-1.0, -1.4950, True),
    ]
    for sx, cy, cap_on in wheels:
        org = Vector((sx * 0.7850, cy, -1.3200))
        ax = Vector((sx, 0.0, 0.0))

        _add_quad_lathe(
            bm, org, ax,
            [(-0.0020, 0.0068), (0.0005, 0.0068), (0.0046, 0.0064), (0.0054, 0.0044)],
            n_seg=6, mat_idx=0, cap_start=True, cap_end=True,
        )
        _add_quad_lathe(
            bm, org, ax,
            [(0.0048, 0.0036), (0.0085, 0.0027), (0.0110, 0.0036), (0.0138, 0.0033), (0.0152, 0.0022)],
            n_seg=12, mat_idx=0, cap_start=False, cap_end=True,
        )

        _add_quad_lathe(
            bm, org, ax,
            [(0.0010, 0.0070), (0.0014, 0.0084), (0.0034, 0.0084), (0.0038, 0.0070)],
            n_seg=12, mat_idx=1, cap_start=False, cap_end=False,
        )
        if cap_on:

            _add_quad_lathe(
                bm, org, ax,
                [(0.0064, 0.0048), (0.0074, 0.0052), (0.0148, 0.0050), (0.0172, 0.0044), (0.0182, 0.0026)],
                n_seg=12, mat_idx=1, cap_start=True, cap_end=True,
                knurl_amp=0.00028, knurl_z_range=(0.0072, 0.0150),
            )
        else:

            tether = _catmull_rom_sample([
                org + Vector((sx * 0.0025, 0.0, -0.0078)),
                org + Vector((sx * 0.0072, 0.0022, -0.0145)),
                org + Vector((sx * 0.0088, 0.0038, -0.0225)),
                org + Vector((sx * 0.0064, 0.0030, -0.0295)),
            ], samples_per_seg=4)
            _add_quad_tube(bm, tether, 0.00105, n_seg=6, mat_idx=1, cap_start=True, cap_end=True)
            cap_org = org + Vector((sx * 0.0062, 0.0028, -0.0285))
            cap_ax = Vector((sx * 0.18, 0.12, -0.97)).normalized()
            _add_quad_lathe(
                bm, cap_org, cap_ax,
                [(0.0, 0.0048), (0.0012, 0.0052), (0.0088, 0.0050), (0.0110, 0.0042), (0.0120, 0.0024)],
                n_seg=12, mat_idx=1, cap_start=True, cap_end=True,
                knurl_amp=0.00025, knurl_z_range=(0.0010, 0.0090),
            )

    return _finalize_detail_mesh_obj(
        "Stand_Trailer_08_Wheel_GreaseZerks", bm, [m_chrome, m_redcap], coll_stand, shadow=True,
    )


def _build_dd1574_serviceable_tag(coll_stand, m_card, m_pvc, m_wire, m_brass):

    bm = bmesh.new()
    uv_l = bm.loops.layers.uv.new("UVMap")


    yaw = math.radians(16.0)
    n_out = Vector((math.cos(yaw), math.sin(yaw), 0.0)).normalized()
    u_right = Vector((-math.sin(yaw), math.cos(yaw), 0.0)).normalized()
    v_up = Vector((0.0, 0.0, 1.0))


    p_anchor = Vector((0.752, -3.742, -0.015))
    p_hole   = Vector((0.782, -3.704, -0.044))
    _add_quad_lathe(bm, p_anchor + Vector((0, 0, -0.003)), Vector((0, 0, 1)),
                    [(0.0, 0.0055), (0.0055, 0.0055), (0.0075, 0.0032)], n_seg=6, mat_idx=2)
    wire_pts = _catmull_rom_sample([
        p_anchor + Vector((+0.000, +0.000, +0.004)),
        p_anchor + Vector((+0.014, +0.016, +0.010)),
        p_hole   + Vector((+0.002, +0.003, +0.012)),
        p_hole   + Vector((0.0,    0.0,    -0.002)),
        p_hole   + Vector((-0.004, -0.004, +0.008)),
        p_anchor + Vector((+0.010, +0.008, -0.002)),
        p_anchor + Vector((+0.000, +0.000, +0.004)),
    ], samples_per_seg=5)
    _add_quad_tube(bm, wire_pts[:-1], 0.00085, n_seg=6, mat_idx=2, closed_loop=True)


    _add_quad_lathe(
        bm, p_hole, n_out,
        [(-0.0014, 0.0032), (-0.0014, 0.0062), (-0.0004, 0.0064),
         (+0.0004, 0.0064), (+0.0014, 0.0062), (+0.0014, 0.0032)],
        n_seg=12, mat_idx=3, cap_start=False, cap_end=False,
    )


    W_card, H_card, T_card = 0.076, 0.114, 0.0007

    p_top_ctr = p_hole + v_up * (0.08 * H_card)

    def _emit_draped_sheet(w_m, h_m, t_m, nu, nv, mat_i, z_extra_top=0.0, z_extra_bot=0.0, uv_pad=0.0):

        grid_f = []
        grid_b = []
        h_tot = h_m + z_extra_top + z_extra_bot
        for j in range(nv + 1):
            fv = j / float(nv)
            z_off = -h_m - z_extra_bot + fv * h_tot

            bend = 0.0022 * math.sin(fv * math.pi) - 0.0014 * ((1.0 - fv) ** 2)
            row_f, row_b = [], []
            for i in range(nu + 1):
                fu = i / float(nu)
                x_off = (fu - 0.5) * w_m
                curl = 0.0009 * ((fu - 0.5) * 2.0) ** 2 * (1.0 - fv)
                base_p = p_top_ctr + u_right * x_off + v_up * z_off + n_out * (bend + curl)
                row_f.append(bm.verts.new(base_p + n_out * (0.5 * t_m)))
                row_b.append(bm.verts.new(base_p - n_out * (0.5 * t_m)))
            grid_f.append(row_f)
            grid_b.append(row_b)

        def _uv(fu, fv):
            u_c = max(0.0, min(1.0, (fu - uv_pad) / max(1e-5, 1.0 - 2.0 * uv_pad)))
            v_c = max(0.0, min(1.0, (fv - uv_pad) / max(1e-5, 1.0 - 2.0 * uv_pad)))
            return (u_c, v_c)


        for j in range(nv):
            fv0, fv1 = j / float(nv), (j + 1) / float(nv)
            for i in range(nu):
                fu0, fu1 = i / float(nu), (i + 1) / float(nu)
                try:
                    ff = bm.faces.new((grid_f[j][i], grid_f[j][i + 1], grid_f[j + 1][i + 1], grid_f[j + 1][i]))
                    ff.material_index = mat_i
                    ff.smooth = True
                    uvs_f = [_uv(fu0, fv0), _uv(fu1, fv0), _uv(fu1, fv1), _uv(fu0, fv1)]
                    for lp, uvc in zip(ff.loops, uvs_f):
                        lp[uv_l].uv = uvc
                except ValueError:
                    pass
                try:
                    fb = bm.faces.new((grid_b[j][i + 1], grid_b[j][i], grid_b[j + 1][i], grid_b[j + 1][i + 1]))
                    fb.material_index = mat_i
                    fb.smooth = True
                    uvs_b = [_uv(1.0 - fu1, fv0), _uv(1.0 - fu0, fv0), _uv(1.0 - fu0, fv1), _uv(1.0 - fu1, fv1)]
                    for lp, uvc in zip(fb.loops, uvs_b):
                        lp[uv_l].uv = uvc
                except ValueError:
                    pass

        for i in range(nu):
            for (v0, v1, v2, v3) in [
                (grid_b[0][i], grid_b[0][i + 1], grid_f[0][i + 1], grid_f[0][i]),
                (grid_f[nv][i], grid_f[nv][i + 1], grid_b[nv][i + 1], grid_b[nv][i]),
            ]:
                try:
                    fe = bm.faces.new((v0, v1, v2, v3))
                    fe.material_index = mat_i
                    fe.smooth = False
                except ValueError:
                    pass
        for j in range(nv):
            for (v0, v1, v2, v3) in [
                (grid_f[j][0], grid_b[j][0], grid_b[j + 1][0], grid_f[j + 1][0]),
                (grid_b[j][nu], grid_f[j][nu], grid_f[j + 1][nu], grid_b[j + 1][nu]),
            ]:
                try:
                    fe = bm.faces.new((v0, v1, v2, v3))
                    fe.material_index = mat_i
                    fe.smooth = False
                except ValueError:
                    pass


    _emit_draped_sheet(W_card, H_card, T_card, nu=6, nv=10, mat_i=0)

    _emit_draped_sheet(W_card + 0.008, H_card, 0.0024, nu=6, nv=10, mat_i=1, z_extra_top=0.005, z_extra_bot=0.005)

    return _finalize_detail_mesh_obj(
        "Stand_Cradle_06_DD1574_ServiceableTag", bm, [m_card, m_pvc, m_wire, m_brass],
        coll_stand, shadow=True, smooth_angle=45.0, recalc_normals=False,
    )


def _build_fod_caps_and_cannon_plugs(coll_core, hull_bvhs, m_redcap, m_yelcap, m_olive, m_steel):

    bm = bmesh.new()


    def _cyl_ray(phi_deg, yv, r_start=0.72):
        pr = math.radians(phi_deg)
        n_rad = Vector((math.sin(pr), 0.0, math.cos(pr)))
        p0 = Vector((r_start * n_rad.x, yv, r_start * n_rad.z))
        hit, nrm = _raycast_hull(hull_bvhs, p0, -n_rad, max_dist=0.30)
        if hit is None:
            return Vector((0.540 * n_rad.x, yv, 0.540 * n_rad.z)), n_rad

        ax = (0.72 * n_rad + 0.28 * nrm).normalized()
        return hit, ax


    fod_specs = [
        (+64.0, -3.96, 0.0180, 0.024, 0),
        (+78.0, -3.20, 0.0155, 0.022, 1),
        (+69.0, -2.88, 0.0165, 0.023, 0),
        (+90.0, -2.72, 0.0160, 0.022, 1),
        (+58.0, -2.04, 0.0175, 0.023, 0),
        (+106.0, -1.87, 0.0145, 0.020, 0),
        (-74.0, -3.48, 0.0155, 0.021, 0),
        (-102.0, -2.72, 0.0145, 0.020, 1),
    ]
    for idx_f, (phi_d, yv, rc, hc, mi) in enumerate(fod_specs):
        base_p, ax = _cyl_ray(phi_d, yv)
        bx, by, bz = _ortho_basis(ax)

        _add_quad_lathe(
            bm, base_p, ax,
            [(-0.004, rc * 1.18), (0.002, rc * 1.18), (0.004, rc * 0.96), (0.008, rc * 0.92)],
            n_seg=12, mat_idx=3, cap_start=True, cap_end=False,
        )

        _add_quad_lathe(
            bm, base_p, ax,
            [(0.005, rc * 1.05), (0.007, rc * 1.14), (0.009, rc * 1.04),
             (0.011, rc * 1.04), (hc - 0.003, rc * 1.02), (hc, rc * 0.88), (hc + 0.0012, rc * 0.62)],
            n_seg=24, mat_idx=mi, cap_start=True, cap_end=True,
            knurl_amp=rc * 0.048, knurl_z_range=(0.010, hc - 0.003),
        )

        _add_quad_box(bm, base_p + ax * (hc + 0.0035), bx, by, bz, rc * 0.45, 0.0018, 0.0032, mat_idx=mi)

        p_t0 = base_p + ax * (hc + 0.0032) + bx * (rc * 0.42)
        p_t2 = base_p + bx * (rc * 1.65) + by * (0.012 if (idx_f % 2 == 0) else -0.012) - ax * 0.001
        p_t1 = 0.5 * (p_t0 + p_t2) + ax * 0.004 + Vector((0.0, 0.0, -0.016))
        t_pts = _catmull_rom_sample([p_t0, p_t1, p_t2], samples_per_seg=6)
        _add_quad_tube(bm, t_pts, 0.00075, n_seg=6, mat_idx=3, cap_start=True, cap_end=True)

        _add_quad_lathe(bm, p_t2, ax, [(-0.002, 0.0034), (0.0025, 0.0034)], n_seg=8, mat_idx=3)


    plug_specs = [
        ("deec_fwd_plus",  +118.0, -3.96, 0.0140, +1.0),
        ("core_mid_plus",  +66.0,  -3.24, 0.0135, +1.0),
        ("core_aft_plus",  +72.0,  -3.02, 0.0130, +1.0),
        ("comb_aft_plus",  +86.0,  -2.48, 0.0140, -1.0),
        ("comb_hi_plus",   +74.0,  -2.34, 0.0128, +1.0),
        ("tec_hi_plus",    +60.0,  -2.06, 0.0125, -1.0),
        ("core_box_neg",   -52.0,  -3.44, 0.0135, +1.0),
        ("comb_aft_neg",   -96.0,  -2.64, 0.0130, -1.0),
    ]
    plug_anchors = {}
    for key, phi_d, yv, rp, _ in plug_specs:
        base_p, ax = _cyl_ray(phi_d, yv)
        bx, by, bz = _ortho_basis(ax)

        _add_quad_box(bm, base_p + ax * 0.001, bx, by, bz, rp * 1.28, rp * 1.28, 0.0032, mat_idx=2)
        for sx_b in (-1.0, +1.0):
            for sy_b in (-1.0, +1.0):
                bp = base_p + bx * (sx_b * rp * 0.96) + by * (sy_b * rp * 0.96) + ax * 0.0040
                _add_quad_lathe(bm, bp, ax, [(0.0, 0.0022), (0.0026, 0.0022)], n_seg=6, mat_idx=3)

        _add_quad_lathe(
            bm, base_p, ax,
            [(0.003, rp * 0.96), (0.011, rp * 0.96), (0.012, rp * 1.18), (0.023, rp * 1.18),
             (0.024, rp * 0.94), (0.032, rp * 0.88), (0.035, rp * 0.72)],
            n_seg=24, mat_idx=2, cap_start=False, cap_end=True,
            knurl_amp=rp * 0.045, knurl_z_range=(0.012, 0.023),
        )

        _add_quad_lathe(
            bm, base_p, ax,
            [(0.0102, rp * 1.02), (0.0118, rp * 1.02)],
            n_seg=12, mat_idx=0, cap_start=False, cap_end=False,
        )
        plug_anchors[key] = (base_p + ax * 0.033, ax, rp)

    obj = _finalize_detail_mesh_obj(
        "Detail_Core_FODCaps_And_CannonPlugs", bm,
        [m_redcap, m_yelcap, m_olive, m_steel], coll_core, shadow=True,
    )
    return obj, plug_anchors


def _build_deec_nomex_harnesses(coll_core, hull_bvhs, plug_anchors,
                                m_nomex, m_clamp_steel, m_adel_blue, m_adel_org, m_lacing, m_boot):

    bm = bmesh.new()


    def _hull_offset_pt(phi_deg, yv, standoff=0.024):
        pr = math.radians(phi_deg)
        n_rad = Vector((math.sin(pr), 0.0, math.cos(pr)))
        p0 = Vector((0.76 * n_rad.x, yv, 0.76 * n_rad.z))
        hit, _ = _raycast_hull(hull_bvhs, p0, -n_rad, max_dist=0.35)
        if hit is None:
            r_def = 0.540
            return Vector(((r_def + standoff) * n_rad.x, yv, (r_def + standoff) * n_rad.z)), n_rad
        return hit + n_rad * standoff, n_rad

    def _emit_harness_run(start_key, end_key, mid_waypoints, r_loom=0.0068, clamp_color_idx=2):
        p_s, ax_s, _ = plug_anchors[start_key]
        p_e, ax_e, _ = plug_anchors[end_key]
        ctrl = [p_s, p_s + ax_s * 0.020]
        for (ph, yv, st) in mid_waypoints:
            mp, _ = _hull_offset_pt(ph, yv, standoff=st)
            ctrl.append(mp)
        ctrl.extend([p_e + ax_e * 0.020, p_e])
        pts = _catmull_rom_sample(ctrl, samples_per_seg=8)


        _add_quad_tube(bm, pts, r_loom, n_seg=10, mat_idx=0, cap_start=True, cap_end=True)


        for p_end, ax_end in ((p_s, ax_s), (p_e, ax_e)):
            _add_quad_lathe(
                bm, p_end - ax_end * 0.003, ax_end,
                [(0.0, r_loom * 1.48), (0.010, r_loom * 1.42), (0.018, r_loom * 1.18), (0.026, r_loom * 1.06)],
                n_seg=12, mat_idx=5, cap_start=True, cap_end=True,
            )


        arc = [0.0]
        for i in range(1, len(pts)):
            arc.append(arc[-1] + (pts[i] - pts[i - 1]).length)
        total_len = arc[-1]

        def _sample_at_s(s_val):
            for i in range(len(arc) - 1):
                if arc[i] <= s_val <= arc[i + 1]:
                    f = (s_val - arc[i]) / max(1e-6, arc[i + 1] - arc[i])
                    p = pts[i].lerp(pts[i + 1], f)
                    t = (pts[i + 1] - pts[i]).normalized()
                    return p, t
            return pts[-1], (pts[-1] - pts[-2]).normalized()


        s_pos = 0.075
        while s_pos < total_len - 0.075:
            pc, tc = _sample_at_s(s_pos)

            for d_s in (-0.0014, +0.0014):
                _add_quad_lathe(
                    bm, pc + tc * d_s, tc,
                    [(-0.0009, r_loom * 0.98), (-0.0004, r_loom * 1.14),
                     (+0.0004, r_loom * 1.14), (+0.0009, r_loom * 0.98)],
                    n_seg=10, mat_idx=4, cap_start=False, cap_end=False,
                )
            bx, by, _ = _ortho_basis(tc)
            r_out = Vector((pc.x, 0.0, pc.z)).normalized()
            _add_quad_box(bm, pc + r_out * (r_loom * 1.15), bx, by, tc, 0.0018, 0.0018, 0.0022, mat_idx=4)
            s_pos += 0.095


        n_clamps = max(2, int(total_len / 0.24))
        for k_c in range(n_clamps):
            s_c = total_len * ((k_c + 0.5) / float(n_clamps))
            pc, tc = _sample_at_s(s_c)
            n_rad = Vector((pc.x, 0.0, pc.z)).normalized()
            s_side = tc.cross(n_rad)
            s_side = s_side.normalized() if s_side.length > 1e-5 else Vector((0, 0, 1))
            n_in = s_side.cross(tc).normalized()
            if n_in.dot(n_rad) > 0:
                n_in = -n_in

            c_mat = clamp_color_idx if (k_c % 2 == 0) else (3 if clamp_color_idx == 2 else 2)
            _add_quad_lathe(
                bm, pc, tc,
                [(-0.0068, r_loom * 1.00), (-0.0068, r_loom * 1.28),
                 (+0.0068, r_loom * 1.28), (+0.0068, r_loom * 1.00)],
                n_seg=12, mat_idx=c_mat, cap_start=False, cap_end=False,
            )

            _add_quad_lathe(
                bm, pc, tc,
                [(-0.0046, r_loom * 1.26), (-0.0046, r_loom * 1.42),
                 (+0.0046, r_loom * 1.42), (+0.0046, r_loom * 1.26)],
                n_seg=12, mat_idx=1, cap_start=False, cap_end=False,
            )

            foot_p = pc + s_side * (r_loom * 1.55) + n_in * 0.012
            _add_quad_box(bm, pc + s_side * (r_loom * 1.45) + n_in * 0.005, s_side, tc, n_in,
                          0.0075, 0.0046, 0.0095, mat_idx=1)
            _add_quad_lathe(bm, foot_p - n_in * 0.004, -n_in, [(0.0, 0.0032), (0.0036, 0.0032)], n_seg=6, mat_idx=1)


    _emit_harness_run(
        "deec_fwd_plus", "core_mid_plus",
        [(+102.0, -3.76, 0.026), (+82.0, -3.52, 0.026), (+70.0, -3.36, 0.026)],
        r_loom=0.0068, clamp_color_idx=2,
    )

    _emit_harness_run(
        "core_aft_plus", "comb_aft_plus",
        [(+76.0, -2.84, 0.028), (+82.0, -2.64, 0.028)],
        r_loom=0.0065, clamp_color_idx=3,
    )

    _emit_harness_run(
        "comb_hi_plus", "tec_hi_plus",
        [(+68.0, -2.20, 0.026)],
        r_loom=0.0058, clamp_color_idx=2,
    )

    _emit_harness_run(
        "core_box_neg", "comb_aft_neg",
        [(-64.0, -3.22, 0.026), (-80.0, -2.96, 0.026), (-90.0, -2.78, 0.026)],
        r_loom=0.0066, clamp_color_idx=3,
    )

    return _finalize_detail_mesh_obj(
        "Detail_DEEC_Nomex_Harnesses", bm,
        [m_nomex, m_clamp_steel, m_adel_blue, m_adel_org, m_lacing, m_boot],
        coll_core, shadow=True,
    )


def _build_actuator_wiper_rings(coll_core, m_wiper):

    bm = bmesh.new()
    y_exit = -0.7170
    ax_y = Vector((0.0, 1.0, 0.0))


    for k in range(8):
        ang = math.radians(k * 45.0)
        org = Vector((0.5750 * math.cos(ang), y_exit, 0.5750 * math.sin(ang)))
        _add_quad_lathe(
            bm, org, ax_y,
            [(-0.0010, 0.0168), (0.0002, 0.0164), (0.0022, 0.0150),
             (0.0046, 0.0136), (0.0058, 0.0126), (0.0052, 0.01245), (-0.0010, 0.01245), (-0.0010, 0.0168)],
            n_seg=16, mat_idx=0, cap_start=False, cap_end=False,
        )

    org_lv = Vector((0.4781, y_exit, 0.3195))
    _add_quad_lathe(
        bm, org_lv, ax_y,
        [(-0.0008, 0.0086), (0.0002, 0.0082), (0.0020, 0.0072),
         (0.0042, 0.0061), (0.0048, 0.0055), (0.0044, 0.00545), (-0.0008, 0.00545), (-0.0008, 0.0086)],
        n_seg=12, mat_idx=0, cap_start=False, cap_end=False,
    )

    return _finalize_detail_mesh_obj(
        "Detail_Actuator_WiperRings", bm, [m_wiper], coll_core, shadow=True,
    )


def build_service_details(ng_global, ng_masks, ng_wrap):

    coll_core = bpy.data.collections.get("01_Static_Engine_Core") or bpy.context.scene.collection
    coll_stand = bpy.data.collections.get("06_Maintenance_Stand_And_Trailer") or bpy.context.scene.collection

    decals_dir = os.path.join(_THIS_DIR, "assets_decals")
    dd1574_png = os.path.join(decals_dir, "decal_dd1574_serviceable_tag.png")


    m_chrome = _build_detail_metal(
        "Mat_Service_Chrome_Zerk", MAT_FASTENER, ng_global, ng_masks, ng_wrap,
        col_a=0xCCD2DA, col_b=0xA2AAB4, crevice_col=0x22252A,
        metallic=0.99, r_min=0.09, r_max=0.20, rough_floor=0.08, f82_tint=0xE8EEF4,
    )
    m_olive = _build_detail_metal(
        "Mat_Service_CannonPlug_OliveCad", MAT_CONNECTOR_38999, ng_global, ng_masks, ng_wrap,
        col_a=0x6A6C4C, col_b=0x484A34, crevice_col=0x1C1D14,
        metallic=0.92, r_min=0.22, r_max=0.38, rough_floor=0.16, f82_tint=0xDCE0C8,
    )
    m_clamp_steel = _build_detail_metal(
        "Mat_Service_ClampSteel", MAT_STAND_STEEL, ng_global, ng_masks, ng_wrap,
        col_a=0xB2B8C0, col_b=0x88909A, crevice_col=0x1E2024,
        metallic=0.97, r_min=0.16, r_max=0.32, rough_floor=0.14, f82_tint=0xE4EAF0,
    )
    m_brass = _build_detail_metal(
        "Mat_Service_Brass_Eyelet", MAT_TRAILER_YELLOW, ng_global, ng_masks, ng_wrap,
        col_a=0xBA9648, col_b=0x84682E, crevice_col=0x2A200E,
        metallic=0.96, r_min=0.18, r_max=0.34, rough_floor=0.14, f82_tint=0xF6E4B4,
    )
    m_redcap = _build_detail_dielectric(
        "Mat_Service_FOD_RedCap", MAT_STAND_ORANGE, ng_global, ng_masks, ng_wrap,
        col_a=0xB81E18, col_b=0x8A1510, crevice_col=0x380806,
        r_min=0.34, r_max=0.48, coat_w=0.18, coat_r=0.24, bump_str=0.030,
    )
    m_yelcap = _build_detail_dielectric(
        "Mat_Service_FOD_YellowCap", MAT_TRAILER_YELLOW, ng_global, ng_masks, ng_wrap,
        col_a=0xD8A816, col_b=0xAA8010, crevice_col=0x443206,
        r_min=0.34, r_max=0.48, coat_w=0.18, coat_r=0.24, bump_str=0.030,
    )
    m_nomex = _build_detail_dielectric(
        "Mat_Service_Nomex_BraidHarness", MAT_YELLOW_HARNESS, ng_global, ng_masks, ng_wrap,
        col_a=0x262930, col_b=0x1A1D22, crevice_col=0x101216,
        r_min=0.68, r_max=0.84, coat_w=0.0, weave_scale=185.0, bump_str=0.12, spec_ior=0.38,
    )
    m_adel_blue = _build_detail_dielectric(
        "Mat_Service_Adel_BlueSilicone", MAT_ADEL_CUSHION, ng_global, ng_masks, ng_wrap,
        col_a=0x1E5EC2, col_b=0x144492, crevice_col=0x0A2248,
        r_min=0.42, r_max=0.58, coat_w=0.0, bump_str=0.025, spec_ior=0.44,
    )
    m_adel_org = _build_detail_dielectric(
        "Mat_Service_Adel_OrangeSilicone", MAT_STAND_ORANGE, ng_global, ng_masks, ng_wrap,
        col_a=0xD44C18, col_b=0xA23610, crevice_col=0x481806,
        r_min=0.42, r_max=0.58, coat_w=0.0, bump_str=0.025, spec_ior=0.44,
    )
    m_lacing = _build_detail_dielectric(
        "Mat_Service_WaxedLacingCord", MAT_BRAID_HOSE, ng_global, ng_masks, ng_wrap,
        col_a=0xD6CEBC, col_b=0xB2A996, crevice_col=0x585244,
        r_min=0.54, r_max=0.72, coat_w=0.14, coat_r=0.32, bump_str=0.055, spec_ior=0.42,
    )
    m_boot = _build_detail_dielectric(
        "Mat_Service_HeatShrink_Boot", MAT_RUBBER_DARK, ng_global, ng_masks, ng_wrap,
        col_a=0x24272D, col_b=0x1A1D22, crevice_col=0x101216,
        r_min=0.48, r_max=0.64, coat_w=0.0, bump_str=0.032, spec_ior=0.42,
    )
    m_wiper = _build_detail_dielectric(
        "Mat_Service_WiperGreaseRing", MAT_NOZ_ACT_AB, ng_global, ng_masks, ng_wrap,
        col_a=0x241E18, col_b=0x16120E, crevice_col=0x0D0B09,
        r_min=0.22, r_max=0.62, coat_w=0.55, coat_r=0.16, bump_str=0.085, spec_ior=0.48,
    )
    m_dd_card = _build_service_tag_card_mat(ng_global, ng_masks, ng_wrap, dd1574_png)
    m_pvc     = _build_pvc_sleeve_mat(ng_global, ng_masks, ng_wrap)


    hull_bvhs = _build_bvh_for_objects([
        "Fwd_02_Fan_Case_VaneLinkage_And_DEEC",
        "Fwd_03_Core_Isogrid_Duct_And_Splitters",
        "Fwd_08_Combustor_Turbine_Case_Fuel_And_Igniters",
        "Static_Forward_TEC_And_Mount",
        "Static_Forward_Waffle_Duct",
    ])


    o_zerk = _build_wheel_grease_zerks(coll_stand, m_chrome, m_redcap)
    o_tag  = _build_dd1574_serviceable_tag(coll_stand, m_dd_card, m_pvc, m_clamp_steel, m_brass)
    o_fod, plug_anchors = _build_fod_caps_and_cannon_plugs(
        coll_core, hull_bvhs, m_redcap, m_yelcap, m_olive, m_clamp_steel,
    )
    o_harn = _build_deec_nomex_harnesses(
        coll_core, hull_bvhs, plug_anchors,
        m_nomex, m_clamp_steel, m_adel_blue, m_adel_org, m_lacing, m_boot,
    )
    o_wip  = _build_actuator_wiper_rings(coll_core, m_wiper)

    tot_f = sum(len(o.data.polygons) for o in (o_zerk, o_tag, o_fod, o_harn, o_wip))
    print(
        f"[Service Detailing] Built 5 quad mesh objects (+{tot_f:,} faces): "
        f"WheelZerks={len(o_zerk.data.polygons)} | DD1574Tag={len(o_tag.data.polygons)} | "
        f"FOD+Cannon={len(o_fod.data.polygons)} | NomexHarness={len(o_harn.data.polygons)} | "
        f"WiperRings={len(o_wip.data.polygons)}",
        flush=True,
    )
    return (o_zerk, o_tag, o_fod, o_harn, o_wip)
