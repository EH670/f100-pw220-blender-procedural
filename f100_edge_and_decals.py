import math
import os
import time
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from f100_geom_primitives import (
    rgba,
    COLOR_ID_SPECS,
    MAT_TI_CASE, MAT_BRAID_HOSE, MAT_YELLOW_HARNESS,
    MAT_PLATE_BASE, MAT_3D_TEXT, MAT_THREAD, MAT_NOZ_ACT_AB,
    MAT_SPINNER_IGV, MAT_FASTENER, MAT_BRASS_FITTING, MAT_MACHINED_ALU,
    MAT_RIGID_TUBE, MAT_STAND_STEEL, MAT_TRAILER_YELLOW, MAT_STAND_ORANGE,
    MAT_RUBBER_DARK, MAT_SHROUD_GOLD,
    attach_weighted_normal_modifier,
    bm_apply_hard_surface_smoothing,
)


def _set_point_float_attr(me, attr_name: str, values: np.ndarray):
    attr = me.attributes.get(attr_name)
    if attr is None or attr.data_type != 'FLOAT' or attr.domain != 'POINT':
        if attr is not None:
            me.attributes.remove(attr)
        attr = me.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
    attr.data.foreach_set("value", np.ascontiguousarray(values, dtype=np.float32))


def _set_face_float_attr(me, attr_name: str, values: np.ndarray):
    attr = me.attributes.get(attr_name)
    if attr is None or attr.data_type != 'FLOAT' or attr.domain != 'FACE':
        if attr is not None:
            me.attributes.remove(attr)
        attr = me.attributes.new(name=attr_name, type='FLOAT', domain='FACE')
    attr.data.foreach_set("value", np.ascontiguousarray(values, dtype=np.float32))


def _compute_mesh_exposure(mesh_name: str, co: np.ndarray) -> np.ndarray:

    n_v = co.shape[0]
    exp = np.full(n_v, 0.25, dtype=np.float32)
    x = co[:, 0]
    y = co[:, 1]
    z = co[:, 2]
    r = np.hypot(x, z)

    if mesh_name == "Stand_Cradle_01_Longitudinal_Rails_And_Crossbeams":

        m_track = (z >= -0.758) & (z <= -0.680) & (np.abs(x) >= 0.58)
        exp = np.where(m_track, 0.92, exp)

        m_fork = (((np.abs(y + 3.22) <= 0.16) | (np.abs(y + 2.52) <= 0.16)) & (z <= -0.892))
        exp = np.where(m_fork, 1.00, exp)

        m_end = (y <= -4.22) | (y >= -0.66)
        exp = np.where(m_end, np.maximum(exp, 0.82), exp)

    elif mesh_name.startswith("Stand_Cradle_0"):

        m_low = (z <= -0.50)
        m_trun = (z >= -0.18) & (z <= 0.10) & (np.abs(x) >= 0.56)
        m_outer = (r >= 0.675)
        exp = np.where(m_low, 0.88, exp)
        exp = np.where(m_trun, np.maximum(exp, 0.90), exp)
        exp = np.where(m_outer, np.maximum(exp, 0.72), exp)

    elif mesh_name.startswith("Stand_Trailer_"):

        m_bumper = (y <= -4.18) | (y >= -0.85)
        m_low = (z <= -1.14)
        exp = np.where(m_bumper, 0.92, exp)
        exp = np.where(m_low, np.maximum(exp, 0.84), exp)

    elif mesh_name == "Static_Nozzle_Outer_Shroud":

        aft_t = np.clip((y - (-0.30)) / 0.23, 0.0, 1.0)
        exp = np.maximum(exp, 0.30 + 0.68 * aft_t)
        m_band = (y >= -0.485) & (y <= -0.355)
        exp = np.where(m_band, np.maximum(exp, 0.88), exp)

    elif mesh_name in (
        "Mesh_DivMaster", "Mesh_DivSeal", "Mesh_ConvMaster", "Mesh_ConvSeal",
        "Mesh_BalanceFlap", "Mesh_SliderCarriage", "Mesh_Bellcrank", "Mesh_Diagonal_Strut",
        "Mesh_TwinFork_PullRod", "Mesh_Boomerang", "Mesh_ShortLink",
    ):

        lip_t = np.clip((y - 0.52) / 0.38, 0.0, 1.0)
        exp = np.maximum(exp, 0.35 + 0.65 * lip_t)
        m_rail = (np.abs(x) <= 0.032) & (y >= 0.36) & (y <= 0.82)
        exp = np.where(m_rail, np.maximum(exp, 0.94), exp)

    elif mesh_name in ("Fwd_01_Inlet_Spinner_Pitot_And_Sensors", "Mesh_Fan_Rotor_3Stage"):

        le_t = np.clip((-4.02 - y) / 0.45, 0.0, 1.0)
        exp = np.maximum(exp, 0.28 + 0.68 * le_t)

    elif mesh_name in ("Static_Base_Casing_Ribbed", "Static_Forward_Waffle_Duct", "Static_Forward_TEC_And_Mount"):

        r_prom = np.clip((r - 0.542) / 0.025, 0.0, 1.0)
        exp = np.maximum(exp, 0.25 + 0.65 * r_prom)

    else:

        ventral = np.clip((-0.10 - z) / 0.48, 0.0, 1.0)
        exp = np.maximum(exp, 0.25 + 0.55 * ventral)

    return np.clip(exp, 0.0, 1.0).astype(np.float32)


def bake_single_mesh_topology_attributes(me):

    n_v = len(me.vertices)
    n_f = len(me.polygons)
    if n_v == 0 or n_f == 0:
        return

    co = np.empty(n_v * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape((n_v, 3))

    exposure = _compute_mesh_exposure(me.name, co)

    n_l = len(me.loops)

    if n_l == n_f * 4:
        l_verts = np.empty(n_l, dtype=np.int32)
        me.loops.foreach_get("vertex_index", l_verts)
        q = l_verts.reshape((n_f, 4))

        v0 = co[q[:, 0]]
        v1 = co[q[:, 1]]
        v2 = co[q[:, 2]]
        v3 = co[q[:, 3]]


        fn = np.cross(v2 - v0, v3 - v1)
        fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
        fn = fn / np.maximum(fn_len, 1e-12)
        fc = 0.25 * (v0 + v1 + v2 + v3)


        e01 = np.linalg.norm(v1 - v0, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)
        e23 = np.linalg.norm(v3 - v2, axis=1)
        e30 = np.linalg.norm(v0 - v3, axis=1)
        wp_02 = 0.5 * (e12 + e30)
        wp_13 = 0.5 * (e01 + e23)
        w_perp_he = np.stack((wp_02, wp_13, wp_02, wp_13), axis=1).ravel()


        u = q.ravel()
        v = np.roll(q, -1, axis=1).ravel()
        fidx = np.repeat(np.arange(n_f, dtype=np.int32), 4)

        ea = np.minimum(u, v).astype(np.int64)
        eb = np.maximum(u, v).astype(np.int64)
        key = ea * np.int64(n_v) + eb

        order = np.argsort(key)
        k_s = key[order]
        f_s = fidx[order]
        ea_s = ea[order]
        eb_s = eb[order]
        wp_s = w_perp_he[order]


        same = (k_s[:-1] == k_s[1:])
        idx_p = np.nonzero(same)[0]

        f0 = f_s[idx_p]
        f1 = f_s[idx_p + 1]
        va = ea_s[idx_p].astype(np.int32)
        vb = eb_s[idx_p].astype(np.int32)
        wp0 = wp_s[idx_p]
        wp1 = wp_s[idx_p + 1]

        n0 = fn[f0]
        n1 = fn[f1]
        c0 = fc[f0]
        c1 = fc[f1]

        dot_n = np.clip(np.sum(n0 * n1, axis=1), -1.0, 1.0)
        bend = np.arccos(dot_n)


        dc = c1 - c0
        sign_val = -(np.sum(n0 * dc, axis=1) - np.sum(n1 * dc, axis=1))


        conv_str = np.clip((bend - 0.54) / 0.52, 0.0, 1.0).astype(np.float32)
        conv_e = np.where((sign_val > 1e-7) & (bend >= 0.54), conv_str, 0.0).astype(np.float32)

        v_conv = np.zeros(n_v, dtype=np.float32)
        np.maximum.at(v_conv, va, conv_e)
        np.maximum.at(v_conv, vb, conv_e)


        conc_str = np.clip((bend - 0.34) / 0.78, 0.0, 1.0).astype(np.float32)
        conc_e = np.where(sign_val < -1e-7, conc_str, 0.0).astype(np.float32)

        v_conc = np.zeros(n_v, dtype=np.float32)
        np.maximum.at(v_conc, va, conc_e)
        np.maximum.at(v_conc, vb, conc_e)


        e_len = np.linalg.norm(co[vb] - co[va], axis=1)
        w_conv_e = np.exp(-((e_len / 0.006) ** 2)).astype(np.float32)
        w_conc_e = np.exp(-((e_len / 0.010) ** 2)).astype(np.float32)

        w_conv_sum = np.zeros(n_v, dtype=np.float32)
        np.add.at(w_conv_sum, va, w_conv_e)
        np.add.at(w_conv_sum, vb, w_conv_e)
        inv_wc = 1.0 / np.maximum(w_conv_sum, 1e-6)

        for keep_w in (0.65, 0.65):
            acc_c = np.zeros(n_v, dtype=np.float32)
            np.add.at(acc_c, va, v_conv[vb] * w_conv_e)
            np.add.at(acc_c, vb, v_conv[va] * w_conv_e)
            has_nbr = (w_conv_sum > 1e-4)
            v_conv = np.where(has_nbr, keep_w * v_conv + (1.0 - keep_w) * (acc_c * inv_wc), v_conv)
        v_conv = np.clip(v_conv, 0.0, 1.0)

        w_conc_sum = np.zeros(n_v, dtype=np.float32)
        np.add.at(w_conc_sum, va, w_conc_e)
        np.add.at(w_conc_sum, vb, w_conc_e)
        inv_wn = 1.0 / np.maximum(w_conc_sum, 1e-6)

        for decay in (0.72, 0.58):
            acc = np.zeros(n_v, dtype=np.float32)
            np.add.at(acc, va, v_conc[vb] * w_conc_e)
            np.add.at(acc, vb, v_conc[va] * w_conc_e)
            v_conc = np.maximum(v_conc, (acc * inv_wn) * decay)

        v_conc = np.clip(v_conc, 0.0, 1.0)
    else:
        v_conv = np.zeros(n_v, dtype=np.float32)
        v_conc = np.zeros(n_v, dtype=np.float32)

    _set_point_float_attr(me, "f15_convexity", v_conv)
    _set_point_float_attr(me, "f15_concavity", v_conc)
    _set_point_float_attr(me, "f15_exposure", exposure)


def bake_all_mesh_geometry_attributes():

    t0 = time.perf_counter()
    count = 0
    tot_v = 0
    for me in bpy.data.meshes:
        if me.name.startswith("Studio_Ground") or me.name.startswith("Decals_"):
            continue
        bake_single_mesh_topology_attributes(me)
        count += 1
        tot_v += len(me.vertices)
    dt = time.perf_counter() - t0
    print(f"[Topology Attributes] Baked {count} meshes / {tot_v:,} vertices -> f15_convexity, f15_concavity, f15_exposure ({dt:.2f}s)")


def _new_node(nt, bl_idname, loc=(0, 0), name=None):
    n = nt.nodes.new(bl_idname)
    n.location = loc
    if name:
        n.name = name
    return n


def _assign_faces_to_new_slot(me, face_indices: list, new_mat):
    if not face_indices:
        return 0
    slot_idx = -1
    for i, m in enumerate(me.materials):
        if m == new_mat:
            slot_idx = i
            break
    if slot_idx < 0:
        me.materials.append(new_mat)
        slot_idx = len(me.materials) - 1
    for fi in face_indices:
        me.polygons[fi].material_index = slot_idx
    return len(face_indices)


def build_ruby_reflector_material(ng_global, ng_masks, ng_wrap):

    mat = bpy.data.materials.get("Mat_ID30_Trailer_RubyReflector")
    if mat is None:
        mat = bpy.data.materials.new("Mat_ID30_Trailer_RubyReflector")
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1420, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1420, -340))
    ctrl.node_tree = ng_global

    masks = _new_node(nt, "ShaderNodeGroup", (-1080, 180))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.022
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00025


    vor_p = _new_node(nt, "ShaderNodeTexVoronoi", (-1080, -240))
    vor_p.feature = 'F1'
    vor_p.distance = 'MANHATTAN'
    vor_p.inputs["Scale"].default_value = 460.0
    vor_p.inputs["Randomness"].default_value = 0.05
    nt.links.new(tc.outputs["Object"], vor_p.inputs["Vector"])

    bw_cell = _new_node(nt, "ShaderNodeRGBToBW", (-840, -360))
    nt.links.new(vor_p.outputs["Color"], bw_cell.inputs["Color"])


    m_pr = _new_node(nt, "ShaderNodeMath", (-660, -220)); m_pr.operation = 'MULTIPLY_ADD'; m_pr.use_clamp = True
    m_pr.inputs[1].default_value = 0.65
    nt.links.new(vor_p.outputs["Distance"], m_pr.inputs[0])
    m_cl = _new_node(nt, "ShaderNodeMath", (-820, -200)); m_cl.operation = 'MULTIPLY'; m_cl.inputs[1].default_value = 0.45
    nt.links.new(bw_cell.outputs["Val"], m_cl.inputs[0])
    nt.links.new(m_cl.outputs[0], m_pr.inputs[2])

    ramp_r = _new_node(nt, "ShaderNodeValToRGB", (-440, 180))
    ramp_r.color_ramp.elements[0].position = 0.14
    ramp_r.color_ramp.elements[0].color = rgba(0x5A0406)
    e_mid = ramp_r.color_ramp.elements.new(0.55)
    e_mid.color = rgba(0xC81012)
    ramp_r.color_ramp.elements[2].position = 0.88
    ramp_r.color_ramp.elements[2].color = rgba(0xD8241C)
    nt.links.new(m_pr.outputs[0], ramp_r.inputs["Fac"])


    cr_w0 = _new_node(nt, "ShaderNodeMath", (-720, 340)); cr_w0.operation = 'MULTIPLY'; cr_w0.use_clamp = True
    cr_w0.inputs[1].default_value = 0.55
    nt.links.new(masks.outputs["Crevice_Mask"], cr_w0.inputs[0])
    cr_w = _new_node(nt, "ShaderNodeMath", (-540, 340)); cr_w.operation = 'MULTIPLY'; cr_w.use_clamp = True
    nt.links.new(cr_w0.outputs[0], cr_w.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], cr_w.inputs[1])
    mix_cr = _new_node(nt, "ShaderNodeMix", (-200, 180)); mix_cr.data_type = 'RGBA'
    mix_cr.inputs[7].default_value = rgba(0x281010)
    nt.links.new(cr_w.outputs[0], mix_cr.inputs[0])
    nt.links.new(ramp_r.outputs["Color"], mix_cr.inputs[6])

    sc_w0 = _new_node(nt, "ShaderNodeMath", (-720, 40)); sc_w0.operation = 'MULTIPLY'; sc_w0.use_clamp = True
    sc_w0.inputs[1].default_value = 0.42
    nt.links.new(masks.outputs["Scratch_Mask"], sc_w0.inputs[0])
    sc_w = _new_node(nt, "ShaderNodeMath", (-540, 40)); sc_w.operation = 'MULTIPLY'; sc_w.use_clamp = True
    nt.links.new(sc_w0.outputs[0], sc_w.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], sc_w.inputs[1])
    mix_sc_c = _new_node(nt, "ShaderNodeMix", (-20, 180)); mix_sc_c.data_type = 'RGBA'
    mix_sc_c.inputs[7].default_value = rgba(0xC49898)
    nt.links.new(sc_w.outputs[0], mix_sc_c.inputs[0])
    nt.links.new(mix_cr.outputs[2], mix_sc_c.inputs[6])


    bump_p = _new_node(nt, "ShaderNodeBump", (-260, -200))
    bump_p.inputs["Strength"].default_value = 0.26
    bump_p.inputs["Distance"].default_value = 0.00045
    nt.links.new(vor_p.outputs["Distance"], bump_p.inputs["Height"])
    nt.links.new(masks.outputs["Bevel_Normal"], bump_p.inputs["Normal"])

    r_lens = _new_node(nt, "ShaderNodeMix", (-20, -40)); r_lens.data_type = 'FLOAT'
    r_lens.inputs[2].default_value = 0.10; r_lens.inputs[3].default_value = 0.44
    nt.links.new(sc_w.outputs[0], r_lens.inputs[0])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (220, 120))
    nt.links.new(mix_sc_c.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    nt.links.new(r_lens.outputs[0], bsdf.inputs["Roughness"])
    bsdf.inputs["IOR"].default_value = 1.58
    bsdf.inputs["Specular IOR Level"].default_value = 0.50
    bsdf.inputs["Coat Weight"].default_value = 1.00
    bsdf.inputs["Coat Roughness"].default_value = 0.030
    bsdf.inputs["Coat IOR"].default_value = 1.58
    nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Coat Normal"])
    bsdf.inputs["Emission Strength"].default_value = 0.0
    nt.links.new(bump_p.outputs["Normal"], bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (540, 120))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(COLOR_ID_SPECS[MAT_THREAD][1])
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (800, 120))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(COLOR_ID_SPECS[MAT_THREAD][1])
    mat["f15_pbr_hex"] = int(0xCC1C12)
    mat.diffuse_color = rgba(0xCC1C12)
    return mat


def build_actuator_dual_band_materials(ng_global, ng_masks, ng_wrap):

    def _make_dual(m_name, id_mid, col_fwd, col_aft, met_fwd, met_aft, r_fwd, r_aft, vp_hex):
        mat = bpy.data.materials.get(m_name)
        if mat is None:
            mat = bpy.data.materials.new(m_name)
        nt = mat.node_tree
        nt.nodes.clear()

        tc = _new_node(nt, "ShaderNodeTexCoord", (-1380, 0))
        ctrl = _new_node(nt, "ShaderNodeGroup", (-1380, -340))
        ctrl.node_tree = ng_global

        masks = _new_node(nt, "ShaderNodeGroup", (-1060, 220))
        masks.node_tree = ng_masks
        nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
        masks.inputs["AO_Dist"].default_value = 0.016
        masks.inputs["Fine_Bevel_Rad"].default_value = 0.00018
        masks.inputs["Micro_Scale"].default_value = 460.0

        sep = _new_node(nt, "ShaderNodeSeparateXYZ", (-1060, -180))
        nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
        cmp = _new_node(nt, "ShaderNodeMath", (-820, -180))
        cmp.operation = 'GREATER_THAN'
        cmp.inputs[1].default_value = -1.001
        nt.links.new(sep.outputs["Y"], cmp.inputs[0])


        mp_br = _new_node(nt, "ShaderNodeMapping", (-1060, -440))
        mp_br.inputs["Scale"].default_value = (12.0, 220.0, 12.0)
        nt.links.new(tc.outputs["Object"], mp_br.inputs["Vector"])
        n_br = _new_node(nt, "ShaderNodeTexNoise", (-840, -440))
        n_br.inputs["Scale"].default_value = 1.0
        n_br.inputs["Detail"].default_value = 6.0
        n_br.inputs["Roughness"].default_value = 0.62
        nt.links.new(mp_br.outputs["Vector"], n_br.inputs["Vector"])

        mix_c = _new_node(nt, "ShaderNodeMix", (-500, 160))
        mix_c.data_type = 'RGBA'
        mix_c.inputs[6].default_value = rgba(col_fwd)
        mix_c.inputs[7].default_value = rgba(col_aft)
        nt.links.new(cmp.outputs[0], mix_c.inputs[0])


        m_gw0 = _new_node(nt, "ShaderNodeMath", (-600, -20)); m_gw0.operation = 'MULTIPLY'; m_gw0.use_clamp = True
        m_gw0.inputs[1].default_value = 0.44
        nt.links.new(masks.outputs["Crevice_Mask"], m_gw0.inputs[0])
        m_gw = _new_node(nt, "ShaderNodeMath", (-420, -20)); m_gw.operation = 'MULTIPLY'; m_gw.use_clamp = True
        nt.links.new(m_gw0.outputs[0], m_gw.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], m_gw.inputs[1])

        mix_gr = _new_node(nt, "ShaderNodeMix", (-220, 160))
        mix_gr.data_type = 'RGBA'
        mix_gr.inputs[7].default_value = rgba(0x221E16)
        nt.links.new(m_gw.outputs[0], mix_gr.inputs[0])
        nt.links.new(mix_c.outputs[2], mix_gr.inputs[6])


        ew_m = _new_node(nt, "ShaderNodeMath", (-420, 80)); ew_m.operation = 'MULTIPLY'; ew_m.use_clamp = True
        ew_m.inputs[1].default_value = 0.34
        nt.links.new(masks.outputs["EdgeWear_Mask"], ew_m.inputs[0])
        mix_ew = _new_node(nt, "ShaderNodeMix", (-20, 160)); mix_ew.data_type = 'RGBA'
        mix_ew.inputs[7].default_value = rgba(0xCDD3DA)
        nt.links.new(ew_m.outputs[0], mix_ew.inputs[0])
        nt.links.new(mix_gr.outputs[2], mix_ew.inputs[6])

        mix_m = _new_node(nt, "ShaderNodeMix", (-440, -180))
        mix_m.data_type = 'FLOAT'
        mix_m.inputs[2].default_value = met_fwd
        mix_m.inputs[3].default_value = met_aft
        nt.links.new(cmp.outputs[0], mix_m.inputs[0])

        mix_r0 = _new_node(nt, "ShaderNodeMix", (-440, -340))
        mix_r0.data_type = 'FLOAT'
        mix_r0.inputs[2].default_value = r_fwd
        mix_r0.inputs[3].default_value = r_aft
        nt.links.new(cmp.outputs[0], mix_r0.inputs[0])

        r_jit = _new_node(nt, "ShaderNodeMapRange", (-440, -480)); r_jit.clamp = True
        r_jit.inputs[1].default_value = 0.14; r_jit.inputs[2].default_value = 0.86
        r_jit.inputs[3].default_value = -0.09; r_jit.inputs[4].default_value = +0.12
        nt.links.new(masks.outputs["Metal_Rough_Breakup"], r_jit.inputs[0])
        mix_r = _new_node(nt, "ShaderNodeMath", (-200, -340)); mix_r.operation = 'ADD'; mix_r.use_clamp = True
        nt.links.new(mix_r0.outputs[0], mix_r.inputs[0])
        nt.links.new(r_jit.outputs[0], mix_r.inputs[1])


        b_h = _new_node(nt, "ShaderNodeMix", (-440, -600)); b_h.data_type = 'FLOAT'
        nt.links.new(cmp.outputs[0], b_h.inputs[0])
        nt.links.new(masks.outputs["Micro_Grain"], b_h.inputs[2])
        nt.links.new(n_br.outputs["Factor"], b_h.inputs[3])
        b_n = _new_node(nt, "ShaderNodeBump", (-180, -520))
        b_n.inputs["Strength"].default_value = 0.042
        b_n.inputs["Distance"].default_value = 0.00028
        nt.links.new(b_h.outputs[0], b_n.inputs["Height"])
        nt.links.new(masks.outputs["Bevel_Normal"], b_n.inputs["Normal"])


        mix_cw = _new_node(nt, "ShaderNodeMix", (-200, -40)); mix_cw.data_type = 'FLOAT'
        mix_cw.inputs[2].default_value = 0.46; mix_cw.inputs[3].default_value = 0.06
        nt.links.new(cmp.outputs[0], mix_cw.inputs[0])
        mix_crg = _new_node(nt, "ShaderNodeMix", (-200, -110)); mix_crg.data_type = 'FLOAT'
        mix_crg.inputs[2].default_value = 0.14; mix_crg.inputs[3].default_value = 0.28
        nt.links.new(cmp.outputs[0], mix_crg.inputs[0])


        bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (200, 120))
        nt.links.new(mix_ew.outputs[2], bsdf.inputs["Base Color"])
        nt.links.new(mix_m.outputs[0], bsdf.inputs["Metallic"])
        nt.links.new(mix_r.outputs[0], bsdf.inputs["Roughness"])
        if "Specular Tint" in bsdf.inputs:
            mix_st = _new_node(nt, "ShaderNodeMix", (-20, 40)); mix_st.data_type = 'RGBA'
            mix_st.inputs[6].default_value = (1.0, 1.0, 1.0, 1.0)
            mix_st.inputs[7].default_value = rgba(0xDCE2EA)
            nt.links.new(cmp.outputs[0], mix_st.inputs[0])
            nt.links.new(mix_st.outputs[2], bsdf.inputs["Specular Tint"])
        bsdf.inputs["Anisotropic"].default_value = 0.0
        nt.links.new(mix_cw.outputs[0], bsdf.inputs["Coat Weight"])
        nt.links.new(mix_crg.outputs[0], bsdf.inputs["Coat Roughness"])
        if "Coat Normal" in bsdf.inputs:
            nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Coat Normal"])
        nt.links.new(b_n.outputs["Normal"], bsdf.inputs["Normal"])

        wrap = _new_node(nt, "ShaderNodeGroup", (540, 120))
        wrap.node_tree = ng_wrap
        nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
        wrap.inputs["ID_Color"].default_value = rgba(COLOR_ID_SPECS[id_mid][1])
        wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
        nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
        nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

        out = _new_node(nt, "ShaderNodeOutputMaterial", (800, 120))
        nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

        mat["f15_id_hex"] = int(COLOR_ID_SPECS[id_mid][1])
        mat["f15_pbr_hex"] = int(vp_hex)
        mat.diffuse_color = rgba(vp_hex)
        return mat


    m_band = _make_dual(
        "Mat_ID18e_Actuator_DualBand", MAT_PLATE_BASE,
        col_fwd=0xD89A0E, col_aft=0x9098A2, met_fwd=0.0, met_aft=0.96, r_fwd=0.38, r_aft=0.28, vp_hex=0xD89A0E,
    )
    m_txt = _make_dual(
        "Mat_ID19f_Actuator_DualText", MAT_3D_TEXT,
        col_fwd=0x16181C, col_aft=0x262A30, met_fwd=0.0, met_aft=0.0, r_fwd=0.45, r_aft=0.46, vp_hex=0x1C1E22,
    )
    return m_band, m_txt


def build_actuator_mos2_struts_material(ng_global, ng_masks, ng_wrap):

    mat = bpy.data.materials.get("Mat_ID24c_Actuator_MoS2_Struts")
    if mat is None:
        mat = bpy.data.materials.new("Mat_ID24c_Actuator_MoS2_Struts")
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1340, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1340, -320))
    ctrl.node_tree = ng_global

    masks = _new_node(nt, "ShaderNodeGroup", (-1020, 180))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.034
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00028
    masks.inputs["Edge_Bevel_Rad"].default_value = 0.0024
    masks.inputs["Edge_Gain"].default_value = 1.85
    masks.inputs["Grime_Scale"].default_value = 18.0
    masks.inputs["Macro_Scale"].default_value = 6.5
    masks.inputs["Micro_Scale"].default_value = 540.0


    mix_base = _new_node(nt, "ShaderNodeMix", (-640, 260))
    mix_base.data_type = 'RGBA'
    mix_base.inputs[6].default_value = rgba(0x4E545E)
    mix_base.inputs[7].default_value = rgba(0x363B42)
    nt.links.new(masks.outputs["Macro_Mottle"], mix_base.inputs[0])


    cr_w0 = _new_node(nt, "ShaderNodeMath", (-820, -40)); cr_w0.operation = 'MULTIPLY'; cr_w0.use_clamp = True
    cr_w0.inputs[1].default_value = 0.62
    nt.links.new(masks.outputs["Crevice_Mask"], cr_w0.inputs[0])
    cr_w = _new_node(nt, "ShaderNodeMath", (-640, -40)); cr_w.operation = 'MULTIPLY'; cr_w.use_clamp = True
    nt.links.new(cr_w0.outputs[0], cr_w.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], cr_w.inputs[1])
    mix_cr = _new_node(nt, "ShaderNodeMix", (-400, 240))
    mix_cr.data_type = 'RGBA'
    mix_cr.inputs[7].default_value = rgba(0x181B20)
    nt.links.new(cr_w.outputs[0], mix_cr.inputs[0])
    nt.links.new(mix_base.outputs[2], mix_cr.inputs[6])


    ew_drv = _new_node(nt, "ShaderNodeMath", (-760, -240)); ew_drv.operation = 'MULTIPLY_ADD'
    ew_drv.inputs[1].default_value = 0.30
    nt.links.new(ctrl.outputs["SurfaceErosion"], ew_drv.inputs[0])
    ew_b = _new_node(nt, "ShaderNodeMath", (-940, -240)); ew_b.operation = 'MULTIPLY'; ew_b.inputs[1].default_value = 0.38
    nt.links.new(ctrl.outputs["EdgeWear"], ew_b.inputs[0])
    nt.links.new(ew_b.outputs[0], ew_drv.inputs[2])

    ew_gate = _new_node(nt, "ShaderNodeMapRange", (-740, -140))
    ew_gate.clamp = True
    ew_gate.inputs[1].default_value = 0.30
    ew_gate.inputs[2].default_value = 0.86
    ew_gate.inputs[3].default_value = 0.0
    ew_gate.inputs[4].default_value = 0.72
    nt.links.new(masks.outputs["EdgeWear_Mask"], ew_gate.inputs[0])

    ew_m = _new_node(nt, "ShaderNodeMath", (-520, -180)); ew_m.operation = 'MULTIPLY'; ew_m.use_clamp = True
    nt.links.new(ew_gate.outputs[0], ew_m.inputs[0])
    nt.links.new(ew_drv.outputs[0], ew_m.inputs[1])

    mix_ew = _new_node(nt, "ShaderNodeMix", (-160, 220))
    mix_ew.data_type = 'RGBA'
    mix_ew.inputs[7].default_value = rgba(0x6E747C)
    nt.links.new(ew_m.outputs[0], mix_ew.inputs[0])
    nt.links.new(mix_cr.outputs[2], mix_ew.inputs[6])


    r_src24c = _new_node(nt, "ShaderNodeMix", (-660, -20)); r_src24c.data_type = 'FLOAT'
    r_src24c.inputs[0].default_value = 0.72
    nt.links.new(masks.outputs["Macro_Mottle"], r_src24c.inputs[2])
    nt.links.new(masks.outputs["Metal_Rough_Breakup"], r_src24c.inputs[3])

    r_m = _new_node(nt, "ShaderNodeMapRange", (-460, -20))
    r_m.clamp = True; r_m.inputs[1].default_value = 0.14; r_m.inputs[2].default_value = 0.86
    r_m.inputs[3].default_value = 0.22; r_m.inputs[4].default_value = 0.46
    nt.links.new(r_src24c.outputs[0], r_m.inputs[0])

    smg_r = _new_node(nt, "ShaderNodeMapRange", (-460, -130))
    smg_r.clamp = True; smg_r.inputs[1].default_value = 0.0; smg_r.inputs[2].default_value = 1.0
    smg_r.inputs[3].default_value = -0.026; smg_r.inputs[4].default_value = +0.085
    nt.links.new(masks.outputs["Photo_Smudge"], smg_r.inputs[0])
    r_m_smg = _new_node(nt, "ShaderNodeMath", (-340, -20)); r_m_smg.operation = 'ADD'; r_m_smg.use_clamp = True
    nt.links.new(r_m.outputs[0], r_m_smg.inputs[0])
    nt.links.new(smg_r.outputs[0], r_m_smg.inputs[1])

    mix_r0 = _new_node(nt, "ShaderNodeMix", (-220, -20))
    mix_r0.data_type = 'FLOAT'; mix_r0.inputs[3].default_value = 0.16
    nt.links.new(ew_m.outputs[0], mix_r0.inputs[0])
    nt.links.new(r_m_smg.outputs[0], mix_r0.inputs[2])

    ol_m0 = _new_node(nt, "ShaderNodeMath", (-560, -280)); ol_m0.operation = 'MULTIPLY'; ol_m0.use_clamp = True
    ol_m0.inputs[1].default_value = 0.48
    nt.links.new(masks.outputs["OilLeak_Mask"], ol_m0.inputs[0])
    ol_m = _new_node(nt, "ShaderNodeMath", (-400, -280)); ol_m.operation = 'MULTIPLY'; ol_m.use_clamp = True
    nt.links.new(ol_m0.outputs[0], ol_m.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], ol_m.inputs[1])
    mix_r = _new_node(nt, "ShaderNodeMix", (-60, -20))
    mix_r.data_type = 'FLOAT'; mix_r.inputs[3].default_value = 0.13
    nt.links.new(ol_m.outputs[0], mix_r.inputs[0])
    nt.links.new(mix_r0.outputs[0], mix_r.inputs[2])

    r_flr24c = _new_node(nt, "ShaderNodeMath", (50, -20)); r_flr24c.operation = 'MAXIMUM'; r_flr24c.use_clamp = True
    r_flr24c.name = "Metal_Roughness_Floor"
    r_flr24c.inputs[1].default_value = 0.13
    nt.links.new(mix_r.outputs[0], r_flr24c.inputs[0])

    mix_met = _new_node(nt, "ShaderNodeMix", (-160, -180))
    mix_met.data_type = 'FLOAT'; mix_met.inputs[2].default_value = 0.82; mix_met.inputs[3].default_value = 0.96
    nt.links.new(ew_m.outputs[0], mix_met.inputs[0])


    b_g = _new_node(nt, "ShaderNodeBump", (-220, -380))
    b_g.inputs["Strength"].default_value = 0.048
    b_g.inputs["Distance"].default_value = 0.00032
    nt.links.new(masks.outputs["Micro_Grain"], b_g.inputs["Height"])
    nt.links.new(masks.outputs["Bevel_Normal"], b_g.inputs["Normal"])

    sc_inv = _new_node(nt, "ShaderNodeMath", (-220, -520)); sc_inv.operation = 'SUBTRACT'; sc_inv.use_clamp = True
    sc_inv.inputs[0].default_value = 1.0
    nt.links.new(masks.outputs["Scratch_Mask"], sc_inv.inputs[1])
    b_sc = _new_node(nt, "ShaderNodeBump", (-20, -380))
    b_sc.inputs["Strength"].default_value = 0.075
    b_sc.inputs["Distance"].default_value = 0.00030
    nt.links.new(sc_inv.outputs[0], b_sc.inputs["Height"])
    nt.links.new(b_g.outputs["Normal"], b_sc.inputs["Normal"])


    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (180, 120))
    nt.links.new(mix_ew.outputs[2], bsdf.inputs["Base Color"])
    nt.links.new(mix_met.outputs[0], bsdf.inputs["Metallic"])
    nt.links.new(r_flr24c.outputs[0], bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.50
    if "Specular Tint" in bsdf.inputs:
        bsdf.inputs["Specular Tint"].default_value = rgba(0xDCE2EA)
    bsdf.inputs["Anisotropic"].default_value = 0.0
    nt.links.new(ol_m.outputs[0], bsdf.inputs["Coat Weight"])
    bsdf.inputs["Coat Roughness"].default_value = 0.14
    if "Coat Normal" in bsdf.inputs:
        nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Coat Normal"])
    nt.links.new(b_sc.outputs["Normal"], bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (500, 120))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(COLOR_ID_SPECS[MAT_NOZ_ACT_AB][1])
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (760, 120))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(COLOR_ID_SPECS[MAT_NOZ_ACT_AB][1])
    mat["f15_pbr_hex"] = int(0x2D3036)
    mat.diffuse_color = rgba(0x2D3036)
    mat.metallic = 0.85
    return mat


def refine_existing_special_face_materials(ng_global, ng_masks, ng_wrap):


    m_refl = build_ruby_reflector_material(ng_global, ng_masks, ng_wrap)
    o_tr = bpy.data.objects.get("Stand_Trailer_05_M3000_Steerable_Axles_And_TowBar")
    n_rf = 0
    if o_tr and o_tr.type == 'MESH':
        me = o_tr.data
        idx_thr = -1
        for i, m in enumerate(me.materials):
            if m and m.name == COLOR_ID_SPECS[MAT_THREAD][0]:
                idx_thr = i
                break
        if idx_thr >= 0:
            refl_faces = []
            for p in me.polygons:
                if p.material_index == idx_thr:
                    c = p.center
                    if (0.358 <= abs(c.x) <= 0.492) and (-0.992 <= c.z <= -0.928) and (c.y <= -4.270 or c.y >= -0.768):
                        refl_faces.append(p.index)
            n_rf = _assign_faces_to_new_slot(me, refl_faces, m_refl)


    m_ab, m_at = build_actuator_dual_band_materials(ng_global, ng_masks, ng_wrap)
    m_mos2 = build_actuator_mos2_struts_material(ng_global, ng_masks, ng_wrap)
    me_act = bpy.data.meshes.get("Static_Hydraulic_System_Detailed")
    n_act = 0
    n_struts = 0
    if me_act is not None:
        id24_slot = -1
        for i, m in enumerate(me_act.materials):
            if m is None:
                continue
            if m.name in (COLOR_ID_SPECS[MAT_PLATE_BASE][0], "Mat_ID18e_Actuator_DualBand"):
                me_act.materials[i] = m_ab
                n_act += 1
            elif m.name in (COLOR_ID_SPECS[MAT_3D_TEXT][0], "Mat_ID19f_Actuator_DualText"):
                me_act.materials[i] = m_at
                n_act += 1
            elif m.name == COLOR_ID_SPECS[MAT_NOZ_ACT_AB][0]:
                id24_slot = i

        if id24_slot >= 0:

            dirs8 = [(math.sin(math.radians(k * 45.0)), math.cos(math.radians(k * 45.0))) for k in range(8)]
            strut_faces = []
            for p in me_act.polygons:
                if p.material_index == id24_slot:
                    c = p.center
                    if c.y >= -0.935:

                        ang = math.atan2(c.x, c.z)
                        k = int(round(math.degrees(ang) / 45.0)) % 8
                        sa, ca = dirs8[k]
                        local_t = abs(c.x * ca - c.z * sa)
                        local_r = c.x * sa + c.z * ca
                        if local_t >= 0.0295 and local_r >= 0.50:
                            strut_faces.append(p.index)
            n_struts = _assign_faces_to_new_slot(me_act, strut_faces, m_mos2)

    print(f"[Component Materials] Reflectors={n_rf} faces | Actuator bands={n_act} | MoS2 struts={n_struts} faces")
    refine_hardware_accents(ng_global, ng_masks, ng_wrap)


def _slot_indices_by_names(me, names):
    name_set = set(names)
    return {i for i, m in enumerate(me.materials) if m is not None and m.name in name_set}


def _collect_whole_islands_subset(me, face_indices: list, target_frac: float, salt: int = 17) -> list:

    if not face_indices:
        return []
    n = len(face_indices)
    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    v_owner = {}
    polys = me.polygons
    for idx_local, fi in enumerate(face_indices):
        for vi in polys[fi].vertices:
            prev = v_owner.get(vi)
            if prev is None:
                v_owner[vi] = idx_local
            else:
                ra, rb = _find(prev), _find(idx_local)
                if ra != rb:
                    parent[rb] = ra

    groups = {}
    for idx_local, fi in enumerate(face_indices):
        r = _find(idx_local)
        groups.setdefault(r, []).append(fi)

    chosen = []
    thresh = int(round(target_frac * 1000.0))
    for g_faces in groups.values():

        if len(g_faces) > 3200 or len(g_faces) < 6:
            continue
        c0 = polys[g_faces[0]].center
        qx = int(round(c0.x * 37.0))
        qy = int(round(c0.y * 37.0))
        qz = int(round(c0.z * 37.0))
        hv = ((qx * 73856093) ^ (qy * 19349663) ^ (qz * 83492791) ^ (salt * 2654435761)) & 0x7FFFFFFF
        if (hv % 1000) < thresh:
            chosen.extend(g_faces)
    return chosen


def _build_accent_metallic_accent(
    mat_name: str, parent_id: int, ng_global, ng_masks, ng_wrap,
    *,
    col_a: int, col_b: int, crevice_col: int, wear_col: int,
    metallic: float = 0.98, r_min: float = 0.08, r_max: float = 0.16, wear_r: float = 0.05,
    ao_dist: float = 0.022, aniso: float = 0.05, coat_w: float = 0.0, coat_r: float = 0.12,
    knurl_scale: float = 0.0, lathe_scale: float = 0.0, tip_heat_y=None,
    inlet_runway_dust: bool = False, wiper_oil_ring=None,
    rough_floor: float = None, tangent_mode: str = 'NONE',
    f82_tint: int = 0xE2E8F0, rough_breakup_amp: float = 0.68,
):

    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1380, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1380, -340))
    ctrl.node_tree = ng_global

    masks = _new_node(nt, "ShaderNodeGroup", (-1060, 180))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = ao_dist
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00022
    masks.inputs["Edge_Bevel_Rad"].default_value = 0.0020
    masks.inputs["Edge_Gain"].default_value = 2.0
    masks.inputs["Grime_Scale"].default_value = 24.0
    masks.inputs["Macro_Scale"].default_value = 7.0
    masks.inputs["Micro_Scale"].default_value = 540.0

    mix_base = _new_node(nt, "ShaderNodeMix", (-720, 280)); mix_base.data_type = 'RGBA'
    mix_base.inputs[6].default_value = rgba(col_a)
    mix_base.inputs[7].default_value = rgba(col_b)
    nt.links.new(masks.outputs["Macro_Mottle"], mix_base.inputs[0])
    cur_col = mix_base.outputs[2]


    if tip_heat_y is not None:
        y_hot, y_cool, tint_hex, tint_amt = tip_heat_y
        sep_ty = _new_node(nt, "ShaderNodeSeparateXYZ", (-1040, 520))
        nt.links.new(tc.outputs["Object"], sep_ty.inputs["Vector"])
        ty_g = _new_node(nt, "ShaderNodeMapRange", (-840, 520)); ty_g.clamp = True
        ty_g.inputs[1].default_value = y_hot
        ty_g.inputs[2].default_value = y_cool
        ty_g.inputs[3].default_value = tint_amt
        ty_g.inputs[4].default_value = 0.0
        nt.links.new(sep_ty.outputs["Y"], ty_g.inputs[0])
        ty_ht = _new_node(nt, "ShaderNodeMath", (-640, 520)); ty_ht.operation = 'MULTIPLY'; ty_ht.use_clamp = True
        nt.links.new(ty_g.outputs[0], ty_ht.inputs[0])
        nt.links.new(ctrl.outputs["HeatTint"], ty_ht.inputs[1])
        mix_ty = _new_node(nt, "ShaderNodeMix", (-500, 320)); mix_ty.data_type = 'RGBA'
        mix_ty.inputs[7].default_value = rgba(tint_hex)
        nt.links.new(ty_ht.outputs[0], mix_ty.inputs[0])
        nt.links.new(cur_col, mix_ty.inputs[6])
        cur_col = mix_ty.outputs[2]


    cr_w = _new_node(nt, "ShaderNodeMath", (-720, -60)); cr_w.operation = 'MULTIPLY'; cr_w.use_clamp = True
    cr_w.inputs[1].default_value = 0.54
    nt.links.new(masks.outputs["Crevice_Mask"], cr_w.inputs[0])
    cr_eff = _new_node(nt, "ShaderNodeMath", (-540, -60)); cr_eff.operation = 'MULTIPLY'; cr_eff.use_clamp = True
    nt.links.new(cr_w.outputs[0], cr_eff.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], cr_eff.inputs[1])
    mix_cr = _new_node(nt, "ShaderNodeMix", (-320, 260)); mix_cr.data_type = 'RGBA'
    mix_cr.inputs[7].default_value = rgba(crevice_col)
    nt.links.new(cr_eff.outputs[0], mix_cr.inputs[0])
    nt.links.new(cur_col, mix_cr.inputs[6])
    cur_col = mix_cr.outputs[2]


    ew_g = _new_node(nt, "ShaderNodeMath", (-720, -220)); ew_g.operation = 'MULTIPLY'; ew_g.use_clamp = True
    ew_g.inputs[1].default_value = 0.50
    nt.links.new(masks.outputs["EdgeWear_Mask"], ew_g.inputs[0])
    ew_eff = _new_node(nt, "ShaderNodeMath", (-540, -220)); ew_eff.operation = 'MULTIPLY'; ew_eff.use_clamp = True
    nt.links.new(ew_g.outputs[0], ew_eff.inputs[0])
    nt.links.new(ctrl.outputs["EdgeWear"], ew_eff.inputs[1])
    mix_ew = _new_node(nt, "ShaderNodeMix", (-120, 260)); mix_ew.data_type = 'RGBA'
    mix_ew.inputs[7].default_value = rgba(wear_col)
    nt.links.new(ew_eff.outputs[0], mix_ew.inputs[0])
    nt.links.new(cur_col, mix_ew.inputs[6])
    cur_col = mix_ew.outputs[2]


    r_src_acc = _new_node(nt, "ShaderNodeMix", (-860, 40)); r_src_acc.data_type = 'FLOAT'
    r_src_acc.inputs[0].default_value = min(0.85, max(0.0, rough_breakup_amp))
    nt.links.new(masks.outputs["Macro_Mottle"], r_src_acc.inputs[2])
    nt.links.new(masks.outputs["Metal_Rough_Breakup"], r_src_acc.inputs[3])

    r_m = _new_node(nt, "ShaderNodeMapRange", (-680, 40)); r_m.clamp = True
    r_m.inputs[1].default_value = 0.12; r_m.inputs[2].default_value = 0.88
    r_m.inputs[3].default_value = r_min; r_m.inputs[4].default_value = r_max
    nt.links.new(r_src_acc.outputs[0], r_m.inputs[0])
    smg_r = _new_node(nt, "ShaderNodeMapRange", (-680, -340)); smg_r.clamp = True
    smg_r.inputs[1].default_value = 0.0; smg_r.inputs[2].default_value = 1.0
    smg_r.inputs[3].default_value = -0.015; smg_r.inputs[4].default_value = +0.055
    nt.links.new(masks.outputs["Photo_Smudge"], smg_r.inputs[0])
    smg_w = _new_node(nt, "ShaderNodeMath", (-480, -340)); smg_w.operation = 'MULTIPLY'
    nt.links.new(smg_r.outputs[0], smg_w.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], smg_w.inputs[1])
    r_add = _new_node(nt, "ShaderNodeMath", (-300, 40)); r_add.operation = 'ADD'; r_add.use_clamp = True
    nt.links.new(r_m.outputs[0], r_add.inputs[0])
    nt.links.new(smg_w.outputs[0], r_add.inputs[1])
    mix_r = _new_node(nt, "ShaderNodeMix", (-100, 40)); mix_r.data_type = 'FLOAT'
    mix_r.inputs[3].default_value = wear_r
    nt.links.new(ew_eff.outputs[0], mix_r.inputs[0])
    nt.links.new(r_add.outputs[0], mix_r.inputs[2])
    cur_rough = mix_r.outputs[0]
    cur_coat_w_sock = None
    cur_coat_r_sock = None


    if inlet_runway_dust:
        geo_id = _new_node(nt, "ShaderNodeNewGeometry", (-760, 620))
        sep_idn = _new_node(nt, "ShaderNodeSeparateXYZ", (-560, 680))
        nt.links.new(geo_id.outputs["Normal"], sep_idn.inputs["Vector"])
        sep_idp = _new_node(nt, "ShaderNodeSeparateXYZ", (-560, 560))
        nt.links.new(tc.outputs["Object"], sep_idp.inputs["Vector"])
        id_nz = _new_node(nt, "ShaderNodeMapRange", (-360, 680)); id_nz.clamp = True
        id_nz.inputs[1].default_value = 0.18; id_nz.inputs[2].default_value = 0.85
        nt.links.new(sep_idn.outputs["Z"], id_nz.inputs[0])
        id_py = _new_node(nt, "ShaderNodeMapRange", (-360, 560)); id_py.clamp = True
        id_py.inputs[1].default_value = -4.36; id_py.inputs[2].default_value = -4.27
        nt.links.new(sep_idp.outputs["Y"], id_py.inputs[0])
        id_m0 = _new_node(nt, "ShaderNodeMath", (-180, 620)); id_m0.operation = 'MULTIPLY'; id_m0.use_clamp = True
        nt.links.new(id_nz.outputs[0], id_m0.inputs[0])
        nt.links.new(id_py.outputs[0], id_m0.inputs[1])
        id_m1 = _new_node(nt, "ShaderNodeMath", (-20, 620)); id_m1.operation = 'MULTIPLY'; id_m1.use_clamp = True
        nt.links.new(id_m0.outputs[0], id_m1.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], id_m1.inputs[1])
        id_cf = _new_node(nt, "ShaderNodeMath", (100, 620)); id_cf.operation = 'MULTIPLY'; id_cf.use_clamp = True
        id_cf.inputs[1].default_value = 0.42
        nt.links.new(id_m1.outputs[0], id_cf.inputs[0])
        mix_id_c = _new_node(nt, "ShaderNodeMix", (80, 280)); mix_id_c.data_type = 'RGBA'
        mix_id_c.inputs[7].default_value = rgba(0x524E48)
        nt.links.new(id_cf.outputs[0], mix_id_c.inputs[0])
        nt.links.new(cur_col, mix_id_c.inputs[6])
        cur_col = mix_id_c.outputs[2]
        mix_id_r = _new_node(nt, "ShaderNodeMix", (80, 60)); mix_id_r.data_type = 'FLOAT'
        mix_id_r.inputs[3].default_value = 0.52
        nt.links.new(id_cf.outputs[0], mix_id_r.inputs[0])
        nt.links.new(cur_rough, mix_id_r.inputs[2])
        cur_rough = mix_id_r.outputs[0]


    if wiper_oil_ring is not None:
        wy_c, wy_hw = wiper_oil_ring
        sep_wy = _new_node(nt, "ShaderNodeSeparateXYZ", (-760, 780))
        nt.links.new(tc.outputs["Object"], sep_wy.inputs["Vector"])

        n_wy = _new_node(nt, "ShaderNodeTexNoise", (-760, 920))
        n_wy.inputs["Scale"].default_value = 34.0
        n_wy.inputs["Detail"].default_value = 5.0
        n_wy.inputs["Roughness"].default_value = 0.62
        nt.links.new(tc.outputs["Object"], n_wy.inputs["Vector"])
        wy_jit = _new_node(nt, "ShaderNodeMapRange", (-580, 920)); wy_jit.clamp = True
        wy_jit.inputs[1].default_value = 0.25; wy_jit.inputs[2].default_value = 0.75
        wy_jit.inputs[3].default_value = -0.0022; wy_jit.inputs[4].default_value = +0.0022
        nt.links.new(n_wy.outputs["Factor"], wy_jit.inputs[0])

        wy_y_eff = _new_node(nt, "ShaderNodeMath", (-580, 780)); wy_y_eff.operation = 'ADD'
        nt.links.new(sep_wy.outputs["Y"], wy_y_eff.inputs[0])
        nt.links.new(wy_jit.outputs[0], wy_y_eff.inputs[1])
        wy_d0 = _new_node(nt, "ShaderNodeMath", (-420, 780)); wy_d0.operation = 'SUBTRACT'
        wy_d0.inputs[1].default_value = wy_c
        nt.links.new(wy_y_eff.outputs[0], wy_d0.inputs[0])
        wy_ad = _new_node(nt, "ShaderNodeMath", (-280, 780)); wy_ad.operation = 'ABSOLUTE'
        nt.links.new(wy_d0.outputs[0], wy_ad.inputs[0])


        wy_ring = _new_node(nt, "ShaderNodeMapRange", (-120, 780)); wy_ring.clamp = True
        wy_ring.inputs[1].default_value = wy_hw * 0.22; wy_ring.inputs[2].default_value = wy_hw
        wy_ring.inputs[3].default_value = 1.0; wy_ring.inputs[4].default_value = 0.0
        nt.links.new(wy_ad.outputs[0], wy_ring.inputs[0])
        wy_eff = _new_node(nt, "ShaderNodeMath", (40, 780)); wy_eff.operation = 'MULTIPLY'; wy_eff.use_clamp = True
        nt.links.new(wy_ring.outputs[0], wy_eff.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], wy_eff.inputs[1])

        wy_dust0 = _new_node(nt, "ShaderNodeMapRange", (-120, 920)); wy_dust0.clamp = True
        wy_dust0.inputs[1].default_value = 0.0012; wy_dust0.inputs[2].default_value = 0.0050
        wy_dust0.inputs[3].default_value = 0.86; wy_dust0.inputs[4].default_value = 0.0
        nt.links.new(wy_ad.outputs[0], wy_dust0.inputs[0])
        wy_dust = _new_node(nt, "ShaderNodeMath", (40, 920)); wy_dust.operation = 'MULTIPLY'; wy_dust.use_clamp = True
        nt.links.new(wy_dust0.outputs[0], wy_dust.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], wy_dust.inputs[1])

        wy_cf = _new_node(nt, "ShaderNodeMath", (160, 780)); wy_cf.operation = 'MULTIPLY'; wy_cf.use_clamp = True
        wy_cf.inputs[1].default_value = 0.50
        nt.links.new(wy_eff.outputs[0], wy_cf.inputs[0])
        mix_wy_c = _new_node(nt, "ShaderNodeMix", (120, 260)); mix_wy_c.data_type = 'RGBA'
        mix_wy_c.inputs[7].default_value = rgba(0x3A220E)
        nt.links.new(wy_cf.outputs[0], mix_wy_c.inputs[0])
        nt.links.new(cur_col, mix_wy_c.inputs[6])

        mix_wd_c = _new_node(nt, "ShaderNodeMix", (200, 260)); mix_wd_c.data_type = 'RGBA'
        mix_wd_c.inputs[7].default_value = rgba(0x1E1812)
        nt.links.new(wy_dust.outputs[0], mix_wd_c.inputs[0])
        nt.links.new(mix_wy_c.outputs[2], mix_wd_c.inputs[6])
        cur_col = mix_wd_c.outputs[2]

        mix_wy_r = _new_node(nt, "ShaderNodeMix", (120, 40)); mix_wy_r.data_type = 'FLOAT'
        mix_wy_r.inputs[3].default_value = 0.055
        nt.links.new(wy_eff.outputs[0], mix_wy_r.inputs[0])
        nt.links.new(cur_rough, mix_wy_r.inputs[2])
        mix_wd_r = _new_node(nt, "ShaderNodeMix", (200, 40)); mix_wd_r.data_type = 'FLOAT'
        mix_wd_r.inputs[3].default_value = 0.58
        nt.links.new(wy_dust.outputs[0], mix_wd_r.inputs[0])
        nt.links.new(mix_wy_r.outputs[0], mix_wd_r.inputs[2])
        cur_rough = mix_wd_r.outputs[0]

        mix_wy_cw = _new_node(nt, "ShaderNodeMix", (120, -80)); mix_wy_cw.data_type = 'FLOAT'
        mix_wy_cw.inputs[2].default_value = coat_w
        mix_wy_cw.inputs[3].default_value = 0.92
        nt.links.new(wy_eff.outputs[0], mix_wy_cw.inputs[0])
        cur_coat_w_sock = mix_wy_cw.outputs[0]
        mix_wy_cr = _new_node(nt, "ShaderNodeMix", (120, -180)); mix_wy_cr.data_type = 'FLOAT'
        mix_wy_cr.inputs[2].default_value = coat_r
        mix_wy_cr.inputs[3].default_value = 0.06
        nt.links.new(wy_eff.outputs[0], mix_wy_cr.inputs[0])
        cur_coat_r_sock = mix_wy_cr.outputs[0]


    cur_norm = masks.outputs["Bevel_Normal"]
    h_acc = _new_node(nt, "ShaderNodeMath", (-360, -460)); h_acc.operation = 'MULTIPLY'
    h_acc.inputs[1].default_value = 0.018
    nt.links.new(masks.outputs["Micro_Grain"], h_acc.inputs[0])
    cur_h = h_acc.outputs[0]

    if lathe_scale > 0.0:
        w_l = _new_node(nt, "ShaderNodeTexWave", (-620, -580))
        w_l.wave_type = 'BANDS'; w_l.bands_direction = 'Y'
        w_l.inputs["Scale"].default_value = lathe_scale
        w_l.inputs["Distortion"].default_value = 0.08
        nt.links.new(tc.outputs["Object"], w_l.inputs["Vector"])
        h_l = _new_node(nt, "ShaderNodeMath", (-240, -540)); h_l.operation = 'MULTIPLY_ADD'
        h_l.inputs[1].default_value = 0.020
        nt.links.new(w_l.outputs["Fac"], h_l.inputs[0])
        nt.links.new(cur_h, h_l.inputs[2])
        cur_h = h_l.outputs[0]
        r_lw = _new_node(nt, "ShaderNodeMapRange", (-240, -400)); r_lw.clamp = True
        r_lw.inputs[1].default_value = 0.0; r_lw.inputs[2].default_value = 1.0
        r_lw.inputs[3].default_value = -0.022; r_lw.inputs[4].default_value = +0.022
        nt.links.new(w_l.outputs["Fac"], r_lw.inputs[0])
        r_add_l = _new_node(nt, "ShaderNodeMath", (10, -20)); r_add_l.operation = 'ADD'; r_add_l.use_clamp = True
        nt.links.new(cur_rough, r_add_l.inputs[0])
        nt.links.new(r_lw.outputs[0], r_add_l.inputs[1])
        cur_rough = r_add_l.outputs[0]

    b_mg = _new_node(nt, "ShaderNodeBump", (-80, -480))
    b_mg.inputs["Strength"].default_value = 1.0
    b_mg.inputs["Distance"].default_value = 0.00024
    nt.links.new(cur_h, b_mg.inputs["Height"])
    nt.links.new(cur_norm, b_mg.inputs["Normal"])
    cur_norm = b_mg.outputs["Normal"]

    if knurl_scale > 0.0:
        w_k1 = _new_node(nt, "ShaderNodeTexWave", (-680, -720))
        w_k1.wave_type = 'BANDS'; w_k1.bands_direction = 'DIAGONAL'
        w_k1.inputs["Scale"].default_value = knurl_scale
        nt.links.new(tc.outputs["Object"], w_k1.inputs["Vector"])
        mp_k2 = _new_node(nt, "ShaderNodeMapping", (-880, -860))
        mp_k2.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(90.0))
        nt.links.new(tc.outputs["Object"], mp_k2.inputs["Vector"])
        w_k2 = _new_node(nt, "ShaderNodeTexWave", (-680, -860))
        w_k2.wave_type = 'BANDS'; w_k2.bands_direction = 'DIAGONAL'
        w_k2.inputs["Scale"].default_value = knurl_scale
        nt.links.new(mp_k2.outputs["Vector"], w_k2.inputs["Vector"])
        k_comb = _new_node(nt, "ShaderNodeMath", (-440, -780)); k_comb.operation = 'MINIMUM'
        nt.links.new(w_k1.outputs["Fac"], k_comb.inputs[0])
        nt.links.new(w_k2.outputs["Fac"], k_comb.inputs[1])
        b_k = _new_node(nt, "ShaderNodeBump", (40, -560))
        b_k.inputs["Strength"].default_value = 0.16
        b_k.inputs["Distance"].default_value = 0.00038
        nt.links.new(k_comb.outputs[0], b_k.inputs["Height"])
        nt.links.new(cur_norm, b_k.inputs["Normal"])
        cur_norm = b_k.outputs["Normal"]

    eff_r_floor = rough_floor if rough_floor is not None else max(0.05, min(r_min * 0.80, wear_r * 0.90))
    r_flr = _new_node(nt, "ShaderNodeMath", (120, -20)); r_flr.operation = 'MAXIMUM'; r_flr.use_clamp = True
    r_flr.name = "Metal_Roughness_Floor"
    r_flr.inputs[1].default_value = eff_r_floor
    nt.links.new(cur_rough, r_flr.inputs[0])
    cur_rough = r_flr.outputs[0]

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (220, 120))
    nt.links.new(cur_col, bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = metallic
    nt.links.new(cur_rough, bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.50
    if "Specular Tint" in bsdf.inputs:
        bsdf.inputs["Specular Tint"].default_value = rgba(f82_tint)
    eff_aniso = 0.0 if tangent_mode == 'NONE' else aniso
    bsdf.inputs["Anisotropic"].default_value = eff_aniso
    if eff_aniso > 1e-4 and tangent_mode != 'NONE':
        tang_acc = _new_node(nt, "ShaderNodeTangent", (0, -240))
        if tangent_mode == 'RADIAL_X':
            tang_acc.direction_type = 'RADIAL'; tang_acc.axis = 'X'
        elif tangent_mode == 'RADIAL_Z':
            tang_acc.direction_type = 'RADIAL'; tang_acc.axis = 'Z'
        elif tangent_mode == 'UV':
            tang_acc.direction_type = 'UV_MAP'
        else:
            tang_acc.direction_type = 'RADIAL'; tang_acc.axis = 'Y'
        nt.links.new(tang_acc.outputs["Tangent"], bsdf.inputs["Tangent"])
    if cur_coat_w_sock is not None:
        nt.links.new(cur_coat_w_sock, bsdf.inputs["Coat Weight"])
    else:
        bsdf.inputs["Coat Weight"].default_value = coat_w
    if cur_coat_r_sock is not None:
        nt.links.new(cur_coat_r_sock, bsdf.inputs["Coat Roughness"])
    else:
        bsdf.inputs["Coat Roughness"].default_value = coat_r
    if (coat_w > 1e-4 or cur_coat_w_sock is not None) and "Coat Normal" in bsdf.inputs:
        nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Coat Normal"])
    nt.links.new(cur_norm, bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (540, 120))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(COLOR_ID_SPECS[parent_id][1])
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (800, 120))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(COLOR_ID_SPECS[parent_id][1])
    mat["f15_pbr_hex"] = int(col_a)
    mat.diffuse_color = rgba(col_a)
    mat.metallic = metallic
    return mat


def _build_accent_elastomer_accent(
    mat_name: str, parent_id: int, ng_global, ng_masks, ng_wrap,
    *,
    col_a: int, col_b: int, crevice_col: int,
    r_min: float = 0.42, r_max: float = 0.58, coat_w: float = 0.0, coat_r: float = 0.16,
    knurl_scale: float = 0.0,
):

    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1280, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1280, -320))
    ctrl.node_tree = ng_global

    masks = _new_node(nt, "ShaderNodeGroup", (-960, 160))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.024
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00025
    masks.inputs["Micro_Scale"].default_value = 720.0

    mix_b = _new_node(nt, "ShaderNodeMix", (-620, 240)); mix_b.data_type = 'RGBA'
    mix_b.inputs[6].default_value = rgba(col_a)
    mix_b.inputs[7].default_value = rgba(col_b)
    nt.links.new(masks.outputs["Macro_Mottle"], mix_b.inputs[0])

    cr_w0 = _new_node(nt, "ShaderNodeMath", (-720, -60)); cr_w0.operation = 'MULTIPLY'; cr_w0.use_clamp = True
    cr_w0.inputs[1].default_value = 0.52
    nt.links.new(masks.outputs["Crevice_Mask"], cr_w0.inputs[0])
    cr_w = _new_node(nt, "ShaderNodeMath", (-540, -60)); cr_w.operation = 'MULTIPLY'; cr_w.use_clamp = True
    nt.links.new(cr_w0.outputs[0], cr_w.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], cr_w.inputs[1])
    mix_c = _new_node(nt, "ShaderNodeMix", (-340, 240)); mix_c.data_type = 'RGBA'
    mix_c.inputs[7].default_value = rgba(crevice_col)
    nt.links.new(cr_w.outputs[0], mix_c.inputs[0])
    nt.links.new(mix_b.outputs[2], mix_c.inputs[6])

    r_m = _new_node(nt, "ShaderNodeMapRange", (-460, 40)); r_m.clamp = True
    r_m.inputs[1].default_value = 0.18; r_m.inputs[2].default_value = 0.84
    r_m.inputs[3].default_value = r_min; r_m.inputs[4].default_value = r_max
    nt.links.new(masks.outputs["Macro_Mottle"], r_m.inputs[0])

    cur_norm = masks.outputs["Bevel_Normal"]
    b_g = _new_node(nt, "ShaderNodeBump", (-260, -280))
    b_g.inputs["Strength"].default_value = 0.055
    b_g.inputs["Distance"].default_value = 0.00025
    nt.links.new(masks.outputs["Micro_Grain"], b_g.inputs["Height"])
    nt.links.new(cur_norm, b_g.inputs["Normal"])
    cur_norm = b_g.outputs["Normal"]

    if knurl_scale > 0.0:
        w1 = _new_node(nt, "ShaderNodeTexWave", (-680, -480))
        w1.wave_type = 'BANDS'; w1.bands_direction = 'DIAGONAL'
        w1.inputs["Scale"].default_value = knurl_scale
        nt.links.new(tc.outputs["Object"], w1.inputs["Vector"])
        mp2 = _new_node(nt, "ShaderNodeMapping", (-880, -620))
        mp2.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(90.0))
        nt.links.new(tc.outputs["Object"], mp2.inputs["Vector"])
        w2 = _new_node(nt, "ShaderNodeTexWave", (-680, -620))
        w2.wave_type = 'BANDS'; w2.bands_direction = 'DIAGONAL'
        w2.inputs["Scale"].default_value = knurl_scale
        nt.links.new(mp2.outputs["Vector"], w2.inputs["Vector"])
        kc = _new_node(nt, "ShaderNodeMath", (-440, -540)); kc.operation = 'MINIMUM'
        nt.links.new(w1.outputs["Fac"], kc.inputs[0]); nt.links.new(w2.outputs["Fac"], kc.inputs[1])
        bk = _new_node(nt, "ShaderNodeBump", (-40, -340))
        bk.inputs["Strength"].default_value = 0.24
        bk.inputs["Distance"].default_value = 0.00060
        nt.links.new(kc.outputs[0], bk.inputs["Height"])
        nt.links.new(cur_norm, bk.inputs["Normal"])
        cur_norm = bk.outputs["Normal"]

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (220, 120))
    nt.links.new(mix_c.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    nt.links.new(r_m.outputs[0], bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.44
    bsdf.inputs["Coat Weight"].default_value = coat_w
    bsdf.inputs["Coat Roughness"].default_value = coat_r
    if coat_w > 1e-4 and "Coat Normal" in bsdf.inputs:
        nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Coat Normal"])
    nt.links.new(cur_norm, bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (540, 120))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(COLOR_ID_SPECS[parent_id][1])
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (800, 120))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(COLOR_ID_SPECS[parent_id][1])
    mat["f15_pbr_hex"] = int(col_a)
    mat.diffuse_color = rgba(col_a)
    return mat


def _build_accent_oil_sight_glass(ng_global, ng_masks, ng_wrap):

    mat = bpy.data.materials.get("Mat_Accent_OilSight_AmberGlass")
    if mat is None:
        mat = bpy.data.materials.new("Mat_Accent_OilSight_AmberGlass")
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1320, 0))
    masks = _new_node(nt, "ShaderNodeGroup", (-1020, -240))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.016
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00018


    vrot = _new_node(nt, "ShaderNodeVectorRotate", (-1020, 180))
    vrot.rotation_type = 'Y_AXIS'
    vrot.inputs["Angle"].default_value = -math.radians(239.4)
    nt.links.new(tc.outputs["Object"], vrot.inputs["Vector"])
    sep = _new_node(nt, "ShaderNodeSeparateXYZ", (-800, 180))
    nt.links.new(vrot.outputs["Vector"], sep.inputs["Vector"])


    air_fac = _new_node(nt, "ShaderNodeMapRange", (-580, 180)); air_fac.clamp = True
    air_fac.inputs[1].default_value = 0.0108
    air_fac.inputs[2].default_value = 0.0142
    air_fac.inputs[3].default_value = 0.0
    air_fac.inputs[4].default_value = 1.0
    nt.links.new(sep.outputs["X"], air_fac.inputs[0])


    oil_grad = _new_node(nt, "ShaderNodeMapRange", (-580, 360)); oil_grad.clamp = True
    oil_grad.inputs[1].default_value = -0.052
    oil_grad.inputs[2].default_value = +0.012
    oil_grad.inputs[3].default_value = 0.0
    oil_grad.inputs[4].default_value = 1.0
    nt.links.new(sep.outputs["X"], oil_grad.inputs[0])


    oil_col = _new_node(nt, "ShaderNodeMix", (-340, 340)); oil_col.data_type = 'RGBA'
    oil_col.inputs[6].default_value = rgba(0xB85608)
    oil_col.inputs[7].default_value = rgba(0xFFB82E)
    nt.links.new(oil_grad.outputs[0], oil_col.inputs[0])


    men_lo = _new_node(nt, "ShaderNodeMapRange", (-580, 480)); men_lo.clamp = True
    men_lo.inputs[1].default_value = 0.0078; men_lo.inputs[2].default_value = 0.0096
    men_lo.inputs[3].default_value = 0.0; men_lo.inputs[4].default_value = 1.0
    nt.links.new(sep.outputs["X"], men_lo.inputs[0])
    men_hi = _new_node(nt, "ShaderNodeMapRange", (-580, 600)); men_hi.clamp = True
    men_hi.inputs[1].default_value = 0.0097; men_hi.inputs[2].default_value = 0.0112
    men_hi.inputs[3].default_value = 1.0; men_hi.inputs[4].default_value = 0.0
    nt.links.new(sep.outputs["X"], men_hi.inputs[0])
    men_ring = _new_node(nt, "ShaderNodeMath", (-360, 520)); men_ring.operation = 'MULTIPLY'; men_ring.use_clamp = True
    nt.links.new(men_lo.outputs[0], men_ring.inputs[0])
    nt.links.new(men_hi.outputs[0], men_ring.inputs[1])

    oil_with_men = _new_node(nt, "ShaderNodeMix", (-220, 340)); oil_with_men.data_type = 'RGBA'
    oil_with_men.inputs[7].default_value = rgba(0xFFE88C)
    nt.links.new(men_ring.outputs[0], oil_with_men.inputs[0])
    nt.links.new(oil_col.outputs[2], oil_with_men.inputs[6])

    mix_col = _new_node(nt, "ShaderNodeMix", (-60, 240)); mix_col.data_type = 'RGBA'
    mix_col.inputs[7].default_value = rgba(0xF5FBF9)
    nt.links.new(air_fac.outputs[0], mix_col.inputs[0])
    nt.links.new(oil_with_men.outputs[2], mix_col.inputs[6])


    tr_mix = _new_node(nt, "ShaderNodeMix", (-100, -40)); tr_mix.data_type = 'FLOAT'
    tr_mix.inputs[2].default_value = 0.92
    tr_mix.inputs[3].default_value = 0.985
    nt.links.new(air_fac.outputs[0], tr_mix.inputs[0])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (220, 140))
    nt.links.new(mix_col.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.014
    bsdf.inputs["IOR"].default_value = 1.52
    bsdf.inputs["Specular IOR Level"].default_value = 0.50
    nt.links.new(tr_mix.outputs[0], bsdf.inputs["Transmission Weight"])
    bsdf.inputs["Coat Weight"].default_value = 0.0
    bsdf.inputs["Emission Strength"].default_value = 0.0
    nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Normal"])


    ctrl_sg = _new_node(nt, "ShaderNodeGroup", (-100, -300))
    ctrl_sg.node_tree = ng_global
    vol_dens = _new_node(nt, "ShaderNodeMix", (100, -180)); vol_dens.data_type = 'FLOAT'
    vol_dens.inputs[2].default_value = 22.0
    vol_dens.inputs[3].default_value = 0.0
    nt.links.new(air_fac.outputs[0], vol_dens.inputs[0])
    vol_gate = _new_node(nt, "ShaderNodeMath", (280, -180)); vol_gate.operation = 'MULTIPLY'
    nt.links.new(vol_dens.outputs[0], vol_gate.inputs[0])
    nt.links.new(ctrl_sg.outputs["Show_PBR"], vol_gate.inputs[1])
    vol_abs = _new_node(nt, "ShaderNodeVolumeAbsorption", (480, -180))
    vol_abs.inputs["Color"].default_value = rgba(0xE8941A)
    nt.links.new(vol_gate.outputs[0], vol_abs.inputs["Density"])

    wrap = _new_node(nt, "ShaderNodeGroup", (520, 140))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(COLOR_ID_SPECS[MAT_PLATE_BASE][1])
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (780, 140))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])
    nt.links.new(vol_abs.outputs["Volume"], out.inputs["Volume"])

    mat["f15_id_hex"] = int(COLOR_ID_SPECS[MAT_PLATE_BASE][1])
    mat["f15_pbr_hex"] = int(0xD8760E)
    mat.diffuse_color = rgba(0xD8760E)
    return mat


def refine_hardware_accents(ng_global, ng_masks, ng_wrap):


    m_pitot = _build_accent_metallic_accent(
        "Mat_Accent_Pitot_MirrorChrome", MAT_SPINNER_IGV, ng_global, ng_masks, ng_wrap,
        col_a=0xCCD2DA, col_b=0xA8B0BA, crevice_col=0x1C1F24, wear_col=0xDCE2E8,
        metallic=0.99, r_min=0.09, r_max=0.22, wear_r=0.07,
        ao_dist=0.016, aniso=0.45, lathe_scale=240.0,
        tip_heat_y=(-4.695, -4.612, 0xD8C28A, 0.30),
        rough_floor=0.07, tangent_mode='RADIAL_Y',
        f82_tint=0xE6ECF2, rough_breakup_amp=0.68,
    )
    m_probes = _build_accent_metallic_accent(
        "Mat_Accent_InletProbes_SatinChrome", MAT_SPINNER_IGV, ng_global, ng_masks, ng_wrap,
        col_a=0xC4CAD2, col_b=0x9AA2AC, crevice_col=0x1E2126, wear_col=0xD6DCE2,
        metallic=0.99, r_min=0.11, r_max=0.26, wear_r=0.08, ao_dist=0.018, aniso=0.0,
        rough_floor=0.08, tangent_mode='NONE',
        f82_tint=0xE2E8F0, rough_breakup_amp=0.70,
    )
    m_sphw = _build_accent_metallic_accent(
        "Mat_Accent_Spinner_BrightHardware", MAT_FASTENER, ng_global, ng_masks, ng_wrap,
        col_a=0xC8CED6, col_b=0xB2A88A, crevice_col=0x22201C, wear_col=0xD8DEE4,
        metallic=0.99, r_min=0.09, r_max=0.22, wear_r=0.07, ao_dist=0.010, aniso=0.0,
        rough_floor=0.07, tangent_mode='NONE',
        f82_tint=0xE6ECF2, rough_breakup_amp=0.68,
    )
    m_sight_glass = _build_accent_oil_sight_glass(ng_global, ng_masks, ng_wrap)
    m_sight_float = _build_accent_elastomer_accent(
        "Mat_Accent_OilSight_FloatRed", MAT_PLATE_BASE, ng_global, ng_masks, ng_wrap,
        col_a=0xCC3016, col_b=0xA4240E, crevice_col=0x481006,
        r_min=0.16, r_max=0.26, coat_w=0.65, coat_r=0.10,
    )
    m_an_blue = _build_accent_metallic_accent(
        "Mat_Accent_AN_AnodizedBlue", MAT_BRASS_FITTING, ng_global, ng_masks, ng_wrap,
        col_a=0x2A6EC8, col_b=0x1C4E98, crevice_col=0x0E2242, wear_col=0xBACCE2,
        metallic=0.94, r_min=0.12, r_max=0.28, wear_r=0.09, ao_dist=0.018, aniso=0.0, coat_w=0.32, coat_r=0.12,
        rough_floor=0.09, tangent_mode='NONE',
        f82_tint=0xDCE8FC, rough_breakup_amp=0.72,
    )
    m_an_red = _build_accent_metallic_accent(
        "Mat_Accent_AN_AnodizedRed", MAT_BRASS_FITTING, ng_global, ng_masks, ng_wrap,
        col_a=0xBE2E28, col_b=0x881E1A, crevice_col=0x300A08, wear_col=0xDABAB8,
        metallic=0.94, r_min=0.12, r_max=0.28, wear_r=0.09, ao_dist=0.018, aniso=0.0, coat_w=0.32, coat_r=0.12,
        rough_floor=0.09, tangent_mode='NONE',
        f82_tint=0xF8D8D8, rough_breakup_amp=0.72,
    )
    m_chrome_rod = _build_accent_metallic_accent(
        "Mat_Accent_Actuator_HardChromeRod", MAT_NOZ_ACT_AB, ng_global, ng_masks, ng_wrap,
        col_a=0xD0D6DE, col_b=0xAEB6C0, crevice_col=0x22262B, wear_col=0xDCE2E8,
        metallic=0.99, r_min=0.06, r_max=0.15, wear_r=0.05,
        ao_dist=0.016, aniso=0.0, coat_w=0.22, coat_r=0.05, lathe_scale=260.0,
        wiper_oil_ring=(-0.678, 0.016), rough_floor=0.05, tangent_mode='NONE',
        f82_tint=0xE8EEF4, rough_breakup_amp=0.66,
    )
    m_bronze_w = _build_accent_metallic_accent(
        "Mat_Accent_Nozzle_PhosphorBronze", MAT_RIGID_TUBE, ng_global, ng_masks, ng_wrap,
        col_a=0xD0A254, col_b=0x8E642A, crevice_col=0x241A0C, wear_col=0xDCC286,
        metallic=0.98, r_min=0.12, r_max=0.30, wear_r=0.09, ao_dist=0.014, aniso=0.0,
        rough_floor=0.09, tangent_mode='NONE',
        f82_tint=0xFCE8BA, rough_breakup_amp=0.76,
    )
    m_port_ring = _build_accent_metallic_accent(
        "Mat_Accent_ShroudPort_ChromeRing", MAT_MACHINED_ALU, ng_global, ng_masks, ng_wrap,
        col_a=0xCAD0D8, col_b=0xA6AEB8, crevice_col=0x222428, wear_col=0xDCE2E8,
        metallic=0.99, r_min=0.09, r_max=0.20, wear_r=0.06, ao_dist=0.012, aniso=0.0,
        rough_floor=0.06, tangent_mode='NONE',
        f82_tint=0xE6ECF2, rough_breakup_amp=0.68,
    )
    m_port_viton = _build_accent_elastomer_accent(
        "Mat_Accent_ShroudPort_VitonSeal", MAT_MACHINED_ALU, ng_global, ng_masks, ng_wrap,
        col_a=0xB8321E, col_b=0x882414, crevice_col=0x381008,
        r_min=0.32, r_max=0.44, coat_w=0.20, coat_r=0.24,
    )
    m_gse_redpin = _build_accent_metallic_accent(
        "Mat_Accent_GSE_AnodizedRedPin", MAT_STAND_STEEL, ng_global, ng_masks, ng_wrap,
        col_a=0xC22E24, col_b=0x8C1E16, crevice_col=0x300A08, wear_col=0xD4DAE0,
        metallic=0.92, r_min=0.12, r_max=0.28, wear_r=0.09, ao_dist=0.022, aniso=0.34, coat_w=0.35, coat_r=0.14,
        rough_floor=0.09, tangent_mode='RADIAL_X',
        f82_tint=0xF8DAD8, rough_breakup_amp=0.72,
    )
    m_gse_grip = _build_accent_elastomer_accent(
        "Mat_Accent_GSE_KnurledRubberGrip", MAT_STAND_STEEL, ng_global, ng_masks, ng_wrap,
        col_a=0x2B2F36, col_b=0x20242A, crevice_col=0x16191E,
        r_min=0.58, r_max=0.72, knurl_scale=340.0,
    )
    m_gse_chrome = _build_accent_metallic_accent(
        "Mat_Accent_GSE_ChromeShaft", MAT_STAND_STEEL, ng_global, ng_masks, ng_wrap,
        col_a=0xCAD0D8, col_b=0xA2AAB4, crevice_col=0x22252A, wear_col=0xDCE2E8,
        metallic=0.99, r_min=0.08, r_max=0.19, wear_r=0.06, ao_dist=0.020, aniso=0.38,
        rough_floor=0.06, tangent_mode='RADIAL_X',
        f82_tint=0xE6ECF2, rough_breakup_amp=0.70,
    )
    m_gse_orange_chrome = _build_accent_metallic_accent(
        "Mat_Accent_GSE_TrunnionChromePin", MAT_STAND_ORANGE, ng_global, ng_masks, ng_wrap,
        col_a=0xCAD0D8, col_b=0xA2AAB4, crevice_col=0x22252A, wear_col=0xDCE2E8,
        metallic=0.99, r_min=0.08, r_max=0.19, wear_r=0.06, ao_dist=0.018, aniso=0.38,
        rough_floor=0.06, tangent_mode='RADIAL_X',
        f82_tint=0xE6ECF2, rough_breakup_amp=0.70,
    )
    m_gse_orange_grip = _build_accent_elastomer_accent(
        "Mat_Accent_GSE_TrunnionBlackKnob", MAT_STAND_ORANGE, ng_global, ng_masks, ng_wrap,
        col_a=0x2B2F36, col_b=0x20242A, crevice_col=0x16191E,
        r_min=0.48, r_max=0.64, knurl_scale=420.0,
    )
    m_tire_n2 = _build_accent_metallic_accent(
        "Mat_Accent_Tire_NitrogenGreenCap", MAT_BRASS_FITTING, ng_global, ng_masks, ng_wrap,
        col_a=0x2CB04C, col_b=0x1C7C32, crevice_col=0x0E3216, wear_col=0xAAD6B2,
        metallic=0.92, r_min=0.12, r_max=0.26, wear_r=0.09, ao_dist=0.010, aniso=0.32, knurl_scale=320.0, coat_w=0.32, coat_r=0.14,
        rough_floor=0.09, tangent_mode='RADIAL_X',
        f82_tint=0xB8ECC4, rough_breakup_amp=0.70,
    )
    m_tire_lead = _build_accent_metallic_accent(
        "Mat_Accent_Tire_LeadWeight", MAT_MACHINED_ALU, ng_global, ng_masks, ng_wrap,
        col_a=0x686E76, col_b=0x4A4F58, crevice_col=0x24282E, wear_col=0x90969E,
        metallic=0.88, r_min=0.44, r_max=0.72, wear_r=0.26, ao_dist=0.014, aniso=0.0,
        rough_floor=0.24, tangent_mode='NONE',
        f82_tint=0xDCE2EA, rough_breakup_amp=0.76,
    )

    stats = {}


    me_inlet = bpy.data.meshes.get("Fwd_01_Inlet_Spinner_Pitot_And_Sensors")
    if me_inlet is not None:
        s_id04 = _slot_indices_by_names(me_inlet, (COLOR_ID_SPECS[MAT_SPINNER_IGV][0], "Mat_Accent_Pitot_MirrorChrome", "Mat_Accent_InletProbes_SatinChrome"))
        s_hw = _slot_indices_by_names(me_inlet, (COLOR_ID_SPECS[MAT_FASTENER][0], COLOR_ID_SPECS[MAT_BRASS_FITTING][0], "Mat_Accent_Spinner_BrightHardware"))
        pitot_f, probe_f, sphw_f = [], [], []
        probe_angs = (142.5, 172.5, 187.5, 217.5)
        for p in me_inlet.polygons:
            c = p.center
            mi = p.material_index
            if mi in s_id04:
                if c.y <= -4.519:
                    pitot_f.append(p.index)
                elif -4.276 <= c.y <= -4.240:
                    r_xy = math.hypot(c.x, c.z)
                    if 0.390 <= r_xy <= 0.494:
                        ang = math.degrees(math.atan2(c.x, c.z)) % 360.0
                        if any(abs(((ang - pa + 180.0) % 360.0) - 180.0) <= 4.0 for pa in probe_angs):
                            probe_f.append(p.index)
            elif mi in s_hw:
                r_xy = math.hypot(c.x, c.z)
                if c.y <= -4.518 or (-4.396 <= c.y <= -4.241 and 0.086 <= r_xy <= 0.170):
                    sphw_f.append(p.index)
        stats["pitot_chrome"] = _assign_faces_to_new_slot(me_inlet, pitot_f, m_pitot)
        stats["inlet_probes"] = _assign_faces_to_new_slot(me_inlet, probe_f, m_probes)
        stats["spinner_hw"] = _assign_faces_to_new_slot(me_inlet, sphw_f, m_sphw)


    me_ot = bpy.data.meshes.get("Fwd_06_Right_Stamped_Saddle_OilTank")
    if me_ot is not None:
        s_pb = _slot_indices_by_names(me_ot, (COLOR_ID_SPECS[MAT_PLATE_BASE][0], "Mat_Accent_OilSight_AmberGlass", "Mat_Accent_OilSight_FloatRed"))
        rot_inv_sg = Matrix.Rotation(-math.radians(239.4), 4, 'Y')
        verts_ot = me_ot.vertices
        glass_f, float_f = [], []
        for p in me_ot.polygons:
            if p.material_index not in s_pb:
                continue
            c = p.center
            if not (-4.072 <= c.y <= -4.046):
                continue
            cl = rot_inv_sg @ c
            if not (-0.053 <= cl.x <= +0.053):
                continue
            d_ax = math.hypot(cl.y - (-4.059), cl.z - 0.5515)
            if d_ax > 0.0067:
                continue

            lxs = [(rot_inv_sg @ verts_ot[vi].co).x for vi in p.vertices]
            x_span = max(lxs) - min(lxs)
            if 0.0096 <= cl.x <= 0.0154 and d_ax >= 0.00545 and x_span < 0.008:
                float_f.append(p.index)
            elif d_ax <= 0.0057 and (x_span > 0.020 or abs(cl.x) >= 0.048):
                glass_f.append(p.index)
        stats["oil_sight_glass"] = _assign_faces_to_new_slot(me_ot, glass_f, m_sight_glass)
        stats["oil_sight_float"] = _assign_faces_to_new_slot(me_ot, float_f, m_sight_float)


    an_blue_tot, an_red_tot = 0, 0
    for mname, target_mat, frac, salt, is_blue in [
        ("Fwd_05_Core_Right_TwinUBleed_RCVV_And_Valves", m_an_blue, 0.24, 31, True),
        ("Fwd_07_Underslung_AGB_FCOC_And_PumpCluster",   m_an_blue, 0.20, 47, True),
        ("Static_Hydraulic_System_Detailed",             m_an_blue, 0.22, 61, True),
        ("Fwd_08_Combustor_Turbine_Case_Fuel_And_Igniters",      m_an_red, 0.24, 73, False),
        ("Fwd_09_Ventral_AugmentorFuel_Metering_And_BellyTrunks", m_an_red, 0.22, 89, False),
    ]:
        me_an = bpy.data.meshes.get(mname)
        if me_an is None:
            continue
        s_br = _slot_indices_by_names(me_an, (COLOR_ID_SPECS[MAT_BRASS_FITTING][0], target_mat.name))
        cand = [p.index for p in me_an.polygons if p.material_index in s_br]
        chosen = _collect_whole_islands_subset(me_an, cand, target_frac=frac, salt=salt)
        cnt = _assign_faces_to_new_slot(me_an, chosen, target_mat)
        if is_blue:
            an_blue_tot += cnt
        else:
            an_red_tot += cnt
    stats["an_blue"] = an_blue_tot
    stats["an_red"] = an_red_tot


    me_sr = bpy.data.meshes.get("Mesh_SyncRing_Pistons")
    if me_sr is not None:
        s_rod = _slot_indices_by_names(me_sr, (COLOR_ID_SPECS[MAT_NOZ_ACT_AB][0], "Mat_Accent_Actuator_HardChromeRod"))
        rod_f = [p.index for p in me_sr.polygons if p.material_index in s_rod and p.center.y <= -0.258]
        stats["chrome_rods"] = _assign_faces_to_new_slot(me_sr, rod_f, m_chrome_rod)


    me_bc = bpy.data.meshes.get("Static_Balance_Chamber_Manifold")
    if me_bc is not None:
        s_bc = _slot_indices_by_names(me_bc, (COLOR_ID_SPECS[MAT_RIGID_TUBE][0], "Mat_Accent_Nozzle_PhosphorBronze"))
        dirs16 = [(math.sin(2.0 * math.pi * (i + 0.5) / 16.0), math.cos(2.0 * math.pi * (i + 0.5) / 16.0)) for i in range(16)]
        btw_xs = (0.0050, 0.0266, 0.0364)
        bw_f = []
        for p in me_bc.polygons:
            if p.material_index not in s_bc:
                continue
            c = p.center
            if not (0.001 <= c.y <= 0.015):
                continue
            ang = math.atan2(c.x, c.z)
            k = int(math.floor((ang % (2.0 * math.pi)) / (2.0 * math.pi / 16.0))) % 16
            sa, ca = dirs16[k]
            lx = abs(c.x * ca - c.z * sa)
            lz = c.x * sa + c.z * ca
            d_yz = math.hypot(c.y - 0.008, lz - 0.562)
            if 0.0033 <= d_yz <= 0.0066 and any(abs(lx - bx) <= 0.00075 for bx in btw_xs):
                bw_f.append(p.index)
        stats["nozzle_bronze_washers"] = _assign_faces_to_new_slot(me_bc, bw_f, m_bronze_w)


    me_sh = bpy.data.meshes.get("Static_Nozzle_Outer_Shroud")
    if me_sh is not None:
        s_sh = _slot_indices_by_names(me_sh, (COLOR_ID_SPECS[MAT_MACHINED_ALU][0], "Mat_Accent_ShroudPort_ChromeRing", "Mat_Accent_ShroudPort_VitonSeal"))
        dirs8p = [(math.sin(math.radians(k * 45.0 + 22.5)), math.cos(math.radians(k * 45.0 + 22.5))) for k in range(8)]
        ring_f, viton_f = [], []
        for p in me_sh.polygons:
            if p.material_index not in s_sh:
                continue
            c = p.center
            if not (-0.283 <= c.y <= -0.253):
                continue
            ang = math.degrees(math.atan2(c.x, c.z)) % 360.0
            k = int(round((ang - 22.5) / 45.0)) % 8
            sa, ca = dirs8p[k]
            lx = c.x * ca - c.z * sa
            rp = math.hypot(lx, c.y - (-0.268))
            if rp <= 0.0125:
                if rp <= 0.0074:
                    viton_f.append(p.index)
                else:
                    ring_f.append(p.index)
        stats["shroud_port_chrome"] = _assign_faces_to_new_slot(me_sh, ring_f, m_port_ring)
        stats["shroud_port_viton"] = _assign_faces_to_new_slot(me_sh, viton_f, m_port_viton)


    me_c2 = bpy.data.meshes.get("Stand_Cradle_02_Pedestals_Rollers_And_Handwheels")
    if me_c2 is not None:
        s_st2 = _slot_indices_by_names(me_c2, (COLOR_ID_SPECS[MAT_STAND_STEEL][0], "Mat_ID27c_Stand_MachinedSteelHardware", "Mat_Accent_GSE_AnodizedRedPin"))
        pin2_f = []
        y_pins2 = (-3.856 - 0.176, -2.333 + 0.176)
        for p in me_c2.polygons:
            if p.material_index not in s_st2:
                continue
            c = p.center
            if abs(c.x) >= 0.785 and (-0.728 <= c.z <= -0.708) and any(abs(c.y - yp) <= 0.028 for yp in y_pins2):
                pin2_f.append(p.index)
        stats["gse_c2_redpins"] = _assign_faces_to_new_slot(me_c2, pin2_f, m_gse_redpin)


    me_c3 = bpy.data.meshes.get("Stand_Cradle_03_RollOver_Arches_And_MountPickups")
    if me_c3 is not None:
        s_st3 = _slot_indices_by_names(me_c3, (
            COLOR_ID_SPECS[MAT_STAND_STEEL][0], "Mat_ID27c_Stand_MachinedSteelHardware",
            "Mat_Accent_GSE_KnurledRubberGrip", "Mat_Accent_GSE_ChromeShaft",
        ))
        s_or3 = _slot_indices_by_names(me_c3, (
            COLOR_ID_SPECS[MAT_STAND_ORANGE][0], "Mat_Accent_GSE_TrunnionChromePin", "Mat_Accent_GSE_TrunnionBlackKnob",
        ))
        pin3_ch_f, pin3_kb_f, grip3_f, arm3_f = [], [], [], []
        y_pins3 = (-3.856 + 0.072, -2.333 + 0.072)
        for p in me_c3.polygons:
            mi = p.material_index
            c = p.center
            if mi in s_or3 and abs(c.x) >= 0.754 and (-0.004 <= c.z <= +0.020) and any(abs(c.y - yp) <= 0.011 for yp in y_pins3):
                if abs(c.x) >= 0.774:
                    pin3_kb_f.append(p.index)
                else:
                    pin3_ch_f.append(p.index)
            elif mi in s_st3:

                if c.x >= 0.809 and (-0.032 <= c.z <= +0.012) and (-4.010 <= c.y <= -3.966):
                    grip3_f.append(p.index)
                elif (0.785 <= c.x < 0.809) and (-0.080 <= c.z <= +0.012) and (-4.010 <= c.y <= -3.830):
                    arm3_f.append(p.index)
        stats["gse_c3_chrome_pins"] = _assign_faces_to_new_slot(me_c3, pin3_ch_f, m_gse_orange_chrome)
        stats["gse_c3_black_knobs"] = _assign_faces_to_new_slot(me_c3, pin3_kb_f, m_gse_orange_grip)
        stats["gse_crank_grip"] = _assign_faces_to_new_slot(me_c3, grip3_f, m_gse_grip)
        stats["gse_crank_chrome"] = _assign_faces_to_new_slot(me_c3, arm3_f, m_gse_chrome)


    me_t5 = bpy.data.meshes.get("Stand_Trailer_05_M3000_Steerable_Axles_And_TowBar")
    if me_t5 is not None:
        s_t5 = _slot_indices_by_names(me_t5, (
            COLOR_ID_SPECS[MAT_STAND_STEEL][0], COLOR_ID_SPECS[MAT_TRAILER_YELLOW][0],
            "Mat_ID27c_Stand_MachinedSteelHardware",
            "Mat_Accent_GSE_AnodizedRedPin", "Mat_Accent_GSE_KnurledRubberGrip",
        ))
        pin5_f, grip5_f = [], []
        for p in me_t5.polygons:
            if p.material_index not in s_t5:
                continue
            c = p.center

            if abs(c.x) <= 0.040 and (-1.356 <= c.y <= -1.330) and (-1.232 <= c.z <= -1.208):
                pin5_f.append(p.index)

            elif c.x >= 0.752 and (-3.615 <= c.y <= -3.565) and (-0.822 <= c.z <= -0.734):
                if c.z > -0.752:
                    pin5_f.append(p.index)
                else:
                    grip5_f.append(p.index)
        stats["gse_t5_redpins"] = _assign_faces_to_new_slot(me_t5, pin5_f, m_gse_redpin)
        stats["gse_brake_grip"] = _assign_faces_to_new_slot(me_t5, grip5_f, m_gse_grip)


    me_t6 = bpy.data.meshes.get("Stand_Trailer_06_Pneumatic_GroundSupport_Wheels")
    if me_t6 is not None:
        s_br6 = _slot_indices_by_names(me_t6, (COLOR_ID_SPECS[MAT_BRASS_FITTING][0], "Mat_Accent_Tire_NitrogenGreenCap"))
        s_al6 = _slot_indices_by_names(me_t6, (COLOR_ID_SPECS[MAT_MACHINED_ALU][0], "Mat_Accent_Tire_LeadWeight"))
        n2_f, lead_f = [], []
        for p in me_t6.polygons:
            c = p.center
            if p.material_index in s_br6 and abs(c.x) >= 0.708 and c.z >= -1.20:
                n2_f.append(p.index)
            elif p.material_index in s_al6 and abs(c.x) >= 0.745 and c.z <= -1.46:
                lead_f.append(p.index)
        stats["tire_n2_green_caps"] = _assign_faces_to_new_slot(me_t6, n2_f, m_tire_n2)
        stats["tire_lead_weights"] = _assign_faces_to_new_slot(me_t6, lead_f, m_tire_lead)

    summary = " | ".join(f"{k}={v}" for k, v in stats.items())
    print(f"[Hardware Accents] {summary}")
    refine_engine_body_details(ng_global, ng_masks, ng_wrap)


def _build_body_detail_woven_sleeve_material(
    mat_name: str, parent_id: int, ng_global, ng_masks, ng_wrap,
    *,
    col_a: int, col_b: int, valley_col: int, crevice_col: int,
    r_min: float = 0.42, r_max: float = 0.58, weave_scale: float = 195.0,
    lacing_bands: bool = False, coat_w: float = 0.0, coat_r: float = 0.30,
    sheen_w: float = 0.35, sss_w: float = 0.06,
):

    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1340, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1340, -340))
    ctrl.node_tree = ng_global
    masks = _new_node(nt, "ShaderNodeGroup", (-1020, 160))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.022
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00024
    masks.inputs["Micro_Scale"].default_value = 520.0

    mix_b = _new_node(nt, "ShaderNodeMix", (-680, 260)); mix_b.data_type = 'RGBA'
    mix_b.inputs[6].default_value = rgba(col_a)
    mix_b.inputs[7].default_value = rgba(col_b)
    nt.links.new(masks.outputs["Macro_Mottle"], mix_b.inputs[0])
    cur_c = mix_b.outputs[2]


    w1 = _new_node(nt, "ShaderNodeTexWave", (-960, -360))
    w1.wave_type = 'BANDS'; w1.bands_direction = 'DIAGONAL'
    w1.inputs["Scale"].default_value = weave_scale; w1.inputs["Distortion"].default_value = 0.55
    nt.links.new(tc.outputs["Object"], w1.inputs["Vector"])
    mp2 = _new_node(nt, "ShaderNodeMapping", (-1160, -520))
    mp2.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(90.0))
    nt.links.new(tc.outputs["Object"], mp2.inputs["Vector"])
    w2 = _new_node(nt, "ShaderNodeTexWave", (-960, -520))
    w2.wave_type = 'BANDS'; w2.bands_direction = 'DIAGONAL'
    w2.inputs["Scale"].default_value = weave_scale; w2.inputs["Distortion"].default_value = 0.55
    nt.links.new(mp2.outputs["Vector"], w2.inputs["Vector"])
    wc = _new_node(nt, "ShaderNodeMath", (-720, -440)); wc.operation = 'MINIMUM'; wc.use_clamp = True
    nt.links.new(w1.outputs["Fac"], wc.inputs[0]); nt.links.new(w2.outputs["Fac"], wc.inputs[1])

    wv_m = _new_node(nt, "ShaderNodeMapRange", (-520, -440)); wv_m.clamp = True
    wv_m.inputs[1].default_value = 0.0; wv_m.inputs[2].default_value = 0.42
    wv_m.inputs[3].default_value = 0.48; wv_m.inputs[4].default_value = 0.0
    nt.links.new(wc.outputs[0], wv_m.inputs[0])
    mix_wv = _new_node(nt, "ShaderNodeMix", (-440, 260)); mix_wv.data_type = 'RGBA'
    mix_wv.inputs[7].default_value = rgba(valley_col)
    nt.links.new(wv_m.outputs[0], mix_wv.inputs[0])
    nt.links.new(cur_c, mix_wv.inputs[6])
    cur_c = mix_wv.outputs[2]


    cr_w = _new_node(nt, "ShaderNodeMath", (-520, -60)); cr_w.operation = 'MULTIPLY'; cr_w.use_clamp = True
    cr_w.inputs[1].default_value = 0.38
    nt.links.new(masks.outputs["Crevice_Mask"], cr_w.inputs[0])
    cr_eff = _new_node(nt, "ShaderNodeMath", (-340, -60)); cr_eff.operation = 'MULTIPLY'; cr_eff.use_clamp = True
    nt.links.new(cr_w.outputs[0], cr_eff.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], cr_eff.inputs[1])
    mix_cr = _new_node(nt, "ShaderNodeMix", (-220, 260)); mix_cr.data_type = 'RGBA'
    mix_cr.inputs[7].default_value = rgba(crevice_col)
    nt.links.new(cr_eff.outputs[0], mix_cr.inputs[0])
    nt.links.new(cur_c, mix_cr.inputs[6])
    cur_c = mix_cr.outputs[2]

    r_m = _new_node(nt, "ShaderNodeMapRange", (-440, 80)); r_m.clamp = True
    r_m.inputs[1].default_value = 0.18; r_m.inputs[2].default_value = 0.84
    r_m.inputs[3].default_value = r_min; r_m.inputs[4].default_value = r_max
    nt.links.new(masks.outputs["Macro_Mottle"], r_m.inputs[0])
    cur_r = r_m.outputs[0]

    lac_bump = None
    if lacing_bands:
        wl = _new_node(nt, "ShaderNodeTexWave", (-720, -680))
        wl.wave_type = 'BANDS'; wl.bands_direction = 'DIAGONAL'
        wl.inputs["Scale"].default_value = 11.0; wl.inputs["Distortion"].default_value = 0.18
        nt.links.new(tc.outputs["Object"], wl.inputs["Vector"])
        lb = _new_node(nt, "ShaderNodeMapRange", (-480, -680)); lb.clamp = True; lb.interpolation_type = 'LINEAR'
        lb.inputs[1].default_value = 0.91; lb.inputs[2].default_value = 0.955
        lb.inputs[3].default_value = 0.0; lb.inputs[4].default_value = 1.0
        nt.links.new(wl.outputs["Fac"], lb.inputs[0])
        lac_bump = lb.outputs[0]
        mix_lc = _new_node(nt, "ShaderNodeMix", (-20, 260)); mix_lc.data_type = 'RGBA'
        mix_lc.inputs[7].default_value = rgba(0x16171B)
        nt.links.new(lac_bump, mix_lc.inputs[0])
        nt.links.new(cur_c, mix_lc.inputs[6])
        cur_c = mix_lc.outputs[2]
        mix_lr = _new_node(nt, "ShaderNodeMix", (-20, 80)); mix_lr.data_type = 'FLOAT'
        mix_lr.inputs[3].default_value = 0.36
        nt.links.new(lac_bump, mix_lr.inputs[0])
        nt.links.new(cur_r, mix_lr.inputs[2])
        cur_r = mix_lr.outputs[0]

    b_wv = _new_node(nt, "ShaderNodeBump", (-140, -340))
    b_wv.inputs["Strength"].default_value = 0.14
    b_wv.inputs["Distance"].default_value = 0.00038
    nt.links.new(wc.outputs[0], b_wv.inputs["Height"])
    nt.links.new(masks.outputs["Bevel_Normal"], b_wv.inputs["Normal"])
    cur_n = b_wv.outputs["Normal"]

    if lac_bump is not None:
        b_lc = _new_node(nt, "ShaderNodeBump", (40, -340))
        b_lc.inputs["Strength"].default_value = 0.22
        b_lc.inputs["Distance"].default_value = 0.0007
        nt.links.new(lac_bump, b_lc.inputs["Height"])
        nt.links.new(cur_n, b_lc.inputs["Normal"])
        cur_n = b_lc.outputs["Normal"]

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (240, 140))
    nt.links.new(cur_c, bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    nt.links.new(cur_r, bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.42
    bsdf.inputs["Subsurface Weight"].default_value = sss_w
    bsdf.inputs["Subsurface Scale"].default_value = 0.003
    bsdf.inputs["Sheen Weight"].default_value = sheen_w
    bsdf.inputs["Sheen Roughness"].default_value = 0.62
    bsdf.inputs["Coat Weight"].default_value = coat_w
    bsdf.inputs["Coat Roughness"].default_value = coat_r
    if coat_w > 1e-4 and "Coat Normal" in bsdf.inputs:
        nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Coat Normal"])
    nt.links.new(cur_n, bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (560, 140))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(COLOR_ID_SPECS[parent_id][1])
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (820, 140))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])
    mat["f15_id_hex"] = int(COLOR_ID_SPECS[parent_id][1])
    mat["f15_pbr_hex"] = int(col_a)
    mat.diffuse_color = rgba(col_a)
    return mat


def _build_body_detail_actuator_bellows_material(ng_global, ng_masks, ng_wrap):

    mat = bpy.data.materials.get("Mat_BodyDetail_Actuator_BellowsGland") or bpy.data.materials.new("Mat_BodyDetail_Actuator_BellowsGland")
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1340, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1340, -320))
    ctrl.node_tree = ng_global
    masks = _new_node(nt, "ShaderNodeGroup", (-1020, 180))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.020
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00020
    masks.inputs["Micro_Scale"].default_value = 540.0


    mix_b = _new_node(nt, "ShaderNodeMix", (-680, 260)); mix_b.data_type = 'RGBA'
    mix_b.inputs[6].default_value = rgba(0x5E646C)
    mix_b.inputs[7].default_value = rgba(0x3C4148)
    nt.links.new(masks.outputs["Macro_Mottle"], mix_b.inputs[0])


    sep_y = _new_node(nt, "ShaderNodeSeparateXYZ", (-960, -140))
    nt.links.new(tc.outputs["Object"], sep_y.inputs["Vector"])
    gland_y = _new_node(nt, "ShaderNodeMapRange", (-740, -140)); gland_y.clamp = True
    gland_y.inputs[1].default_value = -0.742; gland_y.inputs[2].default_value = -0.718
    gland_y.inputs[3].default_value = 0.0; gland_y.inputs[4].default_value = 0.82
    nt.links.new(sep_y.outputs["Y"], gland_y.inputs[0])

    oil_v0 = _new_node(nt, "ShaderNodeMath", (-520, -60)); oil_v0.operation = 'MAXIMUM'; oil_v0.use_clamp = True
    nt.links.new(masks.outputs["Crevice_Mask"], oil_v0.inputs[0])
    nt.links.new(gland_y.outputs[0], oil_v0.inputs[1])
    oil_eff = _new_node(nt, "ShaderNodeMath", (-340, -60)); oil_eff.operation = 'MULTIPLY'; oil_eff.use_clamp = True
    nt.links.new(oil_v0.outputs[0], oil_eff.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], oil_eff.inputs[1])

    mix_oc = _new_node(nt, "ShaderNodeMix", (-180, 240)); mix_oc.data_type = 'RGBA'
    mix_oc.inputs[7].default_value = rgba(0x16120D)
    nt.links.new(oil_eff.outputs[0], mix_oc.inputs[0])
    nt.links.new(mix_b.outputs[2], mix_oc.inputs[6])


    ew_w = _new_node(nt, "ShaderNodeMath", (-340, -200)); ew_w.operation = 'MULTIPLY'; ew_w.use_clamp = True
    ew_w.inputs[1].default_value = 0.45
    nt.links.new(masks.outputs["EdgeWear_Mask"], ew_w.inputs[0])
    mix_ec = _new_node(nt, "ShaderNodeMix", (0, 240)); mix_ec.data_type = 'RGBA'
    mix_ec.inputs[7].default_value = rgba(0xABB2BA)
    nt.links.new(ew_w.outputs[0], mix_ec.inputs[0])
    nt.links.new(mix_oc.outputs[2], mix_ec.inputs[6])


    r_src_bel = _new_node(nt, "ShaderNodeMix", (-640, 80)); r_src_bel.data_type = 'FLOAT'
    r_src_bel.inputs[0].default_value = 0.74
    nt.links.new(masks.outputs["Macro_Mottle"], r_src_bel.inputs[2])
    nt.links.new(masks.outputs["Metal_Rough_Breakup"], r_src_bel.inputs[3])

    r_m = _new_node(nt, "ShaderNodeMapRange", (-460, 80)); r_m.clamp = True
    r_m.inputs[1].default_value = 0.14; r_m.inputs[2].default_value = 0.86
    r_m.inputs[3].default_value = 0.16; r_m.inputs[4].default_value = 0.40
    nt.links.new(r_src_bel.outputs[0], r_m.inputs[0])
    mix_r = _new_node(nt, "ShaderNodeMix", (-120, 60)); mix_r.data_type = 'FLOAT'; mix_r.inputs[3].default_value = 0.10
    nt.links.new(oil_eff.outputs[0], mix_r.inputs[0])
    nt.links.new(r_m.outputs[0], mix_r.inputs[2])

    r_flr_bel = _new_node(nt, "ShaderNodeMath", (60, 60)); r_flr_bel.operation = 'MAXIMUM'; r_flr_bel.use_clamp = True
    r_flr_bel.name = "Metal_Roughness_Floor"
    r_flr_bel.inputs[1].default_value = 0.10
    nt.links.new(mix_r.outputs[0], r_flr_bel.inputs[0])

    coat_w = _new_node(nt, "ShaderNodeMath", (-120, -100)); coat_w.operation = 'MULTIPLY'; coat_w.use_clamp = True
    coat_w.inputs[1].default_value = 0.76
    nt.links.new(oil_eff.outputs[0], coat_w.inputs[0])

    b_g = _new_node(nt, "ShaderNodeBump", (-120, -280))
    b_g.inputs["Strength"].default_value = 0.025; b_g.inputs["Distance"].default_value = 0.00022
    nt.links.new(masks.outputs["Micro_Grain"], b_g.inputs["Height"])
    nt.links.new(masks.outputs["Bevel_Normal"], b_g.inputs["Normal"])


    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (220, 120))
    nt.links.new(mix_ec.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.96
    nt.links.new(r_flr_bel.outputs[0], bsdf.inputs["Roughness"])
    if "Specular Tint" in bsdf.inputs:
        bsdf.inputs["Specular Tint"].default_value = rgba(0xDCE2EA)
    bsdf.inputs["Anisotropic"].default_value = 0.0
    nt.links.new(coat_w.outputs[0], bsdf.inputs["Coat Weight"])
    bsdf.inputs["Coat Roughness"].default_value = 0.08
    if "Coat Normal" in bsdf.inputs:
        nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Coat Normal"])
    nt.links.new(b_g.outputs["Normal"], bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (540, 120))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(COLOR_ID_SPECS[MAT_NOZ_ACT_AB][1])
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (800, 120))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])
    mat["f15_id_hex"] = int(COLOR_ID_SPECS[MAT_NOZ_ACT_AB][1])
    mat["f15_pbr_hex"] = int(0x4E535A)
    mat.diffuse_color = rgba(0x4E535A)
    mat.metallic = 0.96
    return mat


def refine_engine_body_details(ng_global, ng_masks, ng_wrap):


    m_inlet_lip = _build_accent_metallic_accent(
        "Mat_BodyDetail_InletLip_MachinedSilver", MAT_TI_CASE, ng_global, ng_masks, ng_wrap,
        col_a=0xCCD2DA, col_b=0xA4ACB6, crevice_col=0x22262B, wear_col=0xDCE2E8,
        metallic=0.99, r_min=0.09, r_max=0.24, wear_r=0.07,
        ao_dist=0.018, aniso=0.48, lathe_scale=230.0,
        inlet_runway_dust=True, rough_floor=0.07, tangent_mode='RADIAL_Y',
        f82_tint=0xE6ECF2, rough_breakup_amp=0.72,
    )
    m_firesleeve = _build_body_detail_woven_sleeve_material(
        "Mat_BodyDetail_FireZone_Firesleeve", MAT_BRAID_HOSE, ng_global, ng_masks, ng_wrap,
        col_a=0xB83418, col_b=0x82220E, valley_col=0x461207, crevice_col=0x260A04,
        r_min=0.34, r_max=0.48, weave_scale=240.0, lacing_bands=False,
        coat_w=0.34, coat_r=0.22, sheen_w=0.14, sss_w=0.07,
    )
    m_yellow_loom = _build_body_detail_woven_sleeve_material(
        "Mat_BodyDetail_YellowNomex_Braid", MAT_BRAID_HOSE, ng_global, ng_masks, ng_wrap,
        col_a=0xDFB624, col_b=0xB89016, valley_col=0x725814, crevice_col=0x46340E,
        r_min=0.42, r_max=0.58, weave_scale=185.0, lacing_bands=True,
        coat_w=0.0, sheen_w=0.46, sss_w=0.09,
    )
    m_bellows = _build_body_detail_actuator_bellows_material(ng_global, ng_masks, ng_wrap)

    st = {}

    me_f2 = bpy.data.meshes.get("Fwd_02_Fan_Case_VaneLinkage_And_DEEC")
    if me_f2 is not None:
        s_ti = _slot_indices_by_names(me_f2, (COLOR_ID_SPECS[MAT_TI_CASE][0], "Mat_BodyDetail_InletLip_MachinedSilver"))
        lip_f = [
            p.index for p in me_f2.polygons
            if p.material_index in s_ti and p.center.y <= -4.265 and (p.normal.y < -0.15 or p.center.y <= -4.271)
        ]
        st["inlet_lip_silver"] = _assign_faces_to_new_slot(me_f2, lip_f, m_inlet_lip)


    me_f8 = bpy.data.meshes.get("Fwd_08_Combustor_Turbine_Case_Fuel_And_Igniters")
    if me_f8 is not None:
        s_h8 = _slot_indices_by_names(me_f8, (
            COLOR_ID_SPECS[MAT_BRAID_HOSE][0], "Mat_BodyDetail_FireZone_Firesleeve", "Mat_BodyDetail_YellowNomex_Braid",
        ))
        fs8_f, yl8_f = [], []
        for p in me_f8.polygons:
            if p.material_index not in s_h8:
                continue
            c = p.center

            if (-2.370 <= c.y <= -2.342) or (abs(c.x) >= 0.44 and -3.04 <= c.y <= -2.655 and -0.28 <= c.z <= 0.38):
                fs8_f.append(p.index)

            elif -2.655 < c.y <= -2.625:
                yl8_f.append(p.index)
        st["core_firesleeve_f8"] = _assign_faces_to_new_slot(me_f8, fs8_f, m_firesleeve)
        st["core_yellow_f8"] = _assign_faces_to_new_slot(me_f8, yl8_f, m_yellow_loom)

    me_f4 = bpy.data.meshes.get("Fwd_04_Core_Left_SBleed_ServoValve_And_Tubes")
    if me_f4 is not None:
        s_h4 = _slot_indices_by_names(me_f4, (COLOR_ID_SPECS[MAT_BRAID_HOSE][0], "Mat_BodyDetail_YellowNomex_Braid"))
        yl4_f = [
            p.index for p in me_f4.polygons
            if p.material_index in s_h4 and (p.center.x >= 0.34 and -3.75 <= p.center.y <= -3.08 and -0.40 <= p.center.z <= 0.44)
        ]
        st["core_yellow_f4"] = _assign_faces_to_new_slot(me_f4, yl4_f, m_yellow_loom)


    me_f9 = bpy.data.meshes.get("Fwd_09_Ventral_AugmentorFuel_Metering_And_BellyTrunks")
    if me_f9 is not None:
        s_h9 = _slot_indices_by_names(me_f9, (
            COLOR_ID_SPECS[MAT_BRAID_HOSE][0], "Mat_BodyDetail_FireZone_Firesleeve", "Mat_BodyDetail_YellowNomex_Braid",
        ))
        fs9_f, yl9_f = [], []
        for p in me_f9.polygons:
            if p.material_index not in s_h9:
                continue
            c = p.center
            ax = abs(c.x)
            if ax >= 0.32 and -3.02 <= c.y <= -1.90 and -0.46 <= c.z <= -0.38:
                d_upper = (ax - 0.381) ** 2 + (c.z + 0.412) ** 2
                d_lower = (ax - 0.366) ** 2 + (c.z + 0.431) ** 2
                if d_upper <= d_lower:
                    yl9_f.append(p.index)
                else:
                    fs9_f.append(p.index)
        st["belly_firesleeve_f9"] = _assign_faces_to_new_slot(me_f9, fs9_f, m_firesleeve)
        st["belly_yellow_f9"] = _assign_faces_to_new_slot(me_f9, yl9_f, m_yellow_loom)


    me_act = bpy.data.meshes.get("Static_Hydraulic_System_Detailed")
    if me_act is not None:
        s_act = _slot_indices_by_names(me_act, (COLOR_ID_SPECS[MAT_NOZ_ACT_AB][0], "Mat_BodyDetail_Actuator_BellowsGland"))
        dirs8 = [(math.sin(math.radians(k * 45.0)), math.cos(math.radians(k * 45.0))) for k in range(8)]
        bel_f = []
        for p in me_act.polygons:
            if p.material_index not in s_act:
                continue
            c = p.center
            if c.y >= -0.776:
                ang = math.atan2(c.x, c.z)
                k = int(round(math.degrees(ang) / 45.0)) % 8
                sa, ca = dirs8[k]
                local_t = abs(c.x * ca - c.z * sa)
                if local_t < 0.0295:
                    bel_f.append(p.index)
        st["actuator_bellows"] = _assign_faces_to_new_slot(me_act, bel_f, m_bellows)

    print("[Engine Body Details] " + " | ".join(f"{k}={v}" for k, v in st.items()))
    refine_gse_props(ng_global, ng_masks, ng_wrap)


def _build_gse_rbf_twill_material(ng_global, ng_masks, ng_wrap):

    png_path = os.path.join(_THIS_DIR, "assets_decals", "decal_rbf_streamer.png")
    img = _load_packed_decal_img(png_path)
    mat = bpy.data.materials.get("Mat_GSE_RBF_TwillFabric") or bpy.data.materials.new("Mat_GSE_RBF_TwillFabric")
    nt = mat.node_tree
    nt.nodes.clear()

    uv = _new_node(nt, "ShaderNodeUVMap", (-1180, 200))
    tex = _new_node(nt, "ShaderNodeTexImage", (-920, 220))
    tex.image = img
    tex.extension = 'CLIP'
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1180, -180))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1180, -420))
    ctrl.node_tree = ng_global
    masks = _new_node(nt, "ShaderNodeGroup", (-920, -200))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.018
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00018


    dust_f = _new_node(nt, "ShaderNodeMath", (-620, -240)); dust_f.operation = 'MULTIPLY'; dust_f.use_clamp = True
    dust_f.inputs[1].default_value = 0.14
    nt.links.new(ctrl.outputs["Weathering"], dust_f.inputs[0])
    dust_m = _new_node(nt, "ShaderNodeMath", (-420, -240)); dust_m.operation = 'MULTIPLY'; dust_m.use_clamp = True
    nt.links.new(dust_f.outputs[0], dust_m.inputs[0])
    nt.links.new(masks.outputs["Crevice_Mask"], dust_m.inputs[1])

    col_m = _new_node(nt, "ShaderNodeMix", (-220, 200)); col_m.data_type = 'RGBA'
    col_m.inputs[7].default_value = rgba(0x3A1814)
    nt.links.new(dust_m.outputs[0], col_m.inputs[0])
    nt.links.new(tex.outputs["Color"], col_m.inputs[6])


    mp_tw = _new_node(nt, "ShaderNodeMapping", (-920, -440))
    mp_tw.inputs["Scale"].default_value = (4.615, 1.0, 1.0)
    nt.links.new(uv.outputs["UV"], mp_tw.inputs["Vector"])
    wv = _new_node(nt, "ShaderNodeTexWave", (-680, -440))
    wv.wave_type = 'BANDS'; wv.bands_direction = 'DIAGONAL'
    wv.inputs["Scale"].default_value = 92.0
    nt.links.new(mp_tw.outputs["Vector"], wv.inputs["Vector"])

    bw_emb = _new_node(nt, "ShaderNodeRGBToBW", (-680, 40))
    nt.links.new(tex.outputs["Color"], bw_emb.inputs["Color"])
    emb_m = _new_node(nt, "ShaderNodeMapRange", (-480, 40)); emb_m.clamp = True
    emb_m.inputs[1].default_value = 0.56; emb_m.inputs[2].default_value = 0.86
    emb_m.inputs[3].default_value = 0.0; emb_m.inputs[4].default_value = 1.0
    nt.links.new(bw_emb.outputs["Val"], emb_m.inputs[0])

    h_tw = _new_node(nt, "ShaderNodeMath", (-460, -440)); h_tw.operation = 'MULTIPLY'
    h_tw.inputs[1].default_value = 0.42
    nt.links.new(wv.outputs["Fac"], h_tw.inputs[0])
    h_comb = _new_node(nt, "ShaderNodeMath", (-280, -380)); h_comb.operation = 'MULTIPLY_ADD'
    h_comb.inputs[1].default_value = 0.85
    nt.links.new(emb_m.outputs[0], h_comb.inputs[0])
    nt.links.new(h_tw.outputs[0], h_comb.inputs[2])

    bmp = _new_node(nt, "ShaderNodeBump", (-100, -320))
    bmp.inputs["Strength"].default_value = 0.14
    bmp.inputs["Distance"].default_value = 0.00042
    nt.links.new(h_comb.outputs[0], bmp.inputs["Height"])
    nt.links.new(masks.outputs["Bevel_Normal"], bmp.inputs["Normal"])

    r_tw = _new_node(nt, "ShaderNodeMapRange", (-120, -60)); r_tw.clamp = True
    r_tw.inputs[1].default_value = 0.0; r_tw.inputs[2].default_value = 1.0
    r_tw.inputs[3].default_value = 0.76; r_tw.inputs[4].default_value = 0.92
    nt.links.new(wv.outputs["Fac"], r_tw.inputs[0])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (160, 140))
    nt.links.new(col_m.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    nt.links.new(r_tw.outputs[0], bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.24
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.045
    if "Subsurface Scale" in bsdf.inputs:
        bsdf.inputs["Subsurface Scale"].default_value = 0.002
    if "Sheen Weight" in bsdf.inputs:
        bsdf.inputs["Sheen Weight"].default_value = 0.58
    if "Sheen Roughness" in bsdf.inputs:
        bsdf.inputs["Sheen Roughness"].default_value = 0.46
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (480, 140))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(COLOR_ID_SPECS[MAT_STAND_ORANGE][1])
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (740, 140))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(COLOR_ID_SPECS[MAT_STAND_ORANGE][1])
    mat["f15_pbr_hex"] = 0xB81A14
    mat.diffuse_color = rgba(0xB81A14)
    return mat


def _build_gse_chock_hazard_material(ng_global, ng_masks, ng_wrap):

    mat = bpy.data.materials.get("Mat_GSE_Chock_HazardStripe") or bpy.data.materials.new("Mat_GSE_Chock_HazardStripe")
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1240, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1240, -340))
    ctrl.node_tree = ng_global
    masks = _new_node(nt, "ShaderNodeGroup", (-940, -240))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.015
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00022
    masks.inputs["Micro_Scale"].default_value = 620.0


    mp = _new_node(nt, "ShaderNodeMapping", (-960, 160))
    mp.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(45.0))
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    wv = _new_node(nt, "ShaderNodeTexWave", (-740, 160))
    wv.wave_type = 'BANDS'; wv.bands_direction = 'X'; wv.wave_profile = 'SAW'
    wv.inputs["Scale"].default_value = 6.2
    nt.links.new(mp.outputs["Vector"], wv.inputs["Vector"])
    st = _new_node(nt, "ShaderNodeMapRange", (-520, 160)); st.clamp = True
    st.inputs[1].default_value = 0.47; st.inputs[2].default_value = 0.53
    nt.links.new(wv.outputs["Fac"], st.inputs[0])

    col_st = _new_node(nt, "ShaderNodeMix", (-320, 180)); col_st.data_type = 'RGBA'
    col_st.inputs[6].default_value = rgba(0x101216)
    col_st.inputs[7].default_value = rgba(0xCEA016)
    nt.links.new(st.outputs[0], col_st.inputs[0])


    wear_raw = _new_node(nt, "ShaderNodeMath", (-640, -200)); wear_raw.operation = 'MAXIMUM'; wear_raw.use_clamp = True
    nt.links.new(masks.outputs["Photo_Smudge"], wear_raw.inputs[0])
    nt.links.new(masks.outputs["Scratch_Mask"], wear_raw.inputs[1])
    w_bd = _new_node(nt, "ShaderNodeMath", (-640, -360)); w_bd.operation = 'MULTIPLY_ADD'; w_bd.use_clamp = True
    w_bd.inputs[1].default_value = 0.35
    nt.links.new(ctrl.outputs["SurfaceErosion"], w_bd.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], w_bd.inputs[2])
    wear_eff = _new_node(nt, "ShaderNodeMath", (-420, -240)); wear_eff.operation = 'MULTIPLY'; wear_eff.use_clamp = True
    nt.links.new(wear_raw.outputs[0], wear_eff.inputs[0])
    nt.links.new(w_bd.outputs[0], wear_eff.inputs[1])

    cf0 = _new_node(nt, "ShaderNodeMath", (-260, -140)); cf0.operation = 'MULTIPLY_ADD'; cf0.use_clamp = True
    cf0.inputs[1].default_value = 0.46
    nt.links.new(wear_eff.outputs[0], cf0.inputs[0])
    cr_w = _new_node(nt, "ShaderNodeMath", (-420, -100)); cr_w.operation = 'MULTIPLY'; cr_w.use_clamp = True
    cr_w.inputs[1].default_value = 0.34
    nt.links.new(masks.outputs["Crevice_Mask"], cr_w.inputs[0])
    nt.links.new(cr_w.outputs[0], cf0.inputs[2])

    col_m = _new_node(nt, "ShaderNodeMix", (-80, 180)); col_m.data_type = 'RGBA'
    col_m.inputs[7].default_value = rgba(0x282622)
    nt.links.new(cf0.outputs[0], col_m.inputs[0])
    nt.links.new(col_st.outputs[2], col_m.inputs[6])

    r_ch = _new_node(nt, "ShaderNodeMapRange", (-120, 0)); r_ch.clamp = True
    r_ch.inputs[1].default_value = 0.0; r_ch.inputs[2].default_value = 1.0
    r_ch.inputs[3].default_value = 0.26; r_ch.inputs[4].default_value = 0.68
    nt.links.new(wear_eff.outputs[0], r_ch.inputs[0])

    cw_ch = _new_node(nt, "ShaderNodeMapRange", (-120, -140)); cw_ch.clamp = True
    cw_ch.inputs[1].default_value = 0.0; cw_ch.inputs[2].default_value = 0.85
    cw_ch.inputs[3].default_value = 0.48; cw_ch.inputs[4].default_value = 0.06
    nt.links.new(wear_eff.outputs[0], cw_ch.inputs[0])


    sc_inv = _new_node(nt, "ShaderNodeMath", (-320, -400)); sc_inv.operation = 'SUBTRACT'; sc_inv.use_clamp = True
    sc_inv.inputs[0].default_value = 1.0
    nt.links.new(masks.outputs["Scratch_Mask"], sc_inv.inputs[1])
    h_ch = _new_node(nt, "ShaderNodeMath", (-160, -400)); h_ch.operation = 'MULTIPLY_ADD'
    h_ch.inputs[1].default_value = 0.65
    nt.links.new(sc_inv.outputs[0], h_ch.inputs[0])
    mg_ch = _new_node(nt, "ShaderNodeMath", (-320, -520)); mg_ch.operation = 'MULTIPLY'; mg_ch.inputs[1].default_value = 0.35
    nt.links.new(masks.outputs["Micro_Grain"], mg_ch.inputs[0])
    nt.links.new(mg_ch.outputs[0], h_ch.inputs[2])
    b_ch = _new_node(nt, "ShaderNodeBump", (20, -340))
    b_ch.inputs["Strength"].default_value = 0.095
    b_ch.inputs["Distance"].default_value = 0.00035
    nt.links.new(h_ch.outputs[0], b_ch.inputs["Height"])
    nt.links.new(masks.outputs["Bevel_Normal"], b_ch.inputs["Normal"])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (200, 140))
    nt.links.new(col_m.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    nt.links.new(r_ch.outputs[0], bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.50
    nt.links.new(cw_ch.outputs[0], bsdf.inputs["Coat Weight"])
    bsdf.inputs["Coat Roughness"].default_value = 0.18
    if "Coat Normal" in bsdf.inputs:
        nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Coat Normal"])
    nt.links.new(b_ch.outputs["Normal"], bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (500, 140))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(COLOR_ID_SPECS[MAT_PLATE_BASE][1])
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (760, 140))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])
    mat["f15_id_hex"] = int(COLOR_ID_SPECS[MAT_PLATE_BASE][1])
    mat["f15_pbr_hex"] = 0xCEA016
    mat.diffuse_color = rgba(0xCEA016)
    return mat


def _build_gse_drippan_oil_material(ng_global, ng_masks, ng_wrap):

    mat = bpy.data.materials.get("Mat_GSE_DripPan_OilPool") or bpy.data.materials.new("Mat_GSE_DripPan_OilPool")
    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1120, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1120, -340))
    ctrl.node_tree = ng_global
    masks = _new_node(nt, "ShaderNodeGroup", (-840, -240))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.025
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00018

    n_sw = _new_node(nt, "ShaderNodeTexNoise", (-780, 160))
    n_sw.inputs["Scale"].default_value = 6.5
    n_sw.inputs["Detail"].default_value = 5.0
    n_sw.inputs["Distortion"].default_value = 0.85
    nt.links.new(tc.outputs["Object"], n_sw.inputs["Vector"])

    col_m = _new_node(nt, "ShaderNodeMix", (-420, 160)); col_m.data_type = 'RGBA'
    col_m.inputs[6].default_value = rgba(0x2A1405)
    col_m.inputs[7].default_value = rgba(0x562A0A)
    nt.links.new(n_sw.outputs["Factor"], col_m.inputs[0])


    edge_r = _new_node(nt, "ShaderNodeMath", (-460, -140)); edge_r.operation = 'MULTIPLY'; edge_r.use_clamp = True
    nt.links.new(masks.outputs["Photo_Smudge"], edge_r.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], edge_r.inputs[1])
    r_oil = _new_node(nt, "ShaderNodeMapRange", (-220, -20)); r_oil.clamp = True
    r_oil.inputs[1].default_value = 0.0; r_oil.inputs[2].default_value = 1.0
    r_oil.inputs[3].default_value = 0.10; r_oil.inputs[4].default_value = 0.18
    nt.links.new(edge_r.outputs[0], r_oil.inputs[0])

    b_o = _new_node(nt, "ShaderNodeBump", (-180, -180))
    b_o.inputs["Strength"].default_value = 0.045
    b_o.inputs["Distance"].default_value = 0.00045
    nt.links.new(n_sw.outputs["Factor"], b_o.inputs["Height"])
    nt.links.new(masks.outputs["Bevel_Normal"], b_o.inputs["Normal"])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (160, 140))
    nt.links.new(col_m.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    nt.links.new(r_oil.outputs[0], bsdf.inputs["Roughness"])
    bsdf.inputs["IOR"].default_value = 1.47
    bsdf.inputs["Specular IOR Level"].default_value = 0.50
    bsdf.inputs["Coat Weight"].default_value = 0.0
    nt.links.new(b_o.outputs["Normal"], bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (480, 140))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(COLOR_ID_SPECS[MAT_PLATE_BASE][1])
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (740, 140))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])
    mat["f15_id_hex"] = int(COLOR_ID_SPECS[MAT_PLATE_BASE][1])
    mat["f15_pbr_hex"] = 0x1A0B03
    mat.diffuse_color = rgba(0x1A0B03)
    return mat


def refine_gse_props(ng_global, ng_masks, ng_wrap):


    m_dp_ss = _build_accent_metallic_accent(
        "Mat_GSE_DripPan_Stainless", MAT_RIGID_TUBE, ng_global, ng_masks, ng_wrap,
        col_a=0xBAC2CA, col_b=0x808892, crevice_col=0x221E1A, wear_col=0xD4DAE0,
        metallic=0.98, r_min=0.14, r_max=0.38, wear_r=0.10, ao_dist=0.022, aniso=0.0,
        rough_floor=0.10, tangent_mode='NONE',
        f82_tint=0xE2E8F0, rough_breakup_amp=0.84,
    )
    m_dp_oil = _build_gse_drippan_oil_material(ng_global, ng_masks, ng_wrap)
    m_ch_haz = _build_gse_chock_hazard_material(ng_global, ng_masks, ng_wrap)
    m_gc_yel = _build_accent_elastomer_accent(
        "Mat_GSE_GroundCable_Yellow", MAT_TRAILER_YELLOW, ng_global, ng_masks, ng_wrap,
        col_a=0xD8A812, col_b=0xA8800C, crevice_col=0x382A06,
        r_min=0.36, r_max=0.52, coat_w=0.26, coat_r=0.24,
    )
    m_rbf = _build_gse_rbf_twill_material(ng_global, ng_masks, ng_wrap)

    st14 = {}
    me_gse = bpy.data.meshes.get("Stand_Trailer_07_Deployed_GSE_Props")
    if me_gse is not None:
        s_ss = _slot_indices_by_names(me_gse, (COLOR_ID_SPECS[MAT_RIGID_TUBE][0], "Mat_GSE_DripPan_Stainless"))
        s_pb = _slot_indices_by_names(me_gse, (COLOR_ID_SPECS[MAT_PLATE_BASE][0], "Mat_GSE_DripPan_OilPool", "Mat_GSE_Chock_HazardStripe"))
        s_yl = _slot_indices_by_names(me_gse, (COLOR_ID_SPECS[MAT_TRAILER_YELLOW][0], "Mat_GSE_GroundCable_Yellow"))
        dp_ss_f, dp_oil_f, ch_haz_f, gc_yel_f = [], [], [], []
        for p in me_gse.polygons:
            mi = p.material_index
            c = p.center
            if mi in s_ss:
                dp_ss_f.append(p.index)
            elif mi in s_pb:
                if abs(c.x) < 0.44:
                    dp_oil_f.append(p.index)
                else:
                    ch_haz_f.append(p.index)
            elif mi in s_yl:
                gc_yel_f.append(p.index)
        st14["drippan_ss"] = _assign_faces_to_new_slot(me_gse, dp_ss_f, m_dp_ss)
        st14["drippan_oil"] = _assign_faces_to_new_slot(me_gse, dp_oil_f, m_dp_oil)
        st14["chock_hazard"] = _assign_faces_to_new_slot(me_gse, ch_haz_f, m_ch_haz)
        st14["ground_cable"] = _assign_faces_to_new_slot(me_gse, gc_yel_f, m_gc_yel)

    me_rbf = bpy.data.meshes.get("Stand_Cradle_05_RBF_Streamers")
    if me_rbf is not None:
        s_or = _slot_indices_by_names(me_rbf, (COLOR_ID_SPECS[MAT_STAND_ORANGE][0], "Mat_GSE_RBF_TwillFabric"))
        rbf_f = [p.index for p in me_rbf.polygons if p.material_index in s_or]
        st14["rbf_fabric"] = _assign_faces_to_new_slot(me_rbf, rbf_f, m_rbf)


    for _me_clean in (me_gse, me_rbf):
        if _me_clean is None or len(_me_clean.materials) <= 1:
            continue
        used_idx = {p.material_index for p in _me_clean.polygons}
        if len(used_idx) < len(_me_clean.materials):
            old_mats = list(_me_clean.materials)
            remap = {}
            new_mats = []
            for oi, om in enumerate(old_mats):
                if oi in used_idx:
                    remap[oi] = len(new_mats)
                    new_mats.append(om)
            new_indices = [remap[p.material_index] for p in _me_clean.polygons]
            _me_clean.materials.clear()
            for nm in new_mats:
                _me_clean.materials.append(nm)
            _me_clean.polygons.foreach_set("material_index", new_indices)
            _me_clean.update()

    print("[Ground Support Materials] " + " | ".join(f"{k}={v}" for k, v in st14.items()))


def _build_ng_decal_weathering(ng_global):

    old = bpy.data.node_groups.get("NG_F15_DecalWeathering")
    if old is not None:
        bpy.data.node_groups.remove(old, do_unlink=True)
    ng = bpy.data.node_groups.new("NG_F15_DecalWeathering", 'ShaderNodeTree')

    def _in(name, stype, default=None):
        s = ng.interface.new_socket(name=name, in_out='INPUT', socket_type=stype)
        if default is not None and hasattr(s, "default_value"):
            s.default_value = default
        return s

    _in("Decal_Color",   'NodeSocketColor', (1, 1, 1, 1))
    _in("Decal_Alpha",   'NodeSocketFloat', 1.0)
    _in("UV",            'NodeSocketVector')
    _in("Roughness",     'NodeSocketFloat', 0.45)
    _in("Coat_Weight",   'NodeSocketFloat', 0.15)
    _in("Coat_Rough",    'NodeSocketFloat', 0.25)
    _in("Metallic",      'NodeSocketFloat', 0.0)
    _in("Erosion_Gain",  'NodeSocketFloat', 0.55)
    _in("Soot_Gain",     'NodeSocketFloat', 0.40)
    _in("Reflective",    'NodeSocketFloat', 0.0)
    ng.interface.new_socket(name="Shader", in_out='OUTPUT', socket_type='NodeSocketShader')

    inp = _new_node(ng, "NodeGroupInput", (-1480, 0))
    out = _new_node(ng, "NodeGroupOutput", (960, 0))

    ctrl = _new_node(ng, "ShaderNodeGroup", (-1480, -420))
    ctrl.node_tree = ng_global

    tc = _new_node(ng, "ShaderNodeTexCoord", (-1680, 320))


    oi = _new_node(ng, "ShaderNodeObjectInfo", (-1680, 520))
    oi_vec = _new_node(ng, "ShaderNodeVectorMath", (-1500, 520)); oi_vec.operation = 'SCALE'
    oi_vec.inputs[0].default_value = (7.31, 13.17, 5.89)
    ng.links.new(oi.outputs["Random"], oi_vec.inputs["Scale"])
    coord_rand = _new_node(ng, "ShaderNodeVectorMath", (-1440, 340)); coord_rand.operation = 'ADD'
    ng.links.new(tc.outputs["Object"], coord_rand.inputs[0])
    ng.links.new(oi_vec.outputs["Vector"], coord_rand.inputs[1])


    map_lsc = _new_node(ng, "ShaderNodeMapping", (-1300, 340))
    map_lsc.inputs["Scale"].default_value = (2.4, 52.0, 38.0)
    map_lsc.inputs["Rotation"].default_value = (0.18, -0.26, 0.22)
    ng.links.new(coord_rand.outputs["Vector"], map_lsc.inputs["Vector"])
    n_lsc = _new_node(ng, "ShaderNodeTexNoise", (-1060, 340))
    n_lsc.inputs["Scale"].default_value = 1.6
    n_lsc.inputs["Detail"].default_value = 6.0
    n_lsc.inputs["Roughness"].default_value = 0.58
    n_lsc.inputs["Distortion"].default_value = 0.08
    ng.links.new(map_lsc.outputs["Vector"], n_lsc.inputs["Vector"])


    n_sc = _new_node(ng, "ShaderNodeTexNoise", (-1180, 100))
    n_sc.inputs["Scale"].default_value = 18.0
    n_sc.inputs["Detail"].default_value = 10.0
    n_sc.inputs["Roughness"].default_value = 0.78
    n_sc.inputs["Distortion"].default_value = 0.42
    ng.links.new(coord_rand.outputs["Vector"], n_sc.inputs["Vector"])


    sep_uv = _new_node(ng, "ShaderNodeSeparateXYZ", (-1180, -140))
    ng.links.new(inp.outputs["UV"], sep_uv.inputs["Vector"])
    one_u = _new_node(ng, "ShaderNodeMath", (-960, -80)); one_u.operation = 'SUBTRACT'; one_u.inputs[0].default_value = 1.0
    ng.links.new(sep_uv.outputs["X"], one_u.inputs[1])
    min_u = _new_node(ng, "ShaderNodeMath", (-780, -80)); min_u.operation = 'MINIMUM'
    ng.links.new(sep_uv.outputs["X"], min_u.inputs[0]); ng.links.new(one_u.outputs[0], min_u.inputs[1])

    one_v = _new_node(ng, "ShaderNodeMath", (-960, -220)); one_v.operation = 'SUBTRACT'; one_v.inputs[0].default_value = 1.0
    ng.links.new(sep_uv.outputs["Y"], one_v.inputs[1])
    min_v = _new_node(ng, "ShaderNodeMath", (-780, -220)); min_v.operation = 'MINIMUM'
    ng.links.new(sep_uv.outputs["Y"], min_v.inputs[0]); ng.links.new(one_v.outputs[0], min_v.inputs[1])

    edge_d = _new_node(ng, "ShaderNodeMath", (-600, -150)); edge_d.operation = 'MINIMUM'
    ng.links.new(min_u.outputs[0], edge_d.inputs[0]); ng.links.new(min_v.outputs[0], edge_d.inputs[1])

    peel_zone = _new_node(ng, "ShaderNodeMapRange", (-420, -150))
    peel_zone.clamp = True; peel_zone.inputs[1].default_value = 0.0; peel_zone.inputs[2].default_value = 0.14
    peel_zone.inputs[3].default_value = 0.48; peel_zone.inputs[4].default_value = 0.0
    ng.links.new(edge_d.outputs[0], peel_zone.inputs[0])


    img_m_chip = bpy.data.images.get("IMG_Mask4K_PaintChips")
    img_m_schv = bpy.data.images.get("IMG_Mask4K_ScratchesHeavy")
    img_m_smg  = bpy.data.images.get("IMG_Mask4K_SmudgesStains")

    peel_base_sock = n_sc.outputs["Factor"]
    if img_m_chip is not None:
        mp_dc = _new_node(ng, "ShaderNodeMapping", (-1300, 80))
        mp_dc.inputs["Scale"].default_value = (2.8, 2.8, 2.8)
        mp_dc.inputs["Rotation"].default_value = (0.24, -0.35, 0.48)
        ng.links.new(coord_rand.outputs["Vector"], mp_dc.inputs["Vector"])
        t_dc = _new_node(ng, "ShaderNodeTexImage", (-1020, 80))
        t_dc.image = img_m_chip; t_dc.projection = 'BOX'; t_dc.projection_blend = 0.22
        ng.links.new(mp_dc.outputs["Vector"], t_dc.inputs["Vector"])
        dc_mix = _new_node(ng, "ShaderNodeMix", (-800, 100)); dc_mix.data_type = 'FLOAT'
        dc_mix.inputs[0].default_value = 0.45
        ng.links.new(n_sc.outputs["Factor"], dc_mix.inputs[2])
        ng.links.new(t_dc.outputs["Color"], dc_mix.inputs[3])
        peel_base_sock = dc_mix.outputs[0]

    peel_sum = _new_node(ng, "ShaderNodeMath", (-620, 80)); peel_sum.operation = 'ADD'
    ng.links.new(peel_base_sock, peel_sum.inputs[0])
    ng.links.new(peel_zone.outputs[0], peel_sum.inputs[1])

    sc_mask = _new_node(ng, "ShaderNodeMapRange", (-420, 120))
    sc_mask.clamp = True; sc_mask.interpolation_type = 'LINEAR'
    sc_mask.inputs[1].default_value = 0.69; sc_mask.inputs[2].default_value = 0.79
    sc_mask.inputs[3].default_value = 0.0; sc_mask.inputs[4].default_value = 1.0
    ng.links.new(peel_sum.outputs[0], sc_mask.inputs[0])


    lin_cut0 = _new_node(ng, "ShaderNodeMapRange", (-800, 340))
    lin_cut0.clamp = True; lin_cut0.interpolation_type = 'LINEAR'
    lin_cut0.inputs[1].default_value = 0.77; lin_cut0.inputs[2].default_value = 0.85
    lin_cut0.inputs[3].default_value = 0.0; lin_cut0.inputs[4].default_value = 1.0
    ng.links.new(n_lsc.outputs["Factor"], lin_cut0.inputs[0])

    lin_cut_sock = lin_cut0.outputs[0]
    if img_m_schv is not None:
        mp_ds = _new_node(ng, "ShaderNodeMapping", (-1300, 560))
        mp_ds.inputs["Scale"].default_value = (2.4, 2.4, 2.4)
        mp_ds.inputs["Rotation"].default_value = (-0.28, 0.41, -0.19)
        ng.links.new(coord_rand.outputs["Vector"], mp_ds.inputs["Vector"])
        t_ds = _new_node(ng, "ShaderNodeTexImage", (-1020, 560))
        t_ds.image = img_m_schv; t_ds.projection = 'BOX'; t_ds.projection_blend = 0.20
        ng.links.new(mp_ds.outputs["Vector"], t_ds.inputs["Vector"])
        ds_cut = _new_node(ng, "ShaderNodeMapRange", (-800, 540))
        ds_cut.clamp = True; ds_cut.inputs[1].default_value = 0.24; ds_cut.inputs[2].default_value = 0.68
        ds_cut.inputs[3].default_value = 0.0; ds_cut.inputs[4].default_value = 0.92
        ng.links.new(t_ds.outputs["Color"], ds_cut.inputs[0])
        ds_u = _new_node(ng, "ShaderNodeMath", (-660, 440)); ds_u.operation = 'MAXIMUM'; ds_u.use_clamp = True
        ng.links.new(lin_cut0.outputs[0], ds_u.inputs[0])
        ng.links.new(ds_cut.outputs[0], ds_u.inputs[1])
        lin_cut_sock = ds_u.outputs[0]

    bd_v = _new_node(ng, "ShaderNodeMath", (-560, 340)); bd_v.operation = 'MULTIPLY'; bd_v.use_clamp = True
    ng.links.new(lin_cut_sock, bd_v.inputs[0])
    ng.links.new(ctrl.outputs["SurfaceErosion"], bd_v.inputs[1])

    dmg_raw = _new_node(ng, "ShaderNodeMath", (-220, 200)); dmg_raw.operation = 'MAXIMUM'; dmg_raw.use_clamp = True
    ng.links.new(sc_mask.outputs[0], dmg_raw.inputs[0]); ng.links.new(bd_v.outputs[0], dmg_raw.inputs[1])


    w_eb = _new_node(ng, "ShaderNodeMath", (-820, -420)); w_eb.operation = 'MULTIPLY_ADD'
    w_eb.inputs[1].default_value = 0.35
    ng.links.new(ctrl.outputs["EdgeWear"], w_eb.inputs[0])
    w_bd = _new_node(ng, "ShaderNodeMath", (-1020, -480)); w_bd.operation = 'MULTIPLY'; w_bd.inputs[1].default_value = 0.85
    ng.links.new(ctrl.outputs["SurfaceErosion"], w_bd.inputs[0])
    ng.links.new(w_bd.outputs[0], w_eb.inputs[2])

    er_amp = _new_node(ng, "ShaderNodeMath", (-420, -380)); er_amp.operation = 'MULTIPLY'; er_amp.use_clamp = True
    ng.links.new(w_eb.outputs[0], er_amp.inputs[0])
    ng.links.new(inp.outputs["Erosion_Gain"], er_amp.inputs[1])

    er_eff = _new_node(ng, "ShaderNodeMath", (-20, 200)); er_eff.operation = 'MULTIPLY'; er_eff.use_clamp = True
    ng.links.new(dmg_raw.outputs[0], er_eff.inputs[0])
    ng.links.new(er_amp.outputs[0], er_eff.inputs[1])

    keep_a = _new_node(ng, "ShaderNodeMath", (160, 200)); keep_a.operation = 'SUBTRACT'; keep_a.use_clamp = True
    keep_a.inputs[0].default_value = 1.0
    ng.links.new(er_eff.outputs[0], keep_a.inputs[1])

    a0 = _new_node(ng, "ShaderNodeMath", (340, 200)); a0.operation = 'MULTIPLY'; a0.use_clamp = True
    ng.links.new(inp.outputs["Decal_Alpha"], a0.inputs[0])
    ng.links.new(keep_a.outputs[0], a0.inputs[1])


    vis_gate = _new_node(ng, "ShaderNodeMath", (160, -420)); vis_gate.operation = 'MULTIPLY'; vis_gate.use_clamp = True
    ng.links.new(ctrl.outputs["Show_PBR"], vis_gate.inputs[0])
    ng.links.new(ctrl.outputs["Show_Decals"], vis_gate.inputs[1])

    a_final = _new_node(ng, "ShaderNodeMath", (520, 180)); a_final.operation = 'MULTIPLY'; a_final.use_clamp = True
    ng.links.new(a0.outputs[0], a_final.inputs[0])
    ng.links.new(vis_gate.outputs[0], a_final.inputs[1])


    n_gr = _new_node(ng, "ShaderNodeTexNoise", (-820, -640))
    n_gr.inputs["Scale"].default_value = 8.5
    n_gr.inputs["Detail"].default_value = 6.0
    n_gr.inputs["Roughness"].default_value = 0.65
    ng.links.new(coord_rand.outputs["Vector"], n_gr.inputs["Vector"])

    gr_src_sock = n_gr.outputs["Factor"]
    if img_m_smg is not None:
        mp_dsm = _new_node(ng, "ShaderNodeMapping", (-1080, -820))
        mp_dsm.inputs["Scale"].default_value = (2.2, 2.2, 2.2)
        mp_dsm.inputs["Rotation"].default_value = (0.14, -0.32, 0.25)
        ng.links.new(coord_rand.outputs["Vector"], mp_dsm.inputs["Vector"])
        t_dsm = _new_node(ng, "ShaderNodeTexImage", (-840, -820))
        t_dsm.image = img_m_smg; t_dsm.projection = 'BOX'; t_dsm.projection_blend = 0.24
        ng.links.new(mp_dsm.outputs["Vector"], t_dsm.inputs["Vector"])
        gr_mx = _new_node(ng, "ShaderNodeMix", (-660, -720)); gr_mx.data_type = 'FLOAT'
        gr_mx.inputs[0].default_value = 0.48
        ng.links.new(n_gr.outputs["Factor"], gr_mx.inputs[2])
        ng.links.new(t_dsm.outputs["Color"], gr_mx.inputs[3])
        gr_src_sock = gr_mx.outputs[0]

    gr_w = _new_node(ng, "ShaderNodeMath", (-540, -640)); gr_w.operation = 'MULTIPLY'; gr_w.use_clamp = True
    ng.links.new(gr_src_sock, gr_w.inputs[0])
    ng.links.new(inp.outputs["Soot_Gain"], gr_w.inputs[1])

    gr_s = _new_node(ng, "ShaderNodeMath", (-360, -640)); gr_s.operation = 'MULTIPLY'; gr_s.use_clamp = True
    ng.links.new(gr_w.outputs[0], gr_s.inputs[0])
    ng.links.new(ctrl.outputs["Weathering"], gr_s.inputs[1])

    mix_col = _new_node(ng, "ShaderNodeMix", (140, 20))
    mix_col.data_type = 'RGBA'
    mix_col.inputs[7].default_value = rgba(0x1D1B18)
    ng.links.new(gr_s.outputs[0], mix_col.inputs[0])
    ng.links.new(inp.outputs["Decal_Color"], mix_col.inputs[6])


    h_dec_m = _new_node(ng, "ShaderNodeMath", (20, -260)); h_dec_m.operation = 'MULTIPLY'
    h_dec_m.inputs[1].default_value = 0.04 * (0.0003 / 0.0003)
    ng.links.new(n_sc.outputs["Factor"], h_dec_m.inputs[0])
    h_dec_all = _new_node(ng, "ShaderNodeMath", (160, -260)); h_dec_all.operation = 'MULTIPLY_ADD'
    h_dec_all.inputs[1].default_value = 0.075 * (0.00028 / 0.0003)
    ng.links.new(a0.outputs[0], h_dec_all.inputs[0])
    nt_l = ng.links
    nt_l.new(h_dec_m.outputs[0], h_dec_all.inputs[2])
    b_thk = _new_node(ng, "ShaderNodeBump", (310, -240))
    b_thk.inputs["Strength"].default_value = 1.0
    b_thk.inputs["Distance"].default_value = 0.00030
    nt_l.new(h_dec_all.outputs[0], b_thk.inputs["Height"])


    sep_hsv = _new_node(ng, "ShaderNodeSeparateColor", (-120, -480))
    sep_hsv.mode = 'HSV'
    nt_l.new(inp.outputs["Decal_Color"], sep_hsv.inputs["Color"])
    sat_low = _new_node(ng, "ShaderNodeMapRange", (60, -420)); sat_low.clamp = True
    sat_low.inputs[1].default_value = 0.08; sat_low.inputs[2].default_value = 0.22
    sat_low.inputs[3].default_value = 1.0; sat_low.inputs[4].default_value = 0.0
    nt_l.new(sep_hsv.outputs["Green"], sat_low.inputs[0])
    val_mid = _new_node(ng, "ShaderNodeMapRange", (60, -540)); val_mid.clamp = True
    val_mid.inputs[1].default_value = 0.34; val_mid.inputs[2].default_value = 0.56
    val_mid.inputs[3].default_value = 0.0; val_mid.inputs[4].default_value = 1.0
    nt_l.new(sep_hsv.outputs["Blue"], val_mid.inputs[0])
    val_hi = _new_node(ng, "ShaderNodeMapRange", (60, -660)); val_hi.clamp = True
    val_hi.inputs[1].default_value = 0.68; val_hi.inputs[2].default_value = 0.78
    val_hi.inputs[3].default_value = 1.0; val_hi.inputs[4].default_value = 0.0
    nt_l.new(sep_hsv.outputs["Blue"], val_hi.inputs[0])
    val_band = _new_node(ng, "ShaderNodeMath", (180, -580)); val_band.operation = 'MULTIPLY'; val_band.use_clamp = True
    nt_l.new(val_mid.outputs[0], val_band.inputs[0])
    nt_l.new(val_hi.outputs[0], val_band.inputs[1])
    bare_m = _new_node(ng, "ShaderNodeMath", (280, -500)); bare_m.operation = 'MULTIPLY'; bare_m.use_clamp = True
    nt_l.new(sat_low.outputs[0], bare_m.inputs[0])
    nt_l.new(val_band.outputs[0], bare_m.inputs[1])
    met_eff = _new_node(ng, "ShaderNodeMath", (400, -500)); met_eff.operation = 'MULTIPLY'; met_eff.use_clamp = True
    nt_l.new(inp.outputs["Metallic"], met_eff.inputs[0])
    nt_l.new(bare_m.outputs[0], met_eff.inputs[1])


    r_var = _new_node(ng, "ShaderNodeMapRange", (140, -100))
    r_var.clamp = True; r_var.inputs[1].default_value = 0.25; r_var.inputs[2].default_value = 0.75
    r_var.inputs[3].default_value = -0.075; r_var.inputs[4].default_value = +0.105
    nt_l.new(gr_src_sock, r_var.inputs[0])
    r_dyn = _new_node(ng, "ShaderNodeMath", (310, -100)); r_dyn.operation = 'ADD'; r_dyn.use_clamp = True
    nt_l.new(inp.outputs["Roughness"], r_dyn.inputs[0])
    nt_l.new(r_var.outputs[0], r_dyn.inputs[1])


    bsdf = _new_node(ng, "ShaderNodeBsdfPrincipled", (480, -40))
    nt_l.new(mix_col.outputs[2], bsdf.inputs["Base Color"])
    nt_l.new(met_eff.outputs[0], bsdf.inputs["Metallic"])
    nt_l.new(r_dyn.outputs[0], bsdf.inputs["Roughness"])
    nt_l.new(inp.outputs["Coat_Weight"], bsdf.inputs["Coat Weight"])
    nt_l.new(inp.outputs["Coat_Rough"], bsdf.inputs["Coat Roughness"])
    if "Coat Normal" in bsdf.inputs:
        nt_l.new(b_thk.outputs["Normal"], bsdf.inputs["Coat Normal"])
    nt_l.new(b_thk.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Emission Strength"].default_value = 0.0

    trans = _new_node(ng, "ShaderNodeBsdfTransparent", (480, 320))
    mix_sh = _new_node(ng, "ShaderNodeMixShader", (740, 80))
    nt_l.new(a_final.outputs[0], mix_sh.inputs["Fac"])
    nt_l.new(trans.outputs["BSDF"], mix_sh.inputs[1])
    nt_l.new(bsdf.outputs["BSDF"], mix_sh.inputs[2])
    nt_l.new(mix_sh.outputs["Shader"], out.inputs["Shader"])

    return ng


def _get_or_build_decal_material(
    mat_name: str, png_path: str, ng_decal,
    rough=0.58, coat=0.05, coat_r=0.35, metallic=0.0, erosion=0.55, soot=0.36, reflective=0.0,
):
    img_name = "IMG_" + os.path.splitext(os.path.basename(png_path))[0]
    img = bpy.data.images.get(img_name)
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.load(png_path, check_existing=False)
    img.name = img_name
    try:
        img.pack()
    except Exception:
        pass

    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    nt = mat.node_tree
    nt.nodes.clear()

    uv = _new_node(nt, "ShaderNodeUVMap", (-680, 0))
    tex = _new_node(nt, "ShaderNodeTexImage", (-420, 80))
    tex.image = img
    tex.extension = 'CLIP'
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])

    grp = _new_node(nt, "ShaderNodeGroup", (20, 40))
    grp.node_tree = ng_decal
    nt.links.new(tex.outputs["Color"], grp.inputs["Decal_Color"])
    nt.links.new(tex.outputs["Alpha"], grp.inputs["Decal_Alpha"])
    nt.links.new(uv.outputs["UV"], grp.inputs["UV"])
    grp.inputs["Roughness"].default_value = rough
    grp.inputs["Coat_Weight"].default_value = coat
    grp.inputs["Coat_Rough"].default_value = coat_r
    grp.inputs["Metallic"].default_value = metallic
    grp.inputs["Erosion_Gain"].default_value = erosion
    grp.inputs["Soot_Gain"].default_value = soot
    grp.inputs["Reflective"].default_value = reflective

    out = _new_node(nt, "ShaderNodeOutputMaterial", (340, 40))
    nt.links.new(grp.outputs["Shader"], out.inputs["Surface"])

    mat.diffuse_color = (0.85, 0.85, 0.85, 0.0)
    return mat


def _load_packed_decal_img(png_path: str, img_cache: dict = None, non_color: bool = False):
    img_name = "IMG_" + os.path.splitext(os.path.basename(png_path))[0]

    if img_cache is not None and img_name in img_cache:
        img = img_cache[img_name]
        if non_color:
            try:
                img.colorspace_settings.name = 'Non-Color'
            except Exception:
                pass
        return img
    img = bpy.data.images.get(img_name)
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.load(png_path, check_existing=False)
    img.name = img_name
    if non_color:
        try:
            img.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass
    try:
        img.pack()
    except Exception:
        pass
    if img_cache is not None:
        img_cache[img_name] = img
    return img


def _build_wx_scratch_decal_mat(
    mat_name: str, png_path: str, ng_global,
    bare_hex: int = 0xDDE3EA, gain: float = 0.80, img_cache: dict = None,
):

    img = _load_packed_decal_img(png_path, img_cache=img_cache, non_color=True)
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    nt = mat.node_tree
    nt.nodes.clear()

    uv = _new_node(nt, "ShaderNodeUVMap", (-920, 120))
    tex = _new_node(nt, "ShaderNodeTexImage", (-680, 160))
    tex.image = img
    tex.extension = 'CLIP'
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])

    ctrl = _new_node(nt, "ShaderNodeGroup", (-920, -240))
    ctrl.node_tree = ng_global


    a_cl = _new_node(nt, "ShaderNodeMapRange", (-380, 120)); a_cl.clamp = True
    a_cl.inputs[1].default_value = 0.06; a_cl.inputs[2].default_value = 0.72
    a_cl.inputs[3].default_value = 0.00; a_cl.inputs[4].default_value = 1.00
    nt.links.new(tex.outputs["Alpha"], a_cl.inputs[0])

    bd_s = _new_node(nt, "ShaderNodeMath", (-620, -240)); bd_s.operation = 'MULTIPLY_ADD'; bd_s.use_clamp = True
    bd_s.inputs[1].default_value = 0.58; bd_s.inputs[2].default_value = 0.50
    nt.links.new(ctrl.outputs["SurfaceErosion"], bd_s.inputs[0])

    vis0 = _new_node(nt, "ShaderNodeMath", (-420, -160)); vis0.operation = 'MULTIPLY'; vis0.use_clamp = True
    nt.links.new(ctrl.outputs["Show_PBR"], vis0.inputs[0])
    nt.links.new(ctrl.outputs["Show_Decals"], vis0.inputs[1])

    vis1 = _new_node(nt, "ShaderNodeMath", (-240, -160)); vis1.operation = 'MULTIPLY'; vis1.use_clamp = True
    nt.links.new(vis0.outputs[0], vis1.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], vis1.inputs[1])

    vis2 = _new_node(nt, "ShaderNodeMath", (-80, -160)); vis2.operation = 'MULTIPLY'; vis2.use_clamp = True
    nt.links.new(vis1.outputs[0], vis2.inputs[0])
    nt.links.new(bd_s.outputs[0], vis2.inputs[1])

    a_g = _new_node(nt, "ShaderNodeMath", (-160, 120)); a_g.operation = 'MULTIPLY'; a_g.use_clamp = True
    a_g.inputs[1].default_value = gain
    nt.links.new(a_cl.outputs[0], a_g.inputs[0])

    a_fin = _new_node(nt, "ShaderNodeMath", (100, 80)); a_fin.operation = 'MULTIPLY'; a_fin.use_clamp = True
    nt.links.new(a_g.outputs[0], a_fin.inputs[0])
    nt.links.new(vis2.outputs[0], a_fin.inputs[1])


    h_inv = _new_node(nt, "ShaderNodeMath", (-160, -20)); h_inv.operation = 'SUBTRACT'; h_inv.use_clamp = True
    h_inv.inputs[0].default_value = 1.0
    nt.links.new(a_cl.outputs[0], h_inv.inputs[1])
    b_sc = _new_node(nt, "ShaderNodeBump", (80, -140))
    b_sc.inputs["Strength"].default_value = 0.24
    b_sc.inputs["Distance"].default_value = 0.00030
    nt.links.new(h_inv.outputs[0], b_sc.inputs["Height"])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (300, -40))
    bsdf.inputs["Base Color"].default_value = rgba(bare_hex)
    bsdf.inputs["Metallic"].default_value = 0.96
    bsdf.inputs["Roughness"].default_value = 0.16
    bsdf.inputs["Specular IOR Level"].default_value = 0.56
    nt.links.new(b_sc.outputs["Normal"], bsdf.inputs["Normal"])

    trans = _new_node(nt, "ShaderNodeBsdfTransparent", (300, 180))
    mx = _new_node(nt, "ShaderNodeMixShader", (620, 60))
    nt.links.new(a_fin.outputs[0], mx.inputs["Fac"])
    nt.links.new(trans.outputs["BSDF"], mx.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], mx.inputs[2])
    out = _new_node(nt, "ShaderNodeOutputMaterial", (860, 60))
    nt.links.new(mx.outputs["Shader"], out.inputs["Surface"])
    mat.diffuse_color = (0.85, 0.85, 0.85, 0.0)
    return mat


def _build_wx_leak_decal_mat(mat_name: str, png_path: str, ng_global, gain: float = 0.76, img_cache: dict = None):

    img = _load_packed_decal_img(png_path, img_cache=img_cache)
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    nt = mat.node_tree
    nt.nodes.clear()

    uv = _new_node(nt, "ShaderNodeUVMap", (-920, 120))
    tex = _new_node(nt, "ShaderNodeTexImage", (-680, 160))
    tex.image = img
    tex.extension = 'CLIP'
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])

    ctrl = _new_node(nt, "ShaderNodeGroup", (-920, -240))
    ctrl.node_tree = ng_global

    bd_s = _new_node(nt, "ShaderNodeMath", (-620, -240)); bd_s.operation = 'MULTIPLY_ADD'; bd_s.use_clamp = True
    bd_s.inputs[1].default_value = 0.55; bd_s.inputs[2].default_value = 0.48
    nt.links.new(ctrl.outputs["SurfaceErosion"], bd_s.inputs[0])

    vis0 = _new_node(nt, "ShaderNodeMath", (-420, -160)); vis0.operation = 'MULTIPLY'; vis0.use_clamp = True
    nt.links.new(ctrl.outputs["Show_PBR"], vis0.inputs[0])
    nt.links.new(ctrl.outputs["Show_Decals"], vis0.inputs[1])

    vis1 = _new_node(nt, "ShaderNodeMath", (-240, -160)); vis1.operation = 'MULTIPLY'; vis1.use_clamp = True
    nt.links.new(vis0.outputs[0], vis1.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], vis1.inputs[1])

    vis2 = _new_node(nt, "ShaderNodeMath", (-80, -160)); vis2.operation = 'MULTIPLY'; vis2.use_clamp = True
    nt.links.new(vis1.outputs[0], vis2.inputs[0])
    nt.links.new(bd_s.outputs[0], vis2.inputs[1])

    a_g = _new_node(nt, "ShaderNodeMath", (-240, 120)); a_g.operation = 'MULTIPLY'; a_g.use_clamp = True
    a_g.inputs[1].default_value = gain
    nt.links.new(tex.outputs["Alpha"], a_g.inputs[0])

    a_fin = _new_node(nt, "ShaderNodeMath", (100, 100)); a_fin.operation = 'MULTIPLY'; a_fin.use_clamp = True
    nt.links.new(a_g.outputs[0], a_fin.inputs[0])
    nt.links.new(vis2.outputs[0], a_fin.inputs[1])


    wet_m = _new_node(nt, "ShaderNodeMapRange", (-240, -20)); wet_m.clamp = True
    wet_m.inputs[1].default_value = 0.35; wet_m.inputs[2].default_value = 0.85
    wet_m.inputs[3].default_value = 0.82; wet_m.inputs[4].default_value = 0.09
    nt.links.new(tex.outputs["Alpha"], wet_m.inputs[0])

    coat_m = _new_node(nt, "ShaderNodeMapRange", (-240, -180)); coat_m.clamp = True
    coat_m.inputs[1].default_value = 0.40; coat_m.inputs[2].default_value = 0.85
    coat_m.inputs[3].default_value = 0.05; coat_m.inputs[4].default_value = 0.84
    nt.links.new(tex.outputs["Alpha"], coat_m.inputs[0])

    b_lk = _new_node(nt, "ShaderNodeBump", (80, -200))
    b_lk.inputs["Strength"].default_value = 0.08
    b_lk.inputs["Distance"].default_value = 0.0003
    nt.links.new(tex.outputs["Alpha"], b_lk.inputs["Height"])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (300, -40))
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    nt.links.new(wet_m.outputs[0], bsdf.inputs["Roughness"])
    nt.links.new(coat_m.outputs[0], bsdf.inputs["Coat Weight"])
    bsdf.inputs["Coat Roughness"].default_value = 0.06
    nt.links.new(b_lk.outputs["Normal"], bsdf.inputs["Normal"])
    if "Coat Normal" in bsdf.inputs:
        nt.links.new(b_lk.outputs["Normal"], bsdf.inputs["Coat Normal"])

    trans = _new_node(nt, "ShaderNodeBsdfTransparent", (300, 180))
    mx = _new_node(nt, "ShaderNodeMixShader", (620, 60))
    nt.links.new(a_fin.outputs[0], mx.inputs["Fac"])
    nt.links.new(trans.outputs["BSDF"], mx.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], mx.inputs[2])
    out = _new_node(nt, "ShaderNodeOutputMaterial", (860, 60))
    nt.links.new(mx.outputs["Shader"], out.inputs["Surface"])
    mat.diffuse_color = (0.12, 0.10, 0.08, 0.0)
    return mat


def _build_wx_rust_decal_mat(mat_name: str, png_path: str, ng_global, img_cache: dict = None):

    img = _load_packed_decal_img(png_path, img_cache=img_cache)
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    nt = mat.node_tree
    nt.nodes.clear()

    uv = _new_node(nt, "ShaderNodeUVMap", (-920, 120))
    tex = _new_node(nt, "ShaderNodeTexImage", (-680, 160))
    tex.image = img
    tex.extension = 'CLIP'
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])

    ctrl = _new_node(nt, "ShaderNodeGroup", (-920, -240))
    ctrl.node_tree = ng_global


    bd_r = _new_node(nt, "ShaderNodeMapRange", (-620, -220)); bd_r.clamp = True
    bd_r.inputs[1].default_value = 0.30; bd_r.inputs[2].default_value = 1.45
    bd_r.inputs[3].default_value = 0.14; bd_r.inputs[4].default_value = 0.96
    nt.links.new(ctrl.outputs["SurfaceErosion"], bd_r.inputs[0])

    vis0 = _new_node(nt, "ShaderNodeMath", (-420, -140)); vis0.operation = 'MULTIPLY'; vis0.use_clamp = True
    nt.links.new(ctrl.outputs["Show_PBR"], vis0.inputs[0])
    nt.links.new(ctrl.outputs["Show_Decals"], vis0.inputs[1])

    vis1 = _new_node(nt, "ShaderNodeMath", (-240, -140)); vis1.operation = 'MULTIPLY'; vis1.use_clamp = True
    nt.links.new(vis0.outputs[0], vis1.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], vis1.inputs[1])

    vis2 = _new_node(nt, "ShaderNodeMath", (-60, -140)); vis2.operation = 'MULTIPLY'; vis2.use_clamp = True
    nt.links.new(vis1.outputs[0], vis2.inputs[0])
    nt.links.new(bd_r.outputs[0], vis2.inputs[1])

    a_fin = _new_node(nt, "ShaderNodeMath", (120, 100)); a_fin.operation = 'MULTIPLY'; a_fin.use_clamp = True
    nt.links.new(tex.outputs["Alpha"], a_fin.inputs[0])
    nt.links.new(vis2.outputs[0], a_fin.inputs[1])

    b_rs = _new_node(nt, "ShaderNodeBump", (80, -160))
    b_rs.inputs["Strength"].default_value = 0.34
    b_rs.inputs["Distance"].default_value = 0.0007
    nt.links.new(tex.outputs["Alpha"], b_rs.inputs["Height"])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (300, -40))
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.91
    bsdf.inputs["Specular IOR Level"].default_value = 0.30
    nt.links.new(b_rs.outputs["Normal"], bsdf.inputs["Normal"])

    trans = _new_node(nt, "ShaderNodeBsdfTransparent", (300, 180))
    mx = _new_node(nt, "ShaderNodeMixShader", (620, 60))
    nt.links.new(a_fin.outputs[0], mx.inputs["Fac"])
    nt.links.new(trans.outputs["BSDF"], mx.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], mx.inputs[2])
    out = _new_node(nt, "ShaderNodeOutputMaterial", (860, 60))
    nt.links.new(mx.outputs["Shader"], out.inputs["Surface"])
    mat.diffuse_color = (0.36, 0.14, 0.07, 0.0)
    return mat


def _build_wx_grease_smear_mat(mat_name: str, png_path: str, ng_global, gain: float = 0.26, img_cache: dict = None):

    img = _load_packed_decal_img(png_path, img_cache=img_cache)
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    nt = mat.node_tree
    nt.nodes.clear()

    uv = _new_node(nt, "ShaderNodeUVMap", (-920, 120))
    tex = _new_node(nt, "ShaderNodeTexImage", (-680, 160))
    tex.image = img
    tex.extension = 'CLIP'
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])

    ctrl = _new_node(nt, "ShaderNodeGroup", (-920, -240))
    ctrl.node_tree = ng_global

    vis0 = _new_node(nt, "ShaderNodeMath", (-420, -160)); vis0.operation = 'MULTIPLY'; vis0.use_clamp = True
    nt.links.new(ctrl.outputs["Show_PBR"], vis0.inputs[0])
    nt.links.new(ctrl.outputs["Show_Decals"], vis0.inputs[1])

    vis1 = _new_node(nt, "ShaderNodeMath", (-220, -160)); vis1.operation = 'MULTIPLY'; vis1.use_clamp = True
    nt.links.new(vis0.outputs[0], vis1.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], vis1.inputs[1])

    a_g = _new_node(nt, "ShaderNodeMath", (-220, 120)); a_g.operation = 'MULTIPLY'; a_g.use_clamp = True
    a_g.inputs[1].default_value = gain
    nt.links.new(tex.outputs["Alpha"], a_g.inputs[0])

    a_fin = _new_node(nt, "ShaderNodeMath", (100, 100)); a_fin.operation = 'MULTIPLY'; a_fin.use_clamp = True
    nt.links.new(a_g.outputs[0], a_fin.inputs[0])
    nt.links.new(vis1.outputs[0], a_fin.inputs[1])

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (300, -40))
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.10
    bsdf.inputs["Coat Weight"].default_value = 0.86
    bsdf.inputs["Coat Roughness"].default_value = 0.055
    if "Coat Normal" in bsdf.inputs:
        geo_gs = _new_node(nt, "ShaderNodeNewGeometry", (100, -240))
        nt.links.new(geo_gs.outputs["Normal"], bsdf.inputs["Coat Normal"])

    trans = _new_node(nt, "ShaderNodeBsdfTransparent", (300, 180))
    mx = _new_node(nt, "ShaderNodeMixShader", (620, 60))
    nt.links.new(a_fin.outputs[0], mx.inputs["Fac"])
    nt.links.new(trans.outputs["BSDF"], mx.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], mx.inputs[2])
    out = _new_node(nt, "ShaderNodeOutputMaterial", (860, 60))
    nt.links.new(mx.outputs["Shader"], out.inputs["Surface"])
    mat.diffuse_color = (0.14, 0.12, 0.10, 0.0)
    return mat


def _emit_snapped_quad_decal(
    bm, uv_layer, host_obj, mat_idx: int,
    eval_pt_fn, eval_dir_fn, nu: int = 8, nv: int = 4, offset_m: float = 0.00055,
    snap_to_host: bool = True, max_snap_dev: float = 0.015,
    uv_rect: tuple = (0.0, 0.0, 1.0, 1.0), use_raycast: bool = False,
):

    ru0, rv0, ru1, rv1 = uv_rect
    grid = []
    for iu in range(nu + 1):
        u = iu / float(nu)
        col = []
        for iv in range(nv + 1):
            v = iv / float(nv)
            p_nom = Vector(eval_pt_fn(u, v))
            n_nom = Vector(eval_dir_fn(u, v)).normalized()
            if snap_to_host and host_obj is not None:
                if use_raycast:
                    ok, loc, nrm, _ = host_obj.ray_cast(p_nom + n_nom * 0.045, -n_nom, distance=0.045 + max_snap_dev)
                else:
                    ok, loc, nrm, _ = host_obj.closest_point_on_mesh(p_nom)
                if ok and (loc - p_nom).length <= max_snap_dev:
                    if nrm.dot(n_nom) < 0.0:
                        nrm = -nrm
                    p_final = loc + (n_nom if use_raycast else nrm.normalized()) * offset_m
                else:
                    p_final = p_nom + n_nom * offset_m
            else:
                p_final = p_nom + n_nom * offset_m
            col.append(bm.verts.new(p_final))
        grid.append(col)

    for iu in range(nu):
        u0, u1 = iu / float(nu), (iu + 1) / float(nu)
        uu0 = ru0 + (ru1 - ru0) * u0
        uu1 = ru0 + (ru1 - ru0) * u1
        for iv in range(nv):
            v0, v1 = iv / float(nv), (iv + 1) / float(nv)
            vv0 = rv0 + (rv1 - rv0) * v0
            vv1 = rv0 + (rv1 - rv0) * v1
            verts = (grid[iu][iv], grid[iu + 1][iv], grid[iu + 1][iv + 1], grid[iu][iv + 1])
            try:
                f = bm.faces.new(verts)
                f.material_index = mat_idx
                f.smooth = True

                n_ref = Vector(eval_dir_fn(0.5 * (u0 + u1), 0.5 * (v0 + v1)))
                f.normal_update()
                if f.normal.dot(n_ref) < 0.0:
                    f.normal_flip()

                uvs_by_v = {
                    verts[0]: (uu0, vv0),
                    verts[1]: (uu1, vv0),
                    verts[2]: (uu1, vv1),
                    verts[3]: (uu0, vv1),
                }
                for lp in f.loops:
                    lp[uv_layer].uv = uvs_by_v[lp.vert]
            except ValueError:
                pass


def _finalize_decal_mesh_obj(name: str, bm, mat_list: list, coll):
    old = bpy.data.objects.get(name)
    if old is not None:
        me_old = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if me_old and me_old.users == 0:
            bpy.data.meshes.remove(me_old)

    bm_apply_hard_surface_smoothing(bm, angle_deg=42.0)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mat_list:
        me.materials.append(m)
    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    attach_weighted_normal_modifier(obj)

    if hasattr(obj, "visible_shadow"):
        obj.visible_shadow = False
    return obj


def build_conformal_aviation_decals(ng_global, decals_dir: str = None):

    if decals_dir is None:
        decals_dir = os.path.join(_THIS_DIR, "assets_decals")
    if not os.path.exists(decals_dir) or len(os.listdir(decals_dir)) < 10:
        print(f"[Warning] Decal directory not found: {decals_dir}; skipping decals")
        return

    ng_dec = _build_ng_decal_weathering(ng_global)

    def _dmat(key, fn, **kw):
        return _get_or_build_decal_material(f"Mat_Decal_{key}", os.path.join(decals_dir, fn), ng_dec, **kw)


    m_tk9    = _dmat("TK9_Blue",        "decal_tk9_rail_blue.png",      rough=0.64, coat=0.04, erosion=0.60, soot=0.40)
    m_refl   = _dmat("Refl_RedWhite",   "decal_reflector_redwhite.png", rough=0.16, coat=0.92, coat_r=0.06, erosion=0.45, soot=0.30, reflective=0.0)
    m_fork   = _dmat("Forklift_Chev",   "decal_forklift_chevron.png",   rough=0.44, coat=0.22, erosion=0.85, soot=0.52)
    m_m4     = _dmat("M4_Arch_Placard", "decal_m4_arch_placard.png",    rough=0.38, coat=0.20, metallic=0.88, erosion=0.50, soot=0.35)
    m_sblast = _dmat("Shroud_Blast",    "decal_shroud_danger_blast.png",rough=0.66, coat=0.02, erosion=0.72, soot=0.48)
    m_shot   = _dmat("Shroud_Hot",      "decal_shroud_hot_station.png", rough=0.44, coat=0.14, metallic=0.0,  erosion=0.70, soot=0.48)
    m_sband  = _dmat("Shroud_Belly",    "decal_belly_band_latch.png",   rough=0.64, coat=0.02, erosion=0.78, soot=0.42)
    m_intake = _dmat("Intake_Danger",   "decal_intake_danger.png",      rough=0.52, coat=0.10, erosion=0.58, soot=0.32)
    m_tec    = _dmat("TEC_FireZone",    "decal_tec_firezone.png",       rough=0.44, coat=0.14, metallic=0.0,  erosion=0.68, soot=0.48)
    m_oil    = _dmat("OilTank_Spec",    "decal_oiltank_spec.png",       rough=0.56, coat=0.08, erosion=0.62, soot=0.46)
    m_trspec = _dmat("Trailer_Spec",    "decal_trailer_usaf_spec.png",  rough=0.62, coat=0.04, erosion=0.68, soot=0.44)
    m_deec   = _dmat("DEEC_Caution",    "decal_deec_caution.png",       rough=0.46, coat=0.08, metallic=0.0,  erosion=0.48, soot=0.34)

    m_hyd     = _dmat("Micro_HydSpec",   "decal_micro_hyd_spec.png",     rough=0.44, coat=0.14, metallic=0.0,  erosion=0.52, soot=0.36)
    m_pmel    = _dmat("Micro_PMEL_Cal",  "decal_micro_pmel_cal.png",     rough=0.52, coat=0.06, metallic=0.0,  erosion=0.42, soot=0.28)
    m_nostep  = _dmat("Micro_NoStep",    "decal_micro_nostep_hatch.png", rough=0.64, coat=0.02, erosion=0.72, soot=0.46)
    m_hoist   = _dmat("M4_HoistBeam",    "decal_m4_hoist_beam.png",      rough=0.38, coat=0.18, metallic=0.88, erosion=0.55, soot=0.36)
    m_fwdbmp  = _dmat("Stand_FwdBumper", "decal_stand_fwd_bumper.png",   rough=0.58, coat=0.06, erosion=0.64, soot=0.42)
    m_aftidx  = _dmat("Shroud_AftIndex", "decal_shroud_aft_index.png",   rough=0.64, coat=0.02, erosion=0.74, soot=0.50)
    m_spinmk  = _dmat("Spinner_BalMark", "decal_spinner_balance_mark.png",rough=0.52, coat=0.08, erosion=0.56, soot=0.32)

    m_snopush = _dmat("Shroud_NoPush",     "decal_shroud_nopush_plate.png", rough=0.66, coat=0.00, erosion=0.45, soot=0.28)
    m_actband = _dmat("Actuator_DualBand", "decal_actuator_dual_band.png",  rough=0.80, coat=0.00, erosion=0.05, soot=0.05)

    m_noztag   = _dmat("Nozzle_PN_NoStep", "decal_nozzle_pn407808_nostep.png", rough=0.62, coat=0.00, erosion=0.22, soot=0.32)
    m_flbands  = _dmat("FluidLine_Bands",  "decal_fluid_line_bands.png",       rough=0.26, coat=0.48, coat_r=0.16, erosion=0.20, soot=0.22)
    m_oilsight = _dmat("OilSight_Grad",    "decal_oilsight_graduations.png",   rough=0.14, coat=0.60, coat_r=0.06, erosion=0.03, soot=0.04)

    m_m4_rl    = _dmat("M4_RollLock",        "decal_m4_roll_lock.png",           rough=0.46, coat=0.14, erosion=0.22, soot=0.26)
    m_tr_dp    = _dmat("Trailer_DataPlate",  "decal_trailer_dataplate_dual.png", rough=0.48, coat=0.14, erosion=0.26, soot=0.30)
    m_st_atlas = _dmat("Stand_Plates_Atlas", "decal_stand_plates_4k_atlas.png",  rough=0.46, coat=0.14, erosion=0.22, soot=0.24)
    m_foddoc   = _dmat("FODBox_TechDoc",     "decal_fodbox_techdoc_atlas.png",   rough=0.44, coat=0.18, erosion=0.22, soot=0.24)

    m_pmark    = _dmat("PaintMarker_NDT",    "decal_paintmarker_ndt_atlas.png",  rough=0.68, coat=0.04, erosion=0.22, soot=0.30)
    m_petalnum = _dmat("Nozzle_PetalNums",   "decal_petal_numbers_atlas.png",    rough=0.64, coat=0.00, erosion=0.26, soot=0.38)
    m_tireslip = _dmat("Tire_SlippageMark",  "decal_tire_slippage_mark.png",     rough=0.72, coat=0.02, erosion=0.28, soot=0.32)

    wx_img_cache = {}
    m_wx_sc_al = _build_wx_scratch_decal_mat("Mat_Decal_WX_Scratches_Alloy",  os.path.join(decals_dir, "decal_wx_scratches_hero.png"), ng_global, bare_hex=0xA4ACB6, gain=0.42, img_cache=wx_img_cache)
    m_wx_sc_ti = _build_wx_scratch_decal_mat("Mat_Decal_WX_Scratches_GoldTi", os.path.join(decals_dir, "decal_wx_scratches_hero.png"), ng_global, bare_hex=0xD4C498, gain=0.52, img_cache=wx_img_cache)
    m_wx_leak  = _build_wx_leak_decal_mat   ("Mat_Decal_WX_LeakPlume",        os.path.join(decals_dir, "decal_wx_leak_plume.png"),     ng_global, gain=0.46, img_cache=wx_img_cache)
    m_wx_rust  = _build_wx_rust_decal_mat   ("Mat_Decal_WX_RustStrip",        os.path.join(decals_dir, "decal_wx_rust_strip.png"),     ng_global, img_cache=wx_img_cache)
    m_wx_hand  = _build_wx_grease_smear_mat ("Mat_Decal_WX_GreaseSmear",      os.path.join(decals_dir, "decal_wx_grease_smear.png"),   ng_global, gain=0.14, img_cache=wx_img_cache)


    coll_name = "07_Aviation_Decals_And_Markings"
    coll_dec = bpy.data.collections.get(coll_name)
    if coll_dec is None:
        coll_dec = bpy.data.collections.new(coll_name)
        bpy.context.scene.collection.children.link(coll_dec)


    def _r_sh_out(yv: float) -> float:
        if yv <= -0.415:
            t_f = max(0.0, min(1.0, (-0.415 - yv) / 0.305))
            return 0.614 + 0.032 * (1.0 - (t_f ** 1.80))
        t_a = max(0.0, min(1.28, (yv - (-0.415)) / 0.360))
        return 0.646 - 0.0225 * (t_a ** 1.55)

    def _cyl_patch(bm, uvl, host, midx, phi_c_deg, dphi_deg, y0, y1, r_fn, nu=12, nv=4, offset=0.00055, max_snap=0.012, snap_to_host=True):

        is_port = (0.0 < (phi_c_deg % 360.0) < 180.0)
        def _pt(u, v):

            if is_port:
                yv = y0 + (y1 - y0) * u
                ph = math.radians(phi_c_deg + dphi_deg * (0.5 - v))
            else:
                yv = y1 - (y1 - y0) * u
                ph = math.radians(phi_c_deg - dphi_deg * (0.5 - v))
            rv = r_fn(yv)
            return (rv * math.sin(ph), yv, rv * math.cos(ph))
        def _nr(u, v):
            if is_port:
                ph = math.radians(phi_c_deg + dphi_deg * (0.5 - v))
            else:
                ph = math.radians(phi_c_deg - dphi_deg * (0.5 - v))
            return (math.sin(ph), 0.0, math.cos(ph))
        _emit_snapped_quad_decal(bm, uvl, host, midx, _pt, _nr, nu=nu, nv=nv, offset_m=offset, snap_to_host=snap_to_host, max_snap_dev=max_snap)

    def _cyl_leak_aft(bm, uvl, host, midx, phi_c_deg, dphi_deg, y_orig, y_end, r_fn, nu=6, nv=12, offset=0.00042):

        def _pt(u, v):
            yv = y_end + (y_orig - y_end) * v
            ph = math.radians(phi_c_deg + dphi_deg * (u - 0.5))
            rv = r_fn(yv)
            return (rv * math.sin(ph), yv, rv * math.cos(ph))
        def _nr(u, v):
            ph = math.radians(phi_c_deg + dphi_deg * (u - 0.5))
            return (math.sin(ph), 0.0, math.cos(ph))
        _emit_snapped_quad_decal(bm, uvl, host, midx, _pt, _nr, nu=nu, nv=nv, offset_m=offset, snap_to_host=True, max_snap_dev=0.010)


    o_sh = bpy.data.objects.get("Static_Nozzle_Outer_Shroud")
    bm_sh = bmesh.new()
    uv_sh = bm_sh.loops.layers.uv.new("UVMap")

    for phi_d in (67.5, 112.5, 247.5, 292.5):
        _cyl_patch(bm_sh, uv_sh, o_sh, 0, phi_d, 6.33, -0.662, -0.506, _r_sh_out, nu=12, nv=6, offset=0.00062)

    for phi_d in (90.0, 270.0):
        _cyl_patch(bm_sh, uv_sh, o_sh, 1, phi_d, 3.13, -0.334, -0.238, _r_sh_out, nu=10, nv=5, offset=0.00062)

    for phi_d in (45.0, 135.0, 225.0, 315.0):
        _cyl_patch(bm_sh, uv_sh, o_sh, 2, phi_d, 2.55, -0.468, -0.372, lambda yv: _r_sh_out(yv) + 0.0052, nu=10, nv=4, offset=0.00045, snap_to_host=False)

    for phi_d in (22.5, 157.5, 202.5, 337.5):
        _cyl_patch(bm_sh, uv_sh, o_sh, 3, phi_d, 5.20, -0.660, -0.506, _r_sh_out, nu=12, nv=5, offset=0.00062)

    for phi_d, y0, y1 in (
        (78.0,  -0.340, -0.125), (102.0, -0.665, -0.495),
        (258.0, -0.340, -0.125), (282.0, -0.665, -0.495),
    ):
        _cyl_patch(bm_sh, uv_sh, o_sh, 4, phi_d, 10.5, y0, y1, _r_sh_out, nu=10, nv=6, offset=0.00042, max_snap=0.005)

    for phi_d in (77.0, 257.0):
        _cyl_patch(bm_sh, uv_sh, o_sh, 5, phi_d, 6.2, -0.472, -0.364, lambda yv: _r_sh_out(yv) + 0.0024, nu=8, nv=5, offset=0.00045, snap_to_host=False)

    for phi_d in (67.5, 112.5, 247.5, 292.5):
        _cyl_leak_aft(bm_sh, uv_sh, o_sh, 6, phi_d, 2.35, -0.264, -0.156, _r_sh_out, nu=6, nv=12, offset=0.00044)

    for phi_d in (45.0, 90.0, 135.0, 225.0, 270.0, 315.0):
        _cyl_patch(bm_sh, uv_sh, o_sh, 7, phi_d, 2.36, -0.218, -0.088, _r_sh_out, nu=10, nv=4, offset=0.00060)

    for phi_d in (76.0, 256.0):
        _cyl_patch(bm_sh, uv_sh, o_sh, 8, phi_d, 8.2, -0.228, -0.136, _r_sh_out, nu=8, nv=6, offset=0.00043, max_snap=0.005)

    for k_np in range(8):
        ph_npc = k_np * 45.0 + 22.5
        _cyl_patch(
            bm_sh, uv_sh, o_sh, 9, ph_npc, 6.20, -0.1835, -0.0945,
            lambda yv: _r_sh_out(yv) + 0.0006, nu=8, nv=6, offset=0.00042, snap_to_host=False,
        )
    _finalize_decal_mesh_obj(
        "Decals_Engine_Shroud", bm_sh,
        [m_sblast, m_shot, m_sband, m_nostep, m_wx_sc_ti, m_wx_sc_al, m_wx_leak, m_aftidx, m_wx_hand, m_snopush],
        coll_dec,
    )


    bm_co = bmesh.new()
    uv_co = bm_co.loops.layers.uv.new("UVMap")
    o_tec = bpy.data.objects.get("Static_Forward_TEC_And_Mount")
    o_oil = bpy.data.objects.get("Fwd_06_Right_Stamped_Saddle_OilTank")
    o_fan = bpy.data.objects.get("Fwd_02_Fan_Case_VaneLinkage_And_DEEC")
    o_spn = bpy.data.objects.get("Fwd_01_Inlet_Spinner_Pitot_And_Sensors")


    _cyl_patch(bm_co, uv_co, o_tec, 0, 70.0,  7.40, -2.158, -2.002, lambda yv: 0.5437, nu=10, nv=6, offset=0.00062)
    _cyl_patch(bm_co, uv_co, o_tec, 0, 246.0, 7.40, -2.158, -2.002, lambda yv: 0.5437, nu=10, nv=6, offset=0.00062)


    _cyl_patch(bm_co, uv_co, o_oil, 1, 250.8, 5.00, -4.110, -4.017, lambda yv: 0.554, nu=10, nv=5, offset=0.00062)


    _cyl_patch(bm_co, uv_co, o_fan, 2, 86.0,  2.48, -4.114, -4.056, lambda yv: 0.502, nu=10, nv=5, offset=0.00062, max_snap=0.010)
    _cyl_patch(bm_co, uv_co, o_fan, 2, 290.5, 2.48, -4.114, -4.056, lambda yv: 0.502, nu=10, nv=5, offset=0.00062, max_snap=0.010)


    rot_deec = Matrix.Rotation(math.radians(132.0), 4, 'Y')
    deec_ctr_w = rot_deec @ Vector((-0.028, -4.2285, 0.556))
    def _deec_pt(u, v):
        return Vector((deec_ctr_w.x + (u - 0.5) * 0.0320, -4.2285, deec_ctr_w.z + (v - 0.5) * 0.0240))
    def _deec_nr(u, v):
        return (0.0, -1.0, 0.0)
    _emit_snapped_quad_decal(bm_co, uv_co, o_fan, 3, _deec_pt, _deec_nr, nu=4, nv=3, offset_m=0.00062, snap_to_host=True, max_snap_dev=0.004)


    _cyl_patch(bm_co, uv_co, o_tec, 4, 81.8,  3.46, -2.114, -2.032, lambda yv: 0.5437, nu=10, nv=4, offset=0.00062, max_snap=0.003)
    _cyl_patch(bm_co, uv_co, o_tec, 4, 283.0, 3.46, -2.114, -2.032, lambda yv: 0.5437, nu=10, nv=4, offset=0.00062, max_snap=0.003)


    _cyl_patch(bm_co, uv_co, o_tec, 5, 80.0,  10.5, -2.125, -1.995, lambda yv: 0.5437, nu=10, nv=5, offset=0.00042, max_snap=0.005)
    _cyl_patch(bm_co, uv_co, o_tec, 5, 258.0, 10.5, -2.125, -1.995, lambda yv: 0.5437, nu=10, nv=5, offset=0.00042, max_snap=0.005)
    _cyl_patch(bm_co, uv_co, o_oil, 5, 238.5, 8.2,  -4.034, -3.988, lambda yv: 0.554,  nu=6,  nv=5, offset=0.00042, max_snap=0.004)


    _cyl_leak_aft(bm_co, uv_co, o_tec, 6, 84.0,  3.6, -2.122, -1.992, lambda yv: 0.5437, nu=6, nv=10, offset=0.00044)
    _cyl_leak_aft(bm_co, uv_co, o_tec, 6, 260.0, 3.6, -2.122, -1.992, lambda yv: 0.5437, nu=6, nv=10, offset=0.00044)

    def _oil_port_drip_pt(u, v):
        yv = -4.132 + 0.028 * u
        ph = math.radians(228.0 + 9.2 * v)
        return (0.554 * math.sin(ph), yv, 0.554 * math.cos(ph))
    def _oil_port_drip_nr(u, v):
        ph = math.radians(228.0 + 9.2 * v)
        return (math.sin(ph), 0.0, math.cos(ph))
    _emit_snapped_quad_decal(bm_co, uv_co, o_oil, 6, _oil_port_drip_pt, _oil_port_drip_nr, nu=4, nv=10, offset_m=0.00044, snap_to_host=True, max_snap_dev=0.006)

    def _oil_glass_drip_pt(u, v):
        yv = -4.072 + 0.024 * u
        ph = math.radians(225.2 + 6.3 * v)
        return (0.554 * math.sin(ph), yv, 0.554 * math.cos(ph))
    def _oil_glass_drip_nr(u, v):
        ph = math.radians(225.2 + 6.3 * v)
        return (math.sin(ph), 0.0, math.cos(ph))
    _emit_snapped_quad_decal(bm_co, uv_co, o_oil, 6, _oil_glass_drip_pt, _oil_glass_drip_nr, nu=4, nv=8, offset_m=0.00044, snap_to_host=True, max_snap_dev=0.006)


    for phi_sp in (315.0, 135.0):
        def _spin_pt(u, v, _pc=phi_sp):

            yv = -4.376 + 0.034 * v
            rv = 0.0611 + 0.5127 * (yv + 4.45)
            ph = math.radians(_pc + 42.5 * (u - 0.5))
            return (rv * math.sin(ph), yv, rv * math.cos(ph))
        def _spin_nr(u, v, _pc=phi_sp):
            ph = math.radians(_pc + 42.5 * (u - 0.5))
            return (0.89 * math.sin(ph), -0.456, 0.89 * math.cos(ph))
        _emit_snapped_quad_decal(bm_co, uv_co, o_spn, 7, _spin_pt, _spin_nr, nu=12, nv=5, offset_m=0.00055, snap_to_host=True, max_snap_dev=0.008)


    _cyl_patch(bm_co, uv_co, o_oil, 8, 239.0, 6.2, -4.034, -3.986, lambda yv: 0.554, nu=6, nv=6, offset=0.00043, max_snap=0.004)


    r_act_d = 0.575
    r_band_d = 0.0251
    for k_ac in range(8):
        phi_ac = k_ac * 45.0
        rot_ac = Matrix.Rotation(math.radians(phi_ac), 4, 'Y')
        rot_ac3 = rot_ac.to_3x3()
        is_port_ac = (phi_ac <= 180.0)
        for ya_ac, yb_ac, ad_ac, uv_ac in (
            (-1.0365, -1.0055, 28.0, (0.0,  0.15,    0.35, 0.85)),
            (-0.9965, -0.9395, 40.0, (0.40, 0.03125, 1.00, 0.96875)),
        ):
            def _act_pt(u, v, _ya=ya_ac, _yb=yb_ac, _ad=ad_ac, _ip=is_port_ac, _rm=rot_ac):
                if _ip:
                    yv = _ya + (_yb - _ya) * u
                    ang = math.radians(+_ad - 2.0 * _ad * v)
                else:
                    yv = _yb - (_yb - _ya) * u
                    ang = math.radians(-_ad + 2.0 * _ad * v)
                p_loc = Vector((r_band_d * math.sin(ang), yv, r_act_d + r_band_d * math.cos(ang)))
                return _rm @ p_loc
            def _act_nr(u, v, _ad=ad_ac, _ip=is_port_ac, _r3=rot_ac3):
                ang = math.radians((+_ad - 2.0 * _ad * v) if _ip else (-_ad + 2.0 * _ad * v))
                n_loc = Vector((math.sin(ang), 0.0, math.cos(ang)))
                return _r3 @ n_loc
            _emit_snapped_quad_decal(
                bm_co, uv_co, None, 9, _act_pt, _act_nr,
                nu=8, nv=6, offset_m=0.00042, snap_to_host=False, uv_rect=uv_ac,
            )


    def _emit_tube_id_sleeve(bm_t, uvl_t, midx_t, phi_deg: float, y_ctr: float, r_rad: float, tube_r: float,
                             band_uv_v: tuple, length: float = 0.025, n_circ: int = 16):
        ph = math.radians(phi_deg)
        c_ax = Vector((r_rad * math.sin(ph), y_ctr, r_rad * math.cos(ph)))
        e_r = Vector((math.sin(ph), 0.0, math.cos(ph)))
        e_t = Vector((math.cos(ph), 0.0, -math.sin(ph)))
        e_y = Vector((0.0, 1.0, 0.0))
        r_sl = tube_r + 0.00046
        v_lo, v_hi = band_uv_v
        is_port_t = (0.0 < (phi_deg % 360.0) < 180.0)

        def _pt_sl(u, v):
            ang = 2.0 * math.pi * (u if is_port_t else (1.0 - u))
            y_off = ((v - 0.5) if is_port_t else (0.5 - v)) * length
            return c_ax + e_y * y_off + (e_r * math.cos(ang) + e_t * math.sin(ang)) * r_sl
        def _nr_sl(u, v):
            ang = 2.0 * math.pi * (u if is_port_t else (1.0 - u))
            return (e_r * math.cos(ang) + e_t * math.sin(ang))
        _emit_snapped_quad_decal(
            bm_t, uvl_t, None, midx_t, _pt_sl, _nr_sl,
            nu=n_circ, nv=2, offset_m=0.0, snap_to_host=False, uv_rect=(0.0, v_lo, 0.50, v_hi),
        )

    UV_FUEL  = (0.76, 0.99)
    UV_HYD   = (0.51, 0.74)
    UV_LUBE  = (0.26, 0.49)
    UV_BLEED = (0.01, 0.24)
    fluid_sleeve_specs = [

        (79.0,  -3.682, 0.5090, 0.0046, UV_HYD),
        (67.0,  -3.240, 0.5110, 0.0046, UV_HYD),
        (72.0,  -3.655, 0.5115, 0.0043, UV_FUEL),
        (61.5,  -3.220, 0.5130, 0.0043, UV_FUEL),
        (64.0,  -3.625, 0.5100, 0.0042, UV_LUBE),
        (55.5,  -3.200, 0.5110, 0.0042, UV_LUBE),
        (35.0,  -3.455, 0.5080, 0.0045, UV_BLEED),
        (27.5,  -3.360, 0.5060, 0.0040, UV_HYD),
        (85.0,  -3.190, 0.5115, 0.0038, UV_FUEL),
        (18.0,  -3.480, 0.5095, 0.0054, UV_BLEED),

        (280.5, -3.585, 0.5200, 0.0072, UV_BLEED),
        (249.0, -3.415, 0.5100, 0.0046, UV_FUEL),
        (259.0, -3.190, 0.5120, 0.0034, UV_HYD),
        (255.0, -3.260, 0.5100, 0.0034, UV_HYD),
        (304.0, -3.200, 0.5110, 0.0036, UV_HYD),
        (342.0, -3.420, 0.5085, 0.0052, UV_FUEL),
        (262.5, -3.400, 0.5090, 0.0038, UV_LUBE),
        (289.2, -3.828, 0.5220, 0.0056, UV_LUBE),
    ]
    for ph_s, y_s, r_s, tr_s, uv_s in fluid_sleeve_specs:
        _emit_tube_id_sleeve(bm_co, uv_co, 10, ph_s, y_s, r_s, tr_s, uv_s, length=0.018, n_circ=16)


    rot_sg_d = Matrix.Rotation(math.radians(239.4), 4, 'Y')
    rot_sg_d3 = rot_sg_d.to_3x3()
    r_sgg = 0.00568
    def _sg_grad_pt(u, v):
        ang = math.radians(+58.0 - 116.0 * u)
        lx = -0.048 + 0.096 * v
        p_l = Vector((lx, -4.059 + r_sgg * math.sin(ang), 0.5515 + r_sgg * math.cos(ang)))
        return rot_sg_d @ p_l
    def _sg_grad_nr(u, v):
        ang = math.radians(+58.0 - 116.0 * u)
        return rot_sg_d3 @ Vector((0.0, math.sin(ang), math.cos(ang)))
    _emit_snapped_quad_decal(
        bm_co, uv_co, None, 11, _sg_grad_pt, _sg_grad_nr,
        nu=6, nv=14, offset_m=0.0, snap_to_host=False, uv_rect=(0.0, 0.0, 1.0, 1.0),
    )


    o_waf = bpy.data.objects.get("Static_Forward_Waffle_Duct")
    o_tec_m = bpy.data.objects.get("Static_Forward_TEC_And_Mount")
    def _cyl_uv_patch(bm_t, uvl_t, host_t, midx_t, phi_c_deg, dphi_deg, y0, y1, r_fn, uv_rect, nu=8, nv=6, offset=0.00055, max_snap=0.012):
        is_port_t = (0.0 < (phi_c_deg % 360.0) < 180.0)
        def _pt(u, v):
            if is_port_t:
                yv = y0 + (y1 - y0) * u
                ph = math.radians(phi_c_deg + dphi_deg * (0.5 - v))
            else:
                yv = y1 - (y1 - y0) * u
                ph = math.radians(phi_c_deg - dphi_deg * (0.5 - v))
            rv = r_fn(yv)
            return (rv * math.sin(ph), yv, rv * math.cos(ph))
        def _nr(u, v):
            if is_port_t:
                ph = math.radians(phi_c_deg + dphi_deg * (0.5 - v))
            else:
                ph = math.radians(phi_c_deg - dphi_deg * (0.5 - v))
            return (math.sin(ph), 0.0, math.cos(ph))
        _emit_snapped_quad_decal(
            bm_t, uvl_t, host_t, midx_t, _pt, _nr,
            nu=nu, nv=nv, offset_m=offset, snap_to_host=(host_t is not None), max_snap_dev=max_snap, uv_rect=uv_rect, use_raycast=True,
        )


    if o_waf is not None:
        _cyl_uv_patch(bm_co, uv_co, o_waf, 12,  55.6, 4.8, -1.654, -1.594, lambda y: 0.5380, (0.01, 0.51, 0.49, 0.99), nu=8, nv=6, offset=0.00055, max_snap=0.006)
        _cyl_uv_patch(bm_co, uv_co, o_waf, 12,  70.6, 4.8, -1.216, -1.156, lambda y: 0.5387, (0.51, 0.51, 0.99, 0.99), nu=8, nv=6, offset=0.00055, max_snap=0.006)
        _cyl_uv_patch(bm_co, uv_co, o_waf, 12, 235.6, 4.8, -1.654, -1.594, lambda y: 0.5380, (0.01, 0.51, 0.49, 0.99), nu=8, nv=6, offset=0.00055, max_snap=0.006)
        _cyl_uv_patch(bm_co, uv_co, o_waf, 12, 295.6, 4.8, -1.136, -1.076, lambda y: 0.5388, (0.51, 0.51, 0.99, 0.99), nu=8, nv=6, offset=0.00055, max_snap=0.006)

    if o_tec_m is not None:
        _cyl_uv_patch(bm_co, uv_co, o_tec_m, 12,  46.0, 7.2, -2.126, -2.044, lambda y: 0.549, (0.01, 0.01, 0.49, 0.49), nu=8, nv=6, offset=0.00060, max_snap=0.012)
        _cyl_uv_patch(bm_co, uv_co, o_tec_m, 12, 244.0, 7.2, -2.126, -2.044, lambda y: 0.549, (0.01, 0.01, 0.49, 0.49), nu=8, nv=6, offset=0.00060, max_snap=0.012)

        for ph_mm in (52.0, 82.0, 116.0, 248.0, 282.0, 312.0):
            _cyl_uv_patch(bm_co, uv_co, o_tec_m, 12, ph_mm, 3.2, -1.958, -1.904, lambda y: 0.549, (0.51, 0.01, 0.99, 0.49), nu=8, nv=4, offset=0.00060, max_snap=0.012)

    _finalize_decal_mesh_obj(
        "Decals_Engine_Core", bm_co,
        [m_tec, m_oil, m_intake, m_deec, m_hyd, m_wx_sc_al, m_wx_leak, m_spinmk, m_wx_hand, m_actband, m_flbands, m_oilsight, m_pmark],
        coll_dec,
    )


    bm_st = bmesh.new()
    uv_st = bm_st.loops.layers.uv.new("UVMap")
    o_rail = bpy.data.objects.get("Stand_Cradle_01_Longitudinal_Rails_And_Crossbeams")
    o_arch = bpy.data.objects.get("Stand_Cradle_03_RollOver_Arches_And_MountPickups")
    o_topb = bpy.data.objects.get("Stand_Cradle_04_TopTieBar_And_DiagonalBraces")

    def _flat_x_decal(bm, uvl, host, midx, sx, x_plane, y0, y1, z0, z1, nu=6, nv=2, offset=0.00062, max_snap=0.008, uv_rect=(0.0, 0.0, 1.0, 1.0)):
        def _pt(u, v):
            yv = (y0 + (y1 - y0) * u) if sx > 0 else (y1 - (y1 - y0) * u)
            zv = z0 + (z1 - z0) * v
            return (sx * x_plane, yv, zv)
        def _nr(u, v):
            return (sx, 0.0, 0.0)
        _emit_snapped_quad_decal(bm, uvl, host, midx, _pt, _nr, nu=nu, nv=nv, offset_m=offset, snap_to_host=True, max_snap_dev=max_snap, uv_rect=uv_rect)

    def _flat_y_decal(bm, uvl, host, midx, sy, y_plane, x0, x1, z0, z1, nu=8, nv=2, offset=0.00062, max_snap=0.008, uv_rect=(0.0, 0.0, 1.0, 1.0)):

        def _pt(u, v):
            xv = (x0 + (x1 - x0) * u) if sy < 0 else (x1 - (x1 - x0) * u)
            zv = z0 + (z1 - z0) * v
            return (xv, y_plane, zv)
        def _nr(u, v):
            return (0.0, sy, 0.0)
        _emit_snapped_quad_decal(bm, uvl, host, midx, _pt, _nr, nu=nu, nv=nv, offset_m=offset, snap_to_host=True, max_snap_dev=max_snap, uv_rect=uv_rect)

    def _flat_z_decal(bm, uvl, host, midx, z_plane, x0, x1, y0, y1, nu=6, nv=2, offset=0.00045, max_snap=0.004, uv_rect=(0.0, 0.0, 1.0, 1.0)):

        def _pt(u, v):
            xv = x0 + (x1 - x0) * u
            yv = y0 + (y1 - y0) * v
            return (xv, yv, z_plane)
        def _nr(u, v):
            return (0.0, 0.0, 1.0)
        _emit_snapped_quad_decal(bm, uvl, host, midx, _pt, _nr, nu=nu, nv=nv, offset_m=offset, snap_to_host=True, max_snap_dev=max_snap, uv_rect=uv_rect)

    for sx in (-1.0, +1.0):

        _flat_x_decal(bm_st, uv_st, o_rail, 0, sx, 0.684, -3.265, -3.035, -0.865, -0.793, nu=8, nv=3)
        _flat_x_decal(bm_st, uv_st, o_rail, 0, sx, 0.684, -1.660, -1.538, -0.848, -0.810, nu=6, nv=3)


        _flat_x_decal(bm_st, uv_st, o_rail, 1, sx, 0.684, -2.960, -2.650, -0.863, -0.795, nu=8, nv=2)
        _flat_x_decal(bm_st, uv_st, o_rail, 1, sx, 0.684, -0.655, -0.525, -0.843, -0.815, nu=6, nv=2)


        for y_fk in (-3.220, -2.520):
            _flat_x_decal(bm_st, uv_st, o_rail, 2, sx, 0.752, y_fk - 0.095, y_fk + 0.095, -0.9965, -0.9815, nu=8, nv=2)
            _flat_x_decal(bm_st, uv_st, o_rail, 2, sx, 0.752, y_fk - 0.095, y_fk + 0.095, -0.9145, -0.8995, nu=8, nv=2)


        for y_ar in (-3.856, -2.333):
            def _arch_pt(u, v, _sx=sx, _ya=y_ar):
                yv = (_ya - 0.050 + 0.100 * u) if _sx > 0 else (_ya + 0.050 - 0.100 * u)
                deg = _sx * (64.92 - 12.35 * v)
                ar = math.radians(deg)
                return (0.696 * math.sin(ar), yv, 0.696 * math.cos(ar))
            def _arch_nr(u, v, _sx=sx):
                deg = _sx * (64.92 - 12.35 * v)
                ar = math.radians(deg)
                return (math.sin(ar), 0.0, math.cos(ar))
            _emit_snapped_quad_decal(bm_st, uv_st, o_arch, 3, _arch_pt, _arch_nr, nu=6, nv=10, offset_m=0.00062, snap_to_host=True, max_snap_dev=0.010)


        _flat_x_decal(bm_st, uv_st, o_rail, 4, sx, 0.684, -2.096, -1.984, -0.858, -0.802, nu=6, nv=2)


        _flat_x_decal(bm_st, uv_st, o_rail, 5, sx, 0.684, -3.290, -3.035, -0.873, -0.788, nu=8, nv=3, offset=0.00042, max_snap=0.005)
        _flat_x_decal(bm_st, uv_st, o_rail, 5, sx, 0.684, -2.980, -2.640, -0.873, -0.788, nu=8, nv=3, offset=0.00042, max_snap=0.005)


        _flat_x_decal(bm_st, uv_st, o_rail, 6, sx, 0.684, -2.990, -2.635, -0.875, -0.824, nu=10, nv=3, offset=0.00045, max_snap=0.005)
        _flat_x_decal(bm_st, uv_st, o_rail, 6, sx, 0.684, -1.045, -0.895, -0.875, -0.826, nu=6,  nv=3, offset=0.00045, max_snap=0.005)


        for y_hb0, y_hb1 in ((-3.540, -3.280), (-2.900, -2.640)):
            def _hoist_pt(u, v, _sx=sx, _y0=y_hb0, _y1=y_hb1):
                yv = (_y0 + (_y1 - _y0) * u) if _sx > 0 else (_y1 - (_y1 - _y0) * u)
                zc = 0.736 + 0.012 * (yv - (-3.784)) / 1.379
                th = math.radians(-36.3 + 72.6 * v)
                return (_sx * 0.0410 * math.cos(th), yv, zc + 0.0410 * math.sin(th))
            def _hoist_nr(u, v, _sx=sx):
                th = math.radians(-36.3 + 72.6 * v)
                return (_sx * math.cos(th), 0.0, math.sin(th))
            _emit_snapped_quad_decal(bm_st, uv_st, o_topb, 7, _hoist_pt, _hoist_nr, nu=10, nv=6, offset_m=0.00058, snap_to_host=True, max_snap_dev=0.008)


        for y_ar in (-3.856, -2.333):
            _flat_x_decal(bm_st, uv_st, o_arch, 9, sx, 0.7578, y_ar - 0.054, y_ar + 0.054, -0.139, -0.116, nu=6, nv=2, offset=0.00042, max_snap=0.004)


        _flat_x_decal(bm_st, uv_st, o_rail, 10, sx, 0.7420, -4.054, -3.786, -0.860, -0.798, nu=6, nv=2, offset=0.00042, max_snap=0.004, uv_rect=(0.01, 0.61, 0.49, 0.79))
        _flat_x_decal(bm_st, uv_st, o_rail, 10, sx, 0.7420, -1.934, -1.686, -0.860, -0.798, nu=6, nv=2, offset=0.00042, max_snap=0.004, uv_rect=(0.51, 0.61, 0.99, 0.79))
        _flat_x_decal(bm_st, uv_st, o_rail, 10, sx, 0.7420, -1.314, -1.116, -0.860, -0.798, nu=6, nv=2, offset=0.00042, max_snap=0.004, uv_rect=(0.01, 0.41, 0.49, 0.59))

        for ya_w, yb_w in ((-3.562, -3.418), (-2.424, -2.252)):
            _flat_x_decal(bm_st, uv_st, o_rail, 10, sx, 0.7435, ya_w, yb_w, -0.856, -0.800, nu=6, nv=2, offset=0.00042, max_snap=0.004, uv_rect=(0.01, 0.81, 0.49, 0.99))

        for y_fod in (-2.520, -1.460):
            _flat_x_decal(bm_st, uv_st, o_rail, 10, sx, 0.7415, y_fod - 0.055, y_fod + 0.055, -0.847, -0.807, nu=4, nv=2, offset=0.00042, max_snap=0.004, uv_rect=(0.51, 0.81, 0.99, 0.99))

        for y_ar in (-3.856, -2.333):
            for deg_pl_a, hy_mp, hw_ang, uv_rc in (
                (81.5, 0.045, 0.022, (0.01, 0.01, 0.49, 0.19)),
                (74.5, 0.041, 0.018, (0.51, 0.01, 0.99, 0.19)),
            ):
                ar_p = math.radians(deg_pl_a)
                sa_p, ca_p = math.sin(ar_p), math.cos(ar_p)
                r_pl_out = 0.6982
                def _apl_pt(u, v, _sx=sx, _ya=y_ar, _hy=hy_mp, _hw=hw_ang, _sa=sa_p, _ca=ca_p, _ro=r_pl_out):
                    yv = (_ya - _hy + 2.0 * _hy * u) if _sx > 0 else (_ya + _hy - 2.0 * _hy * u)
                    dv = (v - 0.5) * 2.0 * _hw
                    xv = _sx * (_ro * _sa - dv * _ca)
                    zv = _ro * _ca + dv * _sa
                    return (xv, yv, zv)
                def _apl_nr(u, v, _sx=sx, _sa=sa_p, _ca=ca_p):
                    return (_sx * _sa, 0.0, _ca)
                _emit_snapped_quad_decal(bm_st, uv_st, o_arch, 10, _apl_pt, _apl_nr, nu=4, nv=2, offset_m=0.00042, snap_to_host=True, max_snap_dev=0.004, uv_rect=uv_rc)


    for sy, y_pl in ((-1.0, -4.285), (+1.0, -0.605)):
        _flat_y_decal(bm_st, uv_st, o_rail, 8, sy, y_pl, -0.160, +0.160, -0.862, -0.798, nu=10, nv=3, offset=0.00062, max_snap=0.006)
        _flat_y_decal(bm_st, uv_st, o_rail, 1, sy, y_pl, -0.525, -0.275, -0.858, -0.803, nu=8,  nv=2, offset=0.00062, max_snap=0.006)
        _flat_y_decal(bm_st, uv_st, o_rail, 1, sy, y_pl, +0.275, +0.525, -0.858, -0.803, nu=8,  nv=2, offset=0.00062, max_snap=0.006)
        _flat_y_decal(bm_st, uv_st, o_rail, 5, sy, y_pl, -0.450, +0.450, -0.872, -0.790, nu=10, nv=3, offset=0.00042, max_snap=0.005)
        _flat_y_decal(bm_st, uv_st, o_rail, 6, sy, y_pl, -0.540, +0.540, -0.874, -0.828, nu=12, nv=3, offset=0.00045, max_snap=0.005)


    def _fod_lid_pt(u, v):
        yv = -1.855 + 0.330 * u
        xv = +0.412 - 0.082 * v
        return (xv, yv, -0.68895)
    def _fod_lid_nr(u, v):
        return (0.0, 0.0, 1.0)
    _emit_snapped_quad_decal(
        bm_st, uv_st, o_rail, 11, _fod_lid_pt, _fod_lid_nr,
        nu=6, nv=2, offset_m=0.00042, snap_to_host=True, max_snap_dev=0.004,
        uv_rect=(0.01, 0.51, 0.99, 0.99),
    )

    def _tdoc_pt(u, v):
        yv = -1.800 + 0.160 * u
        ang = math.radians(67.0 - 52.0 * v)
        return (-0.455 + 0.042 * math.sin(ang), yv, -0.730 + 0.042 * math.cos(ang))
    def _tdoc_nr(u, v):
        ang = math.radians(67.0 - 52.0 * v)
        return (math.sin(ang), 0.0, math.cos(ang))
    _emit_snapped_quad_decal(
        bm_st, uv_st, o_rail, 11, _tdoc_pt, _tdoc_nr,
        nu=6, nv=6, offset_m=0.00042, snap_to_host=True, max_snap_dev=0.004,
        uv_rect=(0.01, 0.01, 0.99, 0.49),
    )

    _finalize_decal_mesh_obj(
        "Decals_Stand_Cradle", bm_st,
        [m_tk9, m_refl, m_fork, m_m4, m_pmel, m_wx_sc_al, m_wx_rust, m_hoist, m_fwdbmp, m_m4_rl, m_st_atlas, m_foddoc],
        coll_dec,
    )


    bm_tr = bmesh.new()
    uv_tr = bm_tr.loops.layers.uv.new("UVMap")
    o_tr = bpy.data.objects.get("Stand_Trailer_05_M3000_Steerable_Axles_And_TowBar")
    for sx in (-1.0, +1.0):

        _flat_x_decal(bm_tr, uv_tr, o_tr, 0, sx, 0.690, -3.979, -3.791, -0.981, -0.937, nu=8, nv=2)
        _flat_x_decal(bm_tr, uv_tr, o_tr, 0, sx, 0.690, -3.045, -2.874, -0.980, -0.940, nu=8, nv=2)

        _flat_x_decal(bm_tr, uv_tr, o_tr, 1, sx, 0.690, -1.755, -1.545, -0.981, -0.935, nu=6, nv=2)

        _flat_x_decal(bm_tr, uv_tr, o_tr, 2, sx, 0.690, -4.060, -3.710, -0.985, -0.934, nu=8, nv=2, offset=0.00042, max_snap=0.005)
        _flat_x_decal(bm_tr, uv_tr, o_tr, 2, sx, 0.690, -3.060, -2.855, -0.985, -0.934, nu=8, nv=2, offset=0.00042, max_snap=0.005)

        _flat_x_decal(bm_tr, uv_tr, o_tr, 3, sx, 0.690, -4.065, -3.705, -0.987, -0.950, nu=10, nv=3, offset=0.00045, max_snap=0.005)
        _flat_x_decal(bm_tr, uv_tr, o_tr, 3, sx, 0.690, -3.060, -2.855, -0.987, -0.950, nu=8,  nv=3, offset=0.00045, max_snap=0.005)

        _flat_x_decal(bm_tr, uv_tr, o_tr, 4, sx, 0.694, -2.230, -2.020, -0.983, -0.934, nu=6, nv=2, offset=0.00045, max_snap=0.004, uv_rect=(0.0, 0.54, 1.0, 0.96))
        _flat_x_decal(bm_tr, uv_tr, o_tr, 4, sx, 0.694, -1.445, -1.215, -0.983, -0.934, nu=6, nv=2, offset=0.00045, max_snap=0.004, uv_rect=(0.0, 0.04, 1.0, 0.46))

    _flat_y_decal(bm_tr, uv_tr, o_tr, 3, -1.0, -4.282, -0.352, -0.145, -0.988, -0.954, nu=6, nv=3, offset=0.00045, max_snap=0.005)
    _flat_y_decal(bm_tr, uv_tr, o_tr, 3, -1.0, -4.282, +0.145, +0.352, -0.988, -0.954, nu=6, nv=3, offset=0.00045, max_snap=0.005)

    _flat_y_decal(bm_tr, uv_tr, o_tr, 5, -1.0, -4.286, -0.131, +0.131, -0.986, -0.934, nu=6, nv=2, offset=0.00042, max_snap=0.004, uv_rect=(0.51, 0.41, 0.99, 0.59))
    _flat_y_decal(bm_tr, uv_tr, o_tr, 5, +1.0, -1.317, -0.061, +0.061, -1.354, -1.298, nu=6, nv=2, offset=0.00042, max_snap=0.004, uv_rect=(0.01, 0.21, 0.49, 0.39))
    _flat_y_decal(bm_tr, uv_tr, o_tr, 5, -1.0, -4.0995, -0.092, +0.092, -1.372, -1.322, nu=6, nv=2, offset=0.00042, max_snap=0.004, uv_rect=(0.51, 0.21, 0.99, 0.39))
    _finalize_decal_mesh_obj(
        "Decals_Stand_Trailer", bm_tr,
        [m_trspec, m_refl, m_wx_sc_al, m_wx_rust, m_tr_dp, m_st_atlas],
        coll_dec,
    )


    o_wh = bpy.data.objects.get("Stand_Trailer_06_Pneumatic_GroundSupport_Wheels")
    if o_wh is not None:
        bm_wh = bmesh.new()
        uv_wh = bm_wh.loops.layers.uv.new("UVMap")
        wheel_slip_specs = [
            (+1.0, -3.8800,  36.0),
            (+1.0, -1.4950, 138.0),
            (-1.0, -3.8800, 222.0),
            (-1.0, -1.4950, 314.0),
        ]
        cz_wh = -1.3200
        for sx_w, cy_w, clk_deg in wheel_slip_specs:
            ang_w = math.radians(clk_deg)
            e_rad = Vector((0.0, math.cos(ang_w), math.sin(ang_w)))
            e_tan = Vector((0.0, -math.sin(ang_w), math.cos(ang_w)))
            def _slip_pt(u, v, _sx=sx_w, _cy=cy_w, _er=e_rad, _et=e_tan):
                su = ((0.5 - u) if _sx > 0 else (u - 0.5)) * 0.022

                rv = 0.149 + 0.079 * v
                x_nom = _sx * 0.778
                p_yz = Vector((0.0, _cy, cz_wh)) + _er * rv + _et * su
                return (x_nom, p_yz.y, p_yz.z)
            def _slip_nr(u, v, _sx=sx_w):
                return (_sx, 0.0, 0.0)
            _emit_snapped_quad_decal(
                bm_wh, uv_wh, o_wh, 0, _slip_pt, _slip_nr,
                nu=4, nv=16, offset_m=0.00075, snap_to_host=True, max_snap_dev=0.032, use_raycast=True,
            )
        _finalize_decal_mesh_obj("Decals_Stand_Wheels", bm_wh, [m_tireslip], coll_dec)


    arm_noz = bpy.data.objects.get("F15_Nozzle_Rig")
    if arm_noz is not None and arm_noz.type == 'ARMATURE':
        bm_pn = bmesh.new()
        uv_pn = bm_pn.loops.layers.uv.new("UVMap")
        dvert_pn = bm_pn.verts.layers.deform.new()
        bone_names_pn = []
        inv_arm_mw = arm_noz.matrix_world.inverted()

        for i_fl in range(16):
            ob_fl = bpy.data.objects.get(f"Flap_DivMaster_{i_fl:02d}")
            bname = f"B_DivFlap_{i_fl:02d}"
            if ob_fl is None or bname not in arm_noz.data.bones or bname not in arm_noz.pose.bones:
                continue
            g_idx = len(bone_names_pn)
            bone_names_pn.append(bname)


            m_rest = arm_noz.data.bones[bname].matrix_local
            m_pose = arm_noz.pose.bones[bname].matrix
            m_local_to_rest = m_rest @ m_pose.inverted() @ inv_arm_mw @ ob_fl.matrix_world


            is_stbd_fl = (i_fl >= 8)
            col_i = i_fl % 8
            row_i = i_fl // 8
            u0_r = col_i / 8.0 + 0.006
            u1_r = (col_i + 1) / 8.0 - 0.006
            v0_r = (0.506 if row_i == 0 else 0.006)
            v1_r = (0.994 if row_i == 0 else 0.494)


            nu_p, nv_p = 4, 4
            grid_p = []
            for j in range(nv_p + 1):
                fv = j / float(nv_p)
                row_v = []
                for i in range(nu_p + 1):
                    fu = i / float(nu_p)

                    if not is_stbd_fl:
                        yl = 0.806 + 0.034 * fu
                        xl = -0.021 - 0.028 * fv
                    else:
                        yl = 0.840 - 0.034 * fu
                        xl = -0.049 + 0.028 * fv
                    p_try = Vector((xl, yl, 0.370))
                    ok_r, loc_r, nrm_r, _ = ob_fl.ray_cast(p_try, Vector((0.0, 0.0, -1.0)), distance=0.045)
                    if ok_r:
                        p_loc = loc_r + nrm_r.normalized() * 0.00045
                    else:
                        p_loc = Vector((xl, yl, 0.3522))
                    p_rest = m_local_to_rest @ p_loc
                    v_new = bm_pn.verts.new(p_rest)
                    v_new[dvert_pn][g_idx] = 1.0
                    row_v.append(v_new)
                grid_p.append(row_v)

            for j in range(nv_p):
                fv0, fv1 = j / float(nv_p), (j + 1) / float(nv_p)
                for i in range(nu_p):
                    fu0, fu1 = i / float(nu_p), (i + 1) / float(nu_p)

                    try:
                        fq = bm_pn.faces.new((grid_p[j][i], grid_p[j][i + 1], grid_p[j + 1][i + 1], grid_p[j + 1][i]))
                        fq.material_index = 0
                        fq.smooth = True
                        for lp, (ua, va) in zip(fq.loops, [(fu0, fv0), (fu1, fv0), (fu1, fv1), (fu0, fv1)]):
                            lp[uv_pn].uv = (u0_r + (u1_r - u0_r) * ua, v0_r + (v1_r - v0_r) * va)
                    except ValueError:
                        pass

        ob_pn = _finalize_decal_mesh_obj("Decals_Nozzle_PetalNumbers", bm_pn, [m_petalnum], coll_dec)
        ob_pn.matrix_world = arm_noz.matrix_world.copy()
        for bn in bone_names_pn:
            ob_pn.vertex_groups.new(name=bn)
        mod_arm = ob_pn.modifiers.new("Armature", 'ARMATURE')
        mod_arm.object = arm_noz

        try:
            with bpy.context.temp_override(object=ob_pn):
                bpy.ops.object.modifier_move_to_index(modifier=mod_arm.name, index=0)
        except Exception:
            pass


    def _stamp_articulated_nozzle_decal(me_target, is_stbd: bool, uv_rect: tuple, width_m: float, height_m: float, offset_m: float = 0.00028):
        if me_target is None:
            return

        plate_slots = {
            idx for idx, m in enumerate(me_target.materials)
            if m is not None and m.name in ("Mat_ID18c_Nozzle_StampedTag", COLOR_ID_SPECS[MAT_PLATE_BASE][0])
        }
        if not plate_slots:
            return

        cand_faces = [p for p in me_target.polygons if p.material_index in plate_slots and p.normal.z > 0.35]
        if not cand_faces:
            return
        p_dom = max(cand_faces, key=lambda p: p.area)
        n_out = p_dom.normal.normalized()
        flat_verts = []
        for p in cand_faces:
            if p.normal.dot(n_out) > 0.995:
                for vi in p.vertices:
                    flat_verts.append(me_target.vertices[vi].co)
        if not flat_verts:
            return


        t_long = Vector((0.0, n_out.z, -n_out.y)).normalized()
        if t_long.y < 0.0:
            t_long = -t_long
        u_right = (-t_long) if is_stbd else t_long
        v_up = n_out.cross(u_right).normalized()


        h_max = max(v.dot(n_out) for v in flat_verts)
        top_vs = [v for v in flat_verts if v.dot(n_out) >= h_max - 0.00018]
        s_vals = [v.dot(t_long) for v in top_vs]
        s_mid = 0.5 * (min(s_vals) + max(s_vals))
        ctr = t_long * s_mid + n_out * h_max
        ctr.x = 0.0


        dec_slot = None
        for idx, m in enumerate(me_target.materials):
            if m is not None and m.name == m_noztag.name:
                dec_slot = idx
                break
        if dec_slot is None:
            me_target.materials.append(m_noztag)
            dec_slot = len(me_target.materials) - 1

        bm_m = bmesh.new()
        bm_m.from_mesh(me_target)
        old_fs = [f for f in bm_m.faces if f.material_index == dec_slot]
        if old_fs:
            bmesh.ops.delete(bm_m, geom=old_fs, context='FACES')
        uv_l = bm_m.loops.layers.uv.get("UVMap") or bm_m.loops.layers.uv.new("UVMap")

        def _pt_d(u, v):
            return ctr + u_right * ((u - 0.5) * width_m) + v_up * ((v - 0.5) * height_m)
        def _nr_d(u, v):
            return n_out
        _emit_snapped_quad_decal(
            bm_m, uv_l, None, dec_slot, _pt_d, _nr_d,
            nu=4, nv=2, offset_m=offset_m, snap_to_host=False, uv_rect=uv_rect,
        )
        bm_m.to_mesh(me_target)
        bm_m.free()
        me_target.update()


    me_bal_p = bpy.data.meshes.get("Mesh_BalanceFlap")
    if me_bal_p is not None:
        me_bal_s = bpy.data.meshes.get("Mesh_BalanceFlap_Stbd")
        if me_bal_s is None:
            me_bal_s = me_bal_p.copy()
            me_bal_s.name = "Mesh_BalanceFlap_Stbd"
        for i_b in range(16):
            ob_b = bpy.data.objects.get(f"Flap_Balance_{i_b:02d}")
            if ob_b is not None:
                ob_b.data = me_bal_p if i_b < 8 else me_bal_s
        _stamp_articulated_nozzle_decal(me_bal_p, False, (0.0, 0.54, 1.0, 0.96), width_m=0.0320, height_m=0.0074, offset_m=0.00028)
        _stamp_articulated_nozzle_decal(me_bal_s, True,  (0.0, 0.54, 1.0, 0.96), width_m=0.0320, height_m=0.0074, offset_m=0.00028)


    me_ds_p = bpy.data.meshes.get("Mesh_Diagonal_Strut")
    if me_ds_p is not None:
        me_ds_s = bpy.data.meshes.get("Mesh_Diagonal_Strut_Stbd")
        if me_ds_s is None:
            me_ds_s = me_ds_p.copy()
            me_ds_s.name = "Mesh_Diagonal_Strut_Stbd"
        for i_d in range(16):
            ob_d = bpy.data.objects.get(f"Link_DiagStrut_{i_d:02d}")
            if ob_d is not None:
                ob_d.data = me_ds_p if i_d <= 8 else me_ds_s
        _stamp_articulated_nozzle_decal(me_ds_p, False, (0.0, 0.04, 1.0, 0.46), width_m=0.0740, height_m=0.0094, offset_m=0.00028)
        _stamp_articulated_nozzle_decal(me_ds_s, True,  (0.0, 0.04, 1.0, 0.46), width_m=0.0740, height_m=0.0094, offset_m=0.00028)


    ng_masks_ts = bpy.data.node_groups.get("NG_F15_SurfaceMasks")
    ng_wrap_ts  = bpy.data.node_groups.get("NG_F15_ModeWrapper")

    def _build_torque_seal_mat(mat_name: str, hex_col: int, id_hex: int):
        mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
        nt = mat.node_tree
        nt.nodes.clear()
        tc = _new_node(nt, "ShaderNodeTexCoord", (-1180, 0))
        ctrl = _new_node(nt, "ShaderNodeGroup", (-1180, -340))
        ctrl.node_tree = ng_global

        masks = None
        if ng_masks_ts is not None:
            masks = _new_node(nt, "ShaderNodeGroup", (-900, -160))
            masks.node_tree = ng_masks_ts
            nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
            masks.inputs["AO_Dist"].default_value = 0.010
            masks.inputs["Fine_Bevel_Rad"].default_value = 0.00018

        n_b = _new_node(nt, "ShaderNodeTexNoise", (-680, -380))
        n_b.inputs["Scale"].default_value = 340.0
        n_b.inputs["Detail"].default_value = 6.0
        n_b.inputs["Roughness"].default_value = 0.68
        nt.links.new(tc.outputs["Object"], n_b.inputs["Vector"])


        wx_ts = _new_node(nt, "ShaderNodeMath", (-620, -240)); wx_ts.operation = 'MULTIPLY_ADD'; wx_ts.use_clamp = True
        wx_ts.inputs[1].default_value = 0.28
        nt.links.new(ctrl.outputs["SurfaceErosion"], wx_ts.inputs[0])
        wx_w = _new_node(nt, "ShaderNodeMath", (-800, -340)); wx_w.operation = 'MULTIPLY'; wx_w.use_clamp = True
        wx_w.inputs[1].default_value = 0.24
        nt.links.new(ctrl.outputs["Weathering"], wx_w.inputs[0])
        nt.links.new(wx_w.outputs[0], wx_ts.inputs[2])

        if masks is not None:
            cr_ts = _new_node(nt, "ShaderNodeMath", (-440, -160)); cr_ts.operation = 'MULTIPLY'; cr_ts.use_clamp = True
            nt.links.new(masks.outputs["Crevice_Mask"], cr_ts.inputs[0])
            nt.links.new(wx_ts.outputs[0], cr_ts.inputs[1])
            dust_ts = cr_ts.outputs[0]
        else:
            dust_ts = wx_ts.outputs[0]

        col_ts = _new_node(nt, "ShaderNodeMix", (-180, 140)); col_ts.data_type = 'RGBA'
        col_ts.inputs[6].default_value = rgba(hex_col)
        col_ts.inputs[7].default_value = rgba(0x363026)
        nt.links.new(dust_ts, col_ts.inputs[0])

        r_base = _new_node(nt, "ShaderNodeMapRange", (-420, -40)); r_base.clamp = True
        r_base.inputs[1].default_value = 0.25; r_base.inputs[2].default_value = 0.78
        r_base.inputs[3].default_value = 0.31; r_base.inputs[4].default_value = 0.42
        nt.links.new(n_b.outputs["Factor"], r_base.inputs[0])
        r_add = _new_node(nt, "ShaderNodeMath", (-180, -40)); r_add.operation = 'MULTIPLY_ADD'; r_add.use_clamp = True
        r_add.inputs[1].default_value = 0.14
        nt.links.new(wx_ts.outputs[0], r_add.inputs[0])
        nt.links.new(r_base.outputs[0], r_add.inputs[2])

        bmp = _new_node(nt, "ShaderNodeBump", (-180, -240))
        bmp.inputs["Strength"].default_value = 0.085
        bmp.inputs["Distance"].default_value = 0.00028
        nt.links.new(n_b.outputs["Factor"], bmp.inputs["Height"])
        if masks is not None:
            nt.links.new(masks.outputs["Bevel_Normal"], bmp.inputs["Normal"])

        bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (100, 120))
        nt.links.new(col_ts.outputs[2], bsdf.inputs["Base Color"])
        bsdf.inputs["Metallic"].default_value = 0.0
        nt.links.new(r_add.outputs[0], bsdf.inputs["Roughness"])
        bsdf.inputs["Specular IOR Level"].default_value = 0.48
        bsdf.inputs["Coat Weight"].default_value = 0.24
        bsdf.inputs["Coat Roughness"].default_value = 0.25
        if masks is not None and "Coat Normal" in bsdf.inputs:
            nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Coat Normal"])
        nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])

        out = _new_node(nt, "ShaderNodeOutputMaterial", (680, 120))
        if ng_wrap_ts is not None:
            wrap = _new_node(nt, "ShaderNodeGroup", (420, 120))
            wrap.node_tree = ng_wrap_ts
            nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
            wrap.inputs["ID_Color"].default_value = rgba(id_hex)
            wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
            if masks is not None:
                nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
                nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])
            nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])
        else:
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        mat["f15_id_hex"] = int(id_hex)
        mat["f15_pbr_hex"] = int(hex_col)
        mat.diffuse_color = rgba(hex_col)
        return mat

    m_ts_yel = _build_torque_seal_mat("Mat_GSE_TorqueSeal_Yellow", 0xC6A21A, COLOR_ID_SPECS[MAT_TRAILER_YELLOW][1])
    m_ts_org = _build_torque_seal_mat("Mat_GSE_TorqueSeal_Orange", 0xC43E14, COLOR_ID_SPECS[MAT_STAND_ORANGE][1])

    bm_ts_eng = bmesh.new()
    bm_ts_st  = bmesh.new()


    def _emit_torque_seal_quad_bead(bm_q, host_obj, midx_q: int, p_start: Vector, p_end: Vector, n_hint: Vector,
                                    half_w: float = 0.0012, height: float = 0.00090, n_steps: int = 9, wobble_seed: float = 0.0):
        n_ref = n_hint.normalized()
        d_vec = p_end - p_start
        if d_vec.length < 1e-5:
            return
        t_dir = d_vec.normalized()
        s_dir = n_ref.cross(t_dir)
        if s_dir.length < 1e-5:
            return
        s_dir.normalize()

        spine_pts, spine_nrms = [], []
        for i in range(n_steps + 1):
            f = i / float(n_steps)
            wob = 0.00032 * math.sin(f * math.pi * 2.2 + wobble_seed) * math.sin(f * math.pi)
            p_nom = p_start.lerp(p_end, f) + s_dir * wob
            if host_obj is not None:

                ok, loc, nrm, _ = host_obj.ray_cast(p_nom + n_ref * 0.018, -n_ref, distance=0.036)
                if ok and (loc - p_nom).length <= 0.016:
                    if nrm.dot(n_ref) < 0.0:
                        nrm = -nrm
                    spine_pts.append(loc + n_ref * 0.00018)
                    spine_nrms.append((0.75 * n_ref + 0.25 * nrm.normalized()).normalized())
                else:
                    spine_pts.append(p_nom + n_ref * 0.00018)
                    spine_nrms.append(n_ref)
            else:
                spine_pts.append(p_nom + n_ref * 0.00018)
                spine_nrms.append(n_ref)

        prof_u = (-1.0, -0.45, +0.45, +1.0)
        prof_h = (0.0,   1.0,   1.0,   0.0)
        rings = []
        for i in range(n_steps + 1):
            f = i / float(n_steps)
            taper = max(0.22, math.sin(f * math.pi) ** 0.38)
            w_cur = half_w * taper
            h_cur = height * taper
            p_c = spine_pts[i]
            n_c = spine_nrms[i]
            if i == 0:
                tc_v = (spine_pts[1] - spine_pts[0]).normalized()
            elif i == n_steps:
                tc_v = (spine_pts[n_steps] - spine_pts[n_steps - 1]).normalized()
            else:
                tc_v = (spine_pts[i + 1] - spine_pts[i - 1]).normalized()
            sc_v = n_c.cross(tc_v)
            sc_v = s_dir if sc_v.length < 1e-6 else sc_v.normalized()
            rings.append([bm_q.verts.new(p_c + sc_v * (pu * w_cur) + n_c * (ph * h_cur)) for pu, ph in zip(prof_u, prof_h)])

        for i in range(n_steps):
            r0, r1 = rings[i], rings[i + 1]
            for j in range(4):
                jn = (j + 1) % 4
                try:
                    fq = bm_q.faces.new((r0[j], r1[j], r1[jn], r0[jn]))
                    fq.material_index = midx_q
                    fq.smooth = (j != 3)
                except ValueError:
                    pass
        try:
            f0 = bm_q.faces.new((rings[0][0], rings[0][1], rings[0][2], rings[0][3]))
            f0.material_index = midx_q
            f1 = bm_q.faces.new((rings[-1][3], rings[-1][2], rings[-1][1], rings[-1][0]))
            f1.material_index = midx_q
        except ValueError:
            pass


    o_arch3 = bpy.data.objects.get("Stand_Cradle_03_RollOver_Arches_And_MountPickups")
    if o_arch3 is not None:
        seed_i = 0
        for sx_a in (-1.0, +1.0):
            for y_ar in (-3.856, -2.333):
                n_ax = Vector((sx_a, 0.0, 0.0))
                for dy_db, mid_ts in ((-0.044, 0), (+0.044, 0)):
                    seed_i += 1
                    pb = Vector((sx_a * 0.766, y_ar + dy_db, +0.012))
                    _emit_torque_seal_quad_bead(bm_ts_st, o_arch3, mid_ts, pb + Vector((0, 0, -0.005)), pb + Vector((0, 0, +0.010)),
                                                n_ax, half_w=0.0014, height=0.0010, n_steps=8, wobble_seed=seed_i * 1.1)

                x_tr_ts = 0.793 if (sx_a > 0.0 and abs(y_ar - (-3.856)) < 0.02) else 0.772
                for ib_t, mid_ts in ((1, 0), (4, 1), (6, 0), (9, 1)):
                    seed_i += 1
                    ang_b = 2.0 * math.pi * ib_t / 10.0
                    u_rad = Vector((0.0, math.cos(ang_b), math.sin(ang_b)))
                    pb = Vector((sx_a * x_tr_ts, y_ar, -0.054)) + u_rad * 0.044
                    _emit_torque_seal_quad_bead(bm_ts_st, o_arch3, mid_ts, pb - u_rad * 0.003, pb + u_rad * 0.009,
                                                n_ax, half_w=0.00125, height=0.00090, n_steps=11, wobble_seed=seed_i * 0.9)


    if o_oil is not None:
        r_sg_t = 0.5540 - 0.0098 + 0.0096
        for k_s, (sx_sb, sy_sb, dir_y, mid_ts) in enumerate([
            (-0.035, -4.072, -1.0, 0), (+0.012, -4.072, -1.0, 0),
            (-0.012, -4.046, +1.0, 0), (+0.035, -4.046, +1.0, 0),
        ]):
            p0_l = Vector((sx_sb, sy_sb - dir_y * 0.0010, r_sg_t))
            p1_l = Vector((sx_sb, sy_sb + dir_y * 0.0022, r_sg_t))
            _emit_torque_seal_quad_bead(
                bm_ts_eng, o_oil, mid_ts, rot_sg_d @ p0_l, rot_sg_d @ p1_l,
                rot_sg_d3 @ Vector((0.0, 0.0, 1.0)), half_w=0.00072, height=0.00052, n_steps=8, wobble_seed=k_s * 1.4,
            )


    o_hyd = bpy.data.objects.get("Static_Hydraulic_System_Detailed")
    if o_hyd is not None:
        for k_ac in range(8):
            rot_a = Matrix.Rotation(math.radians(k_ac * 45.0), 4, 'Y')
            rot_a3 = rot_a.to_3x3()
            mid_ts = 0 if (k_ac % 2 == 0) else 1
            for side_ar in (-1.0, +1.0):
                p_b0 = rot_a @ Vector((side_ar * 0.0507, -0.741, 0.558))
                p_b1 = rot_a @ Vector((side_ar * 0.0507, -0.751, 0.558))
                _emit_torque_seal_quad_bead(bm_ts_eng, o_hyd, mid_ts, p_b0, p_b1, rot_a3 @ Vector((0, 0, 1)),
                                            half_w=0.00095, height=0.00072, n_steps=7, wobble_seed=k_ac * 1.7 + side_ar)


    o_f4 = bpy.data.objects.get("Fwd_04_Core_Left_SBleed_ServoValve_And_Tubes")
    if o_f4 is not None:
        rot_44 = Matrix.Rotation(math.radians(44.0), 4, 'Y')
        rot_44_3 = rot_44.to_3x3()
        for k_f4, (xl, yl, mid_ts) in enumerate([(-0.024, -3.445, 0), (+0.024, -3.375, 1), (-0.024, -3.375, 0)]):
            p0 = rot_44 @ Vector((xl, yl, 0.517))
            p1 = rot_44 @ Vector((xl + (0.008 if xl > 0 else -0.008), yl, 0.517))
            _emit_torque_seal_quad_bead(bm_ts_eng, o_f4, mid_ts, p0, p1, rot_44_3 @ Vector((0, 0, 1)),
                                        half_w=0.00095, height=0.00072, n_steps=7, wobble_seed=k_f4 * 2.1)

    o_f5 = bpy.data.objects.get("Fwd_05_Core_Right_TwinUBleed_RCVV_And_Valves")
    if o_f5 is not None:
        rot_241 = Matrix.Rotation(math.radians(241.0), 4, 'Y')
        rot_241_3 = rot_241.to_3x3()
        for k_f5, (xr, yr, mid_ts) in enumerate([(-0.018, -3.576, 1), (+0.018, -3.510, 0)]):
            p0 = rot_241 @ Vector((xr, yr, 0.558))
            p1 = rot_241 @ Vector((xr, yr + (0.011 if yr > -3.54 else -0.011), 0.554))
            _emit_torque_seal_quad_bead(bm_ts_eng, o_f5, mid_ts, p0, p1, rot_241_3 @ Vector((0, 0, 1)),
                                        half_w=0.0011, height=0.00080, n_steps=7, wobble_seed=k_f5 * 1.9)

    bmesh.ops.recalc_face_normals(bm_ts_eng, faces=bm_ts_eng.faces)
    _finalize_decal_mesh_obj("Decals_TorqueSeal_Engine", bm_ts_eng, [m_ts_yel, m_ts_org], coll_dec)
    bmesh.ops.recalc_face_normals(bm_ts_st, faces=bm_ts_st.faces)
    _finalize_decal_mesh_obj("Decals_TorqueSeal_Stand", bm_ts_st, [m_ts_yel, m_ts_org], coll_dec)

    print("[Conformal Decals] Generated decal, stencil, fluid-band, and torque-seal mesh groups")
