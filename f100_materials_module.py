import math
import os
import sys
import bpy
import bmesh
from mathutils import Vector, Euler, Matrix

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from f100_geom_primitives import (
    rgba,
    COLOR_ID_SPECS,
    MAT_TI_CASE, MAT_INCONEL_CASE, MAT_WAFFLE_DUCT, MAT_SPINNER_IGV,
    MAT_ROTARY_BLADES, MAT_MACHINED_ALU, MAT_AVIONICS_BOX, MAT_PUMPS_OILTANK,
    MAT_RIGID_TUBE, MAT_BRAID_HOSE, MAT_YELLOW_HARNESS, MAT_BRASS_FITTING,
    MAT_ADEL_CUSHION, MAT_CONNECTOR_38999, MAT_FASTENER, MAT_THREAD,
    MAT_LOCKWIRE, MAT_PLATE_BASE, MAT_3D_TEXT, MAT_SHROUD_GOLD,
    MAT_SHROUD_GREY, MAT_NOZ_EXT_LINK, MAT_NOZ_FLAP_BODY, MAT_NOZ_ACT_AB,
    MAT_HOTGAS_TILE, MAT_STAND_ORANGE, MAT_STAND_STEEL, MAT_TRAILER_YELLOW,
    MAT_RUBBER_DARK,
    attach_weighted_normal_modifier,
    bm_apply_hard_surface_smoothing,
)
from f100_edge_and_decals import (
    bake_all_mesh_geometry_attributes,
    refine_existing_special_face_materials,
    build_conformal_aviation_decals,
)


def _new_node(nt, bl_idname: str, loc=(0, 0), name: str = None, label: str = None):
    n = nt.nodes.new(bl_idname)
    n.location = loc
    if name:
        n.name = name
    if label:
        n.label = label
    return n


def _add_ng_input(ng, name: str, socket_type: str, default=None, min_val=None, max_val=None):
    sock = ng.interface.new_socket(name=name, in_out='INPUT', socket_type=socket_type)
    if default is not None and hasattr(sock, "default_value"):
        sock.default_value = default
    if min_val is not None and hasattr(sock, "min_value"):
        sock.min_value = min_val
    if max_val is not None and hasattr(sock, "max_value"):
        sock.max_value = max_val
    return sock


def _add_ng_output(ng, name: str, socket_type: str):
    return ng.interface.new_socket(name=name, in_out='OUTPUT', socket_type=socket_type)


def _fresh_node_group(name: str):
    old = bpy.data.node_groups.get(name)
    if old is not None:
        bpy.data.node_groups.remove(old, do_unlink=True)
    return bpy.data.node_groups.new(name=name, type='ShaderNodeTree')


def _math(nt, op: str, loc=(0, 0), clamp: bool = False, v0=None, v1=None, v2=None):
    n = _new_node(nt, "ShaderNodeMath", loc)
    n.operation = op
    n.use_clamp = clamp
    if v0 is not None:
        n.inputs[0].default_value = v0
    if v1 is not None and len(n.inputs) > 1:
        n.inputs[1].default_value = v1
    if v2 is not None and len(n.inputs) > 2:
        n.inputs[2].default_value = v2
    return n


def _combine_scalar_heights(nt, terms: list, base_normal, *, ref_distance: float = 0.0004, loc=(600, -820)):

    valid = []
    for sock, strength, dist in terms:
        if sock is not None and abs(float(strength)) > 1e-6 and float(dist) > 1e-7:
            w = float(strength) * (float(dist) / float(ref_distance))
            if abs(w) > 1e-6:
                valid.append((sock, w))
    if not valid:
        return base_normal
    acc = None
    for idx, (sock, w) in enumerate(valid):
        if acc is None:
            m = _math(nt, 'MULTIPLY', (loc[0] - 280, loc[1] - idx * 55), v1=w)
            nt.links.new(sock, m.inputs[0])
            acc = m.outputs[0]
        else:
            ma = _math(nt, 'MULTIPLY_ADD', (loc[0] - 220 + min(idx, 5) * 28, loc[1] - idx * 55), v1=w)
            nt.links.new(sock, ma.inputs[0])
            nt.links.new(acc, ma.inputs[2])
            acc = ma.outputs[0]
    b = _new_node(nt, "ShaderNodeBump", loc)
    b.inputs["Strength"].default_value = 1.0
    b.inputs["Distance"].default_value = float(ref_distance)
    nt.links.new(acc, b.inputs["Height"])
    if base_normal is not None:
        nt.links.new(base_normal, b.inputs["Normal"])
    return b.outputs["Normal"]


def _vmath(nt, op: str, loc=(0, 0)):
    n = _new_node(nt, "ShaderNodeVectorMath", loc)
    n.operation = op
    return n


def _mix_rgba(nt, blend_type='MIX', loc=(0, 0), fac=0.5, col_a=None, col_b=None, clamp_fac=True):
    n = _new_node(nt, "ShaderNodeMix", loc)
    n.data_type = 'RGBA'
    n.blend_type = blend_type
    n.clamp_factor = clamp_fac
    n.inputs[0].default_value = fac
    if col_a is not None:
        n.inputs[6].default_value = col_a
    if col_b is not None:
        n.inputs[7].default_value = col_b
    return n


def _mix_float(nt, loc=(0, 0), fac=0.5, a=0.0, b=1.0, clamp_fac=True):
    n = _new_node(nt, "ShaderNodeMix", loc)
    n.data_type = 'FLOAT'
    n.clamp_factor = clamp_fac
    n.inputs[0].default_value = fac
    n.inputs[2].default_value = a
    n.inputs[3].default_value = b
    return n


def _map_range(nt, fmin, fmax, tmin, tmax, loc=(0, 0), clamp=True, interp='SMOOTHSTEP'):
    n = _new_node(nt, "ShaderNodeMapRange", loc)
    n.clamp = clamp
    n.interpolation_type = interp
    n.inputs[1].default_value = fmin
    n.inputs[2].default_value = fmax
    n.inputs[3].default_value = tmin
    n.inputs[4].default_value = tmax
    return n


def _ramp(nt, stops: list, loc=(0, 0), interp='LINEAR'):
    n = _new_node(nt, "ShaderNodeValToRGB", loc)
    cr = n.color_ramp
    cr.interpolation = interp

    while len(cr.elements) > 2:
        cr.elements.remove(cr.elements[-1])
    cr.elements[0].position = stops[0][0]
    cr.elements[0].color = stops[0][1]
    cr.elements[1].position = stops[-1][0]
    cr.elements[1].color = stops[-1][1]
    for pos, col in stops[1:-1]:
        el = cr.elements.new(pos)
        el.color = col
    return n


def _noise(nt, scale=10.0, detail=6.0, roughness=0.55, distortion=0.0, loc=(0, 0), dim='3D'):
    n = _new_node(nt, "ShaderNodeTexNoise", loc)
    n.noise_dimensions = dim
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = roughness
    n.inputs["Distortion"].default_value = distortion
    return n


def _voronoi(nt, scale=20.0, feature='F1', randomness=1.0, loc=(0, 0)):
    n = _new_node(nt, "ShaderNodeTexVoronoi", loc)
    n.feature = feature
    n.inputs["Scale"].default_value = scale
    n.inputs["Randomness"].default_value = randomness
    return n


def build_ng_global_controls():

    ng = _fresh_node_group("NG_F15_GlobalControls")
    _add_ng_output(ng, "Show_PBR",     'NodeSocketFloat')
    _add_ng_output(ng, "Show_ColorID", 'NodeSocketFloat')
    _add_ng_output(ng, "Weathering",   'NodeSocketFloat')
    _add_ng_output(ng, "HeatTint",     'NodeSocketFloat')
    _add_ng_output(ng, "Soot",         'NodeSocketFloat')
    _add_ng_output(ng, "EdgeWear",     'NodeSocketFloat')
    _add_ng_output(ng, "SurfaceErosion", 'NodeSocketFloat')
    _add_ng_output(ng, "Show_Decals",  'NodeSocketFloat')

    out = _new_node(ng, "NodeGroupOutput", (320, 0))

    specs = [
        ("Val_Show_PBR",      "Show_PBR",      1.0,  260),
        ("Val_Show_ColorID",  "Show_ColorID",  1.0,  180),
        ("Val_Weathering",    "Weathering",    0.95, 100),
        ("Val_HeatTint",      "HeatTint",      1.0,   20),
        ("Val_Soot",          "Soot",          1.0,  -60),
        ("Val_EdgeWear",      "EdgeWear",      0.90,-140),
        ("Val_SurfaceErosion",  "SurfaceErosion",  0.45,-220),
        ("Val_Show_Decals",   "Show_Decals",   1.0, -300),
    ]
    for vname, out_name, val, py in specs:
        vn = _new_node(ng, "ShaderNodeValue", (-100, py), name=vname, label=out_name)
        vn.outputs[0].default_value = val
        ng.links.new(vn.outputs[0], out.inputs[out_name])
    return ng


def _load_and_pack_mask_4k(filename: str, img_name: str):

    path = os.path.join(_THIS_DIR, "assets_masks_4k", filename)
    if not os.path.exists(path):
        print(f"[Notice] Surface mask not found: {path}; using procedural fallback")
        return None
    img = bpy.data.images.get(img_name)
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.load(path, check_existing=False)
    img.name = img_name
    try:
        img.colorspace_settings.name = 'Non-Color'
    except Exception:
        pass
    try:
        img.pack()
    except Exception as e:
        print(f"[Warning] Failed to pack surface mask ({img_name}): {e}")
    return img


def build_ng_surface_masks():

    ng = _fresh_node_group("NG_F15_SurfaceMasks")
    _add_ng_input(ng, "Coord",          'NodeSocketVector')
    _add_ng_input(ng, "AO_Dist",        'NodeSocketFloat', 0.048, 0.002, 0.5)
    _add_ng_input(ng, "Fine_Bevel_Rad", 'NodeSocketFloat', 0.00034, 0.0, 0.01)
    _add_ng_input(ng, "Edge_Bevel_Rad", 'NodeSocketFloat', 0.0042, 0.0002, 0.05)
    _add_ng_input(ng, "Edge_Gain",      'NodeSocketFloat', 1.85, 0.0, 12.0)
    _add_ng_input(ng, "Grime_Scale",    'NodeSocketFloat', 14.0, 0.5, 400.0)
    _add_ng_input(ng, "Macro_Scale",    'NodeSocketFloat', 3.8, 0.2, 100.0)
    _add_ng_input(ng, "Micro_Scale",    'NodeSocketFloat', 180.0, 1.0, 3000.0)

    _add_ng_output(ng, "Bevel_Normal",  'NodeSocketVector')
    _add_ng_output(ng, "Crevice_Mask",  'NodeSocketFloat')
    _add_ng_output(ng, "EdgeWear_Mask", 'NodeSocketFloat')
    _add_ng_output(ng, "Macro_Mottle",  'NodeSocketFloat')
    _add_ng_output(ng, "Micro_Grain",   'NodeSocketFloat')
    _add_ng_output(ng, "AO_Raw",        'NodeSocketFloat')
    _add_ng_output(ng, "Island_Rand",   'NodeSocketFloat')
    _add_ng_output(ng, "Topo_Convex",   'NodeSocketFloat')
    _add_ng_output(ng, "Topo_Concave",  'NodeSocketFloat')
    _add_ng_output(ng, "Exposure_Mask", 'NodeSocketFloat')
    _add_ng_output(ng, "Scratch_Mask",      'NodeSocketFloat')
    _add_ng_output(ng, "OilLeak_Mask",      'NodeSocketFloat')
    _add_ng_output(ng, "Photo_Smudge",      'NodeSocketFloat')
    _add_ng_output(ng, "Photo_Chip",        'NodeSocketFloat')
    _add_ng_output(ng, "Photo_Airflow",     'NodeSocketFloat')
    _add_ng_output(ng, "Photo_Fingerprint", 'NodeSocketFloat')
    _add_ng_output(ng, "Metal_Rough_Breakup", 'NodeSocketFloat')
    _add_ng_output(ng, "Metal_Brush_Grain",   'NodeSocketFloat')
    _add_ng_output(ng, "Galv_Crystal_RGB",    'NodeSocketColor')


    img_sc_fine   = _load_and_pack_mask_4k("mask_4k_scratches_fine.jpg",    "IMG_Mask4K_ScratchesFine")
    img_sc_heavy  = _load_and_pack_mask_4k("mask_4k_scratches_heavy.jpg",   "IMG_Mask4K_ScratchesHeavy")
    img_smudge    = _load_and_pack_mask_4k("mask_4k_smudges_stains.jpg",    "IMG_Mask4K_SmudgesStains")
    img_drip      = _load_and_pack_mask_4k("mask_4k_drip_streaks.jpg",      "IMG_Mask4K_DripStreaks")
    img_chip      = _load_and_pack_mask_4k("mask_4k_paint_chips.jpg",       "IMG_Mask4K_PaintChips")
    img_airflow   = _load_and_pack_mask_4k("mask_4k_airflow_grime.jpg",     "IMG_Mask4K_AirflowGrime")
    img_fprint    = _load_and_pack_mask_4k("mask_4k_fingerprints.jpg",      "IMG_Mask4K_Fingerprints")
    img_m_rbrk    = _load_and_pack_mask_4k("metal_2k_roughness_breakup.png","IMG_Metal2K_RoughBreakup")
    img_m_brush   = _load_and_pack_mask_4k("metal_2k_aniso_grain.png",      "IMG_Metal2K_AnisoGrain")
    img_m_galv    = _load_and_pack_mask_4k("metal_2k_galvanized_crystal.png","IMG_Metal2K_GalvCrystal")

    def _box_antitile(img, coord_sock, scale1, scale2, rot1, rot2, blend_fac_sock, loc_y, blend_mode='MIX', box_blend=0.26):

        mp1 = _new_node(ng, "ShaderNodeMapping", (-560, loc_y))
        mp1.inputs["Scale"].default_value = scale1 if isinstance(scale1, tuple) else (scale1, scale1, scale1)
        mp1.inputs["Rotation"].default_value = rot1
        ng.links.new(coord_sock, mp1.inputs["Vector"])
        t1 = _new_node(ng, "ShaderNodeTexImage", (-320, loc_y))
        t1.image = img; t1.projection = 'BOX'; t1.projection_blend = box_blend
        ng.links.new(mp1.outputs["Vector"], t1.inputs["Vector"])

        mp2 = _new_node(ng, "ShaderNodeMapping", (-560, loc_y - 160))
        mp2.inputs["Scale"].default_value = scale2 if isinstance(scale2, tuple) else (scale2, scale2, scale2)
        mp2.inputs["Rotation"].default_value = rot2
        ng.links.new(coord_sock, mp2.inputs["Vector"])
        t2 = _new_node(ng, "ShaderNodeTexImage", (-320, loc_y - 160))
        t2.image = img; t2.projection = 'BOX'; t2.projection_blend = box_blend
        ng.links.new(mp2.outputs["Vector"], t2.inputs["Vector"])

        if blend_mode == 'MAXIMUM':
            w2 = _math(ng, 'MULTIPLY', (-80, loc_y - 160), clamp=True, v1=0.68)
            ng.links.new(t2.outputs["Color"], w2.inputs[0])
            comb = _math(ng, 'MAXIMUM', (100, loc_y - 80), clamp=True)
            ng.links.new(t1.outputs["Color"], comb.inputs[0])
            ng.links.new(w2.outputs[0], comb.inputs[1])
            return comb.outputs[0]
        else:
            mx = _mix_float(ng, (100, loc_y - 80))
            ng.links.new(blend_fac_sock, mx.inputs[0])
            ng.links.new(t1.outputs["Color"], mx.inputs[2])
            ng.links.new(t2.outputs["Color"], mx.inputs[3])
            return mx.outputs[0]

    inp = _new_node(ng, "NodeGroupInput", (-1280, 0))
    out = _new_node(ng, "NodeGroupOutput", (1180, 0))

    geo = _new_node(ng, "ShaderNodeNewGeometry", (-1060, 420))
    obj_info = _new_node(ng, "ShaderNodeObjectInfo", (-1060, 180))


    attr_conv = _new_node(ng, "ShaderNodeAttribute", (-1060, -140), name="Attr_Convexity")
    attr_conv.attribute_name = "f15_convexity"
    attr_conc = _new_node(ng, "ShaderNodeAttribute", (-1060, -280), name="Attr_Concavity")
    attr_conc.attribute_name = "f15_concavity"
    attr_exp  = _new_node(ng, "ShaderNodeAttribute", (-1060, -420), name="Attr_Exposure")
    attr_exp.attribute_name = "f15_exposure"

    ng.links.new(attr_conv.outputs["Fac"], out.inputs["Topo_Convex"])
    ng.links.new(attr_conc.outputs["Fac"], out.inputs["Topo_Concave"])
    ng.links.new(attr_exp.outputs["Fac"], out.inputs["Exposure_Mask"])


    add_r = _math(ng, 'ADD', (-820, 260))
    ng.links.new(geo.outputs["Random Per Island"], add_r.inputs[0])
    ng.links.new(obj_info.outputs["Random"], add_r.inputs[1])
    frac_r = _math(ng, 'FRACT', (-640, 260))
    ng.links.new(add_r.outputs[0], frac_r.inputs[0])
    ng.links.new(frac_r.outputs[0], out.inputs["Island_Rand"])


    comb_off = _new_node(ng, "ShaderNodeCombineXYZ", (-820, 40))
    m_ox = _math(ng, 'MULTIPLY', (-980, 40), v1=7.31)
    m_oy = _math(ng, 'MULTIPLY', (-980, -30), v1=13.17)
    m_oz = _math(ng, 'MULTIPLY', (-980, -100), v1=5.89)
    ng.links.new(frac_r.outputs[0], m_ox.inputs[0])
    ng.links.new(frac_r.outputs[0], m_oy.inputs[0])
    ng.links.new(frac_r.outputs[0], m_oz.inputs[0])
    ng.links.new(m_ox.outputs[0], comb_off.inputs["X"])
    ng.links.new(m_oy.outputs[0], comb_off.inputs["Y"])
    ng.links.new(m_oz.outputs[0], comb_off.inputs["Z"])

    coord_jit = _vmath(ng, 'ADD', (-600, -20))
    ng.links.new(inp.outputs["Coord"], coord_jit.inputs[0])
    ng.links.new(comb_off.outputs["Vector"], coord_jit.inputs[1])


    bev_fine = _new_node(ng, "ShaderNodeBevel", (-520, 560))
    bev_fine.samples = 4
    ng.links.new(inp.outputs["Fine_Bevel_Rad"], bev_fine.inputs["Radius"])
    ng.links.new(bev_fine.outputs["Normal"], out.inputs["Bevel_Normal"])


    ao = _new_node(ng, "ShaderNodeAmbientOcclusion", (-520, 380))
    ao.samples = 8
    ao.only_local = False
    ng.links.new(inp.outputs["AO_Dist"], ao.inputs["Distance"])
    ng.links.new(ao.outputs["AO"], out.inputs["AO_Raw"])

    inv_ao = _map_range(ng, 0.14, 0.90, 1.0, 0.0, (-260, 400), clamp=True)
    ng.links.new(ao.outputs["AO"], inv_ao.inputs[0])

    conc_w = _math(ng, 'MULTIPLY', (-260, 260), clamp=True, v1=0.76)
    ng.links.new(attr_conc.outputs["Fac"], conc_w.inputs[0])
    crev_geo = _math(ng, 'MAXIMUM', (-40, 360), clamp=True)
    ng.links.new(inv_ao.outputs[0], crev_geo.inputs[0])
    ng.links.new(conc_w.outputs[0], crev_geo.inputs[1])


    bev_edge = _new_node(ng, "ShaderNodeBevel", (-520, 160))
    bev_edge.samples = 8
    ng.links.new(inp.outputs["Edge_Bevel_Rad"], bev_edge.inputs["Radius"])

    dot_n = _vmath(ng, 'DOT_PRODUCT', (-300, 160))
    ng.links.new(geo.outputs["Normal"], dot_n.inputs[0])
    ng.links.new(bev_edge.outputs["Normal"], dot_n.inputs[1])

    one_minus_dot = _math(ng, 'SUBTRACT', (-120, 160), clamp=True, v0=1.0)
    ng.links.new(dot_n.outputs["Value"], one_minus_dot.inputs[1])


    bev_sig = _map_range(ng, 0.005, 0.16, 0.0, 0.72, (0, 160), clamp=True)
    bev_sig.interpolation_type = 'SMOOTHSTEP'
    ng.links.new(one_minus_dot.outputs[0], bev_sig.inputs[0])


    convex_ao = _map_range(ng, 0.36, 0.74, 0.0, 1.0, (-120, 40), clamp=True)
    ng.links.new(ao.outputs["AO"], convex_ao.inputs[0])
    not_conc = _map_range(ng, 0.12, 0.55, 1.0, 0.0, (-120, -80), clamp=True)
    ng.links.new(attr_conc.outputs["Fac"], not_conc.inputs[0])
    convex_gate = _math(ng, 'MULTIPLY', (40, 20), clamp=True)
    ng.links.new(convex_ao.outputs[0], convex_gate.inputs[0])
    ng.links.new(not_conc.outputs[0], convex_gate.inputs[1])

    edge_raw = _math(ng, 'MULTIPLY', (160, 140), clamp=True)
    ng.links.new(bev_sig.outputs[0], edge_raw.inputs[0])
    ng.links.new(convex_gate.outputs[0], edge_raw.inputs[1])


    edge_fused = _math(ng, 'MULTIPLY', (360, 120), clamp=True)
    ng.links.new(edge_raw.outputs[0], edge_fused.inputs[0])
    ng.links.new(inp.outputs["Edge_Gain"], edge_fused.inputs[1])


    n_chip_scale = _math(ng, 'MULTIPLY', (-300, -180), v1=3.4)
    ng.links.new(inp.outputs["Grime_Scale"], n_chip_scale.inputs[0])
    n_chip = _noise(ng, scale=46.0, detail=8.0, roughness=0.70, distortion=0.30, loc=(-80, -180))
    ng.links.new(coord_jit.outputs["Vector"], n_chip.inputs["Vector"])
    ng.links.new(n_chip_scale.outputs[0], n_chip.inputs["Scale"])

    v_fl_scale = _math(ng, 'MULTIPLY', (-300, -340), v1=4.2)
    ng.links.new(inp.outputs["Grime_Scale"], v_fl_scale.inputs[0])
    vor_fl = _voronoi(ng, scale=58.0, feature='F1', randomness=0.92, loc=(-80, -340))
    ng.links.new(coord_jit.outputs["Vector"], vor_fl.inputs["Vector"])
    ng.links.new(v_fl_scale.outputs[0], vor_fl.inputs["Scale"])


    n_at_warp = _noise(ng, scale=1.35, detail=3.0, roughness=0.52, loc=(-820, -480))
    ng.links.new(coord_jit.outputs["Vector"], n_at_warp.inputs["Vector"])
    coord_at = _mix_rgba(ng, 'MIX', (-640, -480), fac=0.042)
    ng.links.new(coord_jit.outputs["Vector"], coord_at.inputs[6])
    ng.links.new(n_at_warp.outputs["Color"], coord_at.inputs[7])
    at_blend = _map_range(ng, 0.30, 0.70, 0.0, 1.0, (-640, -640), clamp=True)
    ng.links.new(n_at_warp.outputs["Factor"], at_blend.inputs[0])


    flake_mix0 = _mix_float(ng, (120, -240), fac=0.34)
    ng.links.new(n_chip.outputs["Factor"], flake_mix0.inputs[2])
    ng.links.new(vor_fl.outputs["Distance"], flake_mix0.inputs[3])

    photo_smudge_out = None
    if img_smudge is not None:
        photo_smudge_out = _box_antitile(
            img_smudge, coord_at.outputs[2],
            1.42, 3.18, (0.18, -0.26, 0.42), (-0.72, 0.54, -0.88),
            at_blend.outputs[0], loc_y=-460, blend_mode='MIX', box_blend=0.28,
        )
        ng.links.new(photo_smudge_out, out.inputs["Photo_Smudge"])
    else:
        out.inputs["Photo_Smudge"].default_value = 0.0

    photo_chip_out = None
    if img_chip is not None:
        photo_chip_out = _box_antitile(
            img_chip, coord_at.outputs[2],
            1.75, 4.10, (0.28, -0.44, 0.19), (-0.82, 0.36, 0.74),
            at_blend.outputs[0], loc_y=-2120, blend_mode='MAXIMUM', box_blend=0.22,
        )
        ng.links.new(photo_chip_out, out.inputs["Photo_Chip"])
    else:
        out.inputs["Photo_Chip"].default_value = 0.0

    if img_airflow is not None:

        photo_af_out = _box_antitile(
            img_airflow, coord_at.outputs[2],
            (2.15, 0.72, 2.15), (4.65, 1.55, 4.65),
            (0.0, 0.0, 0.14), (0.0, 0.0, -0.22),
            at_blend.outputs[0], loc_y=-2460, blend_mode='MIX', box_blend=0.26,
        )
        ng.links.new(photo_af_out, out.inputs["Photo_Airflow"])
    else:
        out.inputs["Photo_Airflow"].default_value = 0.0

    if img_fprint is not None:
        photo_fp_out = _box_antitile(
            img_fprint, coord_at.outputs[2],
            3.10, 5.40, (0.34, -0.28, 0.62), (-0.64, 0.85, -0.40),
            at_blend.outputs[0], loc_y=-2800, blend_mode='MIX', box_blend=0.24,
        )
        ng.links.new(photo_fp_out, out.inputs["Photo_Fingerprint"])
    else:
        out.inputs["Photo_Fingerprint"].default_value = 0.0


    if img_m_rbrk is not None:
        m_rbrk_out = _box_antitile(
            img_m_rbrk, coord_at.outputs[2],
            2.40, 5.20, (0.21, -0.34, 0.48), (-0.77, 0.62, -0.35),
            at_blend.outputs[0], loc_y=-3140, blend_mode='MIX', box_blend=0.28,
        )
        ng.links.new(m_rbrk_out, out.inputs["Metal_Rough_Breakup"])
    else:
        out.inputs["Metal_Rough_Breakup"].default_value = 0.5

    if img_m_brush is not None:
        mp_mb = _new_node(ng, "ShaderNodeMapping", (-560, -3480))
        mp_mb.inputs["Scale"].default_value = (2.8, 2.8, 2.8)
        ng.links.new(coord_jit.outputs["Vector"], mp_mb.inputs["Vector"])
        t_mb = _new_node(ng, "ShaderNodeTexImage", (-320, -3480))
        t_mb.image = img_m_brush
        t_mb.projection = 'BOX'
        t_mb.projection_blend = 0.16
        ng.links.new(mp_mb.outputs["Vector"], t_mb.inputs["Vector"])
        ng.links.new(t_mb.outputs["Color"], out.inputs["Metal_Brush_Grain"])
    else:
        out.inputs["Metal_Brush_Grain"].default_value = 0.5

    if img_m_galv is not None:
        mp_gv = _new_node(ng, "ShaderNodeMapping", (-560, -3720))
        mp_gv.inputs["Scale"].default_value = (7.5, 7.5, 7.5)
        ng.links.new(coord_jit.outputs["Vector"], mp_gv.inputs["Vector"])
        t_gv = _new_node(ng, "ShaderNodeTexImage", (-320, -3720))
        t_gv.image = img_m_galv
        t_gv.projection = 'BOX'
        t_gv.projection_blend = 0.12
        ng.links.new(mp_gv.outputs["Vector"], t_gv.inputs["Vector"])
        ng.links.new(t_gv.outputs["Color"], out.inputs["Galv_Crystal_RGB"])
    else:
        out.inputs["Galv_Crystal_RGB"].default_value = (0.5, 0.5, 0.5, 1.0)


    flake_cur = flake_mix0.outputs[0]
    if photo_smudge_out is not None:
        fm_s = _mix_float(ng, (220, -240), fac=0.16)
        ng.links.new(flake_cur, fm_s.inputs[2])
        ng.links.new(photo_smudge_out, fm_s.inputs[3])
        flake_cur = fm_s.outputs[0]
    if photo_chip_out is not None:
        chip_lift = _map_range(ng, 0.05, 0.78, 0.18, 1.0, (240, -340), clamp=True)
        ng.links.new(photo_chip_out, chip_lift.inputs[0])
        fm_c = _mix_float(ng, (300, -240), fac=0.38)
        ng.links.new(flake_cur, fm_c.inputs[2])
        ng.links.new(chip_lift.outputs[0], fm_c.inputs[3])
        flake_cur = fm_c.outputs[0]


    edge_env = _map_range(ng, 0.015, 0.26, 0.0, 1.0, (540, -60), clamp=True)
    edge_env.interpolation_type = 'SMOOTHSTEP'
    ng.links.new(edge_fused.outputs[0], edge_env.inputs[0])

    flake_off = _map_range(ng, 0.18, 0.84, -0.44, +0.38, (420, -220), clamp=True)
    ng.links.new(flake_cur, flake_off.inputs[0])

    flake_env = _math(ng, 'MULTIPLY', (620, -160), clamp=False)
    ng.links.new(flake_off.outputs[0], flake_env.inputs[0])
    ng.links.new(edge_env.outputs[0], flake_env.inputs[1])

    edge_m1 = _math(ng, 'ADD', (680, 60), clamp=True)
    ng.links.new(edge_fused.outputs[0], edge_m1.inputs[0])
    ng.links.new(flake_env.outputs[0], edge_m1.inputs[1])


    exp_boost = _math(ng, 'MULTIPLY_ADD', (440, -380), v1=0.68, v2=0.62)
    ng.links.new(attr_exp.outputs["Fac"], exp_boost.inputs[0])

    edge_m2 = _math(ng, 'MULTIPLY', (820, 60), clamp=True)
    ng.links.new(edge_m1.outputs[0], edge_m2.inputs[0])
    ng.links.new(exp_boost.outputs[0], edge_m2.inputs[1])

    edge_crisp = _map_range(ng, 0.20, 0.62, 0.0, 1.0, (980, 60), clamp=True)
    ng.links.new(edge_m2.outputs[0], edge_crisp.inputs[0])
    ng.links.new(edge_crisp.outputs[0], out.inputs["EdgeWear_Mask"])


    n_grime = _noise(ng, scale=14.0, detail=7.0, roughness=0.62, distortion=0.35, loc=(-40, 520))
    ng.links.new(coord_jit.outputs["Vector"], n_grime.inputs["Vector"])
    ng.links.new(inp.outputs["Grime_Scale"], n_grime.inputs["Scale"])

    grime_mod = _map_range(ng, 0.22, 0.80, 0.45, 1.35, (220, 440), clamp=True)
    ng.links.new(n_grime.outputs["Factor"], grime_mod.inputs[0])

    crev_mul = _math(ng, 'MULTIPLY', (520, 380), clamp=True)
    ng.links.new(crev_geo.outputs[0], crev_mul.inputs[0])
    ng.links.new(grime_mod.outputs[0], crev_mul.inputs[1])
    ng.links.new(crev_mul.outputs[0], out.inputs["Crevice_Mask"])


    n_macro = _noise(ng, scale=3.8, detail=5.0, roughness=0.58, distortion=0.45, loc=(-40, -540))
    ng.links.new(coord_jit.outputs["Vector"], n_macro.inputs["Vector"])
    ng.links.new(inp.outputs["Macro_Scale"], n_macro.inputs["Scale"])
    ng.links.new(n_macro.outputs["Factor"], out.inputs["Macro_Mottle"])

    n_micro = _noise(ng, scale=180.0, detail=9.0, roughness=0.65, distortion=0.0, loc=(-40, -700))
    ng.links.new(coord_jit.outputs["Vector"], n_micro.inputs["Vector"])
    ng.links.new(inp.outputs["Micro_Scale"], n_micro.inputs["Scale"])
    ng.links.new(n_micro.outputs["Factor"], out.inputs["Micro_Grain"])


    n_swarp = _noise(ng, scale=3.8, detail=3.0, roughness=0.50, loc=(-760, -860))
    ng.links.new(coord_jit.outputs["Vector"], n_swarp.inputs["Vector"])
    v_warp = _mix_rgba(ng, 'MIX', (-540, -860), fac=0.022)
    ng.links.new(coord_jit.outputs["Vector"], v_warp.inputs[6])
    ng.links.new(n_swarp.outputs["Color"], v_warp.inputs[7])

    map_sc1 = _new_node(ng, "ShaderNodeMapping", (-340, -820))
    map_sc1.inputs["Scale"].default_value = (2.0, 92.0, 86.0)
    map_sc1.inputs["Rotation"].default_value = (0.24, -0.35, 0.28)
    ng.links.new(v_warp.outputs[2], map_sc1.inputs["Vector"])
    n_sc1 = _noise(ng, scale=1.5, detail=6.0, roughness=0.56, distortion=0.06, loc=(-120, -820))
    ng.links.new(map_sc1.outputs["Vector"], n_sc1.inputs["Vector"])
    sc_line1 = _map_range(ng, 0.77, 0.85, 0.0, 1.0, (100, -820), clamp=True, interp='LINEAR')
    ng.links.new(n_sc1.outputs["Factor"], sc_line1.inputs[0])

    map_sc2 = _new_node(ng, "ShaderNodeMapping", (-340, -1040))
    map_sc2.inputs["Scale"].default_value = (84.0, 2.4, 78.0)
    map_sc2.inputs["Rotation"].default_value = (-0.42, 0.28, -0.58)
    ng.links.new(v_warp.outputs[2], map_sc2.inputs["Vector"])
    n_sc2 = _noise(ng, scale=1.7, detail=6.0, roughness=0.56, distortion=0.06, loc=(-120, -1040))
    ng.links.new(map_sc2.outputs["Vector"], n_sc2.inputs["Vector"])
    sc_line2 = _map_range(ng, 0.78, 0.86, 0.0, 0.88, (100, -1040), clamp=True, interp='LINEAR')
    ng.links.new(n_sc2.outputs["Factor"], sc_line2.inputs[0])

    sc_lines_proc = _math(ng, 'MAXIMUM', (320, -920), clamp=True)
    ng.links.new(sc_line1.outputs[0], sc_lines_proc.inputs[0])
    ng.links.new(sc_line2.outputs[0], sc_lines_proc.inputs[1])


    sc_src_sig = sc_lines_proc.outputs[0]
    if img_sc_fine is not None:
        mp_sf1 = _new_node(ng, "ShaderNodeMapping", (-560, -1180))
        mp_sf1.inputs["Scale"].default_value = (1.85, 1.85, 1.85)
        mp_sf1.inputs["Rotation"].default_value = (0.12, 0.22, -0.18)
        ng.links.new(coord_jit.outputs["Vector"], mp_sf1.inputs["Vector"])
        t_sf1 = _new_node(ng, "ShaderNodeTexImage", (-320, -1180))
        t_sf1.image = img_sc_fine
        t_sf1.projection = 'BOX'
        t_sf1.projection_blend = 0.24
        ng.links.new(mp_sf1.outputs["Vector"], t_sf1.inputs["Vector"])

        mp_sf2 = _new_node(ng, "ShaderNodeMapping", (-560, -1380))
        mp_sf2.inputs["Scale"].default_value = (4.4, 4.4, 4.4)
        mp_sf2.inputs["Rotation"].default_value = (0.64, -0.92, 0.51)
        ng.links.new(coord_jit.outputs["Vector"], mp_sf2.inputs["Vector"])
        t_sf2 = _new_node(ng, "ShaderNodeTexImage", (-320, -1380))
        t_sf2.image = img_sc_fine
        t_sf2.projection = 'BOX'
        t_sf2.projection_blend = 0.24
        ng.links.new(mp_sf2.outputs["Vector"], t_sf2.inputs["Vector"])

        sf_w2 = _math(ng, 'MULTIPLY', (-60, -1380), clamp=True, v1=0.62)
        ng.links.new(t_sf2.outputs["Color"], sf_w2.inputs[0])
        sf_comb = _math(ng, 'MAXIMUM', (140, -1240), clamp=True)
        ng.links.new(t_sf1.outputs["Color"], sf_comb.inputs[0])
        ng.links.new(sf_w2.outputs[0], sf_comb.inputs[1])

        sf_g = _math(ng, 'MULTIPLY', (320, -1140), clamp=True, v1=0.92)
        ng.links.new(sf_comb.outputs[0], sf_g.inputs[0])
        sc_m_f = _math(ng, 'MAXIMUM', (440, -960), clamp=True)
        ng.links.new(sc_src_sig, sc_m_f.inputs[0])
        ng.links.new(sf_g.outputs[0], sc_m_f.inputs[1])
        sc_src_sig = sc_m_f.outputs[0]

    if img_sc_heavy is not None:
        shv_at = _box_antitile(
            img_sc_heavy, coord_at.outputs[2],
            1.42, 3.28, (-0.31, 0.44, 0.27), (0.86, -0.52, -0.74),
            at_blend.outputs[0], loc_y=-1580, blend_mode='MAXIMUM', box_blend=0.24,
        )
        shv_g = _math(ng, 'MULTIPLY', (320, -1320), clamp=True, v1=1.12)
        ng.links.new(shv_at, shv_g.inputs[0])
        sc_m_h = _math(ng, 'MAXIMUM', (500, -1040), clamp=True)
        ng.links.new(sc_src_sig, sc_m_h.inputs[0])
        ng.links.new(shv_g.outputs[0], sc_m_h.inputs[1])
        sc_src_sig = sc_m_h.outputs[0]


    n_sclust = _noise(ng, scale=5.2, detail=5.0, roughness=0.62, distortion=0.30, loc=(-120, -1480))
    ng.links.new(coord_jit.outputs["Vector"], n_sclust.inputs["Vector"])
    sc_clust = _map_range(ng, 0.44, 0.76, 0.0, 1.0, (120, -1480), clamp=True)
    ng.links.new(n_sclust.outputs["Factor"], sc_clust.inputs[0])

    sc_w0 = _math(ng, 'MULTIPLY_ADD', (120, -1620), v1=0.66, v2=0.26)
    ng.links.new(attr_conv.outputs["Fac"], sc_w0.inputs[0])
    sc_w1 = _math(ng, 'MULTIPLY_ADD', (320, -1620), clamp=True, v1=0.78)
    ng.links.new(attr_exp.outputs["Fac"], sc_w1.inputs[0])
    ng.links.new(sc_w0.outputs[0], sc_w1.inputs[2])

    sc_g1 = _math(ng, 'MULTIPLY', (640, -1000), clamp=True)
    ng.links.new(sc_src_sig, sc_g1.inputs[0])
    ng.links.new(sc_clust.outputs[0], sc_g1.inputs[1])

    sc_final = _math(ng, 'MULTIPLY', (820, -1000), clamp=True)
    ng.links.new(sc_g1.outputs[0], sc_final.inputs[0])
    ng.links.new(sc_w1.outputs[0], sc_final.inputs[1])
    ng.links.new(sc_final.outputs[0], out.inputs["Scratch_Mask"])


    map_oil = _new_node(ng, "ShaderNodeMapping", (-760, -1780))
    map_oil.inputs["Scale"].default_value = (15.0, 15.0, 1.45)
    ng.links.new(geo.outputs["Position"], map_oil.inputs["Vector"])
    n_oil = _noise(ng, scale=1.0, detail=7.0, roughness=0.66, distortion=0.30, loc=(-520, -1780))
    ng.links.new(map_oil.outputs["Vector"], n_oil.inputs["Vector"])
    oil_str0 = _map_range(ng, 0.50, 0.74, 0.0, 1.0, (-280, -1780), clamp=True)
    ng.links.new(n_oil.outputs["Factor"], oil_str0.inputs[0])

    if img_drip is not None:
        mp_dr = _new_node(ng, "ShaderNodeMapping", (-760, -1980))
        mp_dr.inputs["Scale"].default_value = (1.55, 1.55, 0.95)
        ng.links.new(geo.outputs["Position"], mp_dr.inputs["Vector"])
        t_dr = _new_node(ng, "ShaderNodeTexImage", (-520, -1980))
        t_dr.image = img_drip
        t_dr.projection = 'BOX'
        t_dr.projection_blend = 0.20
        ng.links.new(mp_dr.outputs["Vector"], t_dr.inputs["Vector"])
        dr_amp = _math(ng, 'MULTIPLY', (-280, -1980), clamp=True, v1=0.95)
        ng.links.new(t_dr.outputs["Color"], dr_amp.inputs[0])
        oil_str = _math(ng, 'MAXIMUM', (-60, -1840), clamp=True)
        ng.links.new(oil_str0.outputs[0], oil_str.inputs[0])
        ng.links.new(dr_amp.outputs[0], oil_str.inputs[1])
    else:
        oil_str = oil_str0

    n_opatch = _noise(ng, scale=2.1, detail=4.0, roughness=0.55, loc=(-520, -1780))
    ng.links.new(geo.outputs["Position"], n_opatch.inputs["Vector"])
    oil_patch = _map_range(ng, 0.44, 0.72, 0.0, 1.0, (-280, -1780), clamp=True)
    ng.links.new(n_opatch.outputs["Factor"], oil_patch.inputs[0])

    sep_gn = _new_node(ng, "ShaderNodeSeparateXYZ", (-520, -1960))
    ng.links.new(geo.outputs["Normal"], sep_gn.inputs["Vector"])
    grav_gate = _map_range(ng, 0.05, 0.42, 1.0, 0.0, (-280, -1960), clamp=True)
    ng.links.new(sep_gn.outputs["Z"], grav_gate.inputs[0])

    oil_src = _math(ng, 'MULTIPLY_ADD', (-20, -1700), clamp=True, v1=0.75, v2=0.28)
    ng.links.new(crev_mul.outputs[0], oil_src.inputs[0])

    oil_m0 = _math(ng, 'MULTIPLY', (200, -1580), clamp=True)
    ng.links.new(oil_str.outputs[0], oil_m0.inputs[0])
    ng.links.new(oil_patch.outputs[0], oil_m0.inputs[1])

    oil_m1 = _math(ng, 'MULTIPLY', (420, -1580), clamp=True)
    ng.links.new(oil_m0.outputs[0], oil_m1.inputs[0])
    ng.links.new(grav_gate.outputs[0], oil_m1.inputs[1])

    oil_final = _math(ng, 'MULTIPLY', (640, -1580), clamp=True)
    ng.links.new(oil_m1.outputs[0], oil_final.inputs[0])
    ng.links.new(oil_src.outputs[0], oil_final.inputs[1])
    ng.links.new(oil_final.outputs[0], out.inputs["OilLeak_Mask"])

    return ng


def build_ng_heat_oxidize():

    ng = _fresh_node_group("NG_F15_HeatOxidize")
    _add_ng_input(ng, "Base_Color",    'NodeSocketColor', (0.6, 0.6, 0.6, 1.0))
    _add_ng_input(ng, "Heat_Level",    'NodeSocketFloat', 0.0, 0.0, 1.5)
    _add_ng_input(ng, "Grazing_Tint",  'NodeSocketColor', rgba(0x8A8276))
    _add_ng_input(ng, "Grazing_Mix",   'NodeSocketFloat', 0.18, 0.0, 0.6)
    _add_ng_input(ng, "Tint_Mix",      'NodeSocketFloat', 0.35, 0.0, 1.0)
    _add_ng_input(ng, "Global_Heat",   'NodeSocketFloat', 1.0, 0.0, 2.0)
    _add_ng_input(ng, "F82_Tint",      'NodeSocketColor', rgba(0xE2E8F0))

    _add_ng_output(ng, "Color",         'NodeSocketColor')
    _add_ng_output(ng, "Specular_Tint", 'NodeSocketColor')

    inp = _new_node(ng, "NodeGroupInput", (-820, 0))
    out = _new_node(ng, "NodeGroupOutput", (620, 0))

    h_eff = _math(ng, 'MULTIPLY', (-560, 80), clamp=True)
    ng.links.new(inp.outputs["Heat_Level"], h_eff.inputs[0])
    ng.links.new(inp.outputs["Global_Heat"], h_eff.inputs[1])


    ramp_ht = _ramp(ng, [
        (0.00, rgba(0xB4B8BC)),
        (0.22, rgba(0xAE9E7A)),
        (0.45, rgba(0x8D784B)),
        (0.66, rgba(0x63584C)),
        (0.84, rgba(0x4E545E)),
        (1.00, rgba(0x3B4048)),
    ], loc=(-280, 200))
    ng.links.new(h_eff.outputs[0], ramp_ht.inputs["Fac"])

    mix_fac = _math(ng, 'MULTIPLY', (-220, -10), clamp=True)
    ng.links.new(h_eff.outputs[0], mix_fac.inputs[0])
    ng.links.new(inp.outputs["Tint_Mix"], mix_fac.inputs[1])

    mix_col = _mix_rgba(ng, 'MIX', (100, 160))
    ng.links.new(mix_fac.outputs[0], mix_col.inputs[0])
    ng.links.new(inp.outputs["Base_Color"], mix_col.inputs[6])
    ng.links.new(ramp_ht.outputs["Color"], mix_col.inputs[7])


    lw = _new_node(ng, "ShaderNodeLayerWeight", (-560, -240))
    lw.inputs["Blend"].default_value = 0.28
    g_pow = _math(ng, 'POWER', (-340, -240), clamp=True, v1=1.8)
    ng.links.new(lw.outputs["Facing"], g_pow.inputs[0])
    g_mul0 = _math(ng, 'MULTIPLY', (-140, -240), clamp=True)
    ng.links.new(g_pow.outputs[0], g_mul0.inputs[0])
    ng.links.new(inp.outputs["Grazing_Mix"], g_mul0.inputs[1])
    g_mul1 = _math(ng, 'MULTIPLY', (60, -240), clamp=True)
    ng.links.new(g_mul0.outputs[0], g_mul1.inputs[0])
    ng.links.new(h_eff.outputs[0], g_mul1.inputs[1])

    mix_graz = _mix_rgba(ng, 'MIX', (340, 140))
    ng.links.new(g_mul1.outputs[0], mix_graz.inputs[0])
    ng.links.new(mix_col.outputs[2], mix_graz.inputs[6])
    ng.links.new(inp.outputs["Grazing_Tint"], mix_graz.inputs[7])
    ng.links.new(mix_graz.outputs[2], out.inputs["Color"])


    ng.links.new(inp.outputs["F82_Tint"], out.inputs["Specular_Tint"])

    return ng


def build_ng_mode_wrapper(ng_global):

    ng = _fresh_node_group("NG_F15_ModeWrapper")
    _add_ng_input(ng, "PBR_Shader",   'NodeSocketShader')
    _add_ng_input(ng, "ID_Color",     'NodeSocketColor', (0.7, 0.7, 0.7, 1.0))
    _add_ng_input(ng, "Clay_Color",   'NodeSocketColor', rgba(0xC2C6CC))
    _add_ng_input(ng, "AO_Raw",       'NodeSocketFloat', 1.0, 0.0, 1.0)
    _add_ng_input(ng, "Bevel_Normal", 'NodeSocketVector')
    _add_ng_output(ng, "Shader",      'NodeSocketShader')

    inp = _new_node(ng, "NodeGroupInput", (-680, 0))
    out = _new_node(ng, "NodeGroupOutput", (620, 0))

    ctrl = _new_node(ng, "ShaderNodeGroup", (-680, -300))
    ctrl.node_tree = ng_global


    mix_cid = _mix_rgba(ng, 'MIX', (-360, -120))
    mix_cid.name = "ColorID_Mode_Mix"
    ng.links.new(ctrl.outputs["Show_ColorID"], mix_cid.inputs[0])
    ng.links.new(inp.outputs["Clay_Color"], mix_cid.inputs[6])
    ng.links.new(inp.outputs["ID_Color"], mix_cid.inputs[7])

    ramp_ao = _ramp(ng, [(0.12, rgba(0x525760)), (0.90, (1.0, 1.0, 1.0, 1.0))], (-360, -340))
    ng.links.new(inp.outputs["AO_Raw"], ramp_ao.inputs["Fac"])

    mix_ao = _mix_rgba(ng, 'MULTIPLY', (-120, -180), fac=1.0)
    ng.links.new(mix_cid.outputs[2], mix_ao.inputs[6])
    ng.links.new(ramp_ao.outputs["Color"], mix_ao.inputs[7])

    clay_bsdf = _new_node(ng, "ShaderNodeBsdfPrincipled", (120, -180))
    clay_bsdf.inputs["Metallic"].default_value = 0.0
    clay_bsdf.inputs["Roughness"].default_value = 0.52
    clay_bsdf.inputs["Specular IOR Level"].default_value = 0.38
    ng.links.new(mix_ao.outputs[2], clay_bsdf.inputs["Base Color"])
    ng.links.new(inp.outputs["Bevel_Normal"], clay_bsdf.inputs["Normal"])


    mix_sh = _new_node(ng, "ShaderNodeMixShader", (400, 40))
    ng.links.new(ctrl.outputs["Show_PBR"], mix_sh.inputs["Fac"])
    ng.links.new(clay_bsdf.outputs["BSDF"], mix_sh.inputs[1])
    ng.links.new(inp.outputs["PBR_Shader"], mix_sh.inputs[2])
    ng.links.new(mix_sh.outputs["Shader"], out.inputs["Shader"])

    return ng


def _build_metal_material(
    mat, id_hex: int, vp_hex: int, ng_global, ng_masks, ng_heat, ng_wrap,
    *,
    col_a: int, col_b: int,
    crevice_col: int = 0x1A1816,
    wear_col: int = 0xD2D7DC,
    metallic: float = 0.95,
    rough_min: float = 0.20,
    rough_max: float = 0.36,
    crevice_rough: float = 0.42,
    wear_rough: float = 0.14,
    brush_scale=(2.0, 140.0, 2.0),
    brush_detail: float = 6.0,
    brush_bump: float = 0.045,
    brush_rough_var: float = 0.055,
    grain_scale: float = 260.0,
    grain_bump: float = 0.035,
    cast_voronoi_scale: float = 0.0,
    cast_voronoi_bump: float = 0.0,
    heat_base: float = 0.0,
    heat_mottle_amp: float = 0.22,
    heat_y0: float = -10.0,
    heat_y1: float = 10.0,
    heat_axial_amp: float = 0.0,
    grazing_tint: int = 0x7A746C,
    grazing_mix: float = 0.16,
    heat_tint_mix: float = 0.32,
    soot_amp: float = 0.0,
    soot_streak_scale=(22.0, 2.5, 22.0),
    ao_dist: float = 0.045,
    fine_bevel: float = 0.00034,
    edge_bevel: float = 0.0038,
    edge_gain: float = 1.9,
    edge_wear_mix: float = 0.35,
    crevice_mix: float = 0.55,
    grime_scale: float = 14.0,
    macro_scale: float = 3.8,
    island_var: float = 0.14,
    aniso: float = 0.0,
    specular: float = 0.55,
    coat_weight: float = 0.0,
    coat_rough: float = 0.25,
    use_world_y_for_heat: bool = False,
    scratch_amp: float = 0.42,
    oil_leak_amp: float = 0.0,
    rust_amount: float = 0.0,
    torque_seal_amp: float = 0.0,
    shroud_3zone_thermal: bool = False,
    waffle_chem_mill: bool = False,
    oil_canning_bump: float = 0.0,
    rivet_streaks: bool = False,
    lathe_wave_scale: float = 0.0,
    lathe_wave_bump: float = 0.0,
    weld_haz_amp: float = 0.0,
    smudge_rough_amp: float = 0.0,
    corrugation_ash_amp: float = 0.0,
    cast_machined_contrast: float = 0.0,
    swirl_polish_amp: float = 0.0,
    aniso_rot_var: float = 0.0,
    spangle_shimmer: float = 0.0,
    water_stain_amp: float = 0.0,
    oxide_scale_amp: float = 0.0,
    anti_seize_amp: float = 0.0,
    bolt_alloy_var: float = 0.0,
    tube_bend_sheen: float = 0.0,
    thin_film_amp: float = 0.0,
    effusion_holes: bool = False,
    combustor_axial_soot: bool = False,
    exit_lip_soot_amp: float = 0.0,
    braid_micro_weave: bool = False,
    fan_rub_strip: bool = False,
    blade_tip_erosion: bool = False,
    spinner_nose_wear: bool = False,
    igv_leading_edge: bool = False,
    fluid_id_bands: bool = False,
    waffle_aft_transition: bool = False,
    torque_seal_tight: bool = False,
    nozzle_rub_arcs: bool = False,
    rough_floor: float = None,
    tangent_mode: str = 'RADIAL_Y',
    f82_tint: int = 0xE2E8F0,
    rough_breakup_amp: float = 0.62,
    tf_nm_range=(6.0, 20.0),
    aniso_rot_base: float = 0.0,
    wheel_hubcap_lathe: bool = False,
    nozzle_nickel_antiseize: float = 0.0,
):

    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1780, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1560, -440))
    ctrl.node_tree = ng_global

    masks = _new_node(nt, "ShaderNodeGroup", (-1440, 120))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = ao_dist
    masks.inputs["Fine_Bevel_Rad"].default_value = fine_bevel
    masks.inputs["Edge_Bevel_Rad"].default_value = edge_bevel
    masks.inputs["Edge_Gain"].default_value = edge_gain
    masks.inputs["Grime_Scale"].default_value = grime_scale
    masks.inputs["Macro_Scale"].default_value = macro_scale
    masks.inputs["Micro_Scale"].default_value = grain_scale


    mix_base = _mix_rgba(nt, 'MIX', (-1060, 420), col_a=rgba(col_a), col_b=rgba(col_b))
    nt.links.new(masks.outputs["Macro_Mottle"], mix_base.inputs[0])


    waffle_rib_mask_out = None
    waffle_etch_bump_out = None
    if waffle_chem_mill:
        geo_wf = _new_node(nt, "ShaderNodeNewGeometry", (-1700, 760))
        sep_w = _new_node(nt, "ShaderNodeSeparateXYZ", (-1520, 760))
        nt.links.new(geo_wf.outputs["Position"], sep_w.inputs["Vector"])

        attr_wf_conv = _new_node(nt, "ShaderNodeAttribute", (-1520, 920))
        attr_wf_conv.attribute_name = "f15_convexity"
        attr_wf_exp = _new_node(nt, "ShaderNodeAttribute", (-1520, 1040))
        attr_wf_exp.attribute_name = "f15_exposure"

        rib_cv = _map_range(nt, 0.08, 0.28, 0.0, 1.0, (-1320, 920), clamp=True)
        nt.links.new(attr_wf_conv.outputs["Fac"], rib_cv.inputs[0])
        rib_ex = _map_range(nt, 0.42, 0.78, 0.0, 1.0, (-1320, 1040), clamp=True)
        nt.links.new(attr_wf_exp.outputs["Fac"], rib_ex.inputs[0])
        rib_raw = _math(nt, 'MAXIMUM', (-1140, 960), clamp=True)
        nt.links.new(rib_cv.outputs[0], rib_raw.inputs[0])
        nt.links.new(rib_ex.outputs[0], rib_raw.inputs[1])

        rib_not_conc = _map_range(nt, 0.18, 0.55, 1.0, 0.15, (-1140, 820), clamp=True)
        nt.links.new(masks.outputs["Topo_Concave"], rib_not_conc.inputs[0])
        rib_ao_gate = _map_range(nt, 0.45, 0.88, 0.15, 1.0, (-1140, 680), clamp=True)
        nt.links.new(masks.outputs["AO_Raw"], rib_ao_gate.inputs[0])

        rib_m0 = _math(nt, 'MULTIPLY', (-980, 900), clamp=True)
        nt.links.new(rib_raw.outputs[0], rib_m0.inputs[0])
        nt.links.new(rib_not_conc.outputs[0], rib_m0.inputs[1])
        rib_mask = _math(nt, 'MULTIPLY', (-860, 840), clamp=True)
        nt.links.new(rib_m0.outputs[0], rib_mask.inputs[0])
        nt.links.new(rib_ao_gate.outputs[0], rib_mask.inputs[1])
        waffle_rib_mask_out = rib_mask.outputs[0]


        mix_wf = _mix_rgba(nt, 'MIX', (-720, 620), col_b=rgba(0xC2C8D0))
        nt.links.new(rib_mask.outputs[0], mix_wf.inputs[0])
        nt.links.new(mix_base.outputs[2], mix_wf.inputs[6])


        men_w = _math(nt, 'MULTIPLY', (-920, 900), clamp=True, v1=0.44)
        nt.links.new(masks.outputs["Topo_Concave"], men_w.inputs[0])
        men_we = _math(nt, 'MULTIPLY', (-760, 900), clamp=True)
        nt.links.new(men_w.outputs[0], men_we.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], men_we.inputs[1])
        mix_men = _mix_rgba(nt, 'MIX', (-580, 640), col_b=rgba(0x4E3A20))
        nt.links.new(men_we.outputs[0], mix_men.inputs[0])
        nt.links.new(mix_wf.outputs[2], mix_men.inputs[6])
        base_col_src = mix_men.outputs[2]


        if waffle_aft_transition:
            wf_ay = _map_range(nt, -1.08, -0.68, 0.0, 0.34, (-760, 760), clamp=True)
            nt.links.new(sep_w.outputs["Y"], wf_ay.inputs[0])
            wf_am = _map_range(nt, 0.18, 0.84, 0.65, 1.25, (-760, 880), clamp=True)
            nt.links.new(masks.outputs["Macro_Mottle"], wf_am.inputs[0])
            wf_a0 = _math(nt, 'MULTIPLY', (-580, 820), clamp=True)
            nt.links.new(wf_ay.outputs[0], wf_a0.inputs[0])
            nt.links.new(wf_am.outputs[0], wf_a0.inputs[1])
            wf_ae = _math(nt, 'MULTIPLY', (-420, 820), clamp=True)
            nt.links.new(wf_a0.outputs[0], wf_ae.inputs[0])
            nt.links.new(ctrl.outputs["HeatTint"], wf_ae.inputs[1])
            mix_wfa = _mix_rgba(nt, 'MIX', (-400, 640), col_b=rgba(0x8D7E62))
            nt.links.new(wf_ae.outputs[0], mix_wfa.inputs[0])
            nt.links.new(base_col_src, mix_wfa.inputs[6])
            base_col_src = mix_wfa.outputs[2]


        vor_et = _voronoi(nt, scale=920.0, feature='F1', randomness=0.95, loc=(-920, 1080))
        nt.links.new(tc.outputs["Object"], vor_et.inputs["Vector"])
        inv_rib = _math(nt, 'SUBTRACT', (-740, 1080), clamp=True, v0=1.0)
        nt.links.new(rib_mask.outputs[0], inv_rib.inputs[1])
        et_m = _math(nt, 'MULTIPLY', (-580, 1080), clamp=True)
        nt.links.new(vor_et.outputs["Distance"], et_m.inputs[0])
        nt.links.new(inv_rib.outputs[0], et_m.inputs[1])
        waffle_etch_bump_out = et_m.outputs[0]


    elif shroud_3zone_thermal:
        sep_sy = _new_node(nt, "ShaderNodeSeparateXYZ", (-1340, 620))
        nt.links.new(tc.outputs["Object"], sep_sy.inputs["Vector"])
        y_jit = _math(nt, 'MULTIPLY_ADD', (-1140, 620), v1=0.018)
        nt.links.new(masks.outputs["Macro_Mottle"], y_jit.inputs[0])
        nt.links.new(sep_sy.outputs["Y"], y_jit.inputs[2])
        y_norm = _map_range(nt, -0.68, -0.05, 0.0, 1.0, (-960, 620), clamp=True)
        nt.links.new(y_jit.outputs[0], y_norm.inputs[0])
        ramp_sh = _ramp(nt, [
            (0.00, rgba(0xC2AC72)),
            (0.28, rgba(0xA48F54)),
            (0.54, rgba(0x76643A)),
            (0.66, rgba(0x9C8E60)),
            (0.72, rgba(0xB4A46E)),
            (0.81, rgba(0x6D5A34)),
            (0.92, rgba(0x4E3F2B)),
            (1.00, rgba(0x322B26)),
        ], loc=(-720, 620))
        nt.links.new(y_norm.outputs[0], ramp_sh.inputs["Fac"])
        sh_blend = _mix_rgba(nt, 'MIX', (-880, 480), fac=0.80)
        nt.links.new(mix_base.outputs[2], sh_blend.inputs[6])
        nt.links.new(ramp_sh.outputs["Color"], sh_blend.inputs[7])


        scorch_y = _map_range(nt, -0.36, -0.05, 0.0, 1.0, (-960, 780), clamp=True)
        nt.links.new(y_jit.outputs[0], scorch_y.inputs[0])
        scorch_bd = _map_range(nt, 0.35, 1.50, 0.08, 0.72, (-960, 920), clamp=True)
        nt.links.new(ctrl.outputs["SurfaceErosion"], scorch_bd.inputs[0])
        scorch_fac = _math(nt, 'MULTIPLY', (-720, 820), clamp=True)
        nt.links.new(scorch_y.outputs[0], scorch_fac.inputs[0])
        nt.links.new(scorch_bd.outputs[0], scorch_fac.inputs[1])

        sh_scorch = _mix_rgba(nt, 'MIX', (-540, 560), col_b=rgba(0x221E1A))
        nt.links.new(scorch_fac.outputs[0], sh_scorch.inputs[0])
        nt.links.new(sh_blend.outputs[2], sh_scorch.inputs[6])


        sh_ang = _math(nt, 'ARCTAN2', (-1340, 960))
        nt.links.new(sep_sy.outputs["X"], sh_ang.inputs[0])
        nt.links.new(sep_sy.outputs["Z"], sh_ang.inputs[1])
        sh_sec = _math(nt, 'MULTIPLY_ADD', (-1160, 960), v1=8.0 / (2.0 * math.pi), v2=4.5)
        nt.links.new(sh_ang.outputs[0], sh_sec.inputs[0])
        sh_flr = _math(nt, 'FLOOR', (-980, 960))
        nt.links.new(sh_sec.outputs[0], sh_flr.inputs[0])
        sh_half = _math(nt, 'GREATER_THAN', (-1160, 1080), v1=-0.415)
        nt.links.new(sep_sy.outputs["Y"], sh_half.inputs[0])
        sh_pid = _math(nt, 'MULTIPLY_ADD', (-820, 1020), v1=17.0)
        nt.links.new(sh_half.outputs[0], sh_pid.inputs[0])
        nt.links.new(sh_flr.outputs[0], sh_pid.inputs[2])
        wn_sh = _new_node(nt, "ShaderNodeTexWhiteNoise", (-640, 1020))
        wn_sh.noise_dimensions = '1D'
        nt.links.new(sh_pid.outputs[0], wn_sh.inputs["W"])
        sh_pv = _map_range(nt, 0.0, 1.0, 0.93, 1.07, (-460, 1020), clamp=True)
        nt.links.new(wn_sh.outputs["Value"], sh_pv.inputs[0])
        sh_ph = _map_range(nt, 0.0, 1.0, 0.491, 0.509, (-460, 1140), clamp=True)
        nt.links.new(wn_sh.outputs["Value"], sh_ph.inputs[0])
        sh_hsv = _new_node(nt, "ShaderNodeHueSaturation", (-320, 620))
        nt.links.new(sh_ph.outputs[0], sh_hsv.inputs["Hue"])
        nt.links.new(sh_pv.outputs[0], sh_hsv.inputs["Value"])
        nt.links.new(sh_scorch.outputs[2], sh_hsv.inputs["Color"])
        base_col_src = sh_hsv.outputs["Color"]
    else:
        base_col_src = mix_base.outputs[2]


    hsv_isl = _new_node(nt, "ShaderNodeHueSaturation", (-680, 420))
    val_isl = _map_range(nt, 0.0, 1.0, 1.0 - island_var, 1.0 + island_var, (-1060, 210), clamp=True)
    sat_isl = _map_range(nt, 0.0, 1.0, 0.85, 1.18, (-1060, 40), clamp=True)
    nt.links.new(masks.outputs["Island_Rand"], val_isl.inputs[0])
    nt.links.new(masks.outputs["Island_Rand"], sat_isl.inputs[0])
    nt.links.new(val_isl.outputs[0], hsv_isl.inputs["Value"])
    nt.links.new(sat_isl.outputs[0], hsv_isl.inputs["Saturation"])
    nt.links.new(base_col_src, hsv_isl.inputs["Color"])


    map_br = _new_node(nt, "ShaderNodeMapping", (-1440, -180))
    map_br.inputs["Scale"].default_value = brush_scale
    nt.links.new(tc.outputs["Object"], map_br.inputs["Vector"])
    n_brush = _noise(nt, scale=1.0, detail=brush_detail, roughness=0.62, distortion=0.04, loc=(-1200, -180))
    nt.links.new(map_br.outputs["Vector"], n_brush.inputs["Vector"])


    m_hmot = _math(nt, 'MULTIPLY', (-980, -340), v1=heat_mottle_amp)
    nt.links.new(masks.outputs["Macro_Mottle"], m_hmot.inputs[0])
    add_h0 = _math(nt, 'ADD', (-800, -300), v0=heat_base)
    nt.links.new(m_hmot.outputs[0], add_h0.inputs[1])

    if abs(heat_axial_amp) > 1e-5:
        sep_y = _new_node(nt, "ShaderNodeSeparateXYZ", (-1200, -480))
        if use_world_y_for_heat:
            geo_w = _new_node(nt, "ShaderNodeNewGeometry", (-1420, -480))
            nt.links.new(geo_w.outputs["Position"], sep_y.inputs["Vector"])
        else:
            nt.links.new(tc.outputs["Object"], sep_y.inputs["Vector"])
        y_grad = _map_range(nt, heat_y0, heat_y1, 0.0, heat_axial_amp, (-980, -480), clamp=True)
        nt.links.new(sep_y.outputs["Y"], y_grad.inputs[0])
        add_h1 = _math(nt, 'ADD', (-620, -320), clamp=True)
        nt.links.new(add_h0.outputs[0], add_h1.inputs[0])
        nt.links.new(y_grad.outputs[0], add_h1.inputs[1])
        heat_sig = add_h1.outputs[0]
    else:
        heat_sig = add_h0.outputs[0]

    isl_h = _map_range(nt, 0.0, 1.0, -0.07, 0.08, (-800, -540), clamp=True)
    nt.links.new(masks.outputs["Island_Rand"], isl_h.inputs[0])
    add_h2 = _math(nt, 'ADD', (-440, -320), clamp=True)
    nt.links.new(heat_sig, add_h2.inputs[0])
    nt.links.new(isl_h.outputs[0], add_h2.inputs[1])

    ht_grp = _new_node(nt, "ShaderNodeGroup", (-240, 260))
    ht_grp.node_tree = ng_heat
    nt.links.new(hsv_isl.outputs["Color"], ht_grp.inputs["Base_Color"])
    nt.links.new(add_h2.outputs[0], ht_grp.inputs["Heat_Level"])
    ht_grp.inputs["Grazing_Tint"].default_value = rgba(grazing_tint)
    ht_grp.inputs["Grazing_Mix"].default_value = grazing_mix
    ht_grp.inputs["Tint_Mix"].default_value = heat_tint_mix
    nt.links.new(ctrl.outputs["HeatTint"], ht_grp.inputs["Global_Heat"])
    ht_grp.inputs["F82_Tint"].default_value = rgba(f82_tint)

    cur_col = ht_grp.outputs["Color"]


    tf_thickness_out = None
    if thin_film_amp > 1e-4:
        tf_hm = _math(nt, 'MULTIPLY_ADD', (20, 1520), clamp=True, v1=0.22)
        nt.links.new(masks.outputs["Macro_Mottle"], tf_hm.inputs[0])
        nt.links.new(add_h2.outputs[0], tf_hm.inputs[2])
        tf_nm = _map_range(nt, 0.10, 0.95, float(tf_nm_range[0]), float(tf_nm_range[1]), (200, 1520), clamp=True)
        nt.links.new(tf_hm.outputs[0], tf_nm.inputs[0])
        tf_ht_gate = _map_range(nt, 0.02, 0.25, 0.0, 1.0, (200, 1660), clamp=True)
        nt.links.new(ctrl.outputs["HeatTint"], tf_ht_gate.inputs[0])
        tf_nm_out = _math(nt, 'MULTIPLY', (380, 1560), clamp=False)
        nt.links.new(tf_nm.outputs[0], tf_nm_out.inputs[0])
        nt.links.new(tf_ht_gate.outputs[0], tf_nm_out.inputs[1])
        tf_thickness_out = tf_nm_out.outputs[0]


    if rough_breakup_amp > 1e-4:
        r_src_mix = _mix_float(nt, (-820, 20), fac=min(0.85, max(0.0, rough_breakup_amp)))
        nt.links.new(masks.outputs["Macro_Mottle"], r_src_mix.inputs[2])
        nt.links.new(masks.outputs["Metal_Rough_Breakup"], r_src_mix.inputs[3])
        r_macro_src = r_src_mix.outputs[0]
    else:
        r_macro_src = masks.outputs["Macro_Mottle"]

    r_macro = _map_range(nt, 0.12, 0.88, rough_min, rough_max, (-640, 20), clamp=True)
    nt.links.new(r_macro_src, r_macro.inputs[0])


    br_fused = _mix_float(nt, (-820, -140), fac=0.48)
    nt.links.new(n_brush.outputs["Factor"], br_fused.inputs[2])
    nt.links.new(masks.outputs["Metal_Brush_Grain"], br_fused.inputs[3])

    r_br_off = _map_range(nt, 0.0, 1.0, -brush_rough_var, +brush_rough_var, (-640, -140), clamp=True)
    nt.links.new(br_fused.outputs[0], r_br_off.inputs[0])

    cur_rough_node = _math(nt, 'ADD', (-400, -40), clamp=True)
    nt.links.new(r_macro.outputs[0], cur_rough_node.inputs[0])
    nt.links.new(r_br_off.outputs[0], cur_rough_node.inputs[1])
    cur_rough = cur_rough_node.outputs[0]


    if smudge_rough_amp > 1e-4:
        n_smg = _noise(nt, scale=4.8, detail=4.5, roughness=0.58, distortion=1.15, loc=(-640, -300))
        nt.links.new(tc.outputs["Object"], n_smg.inputs["Vector"])
        r_smg = _map_range(nt, 0.22, 0.78, -smudge_rough_amp, +smudge_rough_amp, (-420, -240), clamp=True)
        nt.links.new(n_smg.outputs["Factor"], r_smg.inputs[0])
        r_add_smg = _math(nt, 'ADD', (-240, -120), clamp=True)
        nt.links.new(cur_rough, r_add_smg.inputs[0])
        nt.links.new(r_smg.outputs[0], r_add_smg.inputs[1])
        cur_rough = r_add_smg.outputs[0]


    if waffle_rib_mask_out is not None:
        mix_wf_r = _mix_float(nt, (-200, 120), b=0.125)
        nt.links.new(waffle_rib_mask_out, mix_wf_r.inputs[0])
        nt.links.new(cur_rough, mix_wf_r.inputs[2])
        cur_rough = mix_wf_r.outputs[0]


    lathe_wave_out = None
    wheel_local_vec_out = None
    if wheel_hubcap_lathe or tangent_mode == 'WHEEL_AXLE_X':
        sep_wh = _new_node(nt, "ShaderNodeSeparateXYZ", (-1180, -680))
        nt.links.new(tc.outputs["Object"], sep_wh.inputs["Vector"])

        wh_y0 = _math(nt, 'ADD', (-1080, -640), v1=2.6875)
        nt.links.new(sep_wh.outputs["Y"], wh_y0.inputs[0])
        wh_ya = _math(nt, 'ABSOLUTE', (-1000, -640))
        nt.links.new(wh_y0.outputs[0], wh_ya.inputs[0])
        wh_dy = _math(nt, 'SUBTRACT', (-920, -640), v1=1.1925)
        nt.links.new(wh_ya.outputs[0], wh_dy.inputs[0])
        wh_dz = _math(nt, 'ADD', (-1000, -740), v1=1.320)
        nt.links.new(sep_wh.outputs["Z"], wh_dz.inputs[0])
        wh_comb = _new_node(nt, "ShaderNodeCombineXYZ", (-840, -680))
        wh_comb.inputs["X"].default_value = 0.0
        nt.links.new(wh_dy.outputs[0], wh_comb.inputs["Y"])
        nt.links.new(wh_dz.outputs[0], wh_comb.inputs["Z"])
        wheel_local_vec_out = wh_comb.outputs["Vector"]

    if lathe_wave_scale > 0.0:
        w_lth = _new_node(nt, "ShaderNodeTexWave", (-700, -680))
        if wheel_hubcap_lathe and wheel_local_vec_out is not None:
            w_lth.wave_type = 'RINGS'
            w_lth.rings_direction = 'X'
            nt.links.new(wheel_local_vec_out, w_lth.inputs["Vector"])
        else:
            w_lth.wave_type = 'BANDS'
            w_lth.bands_direction = 'Y'
            nt.links.new(tc.outputs["Object"], w_lth.inputs["Vector"])
        w_lth.inputs["Scale"].default_value = lathe_wave_scale
        w_lth.inputs["Distortion"].default_value = 0.10
        w_lth.inputs["Detail"].default_value = 2.0
        lathe_wave_out = w_lth.outputs["Fac"]
        r_lth = _map_range(nt, 0.0, 1.0, -0.036, +0.036, (-400, -680), clamp=True)
        nt.links.new(lathe_wave_out, r_lth.inputs[0])
        r_add_l = _math(nt, 'ADD', (-180, -300), clamp=True)
        nt.links.new(cur_rough, r_add_l.inputs[0])
        nt.links.new(r_lth.outputs[0], r_add_l.inputs[1])
        cur_rough = r_add_l.outputs[0]


    if weld_haz_amp > 1e-4:
        n_haz = _noise(nt, scale=18.0, detail=5.0, roughness=0.58, distortion=0.45, loc=(-560, 500))
        nt.links.new(tc.outputs["Object"], n_haz.inputs["Vector"])
        haz_mod = _map_range(nt, 0.25, 0.78, 0.58, 1.35, (-380, 500), clamp=True)
        nt.links.new(n_haz.outputs["Factor"], haz_mod.inputs[0])
        haz_raw = _math(nt, 'MULTIPLY', (-220, 500), clamp=True)
        nt.links.new(masks.outputs["Topo_Concave"], haz_raw.inputs[0])
        nt.links.new(haz_mod.outputs[0], haz_raw.inputs[1])

        ramp_haz = _ramp(nt, [
            (0.00, rgba(0xB8A882)),
            (0.32, rgba(0x96754A)),
            (0.60, rgba(0x5E536E)),
            (0.84, rgba(0x3D4C60)),
            (1.00, rgba(0x2A323D)),
        ], loc=(-220, 660))
        nt.links.new(haz_raw.outputs[0], ramp_haz.inputs["Fac"])

        haz_band = _map_range(nt, 0.16, 0.76, 0.0, weld_haz_amp, (-40, 500), clamp=True)
        nt.links.new(haz_raw.outputs[0], haz_band.inputs[0])
        haz_h = _math(nt, 'MULTIPLY', (120, 500), clamp=True)
        nt.links.new(haz_band.outputs[0], haz_h.inputs[0])
        nt.links.new(ctrl.outputs["HeatTint"], haz_h.inputs[1])
        mix_haz = _mix_rgba(nt, 'MIX', (240, 460))
        nt.links.new(haz_h.outputs[0], mix_haz.inputs[0])
        nt.links.new(cur_col, mix_haz.inputs[6])
        nt.links.new(ramp_haz.outputs["Color"], mix_haz.inputs[7])
        cur_col = mix_haz.outputs[2]


    swirl_bump_out = None
    if swirl_polish_amp > 1e-4:
        w_sw = _new_node(nt, "ShaderNodeTexWave", (-640, -820))
        w_sw.wave_type = 'RINGS'
        w_sw.rings_direction = 'SPHERICAL'
        w_sw.inputs["Scale"].default_value = 185.0
        w_sw.inputs["Distortion"].default_value = 14.0
        w_sw.inputs["Detail"].default_value = 6.0
        w_sw.inputs["Detail Scale"].default_value = 2.6
        nt.links.new(tc.outputs["Object"], w_sw.inputs["Vector"])
        swirl_bump_out = w_sw.outputs["Fac"]
        r_sw = _map_range(nt, 0.0, 1.0, -0.018 * swirl_polish_amp, +0.018 * swirl_polish_amp, (-380, -820), clamp=True)
        nt.links.new(swirl_bump_out, r_sw.inputs[0])
        r_add_sw = _math(nt, 'ADD', (-140, -440), clamp=True)
        nt.links.new(cur_rough, r_add_sw.inputs[0])
        nt.links.new(r_sw.outputs[0], r_add_sw.inputs[1])
        cur_rough = r_add_sw.outputs[0]


    if water_stain_amp > 1e-4:
        n_ws = _noise(nt, scale=9.5, detail=5.5, roughness=0.58, distortion=0.82, loc=(-640, -980))
        nt.links.new(tc.outputs["Object"], n_ws.inputs["Vector"])
        ws_sub = _math(nt, 'SUBTRACT', (-440, -980), v1=0.52)
        nt.links.new(n_ws.outputs["Factor"], ws_sub.inputs[0])
        ws_abs = _math(nt, 'ABSOLUTE', (-280, -980))
        nt.links.new(ws_sub.outputs[0], ws_abs.inputs[0])
        ws_ring = _map_range(nt, 0.0, 0.016, 1.0, 0.0, (-120, -980), clamp=True)
        nt.links.new(ws_abs.outputs[0], ws_ring.inputs[0])

        n_ws_p = _noise(nt, scale=2.6, detail=3.0, roughness=0.50, loc=(-440, -1120))
        nt.links.new(tc.outputs["Object"], n_ws_p.inputs["Vector"])
        ws_gate = _map_range(nt, 0.60, 0.78, 0.0, water_stain_amp, (-240, -1120), clamp=True)
        nt.links.new(n_ws_p.outputs["Factor"], ws_gate.inputs[0])

        ws_m0 = _math(nt, 'MULTIPLY', (40, -1040), clamp=True)
        nt.links.new(ws_ring.outputs[0], ws_m0.inputs[0])
        nt.links.new(ws_gate.outputs[0], ws_m0.inputs[1])
        ws_eff = _math(nt, 'MULTIPLY', (180, -1040), clamp=True)
        nt.links.new(ws_m0.outputs[0], ws_eff.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], ws_eff.inputs[1])

        mix_ws_c = _mix_rgba(nt, 'MIX', (280, 420), col_b=rgba(0x7E7668))
        nt.links.new(ws_eff.outputs[0], mix_ws_c.inputs[0])
        nt.links.new(cur_col, mix_ws_c.inputs[6])
        cur_col = mix_ws_c.outputs[2]
        mix_ws_r = _mix_float(nt, (280, 180), b=0.52)
        nt.links.new(ws_eff.outputs[0], mix_ws_r.inputs[0])
        nt.links.new(cur_rough, mix_ws_r.inputs[2])
        cur_rough = mix_ws_r.outputs[0]


    spangle_bump_out = None
    galv_tilt_vec_out = None
    if spangle_shimmer > 1e-4:
        sep_gv = _new_node(nt, "ShaderNodeSeparateColor", (-620, 1280))
        nt.links.new(masks.outputs["Galv_Crystal_RGB"], sep_gv.inputs["Color"])
        spangle_bump_out = sep_gv.outputs["Red"]

        spg_hsv = _new_node(nt, "ShaderNodeHueSaturation", (260, 520))
        spg_val = _map_range(nt, 0.0, 1.0, 1.0 - 0.065 * spangle_shimmer, 1.0 + 0.065 * spangle_shimmer, (-240, 1280), clamp=True)
        nt.links.new(spangle_bump_out, spg_val.inputs[0])
        nt.links.new(spg_val.outputs[0], spg_hsv.inputs["Value"])
        nt.links.new(cur_col, spg_hsv.inputs["Color"])
        cur_col = spg_hsv.outputs["Color"]

        spg_roff = _map_range(nt, 0.0, 1.0, -0.042 * spangle_shimmer, +0.042 * spangle_shimmer, (-240, 1140), clamp=True)
        nt.links.new(spangle_bump_out, spg_roff.inputs[0])
        r_add_spg = _math(nt, 'ADD', (0, 260), clamp=True)
        nt.links.new(cur_rough, r_add_spg.inputs[0])
        nt.links.new(spg_roff.outputs[0], r_add_spg.inputs[1])
        cur_rough = r_add_spg.outputs[0]


        gv_dx = _map_range(nt, 0.0, 1.0, -1.0, +1.0, (-420, 1360), clamp=True)
        nt.links.new(sep_gv.outputs["Green"], gv_dx.inputs[0])
        gv_dy = _map_range(nt, 0.0, 1.0, -1.0, +1.0, (-420, 1200), clamp=True)
        nt.links.new(sep_gv.outputs["Blue"], gv_dy.inputs[0])
        gv_dz0 = _math(nt, 'SUBTRACT', (-240, 1360))
        nt.links.new(gv_dx.outputs[0], gv_dz0.inputs[0])
        nt.links.new(gv_dy.outputs[0], gv_dz0.inputs[1])
        gv_dz = _math(nt, 'MULTIPLY', (-150, 1360), v1=0.5)
        nt.links.new(gv_dz0.outputs[0], gv_dz.inputs[0])
        gv_comb = _new_node(nt, "ShaderNodeCombineXYZ", (-60, 1320))
        nt.links.new(gv_dx.outputs[0], gv_comb.inputs["X"])
        nt.links.new(gv_dy.outputs[0], gv_comb.inputs["Y"])
        nt.links.new(gv_dz.outputs[0], gv_comb.inputs["Z"])
        gv_scl = _vmath(nt, 'SCALE', (120, 1320))
        gv_scl.inputs["Scale"].default_value = 0.014 * spangle_shimmer
        nt.links.new(gv_comb.outputs["Vector"], gv_scl.inputs[0])
        galv_tilt_vec_out = gv_scl.outputs["Vector"]


    oxide_bump_out = None
    if oxide_scale_amp > 1e-4:
        n_ox = _noise(nt, scale=145.0, detail=10.0, roughness=0.78, distortion=0.35, loc=(-640, 1440))
        nt.links.new(tc.outputs["Object"], n_ox.inputs["Vector"])
        oxide_bump_out = n_ox.outputs["Factor"]
        ox_roff = _map_range(nt, 0.20, 0.80, -0.075 * oxide_scale_amp, +0.095 * oxide_scale_amp, (-400, 1440), clamp=True)
        nt.links.new(oxide_bump_out, ox_roff.inputs[0])
        r_add_ox = _math(nt, 'ADD', (-80, 320), clamp=True)
        nt.links.new(cur_rough, r_add_ox.inputs[0])
        nt.links.new(ox_roff.outputs[0], r_add_ox.inputs[1])
        cur_rough = r_add_ox.outputs[0]


        geo_ox = _new_node(nt, "ShaderNodeNewGeometry", (-640, 1580))
        sep_ox = _new_node(nt, "ShaderNodeSeparateXYZ", (-460, 1580))
        nt.links.new(geo_ox.outputs["Position"], sep_ox.inputs["Vector"])
        ox_yg = _map_range(nt, 0.30, 0.88, 0.0, 0.34 * oxide_scale_amp, (-280, 1580), clamp=True)
        nt.links.new(sep_ox.outputs["Y"], ox_yg.inputs[0])
        ox_ht = _math(nt, 'MULTIPLY', (-100, 1580), clamp=True)
        nt.links.new(ox_yg.outputs[0], ox_ht.inputs[0])
        nt.links.new(ctrl.outputs["HeatTint"], ox_ht.inputs[1])
        mix_ox_c = _mix_rgba(nt, 'MIX', (300, 340), col_b=rgba(0x364458))
        nt.links.new(ox_ht.outputs[0], mix_ox_c.inputs[0])
        nt.links.new(cur_col, mix_ox_c.inputs[6])
        cur_col = mix_ox_c.outputs[2]


    if anti_seize_amp > 1e-4:
        n_as = _noise(nt, scale=280.0, detail=6.0, roughness=0.72, loc=(-460, 1720))
        nt.links.new(tc.outputs["Object"], n_as.inputs["Vector"])
        as_conc0 = _map_range(nt, 0.28, 0.80, 0.0, anti_seize_amp, (-240, 1720), clamp=True)
        nt.links.new(masks.outputs["Topo_Concave"], as_conc0.inputs[0])
        as_conc = _math(nt, 'MULTIPLY', (-60, 1720), clamp=True)
        nt.links.new(as_conc0.outputs[0], as_conc.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], as_conc.inputs[1])
        as_paste = _mix_rgba(nt, 'MIX', (140, 1720), col_a=rgba(0x9C7242), col_b=rgba(0x88847A))
        nt.links.new(n_as.outputs["Factor"], as_paste.inputs[0])
        mix_as_c = _mix_rgba(nt, 'MIX', (340, 300))
        nt.links.new(as_conc.outputs[0], mix_as_c.inputs[0])
        nt.links.new(cur_col, mix_as_c.inputs[6])
        nt.links.new(as_paste.outputs[2], mix_as_c.inputs[7])
        cur_col = mix_as_c.outputs[2]
        mix_as_r = _mix_float(nt, (340, 120), b=0.34)
        nt.links.new(as_conc.outputs[0], mix_as_r.inputs[0])
        nt.links.new(cur_rough, mix_as_r.inputs[2])
        cur_rough = mix_as_r.outputs[0]


    nickel_as_bump_out = None
    if nozzle_nickel_antiseize > 1e-4:
        sep_nas = _new_node(nt, "ShaderNodeSeparateXYZ", (-660, 1780))
        nt.links.new(tc.outputs["Object"], sep_nas.inputs["Vector"])

        nas_h0 = _map_range(nt, 0.352, 0.376, 0.0, 1.0, (-480, 1840), clamp=True)
        nt.links.new(sep_nas.outputs["Y"], nas_h0.inputs[0])
        nas_h1 = _map_range(nt, 0.392, 0.422, 1.0, 0.0, (-480, 1740), clamp=True)
        nt.links.new(sep_nas.outputs["Y"], nas_h1.inputs[0])
        nas_hz = _math(nt, 'MULTIPLY', (-300, 1800), clamp=True)
        nt.links.new(nas_h0.outputs[0], nas_hz.inputs[0])
        nt.links.new(nas_h1.outputs[0], nas_hz.inputs[1])

        nas_t0 = _map_range(nt, 0.542, 0.576, 0.0, 1.0, (-480, 1640), clamp=True)
        nt.links.new(sep_nas.outputs["Y"], nas_t0.inputs[0])
        nas_t1 = _map_range(nt, 0.618, 0.652, 1.0, 0.0, (-480, 1540), clamp=True)
        nt.links.new(sep_nas.outputs["Y"], nas_t1.inputs[0])
        nas_tz = _math(nt, 'MULTIPLY', (-300, 1600), clamp=True)
        nt.links.new(nas_t0.outputs[0], nas_tz.inputs[0])
        nt.links.new(nas_t1.outputs[0], nas_tz.inputs[1])
        nas_zone = _math(nt, 'MAXIMUM', (-140, 1700), clamp=True)
        nt.links.new(nas_hz.outputs[0], nas_zone.inputs[0])
        nt.links.new(nas_tz.outputs[0], nas_zone.inputs[1])

        n_nas = _noise(nt, scale=290.0, detail=7.0, roughness=0.72, distortion=0.35, loc=(-300, 1940))
        nt.links.new(tc.outputs["Object"], n_nas.inputs["Vector"])
        nas_pat = _map_range(nt, 0.36, 0.72, 0.0, 1.0, (-120, 1940), clamp=True)
        nt.links.new(n_nas.outputs["Factor"], nas_pat.inputs[0])

        nas_cc = _map_range(nt, 0.28, 0.78, 0.0, nozzle_nickel_antiseize, (-120, 1820), clamp=True)
        nt.links.new(masks.outputs["Topo_Concave"], nas_cc.inputs[0])
        nas_m0 = _math(nt, 'MULTIPLY', (40, 1800), clamp=True)
        nt.links.new(nas_zone.outputs[0], nas_m0.inputs[0])
        nt.links.new(nas_cc.outputs[0], nas_m0.inputs[1])
        nas_m1 = _math(nt, 'MULTIPLY', (180, 1800), clamp=True)
        nt.links.new(nas_m0.outputs[0], nas_m1.inputs[0])
        nt.links.new(nas_pat.outputs[0], nas_m1.inputs[1])
        nas_eff = _math(nt, 'MULTIPLY', (310, 1800), clamp=True)
        nt.links.new(nas_m1.outputs[0], nas_eff.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], nas_eff.inputs[1])
        nickel_as_bump_out = nas_eff.outputs[0]


        nas_paste = _mix_rgba(nt, 'MIX', (240, 1960), col_a=rgba(0x8C929A), col_b=rgba(0x5E636B))
        nt.links.new(n_nas.outputs["Factor"], nas_paste.inputs[0])
        mix_nas_c = _mix_rgba(nt, 'MIX', (360, 320))
        nt.links.new(nas_eff.outputs[0], mix_nas_c.inputs[0])
        nt.links.new(cur_col, mix_nas_c.inputs[6])
        nt.links.new(nas_paste.outputs[2], mix_nas_c.inputs[7])
        cur_col = mix_nas_c.outputs[2]
        mix_nas_r = _mix_float(nt, (360, 100), b=0.36)
        nt.links.new(nas_eff.outputs[0], mix_nas_r.inputs[0])
        nt.links.new(cur_rough, mix_nas_r.inputs[2])
        cur_rough = mix_nas_r.outputs[0]


    if bolt_alloy_var > 1e-4:
        ramp_ba = _ramp(nt, [
            (0.00, rgba(0xC6CCD4)),
            (0.45, rgba(0xA4ACB6)),
            (0.72, rgba(0xC8B47E)),
            (0.90, rgba(0x6E7886)),
            (1.00, rgba(0xD6C28A)),
        ], loc=(-220, 1860))
        nt.links.new(masks.outputs["Island_Rand"], ramp_ba.inputs["Fac"])
        mix_ba = _mix_rgba(nt, 'MIX', (380, 260), fac=bolt_alloy_var)
        nt.links.new(cur_col, mix_ba.inputs[6])
        nt.links.new(ramp_ba.outputs["Color"], mix_ba.inputs[7])
        cur_col = mix_ba.outputs[2]


    if tube_bend_sheen > 1e-4:
        tb_cv = _map_range(nt, 0.35, 0.85, 0.0, 0.055 * tube_bend_sheen, (-240, 1980), clamp=True)
        nt.links.new(masks.outputs["Topo_Convex"], tb_cv.inputs[0])
        r_add_tb = _math(nt, 'ADD', (40, 180), clamp=True)
        nt.links.new(cur_rough, r_add_tb.inputs[0])
        nt.links.new(tb_cv.outputs[0], r_add_tb.inputs[1])
        cur_rough = r_add_tb.outputs[0]

        tb_cc = _map_range(nt, 0.38, 0.82, 0.0, 0.42 * tube_bend_sheen, (-240, 2100), clamp=True)
        nt.links.new(masks.outputs["Topo_Concave"], tb_cc.inputs[0])
        tb_cw = _math(nt, 'MULTIPLY', (-60, 2100), clamp=True)
        nt.links.new(tb_cc.outputs[0], tb_cw.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], tb_cw.inputs[1])
        mix_tb_c = _mix_rgba(nt, 'MIX', (420, 220), col_b=rgba(0x4A4236))
        nt.links.new(tb_cw.outputs[0], mix_tb_c.inputs[0])
        nt.links.new(cur_col, mix_tb_c.inputs[6])
        cur_col = mix_tb_c.outputs[2]


    if rivet_streaks:
        map_rv = _new_node(nt, "ShaderNodeMapping", (-960, 1140))
        map_rv.inputs["Scale"].default_value = (86.0, 1.55, 86.0)
        nt.links.new(tc.outputs["Object"], map_rv.inputs["Vector"])
        n_rv = _noise(nt, scale=1.0, detail=7.0, roughness=0.66, distortion=0.14, loc=(-720, 1140))
        nt.links.new(map_rv.outputs["Vector"], n_rv.inputs["Vector"])
        rv_str = _map_range(nt, 0.56, 0.77, 0.0, 1.0, (-480, 1140), clamp=True)
        nt.links.new(n_rv.outputs["Factor"], rv_str.inputs[0])
        rv_src = _math(nt, 'MULTIPLY_ADD', (-480, 1000), clamp=True, v1=0.78, v2=0.22)
        nt.links.new(masks.outputs["Crevice_Mask"], rv_src.inputs[0])
        rv_m0 = _math(nt, 'MULTIPLY', (-260, 1100), clamp=True)
        nt.links.new(rv_str.outputs[0], rv_m0.inputs[0])
        nt.links.new(rv_src.outputs[0], rv_m0.inputs[1])
        rv_w = _math(nt, 'MULTIPLY', (-80, 1100), clamp=True, v1=0.78)
        nt.links.new(rv_m0.outputs[0], rv_w.inputs[0])
        rv_eff = _math(nt, 'MULTIPLY', (100, 1100), clamp=True)
        nt.links.new(rv_w.outputs[0], rv_eff.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], rv_eff.inputs[1])

        mix_rv_c = _mix_rgba(nt, 'MIX', (240, 560), col_b=rgba(0x181410))
        nt.links.new(rv_eff.outputs[0], mix_rv_c.inputs[0])
        nt.links.new(cur_col, mix_rv_c.inputs[6])
        cur_col = mix_rv_c.outputs[2]

        rv_ro = _math(nt, 'MULTIPLY', (100, 980), v1=-0.055)
        nt.links.new(rv_eff.outputs[0], rv_ro.inputs[0])
        r_after_rv = _math(nt, 'ADD', (240, 420), clamp=True)
        nt.links.new(cur_rough, r_after_rv.inputs[0])
        nt.links.new(rv_ro.outputs[0], r_after_rv.inputs[1])
        cur_rough = r_after_rv.outputs[0]


    if corrugation_ash_amp > 1e-4:
        ash_cv = _map_range(nt, 0.25, 0.78, 0.0, 1.0, (-220, 740), clamp=True)
        nt.links.new(masks.outputs["Topo_Convex"], ash_cv.inputs[0])
        ash_m0 = _math(nt, 'MULTIPLY', (-20, 740), clamp=True)
        nt.links.new(ash_cv.outputs[0], ash_m0.inputs[0])
        nt.links.new(masks.outputs["Macro_Mottle"], ash_m0.inputs[1])
        ash_eff = _math(nt, 'MULTIPLY', (160, 740), clamp=True, v1=corrugation_ash_amp)
        nt.links.new(ash_m0.outputs[0], ash_eff.inputs[0])
        mix_ash_c = _mix_rgba(nt, 'MIX', (320, 680), col_b=rgba(0x8D8A82))
        nt.links.new(ash_eff.outputs[0], mix_ash_c.inputs[0])
        nt.links.new(cur_col, mix_ash_c.inputs[6])
        cur_col = mix_ash_c.outputs[2]
        mix_ash_r = _mix_float(nt, (320, 480), b=0.76)
        nt.links.new(ash_eff.outputs[0], mix_ash_r.inputs[0])
        nt.links.new(cur_rough, mix_ash_r.inputs[2])
        cur_rough = mix_ash_r.outputs[0]


    if cast_machined_contrast > 1e-4:
        cm_cv = _map_range(nt, 0.42, 0.80, 0.0, cast_machined_contrast, (-220, 880), clamp=True)
        nt.links.new(masks.outputs["Topo_Convex"], cm_cv.inputs[0])
        mix_cm_c = _mix_rgba(nt, 'MIX', (360, 820), col_b=rgba(0xCBD1D8))
        nt.links.new(cm_cv.outputs[0], mix_cm_c.inputs[0])
        nt.links.new(cur_col, mix_cm_c.inputs[6])
        cur_col = mix_cm_c.outputs[2]
        mix_cm_r = _mix_float(nt, (360, 620), b=0.12)
        nt.links.new(cm_cv.outputs[0], mix_cm_r.inputs[0])
        nt.links.new(cur_rough, mix_cm_r.inputs[2])
        cur_rough = mix_cm_r.outputs[0]


    crev_eff = _math(nt, 'MULTIPLY', (-180, -40), clamp=True)
    nt.links.new(masks.outputs["Crevice_Mask"], crev_eff.inputs[0])
    crev_w = _math(nt, 'MULTIPLY', (-380, -200), v0=crevice_mix)
    nt.links.new(ctrl.outputs["Weathering"], crev_w.inputs[1])
    nt.links.new(crev_w.outputs[0], crev_eff.inputs[1])

    mix_crev_c = _mix_rgba(nt, 'MIX', (60, 280), col_b=rgba(crevice_col))
    nt.links.new(crev_eff.outputs[0], mix_crev_c.inputs[0])
    nt.links.new(cur_col, mix_crev_c.inputs[6])
    cur_col = mix_crev_c.outputs[2]

    mix_crev_r = _mix_float(nt, (60, 40), b=crevice_rough)
    nt.links.new(crev_eff.outputs[0], mix_crev_r.inputs[0])
    nt.links.new(cur_rough, mix_crev_r.inputs[2])
    cur_rough = mix_crev_r.outputs[0]

    crev_m_drop = _math(nt, 'MULTIPLY', (60, -120), v1=0.35)
    nt.links.new(crev_eff.outputs[0], crev_m_drop.inputs[0])
    cur_met_node = _math(nt, 'SUBTRACT', (240, -120), clamp=True, v0=metallic)
    nt.links.new(crev_m_drop.outputs[0], cur_met_node.inputs[1])
    cur_met = cur_met_node.outputs[0]

    cur_coat_w = None
    cur_coat_r = coat_rough


    if soot_amp > 1e-4:
        map_st = _new_node(nt, "ShaderNodeMapping", (-960, -760))
        map_st.inputs["Scale"].default_value = soot_streak_scale
        nt.links.new(tc.outputs["Object"], map_st.inputs["Vector"])
        n_st = _noise(nt, scale=1.0, detail=7.0, roughness=0.66, distortion=0.45, loc=(-720, -760))
        nt.links.new(map_st.outputs["Vector"], n_st.inputs["Vector"])

        st_ramp = _map_range(nt, 0.34, 0.78, 0.0, 1.0, (-480, -760), clamp=True)
        nt.links.new(n_st.outputs["Factor"], st_ramp.inputs[0])


        st_bd = _math(nt, 'MULTIPLY_ADD', (-480, -900), v1=0.38, v2=0.82)
        nt.links.new(ctrl.outputs["SurfaceErosion"], st_bd.inputs[0])
        st_sc = _math(nt, 'MULTIPLY', (-280, -900))
        nt.links.new(ctrl.outputs["Soot"], st_sc.inputs[0])
        nt.links.new(st_bd.outputs[0], st_sc.inputs[1])

        st_mul0 = _math(nt, 'MULTIPLY', (-260, -760), v1=soot_amp)
        nt.links.new(st_ramp.outputs[0], st_mul0.inputs[0])
        st_mul1 = _math(nt, 'MULTIPLY', (-80, -760), clamp=True)
        nt.links.new(st_mul0.outputs[0], st_mul1.inputs[0])
        nt.links.new(st_sc.outputs[0], st_mul1.inputs[1])

        mix_st_c = _mix_rgba(nt, 'MIX', (280, 280), col_b=rgba(0x161618))
        nt.links.new(st_mul1.outputs[0], mix_st_c.inputs[0])
        nt.links.new(cur_col, mix_st_c.inputs[6])
        cur_col = mix_st_c.outputs[2]

        mix_st_r = _mix_float(nt, (280, 40), b=0.84)
        nt.links.new(st_mul1.outputs[0], mix_st_r.inputs[0])
        nt.links.new(cur_rough, mix_st_r.inputs[2])
        cur_rough = mix_st_r.outputs[0]

        st_m_drop = _math(nt, 'MULTIPLY', (240, -260), v1=0.72)
        nt.links.new(st_mul1.outputs[0], st_m_drop.inputs[0])
        m_after_st = _math(nt, 'SUBTRACT', (420, -140), clamp=True)
        nt.links.new(cur_met, m_after_st.inputs[0])
        nt.links.new(st_m_drop.outputs[0], m_after_st.inputs[1])
        cur_met = m_after_st.outputs[0]


    rust_height_out = None
    if rust_amount > 1e-4:
        map_rs = _new_node(nt, "ShaderNodeMapping", (-480, -1040))
        map_rs.inputs["Scale"].default_value = (14.0, 2.2, 14.0)
        nt.links.new(tc.outputs["Object"], map_rs.inputs["Vector"])
        n_rs = _noise(nt, scale=1.0, detail=8.0, roughness=0.72, distortion=0.38, loc=(-240, -1040))
        nt.links.new(map_rs.outputs["Vector"], n_rs.inputs["Vector"])
        rs_pit = _map_range(nt, 0.46, 0.74, 0.0, 1.0, (0, -1040), clamp=True)
        nt.links.new(n_rs.outputs["Factor"], rs_pit.inputs[0])


        rs_ew = _math(nt, 'MULTIPLY', (-180, -1180), v1=0.45)
        nt.links.new(masks.outputs["EdgeWear_Mask"], rs_ew.inputs[0])
        rs_loc = _math(nt, 'MAXIMUM', (0, -1180), clamp=True)
        nt.links.new(masks.outputs["Crevice_Mask"], rs_loc.inputs[0])
        nt.links.new(rs_ew.outputs[0], rs_loc.inputs[1])

        rs_exp = _math(nt, 'MULTIPLY', (200, -1140), clamp=True)
        nt.links.new(rs_loc.outputs[0], rs_exp.inputs[0])
        nt.links.new(masks.outputs["Exposure_Mask"], rs_exp.inputs[1])

        rs_bd = _math(nt, 'MULTIPLY_ADD', (200, -1280), clamp=True, v1=0.92, v2=0.06)
        nt.links.new(ctrl.outputs["SurfaceErosion"], rs_bd.inputs[0])

        rs_m0 = _math(nt, 'MULTIPLY', (400, -1080), clamp=True)
        nt.links.new(rs_pit.outputs[0], rs_m0.inputs[0])
        nt.links.new(rs_exp.outputs[0], rs_m0.inputs[1])

        rs_m1 = _math(nt, 'MULTIPLY', (580, -1080), clamp=True)
        nt.links.new(rs_m0.outputs[0], rs_m1.inputs[0])
        nt.links.new(rs_bd.outputs[0], rs_m1.inputs[1])

        rs_mask = _math(nt, 'MULTIPLY', (740, -1080), clamp=True, v1=rust_amount)
        nt.links.new(rs_m1.outputs[0], rs_mask.inputs[0])
        rust_height_out = rs_mask.outputs[0]

        rust_col_mix = _mix_rgba(nt, 'MIX', (420, 440), col_a=rgba(0x6C2E11), col_b=rgba(0x361407))
        nt.links.new(n_rs.outputs["Factor"], rust_col_mix.inputs[0])

        mix_rs_c = _mix_rgba(nt, 'MIX', (600, 400))
        nt.links.new(rs_mask.outputs[0], mix_rs_c.inputs[0])
        nt.links.new(cur_col, mix_rs_c.inputs[6])
        nt.links.new(rust_col_mix.outputs[2], mix_rs_c.inputs[7])
        cur_col = mix_rs_c.outputs[2]

        mix_rs_m = _mix_float(nt, (600, -180), b=0.0)
        nt.links.new(rs_mask.outputs[0], mix_rs_m.inputs[0])
        nt.links.new(cur_met, mix_rs_m.inputs[2])
        cur_met = mix_rs_m.outputs[0]

        mix_rs_r = _mix_float(nt, (600, 20), b=0.90)
        nt.links.new(rs_mask.outputs[0], mix_rs_r.inputs[0])
        nt.links.new(cur_rough, mix_rs_r.inputs[2])
        cur_rough = mix_rs_r.outputs[0]


    ew_drive = _math(nt, 'MULTIPLY_ADD', (60, -420), v1=0.52)
    nt.links.new(ctrl.outputs["SurfaceErosion"], ew_drive.inputs[0])
    ew_base = _math(nt, 'MULTIPLY', (-120, -420), v1=0.78)
    nt.links.new(ctrl.outputs["EdgeWear"], ew_base.inputs[0])
    nt.links.new(ew_base.outputs[0], ew_drive.inputs[2])

    ew_w = _math(nt, 'MULTIPLY', (240, -400), v0=edge_wear_mix)
    nt.links.new(ew_drive.outputs[0], ew_w.inputs[1])
    ew_eff = _math(nt, 'MULTIPLY', (420, -340), clamp=True)
    nt.links.new(masks.outputs["EdgeWear_Mask"], ew_eff.inputs[0])
    nt.links.new(ew_w.outputs[0], ew_eff.inputs[1])

    mix_ew_c = _mix_rgba(nt, 'MIX', (680, 260), col_b=rgba(wear_col))
    nt.links.new(ew_eff.outputs[0], mix_ew_c.inputs[0])
    nt.links.new(cur_col, mix_ew_c.inputs[6])
    cur_col = mix_ew_c.outputs[2]

    mix_ew_r = _mix_float(nt, (680, 40), b=wear_rough)
    nt.links.new(ew_eff.outputs[0], mix_ew_r.inputs[0])
    nt.links.new(cur_rough, mix_ew_r.inputs[2])
    cur_rough = mix_ew_r.outputs[0]

    mix_ew_m = _mix_float(nt, (680, -140), b=0.98)
    nt.links.new(ew_eff.outputs[0], mix_ew_m.inputs[0])
    nt.links.new(cur_met, mix_ew_m.inputs[2])
    cur_met = mix_ew_m.outputs[0]


    scratch_eff_out = None
    if scratch_amp > 1e-4:
        sc_bd = _math(nt, 'MULTIPLY_ADD', (420, -480), clamp=True, v1=0.95, v2=0.15)
        nt.links.new(ctrl.outputs["SurfaceErosion"], sc_bd.inputs[0])
        sc_g0 = _math(nt, 'MULTIPLY', (600, -480), clamp=True, v1=scratch_amp)
        nt.links.new(sc_bd.outputs[0], sc_g0.inputs[0])
        sc_eff = _math(nt, 'MULTIPLY', (760, -480), clamp=True)
        nt.links.new(masks.outputs["Scratch_Mask"], sc_eff.inputs[0])
        nt.links.new(sc_g0.outputs[0], sc_eff.inputs[1])
        scratch_eff_out = sc_eff.outputs[0]

        mix_sc_c = _mix_rgba(nt, 'MIX', (820, 260), col_b=rgba(wear_col))
        nt.links.new(sc_eff.outputs[0], mix_sc_c.inputs[0])
        nt.links.new(cur_col, mix_sc_c.inputs[6])
        cur_col = mix_sc_c.outputs[2]

        mix_sc_r = _mix_float(nt, (820, 40), b=max(0.08, wear_rough * 0.85))
        nt.links.new(sc_eff.outputs[0], mix_sc_r.inputs[0])
        nt.links.new(cur_rough, mix_sc_r.inputs[2])
        cur_rough = mix_sc_r.outputs[0]

        mix_sc_m = _mix_float(nt, (820, -140), b=0.99)
        nt.links.new(sc_eff.outputs[0], mix_sc_m.inputs[0])
        nt.links.new(cur_met, mix_sc_m.inputs[2])
        cur_met = mix_sc_m.outputs[0]


    if oil_leak_amp > 1e-4:
        ol_drv = _math(nt, 'MULTIPLY_ADD', (420, -640), v1=0.68)
        nt.links.new(ctrl.outputs["SurfaceErosion"], ol_drv.inputs[0])
        ol_w0 = _math(nt, 'MULTIPLY', (240, -640), v1=0.45)
        nt.links.new(ctrl.outputs["Weathering"], ol_w0.inputs[0])
        nt.links.new(ol_w0.outputs[0], ol_drv.inputs[2])

        ol_g = _math(nt, 'MULTIPLY', (600, -640), clamp=True, v1=oil_leak_amp)
        nt.links.new(ol_drv.outputs[0], ol_g.inputs[0])
        ol_eff = _math(nt, 'MULTIPLY', (760, -640), clamp=True)
        nt.links.new(masks.outputs["OilLeak_Mask"], ol_eff.inputs[0])
        nt.links.new(ol_g.outputs[0], ol_eff.inputs[1])


        ol_wet = _map_range(nt, 0.36, 0.80, 0.0, 1.0, (820, -560), clamp=True)
        nt.links.new(ol_eff.outputs[0], ol_wet.inputs[0])
        ol_ring0 = _map_range(nt, 0.08, 0.46, 0.0, 0.62, (820, -720), clamp=True)
        nt.links.new(ol_eff.outputs[0], ol_ring0.inputs[0])
        ol_not_w = _math(nt, 'SUBTRACT', (920, -640), clamp=True, v0=1.0)
        nt.links.new(ol_wet.outputs[0], ol_not_w.inputs[1])
        ol_halo = _math(nt, 'MULTIPLY', (1020, -680), clamp=True)
        nt.links.new(ol_ring0.outputs[0], ol_halo.inputs[0])
        nt.links.new(ol_not_w.outputs[0], ol_halo.inputs[1])

        mix_oh_c = _mix_rgba(nt, 'MIX', (900, 320), col_b=rgba(0x3A332A))
        nt.links.new(ol_halo.outputs[0], mix_oh_c.inputs[0])
        nt.links.new(cur_col, mix_oh_c.inputs[6])
        cur_col = mix_oh_c.outputs[2]

        mix_oh_r = _mix_float(nt, (900, 100), b=0.82)
        nt.links.new(ol_halo.outputs[0], mix_oh_r.inputs[0])
        nt.links.new(cur_rough, mix_oh_r.inputs[2])
        cur_rough = mix_oh_r.outputs[0]


        mix_ol_c = _mix_rgba(nt, 'MIX', (1020, 280), col_b=rgba(0x130F09))
        nt.links.new(ol_wet.outputs[0], mix_ol_c.inputs[0])
        nt.links.new(cur_col, mix_ol_c.inputs[6])
        cur_col = mix_ol_c.outputs[2]

        mix_ol_r = _mix_float(nt, (1020, 60), b=0.065)
        nt.links.new(ol_wet.outputs[0], mix_ol_r.inputs[0])
        nt.links.new(cur_rough, mix_ol_r.inputs[2])
        cur_rough = mix_ol_r.outputs[0]

        mix_ol_m = _mix_float(nt, (1020, -120), b=0.0)
        nt.links.new(ol_wet.outputs[0], mix_ol_m.inputs[0])
        nt.links.new(cur_met, mix_ol_m.inputs[2])
        cur_met = mix_ol_m.outputs[0]

        ol_coat = _math(nt, 'MULTIPLY_ADD', (1020, -280), clamp=True, v1=0.86, v2=coat_weight)
        nt.links.new(ol_wet.outputs[0], ol_coat.inputs[0])
        cur_coat_w = ol_coat.outputs[0]
        cur_coat_r = 0.055


    effusion_bump_out = None
    if effusion_holes:
        sep_ef = _new_node(nt, "ShaderNodeSeparateXYZ", (-1120, 1740))
        nt.links.new(tc.outputs["Object"], sep_ef.inputs["Vector"])
        ef_ang = _math(nt, 'ARCTAN2', (-940, 1800))
        nt.links.new(sep_ef.outputs["X"], ef_ang.inputs[0])
        nt.links.new(sep_ef.outputs["Z"], ef_ang.inputs[1])

        ef_u0 = _math(nt, 'MULTIPLY_ADD', (-760, 1820), v1=210.0 / (2.0 * math.pi), v2=105.0)
        nt.links.new(ef_ang.outputs[0], ef_u0.inputs[0])
        ef_v0 = _math(nt, 'MULTIPLY', (-760, 1680), v1=92.0)
        nt.links.new(sep_ef.outputs["Y"], ef_v0.inputs[0])

        ef_row = _math(nt, 'FLOOR', (-580, 1620))
        nt.links.new(ef_v0.outputs[0], ef_row.inputs[0])
        ef_par = _math(nt, 'MODULO', (-420, 1620), v1=2.0)
        nt.links.new(ef_row.outputs[0], ef_par.inputs[0])
        ef_ust = _math(nt, 'MULTIPLY_ADD', (-260, 1780), v1=0.5)
        nt.links.new(ef_par.outputs[0], ef_ust.inputs[0])
        nt.links.new(ef_u0.outputs[0], ef_ust.inputs[2])

        ef_fu = _math(nt, 'FRACT', (-100, 1780))
        nt.links.new(ef_ust.outputs[0], ef_fu.inputs[0])
        ef_fv = _math(nt, 'FRACT', (-100, 1660))
        nt.links.new(ef_v0.outputs[0], ef_fv.inputs[0])
        ef_du = _math(nt, 'SUBTRACT', (60, 1780), v1=0.5)
        nt.links.new(ef_fu.outputs[0], ef_du.inputs[0])
        ef_dv = _math(nt, 'SUBTRACT', (60, 1660), v1=0.5)
        nt.links.new(ef_fv.outputs[0], ef_dv.inputs[0])

        ef_du2 = _math(nt, 'MULTIPLY', (220, 1800))
        nt.links.new(ef_du.outputs[0], ef_du2.inputs[0])
        nt.links.new(ef_du.outputs[0], ef_du2.inputs[1])
        ef_dv2 = _math(nt, 'MULTIPLY', (220, 1680))
        nt.links.new(ef_dv.outputs[0], ef_dv2.inputs[0])
        nt.links.new(ef_dv.outputs[0], ef_dv2.inputs[1])
        ef_r2 = _math(nt, 'ADD', (380, 1740))
        nt.links.new(ef_du2.outputs[0], ef_r2.inputs[0])
        nt.links.new(ef_dv2.outputs[0], ef_r2.inputs[1])
        ef_d = _math(nt, 'SQRT', (520, 1740))
        nt.links.new(ef_r2.outputs[0], ef_d.inputs[0])


        ef_hole = _map_range(nt, 0.14, 0.21, 1.0, 0.0, (680, 1780), clamp=True)
        nt.links.new(ef_d.outputs[0], ef_hole.inputs[0])
        ef_dimp = _map_range(nt, 0.13, 0.34, 0.0, 1.0, (680, 1640), clamp=True)
        nt.links.new(ef_d.outputs[0], ef_dimp.inputs[0])
        effusion_bump_out = ef_dimp.outputs[0]


        ef_adu = _math(nt, 'ABSOLUTE', (220, 1540))
        nt.links.new(ef_du.outputs[0], ef_adu.inputs[0])
        ef_tu = _map_range(nt, 0.09, 0.26, 1.0, 0.0, (380, 1540), clamp=True)
        nt.links.new(ef_adu.outputs[0], ef_tu.inputs[0])
        ef_tv1 = _map_range(nt, -0.02, 0.08, 0.0, 1.0, (380, 1420), clamp=True)
        nt.links.new(ef_dv.outputs[0], ef_tv1.inputs[0])
        ef_tv2 = _map_range(nt, 0.08, 0.48, 1.0, 0.0, (380, 1300), clamp=True)
        nt.links.new(ef_dv.outputs[0], ef_tv2.inputs[0])
        ef_tv = _math(nt, 'MULTIPLY', (540, 1380), clamp=True)
        nt.links.new(ef_tv1.outputs[0], ef_tv.inputs[0])
        nt.links.new(ef_tv2.outputs[0], ef_tv.inputs[1])
        ef_tm0 = _math(nt, 'MULTIPLY', (680, 1460), clamp=True)
        nt.links.new(ef_tu.outputs[0], ef_tm0.inputs[0])
        nt.links.new(ef_tv.outputs[0], ef_tm0.inputs[1])
        ef_tm1 = _math(nt, 'MULTIPLY', (820, 1460), clamp=True, v1=0.45)
        nt.links.new(ef_tm0.outputs[0], ef_tm1.inputs[0])
        ef_tail = _math(nt, 'MULTIPLY', (960, 1460), clamp=True)
        nt.links.new(ef_tm1.outputs[0], ef_tail.inputs[0])
        nt.links.new(ctrl.outputs["Soot"], ef_tail.inputs[1])


        mix_et_c = _mix_rgba(nt, 'MIX', (1040, 420), col_b=rgba(0x181614))
        nt.links.new(ef_tail.outputs[0], mix_et_c.inputs[0])
        nt.links.new(cur_col, mix_et_c.inputs[6])
        cur_col = mix_et_c.outputs[2]

        mix_eh_c = _mix_rgba(nt, 'MIX', (1120, 420), col_b=rgba(0x070709))
        nt.links.new(ef_hole.outputs[0], mix_eh_c.inputs[0])
        nt.links.new(cur_col, mix_eh_c.inputs[6])
        cur_col = mix_eh_c.outputs[2]

        mix_eh_m = _mix_float(nt, (1120, -220), b=0.0)
        nt.links.new(ef_hole.outputs[0], mix_eh_m.inputs[0])
        nt.links.new(cur_met, mix_eh_m.inputs[2])
        cur_met = mix_eh_m.outputs[0]

        mix_eh_r = _mix_float(nt, (1120, 180), b=0.95)
        nt.links.new(ef_hole.outputs[0], mix_eh_r.inputs[0])
        nt.links.new(cur_rough, mix_eh_r.inputs[2])
        cur_rough = mix_eh_r.outputs[0]


    if combustor_axial_soot:
        sep_cb = _new_node(nt, "ShaderNodeSeparateXYZ", (-1120, 1940))
        nt.links.new(tc.outputs["Object"], sep_cb.inputs["Vector"])

        cb_yg = _map_range(nt, -1.65, -0.05, 0.92, 0.22, (-860, 1940), clamp=True)
        nt.links.new(sep_cb.outputs["Y"], cb_yg.inputs[0])
        cb_af = _map_range(nt, 0.15, 0.85, 0.68, 1.25, (-860, 2060), clamp=True)
        nt.links.new(masks.outputs["Photo_Airflow"], cb_af.inputs[0])
        cb_m0 = _math(nt, 'MULTIPLY', (-640, 1980), clamp=True)
        nt.links.new(cb_yg.outputs[0], cb_m0.inputs[0])
        nt.links.new(cb_af.outputs[0], cb_m0.inputs[1])
        cb_eff = _math(nt, 'MULTIPLY', (-460, 1980), clamp=True)
        nt.links.new(cb_m0.outputs[0], cb_eff.inputs[0])
        nt.links.new(ctrl.outputs["Soot"], cb_eff.inputs[1])

        mix_cb_c = _mix_rgba(nt, 'MIX', (1160, 440), col_b=rgba(0x111012))
        nt.links.new(cb_eff.outputs[0], mix_cb_c.inputs[0])
        nt.links.new(cur_col, mix_cb_c.inputs[6])
        cur_col = mix_cb_c.outputs[2]

        mix_cb_r = _mix_float(nt, (1160, 200), b=0.90)
        nt.links.new(cb_eff.outputs[0], mix_cb_r.inputs[0])
        nt.links.new(cur_rough, mix_cb_r.inputs[2])
        cur_rough = mix_cb_r.outputs[0]

        cb_md = _math(nt, 'MULTIPLY', (960, -260), clamp=True, v1=0.68)
        nt.links.new(cb_eff.outputs[0], cb_md.inputs[0])
        m_after_cb = _math(nt, 'SUBTRACT', (1160, -240), clamp=True)
        nt.links.new(cur_met, m_after_cb.inputs[0])
        nt.links.new(cb_md.outputs[0], m_after_cb.inputs[1])
        cur_met = m_after_cb.outputs[0]


    if exit_lip_soot_amp > 1e-4:
        geo_ex = _new_node(nt, "ShaderNodeNewGeometry", (-1120, 2140))
        sep_ex = _new_node(nt, "ShaderNodeSeparateXYZ", (-940, 2140))
        nt.links.new(geo_ex.outputs["Position"], sep_ex.inputs["Vector"])
        ex_yg = _map_range(nt, 0.22, 0.88, 0.0, 1.0, (-740, 2140), clamp=True)
        nt.links.new(sep_ex.outputs["Y"], ex_yg.inputs[0])
        ex_af = _map_range(nt, 0.12, 0.85, 0.55, 1.25, (-740, 2260), clamp=True)
        nt.links.new(masks.outputs["Photo_Airflow"], ex_af.inputs[0])
        ex_m0 = _math(nt, 'MULTIPLY', (-540, 2180), clamp=True)
        nt.links.new(ex_yg.outputs[0], ex_m0.inputs[0])
        nt.links.new(ex_af.outputs[0], ex_m0.inputs[1])
        ex_g = _math(nt, 'MULTIPLY', (-380, 2180), clamp=True, v1=exit_lip_soot_amp)
        nt.links.new(ex_m0.outputs[0], ex_g.inputs[0])
        ex_eff = _math(nt, 'MULTIPLY', (-220, 2180), clamp=True)
        nt.links.new(ex_g.outputs[0], ex_eff.inputs[0])
        nt.links.new(ctrl.outputs["Soot"], ex_eff.inputs[1])


        ex_col = _mix_rgba(nt, 'MIX', (940, 440), col_a=rgba(0x283546), col_b=rgba(0x161517))
        nt.links.new(ex_yg.outputs[0], ex_col.inputs[0])
        mix_ex_c = _mix_rgba(nt, 'MIX', (1180, 400))
        nt.links.new(ex_eff.outputs[0], mix_ex_c.inputs[0])
        nt.links.new(cur_col, mix_ex_c.inputs[6])
        nt.links.new(ex_col.outputs[2], mix_ex_c.inputs[7])
        cur_col = mix_ex_c.outputs[2]

        mix_ex_r = _mix_float(nt, (1180, 160), b=0.82)
        nt.links.new(ex_eff.outputs[0], mix_ex_r.inputs[0])
        nt.links.new(cur_rough, mix_ex_r.inputs[2])
        cur_rough = mix_ex_r.outputs[0]

        ex_md = _math(nt, 'MULTIPLY', (980, -300), clamp=True, v1=0.55)
        nt.links.new(ex_eff.outputs[0], ex_md.inputs[0])
        m_after_ex = _math(nt, 'SUBTRACT', (1180, -280), clamp=True)
        nt.links.new(cur_met, m_after_ex.inputs[0])
        nt.links.new(ex_md.outputs[0], m_after_ex.inputs[1])
        cur_met = m_after_ex.outputs[0]


    braid_bump_out = None
    if braid_micro_weave:
        w_br1 = _new_node(nt, "ShaderNodeTexWave", (-940, 2400))
        w_br1.wave_type = 'BANDS'
        w_br1.bands_direction = 'DIAGONAL'
        w_br1.inputs["Scale"].default_value = 340.0
        w_br1.inputs["Distortion"].default_value = 0.28
        nt.links.new(tc.outputs["Object"], w_br1.inputs["Vector"])

        mp_br2 = _new_node(nt, "ShaderNodeMapping", (-1120, 2540))
        mp_br2.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(90.0))
        nt.links.new(tc.outputs["Object"], mp_br2.inputs["Vector"])
        w_br2 = _new_node(nt, "ShaderNodeTexWave", (-940, 2540))
        w_br2.wave_type = 'BANDS'
        w_br2.bands_direction = 'DIAGONAL'
        w_br2.inputs["Scale"].default_value = 340.0
        w_br2.inputs["Distortion"].default_value = 0.28
        nt.links.new(mp_br2.outputs["Vector"], w_br2.inputs["Vector"])

        br_comb = _math(nt, 'MINIMUM', (-720, 2460), clamp=True)
        nt.links.new(w_br1.outputs["Fac"], br_comb.inputs[0])
        nt.links.new(w_br2.outputs["Fac"], br_comb.inputs[1])
        braid_bump_out = br_comb.outputs[0]


        br_val = _map_range(nt, 0.0, 0.36, 0.56, 0.0, (-520, 2460), clamp=True)
        nt.links.new(braid_bump_out, br_val.inputs[0])
        mix_br_c = _mix_rgba(nt, 'MIX', (1180, 360), col_b=rgba(0x101215))
        nt.links.new(br_val.outputs[0], mix_br_c.inputs[0])
        nt.links.new(cur_col, mix_br_c.inputs[6])
        cur_col = mix_br_c.outputs[2]
        mix_br_r = _mix_float(nt, (1180, 120), b=0.52)
        nt.links.new(br_val.outputs[0], mix_br_r.inputs[0])
        nt.links.new(cur_rough, mix_br_r.inputs[2])
        cur_rough = mix_br_r.outputs[0]


    rub_bump_out = None
    acoustic_bump_out = None
    if fan_rub_strip:
        sep_rb = _new_node(nt, "ShaderNodeSeparateXYZ", (-1120, 2120))
        nt.links.new(tc.outputs["Object"], sep_rb.inputs["Vector"])
        comb_rb = _new_node(nt, "ShaderNodeCombineXYZ", (-940, 2180))
        nt.links.new(sep_rb.outputs["X"], comb_rb.inputs["X"])
        nt.links.new(sep_rb.outputs["Z"], comb_rb.inputs["Z"])
        rad_rb = _vmath(nt, 'LENGTH', (-760, 2180))
        nt.links.new(comb_rb.outputs["Vector"], rad_rb.inputs[0])
        rb_rm = _map_range(nt, 0.483, 0.493, 1.0, 0.0, (-560, 2180), clamp=True)
        nt.links.new(rad_rb.outputs["Value"], rb_rm.inputs[0])
        rb_y1 = _map_range(nt, -4.15, -4.11, 0.0, 1.0, (-560, 2060), clamp=True)
        nt.links.new(sep_rb.outputs["Y"], rb_y1.inputs[0])
        rb_y2 = _map_range(nt, -3.80, -3.75, 1.0, 0.0, (-560, 1960), clamp=True)
        nt.links.new(sep_rb.outputs["Y"], rb_y2.inputs[0])
        rb_ym = _math(nt, 'MULTIPLY', (-380, 2020), clamp=True)
        nt.links.new(rb_y1.outputs[0], rb_ym.inputs[0])
        nt.links.new(rb_y2.outputs[0], rb_ym.inputs[1])
        rb_mask = _math(nt, 'MULTIPLY', (-200, 2100), clamp=True)
        nt.links.new(rb_rm.outputs[0], rb_mask.inputs[0])
        nt.links.new(rb_ym.outputs[0], rb_mask.inputs[1])

        w_rub = _new_node(nt, "ShaderNodeTexWave", (-560, 2320))
        w_rub.wave_type = 'BANDS'
        w_rub.bands_direction = 'Y'
        w_rub.inputs["Scale"].default_value = 380.0
        w_rub.inputs["Distortion"].default_value = 0.25
        nt.links.new(tc.outputs["Object"], w_rub.inputs["Vector"])
        rb_b = _math(nt, 'MULTIPLY', (-200, 2260), clamp=True)
        nt.links.new(w_rub.outputs["Fac"], rb_b.inputs[0])
        nt.links.new(rb_mask.outputs[0], rb_b.inputs[1])
        rub_bump_out = rb_b.outputs[0]

        rb_col = _mix_rgba(nt, 'MIX', (60, 2200), col_a=rgba(0x363A40), col_b=rgba(0x747A82))
        nt.links.new(w_rub.outputs["Fac"], rb_col.inputs[0])
        mix_rb_c = _mix_rgba(nt, 'MIX', (1060, 380))
        nt.links.new(rb_mask.outputs[0], mix_rb_c.inputs[0])
        nt.links.new(cur_col, mix_rb_c.inputs[6])
        nt.links.new(rb_col.outputs[2], mix_rb_c.inputs[7])
        cur_col = mix_rb_c.outputs[2]
        mix_rb_m = _mix_float(nt, (1060, -160), b=0.42)
        nt.links.new(rb_mask.outputs[0], mix_rb_m.inputs[0])
        nt.links.new(cur_met, mix_rb_m.inputs[2])
        cur_met = mix_rb_m.outputs[0]
        mix_rb_r = _mix_float(nt, (1060, 140), b=0.66)
        nt.links.new(rb_mask.outputs[0], mix_rb_r.inputs[0])
        nt.links.new(cur_rough, mix_rb_r.inputs[2])
        cur_rough = mix_rb_r.outputs[0]


        ac_y1 = _map_range(nt, -4.262, -4.250, 0.0, 1.0, (-560, 1840), clamp=True)
        nt.links.new(sep_rb.outputs["Y"], ac_y1.inputs[0])
        ac_y2 = _map_range(nt, -4.145, -4.115, 1.0, 0.0, (-560, 1740), clamp=True)
        nt.links.new(sep_rb.outputs["Y"], ac_y2.inputs[0])
        ac_ym = _math(nt, 'MULTIPLY', (-380, 1790), clamp=True)
        nt.links.new(ac_y1.outputs[0], ac_ym.inputs[0])
        nt.links.new(ac_y2.outputs[0], ac_ym.inputs[1])
        ac_mask = _math(nt, 'MULTIPLY', (-200, 1860), clamp=True)
        nt.links.new(rb_rm.outputs[0], ac_mask.inputs[0])
        nt.links.new(ac_ym.outputs[0], ac_mask.inputs[1])

        vor_ac = _voronoi(nt, scale=360.0, feature='F1', randomness=0.06, loc=(-380, 1620))
        nt.links.new(tc.outputs["Object"], vor_ac.inputs["Vector"])
        ac_dot = _map_range(nt, 0.16, 0.28, 1.0, 0.0, (-180, 1620), clamp=True)
        nt.links.new(vor_ac.outputs["Distance"], ac_dot.inputs[0])
        ac_h = _math(nt, 'MULTIPLY', (0, 1660), clamp=True)
        nt.links.new(ac_dot.outputs[0], ac_h.inputs[0])
        nt.links.new(ac_mask.outputs[0], ac_h.inputs[1])
        acoustic_bump_out = ac_h.outputs[0]

        ac_col = _mix_rgba(nt, 'MIX', (160, 1760), col_a=rgba(0x363A42), col_b=rgba(0x0D0F13))
        nt.links.new(ac_dot.outputs[0], ac_col.inputs[0])
        mix_ac_c = _mix_rgba(nt, 'MIX', (1080, 360))
        nt.links.new(ac_mask.outputs[0], mix_ac_c.inputs[0])
        nt.links.new(cur_col, mix_ac_c.inputs[6])
        nt.links.new(ac_col.outputs[2], mix_ac_c.inputs[7])
        cur_col = mix_ac_c.outputs[2]
        mix_ac_m = _mix_float(nt, (1080, -180), b=0.28)
        nt.links.new(ac_mask.outputs[0], mix_ac_m.inputs[0])
        nt.links.new(cur_met, mix_ac_m.inputs[2])
        cur_met = mix_ac_m.outputs[0]
        mix_ac_r = _mix_float(nt, (1080, 120), b=0.72)
        nt.links.new(ac_mask.outputs[0], mix_ac_r.inputs[0])
        nt.links.new(cur_rough, mix_ac_r.inputs[2])
        cur_rough = mix_ac_r.outputs[0]


    if blade_tip_erosion:
        geo_bl = _new_node(nt, "ShaderNodeNewGeometry", (-1300, 2460))
        sep_bl = _new_node(nt, "ShaderNodeSeparateXYZ", (-1120, 2460))
        nt.links.new(tc.outputs["Object"], sep_bl.inputs["Vector"])
        comb_bl = _new_node(nt, "ShaderNodeCombineXYZ", (-940, 2460))
        nt.links.new(sep_bl.outputs["X"], comb_bl.inputs["X"])
        nt.links.new(sep_bl.outputs["Z"], comb_bl.inputs["Z"])
        rad_bl = _vmath(nt, 'LENGTH', (-760, 2460))
        nt.links.new(comb_bl.outputs["Vector"], rad_bl.inputs[0])
        bl_tip = _map_range(nt, 0.345, 0.472, 0.0, 0.68, (-560, 2460), clamp=True)
        nt.links.new(rad_bl.outputs["Value"], bl_tip.inputs[0])
        bl_e = _math(nt, 'MULTIPLY', (-380, 2460), clamp=True)
        nt.links.new(bl_tip.outputs[0], bl_e.inputs[0])
        nt.links.new(masks.outputs["EdgeWear_Mask"], bl_e.inputs[1])
        bl_ef0 = _math(nt, 'MULTIPLY_ADD', (-200, 2460), clamp=True, v1=0.58)
        nt.links.new(bl_tip.outputs[0], bl_ef0.inputs[0])
        nt.links.new(bl_e.outputs[0], bl_ef0.inputs[2])


        sep_bln = _new_node(nt, "ShaderNodeSeparateXYZ", (-1120, 2580))
        nt.links.new(geo_bl.outputs["Normal"], sep_bln.inputs["Vector"])
        bl_le = _map_range(nt, -0.32, -0.82, 0.0, 0.88, (-560, 2580), clamp=True)
        nt.links.new(sep_bln.outputs["Y"], bl_le.inputs[0])
        bl_ef = _math(nt, 'MAXIMUM', (-40, 2520), clamp=True)
        nt.links.new(bl_ef0.outputs[0], bl_ef.inputs[0])
        nt.links.new(bl_le.outputs[0], bl_ef.inputs[1])


        mix_bl_c = _mix_rgba(nt, 'MIX', (1060, 340), col_b=rgba(0xC6CED6))
        nt.links.new(bl_ef.outputs[0], mix_bl_c.inputs[0])
        nt.links.new(cur_col, mix_bl_c.inputs[6])
        cur_col = mix_bl_c.outputs[2]
        mix_bl_r = _mix_float(nt, (1060, 100), b=0.14)
        nt.links.new(bl_ef.outputs[0], mix_bl_r.inputs[0])
        nt.links.new(cur_rough, mix_bl_r.inputs[2])
        cur_rough = mix_bl_r.outputs[0]


    if igv_leading_edge:
        geo_ig = _new_node(nt, "ShaderNodeNewGeometry", (-1300, 2760))
        sep_igp = _new_node(nt, "ShaderNodeSeparateXYZ", (-1120, 2760))
        nt.links.new(tc.outputs["Object"], sep_igp.inputs["Vector"])
        comb_ig = _new_node(nt, "ShaderNodeCombineXYZ", (-940, 2760))
        nt.links.new(sep_igp.outputs["X"], comb_ig.inputs["X"])
        nt.links.new(sep_igp.outputs["Z"], comb_ig.inputs["Z"])
        rad_ig = _vmath(nt, 'LENGTH', (-760, 2760))
        nt.links.new(comb_ig.outputs["Vector"], rad_ig.inputs[0])
        ig_rm = _map_range(nt, 0.174, 0.195, 0.0, 1.0, (-560, 2760), clamp=True)
        nt.links.new(rad_ig.outputs["Value"], ig_rm.inputs[0])

        sep_ign = _new_node(nt, "ShaderNodeSeparateXYZ", (-1120, 2880))
        nt.links.new(geo_ig.outputs["Normal"], sep_ign.inputs["Vector"])
        ig_le = _map_range(nt, -0.40, -0.86, 0.0, 0.85, (-560, 2880), clamp=True)
        nt.links.new(sep_ign.outputs["Y"], ig_le.inputs[0])
        ig_eff = _math(nt, 'MULTIPLY', (-320, 2820), clamp=True)
        nt.links.new(ig_rm.outputs[0], ig_eff.inputs[0])
        nt.links.new(ig_le.outputs[0], ig_eff.inputs[1])


        vor_ige = _voronoi(nt, scale=680.0, feature='F1', randomness=0.95, loc=(-760, 2940))
        nt.links.new(tc.outputs["Object"], vor_ige.inputs["Vector"])
        ige_dot = _map_range(nt, 0.0, 0.22, 0.65, 0.0, (-560, 2940), clamp=True)
        nt.links.new(vor_ige.outputs["Distance"], ige_dot.inputs[0])
        ige_m0 = _math(nt, 'MULTIPLY', (-320, 2940), clamp=True)
        nt.links.new(ig_eff.outputs[0], ige_m0.inputs[0])
        nt.links.new(ige_dot.outputs[0], ige_m0.inputs[1])
        ige_w = _math(nt, 'MULTIPLY', (-140, 2940), clamp=True)
        nt.links.new(ige_m0.outputs[0], ige_w.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], ige_w.inputs[1])


        mix_ig_c = _mix_rgba(nt, 'MIX', (1080, 320), col_b=rgba(0xC4CCD4))
        nt.links.new(ig_eff.outputs[0], mix_ig_c.inputs[0])
        nt.links.new(cur_col, mix_ig_c.inputs[6])
        mix_ige_c = _mix_rgba(nt, 'MIX', (1120, 320), col_b=rgba(0xB8C0C8))
        nt.links.new(ige_w.outputs[0], mix_ige_c.inputs[0])
        nt.links.new(mix_ig_c.outputs[2], mix_ige_c.inputs[6])
        cur_col = mix_ige_c.outputs[2]
        mix_ig_r = _mix_float(nt, (1080, 80), b=0.14)
        nt.links.new(ig_eff.outputs[0], mix_ig_r.inputs[0])
        nt.links.new(cur_rough, mix_ig_r.inputs[2])
        mix_ige_r = _mix_float(nt, (1120, 80), b=0.46)
        nt.links.new(ige_w.outputs[0], mix_ige_r.inputs[0])
        nt.links.new(mix_ig_r.outputs[0], mix_ige_r.inputs[2])
        cur_rough = mix_ige_r.outputs[0]


    fluid_band_bump_out = None
    if fluid_id_bands:
        geo_fb = _new_node(nt, "ShaderNodeNewGeometry", (-1300, 3040))
        sep_fb = _new_node(nt, "ShaderNodeSeparateXYZ", (-1120, 3040))
        nt.links.new(geo_fb.outputs["Position"], sep_fb.inputs["Vector"])

        fb_core = _map_range(nt, -1.24, -1.16, 1.0, 0.0, (-920, 3040), clamp=True)
        nt.links.new(sep_fb.outputs["Y"], fb_core.inputs[0])

        fb_i1 = _map_range(nt, 0.155, 0.165, 0.0, 1.0, (-920, 3160), clamp=True, interp='LINEAR')
        nt.links.new(masks.outputs["Island_Rand"], fb_i1.inputs[0])
        fb_i2 = _map_range(nt, 0.375, 0.385, 1.0, 0.0, (-920, 3260), clamp=True, interp='LINEAR')
        nt.links.new(masks.outputs["Island_Rand"], fb_i2.inputs[0])
        fb_im = _math(nt, 'MULTIPLY', (-720, 3200), clamp=True)
        nt.links.new(fb_i1.outputs[0], fb_im.inputs[0])
        nt.links.new(fb_i2.outputs[0], fb_im.inputs[1])


        fb_yz = _math(nt, 'MULTIPLY_ADD', (-920, 2940), v1=0.42)
        nt.links.new(sep_fb.outputs["Z"], fb_yz.inputs[0])
        nt.links.new(sep_fb.outputs["Y"], fb_yz.inputs[2])
        fb_ph = _math(nt, 'MULTIPLY_ADD', (-720, 2940), v1=3.20)
        nt.links.new(fb_yz.outputs[0], fb_ph.inputs[0])
        nt.links.new(masks.outputs["Island_Rand"], fb_ph.inputs[2])
        fb_fr = _math(nt, 'FRACT', (-560, 2940))
        nt.links.new(fb_ph.outputs[0], fb_fr.inputs[0])
        fb_df = _math(nt, 'SUBTRACT', (-400, 2940), v1=0.50)
        nt.links.new(fb_fr.outputs[0], fb_df.inputs[0])
        fb_ad = _math(nt, 'ABSOLUTE', (-240, 2940))
        nt.links.new(fb_df.outputs[0], fb_ad.inputs[0])
        fb_stripe = _map_range(nt, 0.020, 0.024, 1.0, 0.0, (-80, 2940), clamp=True, interp='LINEAR')
        nt.links.new(fb_ad.outputs[0], fb_stripe.inputs[0])

        fb_m0 = _math(nt, 'MULTIPLY', (100, 3040), clamp=True)
        nt.links.new(fb_core.outputs[0], fb_m0.inputs[0])
        nt.links.new(fb_im.outputs[0], fb_m0.inputs[1])
        fb_eff = _math(nt, 'MULTIPLY', (260, 3040), clamp=True)
        nt.links.new(fb_m0.outputs[0], fb_eff.inputs[0])
        nt.links.new(fb_stripe.outputs[0], fb_eff.inputs[1])
        fluid_band_bump_out = fb_eff.outputs[0]


        fb_sel = _map_range(nt, 0.16, 0.38, 0.0, 1.0, (-400, 3160), clamp=True)
        nt.links.new(masks.outputs["Island_Rand"], fb_sel.inputs[0])
        ramp_fb = _ramp(nt, [
            (0.00, rgba(0xC22018)),
            (0.28, rgba(0xC22018)),
            (0.32, rgba(0x1E6CC8)),
            (0.60, rgba(0x1E6CC8)),
            (0.64, rgba(0xDFAA14)),
            (0.84, rgba(0xDFAA14)),
            (0.88, rgba(0xD8DEE4)),
            (1.00, rgba(0xD8DEE4)),
        ], loc=(-180, 3160))
        nt.links.new(fb_sel.outputs[0], ramp_fb.inputs["Fac"])


        fb_ctr = _map_range(nt, 0.006, 0.008, 0.55, 0.0, (-80, 3060), clamp=True, interp='LINEAR')
        nt.links.new(fb_ad.outputs[0], fb_ctr.inputs[0])
        fb_col_mix = _mix_rgba(nt, 'MIX', (100, 3160), col_b=rgba(0xECBC1C))
        nt.links.new(fb_ctr.outputs[0], fb_col_mix.inputs[0])
        nt.links.new(ramp_fb.outputs["Color"], fb_col_mix.inputs[6])

        mix_fb_c = _mix_rgba(nt, 'MIX', (1100, 300))
        nt.links.new(fb_eff.outputs[0], mix_fb_c.inputs[0])
        nt.links.new(cur_col, mix_fb_c.inputs[6])
        nt.links.new(fb_col_mix.outputs[2], mix_fb_c.inputs[7])
        cur_col = mix_fb_c.outputs[2]
        mix_fb_m = _mix_float(nt, (1100, -200), b=0.0)
        nt.links.new(fb_eff.outputs[0], mix_fb_m.inputs[0])
        nt.links.new(cur_met, mix_fb_m.inputs[2])
        cur_met = mix_fb_m.outputs[0]
        mix_fb_r = _mix_float(nt, (1100, 60), b=0.22)
        nt.links.new(fb_eff.outputs[0], mix_fb_r.inputs[0])
        nt.links.new(cur_rough, mix_fb_r.inputs[2])
        cur_rough = mix_fb_r.outputs[0]


    pit_bump_out = None
    if spinner_nose_wear:
        sep_sn = _new_node(nt, "ShaderNodeSeparateXYZ", (-1120, 2660))
        nt.links.new(tc.outputs["Object"], sep_sn.inputs["Vector"])
        sn_ym = _map_range(nt, -4.32, -4.58, 0.0, 1.0, (-860, 2660), clamp=True)
        nt.links.new(sep_sn.outputs["Y"], sn_ym.inputs[0])

        vor_pt1 = _voronoi(nt, scale=460.0, feature='F1', randomness=0.96, loc=(-860, 2800))
        nt.links.new(tc.outputs["Object"], vor_pt1.inputs["Vector"])
        pt_d1 = _map_range(nt, 0.0, 0.19, 1.0, 0.0, (-640, 2800), clamp=True)
        nt.links.new(vor_pt1.outputs["Distance"], pt_d1.inputs[0])

        vor_pt2 = _voronoi(nt, scale=820.0, feature='F1', randomness=0.97, loc=(-860, 2900))
        nt.links.new(tc.outputs["Object"], vor_pt2.inputs["Vector"])
        pt_d2 = _map_range(nt, 0.0, 0.24, 0.72, 0.0, (-640, 2900), clamp=True)
        nt.links.new(vor_pt2.outputs["Distance"], pt_d2.inputs[0])

        pt_comb = _math(nt, 'MAXIMUM', (-540, 2820), clamp=True)
        nt.links.new(pt_d1.outputs[0], pt_comb.inputs[0])
        nt.links.new(pt_d2.outputs[0], pt_comb.inputs[1])

        pt_m0 = _math(nt, 'MULTIPLY', (-440, 2740), clamp=True)
        nt.links.new(pt_comb.outputs[0], pt_m0.inputs[0])
        nt.links.new(sn_ym.outputs[0], pt_m0.inputs[1])
        pt_m = _math(nt, 'MULTIPLY', (-300, 2740), clamp=True)
        nt.links.new(pt_m0.outputs[0], pt_m.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], pt_m.inputs[1])
        pit_bump_out = pt_m.outputs[0]


        sn_h = _math(nt, 'MULTIPLY', (-440, 2600), clamp=True, v1=0.22)
        nt.links.new(sn_ym.outputs[0], sn_h.inputs[0])
        mix_sn_h = _mix_rgba(nt, 'MIX', (1040, 320), col_b=rgba(0xB6A482))
        nt.links.new(sn_h.outputs[0], mix_sn_h.inputs[0])
        nt.links.new(cur_col, mix_sn_h.inputs[6])
        mix_sn_p = _mix_rgba(nt, 'MIX', (1100, 320), col_b=rgba(0xDFE5EC))
        nt.links.new(pt_m.outputs[0], mix_sn_p.inputs[0])
        nt.links.new(mix_sn_h.outputs[2], mix_sn_p.inputs[6])
        cur_col = mix_sn_p.outputs[2]

        mix_sn_r = _mix_float(nt, (1100, 80), b=0.52)
        nt.links.new(pt_m.outputs[0], mix_sn_r.inputs[0])
        nt.links.new(cur_rough, mix_sn_r.inputs[2])
        cur_rough = mix_sn_r.outputs[0]


    if nozzle_rub_arcs:
        sep_nr = _new_node(nt, "ShaderNodeSeparateXYZ", (-1120, 3320))
        nt.links.new(tc.outputs["Object"], sep_nr.inputs["Vector"])
        nr_ax = _math(nt, 'ABSOLUTE', (-940, 3320))
        nt.links.new(sep_nr.outputs["X"], nr_ax.inputs[0])

        nr_x0 = _map_range(nt, 0.028, 0.042, 0.0, 1.0, (-760, 3380), clamp=True)
        nt.links.new(nr_ax.outputs[0], nr_x0.inputs[0])
        nr_x1 = _map_range(nt, 0.062, 0.076, 1.0, 0.0, (-760, 3260), clamp=True)
        nt.links.new(nr_ax.outputs[0], nr_x1.inputs[0])
        nr_xm = _math(nt, 'MULTIPLY', (-580, 3320), clamp=True)
        nt.links.new(nr_x0.outputs[0], nr_xm.inputs[0])
        nt.links.new(nr_x1.outputs[0], nr_xm.inputs[1])


        nr_y0 = _map_range(nt, 0.01, 0.08, 0.0, 1.0, (-760, 3140), clamp=True)
        nt.links.new(sep_nr.outputs["Y"], nr_y0.inputs[0])
        nr_y1 = _map_range(nt, 0.80, 0.88, 1.0, 0.0, (-760, 3040), clamp=True)
        nt.links.new(sep_nr.outputs["Y"], nr_y1.inputs[0])
        nr_ym = _math(nt, 'MULTIPLY', (-580, 3100), clamp=True)
        nt.links.new(nr_y0.outputs[0], nr_ym.inputs[0])
        nt.links.new(nr_y1.outputs[0], nr_ym.inputs[1])

        nr_zone = _math(nt, 'MULTIPLY', (-400, 3240), clamp=True)
        nt.links.new(nr_xm.outputs[0], nr_zone.inputs[0])
        nt.links.new(nr_ym.outputs[0], nr_zone.inputs[1])


        mp_nr = _new_node(nt, "ShaderNodeMapping", (-960, 3500))
        mp_nr.inputs["Scale"].default_value = (82.0, 3.6, 82.0)
        nt.links.new(tc.outputs["Object"], mp_nr.inputs["Vector"])
        n_nr = _noise(nt, scale=1.0, detail=7.0, roughness=0.62, distortion=0.14, loc=(-760, 3500))
        nt.links.new(mp_nr.outputs["Vector"], n_nr.inputs["Vector"])


        nr_mos0 = _map_range(nt, 0.36, 0.58, 0.0, 0.44, (-540, 3560), clamp=True)
        nt.links.new(n_nr.outputs["Factor"], nr_mos0.inputs[0])
        nr_mos1 = _math(nt, 'MULTIPLY', (-340, 3520), clamp=True)
        nt.links.new(nr_mos0.outputs[0], nr_mos1.inputs[0])
        nt.links.new(nr_zone.outputs[0], nr_mos1.inputs[1])
        nr_mos = _math(nt, 'MULTIPLY', (-180, 3520), clamp=True)
        nt.links.new(nr_mos1.outputs[0], nr_mos.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], nr_mos.inputs[1])

        mix_mos_c = _mix_rgba(nt, 'MIX', (1120, 280), col_b=rgba(0x181A1E))
        nt.links.new(nr_mos.outputs[0], mix_mos_c.inputs[0])
        nt.links.new(cur_col, mix_mos_c.inputs[6])
        cur_col = mix_mos_c.outputs[2]
        mix_mos_r = _mix_float(nt, (1120, 40), b=0.48)
        nt.links.new(nr_mos.outputs[0], mix_mos_r.inputs[0])
        nt.links.new(cur_rough, mix_mos_r.inputs[2])
        cur_rough = mix_mos_r.outputs[0]


        nr_rub0 = _map_range(nt, 0.57, 0.76, 0.0, 0.72, (-540, 3420), clamp=True)
        nt.links.new(n_nr.outputs["Factor"], nr_rub0.inputs[0])
        nr_rub1 = _math(nt, 'MULTIPLY', (-340, 3380), clamp=True)
        nt.links.new(nr_rub0.outputs[0], nr_rub1.inputs[0])
        nt.links.new(nr_zone.outputs[0], nr_rub1.inputs[1])
        nr_rub = _math(nt, 'MULTIPLY', (-180, 3380), clamp=True)
        nt.links.new(nr_rub1.outputs[0], nr_rub.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], nr_rub.inputs[1])

        mix_rub_c = _mix_rgba(nt, 'MIX', (1160, 280), col_b=rgba(0xBAC2CC))
        nt.links.new(nr_rub.outputs[0], mix_rub_c.inputs[0])
        nt.links.new(cur_col, mix_rub_c.inputs[6])
        cur_col = mix_rub_c.outputs[2]
        mix_rub_r = _mix_float(nt, (1160, 40), b=0.12)
        nt.links.new(nr_rub.outputs[0], mix_rub_r.inputs[0])
        nt.links.new(cur_rough, mix_rub_r.inputs[2])
        cur_rough = mix_rub_r.outputs[0]


    if torque_seal_amp > 1e-4:
        ts_isl = _map_range(nt, 0.872, 0.888, 0.0, 1.0, (680, 560), clamp=True, interp='LINEAR')
        nt.links.new(masks.outputs["Island_Rand"], ts_isl.inputs[0])
        cv_lo = 0.54 if torque_seal_tight else 0.44
        cv_hi = 0.84 if torque_seal_tight else 0.78
        ts_cv = _map_range(nt, cv_lo, cv_hi, 0.0, 1.0, (680, 440), clamp=True)
        nt.links.new(masks.outputs["Topo_Convex"], ts_cv.inputs[0])


        geo_ts = _new_node(nt, "ShaderNodeNewGeometry", (320, 760))
        sep_ts = _new_node(nt, "ShaderNodeSeparateXYZ", (480, 760))
        nt.links.new(geo_ts.outputs["Position"], sep_ts.inputs["Vector"])
        comb_tsr = _new_node(nt, "ShaderNodeCombineXYZ", (640, 820))
        nt.links.new(sep_ts.outputs["X"], comb_tsr.inputs["X"])
        nt.links.new(sep_ts.outputs["Z"], comb_tsr.inputs["Z"])
        rad_ts = _vmath(nt, 'LENGTH', (780, 820))
        nt.links.new(comb_tsr.outputs["Vector"], rad_ts.inputs[0])
        ts_out_r = _map_range(nt, 0.522, 0.542, 0.0, 1.0, (920, 820), clamp=True)
        nt.links.new(rad_ts.outputs["Value"], ts_out_r.inputs[0])
        ts_fwd_y = _map_range(nt, -1.92, -1.84, 1.0, 0.0, (920, 700), clamp=True)
        nt.links.new(sep_ts.outputs["Y"], ts_fwd_y.inputs[0])
        ts_safe = _math(nt, 'MAXIMUM', (1060, 760), clamp=True)
        nt.links.new(ts_out_r.outputs[0], ts_safe.inputs[0])
        nt.links.new(ts_fwd_y.outputs[0], ts_safe.inputs[1])

        ts_m0 = _math(nt, 'MULTIPLY', (860, 500), clamp=True)
        nt.links.new(ts_isl.outputs[0], ts_m0.inputs[0])
        nt.links.new(ts_cv.outputs[0], ts_m0.inputs[1])

        if torque_seal_tight:
            n_tspot = _noise(nt, scale=68.0, detail=3.0, roughness=0.45, loc=(680, 320))
            nt.links.new(geo_ts.outputs["Position"], n_tspot.inputs["Vector"])
            ts_sp = _map_range(nt, 0.54, 0.66, 0.0, 1.0, (820, 340), clamp=True)
            nt.links.new(n_tspot.outputs["Factor"], ts_sp.inputs[0])
            ts_m0b = _math(nt, 'MULTIPLY', (910, 440), clamp=True)
            nt.links.new(ts_m0.outputs[0], ts_m0b.inputs[0])
            nt.links.new(ts_sp.outputs[0], ts_m0b.inputs[1])
            ts_m0_out = ts_m0b.outputs[0]
        else:
            ts_m0_out = ts_m0.outputs[0]
        ts_m1 = _math(nt, 'MULTIPLY', (960, 500), clamp=True)
        nt.links.new(ts_m0_out, ts_m1.inputs[0])
        nt.links.new(ts_safe.outputs[0], ts_m1.inputs[1])
        ts_eff = _math(nt, 'MULTIPLY', (1060, 500), clamp=True, v1=torque_seal_amp)
        nt.links.new(ts_m1.outputs[0], ts_eff.inputs[0])


        ts_sel = _map_range(nt, 0.870, 0.995, 0.0, 1.0, (860, 640), clamp=True)
        nt.links.new(masks.outputs["Island_Rand"], ts_sel.inputs[0])
        ramp_ts = _ramp(nt, [
            (0.00, rgba(0xF0820A)),
            (0.42, rgba(0xF0820A)),
            (0.48, rgba(0xD4D814)),
            (0.74, rgba(0xD4D814)),
            (0.80, rgba(0xD6221A)),
            (1.00, rgba(0xD6221A)),
        ], loc=(980, 660))
        nt.links.new(ts_sel.outputs[0], ramp_ts.inputs["Fac"])

        mix_ts_c = _mix_rgba(nt, 'MIX', (1180, 300))
        nt.links.new(ts_eff.outputs[0], mix_ts_c.inputs[0])
        nt.links.new(cur_col, mix_ts_c.inputs[6])
        nt.links.new(ramp_ts.outputs["Color"], mix_ts_c.inputs[7])
        cur_col = mix_ts_c.outputs[2]

        mix_ts_m = _mix_float(nt, (1180, -100), b=0.0)
        nt.links.new(ts_eff.outputs[0], mix_ts_m.inputs[0])
        nt.links.new(cur_met, mix_ts_m.inputs[2])
        cur_met = mix_ts_m.outputs[0]

        mix_ts_r = _mix_float(nt, (1180, 80), b=0.32)
        nt.links.new(ts_eff.outputs[0], mix_ts_r.inputs[0])
        nt.links.new(cur_rough, mix_ts_r.inputs[2])
        cur_rough = mix_ts_r.outputs[0]


        ts_coat = _math(nt, 'MULTIPLY', (1180, -240), clamp=True, v1=0.24)
        nt.links.new(ts_eff.outputs[0], ts_coat.inputs[0])
        if cur_coat_w is not None:
            ts_cmax = _math(nt, 'MAXIMUM', (1260, -240), clamp=True)
            nt.links.new(cur_coat_w, ts_cmax.inputs[0])
            nt.links.new(ts_coat.outputs[0], ts_cmax.inputs[1])
            cur_coat_w = ts_cmax.outputs[0]
        else:
            cur_coat_w = ts_coat.outputs[0]


    cur_norm = masks.outputs["Bevel_Normal"]
    macro_h_terms = []
    micro_h_terms = []

    if oil_canning_bump > 1e-4:
        map_oc = _new_node(nt, "ShaderNodeMapping", (-120, -1180))
        map_oc.inputs["Scale"].default_value = (5.5, 8.5, 5.5)
        nt.links.new(tc.outputs["Object"], map_oc.inputs["Vector"])
        n_oc = _noise(nt, scale=1.0, detail=2.5, roughness=0.45, distortion=0.20, loc=(60, -1180))
        nt.links.new(map_oc.outputs["Vector"], n_oc.inputs["Vector"])
        macro_h_terms.append((n_oc.outputs["Factor"], oil_canning_bump, 0.0028))

    if rust_height_out is not None:
        macro_h_terms.append((rust_height_out, 0.18, 0.0007))

    if effusion_bump_out is not None:
        macro_h_terms.append((effusion_bump_out, 0.24, 0.0006))

    if lathe_wave_out is not None and lathe_wave_bump > 1e-4:
        micro_h_terms.append((lathe_wave_out, lathe_wave_bump, 0.00025))


    if brush_bump > 1e-4:
        micro_h_terms.append((br_fused.outputs[0], brush_bump, 0.00035))

    if grain_bump > 1e-4:
        micro_h_terms.append((masks.outputs["Micro_Grain"], grain_bump, 0.00040))

    if waffle_etch_bump_out is not None:
        micro_h_terms.append((waffle_etch_bump_out, 0.022, 0.00025))

    if spangle_bump_out is not None:
        micro_h_terms.append((spangle_bump_out, 0.018 * spangle_shimmer, 0.00022))

    if oxide_bump_out is not None:
        micro_h_terms.append((oxide_bump_out, 0.036 * oxide_scale_amp, 0.00030))

    if cast_voronoi_scale > 0.0 and cast_voronoi_bump > 1e-4:
        vor = _voronoi(nt, scale=cast_voronoi_scale, feature='F1', randomness=0.92, loc=(240, -1020))
        nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
        v_dist = 0.00024 if cast_voronoi_scale > 900.0 else 0.00060
        micro_h_terms.append((vor.outputs["Distance"], cast_voronoi_bump, v_dist))

    if scratch_eff_out is not None:
        sc_inv = _math(nt, 'SUBTRACT', (820, -960), clamp=True, v0=1.0)
        nt.links.new(scratch_eff_out, sc_inv.inputs[1])
        micro_h_terms.append((sc_inv.outputs[0], 0.14, 0.00045))

    if rub_bump_out is not None:
        micro_h_terms.append((rub_bump_out, 0.075, 0.00035))

    if acoustic_bump_out is not None:
        ac_inv = _math(nt, 'SUBTRACT', (1140, -1140), clamp=True, v0=1.0)
        nt.links.new(acoustic_bump_out, ac_inv.inputs[1])
        micro_h_terms.append((ac_inv.outputs[0], 0.16, 0.00040))

    if fluid_band_bump_out is not None:
        micro_h_terms.append((fluid_band_bump_out, 0.065, 0.00018))

    if pit_bump_out is not None:
        b_pt_inv = _math(nt, 'SUBTRACT', (1140, -1200), clamp=True, v0=1.0)
        nt.links.new(pit_bump_out, b_pt_inv.inputs[1])
        micro_h_terms.append((b_pt_inv.outputs[0], 0.065, 0.00025))

    if braid_bump_out is not None:
        micro_h_terms.append((braid_bump_out, 0.048, 0.00018))

    if nickel_as_bump_out is not None:
        micro_h_terms.append((nickel_as_bump_out, 0.055, 0.00032))

    cur_norm = _combine_scalar_heights(nt, macro_h_terms, cur_norm, ref_distance=0.0014, loc=(660, -880))
    cur_norm = _combine_scalar_heights(nt, micro_h_terms, cur_norm, ref_distance=0.0004, loc=(1100, -880))


    if galv_tilt_vec_out is not None:
        gn_add = _vmath(nt, 'ADD', (1200, -880))
        nt.links.new(cur_norm, gn_add.inputs[0])
        nt.links.new(galv_tilt_vec_out, gn_add.inputs[1])
        gn_nrm = _vmath(nt, 'NORMALIZE', (1300, -880))
        nt.links.new(gn_add.outputs["Vector"], gn_nrm.inputs[0])
        cur_norm = gn_nrm.outputs["Vector"]


    ps_w = _math(nt, 'MULTIPLY', (1120, -340), clamp=True, v1=0.82)
    nt.links.new(masks.outputs["Photo_Smudge"], ps_w.inputs[0])
    ps_eff = _math(nt, 'MULTIPLY', (1200, -340), clamp=True)
    nt.links.new(ps_w.outputs[0], ps_eff.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], ps_eff.inputs[1])

    ps_roff = _map_range(nt, 0.0, 1.0, -0.028, +0.068, (1240, -180), clamp=True)
    nt.links.new(ps_eff.outputs[0], ps_roff.inputs[0])
    r_after_ps = _math(nt, 'ADD', (1280, 40), clamp=True)
    nt.links.new(cur_rough, r_after_ps.inputs[0])
    nt.links.new(ps_roff.outputs[0], r_after_ps.inputs[1])
    cur_rough = r_after_ps.outputs[0]

    ps_col_f = _math(nt, 'MULTIPLY', (1240, 220), clamp=True, v1=0.13)
    nt.links.new(ps_eff.outputs[0], ps_col_f.inputs[0])
    mix_ps_c = _mix_rgba(nt, 'MIX', (1280, 260), col_b=rgba(0x282520))
    nt.links.new(ps_col_f.outputs[0], mix_ps_c.inputs[0])
    nt.links.new(cur_col, mix_ps_c.inputs[6])
    cur_col = mix_ps_c.outputs[2]


    fp_m0 = _math(nt, 'MULTIPLY', (1040, -480), clamp=True)
    nt.links.new(masks.outputs["Photo_Fingerprint"], fp_m0.inputs[0])
    nt.links.new(masks.outputs["Exposure_Mask"], fp_m0.inputs[1])
    fp_eff = _math(nt, 'MULTIPLY', (1140, -480), clamp=True)
    nt.links.new(fp_m0.outputs[0], fp_eff.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], fp_eff.inputs[1])
    fp_roff = _map_range(nt, 0.36, 0.82, 0.0, +0.036, (1220, -480), clamp=True)
    nt.links.new(fp_eff.outputs[0], fp_roff.inputs[0])
    r_after_fp = _math(nt, 'ADD', (1295, 20), clamp=True)
    nt.links.new(cur_rough, r_after_fp.inputs[0])
    nt.links.new(fp_roff.outputs[0], r_after_fp.inputs[1])
    cur_rough = r_after_fp.outputs[0]


    if soot_amp > 1e-4 or rivet_streaks:
        af_gain = max(0.22, min(0.54, soot_amp * 1.10 + (0.18 if rivet_streaks else 0.0)))
        af_m0 = _math(nt, 'MULTIPLY', (1040, -640), clamp=True, v1=af_gain)
        nt.links.new(masks.outputs["Photo_Airflow"], af_m0.inputs[0])
        af_eff = _math(nt, 'MULTIPLY', (1140, -640), clamp=True)
        nt.links.new(af_m0.outputs[0], af_eff.inputs[0])
        nt.links.new(ctrl.outputs["Soot"], af_eff.inputs[1])

        af_cf = _math(nt, 'MULTIPLY', (1220, 320), clamp=True, v1=0.35)
        nt.links.new(af_eff.outputs[0], af_cf.inputs[0])
        mix_af_mc = _mix_rgba(nt, 'MIX', (1295, 320), col_b=rgba(0x181614))
        nt.links.new(af_cf.outputs[0], mix_af_mc.inputs[0])
        nt.links.new(cur_col, mix_af_mc.inputs[6])
        cur_col = mix_af_mc.outputs[2]

        af_rf = _math(nt, 'MULTIPLY', (1220, -640), clamp=True, v1=0.082)
        nt.links.new(af_eff.outputs[0], af_rf.inputs[0])
        r_after_af = _math(nt, 'ADD', (1305, 0), clamp=True)
        nt.links.new(cur_rough, r_after_af.inputs[0])
        nt.links.new(af_rf.outputs[0], r_after_af.inputs[1])
        cur_rough = r_after_af.outputs[0]


    eff_r_floor = rough_floor if rough_floor is not None else max(0.09, min(rough_min * 0.72, wear_rough * 0.90))
    r_floor_node = _math(nt, 'MAXIMUM', (1320, -40), clamp=True, v1=eff_r_floor)
    r_floor_node.name = "Metal_Roughness_Floor"
    nt.links.new(cur_rough, r_floor_node.inputs[0])
    cur_rough = r_floor_node.outputs[0]


    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (1340, 120))
    nt.links.new(cur_col, bsdf.inputs["Base Color"])
    nt.links.new(cur_met, bsdf.inputs["Metallic"])
    nt.links.new(cur_rough, bsdf.inputs["Roughness"])
    nt.links.new(cur_norm, bsdf.inputs["Normal"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.50 if metallic <= 0.5 else specular
    if metallic > 0.5:
        nt.links.new(ht_grp.outputs["Specular_Tint"], bsdf.inputs["Specular Tint"])
    eff_aniso = 0.0 if tangent_mode == 'NONE' else aniso
    bsdf.inputs["Anisotropic"].default_value = eff_aniso
    if eff_aniso > 1e-4 and tangent_mode != 'NONE':
        if tangent_mode == 'WHEEL_AXLE_X' and wheel_local_vec_out is not None:
            wh_crs = _vmath(nt, 'CROSS_PRODUCT', (1020, -340))
            wh_crs.inputs[0].default_value = (1.0, 0.0, 0.0)
            nt.links.new(wheel_local_vec_out, wh_crs.inputs[1])
            wh_nrm = _vmath(nt, 'NORMALIZE', (1180, -340))
            nt.links.new(wh_crs.outputs["Vector"], wh_nrm.inputs[0])
            nt.links.new(wh_nrm.outputs["Vector"], bsdf.inputs["Tangent"])
        else:
            tang = _new_node(nt, "ShaderNodeTangent", (1020, -340))
            if tangent_mode == 'RADIAL_X':
                tang.direction_type = 'RADIAL'
                tang.axis = 'X'
            elif tangent_mode == 'RADIAL_Z':
                tang.direction_type = 'RADIAL'
                tang.axis = 'Z'
            elif tangent_mode == 'UV':
                tang.direction_type = 'UV_MAP'
            else:
                tang.direction_type = 'RADIAL'
                tang.axis = 'Y'
            nt.links.new(tang.outputs["Tangent"], bsdf.inputs["Tangent"])
        if aniso_rot_var > 1e-4:
            n_arot = _noise(nt, scale=3.6, detail=3.0, roughness=0.55, distortion=0.85, loc=(1020, -480))
            nt.links.new(tc.outputs["Object"], n_arot.inputs["Vector"])
            arot_m = _map_range(nt, 0.20, 0.80, aniso_rot_base, aniso_rot_base + aniso_rot_var, (1180, -480), clamp=True)
            nt.links.new(n_arot.outputs["Factor"], arot_m.inputs[0])
            nt.links.new(arot_m.outputs[0], bsdf.inputs["Anisotropic Rotation"])
        elif abs(aniso_rot_base) > 1e-4:
            bsdf.inputs["Anisotropic Rotation"].default_value = aniso_rot_base
    if tf_thickness_out is not None and "Thin Film Thickness" in bsdf.inputs:
        nt.links.new(tf_thickness_out, bsdf.inputs["Thin Film Thickness"])
        if "Thin Film IOR" in bsdf.inputs:
            bsdf.inputs["Thin Film IOR"].default_value = 2.25
    if cur_coat_w is not None:
        nt.links.new(cur_coat_w, bsdf.inputs["Coat Weight"])
    else:
        bsdf.inputs["Coat Weight"].default_value = coat_weight
    bsdf.inputs["Coat Roughness"].default_value = cur_coat_r
    if (cur_coat_w is not None or coat_weight > 1e-4) and "Coat Normal" in bsdf.inputs:
        nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Coat Normal"])


    wrap = _new_node(nt, "ShaderNodeGroup", (1660, 120))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(id_hex)
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (1900, 120))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])


    mat["f15_id_hex"] = int(id_hex)
    mat["f15_pbr_hex"] = int(vp_hex)
    mat.diffuse_color = rgba(vp_hex)
    mat.metallic = min(1.0, max(0.0, metallic))
    mat.roughness = 0.5 * (rough_min + rough_max)
    return mat


def _build_painted_metal_material(
    mat, id_hex: int, vp_hex: int, ng_global, ng_masks, ng_wrap,
    *,
    paint_col_a: int, paint_col_b: int,
    primer_col: int = 0x8A8E82,
    bare_metal_col: int = 0x9CA2A8,
    crevice_col: int = 0x161412,
    paint_metallic: float = 0.0,
    rough_min: float = 0.28,
    rough_max: float = 0.44,
    crevice_rough: float = 0.62,
    bare_rough: float = 0.24,
    coat_weight: float = 0.30,
    coat_rough: float = 0.20,
    peel_scale: float = 280.0,
    peel_bump: float = 0.045,
    chip_depth_bump: float = 0.14,
    ao_dist: float = 0.065,
    fine_bevel: float = 0.00038,
    edge_bevel: float = 0.0055,
    edge_gain: float = 2.4,
    chip_amount: float = 0.72,
    crevice_mix: float = 0.58,
    grime_scale: float = 10.0,
    macro_scale: float = 2.6,
    low_z_dust_amp: float = 0.0,
    low_z_min: float = -1.60,
    low_z_max: float = -1.00,
    dust_col: int = 0x4E4A44,
    scuff_blotch_amp: float = 0.0,
    scuff_scale: float = 12.0,
    rust_amount: float = 0.0,
    rust_col_a: int = 0x5C260E,
    rust_col_b: int = 0x2F1207,
    scratch_amp: float = 0.50,
    oil_leak_amp: float = 0.22,
    top_dust_fade_amp: float = 0.0,
    grease_smudge_amp: float = 0.0,
    blister_amp: float = 0.0,
    spatter_amp: float = 0.0,
    touchup_amp: float = 0.0,
    clearcoat_swirl_amp: float = 0.0,
    airflow_streaks_amp: float = 0.0,
    chip_thresh: float = 0.45,
    steel_warp_bump: float = 0.0,
    smudge_coat_scale: float = 1.0,
):

    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1780, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1560, -440))
    ctrl.node_tree = ng_global

    masks = _new_node(nt, "ShaderNodeGroup", (-1440, 120))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = ao_dist
    masks.inputs["Fine_Bevel_Rad"].default_value = fine_bevel
    masks.inputs["Edge_Bevel_Rad"].default_value = edge_bevel
    masks.inputs["Edge_Gain"].default_value = edge_gain
    masks.inputs["Grime_Scale"].default_value = grime_scale
    masks.inputs["Macro_Scale"].default_value = macro_scale
    masks.inputs["Micro_Scale"].default_value = peel_scale


    mix_paint = _mix_rgba(nt, 'MIX', (-1060, 420), col_a=rgba(paint_col_a), col_b=rgba(paint_col_b))
    nt.links.new(masks.outputs["Macro_Mottle"], mix_paint.inputs[0])
    cur_col = mix_paint.outputs[2]

    r_paint = _map_range(nt, 0.16, 0.85, rough_min, rough_max, (-1060, 140), clamp=True)
    nt.links.new(masks.outputs["Macro_Mottle"], r_paint.inputs[0])
    cur_rough = r_paint.outputs[0]


    up_coat_drop_out = None
    if top_dust_fade_amp > 1e-4:
        geo_up = _new_node(nt, "ShaderNodeNewGeometry", (-1440, 620))
        sep_up = _new_node(nt, "ShaderNodeSeparateXYZ", (-1220, 620))
        nt.links.new(geo_up.outputs["Normal"], sep_up.inputs["Vector"])
        up_gate = _map_range(nt, 0.40, 0.88, 0.0, top_dust_fade_amp, (-1000, 620), clamp=True)
        nt.links.new(sep_up.outputs["Z"], up_gate.inputs[0])
        up_m0 = _math(nt, 'MULTIPLY', (-800, 620), clamp=True)
        nt.links.new(up_gate.outputs[0], up_m0.inputs[0])
        nt.links.new(masks.outputs["Macro_Mottle"], up_m0.inputs[1])
        up_eff = _math(nt, 'MULTIPLY', (-620, 620), clamp=True)
        nt.links.new(up_m0.outputs[0], up_eff.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], up_eff.inputs[1])
        up_coat_drop_out = up_eff.outputs[0]

        mix_up_c = _mix_rgba(nt, 'MIX', (-760, 420), col_b=rgba(0x8A857C))
        nt.links.new(up_eff.outputs[0], mix_up_c.inputs[0])
        nt.links.new(cur_col, mix_up_c.inputs[6])
        cur_col = mix_up_c.outputs[2]

        mix_up_r = _mix_float(nt, (-760, 160), b=0.72)
        nt.links.new(up_eff.outputs[0], mix_up_r.inputs[0])
        nt.links.new(cur_rough, mix_up_r.inputs[2])
        cur_rough = mix_up_r.outputs[0]


        dn_gate = _map_range(nt, -0.25, -0.85, 0.0, 0.14, (-1000, 740), clamp=True)
        nt.links.new(sep_up.outputs["Z"], dn_gate.inputs[0])
        dn_hsv = _new_node(nt, "ShaderNodeHueSaturation", (-680, 520))
        dn_hsv.inputs["Saturation"].default_value = 1.08
        dn_hsv.inputs["Value"].default_value = 0.88
        nt.links.new(cur_col, dn_hsv.inputs["Color"])
        mix_dn_c = _mix_rgba(nt, 'MIX', (-640, 420))
        nt.links.new(dn_gate.outputs[0], mix_dn_c.inputs[0])
        nt.links.new(cur_col, mix_dn_c.inputs[6])
        nt.links.new(dn_hsv.outputs["Color"], mix_dn_c.inputs[7])
        cur_col = mix_dn_c.outputs[2]


    if airflow_streaks_amp > 1e-4:
        mp_af = _new_node(nt, "ShaderNodeMapping", (-1440, 1340))
        mp_af.inputs["Scale"].default_value = (48.0, 1.9, 48.0)
        nt.links.new(tc.outputs["Object"], mp_af.inputs["Vector"])
        n_af = _noise(nt, scale=1.0, detail=6.0, roughness=0.64, distortion=0.22, loc=(-1200, 1340))
        nt.links.new(mp_af.outputs["Vector"], n_af.inputs["Vector"])
        af_mix = _mix_float(nt, (-1080, 1340), fac=0.52)
        nt.links.new(n_af.outputs["Factor"], af_mix.inputs[2])
        nt.links.new(masks.outputs["Photo_Airflow"], af_mix.inputs[3])
        af_str = _map_range(nt, 0.42, 0.78, 0.0, airflow_streaks_amp, (-940, 1340), clamp=True)
        nt.links.new(af_mix.outputs[0], af_str.inputs[0])
        af_w = _math(nt, 'MULTIPLY', (-780, 1340), clamp=True)
        nt.links.new(af_str.outputs[0], af_w.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], af_w.inputs[1])

        mix_af_c = _mix_rgba(nt, 'MIX', (-540, 440), col_b=rgba(0x1B1D21))
        nt.links.new(af_w.outputs[0], mix_af_c.inputs[0])
        nt.links.new(cur_col, mix_af_c.inputs[6])
        cur_col = mix_af_c.outputs[2]

        mix_af_r = _mix_float(nt, (-540, 180), b=0.34)
        nt.links.new(af_w.outputs[0], mix_af_r.inputs[0])
        nt.links.new(cur_rough, mix_af_r.inputs[2])
        cur_rough = mix_af_r.outputs[0]


    if grease_smudge_amp > 1e-4:
        n_gsm = _noise(nt, scale=9.5, detail=6.0, roughness=0.66, distortion=0.95, loc=(-1220, 820))
        nt.links.new(tc.outputs["Object"], n_gsm.inputs["Vector"])
        gsm_thr = _map_range(nt, 0.56, 0.78, 0.0, grease_smudge_amp, (-980, 820), clamp=True)
        nt.links.new(n_gsm.outputs["Factor"], gsm_thr.inputs[0])
        gsm_exp = _math(nt, 'MULTIPLY', (-780, 820), clamp=True)
        nt.links.new(gsm_thr.outputs[0], gsm_exp.inputs[0])
        nt.links.new(masks.outputs["Exposure_Mask"], gsm_exp.inputs[1])
        gsm_eff = _math(nt, 'MULTIPLY', (-600, 820), clamp=True)
        nt.links.new(gsm_exp.outputs[0], gsm_eff.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], gsm_eff.inputs[1])

        mix_gs_c = _mix_rgba(nt, 'MIX', (-600, 420), col_b=rgba(0x1D1B18))
        nt.links.new(gsm_eff.outputs[0], mix_gs_c.inputs[0])
        nt.links.new(cur_col, mix_gs_c.inputs[6])
        cur_col = mix_gs_c.outputs[2]

        mix_gs_r = _mix_float(nt, (-600, 160), b=0.22)
        nt.links.new(gsm_eff.outputs[0], mix_gs_r.inputs[0])
        nt.links.new(cur_rough, mix_gs_r.inputs[2])
        cur_rough = mix_gs_r.outputs[0]


    touchup_bump_out = None
    if touchup_amp > 1e-4:
        n_tu = _noise(nt, scale=4.2, detail=5.0, roughness=0.60, distortion=0.85, loc=(-1220, 1000))
        nt.links.new(tc.outputs["Object"], n_tu.inputs["Vector"])
        tu_mask = _map_range(nt, 0.68, 0.73, 0.0, touchup_amp, (-980, 1000), clamp=True)
        nt.links.new(n_tu.outputs["Factor"], tu_mask.inputs[0])
        touchup_bump_out = tu_mask.outputs[0]

        tu_hsv = _new_node(nt, "ShaderNodeHueSaturation", (-520, 560))
        tu_hsv.inputs["Hue"].default_value = 0.506
        tu_hsv.inputs["Saturation"].default_value = 1.07
        tu_hsv.inputs["Value"].default_value = 1.08
        nt.links.new(cur_col, tu_hsv.inputs["Color"])
        mix_tu_c = _mix_rgba(nt, 'MIX', (-420, 460))
        nt.links.new(tu_mask.outputs[0], mix_tu_c.inputs[0])
        nt.links.new(cur_col, mix_tu_c.inputs[6])
        nt.links.new(tu_hsv.outputs["Color"], mix_tu_c.inputs[7])
        cur_col = mix_tu_c.outputs[2]

        mix_tu_r = _mix_float(nt, (-420, 220), b=max(0.14, rough_min * 0.72))
        nt.links.new(tu_mask.outputs[0], mix_tu_r.inputs[0])
        nt.links.new(cur_rough, mix_tu_r.inputs[2])
        cur_rough = mix_tu_r.outputs[0]


    spatter_bump_out = None
    if spatter_amp > 1e-4:
        vor_sp = _voronoi(nt, scale=185.0, feature='F1', randomness=0.98, loc=(-1220, 1180))
        nt.links.new(tc.outputs["Object"], vor_sp.inputs["Vector"])
        sp_dot = _map_range(nt, 0.0, 0.14, 1.0, 0.0, (-980, 1180), clamp=True)
        nt.links.new(vor_sp.outputs["Distance"], sp_dot.inputs[0])
        n_spg = _noise(nt, scale=5.5, detail=4.0, roughness=0.60, loc=(-980, 1300))
        nt.links.new(tc.outputs["Object"], n_spg.inputs["Vector"])
        sp_cl = _map_range(nt, 0.56, 0.74, 0.0, spatter_amp, (-760, 1300), clamp=True)
        nt.links.new(n_spg.outputs["Factor"], sp_cl.inputs[0])
        sp_m0 = _math(nt, 'MULTIPLY', (-560, 1220), clamp=True)
        nt.links.new(sp_dot.outputs[0], sp_m0.inputs[0])
        nt.links.new(sp_cl.outputs[0], sp_m0.inputs[1])
        sp_eff = _math(nt, 'MULTIPLY', (-380, 1220), clamp=True)
        nt.links.new(sp_m0.outputs[0], sp_eff.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], sp_eff.inputs[1])
        spatter_bump_out = sp_eff.outputs[0]

        mix_sp_c = _mix_rgba(nt, 'MIX', (-300, 440), col_b=rgba(0x141210))
        nt.links.new(sp_eff.outputs[0], mix_sp_c.inputs[0])
        nt.links.new(cur_col, mix_sp_c.inputs[6])
        cur_col = mix_sp_c.outputs[2]

        mix_sp_r = _mix_float(nt, (-300, 180), b=0.14)
        nt.links.new(sp_eff.outputs[0], mix_sp_r.inputs[0])
        nt.links.new(cur_rough, mix_sp_r.inputs[2])
        cur_rough = mix_sp_r.outputs[0]


    crev_w = _math(nt, 'MULTIPLY', (-920, -140), v0=crevice_mix)
    nt.links.new(ctrl.outputs["Weathering"], crev_w.inputs[1])
    crev_eff = _math(nt, 'MULTIPLY', (-720, -140), clamp=True)
    nt.links.new(masks.outputs["Crevice_Mask"], crev_eff.inputs[0])
    nt.links.new(crev_w.outputs[0], crev_eff.inputs[1])

    mix_cr_c = _mix_rgba(nt, 'MIX', (-460, 380), col_b=rgba(crevice_col))
    nt.links.new(crev_eff.outputs[0], mix_cr_c.inputs[0])
    nt.links.new(cur_col, mix_cr_c.inputs[6])
    cur_col = mix_cr_c.outputs[2]

    mix_cr_r = _mix_float(nt, (-460, 120), b=crevice_rough)
    nt.links.new(crev_eff.outputs[0], mix_cr_r.inputs[0])
    nt.links.new(cur_rough, mix_cr_r.inputs[2])
    cur_rough = mix_cr_r.outputs[0]


    if low_z_dust_amp > 1e-4:
        geo_z = _new_node(nt, "ShaderNodeNewGeometry", (-1200, -640))
        sep_z = _new_node(nt, "ShaderNodeSeparateXYZ", (-980, -640))
        nt.links.new(geo_z.outputs["Position"], sep_z.inputs["Vector"])
        z_grad = _map_range(nt, low_z_min, low_z_max, low_z_dust_amp, 0.0, (-760, -640), clamp=True)
        nt.links.new(sep_z.outputs["Z"], z_grad.inputs[0])
        z_noise = _math(nt, 'MULTIPLY', (-540, -640), clamp=True)
        nt.links.new(z_grad.outputs[0], z_noise.inputs[0])
        nt.links.new(masks.outputs["Macro_Mottle"], z_noise.inputs[1])
        z_weath = _math(nt, 'MULTIPLY', (-340, -640), clamp=True)
        nt.links.new(z_noise.outputs[0], z_weath.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], z_weath.inputs[1])

        mix_dz_c = _mix_rgba(nt, 'MIX', (-200, 380), col_b=rgba(dust_col))
        nt.links.new(z_weath.outputs[0], mix_dz_c.inputs[0])
        nt.links.new(cur_col, mix_dz_c.inputs[6])
        cur_col = mix_dz_c.outputs[2]

        mix_dz_r = _mix_float(nt, (-200, 120), b=0.82)
        nt.links.new(z_weath.outputs[0], mix_dz_r.inputs[0])
        nt.links.new(cur_rough, mix_dz_r.inputs[2])
        cur_rough = mix_dz_r.outputs[0]


    ew_mix_bd = _math(nt, 'MULTIPLY_ADD', (-960, -380), v1=0.82)
    nt.links.new(ctrl.outputs["SurfaceErosion"], ew_mix_bd.inputs[0])
    ew_p = _math(nt, 'MULTIPLY', (-1160, -380), v1=0.60)
    nt.links.new(ctrl.outputs["EdgeWear"], ew_p.inputs[0])
    nt.links.new(ew_p.outputs[0], ew_mix_bd.inputs[2])

    ew_w = _math(nt, 'MULTIPLY', (-720, -360), v0=chip_amount)
    nt.links.new(ew_mix_bd.outputs[0], ew_w.inputs[1])

    n_patch = _noise(nt, scale=3.6, detail=6.0, roughness=0.66, distortion=0.42, loc=(-960, -240))
    nt.links.new(tc.outputs["Object"], n_patch.inputs["Vector"])
    patch_gate = _map_range(nt, 0.34, 0.72, 0.06, 1.38, (-720, -240), clamp=True)
    nt.links.new(n_patch.outputs["Factor"], patch_gate.inputs[0])


    exp_chip = _math(nt, 'MULTIPLY_ADD', (-720, -500), v1=0.82, v2=0.45)
    nt.links.new(masks.outputs["Exposure_Mask"], exp_chip.inputs[0])

    chip_raw0 = _math(nt, 'MULTIPLY', (-540, -340), clamp=True)
    nt.links.new(masks.outputs["EdgeWear_Mask"], chip_raw0.inputs[0])
    nt.links.new(ew_w.outputs[0], chip_raw0.inputs[1])

    chip_raw1 = _math(nt, 'MULTIPLY', (-400, -340), clamp=True)
    nt.links.new(chip_raw0.outputs[0], chip_raw1.inputs[0])
    nt.links.new(patch_gate.outputs[0], chip_raw1.inputs[1])

    chip_sig = _math(nt, 'MULTIPLY', (-260, -340), clamp=True)
    nt.links.new(chip_raw1.outputs[0], chip_sig.inputs[0])
    nt.links.new(exp_chip.outputs[0], chip_sig.inputs[1])

    if scuff_blotch_amp > 1e-4:
        n_scuff = _noise(nt, scale=scuff_scale, detail=9.0, roughness=0.72, distortion=0.65, loc=(-960, -560))
        nt.links.new(tc.outputs["Object"], n_scuff.inputs["Vector"])

        sc_f0 = _map_range(nt, 0.76, 0.87, 0.0, 1.0, (-800, -580), clamp=True)
        nt.links.new(n_scuff.outputs["Factor"], sc_f0.inputs[0])
        sc_pc = _map_range(nt, 0.42, 0.88, 0.0, 0.48, (-800, -700), clamp=True)
        nt.links.new(masks.outputs["Photo_Chip"], sc_pc.inputs[0])
        sc_u = _math(nt, 'MAXIMUM', (-660, -600), clamp=True)
        nt.links.new(sc_f0.outputs[0], sc_u.inputs[0])
        nt.links.new(sc_pc.outputs[0], sc_u.inputs[1])
        sc_thr = _math(nt, 'MULTIPLY', (-540, -600), clamp=True, v1=scuff_blotch_amp)
        nt.links.new(sc_u.outputs[0], sc_thr.inputs[0])
        sc_w = _math(nt, 'MULTIPLY', (-420, -580), clamp=True)
        nt.links.new(sc_thr.outputs[0], sc_w.inputs[0])
        nt.links.new(ew_mix_bd.outputs[0], sc_w.inputs[1])
        chip_comb = _math(nt, 'MAXIMUM', (-120, -380), clamp=True)
        nt.links.new(chip_sig.outputs[0], chip_comb.inputs[0])
        nt.links.new(sc_w.outputs[0], chip_comb.inputs[1])
        chip_raw = chip_comb.outputs[0]
    else:
        chip_raw = chip_sig.outputs[0]


    pr_lo = max(0.08, chip_thresh - 0.045)
    bm_hi = min(0.95, chip_thresh + 0.065)
    primer_mask = _map_range(nt, pr_lo, chip_thresh, 0.0, 1.0, (40, -280), clamp=True)
    nt.links.new(chip_raw, primer_mask.inputs[0])
    bare_mask = _map_range(nt, chip_thresh, bm_hi, 0.0, 1.0, (40, -440), clamp=True)
    nt.links.new(chip_raw, bare_mask.inputs[0])

    mix_pr_c = _mix_rgba(nt, 'MIX', (160, 380), col_b=rgba(primer_col))
    nt.links.new(primer_mask.outputs[0], mix_pr_c.inputs[0])
    nt.links.new(cur_col, mix_pr_c.inputs[6])

    mix_bm_c = _mix_rgba(nt, 'MIX', (360, 380), col_b=rgba(bare_metal_col))
    nt.links.new(bare_mask.outputs[0], mix_bm_c.inputs[0])
    nt.links.new(mix_pr_c.outputs[2], mix_bm_c.inputs[6])
    cur_col = mix_bm_c.outputs[2]

    mix_bm_m = _mix_float(nt, (360, 140), a=paint_metallic, b=0.96)
    nt.links.new(bare_mask.outputs[0], mix_bm_m.inputs[0])
    cur_met = mix_bm_m.outputs[0]

    mix_bm_r = _mix_float(nt, (360, -40), b=bare_rough)
    nt.links.new(bare_mask.outputs[0], mix_bm_r.inputs[0])
    nt.links.new(cur_rough, mix_bm_r.inputs[2])
    cur_rough = mix_bm_r.outputs[0]


    rust_eff_out = None
    if rust_amount > 1e-4:
        n_rust = _noise(nt, scale=14.0, detail=9.0, roughness=0.76, distortion=0.58, loc=(-240, -820))
        nt.links.new(tc.outputs["Object"], n_rust.inputs["Vector"])
        r_pit = _map_range(nt, 0.32, 0.68, 0.0, 1.0, (0, -820), clamp=True)
        nt.links.new(n_rust.outputs["Factor"], r_pit.inputs[0])


        r_halo = _map_range(nt, 0.14, 0.36, 0.0, 1.0, (-40, -880), clamp=True)
        nt.links.new(chip_raw, r_halo.inputs[0])

        r_seam = _math(nt, 'MULTIPLY', (-40, -1020), clamp=True)
        nt.links.new(masks.outputs["Crevice_Mask"], r_seam.inputs[0])
        nt.links.new(masks.outputs["Exposure_Mask"], r_seam.inputs[1])
        r_seam_s = _math(nt, 'MULTIPLY', (140, -1020), clamp=True, v1=0.95)
        nt.links.new(r_seam.outputs[0], r_seam_s.inputs[0])

        r_zone = _math(nt, 'MAXIMUM', (300, -900), clamp=True)
        nt.links.new(r_halo.outputs[0], r_zone.inputs[0])
        nt.links.new(r_seam_s.outputs[0], r_zone.inputs[1])


        r_bd = _map_range(nt, 0.25, 1.45, 0.06, 1.15, (140, -1160), clamp=True)
        nt.links.new(ctrl.outputs["SurfaceErosion"], r_bd.inputs[0])

        r_m0 = _math(nt, 'MULTIPLY', (460, -840), clamp=True)
        nt.links.new(r_zone.outputs[0], r_m0.inputs[0])
        nt.links.new(r_pit.outputs[0], r_m0.inputs[1])

        r_m1 = _math(nt, 'MULTIPLY', (620, -840), clamp=True)
        nt.links.new(r_m0.outputs[0], r_m1.inputs[0])
        nt.links.new(r_bd.outputs[0], r_m1.inputs[1])

        r_mask = _math(nt, 'MULTIPLY', (780, -840), clamp=True, v1=rust_amount)
        nt.links.new(r_m1.outputs[0], r_mask.inputs[0])
        rust_eff_out = r_mask.outputs[0]

        rust_col = _mix_rgba(nt, 'MIX', (420, 560), col_a=rgba(rust_col_a), col_b=rgba(rust_col_b))
        nt.links.new(n_rust.outputs["Factor"], rust_col.inputs[0])

        mix_rs_c = _mix_rgba(nt, 'MIX', (560, 380))
        nt.links.new(r_mask.outputs[0], mix_rs_c.inputs[0])
        nt.links.new(cur_col, mix_rs_c.inputs[6])
        nt.links.new(rust_col.outputs[2], mix_rs_c.inputs[7])
        cur_col = mix_rs_c.outputs[2]

        mix_rs_m = _mix_float(nt, (560, 140), b=0.0)
        nt.links.new(r_mask.outputs[0], mix_rs_m.inputs[0])
        nt.links.new(cur_met, mix_rs_m.inputs[2])
        cur_met = mix_rs_m.outputs[0]

        mix_rs_r = _mix_float(nt, (560, -40), b=0.90)
        nt.links.new(r_mask.outputs[0], mix_rs_r.inputs[0])
        nt.links.new(cur_rough, mix_rs_r.inputs[2])
        cur_rough = mix_rs_r.outputs[0]


    scratch_eff_out = None
    if scratch_amp > 1e-4:
        sc_bd = _math(nt, 'MULTIPLY_ADD', (320, -680), clamp=True, v1=0.95, v2=0.14)
        nt.links.new(ctrl.outputs["SurfaceErosion"], sc_bd.inputs[0])
        sc_g = _math(nt, 'MULTIPLY', (500, -680), clamp=True, v1=scratch_amp)
        nt.links.new(sc_bd.outputs[0], sc_g.inputs[0])
        sc_eff = _math(nt, 'MULTIPLY', (680, -680), clamp=True)
        nt.links.new(masks.outputs["Scratch_Mask"], sc_eff.inputs[0])
        nt.links.new(sc_g.outputs[0], sc_eff.inputs[1])
        scratch_eff_out = sc_eff.outputs[0]

        mix_sc_c = _mix_rgba(nt, 'MIX', (740, 380), col_b=rgba(bare_metal_col))
        nt.links.new(sc_eff.outputs[0], mix_sc_c.inputs[0])
        nt.links.new(cur_col, mix_sc_c.inputs[6])
        cur_col = mix_sc_c.outputs[2]

        mix_sc_m = _mix_float(nt, (740, 140), b=0.96)
        nt.links.new(sc_eff.outputs[0], mix_sc_m.inputs[0])
        nt.links.new(cur_met, mix_sc_m.inputs[2])
        cur_met = mix_sc_m.outputs[0]

        mix_sc_r = _mix_float(nt, (740, -40), b=0.16)
        nt.links.new(sc_eff.outputs[0], mix_sc_r.inputs[0])
        nt.links.new(cur_rough, mix_sc_r.inputs[2])
        cur_rough = mix_sc_r.outputs[0]


    coat_drop0 = _math(nt, 'MAXIMUM', (200, -240), clamp=True)
    nt.links.new(primer_mask.outputs[0], coat_drop0.inputs[0])
    nt.links.new(crev_eff.outputs[0], coat_drop0.inputs[1])
    cd_sig = coat_drop0.outputs[0]
    if scratch_eff_out is not None:
        coat_drop_sc = _math(nt, 'MAXIMUM', (280, -240), clamp=True)
        nt.links.new(cd_sig, coat_drop_sc.inputs[0])
        nt.links.new(scratch_eff_out, coat_drop_sc.inputs[1])
        cd_sig = coat_drop_sc.outputs[0]
    if rust_eff_out is not None:
        coat_drop = _math(nt, 'MAXIMUM', (360, -240), clamp=True)
        nt.links.new(cd_sig, coat_drop.inputs[0])
        nt.links.new(rust_eff_out, coat_drop.inputs[1])
        cd_sig = coat_drop.outputs[0]

    coat_inv = _math(nt, 'SUBTRACT', (520, -240), clamp=True, v0=1.0)
    nt.links.new(cd_sig, coat_inv.inputs[1])
    if up_coat_drop_out is not None:
        up_cd = _math(nt, 'MULTIPLY', (520, -340), clamp=True, v1=0.52)
        nt.links.new(up_coat_drop_out, up_cd.inputs[0])
        coat_inv2 = _math(nt, 'SUBTRACT', (600, -240), clamp=True)
        nt.links.new(coat_inv.outputs[0], coat_inv2.inputs[0])
        nt.links.new(up_cd.outputs[0], coat_inv2.inputs[1])
        coat_base_sig = coat_inv2.outputs[0]
    else:
        coat_base_sig = coat_inv.outputs[0]
    coat_eff_node = _math(nt, 'MULTIPLY', (680, -240), v1=coat_weight)
    nt.links.new(coat_base_sig, coat_eff_node.inputs[0])
    cur_coat_w = coat_eff_node.outputs[0]
    cur_coat_r_sig = None


    if oil_leak_amp > 1e-4:
        ol_drv = _math(nt, 'MULTIPLY_ADD', (520, -1040), v1=0.68)
        nt.links.new(ctrl.outputs["SurfaceErosion"], ol_drv.inputs[0])
        ol_w0 = _math(nt, 'MULTIPLY', (340, -1040), v1=0.40)
        nt.links.new(ctrl.outputs["Weathering"], ol_w0.inputs[0])
        nt.links.new(ol_w0.outputs[0], ol_drv.inputs[2])

        ol_g = _math(nt, 'MULTIPLY', (680, -1040), clamp=True, v1=oil_leak_amp)
        nt.links.new(ol_drv.outputs[0], ol_g.inputs[0])
        ol_eff = _math(nt, 'MULTIPLY', (840, -1040), clamp=True)
        nt.links.new(masks.outputs["OilLeak_Mask"], ol_eff.inputs[0])
        nt.links.new(ol_g.outputs[0], ol_eff.inputs[1])

        ol_wet = _map_range(nt, 0.36, 0.80, 0.0, 1.0, (880, -960), clamp=True)
        nt.links.new(ol_eff.outputs[0], ol_wet.inputs[0])
        ol_ring0 = _map_range(nt, 0.08, 0.46, 0.0, 0.60, (880, -1120), clamp=True)
        nt.links.new(ol_eff.outputs[0], ol_ring0.inputs[0])
        ol_not_w = _math(nt, 'SUBTRACT', (980, -1040), clamp=True, v0=1.0)
        nt.links.new(ol_wet.outputs[0], ol_not_w.inputs[1])
        ol_halo = _math(nt, 'MULTIPLY', (1080, -1080), clamp=True)
        nt.links.new(ol_ring0.outputs[0], ol_halo.inputs[0])
        nt.links.new(ol_not_w.outputs[0], ol_halo.inputs[1])


        mix_oh_c = _mix_rgba(nt, 'MIX', (860, 400), col_b=rgba(0x3B342B))
        nt.links.new(ol_halo.outputs[0], mix_oh_c.inputs[0])
        nt.links.new(cur_col, mix_oh_c.inputs[6])
        cur_col = mix_oh_c.outputs[2]
        mix_oh_r = _mix_float(nt, (860, -20), b=0.84)
        nt.links.new(ol_halo.outputs[0], mix_oh_r.inputs[0])
        nt.links.new(cur_rough, mix_oh_r.inputs[2])
        cur_rough = mix_oh_r.outputs[0]


        mix_ol_c = _mix_rgba(nt, 'MIX', (960, 380), col_b=rgba(0x130E08))
        nt.links.new(ol_wet.outputs[0], mix_ol_c.inputs[0])
        nt.links.new(cur_col, mix_ol_c.inputs[6])
        cur_col = mix_ol_c.outputs[2]

        mix_ol_r = _mix_float(nt, (960, -40), b=0.065)
        nt.links.new(ol_wet.outputs[0], mix_ol_r.inputs[0])
        nt.links.new(cur_rough, mix_ol_r.inputs[2])
        cur_rough = mix_ol_r.outputs[0]

        ol_coat_w = _mix_float(nt, (960, -240), b=0.86)
        nt.links.new(ol_wet.outputs[0], ol_coat_w.inputs[0])
        nt.links.new(cur_coat_w, ol_coat_w.inputs[2])
        cur_coat_w = ol_coat_w.outputs[0]

        ol_coat_r = _mix_float(nt, (960, -380), a=coat_rough, b=0.055)
        nt.links.new(ol_wet.outputs[0], ol_coat_r.inputs[0])
        cur_coat_r_sig = ol_coat_r.outputs[0]


    cur_norm = masks.outputs["Bevel_Normal"]
    p_macro_terms = []
    p_micro_terms = []

    if peel_bump > 1e-4:
        p_micro_terms.append((masks.outputs["Micro_Grain"], peel_bump, 0.00050))
    if steel_warp_bump > 1e-4:

        n_sw1 = _noise(nt, scale=11.0, detail=4.0, roughness=0.52, distortion=0.25, loc=(40, -1060))
        nt.links.new(tc.outputs["Object"], n_sw1.inputs["Vector"])
        n_sw2 = _noise(nt, scale=54.0, detail=6.0, roughness=0.62, distortion=0.18, loc=(240, -1060))
        nt.links.new(tc.outputs["Object"], n_sw2.inputs["Vector"])
        p_macro_terms.append((n_sw1.outputs["Factor"], steel_warp_bump, 0.00180))
        p_micro_terms.append((n_sw2.outputs["Factor"], steel_warp_bump * 0.90, 0.00065))
    if touchup_bump_out is not None:
        p_micro_terms.append((touchup_bump_out, 0.045, 0.00025))
    if spatter_bump_out is not None:
        p_micro_terms.append((spatter_bump_out, 0.055, 0.00022))


    if blister_amp > 1e-4 and rust_eff_out is not None:
        vor_bl = _voronoi(nt, scale=135.0, feature='F1', randomness=0.95, loc=(40, -1240))
        nt.links.new(tc.outputs["Object"], vor_bl.inputs["Vector"])
        bl_dome = _map_range(nt, 0.0, 0.42, 1.0, 0.0, (240, -1240), clamp=True)
        nt.links.new(vor_bl.outputs["Distance"], bl_dome.inputs[0])
        bl_not_bare = _math(nt, 'SUBTRACT', (240, -1380), clamp=True, v0=1.0)
        nt.links.new(bare_mask.outputs[0], bl_not_bare.inputs[1])
        bl_zone = _math(nt, 'MULTIPLY', (420, -1300), clamp=True)
        nt.links.new(rust_eff_out, bl_zone.inputs[0])
        nt.links.new(bl_not_bare.outputs[0], bl_zone.inputs[1])
        bl_bd_gate = _map_range(nt, 0.85, 1.40, 0.0, 1.0, (420, -1440), clamp=True)
        nt.links.new(ctrl.outputs["SurfaceErosion"], bl_bd_gate.inputs[0])
        bl_zone_g = _math(nt, 'MULTIPLY', (520, -1360), clamp=True)
        nt.links.new(bl_zone.outputs[0], bl_zone_g.inputs[0])
        nt.links.new(bl_bd_gate.outputs[0], bl_zone_g.inputs[1])
        bl_h = _math(nt, 'MULTIPLY', (640, -1260), clamp=True)
        nt.links.new(bl_dome.outputs[0], bl_h.inputs[0])
        nt.links.new(bl_zone_g.outputs[0], bl_h.inputs[1])
        p_macro_terms.append((bl_h.outputs[0], 0.18 * blister_amp, 0.00090))

    chip_h = _math(nt, 'SUBTRACT', (260, -640), clamp=True, v0=1.0)
    nt.links.new(bare_mask.outputs[0], chip_h.inputs[1])
    p_macro_terms.append((chip_h.outputs[0], chip_depth_bump, 0.00055))

    if scratch_eff_out is not None:
        sc_inv = _math(nt, 'SUBTRACT', (760, -540), clamp=True, v0=1.0)
        nt.links.new(scratch_eff_out, sc_inv.inputs[1])
        p_micro_terms.append((sc_inv.outputs[0], 0.16, 0.00045))

    if rust_eff_out is not None:
        p_macro_terms.append((rust_eff_out, 0.22, 0.00080))

    cur_norm = _combine_scalar_heights(nt, p_macro_terms, cur_norm, ref_distance=0.0008, loc=(660, -540))
    cur_norm = _combine_scalar_heights(nt, p_micro_terms, cur_norm, ref_distance=0.0004, loc=(1020, -540))


    if clearcoat_swirl_amp > 1e-4 and coat_weight > 1e-4:
        w_csw = _new_node(nt, "ShaderNodeTexWave", (780, -340))
        w_csw.wave_type = 'RINGS'
        w_csw.rings_direction = 'SPHERICAL'
        w_csw.inputs["Scale"].default_value = 28.0
        w_csw.inputs["Distortion"].default_value = 5.5
        w_csw.inputs["Detail"].default_value = 3.0
        nt.links.new(tc.outputs["Object"], w_csw.inputs["Vector"])
        csw_off = _map_range(nt, 0.0, 1.0, -0.045 * clearcoat_swirl_amp, +0.065 * clearcoat_swirl_amp, (980, -340), clamp=True)
        nt.links.new(w_csw.outputs["Fac"], csw_off.inputs[0])
        csw_add = _math(nt, 'ADD', (1120, -340), clamp=True)
        if cur_coat_r_sig is not None:
            nt.links.new(cur_coat_r_sig, csw_add.inputs[0])
        else:
            csw_add.inputs[0].default_value = coat_rough
        nt.links.new(csw_off.outputs[0], csw_add.inputs[1])
        cur_coat_r_sig = csw_add.outputs[0]


    ps_pw = _math(nt, 'MULTIPLY', (1040, -160), clamp=True, v1=0.85)
    nt.links.new(masks.outputs["Photo_Smudge"], ps_pw.inputs[0])
    ps_peff = _math(nt, 'MULTIPLY', (1120, -160), clamp=True)
    nt.links.new(ps_pw.outputs[0], ps_peff.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], ps_peff.inputs[1])

    pf_m0 = _math(nt, 'MULTIPLY', (1040, -400), clamp=True)
    nt.links.new(masks.outputs["Photo_Fingerprint"], pf_m0.inputs[0])
    nt.links.new(masks.outputs["Exposure_Mask"], pf_m0.inputs[1])
    pf_eff = _math(nt, 'MULTIPLY', (1120, -400), clamp=True)
    nt.links.new(pf_m0.outputs[0], pf_eff.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], pf_eff.inputs[1])

    ps_pr = _map_range(nt, 0.0, 1.0, -0.02, +0.12 * smudge_coat_scale, (1180, -40), clamp=True)
    nt.links.new(ps_peff.outputs[0], ps_pr.inputs[0])

    ps_not_bm = _math(nt, 'MULTIPLY_ADD', (1180, -110), clamp=True, v1=-0.80, v2=1.0)
    nt.links.new(bare_mask.outputs[0], ps_not_bm.inputs[0])
    ps_pr_g = _math(nt, 'MULTIPLY', (1200, -40), clamp=False)
    nt.links.new(ps_pr.outputs[0], ps_pr_g.inputs[0])
    nt.links.new(ps_not_bm.outputs[0], ps_pr_g.inputs[1])
    r_p_add = _math(nt, 'ADD', (1220, 40), clamp=True)
    nt.links.new(cur_rough, r_p_add.inputs[0])
    nt.links.new(ps_pr_g.outputs[0], r_p_add.inputs[1])
    cur_rough = r_p_add.outputs[0]

    ps_cr0 = _math(nt, 'MULTIPLY', (1140, -280), clamp=True, v1=0.18 * smudge_coat_scale)
    nt.links.new(ps_peff.outputs[0], ps_cr0.inputs[0])
    pf_cr = _math(nt, 'MULTIPLY_ADD', (1180, -340), clamp=True, v1=0.16 * smudge_coat_scale)
    nt.links.new(pf_eff.outputs[0], pf_cr.inputs[0])
    nt.links.new(ps_cr0.outputs[0], pf_cr.inputs[2])
    cr_p_add = _math(nt, 'ADD', (1220, -280), clamp=True)
    if cur_coat_r_sig is not None:
        nt.links.new(cur_coat_r_sig, cr_p_add.inputs[0])
    else:
        cr_p_add.inputs[0].default_value = coat_rough
    nt.links.new(pf_cr.outputs[0], cr_p_add.inputs[1])
    cur_coat_r_sig = cr_p_add.outputs[0]

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (1280, 140))
    nt.links.new(cur_col, bsdf.inputs["Base Color"])
    nt.links.new(cur_met, bsdf.inputs["Metallic"])
    nt.links.new(cur_rough, bsdf.inputs["Roughness"])
    nt.links.new(cur_norm, bsdf.inputs["Normal"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.50
    nt.links.new(cur_coat_w, bsdf.inputs["Coat Weight"])
    if cur_coat_r_sig is not None:
        nt.links.new(cur_coat_r_sig, bsdf.inputs["Coat Roughness"])
    else:
        bsdf.inputs["Coat Roughness"].default_value = coat_rough
    if "Coat Normal" in bsdf.inputs:
        nt.links.new(masks.outputs["Bevel_Normal"], bsdf.inputs["Coat Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (1580, 140))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(id_hex)
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (1820, 140))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(id_hex)
    mat["f15_pbr_hex"] = int(vp_hex)
    mat.diffuse_color = rgba(vp_hex)
    mat.metallic = paint_metallic
    mat.roughness = 0.5 * (rough_min + rough_max)
    return mat


def _build_ceramic_soot_material(
    mat, id_hex: int, vp_hex: int, ng_global, ng_masks, ng_wrap,
    *,
    ceramic_a: int = 0xC6C3BA,
    ceramic_b: int = 0x9C9990,
    soot_mid: int = 0x3A3836,
    soot_dark: int = 0x121214,
    metallic_base: float = 0.05,
    rough_ceramic: float = 0.86,
    rough_soot: float = 0.92,
    streak_scale_lo=(16.0, 1.4, 16.0),
    streak_scale_hi=(54.0, 3.8, 54.0),
    y_grad_start: float = 0.05,
    y_grad_end: float = 0.88,
    y_grad_gain: float = 0.45,
    crevice_soot_gain: float = 0.88,
    base_soot_bias: float = 0.32,
    spall_amount: float = 0.95,
):

    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1780, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1560, -480))
    ctrl.node_tree = ng_global

    masks = _new_node(nt, "ShaderNodeGroup", (-1440, 180))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = 0.026
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00036
    masks.inputs["Edge_Bevel_Rad"].default_value = 0.0030
    masks.inputs["Edge_Gain"].default_value = 1.6
    masks.inputs["Grime_Scale"].default_value = 18.0
    masks.inputs["Macro_Scale"].default_value = 4.5
    masks.inputs["Micro_Scale"].default_value = 420.0


    comb_seed = _new_node(nt, "ShaderNodeCombineXYZ", (-1180, -320))
    msx = _math(nt, 'MULTIPLY', (-1340, -280), v1=11.7)
    msy = _math(nt, 'MULTIPLY', (-1340, -360), v1=2.3)
    msz = _math(nt, 'MULTIPLY', (-1340, -440), v1=17.3)
    nt.links.new(masks.outputs["Island_Rand"], msx.inputs[0])
    nt.links.new(masks.outputs["Island_Rand"], msy.inputs[0])
    nt.links.new(masks.outputs["Island_Rand"], msz.inputs[0])
    nt.links.new(msx.outputs[0], comb_seed.inputs["X"])
    nt.links.new(msy.outputs[0], comb_seed.inputs["Y"])
    nt.links.new(msz.outputs[0], comb_seed.inputs["Z"])

    coord_p = _vmath(nt, 'ADD', (-980, -260))
    nt.links.new(tc.outputs["Object"], coord_p.inputs[0])
    nt.links.new(comb_seed.outputs["Vector"], coord_p.inputs[1])


    m_lo = _new_node(nt, "ShaderNodeMapping", (-780, -160))
    m_lo.inputs["Scale"].default_value = streak_scale_lo
    nt.links.new(coord_p.outputs["Vector"], m_lo.inputs["Vector"])
    n_lo = _noise(nt, scale=1.0, detail=7.0, roughness=0.68, distortion=0.72, loc=(-540, -160))
    nt.links.new(m_lo.outputs["Vector"], n_lo.inputs["Vector"])

    m_hi = _new_node(nt, "ShaderNodeMapping", (-780, -420))
    m_hi.inputs["Scale"].default_value = streak_scale_hi
    nt.links.new(coord_p.outputs["Vector"], m_hi.inputs["Vector"])
    n_hi = _noise(nt, scale=1.0, detail=8.0, roughness=0.70, distortion=0.58, loc=(-540, -420))
    nt.links.new(m_hi.outputs["Vector"], n_hi.inputs["Vector"])

    n_turb = _noise(nt, scale=8.5, detail=8.0, roughness=0.72, distortion=1.05, loc=(-540, -560))
    nt.links.new(coord_p.outputs["Vector"], n_turb.inputs["Vector"])

    st_mix0 = _mix_float(nt, (-340, -240), fac=0.22)
    nt.links.new(n_lo.outputs["Factor"], st_mix0.inputs[2])
    nt.links.new(n_hi.outputs["Factor"], st_mix0.inputs[3])

    st_mix1 = _mix_float(nt, (-260, -240), fac=0.58)
    nt.links.new(st_mix0.outputs[0], st_mix1.inputs[2])
    nt.links.new(n_turb.outputs["Factor"], st_mix1.inputs[3])


    st_mix2 = _mix_float(nt, (-180, -240), fac=0.46)
    nt.links.new(st_mix1.outputs[0], st_mix2.inputs[2])
    nt.links.new(masks.outputs["Photo_Airflow"], st_mix2.inputs[3])

    st_mix = _mix_float(nt, (-110, -240), fac=0.24)
    nt.links.new(st_mix2.outputs[0], st_mix.inputs[2])
    nt.links.new(masks.outputs["Photo_Smudge"], st_mix.inputs[3])

    st_contrast = _map_range(nt, 0.30, 0.78, 0.0, 1.0, (-20, -240), clamp=True)
    nt.links.new(st_mix.outputs[0], st_contrast.inputs[0])

    sep_y = _new_node(nt, "ShaderNodeSeparateXYZ", (-780, -680))
    nt.links.new(tc.outputs["Object"], sep_y.inputs["Vector"])

    y_soot_aft = _map_range(nt, y_grad_start, y_grad_end, 0.0, y_grad_gain, (-520, -680), clamp=True)
    nt.links.new(sep_y.outputs["Y"], y_soot_aft.inputs[0])
    y_soot_fwd = _map_range(nt, -0.15, 0.54, 0.42, 0.14, (-520, -780), clamp=True)
    nt.links.new(sep_y.outputs["Y"], y_soot_fwd.inputs[0])
    y_soot = _math(nt, 'MAXIMUM', (-340, -720), clamp=True)
    nt.links.new(y_soot_aft.outputs[0], y_soot.inputs[0])
    nt.links.new(y_soot_fwd.outputs[0], y_soot.inputs[1])

    crev_tight = _map_range(nt, 0.62, 0.94, 0.0, 1.0, (-460, -520), clamp=True)
    nt.links.new(masks.outputs["Crevice_Mask"], crev_tight.inputs[0])
    crev_soot = _math(nt, 'MULTIPLY', (-240, -520), v1=crevice_soot_gain)
    nt.links.new(crev_tight.outputs[0], crev_soot.inputs[0])

    st_sharp = _map_range(nt, 0.28, 0.84, 0.0, 1.0, (100, -240), clamp=True)
    nt.links.new(st_contrast.outputs[0], st_sharp.inputs[0])


    s_sum0 = _math(nt, 'ADD', (280, -240), v0=base_soot_bias)
    nt.links.new(st_sharp.outputs[0], s_sum0.inputs[1])
    s_scaled = _math(nt, 'MULTIPLY', (420, -240), v1=0.68)
    nt.links.new(s_sum0.outputs[0], s_scaled.inputs[0])
    s_sum1 = _math(nt, 'ADD', (540, -240))
    nt.links.new(s_scaled.outputs[0], s_sum1.inputs[0])
    nt.links.new(y_soot.outputs[0], s_sum1.inputs[1])
    s_sum2 = _math(nt, 'MAXIMUM', (660, -240), clamp=True)
    nt.links.new(s_sum1.outputs[0], s_sum2.inputs[0])
    nt.links.new(crev_soot.outputs[0], s_sum2.inputs[1])

    soot_final = _math(nt, 'MULTIPLY', (790, -240), clamp=True)
    nt.links.new(s_sum2.outputs[0], soot_final.inputs[0])
    nt.links.new(ctrl.outputs["Soot"], soot_final.inputs[1])


    oinfo_c = _new_node(nt, "ShaderNodeObjectInfo", (-780, 560))

    rad_x_sc = _math(nt, 'MULTIPLY', (-560, 520), v1=52.0)
    nt.links.new(sep_y.outputs["X"], rad_x_sc.inputs[0])
    rad_cos = _math(nt, 'COSINE', (-400, 520))
    nt.links.new(rad_x_sc.outputs[0], rad_cos.inputs[0])
    rad_amp = _map_range(nt, -1.0, 1.0, -0.07, +0.10, (-220, 520), clamp=True)
    nt.links.new(rad_cos.outputs[0], rad_amp.inputs[0])
    rad_soot = _math(nt, 'MULTIPLY', (-40, 520))
    nt.links.new(rad_amp.outputs[0], rad_soot.inputs[0])
    nt.links.new(ctrl.outputs["Soot"], rad_soot.inputs[1])


    tile_id_x = _math(nt, 'MULTIPLY', (-560, 680), v1=37.0)
    nt.links.new(oinfo_c.outputs["Random"], tile_id_x.inputs[0])
    tile_id_xf = _math(nt, 'FLOOR', (-400, 680))
    nt.links.new(tile_id_x.outputs[0], tile_id_xf.inputs[0])

    tile_id_y = _math(nt, 'MULTIPLY', (-560, 780), v1=4.2)
    nt.links.new(sep_y.outputs["Y"], tile_id_y.inputs[0])
    tile_id_yf = _math(nt, 'FLOOR', (-400, 780))
    nt.links.new(tile_id_y.outputs[0], tile_id_yf.inputs[0])

    tile_vec = _new_node(nt, "ShaderNodeCombineXYZ", (-220, 740))
    nt.links.new(tile_id_xf.outputs[0], tile_vec.inputs["X"])
    nt.links.new(tile_id_yf.outputs[0], tile_vec.inputs["Y"])
    wn_tile = _new_node(nt, "ShaderNodeTexWhiteNoise", (-40, 740))
    wn_tile.noise_dimensions = '2D'
    nt.links.new(tile_vec.outputs["Vector"], wn_tile.inputs["Vector"])


    new_tile_mask = _map_range(nt, 0.868, 0.874, 0.0, 1.0, (140, 740), clamp=True, interp='LINEAR')
    nt.links.new(wn_tile.outputs["Value"], new_tile_mask.inputs[0])


    soot_with_rad = _math(nt, 'ADD', (880, -240), clamp=True)
    nt.links.new(soot_final.outputs[0], soot_with_rad.inputs[0])
    nt.links.new(rad_soot.outputs[0], soot_with_rad.inputs[1])
    soot_new_scale = _mix_float(nt, (880, -360), a=1.0, b=0.28)
    nt.links.new(new_tile_mask.outputs[0], soot_new_scale.inputs[0])
    soot_effective = _math(nt, 'MULTIPLY', (960, -240), clamp=True)
    nt.links.new(soot_with_rad.outputs[0], soot_effective.inputs[0])
    nt.links.new(soot_new_scale.outputs[0], soot_effective.inputs[1])


    mix_cer0 = _mix_rgba(nt, 'MIX', (120, 320), col_a=rgba(ceramic_a), col_b=rgba(ceramic_b))
    nt.links.new(masks.outputs["Macro_Mottle"], mix_cer0.inputs[0])
    vor_stip = _voronoi(nt, scale=680.0, feature='F1', randomness=0.95, loc=(-60, 420))
    nt.links.new(tc.outputs["Object"], vor_stip.inputs["Vector"])
    stip_v = _map_range(nt, 0.0, 0.85, 0.93, 1.05, (140, 440), clamp=True)
    nt.links.new(vor_stip.outputs["Distance"], stip_v.inputs[0])
    mix_cer = _new_node(nt, "ShaderNodeHueSaturation", (300, 340))
    nt.links.new(stip_v.outputs[0], mix_cer.inputs["Value"])
    nt.links.new(mix_cer0.outputs[2], mix_cer.inputs["Color"])


    nt_col_w = _math(nt, 'MULTIPLY', (300, 740), clamp=True, v1=0.58)
    nt.links.new(new_tile_mask.outputs[0], nt_col_w.inputs[0])
    mix_cer_nt = _mix_rgba(nt, 'MIX', (420, 360), col_b=rgba(0xCEC6B5))
    nt.links.new(nt_col_w.outputs[0], mix_cer_nt.inputs[0])
    nt.links.new(mix_cer.outputs["Color"], mix_cer_nt.inputs[6])

    ramp_soot = _ramp(nt, [
        (0.00, rgba(0xDFD7C8)),
        (0.22, rgba(0xBEB5A6)),
        (0.42, rgba(0x968974)),
        (0.62, rgba(0x5E574F)),
        (0.78, rgba(soot_mid)),
        (0.91, rgba(soot_dark)),
        (1.00, rgba(0x141416)),
    ], loc=(460, 200))
    nt.links.new(soot_effective.outputs[0], ramp_soot.inputs["Fac"])

    col_cer0 = _mix_rgba(nt, 'MULTIPLY', (640, 280), fac=1.0)
    nt.links.new(mix_cer_nt.outputs[2], col_cer0.inputs[6])
    nt.links.new(ramp_soot.outputs["Color"], col_cer0.inputs[7])


    vor_crz = _voronoi(nt, scale=175.0, feature='DISTANCE_TO_EDGE', randomness=0.92, loc=(140, 560))
    nt.links.new(coord_p.outputs["Vector"], vor_crz.inputs["Vector"])
    crz_ln = _map_range(nt, 0.0, 0.022, 1.0, 0.0, (360, 560), clamp=True, interp='LINEAR')
    nt.links.new(vor_crz.outputs["Distance"], crz_ln.inputs[0])
    crz_bd = _map_range(nt, 0.35, 1.35, 0.07, 0.34, (360, 440), clamp=True)
    nt.links.new(ctrl.outputs["SurfaceErosion"], crz_bd.inputs[0])
    crz_w = _math(nt, 'MULTIPLY', (540, 560), clamp=True)
    nt.links.new(crz_ln.outputs[0], crz_w.inputs[0])
    nt.links.new(crz_bd.outputs[0], crz_w.inputs[1])

    col_cer = _mix_rgba(nt, 'MIX', (740, 280), col_b=rgba(0x282624))
    nt.links.new(crz_w.outputs[0], col_cer.inputs[0])
    nt.links.new(col_cer0.outputs[2], col_cer.inputs[6])


    n_slc = _noise(nt, scale=28.0, detail=7.0, roughness=0.72, distortion=0.65, loc=(340, 680))
    nt.links.new(coord_p.outputs["Vector"], n_slc.inputs["Vector"])
    slc_m = _map_range(nt, 0.62, 0.78, 0.0, 0.28, (540, 680), clamp=True)
    nt.links.new(n_slc.outputs["Factor"], slc_m.inputs[0])
    inv_soot = _math(nt, 'SUBTRACT', (540, 460), clamp=True, v0=1.0)
    nt.links.new(soot_final.outputs[0], inv_soot.inputs[1])
    slc_eff = _math(nt, 'MULTIPLY', (680, 640), clamp=True)
    nt.links.new(slc_m.outputs[0], slc_eff.inputs[0])
    nt.links.new(inv_soot.outputs[0], slc_eff.inputs[1])

    col_cer_crust = _mix_rgba(nt, 'MIX', (800, 340), col_b=rgba(0xB8B09E))
    nt.links.new(slc_eff.outputs[0], col_cer_crust.inputs[0])
    nt.links.new(col_cer.outputs[2], col_cer_crust.inputs[6])

    r_cer0 = _mix_float(nt, (720, 40), a=rough_ceramic, b=rough_soot)
    nt.links.new(soot_effective.outputs[0], r_cer0.inputs[0])
    r_cer = _mix_float(nt, (840, 40), b=0.96)
    nt.links.new(slc_eff.outputs[0], r_cer.inputs[0])
    nt.links.new(r_cer0.outputs[0], r_cer.inputs[2])


    vor_sp = _voronoi(nt, scale=18.0, feature='F1', randomness=0.95, loc=(-240, -900))
    nt.links.new(coord_p.outputs["Vector"], vor_sp.inputs["Vector"])
    n_sp = _noise(nt, scale=10.5, detail=9.0, roughness=0.74, distortion=0.72, loc=(-240, -1080))
    nt.links.new(coord_p.outputs["Vector"], n_sp.inputs["Vector"])

    sp_pat0 = _mix_float(nt, (-60, -960), fac=0.34)
    nt.links.new(n_sp.outputs["Factor"], sp_pat0.inputs[2])
    nt.links.new(vor_sp.outputs["Distance"], sp_pat0.inputs[3])


    sp_pc = _map_range(nt, 0.04, 0.72, 0.22, 1.05, (-60, -1100), clamp=True)
    nt.links.new(masks.outputs["Photo_Chip"], sp_pc.inputs[0])
    sp_pat = _mix_float(nt, (60, -960), fac=0.42)
    nt.links.new(sp_pat0.outputs[0], sp_pat.inputs[2])
    nt.links.new(sp_pc.outputs[0], sp_pat.inputs[3])


    sp_edge = _math(nt, 'MULTIPLY_ADD', (0, -1140), v1=0.42, v2=0.42)
    nt.links.new(masks.outputs["EdgeWear_Mask"], sp_edge.inputs[0])
    sp_exp = _math(nt, 'MULTIPLY_ADD', (0, -1280), v1=1.15, v2=0.18)
    nt.links.new(masks.outputs["Exposure_Mask"], sp_exp.inputs[0])

    sp_geo = _math(nt, 'MULTIPLY', (200, -1160), clamp=True)
    nt.links.new(sp_edge.outputs[0], sp_geo.inputs[0])
    nt.links.new(sp_exp.outputs[0], sp_geo.inputs[1])

    sp_bd = _math(nt, 'MULTIPLY', (200, -1320), v1=spall_amount)
    nt.links.new(ctrl.outputs["SurfaceErosion"], sp_bd.inputs[0])

    sp_m0 = _math(nt, 'MULTIPLY', (400, -1040))
    nt.links.new(sp_pat.outputs[0], sp_m0.inputs[0])
    nt.links.new(sp_geo.outputs[0], sp_m0.inputs[1])

    sp_m1 = _math(nt, 'MULTIPLY', (580, -1040))
    nt.links.new(sp_m0.outputs[0], sp_m1.inputs[0])
    nt.links.new(sp_bd.outputs[0], sp_m1.inputs[1])


    spall_halo = _map_range(nt, 0.30, 0.39, 0.0, 1.0, (740, -900), clamp=True, interp='LINEAR')
    nt.links.new(sp_m1.outputs[0], spall_halo.inputs[0])
    spall_mask = _map_range(nt, 0.38, 0.44, 0.0, 1.0, (740, -1080), clamp=True, interp='LINEAR')
    nt.links.new(sp_m1.outputs[0], spall_mask.inputs[0])


    col_halo = _mix_rgba(nt, 'MIX', (860, 280), col_b=rgba(0x161514))
    nt.links.new(spall_halo.outputs[0], col_halo.inputs[0])
    nt.links.new(col_cer_crust.outputs[2], col_halo.inputs[6])

    sub_col0 = _mix_rgba(nt, 'MIX', (720, 480), col_a=rgba(0x4E5460), col_b=rgba(0x6A6156))
    nt.links.new(masks.outputs["Macro_Mottle"], sub_col0.inputs[0])
    sub_col = _mix_rgba(nt, 'MIX', (880, 480), col_b=rgba(0xA4ACB6))
    nt.links.new(masks.outputs["Scratch_Mask"], sub_col.inputs[0])
    nt.links.new(sub_col0.outputs[2], sub_col.inputs[6])

    col_out = _mix_rgba(nt, 'MIX', (1040, 280))
    nt.links.new(spall_mask.outputs[0], col_out.inputs[0])
    nt.links.new(col_halo.outputs[2], col_out.inputs[6])
    nt.links.new(sub_col.outputs[2], col_out.inputs[7])

    m_out = _mix_float(nt, (1040, -100), a=metallic_base, b=0.92)
    nt.links.new(spall_mask.outputs[0], m_out.inputs[0])

    r_sub_sc = _mix_float(nt, (880, -160), a=0.28, b=0.14)
    nt.links.new(masks.outputs["Scratch_Mask"], r_sub_sc.inputs[0])
    r_out = _mix_float(nt, (1040, 60))
    nt.links.new(spall_mask.outputs[0], r_out.inputs[0])
    nt.links.new(r_cer.outputs[0], r_out.inputs[2])
    nt.links.new(r_sub_sc.outputs[0], r_out.inputs[3])


    vor_cer = _voronoi(nt, scale=680.0, feature='F1', randomness=0.95, loc=(320, -560))
    nt.links.new(tc.outputs["Object"], vor_cer.inputs["Vector"])
    n_splat = _noise(nt, scale=310.0, detail=8.0, roughness=0.72, distortion=0.45, loc=(320, -700))
    nt.links.new(coord_p.outputs["Vector"], n_splat.inputs["Vector"])


    sp_h = _math(nt, 'SUBTRACT', (760, -760), clamp=True, v0=1.0)
    nt.links.new(spall_mask.outputs[0], sp_h.inputs[1])

    crz_intact = _math(nt, 'MULTIPLY', (760, -900), clamp=True)
    nt.links.new(crz_ln.outputs[0], crz_intact.inputs[0])
    nt.links.new(sp_h.outputs[0], crz_intact.inputs[1])
    crz_inv = _math(nt, 'SUBTRACT', (920, -900), clamp=True, v0=1.0)
    nt.links.new(crz_intact.outputs[0], crz_inv.inputs[1])

    cer_norm = _combine_scalar_heights(
        nt, [(sp_h.outputs[0], 0.36, 0.00080)], masks.outputs["Bevel_Normal"],
        ref_distance=0.0008, loc=(760, -520),
    )
    cer_norm = _combine_scalar_heights(
        nt,
        [
            (masks.outputs["Micro_Grain"], 0.14, 0.00050),
            (vor_cer.outputs["Distance"], 0.08, 0.00030),
            (n_splat.outputs["Factor"], 0.075, 0.00035),
            (crz_inv.outputs[0], 0.028, 0.00020),
        ],
        cer_norm,
        ref_distance=0.0004,
        loc=(1040, -520),
    )

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (1220, 140))
    nt.links.new(col_out.outputs[2], bsdf.inputs["Base Color"])
    nt.links.new(m_out.outputs[0], bsdf.inputs["Metallic"])
    nt.links.new(r_out.outputs[0], bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.32
    nt.links.new(cer_norm, bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (1500, 140))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(id_hex)
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (1740, 140))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(id_hex)
    mat["f15_pbr_hex"] = int(vp_hex)
    mat.diffuse_color = rgba(vp_hex)
    mat.metallic = metallic_base
    mat.roughness = rough_ceramic
    return mat


def _build_elastomer_or_fabric_material(
    mat, id_hex: int, vp_hex: int, ng_global, ng_masks, ng_wrap,
    *,
    col_a: int, col_b: int,
    crevice_col: int = 0x2E2E30,
    rough_min: float = 0.64,
    rough_max: float = 0.84,
    crevice_rough: float = 0.90,
    crevice_mix: float = 0.55,
    grain_scale: float = 480.0,
    grain_bump: float = 0.08,
    weave_scale: float = 0.0,
    weave_bump: float = 0.0,
    weave_criss_cross: bool = False,
    lacing_bands: bool = False,
    rubber_bloom_amp: float = 0.0,
    tire_tread_wear: bool = False,
    sss_weight: float = 0.0,
    sheen_weight: float = 0.0,
    specular: float = 0.36,
    ao_dist: float = 0.032,
):

    nt = mat.node_tree
    nt.nodes.clear()

    tc = _new_node(nt, "ShaderNodeTexCoord", (-1620, 0))
    ctrl = _new_node(nt, "ShaderNodeGroup", (-1420, -400))
    ctrl.node_tree = ng_global

    masks = _new_node(nt, "ShaderNodeGroup", (-1300, 120))
    masks.node_tree = ng_masks
    nt.links.new(tc.outputs["Object"], masks.inputs["Coord"])
    masks.inputs["AO_Dist"].default_value = ao_dist
    masks.inputs["Fine_Bevel_Rad"].default_value = 0.00035
    masks.inputs["Edge_Bevel_Rad"].default_value = 0.0030
    masks.inputs["Edge_Gain"].default_value = 1.4
    masks.inputs["Grime_Scale"].default_value = 18.0
    masks.inputs["Macro_Scale"].default_value = 5.0
    masks.inputs["Micro_Scale"].default_value = grain_scale

    mix_base = _mix_rgba(nt, 'MIX', (-880, 320), col_a=rgba(col_a), col_b=rgba(col_b))
    nt.links.new(masks.outputs["Macro_Mottle"], mix_base.inputs[0])
    cur_col = mix_base.outputs[2]

    r_base = _map_range(nt, 0.15, 0.85, rough_min, rough_max, (-640, 80), clamp=True)
    nt.links.new(masks.outputs["Macro_Mottle"], r_base.inputs[0])
    cur_r = r_base.outputs[0]


    tire_gravel_bump = None
    if tire_tread_wear:
        tr_cv = _map_range(nt, 0.25, 0.72, 0.0, 0.72, (-640, 480), clamp=True)
        nt.links.new(masks.outputs["Topo_Convex"], tr_cv.inputs[0])
        mix_tr_c = _mix_rgba(nt, 'MIX', (-460, 440), col_b=rgba(0x2F3136))
        nt.links.new(tr_cv.outputs[0], mix_tr_c.inputs[0])
        nt.links.new(cur_col, mix_tr_c.inputs[6])
        cur_col = mix_tr_c.outputs[2]

        mix_tr_r = _mix_float(nt, (-460, 220), b=0.88)
        nt.links.new(tr_cv.outputs[0], mix_tr_r.inputs[0])
        nt.links.new(cur_r, mix_tr_r.inputs[2])
        cur_r = mix_tr_r.outputs[0]


        sc_t = _math(nt, 'MULTIPLY', (-460, 620), clamp=True, v1=0.38)
        nt.links.new(masks.outputs["EdgeWear_Mask"], sc_t.inputs[0])
        mix_tsc = _mix_rgba(nt, 'MIX', (-320, 460), col_b=rgba(0x46484E))
        nt.links.new(sc_t.outputs[0], mix_tsc.inputs[0])
        nt.links.new(cur_col, mix_tsc.inputs[6])
        cur_col = mix_tsc.outputs[2]


        vor_grv = _voronoi(nt, scale=420.0, feature='F1', randomness=0.96, loc=(-640, 760))
        nt.links.new(tc.outputs["Object"], vor_grv.inputs["Vector"])
        grv_dot = _map_range(nt, 0.0, 0.22, 1.0, 0.0, (-440, 760), clamp=True)
        nt.links.new(vor_grv.outputs["Distance"], grv_dot.inputs[0])
        grv_mask = _math(nt, 'MULTIPLY', (-260, 740), clamp=True)
        nt.links.new(grv_dot.outputs[0], grv_mask.inputs[0])
        nt.links.new(masks.outputs["Topo_Concave"], grv_mask.inputs[1])
        tire_gravel_bump = grv_mask.outputs[0]

        mix_grv_c = _mix_rgba(nt, 'MIX', (-160, 500), col_b=rgba(0x8A8478))
        nt.links.new(grv_mask.outputs[0], mix_grv_c.inputs[0])
        nt.links.new(cur_col, mix_grv_c.inputs[6])
        cur_col = mix_grv_c.outputs[2]


        w_tm = _new_node(nt, "ShaderNodeTexWave", (-640, 920))
        w_tm.wave_type = 'RINGS'
        w_tm.rings_direction = 'X'
        w_tm.inputs["Scale"].default_value = 34.0
        w_tm.inputs["Distortion"].default_value = 2.6
        nt.links.new(tc.outputs["Object"], w_tm.inputs["Vector"])
        tm_roff = _map_range(nt, 0.0, 1.0, -0.024, +0.024, (-420, 920), clamp=True)
        nt.links.new(w_tm.outputs["Fac"], tm_roff.inputs[0])
        r_add_tm = _math(nt, 'ADD', (-300, 200), clamp=True)
        nt.links.new(cur_r, r_add_tm.inputs[0])
        nt.links.new(tm_roff.outputs[0], r_add_tm.inputs[1])
        cur_r = r_add_tm.outputs[0]


        rub_sc_p = _map_range(nt, 0.44, 0.86, 0.0, 0.18, (-640, 1080), clamp=True)
        nt.links.new(masks.outputs["Photo_Chip"], rub_sc_p.inputs[0])
        rub_sc_s = _math(nt, 'MULTIPLY', (-640, 1200), clamp=True, v1=0.22)
        nt.links.new(masks.outputs["Scratch_Mask"], rub_sc_s.inputs[0])
        rub_sc0 = _math(nt, 'MAXIMUM', (-460, 1140), clamp=True)
        nt.links.new(rub_sc_p.outputs[0], rub_sc0.inputs[0])
        nt.links.new(rub_sc_s.outputs[0], rub_sc0.inputs[1])
        rub_groov = _math(nt, 'MAXIMUM', (-560, 1260), clamp=True)
        nt.links.new(masks.outputs["Crevice_Mask"], rub_groov.inputs[0])
        nt.links.new(masks.outputs["Topo_Concave"], rub_groov.inputs[1])
        rub_gate = _map_range(nt, 0.0, 1.0, 0.28, 1.0, (-400, 1260), clamp=True)
        nt.links.new(rub_groov.outputs[0], rub_gate.inputs[0])
        rub_sc_g = _math(nt, 'MULTIPLY', (-340, 1180), clamp=True)
        nt.links.new(rub_sc0.outputs[0], rub_sc_g.inputs[0])
        nt.links.new(rub_gate.outputs[0], rub_sc_g.inputs[1])
        rub_sc_eff = _math(nt, 'MULTIPLY', (-220, 1140), clamp=True)
        nt.links.new(rub_sc_g.outputs[0], rub_sc_eff.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], rub_sc_eff.inputs[1])

        mix_rsc_c = _mix_rgba(nt, 'MIX', (-120, 560), col_b=rgba(0x4E4D48))
        nt.links.new(rub_sc_eff.outputs[0], mix_rsc_c.inputs[0])
        nt.links.new(cur_col, mix_rsc_c.inputs[6])
        cur_col = mix_rsc_c.outputs[2]
        mix_rsc_r = _mix_float(nt, (-120, 260), b=0.92)
        nt.links.new(rub_sc_eff.outputs[0], mix_rsc_r.inputs[0])
        nt.links.new(cur_r, mix_rsc_r.inputs[2])
        cur_r = mix_rsc_r.outputs[0]


        geo_rd = _new_node(nt, "ShaderNodeNewGeometry", (-820, 1340))
        sep_rd = _new_node(nt, "ShaderNodeSeparateXYZ", (-640, 1340))
        nt.links.new(geo_rd.outputs["Position"], sep_rd.inputs["Vector"])
        rd_zg = _map_range(nt, -1.615, -1.48, 0.34, 0.0, (-460, 1340), clamp=True)
        nt.links.new(sep_rd.outputs["Z"], rd_zg.inputs[0])
        rd_cc = _math(nt, 'MULTIPLY', (-460, 1460), clamp=True, v1=0.38)
        nt.links.new(masks.outputs["Topo_Concave"], rd_cc.inputs[0])
        rd_zmax = _math(nt, 'MAXIMUM', (-300, 1380), clamp=True)
        nt.links.new(rd_zg.outputs[0], rd_zmax.inputs[0])
        nt.links.new(rd_cc.outputs[0], rd_zmax.inputs[1])
        rd_mod = _math(nt, 'MULTIPLY', (-160, 1380), clamp=True)
        nt.links.new(rd_zmax.outputs[0], rd_mod.inputs[0])
        nt.links.new(masks.outputs["Photo_Smudge"], rd_mod.inputs[1])
        rd_eff = _math(nt, 'MULTIPLY', (-20, 1380), clamp=True)
        nt.links.new(rd_mod.outputs[0], rd_eff.inputs[0])
        nt.links.new(ctrl.outputs["Weathering"], rd_eff.inputs[1])

        mix_rd_c = _mix_rgba(nt, 'MIX', (20, 560), col_b=rgba(0x6D6B64))
        nt.links.new(rd_eff.outputs[0], mix_rd_c.inputs[0])
        nt.links.new(cur_col, mix_rd_c.inputs[6])
        cur_col = mix_rd_c.outputs[2]
        mix_rd_r = _mix_float(nt, (20, 260), b=0.96)
        nt.links.new(rd_eff.outputs[0], mix_rd_r.inputs[0])
        nt.links.new(cur_r, mix_rd_r.inputs[2])
        cur_r = mix_rd_r.outputs[0]


    el_smg_w = _math(nt, 'MULTIPLY', (-420, -240), clamp=True)
    nt.links.new(masks.outputs["Photo_Smudge"], el_smg_w.inputs[0])
    nt.links.new(ctrl.outputs["Weathering"], el_smg_w.inputs[1])
    el_smg_r = _map_range(nt, 0.0, 1.0, -0.055, +0.135, (-240, -240), clamp=True)
    nt.links.new(el_smg_w.outputs[0], el_smg_r.inputs[0])
    r_add_esmg = _math(nt, 'ADD', (-100, 120), clamp=True)
    nt.links.new(cur_r, r_add_esmg.inputs[0])
    nt.links.new(el_smg_r.outputs[0], r_add_esmg.inputs[1])
    cur_r = r_add_esmg.outputs[0]
    el_smg_cf = _math(nt, 'MULTIPLY', (-240, -340), clamp=True, v1=0.16)
    nt.links.new(el_smg_w.outputs[0], el_smg_cf.inputs[0])
    mix_esmg_c = _mix_rgba(nt, 'MIX', (-80, 420), col_b=rgba(0x2D2B27))
    nt.links.new(el_smg_cf.outputs[0], mix_esmg_c.inputs[0])
    nt.links.new(cur_col, mix_esmg_c.inputs[6])
    cur_col = mix_esmg_c.outputs[2]


    if rubber_bloom_amp > 1e-4:
        rb_m = _math(nt, 'MULTIPLY', (-420, 340), clamp=True, v1=rubber_bloom_amp)
        nt.links.new(masks.outputs["EdgeWear_Mask"], rb_m.inputs[0])
        mix_rb = _mix_rgba(nt, 'MIX', (-240, 460), col_b=rgba(0xA4B4C4))
        nt.links.new(rb_m.outputs[0], mix_rb.inputs[0])
        nt.links.new(cur_col, mix_rb.inputs[6])
        cur_col = mix_rb.outputs[2]

    crev_w = _math(nt, 'MULTIPLY', (-880, -140), v0=crevice_mix)
    nt.links.new(ctrl.outputs["Weathering"], crev_w.inputs[1])
    crev_eff = _math(nt, 'MULTIPLY', (-640, -140), clamp=True)
    nt.links.new(masks.outputs["Crevice_Mask"], crev_eff.inputs[0])
    nt.links.new(crev_w.outputs[0], crev_eff.inputs[1])

    mix_col = _mix_rgba(nt, 'MIX', (-240, 300), col_b=rgba(crevice_col))
    nt.links.new(crev_eff.outputs[0], mix_col.inputs[0])
    nt.links.new(cur_col, mix_col.inputs[6])
    cur_col = mix_col.outputs[2]

    mix_r = _mix_float(nt, (-240, 60), b=crevice_rough)
    nt.links.new(crev_eff.outputs[0], mix_r.inputs[0])
    nt.links.new(cur_r, mix_r.inputs[2])
    cur_r = mix_r.outputs[0]


    lacing_bump_out = None
    if lacing_bands:
        w_lac = _new_node(nt, "ShaderNodeTexWave", (-640, -860))
        w_lac.wave_type = 'BANDS'
        w_lac.bands_direction = 'Y'
        w_lac.inputs["Scale"].default_value = 9.5
        w_lac.inputs["Distortion"].default_value = 0.25
        nt.links.new(tc.outputs["Object"], w_lac.inputs["Vector"])
        lac_band = _map_range(nt, 0.90, 0.95, 0.0, 1.0, (-400, -860), clamp=True, interp='LINEAR')
        nt.links.new(w_lac.outputs["Fac"], lac_band.inputs[0])
        lacing_bump_out = lac_band.outputs[0]

        mix_lac_c = _mix_rgba(nt, 'MIX', (-40, 340), col_b=rgba(0x16171A))
        nt.links.new(lac_band.outputs[0], mix_lac_c.inputs[0])
        nt.links.new(cur_col, mix_lac_c.inputs[6])
        cur_col = mix_lac_c.outputs[2]
        mix_lac_r = _mix_float(nt, (-40, 120), b=0.38)
        nt.links.new(lac_band.outputs[0], mix_lac_r.inputs[0])
        nt.links.new(cur_r, mix_lac_r.inputs[2])
        cur_r = mix_lac_r.outputs[0]


    cur_norm = masks.outputs["Bevel_Normal"]
    el_macro_terms = []
    el_micro_terms = []

    if grain_bump > 1e-4:
        el_micro_terms.append((masks.outputs["Micro_Grain"], grain_bump, 0.00045))

    if tire_tread_wear:
        n_epdm = _noise(nt, scale=1450.0, detail=6.0, roughness=0.72, loc=(-260, -480))
        nt.links.new(tc.outputs["Object"], n_epdm.inputs["Vector"])
        el_micro_terms.append((n_epdm.outputs["Factor"], 0.055, 0.00022))

        rub_sc_inv = _math(nt, 'SUBTRACT', (-180, -600), clamp=True, v0=1.0)
        nt.links.new(masks.outputs["Scratch_Mask"], rub_sc_inv.inputs[1])
        el_micro_terms.append((rub_sc_inv.outputs[0], 0.048, 0.00032))

    if tire_gravel_bump is not None:
        el_macro_terms.append((tire_gravel_bump, 0.14, 0.00060))

    if weave_scale > 0.0 and weave_bump > 1e-4:
        w1 = _new_node(nt, "ShaderNodeTexWave", (-640, -480))
        w1.wave_type = 'BANDS'
        w1.bands_direction = 'DIAGONAL'
        w1.inputs["Scale"].default_value = weave_scale
        w1.inputs["Distortion"].default_value = 0.65
        nt.links.new(tc.outputs["Object"], w1.inputs["Vector"])

        if weave_criss_cross:
            mp_w2 = _new_node(nt, "ShaderNodeMapping", (-880, -680))
            mp_w2.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(90.0))
            nt.links.new(tc.outputs["Object"], mp_w2.inputs["Vector"])
            w2 = _new_node(nt, "ShaderNodeTexWave", (-640, -680))
            w2.wave_type = 'BANDS'
            w2.bands_direction = 'DIAGONAL'
            w2.inputs["Scale"].default_value = weave_scale
            w2.inputs["Distortion"].default_value = 0.65
            nt.links.new(mp_w2.outputs["Vector"], w2.inputs["Vector"])

            w_comb = _math(nt, 'MINIMUM', (-400, -560), clamp=True)
            nt.links.new(w1.outputs["Fac"], w_comb.inputs[0])
            nt.links.new(w2.outputs["Fac"], w_comb.inputs[1])
            weave_h = w_comb.outputs[0]

            w_val = _map_range(nt, 0.0, 0.45, 0.55, 0.0, (-200, -560), clamp=True)
            nt.links.new(weave_h, w_val.inputs[0])
            mix_wv_c = _mix_rgba(nt, 'MIX', (60, 280), col_b=rgba(0x463612))
            nt.links.new(w_val.outputs[0], mix_wv_c.inputs[0])
            nt.links.new(cur_col, mix_wv_c.inputs[6])
            cur_col = mix_wv_c.outputs[2]
        else:
            weave_h = w1.outputs["Fac"]

        el_micro_terms.append((weave_h, weave_bump, 0.00040))

    if lacing_bump_out is not None:
        el_macro_terms.append((lacing_bump_out, 0.22, 0.00080))

    cur_norm = _combine_scalar_heights(nt, el_macro_terms, cur_norm, ref_distance=0.0007, loc=(80, -340))
    cur_norm = _combine_scalar_heights(nt, el_micro_terms, cur_norm, ref_distance=0.0004, loc=(240, -340))

    bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", (420, 140))
    nt.links.new(cur_col, bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    nt.links.new(cur_r, bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = specular
    bsdf.inputs["Subsurface Weight"].default_value = sss_weight
    bsdf.inputs["Subsurface Scale"].default_value = 0.003
    bsdf.inputs["Sheen Weight"].default_value = sheen_weight
    bsdf.inputs["Sheen Roughness"].default_value = 0.65
    nt.links.new(cur_norm, bsdf.inputs["Normal"])

    wrap = _new_node(nt, "ShaderNodeGroup", (760, 140))
    wrap.node_tree = ng_wrap
    nt.links.new(bsdf.outputs["BSDF"], wrap.inputs["PBR_Shader"])
    wrap.inputs["ID_Color"].default_value = rgba(id_hex)
    wrap.inputs["Clay_Color"].default_value = rgba(0xC2C6CC)
    nt.links.new(masks.outputs["AO_Raw"], wrap.inputs["AO_Raw"])
    nt.links.new(masks.outputs["Bevel_Normal"], wrap.inputs["Bevel_Normal"])

    out = _new_node(nt, "ShaderNodeOutputMaterial", (1000, 140))
    nt.links.new(wrap.outputs["Shader"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(id_hex)
    mat["f15_pbr_hex"] = int(vp_hex)
    mat.diffuse_color = rgba(vp_hex)
    mat.metallic = 0.0
    mat.roughness = 0.5 * (rough_min + rough_max)
    return mat


def _get_or_new_mat(name: str):
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
    return m


def _swap_mesh_mat_slot(mesh_name: str, old_mat_name: str, new_mat):

    me = bpy.data.meshes.get(mesh_name)
    if me is None:
        return False
    changed = False
    for idx, m in enumerate(me.materials):
        if m is not None and m.name == old_mat_name:
            me.materials[idx] = new_mat
            changed = True
    return changed


def build_all_f100_procedural_materials():

    ng_global = build_ng_global_controls()
    ng_masks  = build_ng_surface_masks()
    ng_heat   = build_ng_heat_oxidize()
    ng_wrap   = build_ng_mode_wrapper(ng_global)

    def _id_hex(mid):
        return COLOR_ID_SPECS[mid][1]

    def _base_mat(mid):
        return _get_or_new_mat(COLOR_ID_SPECS[mid][0])


    _build_metal_material(
        _base_mat(MAT_TI_CASE), _id_hex(MAT_TI_CASE), 0x9CA2A8,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xA6ACB2, col_b=0x828890, crevice_col=0x1E1C1A, wear_col=0xCED4DA,
        metallic=0.98, rough_min=0.18, rough_max=0.38, crevice_rough=0.48, wear_rough=0.14,
        brush_scale=(2.2, 165.0, 2.2), brush_bump=0.048, brush_rough_var=0.062,
        grain_scale=320.0, grain_bump=0.024,
        heat_base=0.04, heat_mottle_amp=0.10, heat_y0=-3.60, heat_y1=-2.80, heat_axial_amp=0.08,
        grazing_tint=0x909498, grazing_mix=0.10, heat_tint_mix=0.14,
        ao_dist=0.052, edge_bevel=0.0036, edge_gain=2.1, edge_wear_mix=0.42, crevice_mix=0.58,
        macro_scale=3.2, island_var=0.07, aniso=0.42,
        scratch_amp=0.48, oil_leak_amp=0.36, oil_canning_bump=0.024,
        lathe_wave_scale=210.0, lathe_wave_bump=0.022, weld_haz_amp=0.48, smudge_rough_amp=0.062,
        swirl_polish_amp=0.65, aniso_rot_var=0.22, water_stain_amp=0.10, fan_rub_strip=True,
        rough_floor=0.16, tangent_mode='RADIAL_Y',
        f82_tint=0xE4EAEF, rough_breakup_amp=0.46,
    )


    _build_metal_material(
        _base_mat(MAT_INCONEL_CASE), _id_hex(MAT_INCONEL_CASE), 0x7A7670,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x8E887F, col_b=0x64605A, crevice_col=0x161412, wear_col=0xC2BDB6,
        metallic=0.97, rough_min=0.20, rough_max=0.42, crevice_rough=0.56, wear_rough=0.15,
        brush_scale=(2.0, 145.0, 2.0), brush_bump=0.050, brush_rough_var=0.068,
        grain_scale=280.0, grain_bump=0.036,
        heat_base=0.18, heat_mottle_amp=0.20, heat_y0=-3.10, heat_y1=-1.55, heat_axial_amp=0.26,
        grazing_tint=0x786E60, grazing_mix=0.20, heat_tint_mix=0.34, use_world_y_for_heat=True,
        soot_amp=0.14, soot_streak_scale=(14.0, 4.0, 14.0),
        ao_dist=0.058, edge_bevel=0.0040, edge_gain=2.0, edge_wear_mix=0.40, crevice_mix=0.70,
        macro_scale=2.8, island_var=0.09, aniso=0.38,
        scratch_amp=0.48, oil_leak_amp=0.66, oil_canning_bump=0.024,
        lathe_wave_scale=190.0, lathe_wave_bump=0.018, weld_haz_amp=0.64, smudge_rough_amp=0.068, rivet_streaks=True,
        swirl_polish_amp=0.55, aniso_rot_var=0.18, water_stain_amp=0.16, oxide_scale_amp=0.34, thin_film_amp=0.18,
        anti_seize_amp=0.38, rough_floor=0.18, tangent_mode='RADIAL_Y',
        f82_tint=0xE6DDD2, rough_breakup_amp=0.48,
    )


    m_id02b = _build_metal_material(
        _get_or_new_mat("Mat_ID02b_Interior_CombustionInconel"), _id_hex(MAT_INCONEL_CASE), 0x353331,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x524D46, col_b=0x2B2926, crevice_col=0x0D0D0F, wear_col=0x968E84,
        metallic=0.84, rough_min=0.36, rough_max=0.72, crevice_rough=0.92, wear_rough=0.22,
        brush_scale=(12.0, 4.0, 12.0), brush_bump=0.044, brush_rough_var=0.085,
        grain_scale=360.0, grain_bump=0.075,
        heat_base=0.48, heat_mottle_amp=0.24, grazing_tint=0x5A5248, grazing_mix=0.22, heat_tint_mix=0.28,
        soot_amp=0.56, soot_streak_scale=(18.0, 2.2, 18.0),
        ao_dist=0.060, edge_bevel=0.0032, edge_gain=2.1, edge_wear_mix=0.42, crevice_mix=0.82,
        macro_scale=4.5, island_var=0.10,
        scratch_amp=0.38, oxide_scale_amp=0.68, corrugation_ash_amp=0.46, weld_haz_amp=0.42,
        rough_floor=0.22, f82_tint=0xDFD8CE, rough_breakup_amp=0.76,
    )


    _build_metal_material(
        _base_mat(MAT_WAFFLE_DUCT), _id_hex(MAT_WAFFLE_DUCT), 0x9EA4AC,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x848B94, col_b=0x626872, crevice_col=0x221D18, wear_col=0xCAD0D8,
        metallic=0.97, rough_min=0.34, rough_max=0.52, crevice_rough=0.58, wear_rough=0.12,
        brush_scale=(2.6, 110.0, 2.6), brush_bump=0.030, brush_rough_var=0.052,
        grain_scale=360.0, grain_bump=0.024,
        heat_base=0.04, heat_mottle_amp=0.14, heat_y0=-1.45, heat_y1=-0.68, heat_axial_amp=0.26,
        grazing_tint=0x9E9380, grazing_mix=0.18, heat_tint_mix=0.28,
        ao_dist=0.045, edge_bevel=0.0034, edge_gain=2.2, edge_wear_mix=0.42, crevice_mix=0.60,
        macro_scale=3.4, island_var=0.05, aniso=0.36,
        scratch_amp=0.48, oil_leak_amp=0.42, oil_canning_bump=0.020,
        waffle_chem_mill=True, waffle_aft_transition=True, weld_haz_amp=0.46, smudge_rough_amp=0.048,
        swirl_polish_amp=0.45, aniso_rot_var=0.20, water_stain_amp=0.11,
        rough_floor=0.12, tangent_mode='RADIAL_Y',
        f82_tint=0xE4EAEF, rough_breakup_amp=0.54,
    )


    _build_metal_material(
        _base_mat(MAT_SPINNER_IGV), _id_hex(MAT_SPINNER_IGV), 0x82868B,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x969BA2, col_b=0x6C7178, crevice_col=0x1A1B1E, wear_col=0xC8D0D8,
        metallic=0.99, rough_min=0.16, rough_max=0.34, crevice_rough=0.44, wear_rough=0.13,
        brush_scale=(1.4, 240.0, 1.4), brush_bump=0.042, brush_rough_var=0.062,
        grain_scale=380.0, grain_bump=0.016,
        heat_base=0.02, heat_mottle_amp=0.06, grazing_tint=0x888C92, grazing_mix=0.10, heat_tint_mix=0.08,
        ao_dist=0.045, edge_bevel=0.0028, edge_gain=2.3, edge_wear_mix=0.45, crevice_mix=0.50,
        macro_scale=5.0, island_var=0.08, aniso=0.52,
        scratch_amp=0.54,
        lathe_wave_scale=230.0, lathe_wave_bump=0.045, smudge_rough_amp=0.024,
        swirl_polish_amp=0.38, aniso_rot_var=0.12, spinner_nose_wear=True, igv_leading_edge=True,
        rough_floor=0.15, tangent_mode='RADIAL_Y',
        f82_tint=0xE6ECF2, rough_breakup_amp=0.72,
    )


    m_id04b = _build_metal_material(
        _get_or_new_mat("Mat_ID04b_Afterburner_HotSpraybars"), _id_hex(MAT_SPINNER_IGV), 0x46423D,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x58524A, col_b=0x34312E, crevice_col=0x0F0F11, wear_col=0x968E84,
        metallic=0.86, rough_min=0.34, rough_max=0.68, crevice_rough=0.88, wear_rough=0.20,
        brush_scale=(16.0, 3.0, 16.0), brush_bump=0.038, brush_rough_var=0.075,
        grain_scale=380.0, grain_bump=0.062,
        heat_base=0.54, heat_mottle_amp=0.22, grazing_tint=0x5E564C, grazing_mix=0.20, heat_tint_mix=0.30,
        soot_amp=0.48, soot_streak_scale=(16.0, 2.0, 16.0),
        ao_dist=0.048, edge_bevel=0.0028, edge_gain=2.0, edge_wear_mix=0.36, crevice_mix=0.78,
        macro_scale=5.0, island_var=0.12,
        oxide_scale_amp=0.66, corrugation_ash_amp=0.42, weld_haz_amp=0.38,
        rough_floor=0.20, f82_tint=0xE0D8CE, rough_breakup_amp=0.72,
    )


    _build_metal_material(
        _base_mat(MAT_ROTARY_BLADES), _id_hex(MAT_ROTARY_BLADES), 0x7A828C,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x929AA4, col_b=0x626A74, crevice_col=0x181A1D, wear_col=0xC8D0D8,
        metallic=0.99, rough_min=0.16, rough_max=0.36, crevice_rough=0.44, wear_rough=0.13,
        brush_scale=(18.0, 160.0, 18.0), brush_bump=0.038, brush_rough_var=0.060,
        grain_scale=420.0, grain_bump=0.018,
        ao_dist=0.042, edge_bevel=0.0026, edge_gain=2.5, edge_wear_mix=0.56, crevice_mix=0.48,
        macro_scale=6.0, island_var=0.10, aniso=0.50,
        scratch_amp=0.54, smudge_rough_amp=0.048,
        swirl_polish_amp=0.34, aniso_rot_var=0.14, blade_tip_erosion=True,
        rough_floor=0.15, tangent_mode='RADIAL_Y',
        f82_tint=0xE4EAEF, rough_breakup_amp=0.74,
    )


    m_id05b = _build_metal_material(
        _get_or_new_mat("Mat_ID05b_Hot_LPTTurbine"), _id_hex(MAT_ROTARY_BLADES), 0x4E4742,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x645B52, col_b=0x3B3733, crevice_col=0x111011, wear_col=0xA49C92,
        metallic=0.94, rough_min=0.24, rough_max=0.52, crevice_rough=0.74, wear_rough=0.14,
        brush_scale=(14.0, 120.0, 14.0), brush_bump=0.044, brush_rough_var=0.070,
        grain_scale=360.0, grain_bump=0.048,
        heat_base=0.52, heat_mottle_amp=0.22, grazing_tint=0x5C544C, grazing_mix=0.20, heat_tint_mix=0.32,
        soot_amp=0.38, soot_streak_scale=(20.0, 3.0, 20.0),
        ao_dist=0.045, edge_bevel=0.0026, edge_gain=2.2, edge_wear_mix=0.42, crevice_mix=0.72,
        macro_scale=5.5, island_var=0.12,
        corrugation_ash_amp=0.38, oxide_scale_amp=0.62, swirl_polish_amp=0.25,
        rough_floor=0.14, f82_tint=0xE4DDD4, rough_breakup_amp=0.78,
    )


    m_id05c = _build_metal_material(
        _get_or_new_mat("Mat_ID05c_Hot_LPTSplineCap"), _id_hex(MAT_BRASS_FITTING), 0x363330,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x44403C, col_b=0x322F2C, crevice_col=0x161515, wear_col=0x605C58,
        metallic=0.84, rough_min=0.58, rough_max=0.86, crevice_rough=0.94, wear_rough=0.42,
        brush_scale=(18.0, 18.0, 18.0), brush_bump=0.032, brush_rough_var=0.065,
        grain_scale=420.0, grain_bump=0.06,
        heat_base=0.62, heat_mottle_amp=0.18, grazing_tint=0x2A2828, grazing_mix=0.10, heat_tint_mix=0.18,
        soot_amp=0.78, soot_streak_scale=(14.0, 2.0, 14.0),
        ao_dist=0.055, edge_bevel=0.0024, edge_gain=1.2, edge_wear_mix=0.12, crevice_mix=0.85,
        macro_scale=5.0, island_var=0.06, specular=0.26, rough_floor=0.40,
        f82_tint=0xDCD8D2, rough_breakup_amp=0.68,
    )


    _build_metal_material(
        _base_mat(MAT_MACHINED_ALU), _id_hex(MAT_MACHINED_ALU), 0xB4BAC0,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xBCC2C8, col_b=0x90969E, crevice_col=0x22201E, wear_col=0xD4DAE0,
        metallic=0.99, rough_min=0.11, rough_max=0.30, crevice_rough=0.44, wear_rough=0.09,
        brush_scale=(45.0, 120.0, 45.0), brush_bump=0.036, brush_rough_var=0.058,
        grain_scale=340.0, grain_bump=0.018,
        ao_dist=0.036, edge_bevel=0.0028, edge_gain=2.2, edge_wear_mix=0.46, crevice_mix=0.54,
        macro_scale=6.0, island_var=0.11, aniso=0.06,
        lathe_wave_scale=220.0, lathe_wave_bump=0.016,
        swirl_polish_amp=0.55, weld_haz_amp=0.24, rough_floor=0.09, tangent_mode='NONE',
        f82_tint=0xEAF0F6, rough_breakup_amp=0.74,
    )


    m_id06b = _build_metal_material(
        _get_or_new_mat("Mat_ID06b_Nozzle_MachinedTracks"), _id_hex(MAT_MACHINED_ALU), 0x565759,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x6E7075, col_b=0x44464B, crevice_col=0x131416, wear_col=0xC6CBD2,
        metallic=0.97, rough_min=0.14, rough_max=0.40, crevice_rough=0.62, wear_rough=0.11,
        brush_scale=(35.0, 8.0, 35.0), brush_bump=0.044, brush_rough_var=0.068,
        grain_scale=320.0, grain_bump=0.034,
        heat_base=0.58, heat_mottle_amp=0.22, heat_y0=0.0, heat_y1=0.75, heat_axial_amp=0.18,
        grazing_tint=0x525864, grazing_mix=0.22, heat_tint_mix=0.28,
        soot_amp=0.22, soot_streak_scale=(22.0, 3.0, 22.0),
        ao_dist=0.042, edge_bevel=0.0032, edge_gain=2.5, edge_wear_mix=0.60, crevice_mix=0.74,
        macro_scale=5.0, island_var=0.10,
        scratch_amp=0.74, oxide_scale_amp=0.62, swirl_polish_amp=0.35, rough_floor=0.11,
        f82_tint=0xDFE6EE, rough_breakup_amp=0.84, nozzle_nickel_antiseize=0.78,
    )


    _build_painted_metal_material(
        _base_mat(MAT_AVIONICS_BOX), _id_hex(MAT_AVIONICS_BOX), 0x262B33,
        ng_global, ng_masks, ng_wrap,
        paint_col_a=0x2A2F38, paint_col_b=0x20242C,
        primer_col=0x5D633A, bare_metal_col=0xC2CAD2, crevice_col=0x101216,
        paint_metallic=0.0, rough_min=0.36, rough_max=0.52, crevice_rough=0.68, bare_rough=0.19,
        coat_weight=0.22, coat_rough=0.26,
        peel_scale=900.0, peel_bump=0.095, chip_depth_bump=0.10,
        ao_dist=0.044, edge_bevel=0.0032, edge_gain=2.7, chip_amount=0.52, crevice_mix=0.66,
        grime_scale=18.0, macro_scale=6.0,
        rust_amount=0.0, scratch_amp=0.58, oil_leak_amp=0.34,
        top_dust_fade_amp=0.22, grease_smudge_amp=0.45,
        spatter_amp=0.30, touchup_amp=0.22, clearcoat_swirl_amp=0.52,
        chip_thresh=0.28,
    )


    m_id07b = _build_metal_material(
        _get_or_new_mat("Mat_ID07b_Nozzle_DarkOxidizedSkin"), _id_hex(MAT_AVIONICS_BOX), 0x444850,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x52565E, col_b=0x3A3E45, crevice_col=0x121316, wear_col=0x9096A0,
        metallic=0.84, rough_min=0.32, rough_max=0.62, crevice_rough=0.80, wear_rough=0.18,
        brush_scale=(24.0, 5.0, 24.0), brush_bump=0.048, brush_rough_var=0.075,
        grain_scale=360.0, grain_bump=0.055,
        heat_base=0.72, heat_mottle_amp=0.18, grazing_tint=0x404652, grazing_mix=0.18, heat_tint_mix=0.26,
        soot_amp=0.36, soot_streak_scale=(26.0, 2.8, 26.0),
        ao_dist=0.038, edge_bevel=0.0028, edge_gain=2.3, edge_wear_mix=0.44, crevice_mix=0.74,
        macro_scale=6.0, island_var=0.15,
        scratch_amp=0.68, corrugation_ash_amp=0.28, smudge_rough_amp=0.08, oxide_scale_amp=0.75,
        exit_lip_soot_amp=0.42, rough_floor=0.18,
        f82_tint=0xDCE2EA, rough_breakup_amp=0.82,
    )


    _build_metal_material(
        _base_mat(MAT_PUMPS_OILTANK), _id_hex(MAT_PUMPS_OILTANK), 0x94999E,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xA0A5AC, col_b=0x787D84, crevice_col=0x1D1B18, wear_col=0xD4DAE0,
        metallic=0.94, rough_min=0.32, rough_max=0.56, crevice_rough=0.46, wear_rough=0.12,
        brush_scale=(22.0, 22.0, 22.0), brush_bump=0.016, brush_rough_var=0.055,
        grain_scale=640.0, grain_bump=0.060,
        cast_voronoi_scale=480.0, cast_voronoi_bump=0.065,
        ao_dist=0.048, edge_bevel=0.0038, edge_gain=2.0, edge_wear_mix=0.46, crevice_mix=0.64,
        macro_scale=4.5, island_var=0.10,
        scratch_amp=0.40, oil_leak_amp=0.76,
        cast_machined_contrast=0.85, smudge_rough_amp=0.072, water_stain_amp=0.15, swirl_polish_amp=0.28,
        rough_floor=0.11, f82_tint=0xE4EAEF, rough_breakup_amp=0.78,
    )


    _build_metal_material(
        _base_mat(MAT_RIGID_TUBE), _id_hex(MAT_RIGID_TUBE), 0xBCC3CC,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xC2C8D0, col_b=0x9AA2AC, crevice_col=0x2A2E34, wear_col=0xD6DCE2,
        metallic=0.99, rough_min=0.15, rough_max=0.32, crevice_rough=0.38, wear_rough=0.12,
        brush_scale=(85.0, 10.0, 85.0), brush_bump=0.022, brush_rough_var=0.048,
        grain_scale=420.0, grain_bump=0.008,
        heat_base=0.01, heat_mottle_amp=0.04, heat_y0=-3.40, heat_y1=-1.20, heat_axial_amp=0.04,
        grazing_tint=0xBCC4CE, grazing_mix=0.08, heat_tint_mix=0.06,
        ao_dist=0.014, edge_bevel=0.0020, edge_gain=1.7, edge_wear_mix=0.32, crevice_mix=0.26,
        macro_scale=6.5, island_var=0.06, aniso=0.05,
        scratch_amp=0.30, oil_leak_amp=0.16,
        tube_bend_sheen=0.85, swirl_polish_amp=0.45, weld_haz_amp=0.28,
        fluid_id_bands=False, rough_floor=0.14, tangent_mode='NONE',
        f82_tint=0xE6ECF2, rough_breakup_amp=0.72,
    )


    m_id09b = _build_metal_material(
        _get_or_new_mat("Mat_ID09b_Nozzle_HotTieRods"), _id_hex(MAT_RIGID_TUBE), 0x6A6C70,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x808388, col_b=0x52555B, crevice_col=0x161618, wear_col=0xC6CAD0,
        metallic=0.97, rough_min=0.16, rough_max=0.42, crevice_rough=0.64, wear_rough=0.12,
        brush_scale=(55.0, 12.0, 55.0), brush_bump=0.032, brush_rough_var=0.062,
        grain_scale=360.0, grain_bump=0.032,
        heat_base=0.46, heat_mottle_amp=0.20, heat_y0=-0.20, heat_y1=0.80, heat_axial_amp=0.22,
        grazing_tint=0x606672, grazing_mix=0.22, heat_tint_mix=0.28, use_world_y_for_heat=True,
        soot_amp=0.22, soot_streak_scale=(22.0, 3.0, 22.0),
        ao_dist=0.024, edge_bevel=0.0022, edge_gain=2.1, edge_wear_mix=0.46, crevice_mix=0.66,
        macro_scale=6.0, island_var=0.12, aniso=0.05,
        scratch_amp=0.54, oxide_scale_amp=0.48, swirl_polish_amp=0.30, thin_film_amp=0.28,
        rough_floor=0.12, tangent_mode='NONE',
        f82_tint=0xE0E6EE, rough_breakup_amp=0.80,
    )


    _build_metal_material(
        _base_mat(MAT_BRAID_HOSE), _id_hex(MAT_BRAID_HOSE), 0xB4BCC4,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xBEC6CE, col_b=0x848C96, crevice_col=0x0D0F11, wear_col=0xD4DAE2,
        metallic=0.98, rough_min=0.16, rough_max=0.36, crevice_rough=0.52, wear_rough=0.12,
        brush_scale=(180.0, 180.0, 180.0), brush_bump=0.028, brush_rough_var=0.052,
        grain_scale=620.0, grain_bump=0.015,
        ao_dist=0.014, fine_bevel=0.00016, edge_bevel=0.0011, edge_gain=1.5, edge_wear_mix=0.16,
        crevice_mix=0.70, grime_scale=32.0, macro_scale=7.0, island_var=0.10, aniso=0.05,
        oil_leak_amp=0.30, braid_micro_weave=True, rough_floor=0.12, tangent_mode='NONE',
        f82_tint=0xE6ECF2, rough_breakup_amp=0.70,
    )


    _build_elastomer_or_fabric_material(
        _base_mat(MAT_YELLOW_HARNESS), _id_hex(MAT_YELLOW_HARNESS), 0xD2AA1E,
        ng_global, ng_masks, ng_wrap,
        col_a=0xDFB624, col_b=0xB89016, crevice_col=0x4E3E14,
        rough_min=0.42, rough_max=0.60, crevice_rough=0.72, crevice_mix=0.36,
        grain_scale=460.0, grain_bump=0.052, weave_scale=185.0, weave_bump=0.14,
        weave_criss_cross=True, lacing_bands=True,
        sss_weight=0.08, sheen_weight=0.44, specular=0.34, ao_dist=0.022,
    )


    _build_metal_material(
        _base_mat(MAT_BRASS_FITTING), _id_hex(MAT_BRASS_FITTING), 0xC4AD78,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xCCB47C, col_b=0x9A845A, crevice_col=0x261F14, wear_col=0xDACB9E,
        metallic=0.99, rough_min=0.12, rough_max=0.34, crevice_rough=0.44, wear_rough=0.09,
        brush_scale=(50.0, 140.0, 50.0), brush_bump=0.032, brush_rough_var=0.056,
        grain_scale=400.0, grain_bump=0.015,
        ao_dist=0.022, fine_bevel=0.00024, edge_bevel=0.0022, edge_gain=2.3, edge_wear_mix=0.48,
        crevice_mix=0.56, macro_scale=8.0, island_var=0.18,
        oil_leak_amp=0.44, torque_seal_amp=0.58, torque_seal_tight=True, bolt_alloy_var=0.24, swirl_polish_amp=0.35,
        rough_floor=0.09, f82_tint=0xFCECC6, rough_breakup_amp=0.76,
    )


    m_id12b = _build_metal_material(
        _get_or_new_mat("Mat_ID12b_Nozzle_HotBushings"), _id_hex(MAT_BRASS_FITTING), 0x987D4E,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xAE8E56, col_b=0x6E5632, crevice_col=0x14120E, wear_col=0xD6BE84,
        metallic=0.98, rough_min=0.13, rough_max=0.38, crevice_rough=0.58, wear_rough=0.10,
        brush_scale=(40.0, 120.0, 40.0), brush_bump=0.034, brush_rough_var=0.062,
        heat_base=0.40, heat_mottle_amp=0.18, grazing_tint=0x90784C, grazing_mix=0.20, heat_tint_mix=0.26,
        soot_amp=0.18, ao_dist=0.024, fine_bevel=0.00024, edge_bevel=0.0022, edge_gain=2.4,
        edge_wear_mix=0.54, crevice_mix=0.66, macro_scale=8.0, island_var=0.16,
        oxide_scale_amp=0.42, anti_seize_amp=0.36, swirl_polish_amp=0.32, rough_floor=0.10,
        f82_tint=0xF6E4B8, rough_breakup_amp=0.80,
    )


    _build_elastomer_or_fabric_material(
        _base_mat(MAT_ADEL_CUSHION), _id_hex(MAT_ADEL_CUSHION), 0x4890C6,
        ng_global, ng_masks, ng_wrap,
        col_a=0x529ED4, col_b=0x3A7EB4, crevice_col=0x1C3E5E,
        rough_min=0.32, rough_max=0.46, crevice_rough=0.58, crevice_mix=0.28,
        grain_scale=520.0, grain_bump=0.040, rubber_bloom_amp=0.28, sss_weight=0.14, sheen_weight=0.0, specular=0.48,
        ao_dist=0.016,
    )


    _build_metal_material(
        _base_mat(MAT_CONNECTOR_38999), _id_hex(MAT_CONNECTOR_38999), 0x646A48,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x727850, col_b=0x4D5336, crevice_col=0x1A1D13, wear_col=0xC8CEC9,
        metallic=0.92, rough_min=0.20, rough_max=0.44, crevice_rough=0.56, wear_rough=0.11,
        brush_scale=(30.0, 140.0, 30.0), brush_bump=0.034, brush_rough_var=0.060,
        grain_scale=420.0, grain_bump=0.03,
        heat_base=0.16, heat_mottle_amp=0.12, grazing_tint=0x54786C, grazing_mix=0.26, heat_tint_mix=0.14,
        ao_dist=0.022, fine_bevel=0.00022, edge_bevel=0.0018, edge_gain=2.8, edge_wear_mix=0.58,
        crevice_mix=0.64, macro_scale=9.0, island_var=0.16, swirl_polish_amp=0.30,
        rough_floor=0.11, f82_tint=0xDCE6DC, rough_breakup_amp=0.76,
    )


    _build_metal_material(
        _base_mat(MAT_FASTENER), _id_hex(MAT_FASTENER), 0x9EA4AC,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xB6BCC4, col_b=0x7A8088, crevice_col=0x181715, wear_col=0xD6DCE2,
        metallic=0.99, rough_min=0.11, rough_max=0.32, crevice_rough=0.48, wear_rough=0.09,
        brush_scale=(65.0, 65.0, 65.0), brush_bump=0.026, brush_rough_var=0.055,
        grain_scale=460.0, grain_bump=0.014,
        heat_base=0.04, heat_mottle_amp=0.12, heat_y0=-1.60, heat_y1=0.85, heat_axial_amp=0.34,
        grazing_tint=0x827A6E, grazing_mix=0.16, heat_tint_mix=0.26, use_world_y_for_heat=True,
        ao_dist=0.016, fine_bevel=0.00018, edge_bevel=0.0014, edge_gain=2.8, edge_wear_mix=0.60,
        crevice_mix=0.60, macro_scale=12.0, island_var=0.20,
        oil_leak_amp=0.26, torque_seal_amp=0.88, bolt_alloy_var=0.38, rough_floor=0.09,
        f82_tint=0xE6ECF2, rough_breakup_amp=0.74,
    )


    m_id15b = _build_metal_material(
        _get_or_new_mat("Mat_ID15b_Interior_HotFasteners"), _id_hex(MAT_FASTENER), 0x3A3734,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x48443E, col_b=0x262422, crevice_col=0x0E0E10, wear_col=0x6A655E,
        metallic=0.84, rough_min=0.36, rough_max=0.70, crevice_rough=0.92, wear_rough=0.24,
        brush_scale=(30.0, 30.0, 30.0), brush_bump=0.026, brush_rough_var=0.065,
        grain_scale=420.0, grain_bump=0.045,
        heat_base=0.58, heat_mottle_amp=0.20, grazing_tint=0x4E4842, grazing_mix=0.18, heat_tint_mix=0.28,
        soot_amp=0.62, soot_streak_scale=(18.0, 2.5, 18.0),
        ao_dist=0.018, fine_bevel=0.00018, edge_bevel=0.0014, edge_gain=1.6, edge_wear_mix=0.22,
        crevice_mix=0.82, macro_scale=12.0, island_var=0.14,
        oxide_scale_amp=0.55, rough_floor=0.24,
        f82_tint=0xE0DAD2, rough_breakup_amp=0.72,
    )


    _build_metal_material(
        _base_mat(MAT_THREAD), _id_hex(MAT_THREAD), 0x868C94,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x9CA2AA, col_b=0x6E747C, crevice_col=0x221C15, wear_col=0xD4DAE0,
        metallic=0.98, rough_min=0.13, rough_max=0.36, crevice_rough=0.50, wear_rough=0.09,
        brush_scale=(80.0, 80.0, 80.0), brush_bump=0.028, brush_rough_var=0.052,
        ao_dist=0.012, fine_bevel=0.00016, edge_bevel=0.0012, edge_gain=2.6, edge_wear_mix=0.52,
        crevice_mix=0.74, macro_scale=14.0, island_var=0.14, anti_seize_amp=0.68, rough_floor=0.09,
        f82_tint=0xE4EAEF, rough_breakup_amp=0.74,
    )


    _build_metal_material(
        _base_mat(MAT_LOCKWIRE), _id_hex(MAT_LOCKWIRE), 0xC2C8D0,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xC8CED6, col_b=0x9EA6B0, crevice_col=0x2E3034, wear_col=0xD8DEE4,
        metallic=0.99, rough_min=0.09, rough_max=0.22, crevice_rough=0.30, wear_rough=0.08,
        brush_scale=(120.0, 120.0, 120.0), brush_bump=0.018, brush_rough_var=0.032,
        grain_scale=600.0, grain_bump=0.008,
        ao_dist=0.008, fine_bevel=0.00014, edge_bevel=0.0008, edge_gain=1.5, edge_wear_mix=0.32,
        crevice_mix=0.24, macro_scale=16.0, island_var=0.08, rough_floor=0.08,
        f82_tint=0xEAEFF5, rough_breakup_amp=0.64,
    )


    _build_painted_metal_material(
        _base_mat(MAT_PLATE_BASE), _id_hex(MAT_PLATE_BASE), 0x24282F,
        ng_global, ng_masks, ng_wrap,
        paint_col_a=0x282C33, paint_col_b=0x20242B,
        primer_col=0x4E535A, bare_metal_col=0xB8C0C8, crevice_col=0x0E1014,
        paint_metallic=0.0, rough_min=0.28, rough_max=0.40, crevice_rough=0.54, bare_rough=0.18,
        coat_weight=0.20, coat_rough=0.24, peel_scale=520.0, peel_bump=0.018, chip_depth_bump=0.04,
        ao_dist=0.018, fine_bevel=0.00020, edge_bevel=0.0016, edge_gain=1.6, chip_amount=0.22,
        crevice_mix=0.48, scratch_amp=0.05, oil_leak_amp=0.12, grease_smudge_amp=0.24, clearcoat_swirl_amp=0.18,
    )
    _build_metal_material(
        _base_mat(MAT_3D_TEXT), _id_hex(MAT_3D_TEXT), 0xC4CAD0,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xC8CED4, col_b=0xA4AAB2, crevice_col=0x34383E, wear_col=0xD8DEE4,
        metallic=0.96, rough_min=0.16, rough_max=0.36, crevice_rough=0.46, wear_rough=0.11,
        brush_scale=(60.0, 60.0, 60.0), brush_bump=0.018, brush_rough_var=0.042,
        ao_dist=0.008, fine_bevel=0.00014, edge_bevel=0.0008, edge_gain=1.6, edge_wear_mix=0.30,
        crevice_mix=0.32, island_var=0.06, swirl_polish_amp=0.25, rough_floor=0.11,
        f82_tint=0xE8EEF4, rough_breakup_amp=0.68,
    )


    m_id18b = _build_metal_material(
        _get_or_new_mat("Mat_ID18b_Shroud_StencilPlate"), _id_hex(MAT_PLATE_BASE), 0x34322F,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x403D39, col_b=0x242220, crevice_col=0x100F0D, wear_col=0x7C7870,
        metallic=0.90, rough_min=0.36, rough_max=0.60, crevice_rough=0.74, wear_rough=0.20,
        brush_scale=(1.8, 185.0, 1.8), brush_bump=0.044, brush_rough_var=0.065,
        heat_base=0.15, heat_mottle_amp=0.10, grazing_tint=0x6A665E, grazing_mix=0.14, heat_tint_mix=0.16,
        ao_dist=0.022, fine_bevel=0.00022, edge_bevel=0.0020, edge_gain=2.0, edge_wear_mix=0.44,
        crevice_mix=0.56, island_var=0.05, aniso=0.28, swirl_polish_amp=0.45, smudge_rough_amp=0.065,
        rough_floor=0.20, tangent_mode='RADIAL_Y',
        f82_tint=0xE2DDD4, rough_breakup_amp=0.72,
    )
    m_id19b = _build_painted_metal_material(
        _get_or_new_mat("Mat_ID19b_Shroud_RedStencil"), _id_hex(MAT_3D_TEXT), 0xB43320,
        ng_global, ng_masks, ng_wrap,
        paint_col_a=0xC23824, paint_col_b=0x902618,
        primer_col=0x6A5C48, bare_metal_col=0x8E7D58, crevice_col=0x38120C,
        paint_metallic=0.0, rough_min=0.50, rough_max=0.68, crevice_rough=0.76, bare_rough=0.34,
        coat_weight=0.0, peel_scale=460.0, peel_bump=0.04, chip_depth_bump=0.06,
        ao_dist=0.010, fine_bevel=0.00014, edge_bevel=0.0010, edge_gain=2.0, chip_amount=0.32,
        crevice_mix=0.28,
    )


    m_id18c = _build_metal_material(
        _get_or_new_mat("Mat_ID18c_Nozzle_StampedTag"), _id_hex(MAT_PLATE_BASE), 0x666C74,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x6A7078, col_b=0x4A5058, crevice_col=0x1A1C20, wear_col=0x989EA6,
        metallic=0.90, rough_min=0.40, rough_max=0.56, crevice_rough=0.68, wear_rough=0.26,
        brush_scale=(2.0, 180.0, 2.0), brush_bump=0.028, brush_rough_var=0.042,
        heat_base=0.18, heat_mottle_amp=0.12, grazing_tint=0x646A72, grazing_mix=0.12, heat_tint_mix=0.14,
        soot_amp=0.18, ao_dist=0.014, fine_bevel=0.00018, edge_bevel=0.0014, edge_gain=1.8,
        edge_wear_mix=0.26, crevice_mix=0.48, island_var=0.08, aniso=0.0, swirl_polish_amp=0.20,
        rough_floor=0.26, tangent_mode='NONE',
        f82_tint=0xE2E8EE, rough_breakup_amp=0.55,
    )
    m_id19d = _build_metal_material(
        _get_or_new_mat("Mat_ID19d_Nozzle_StampedText"), _id_hex(MAT_3D_TEXT), 0x202226,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x26282C, col_b=0x14161A, crevice_col=0x0C0D0F, wear_col=0x464A52,
        metallic=0.0, rough_min=0.46, rough_max=0.62, crevice_rough=0.72, wear_rough=0.32,
        heat_base=0.14, heat_mottle_amp=0.08, grazing_tint=0x30343A, grazing_mix=0.08, heat_tint_mix=0.10,
        soot_amp=0.14, ao_dist=0.008, fine_bevel=0.00014, edge_bevel=0.0008, edge_gain=1.4,
        edge_wear_mix=0.20, crevice_mix=0.40, island_var=0.06, rough_floor=0.30,
    )


    m_id18d = _build_painted_metal_material(
        _get_or_new_mat("Mat_ID18d_Stand_DataPlate"), _id_hex(MAT_PLATE_BASE), 0x24282F,
        ng_global, ng_masks, ng_wrap,
        paint_col_a=0x282C33, paint_col_b=0x20242B,
        primer_col=0x585C60, bare_metal_col=0xB8BEC4, crevice_col=0x101216,
        paint_metallic=0.0, rough_min=0.32, rough_max=0.48, crevice_rough=0.64, bare_rough=0.20,
        coat_weight=0.24, coat_rough=0.24, peel_scale=420.0, peel_bump=0.032, chip_depth_bump=0.08,
        ao_dist=0.025, fine_bevel=0.00024, edge_bevel=0.0022, edge_gain=2.0, chip_amount=0.44,
        crevice_mix=0.52, grease_smudge_amp=0.30, top_dust_fade_amp=0.18, clearcoat_swirl_amp=0.48,
    )
    m_id19e = _build_painted_metal_material(
        _get_or_new_mat("Mat_ID19e_Stand_StencilText"), _id_hex(MAT_3D_TEXT), 0xD8DCE0,
        ng_global, ng_masks, ng_wrap,
        paint_col_a=0xDCE0E4, paint_col_b=0xBEC2C8,
        primer_col=0x787C82, bare_metal_col=0xAEB4BA, crevice_col=0x383B40,
        paint_metallic=0.0, rough_min=0.42, rough_max=0.58, crevice_rough=0.68, bare_rough=0.26,
        coat_weight=0.08, coat_rough=0.34, peel_scale=460.0, peel_bump=0.03, chip_depth_bump=0.05,
        ao_dist=0.014, fine_bevel=0.00018, edge_bevel=0.0012, edge_gain=1.6, chip_amount=0.28,
        crevice_mix=0.36,
    )


    m_id19c = _build_elastomer_or_fabric_material(
        _get_or_new_mat("Mat_ID19c_Tire_MoldedRubberText"), _id_hex(MAT_3D_TEXT), 0x26292E,
        ng_global, ng_masks, ng_wrap,
        col_a=0x2A2D33, col_b=0x22252A, crevice_col=0x44423E,
        rough_min=0.66, rough_max=0.82, crevice_rough=0.90, crevice_mix=0.52,
        grain_scale=620.0, grain_bump=0.075, specular=0.30, ao_dist=0.012,
    )


    _build_metal_material(
        _base_mat(MAT_SHROUD_GOLD), _id_hex(MAT_SHROUD_GOLD), 0x94804C,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xB49958, col_b=0x786336, crevice_col=0x1D170E, wear_col=0xD0B97C,
        metallic=0.98, rough_min=0.17, rough_max=0.36, crevice_rough=0.54, wear_rough=0.14,
        brush_scale=(1.8, 175.0, 1.8), brush_bump=0.040, brush_rough_var=0.062,
        grain_scale=320.0, grain_bump=0.020,
        heat_base=0.32, heat_mottle_amp=0.22, heat_y0=-0.68, heat_y1=-0.06, heat_axial_amp=0.28,
        grazing_tint=0x78643E, grazing_mix=0.22, heat_tint_mix=0.36,
        soot_amp=0.16, soot_streak_scale=(16.0, 3.2, 16.0),
        ao_dist=0.044, fine_bevel=0.00032, edge_bevel=0.0038, edge_gain=1.9, edge_wear_mix=0.36,
        crevice_mix=0.64, grime_scale=11.0, macro_scale=2.8, island_var=0.04, aniso=0.46,
        scratch_amp=0.64, oil_leak_amp=0.26, shroud_3zone_thermal=True,
        oil_canning_bump=0.032, rivet_streaks=True, smudge_rough_amp=0.055,
        weld_haz_amp=0.48, swirl_polish_amp=0.60, aniso_rot_var=0.22, water_stain_amp=0.11,
        thin_film_amp=0.34, tf_nm_range=(8.0, 22.0), rough_floor=0.14, tangent_mode='RADIAL_Y',
        f82_tint=0xECE2C4, rough_breakup_amp=0.42,
    )


    _build_painted_metal_material(
        _base_mat(MAT_SHROUD_GREY), _id_hex(MAT_SHROUD_GREY), 0x4E5762,
        ng_global, ng_masks, ng_wrap,
        paint_col_a=0x525B66, paint_col_b=0x404852,
        primer_col=0x6C6E6A, bare_metal_col=0x9098A2, crevice_col=0x1B1E24,
        paint_metallic=0.0, rough_min=0.34, rough_max=0.56, crevice_rough=0.76, bare_rough=0.26,
        coat_weight=0.16, coat_rough=0.28,
        peel_scale=420.0, peel_bump=0.040, chip_depth_bump=0.11,
        ao_dist=0.038, fine_bevel=0.00030, edge_bevel=0.0038, edge_gain=1.65, chip_amount=0.46,
        crevice_mix=0.58, grime_scale=14.0, macro_scale=4.2,
        scuff_blotch_amp=0.34, scuff_scale=9.5,
        rust_amount=0.0, scratch_amp=0.42, oil_leak_amp=0.30,
        top_dust_fade_amp=0.22, grease_smudge_amp=0.54,
        spatter_amp=0.30, touchup_amp=0.32, clearcoat_swirl_amp=0.28, airflow_streaks_amp=0.52,
        steel_warp_bump=0.032,
    )


    _build_metal_material(
        _base_mat(MAT_NOZ_EXT_LINK), _id_hex(MAT_NOZ_EXT_LINK), 0x4C5460,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x626C7C, col_b=0x3E4552, crevice_col=0x101216, wear_col=0xACB6C2,
        metallic=0.97, rough_min=0.16, rough_max=0.40, crevice_rough=0.66, wear_rough=0.14,
        brush_scale=(26.0, 6.0, 26.0), brush_bump=0.044, brush_rough_var=0.064,
        grain_scale=340.0, grain_bump=0.032,
        heat_base=0.72, heat_mottle_amp=0.20, heat_y0=0.0, heat_y1=0.90, heat_axial_amp=0.20,
        grazing_tint=0x526278, grazing_mix=0.30, heat_tint_mix=0.36,
        soot_amp=0.34, soot_streak_scale=(24.0, 2.4, 24.0),
        ao_dist=0.042, fine_bevel=0.00032, edge_bevel=0.0034, edge_gain=2.3, edge_wear_mix=0.52,
        crevice_mix=0.78, macro_scale=4.5, island_var=0.16, aniso=0.36,
        scratch_amp=0.68, corrugation_ash_amp=0.46, smudge_rough_amp=0.056,
        oxide_scale_amp=0.72, swirl_polish_amp=0.38, aniso_rot_var=0.18, aniso_rot_base=0.25,
        thin_film_amp=0.28, tf_nm_range=(34.0, 50.0),
        exit_lip_soot_amp=0.58, nozzle_rub_arcs=True, rough_floor=0.14, tangent_mode='RADIAL_Y',
        f82_tint=0xDCE6F2, rough_breakup_amp=0.56,
    )


    m_id22b = _build_metal_material(
        _get_or_new_mat("Mat_ID22b_Nozzle_DivSealBlueMetal"), _id_hex(MAT_NOZ_EXT_LINK), 0x222E44,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x28364E, col_b=0x141C2B, crevice_col=0x090C12, wear_col=0x5A6A84,
        metallic=0.96, rough_min=0.20, rough_max=0.36, crevice_rough=0.68, wear_rough=0.16,
        brush_scale=(26.0, 6.0, 26.0), brush_bump=0.042, brush_rough_var=0.056,
        grain_scale=340.0, grain_bump=0.028,
        heat_base=0.78, heat_mottle_amp=0.22, heat_y0=0.0, heat_y1=0.90, heat_axial_amp=0.20,
        grazing_tint=0x2A2948, grazing_mix=0.34, heat_tint_mix=0.08,
        soot_amp=0.22, soot_streak_scale=(24.0, 2.4, 24.0),
        ao_dist=0.042, fine_bevel=0.00032, edge_bevel=0.0034, edge_gain=2.1, edge_wear_mix=0.30,
        crevice_mix=0.74, macro_scale=4.5, island_var=0.14, aniso=0.42,
        scratch_amp=0.40, corrugation_ash_amp=0.04, smudge_rough_amp=0.052,
        oxide_scale_amp=0.28, swirl_polish_amp=0.36, aniso_rot_var=0.18, aniso_rot_base=0.25,
        thin_film_amp=0.0,
        exit_lip_soot_amp=0.36, nozzle_rub_arcs=False, rough_floor=0.16, tangent_mode='RADIAL_Y',
        f82_tint=0xCCD6E6, rough_breakup_amp=0.48,
    )
    m_id22b.use_fake_user = True


    m_id22c = _build_metal_material(
        _get_or_new_mat("Mat_ID22c_Nozzle_SealTempered"), _id_hex(MAT_NOZ_EXT_LINK), 0x32507E,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x3A5E94, col_b=0x223A62, crevice_col=0x0C111A, wear_col=0x8AAEE0,
        metallic=0.97, rough_min=0.16, rough_max=0.36, crevice_rough=0.64, wear_rough=0.13,
        brush_scale=(26.0, 6.0, 26.0), brush_bump=0.044, brush_rough_var=0.062,
        grain_scale=340.0, grain_bump=0.030,
        heat_base=0.68, heat_mottle_amp=0.22, heat_y0=0.0, heat_y1=0.90, heat_axial_amp=0.22,
        grazing_tint=0x3E5A88, grazing_mix=0.24, heat_tint_mix=0.14,
        soot_amp=0.28, soot_streak_scale=(24.0, 2.4, 24.0),
        ao_dist=0.042, fine_bevel=0.00032, edge_bevel=0.0034, edge_gain=2.3, edge_wear_mix=0.52,
        crevice_mix=0.72, macro_scale=4.5, island_var=0.16, aniso=0.38,
        scratch_amp=0.62, corrugation_ash_amp=0.36, smudge_rough_amp=0.056,
        oxide_scale_amp=0.64, swirl_polish_amp=0.36, aniso_rot_var=0.18, aniso_rot_base=0.25,
        thin_film_amp=0.32, tf_nm_range=(36.0, 54.0),
        exit_lip_soot_amp=0.54, nozzle_rub_arcs=True, rough_floor=0.14, tangent_mode='RADIAL_Y',
        f82_tint=0xC8DCFA, rough_breakup_amp=0.50,
    )
    m_id22c.use_fake_user = True


    _build_metal_material(
        _base_mat(MAT_NOZ_FLAP_BODY), _id_hex(MAT_NOZ_FLAP_BODY), 0x666460,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x7E7B74, col_b=0x52504C, crevice_col=0x111113, wear_col=0xC2C6CC,
        metallic=0.97, rough_min=0.16, rough_max=0.40, crevice_rough=0.64, wear_rough=0.14,
        brush_scale=(22.0, 5.0, 22.0), brush_bump=0.042, brush_rough_var=0.062,
        grain_scale=340.0, grain_bump=0.030,
        heat_base=0.48, heat_mottle_amp=0.22, heat_y0=0.0, heat_y1=0.92, heat_axial_amp=0.28,
        grazing_tint=0x6A6156, grazing_mix=0.24, heat_tint_mix=0.36,
        soot_amp=0.34, soot_streak_scale=(22.0, 2.2, 22.0),
        ao_dist=0.045, fine_bevel=0.00034, edge_bevel=0.0036, edge_gain=2.4, edge_wear_mix=0.52,
        crevice_mix=0.76, macro_scale=4.2, island_var=0.16, aniso=0.38,
        scratch_amp=0.68, corrugation_ash_amp=0.32, smudge_rough_amp=0.058,
        oxide_scale_amp=0.65, swirl_polish_amp=0.38, aniso_rot_var=0.18, aniso_rot_base=0.25,
        thin_film_amp=0.26, tf_nm_range=(12.0, 26.0),
        exit_lip_soot_amp=0.54, nozzle_rub_arcs=True, rough_floor=0.14, tangent_mode='RADIAL_Y',
        f82_tint=0xE6DFD6, rough_breakup_amp=0.56, nozzle_nickel_antiseize=0.68,
    )


    _build_metal_material(
        _base_mat(MAT_NOZ_ACT_AB), _id_hex(MAT_NOZ_ACT_AB), 0xB2B9C0,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xC0C6CC, col_b=0x9098A0, crevice_col=0x1E1C1A, wear_col=0xD4DAE0,
        metallic=0.97, rough_min=0.13, rough_max=0.34, crevice_rough=0.46, wear_rough=0.10,
        brush_scale=(3.0, 220.0, 3.0), brush_bump=0.032, brush_rough_var=0.056,
        grain_scale=480.0, grain_bump=0.030,
        ao_dist=0.038, fine_bevel=0.00030, edge_bevel=0.0030, edge_gain=2.1, edge_wear_mix=0.46,
        crevice_mix=0.60, macro_scale=5.0, island_var=0.09, aniso=0.05,
        scratch_amp=0.44, oil_leak_amp=0.70,
        lathe_wave_scale=240.0, lathe_wave_bump=0.018, smudge_rough_amp=0.055,
        swirl_polish_amp=0.55, water_stain_amp=0.14, rough_floor=0.10, tangent_mode='NONE',
        f82_tint=0xEAEFF5, rough_breakup_amp=0.76,
    )


    m_id24b = _build_metal_material(
        _get_or_new_mat("Mat_ID24b_Augmentor_SootedLiner"), _id_hex(MAT_NOZ_ACT_AB), 0x3D3A36,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x544E46, col_b=0x302D28, crevice_col=0x0E0E10, wear_col=0x7C776E,
        metallic=0.84, rough_min=0.32, rough_max=0.68, crevice_rough=0.88, wear_rough=0.22,
        brush_scale=(14.0, 2.2, 14.0), brush_bump=0.032, brush_rough_var=0.072,
        grain_scale=380.0, grain_bump=0.055,
        heat_base=0.48, heat_mottle_amp=0.22, heat_y0=-1.85, heat_y1=0.0, heat_axial_amp=0.26,
        grazing_tint=0x564F45, grazing_mix=0.24, heat_tint_mix=0.32,
        soot_amp=0.58, soot_streak_scale=(10.0, 1.4, 10.0),
        ao_dist=0.058, fine_bevel=0.00030, edge_bevel=0.0028, edge_gain=1.6, edge_wear_mix=0.22,
        crevice_mix=0.82, macro_scale=2.6, island_var=0.06, aniso=0.28,
        scratch_amp=0.34, lathe_wave_scale=240.0, lathe_wave_bump=0.014,
        corrugation_ash_amp=0.28, oxide_scale_amp=0.52, effusion_holes=True,
        combustor_axial_soot=True, rough_floor=0.22, tangent_mode='RADIAL_Y',
        f82_tint=0xE2DCD4, rough_breakup_amp=0.78,
    )


    _build_ceramic_soot_material(
        _base_mat(MAT_HOTGAS_TILE), _id_hex(MAT_HOTGAS_TILE), 0x8E867A,
        ng_global, ng_masks, ng_wrap,
        ceramic_a=0x9A9082, ceramic_b=0x6C655A, soot_mid=0x44413E, soot_dark=0x1A1A1C,
        metallic_base=0.0, rough_ceramic=0.88, rough_soot=0.95,
        streak_scale_lo=(6.5, 1.4, 6.5), streak_scale_hi=(11.5, 2.8, 11.5),
        y_grad_start=0.44, y_grad_end=0.92, y_grad_gain=0.70,
        crevice_soot_gain=0.72, base_soot_bias=0.42,
        spall_amount=1.05,
    )


    _build_painted_metal_material(
        _base_mat(MAT_STAND_ORANGE), _id_hex(MAT_STAND_ORANGE), 0xB83C12,
        ng_global, ng_masks, ng_wrap,
        paint_col_a=0xBC3E12, paint_col_b=0x8A2A0C,
        primer_col=0x726E3C, bare_metal_col=0xBCC4CC, crevice_col=0x260E04,
        paint_metallic=0.0, rough_min=0.27, rough_max=0.40, crevice_rough=0.62, bare_rough=0.22,
        coat_weight=0.24, coat_rough=0.18,
        peel_scale=260.0, peel_bump=0.048, chip_depth_bump=0.20,
        ao_dist=0.075, fine_bevel=0.00040, edge_bevel=0.0045, edge_gain=2.1, chip_amount=0.66,
        crevice_mix=0.68, grime_scale=9.0, macro_scale=2.4,
        scuff_blotch_amp=0.08, scuff_scale=9.5,
        rust_amount=1.05, rust_col_a=0x722F10, rust_col_b=0x331206,
        scratch_amp=0.50, oil_leak_amp=0.62,
        top_dust_fade_amp=0.14, grease_smudge_amp=0.40,
        blister_amp=0.82, spatter_amp=0.44, touchup_amp=0.42, clearcoat_swirl_amp=0.65,
        chip_thresh=0.30, steel_warp_bump=0.048,
    )


    m_id26b = _build_painted_metal_material(
        _get_or_new_mat("Mat_ID26b_Engine_HoistPointRed"), _id_hex(MAT_STAND_ORANGE), 0xB22618,
        ng_global, ng_masks, ng_wrap,
        paint_col_a=0xC02C1C, paint_col_b=0x8E1E12,
        primer_col=0x6A644E, bare_metal_col=0xA2AAB2, crevice_col=0x220B07,
        paint_metallic=0.0, rough_min=0.28, rough_max=0.46, crevice_rough=0.62, bare_rough=0.16,
        coat_weight=0.28, coat_rough=0.22,
        peel_scale=380.0, peel_bump=0.045, chip_depth_bump=0.16,
        ao_dist=0.036, fine_bevel=0.00032, edge_bevel=0.0038, edge_gain=2.8, chip_amount=0.92,
        crevice_mix=0.60, grime_scale=16.0, macro_scale=5.0,
        rust_amount=0.42, scratch_amp=0.78, oil_leak_amp=0.26,
        blister_amp=0.58, spatter_amp=0.32, clearcoat_swirl_amp=0.50,
    )


    _build_metal_material(
        _base_mat(MAT_STAND_STEEL), _id_hex(MAT_STAND_STEEL), 0x98A0A8,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x9CA4AE, col_b=0x767E88, crevice_col=0x161719, wear_col=0xD4DAE0,
        metallic=0.99, rough_min=0.16, rough_max=0.34, crevice_rough=0.62, wear_rough=0.11,
        brush_scale=(48.0, 2.0, 48.0), brush_bump=0.036, brush_rough_var=0.068,
        grain_scale=340.0, grain_bump=0.022,
        cast_voronoi_scale=165.0, cast_voronoi_bump=0.012,
        oil_canning_bump=0.030,
        ao_dist=0.068, fine_bevel=0.00042, edge_bevel=0.0065, edge_gain=2.8, edge_wear_mix=0.88,
        crevice_mix=0.72, grime_scale=22.0, macro_scale=1.8, island_var=0.10, aniso=0.06,
        scratch_amp=0.66, oil_leak_amp=0.44, rust_amount=1.05, smudge_rough_amp=0.056,
        spangle_shimmer=0.45, swirl_polish_amp=0.62, water_stain_amp=0.22,
        rough_floor=0.11, tangent_mode='NONE',
        f82_tint=0xE2E8F0, rough_breakup_amp=0.46,
    )


    m_id27c = _build_metal_material(
        _get_or_new_mat("Mat_ID27c_Stand_MachinedSteelHardware"), _id_hex(MAT_STAND_STEEL), 0xA8B1BA,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xAEB6C0, col_b=0x828A94, crevice_col=0x141312, wear_col=0xD6DCE2,
        metallic=0.99, rough_min=0.12, rough_max=0.28, crevice_rough=0.48, wear_rough=0.10,
        brush_scale=(28.0, 28.0, 28.0), brush_bump=0.032, brush_rough_var=0.056,
        grain_scale=420.0, grain_bump=0.016,
        oil_canning_bump=0.014,
        ao_dist=0.036, fine_bevel=0.00034, edge_bevel=0.0038, edge_gain=2.7, edge_wear_mix=0.76,
        crevice_mix=0.60, grime_scale=18.0, macro_scale=3.2, island_var=0.10, aniso=0.24,
        scratch_amp=0.72, oil_leak_amp=0.44, rust_amount=0.42, smudge_rough_amp=0.048,
        lathe_wave_scale=0.0, lathe_wave_bump=0.0, swirl_polish_amp=0.60, aniso_rot_var=0.24,
        rough_floor=0.10, tangent_mode='RADIAL_X',
        f82_tint=0xE4EAEF, rough_breakup_amp=0.42,
    )


    m_id27e = _build_metal_material(
        _get_or_new_mat("Mat_ID27e_Wheel_Hubcap_Lathe"), _id_hex(MAT_STAND_STEEL), 0xA8B1BA,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0xAEB6C0, col_b=0x828A94, crevice_col=0x141312, wear_col=0xD6DCE2,
        metallic=0.99, rough_min=0.12, rough_max=0.26, crevice_rough=0.50, wear_rough=0.10,
        brush_scale=(32.0, 32.0, 32.0), brush_bump=0.028, brush_rough_var=0.052,
        grain_scale=420.0, grain_bump=0.014,
        ao_dist=0.032, fine_bevel=0.00032, edge_bevel=0.0034, edge_gain=2.5, edge_wear_mix=0.72,
        crevice_mix=0.62, grime_scale=20.0, macro_scale=3.6, island_var=0.08, aniso=0.46,
        scratch_amp=0.58, oil_leak_amp=0.48, rust_amount=0.35, smudge_rough_amp=0.042,
        lathe_wave_scale=150.0, lathe_wave_bump=0.032, wheel_hubcap_lathe=True,
        swirl_polish_amp=0.42, aniso_rot_var=0.12,
        rough_floor=0.10, tangent_mode='WHEEL_AXLE_X',
        f82_tint=0xE4EAEF, rough_breakup_amp=0.40,
    )


    m_id27b = _build_metal_material(
        _get_or_new_mat("Mat_ID27b_Nozzle_HardenedRollers"), _id_hex(MAT_STAND_STEEL), 0x626870,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x787E86, col_b=0x4A5058, crevice_col=0x121315, wear_col=0xD2D8DF,
        metallic=0.98, rough_min=0.10, rough_max=0.30, crevice_rough=0.54, wear_rough=0.09,
        brush_scale=(120.0, 120.0, 120.0), brush_bump=0.024, brush_rough_var=0.052,
        heat_base=0.26, heat_mottle_amp=0.14, grazing_tint=0x666C76, grazing_mix=0.16, heat_tint_mix=0.20,
        soot_amp=0.16, ao_dist=0.018, fine_bevel=0.00020, edge_bevel=0.0016, edge_gain=2.3,
        edge_wear_mix=0.58, crevice_mix=0.68, island_var=0.14,
        scratch_amp=0.54, oxide_scale_amp=0.32, swirl_polish_amp=0.30, rough_floor=0.09,
        f82_tint=0xE4EAEF, rough_breakup_amp=0.78,
    )


    _build_painted_metal_material(
        _base_mat(MAT_TRAILER_YELLOW), _id_hex(MAT_TRAILER_YELLOW), 0xB6740C,
        ng_global, ng_masks, ng_wrap,
        paint_col_a=0xB6740C, paint_col_b=0x784906,
        primer_col=0x585C34, bare_metal_col=0xC6CED6, crevice_col=0x140E06,
        paint_metallic=0.0, rough_min=0.20, rough_max=0.40, crevice_rough=0.80, bare_rough=0.14,
        coat_weight=0.36, coat_rough=0.16,
        peel_scale=125.0, peel_bump=0.105, chip_depth_bump=0.34,
        ao_dist=0.095, fine_bevel=0.00042, edge_bevel=0.0055, edge_gain=3.4, chip_amount=1.05,
        crevice_mix=0.82, grime_scale=7.5, macro_scale=2.0,
        low_z_dust_amp=0.44, low_z_min=-1.60, low_z_max=-0.96, dust_col=0x34312C,
        scuff_blotch_amp=0.30, scuff_scale=9.0,
        rust_amount=1.35, rust_col_a=0x6E2C0E, rust_col_b=0x2D1005,
        scratch_amp=0.54, oil_leak_amp=0.78,
        top_dust_fade_amp=0.22, grease_smudge_amp=0.55,
        blister_amp=0.95, spatter_amp=0.72, touchup_amp=0.52, clearcoat_swirl_amp=0.42,
        chip_thresh=0.25, steel_warp_bump=0.055, smudge_coat_scale=0.42,
    )


    m_id28b = _build_painted_metal_material(
        _get_or_new_mat("Mat_ID28b_Trailer_UndercarriageBlack"), _id_hex(MAT_TRAILER_YELLOW), 0x22262D,
        ng_global, ng_masks, ng_wrap,
        paint_col_a=0x262B32, paint_col_b=0x1D2127,
        primer_col=0x46453C, bare_metal_col=0xC2CAD2, crevice_col=0x0E1014,
        paint_metallic=0.0, rough_min=0.22, rough_max=0.40, crevice_rough=0.80, bare_rough=0.14,
        coat_weight=0.28, coat_rough=0.19,
        peel_scale=115.0, peel_bump=0.110, chip_depth_bump=0.32,
        ao_dist=0.088, fine_bevel=0.00040, edge_bevel=0.0052, edge_gain=3.2, chip_amount=1.00,
        crevice_mix=0.82, grime_scale=8.5, macro_scale=2.4,
        low_z_dust_amp=0.38, low_z_min=-1.60, low_z_max=-1.05, dust_col=0x302D28,
        scuff_blotch_amp=0.22, scuff_scale=10.0,
        rust_amount=1.18, rust_col_a=0x6A2A0E, rust_col_b=0x2C1005,
        scratch_amp=0.46, oil_leak_amp=0.85,
        top_dust_fade_amp=0.16, grease_smudge_amp=0.58,
        blister_amp=0.78, spatter_amp=0.68, touchup_amp=0.35, clearcoat_swirl_amp=0.38,
        chip_thresh=0.28, steel_warp_bump=0.052, smudge_coat_scale=0.42,
    )


    m_id27d = _build_metal_material(
        _get_or_new_mat("Mat_ID27d_Trailer_ForgedSpringSteel"), _id_hex(MAT_STAND_STEEL), 0x505660,
        ng_global, ng_masks, ng_heat, ng_wrap,
        col_a=0x5E6570, col_b=0x424852, crevice_col=0x101012, wear_col=0xC4CCD4,
        metallic=0.98, rough_min=0.16, rough_max=0.46, crevice_rough=0.76, wear_rough=0.12,
        brush_scale=(55.0, 12.0, 55.0), brush_bump=0.036, brush_rough_var=0.078,
        grain_scale=340.0, grain_bump=0.028,
        cast_voronoi_scale=220.0, cast_voronoi_bump=0.032,
        oil_canning_bump=0.026,
        ao_dist=0.056, fine_bevel=0.00038, edge_bevel=0.0044, edge_gain=2.9, edge_wear_mix=0.84,
        crevice_mix=0.84, grime_scale=16.0, macro_scale=2.6, island_var=0.12, aniso=0.05,
        scratch_amp=0.84, oil_leak_amp=0.74, rust_amount=1.20, smudge_rough_amp=0.062,
        oxide_scale_amp=0.32, swirl_polish_amp=0.38,
        rough_floor=0.12, tangent_mode='NONE',
        f82_tint=0xDCE2EA, rough_breakup_amp=0.58,
    )


    _build_elastomer_or_fabric_material(
        _base_mat(MAT_RUBBER_DARK), _id_hex(MAT_RUBBER_DARK), 0x24272C,
        ng_global, ng_masks, ng_wrap,
        col_a=0x282B30, col_b=0x202328, crevice_col=0x5C5952,
        rough_min=0.58, rough_max=0.76, crevice_rough=0.95, crevice_mix=0.72,
        grain_scale=640.0, grain_bump=0.085, specular=0.30, ao_dist=0.036,
        tire_tread_wear=True,
    )


    m_id29b = _build_elastomer_or_fabric_material(
        _get_or_new_mat("Mat_ID29b_Shroud_RedSiliconeSeal"), _id_hex(MAT_RUBBER_DARK), 0xB8361B,
        ng_global, ng_masks, ng_wrap,
        col_a=0xC63B1E, col_b=0x9A2B14, crevice_col=0x38120A,
        rough_min=0.44, rough_max=0.62, crevice_rough=0.72, crevice_mix=0.45,
        grain_scale=460.0, grain_bump=0.05, rubber_bloom_amp=0.22, sss_weight=0.06, specular=0.42, ao_dist=0.018,
    )


    m_glow = _get_or_new_mat("Mat_Clay_AB_Accent")
    ntg = m_glow.node_tree
    ntg.nodes.clear()
    tc_g = _new_node(ntg, "ShaderNodeTexCoord", (-1280, 0))
    masks_g = _new_node(ntg, "ShaderNodeGroup", (-960, 140))
    masks_g.node_tree = ng_masks
    ntg.links.new(tc_g.outputs["Object"], masks_g.inputs["Coord"])
    masks_g.inputs["AO_Dist"].default_value = 0.045
    masks_g.inputs["Fine_Bevel_Rad"].default_value = 0.00028
    masks_g.inputs["Micro_Scale"].default_value = 420.0

    col_g0 = _mix_rgba(ntg, 'MIX', (-520, 260), col_a=rgba(0x403A33), col_b=rgba(0x1B1917))
    ntg.links.new(masks_g.outputs["Macro_Mottle"], col_g0.inputs[0])
    col_g1 = _mix_rgba(ntg, 'MIX', (-300, 260), col_b=rgba(0x0E0D0F))
    ntg.links.new(masks_g.outputs["Crevice_Mask"], col_g1.inputs[0])
    ntg.links.new(col_g0.outputs[2], col_g1.inputs[6])

    r_g_mix = _mix_float(ntg, (-720, 20), fac=0.70)
    ntg.links.new(masks_g.outputs["Macro_Mottle"], r_g_mix.inputs[2])
    ntg.links.new(masks_g.outputs["Metal_Rough_Breakup"], r_g_mix.inputs[3])
    r_g = _map_range(ntg, 0.14, 0.86, 0.30, 0.68, (-520, 20), clamp=True)
    ntg.links.new(r_g_mix.outputs[0], r_g.inputs[0])

    b_g = _new_node(ntg, "ShaderNodeBump", (-300, -160))
    b_g.inputs["Strength"].default_value = 0.068
    b_g.inputs["Distance"].default_value = 0.0004
    ntg.links.new(masks_g.outputs["Micro_Grain"], b_g.inputs["Height"])
    ntg.links.new(masks_g.outputs["Bevel_Normal"], b_g.inputs["Normal"])


    n_em_coarse = _noise(ntg, scale=4.2, detail=4.5, roughness=0.62, distortion=1.35, loc=(-1040, -280))
    ntg.links.new(tc_g.outputs["Object"], n_em_coarse.inputs["Vector"])
    n_em_fine = _noise(ntg, scale=19.0, detail=6.0, roughness=0.68, distortion=0.85, loc=(-1040, -440))
    ntg.links.new(tc_g.outputs["Object"], n_em_fine.inputs["Vector"])
    em_mul = _math(ntg, 'MULTIPLY', (-820, -340), clamp=True)
    ntg.links.new(n_em_coarse.outputs["Factor"], em_mul.inputs[0])
    ntg.links.new(n_em_fine.outputs["Factor"], em_mul.inputs[1])
    em_pkt = _map_range(ntg, 0.14, 0.44, 0.0, 1.0, (-640, -340), clamp=True)
    ntg.links.new(em_mul.outputs[0], em_pkt.inputs[0])
    ramp_em = _ramp(ntg, [
        (0.00, rgba(0x0B0200)),
        (0.32, rgba(0x4A0B02)),
        (0.62, rgba(0xD2480A)),
        (0.84, rgba(0xFF982A)),
        (1.00, rgba(0xFFD278)),
    ], loc=(-440, -340))
    ntg.links.new(em_pkt.outputs[0], ramp_em.inputs["Fac"])


    em_gain = _map_range(ntg, 0.0, 1.0, 0.12, 1.85, (-440, -520), clamp=True)
    ntg.links.new(em_pkt.outputs[0], em_gain.inputs[0])
    em_scaled = _vmath(ntg, 'SCALE', (-180, -380))
    ntg.links.new(ramp_em.outputs["Color"], em_scaled.inputs[0])
    ntg.links.new(em_gain.outputs[0], em_scaled.inputs["Scale"])

    out_g = _new_node(ntg, "ShaderNodeOutputMaterial", (740, 0))
    bsdf_g = _new_node(ntg, "ShaderNodeBsdfPrincipled", (160, 80), name="AB_Glow_BSDF")
    ntg.links.new(col_g1.outputs[2], bsdf_g.inputs["Base Color"])
    bsdf_g.inputs["Metallic"].default_value = 0.86
    ntg.links.new(r_g.outputs[0], bsdf_g.inputs["Roughness"])
    if "Specular Tint" in bsdf_g.inputs:
        bsdf_g.inputs["Specular Tint"].default_value = rgba(0xE2D8C8)
    ntg.links.new(b_g.outputs["Normal"], bsdf_g.inputs["Normal"])
    ntg.links.new(em_scaled.outputs["Vector"], bsdf_g.inputs["Emission Color"])
    bsdf_g.inputs["Emission Strength"].default_value = 0.0

    wrap_g = _new_node(ntg, "ShaderNodeGroup", (480, 60))
    wrap_g.node_tree = ng_wrap
    ntg.links.new(bsdf_g.outputs["BSDF"], wrap_g.inputs["PBR_Shader"])
    wrap_g.inputs["ID_Color"].default_value = rgba(0xFFE066)
    wrap_g.inputs["Clay_Color"].default_value = rgba(0xD8DCE2)
    ntg.links.new(masks_g.outputs["AO_Raw"], wrap_g.inputs["AO_Raw"])
    ntg.links.new(masks_g.outputs["Bevel_Normal"], wrap_g.inputs["Bevel_Normal"])
    ntg.links.new(wrap_g.outputs["Shader"], out_g.inputs["Surface"])
    m_glow["f15_id_hex"] = int(0xFFE066)
    m_glow["f15_pbr_hex"] = int(0x3D3935)
    m_glow.diffuse_color = rgba(0x3D3935)


    reassigned = 0

    for old_n, new_m in [
        (COLOR_ID_SPECS[MAT_RUBBER_DARK][0], m_id29b),
        (COLOR_ID_SPECS[MAT_PLATE_BASE][0],  m_id18b),
        (COLOR_ID_SPECS[MAT_3D_TEXT][0],     m_id19b),
    ]:
        if _swap_mesh_mat_slot("Static_Nozzle_Outer_Shroud", old_n, new_m):
            reassigned += 1


    if _swap_mesh_mat_slot("Mesh_DivMaster", COLOR_ID_SPECS[MAT_AVIONICS_BOX][0], m_id07b):
        reassigned += 1
    for seal_mesh in ("Mesh_DivSeal", "Mesh_ConvSeal"):
        for old_seal_mat in (COLOR_ID_SPECS[MAT_NOZ_EXT_LINK][0], "Mat_ID22c_Nozzle_SealTempered"):
            if _swap_mesh_mat_slot(seal_mesh, old_seal_mat, m_id22b):
                reassigned += 1


    nozzle_moving_meshes = [
        "Mesh_DivMaster", "Mesh_ConvMaster", "Mesh_DivSeal", "Mesh_ConvSeal",
        "Mesh_BalanceFlap", "Mesh_BalanceFlap_Stbd",
        "Mesh_Diagonal_Strut", "Mesh_Diagonal_Strut_Stbd",
        "Mesh_TwinFork_PullRod",
        "Mesh_Bellcrank", "Mesh_Boomerang", "Mesh_ShortLink", "Mesh_SliderCarriage",
        "Mesh_RingKeeperSaddle", "Mesh_RingStrap_L", "Mesh_RingStrap_R",
        "Mesh_BoomGuideCyl", "Mesh_BoomFwdRod", "Mesh_SyncRing_Pistons",
        "Mesh_TransRodA", "Mesh_TransRodB",
    ]
    for mn in nozzle_moving_meshes:
        for old_n, new_m in [
            (COLOR_ID_SPECS[MAT_MACHINED_ALU][0],  m_id06b),
            (COLOR_ID_SPECS[MAT_RIGID_TUBE][0],    m_id09b),
            (COLOR_ID_SPECS[MAT_BRASS_FITTING][0], m_id12b),
            (COLOR_ID_SPECS[MAT_PLATE_BASE][0],    m_id18c),
            (COLOR_ID_SPECS[MAT_3D_TEXT][0],       m_id19d),
        ]:
            if _swap_mesh_mat_slot(mn, old_n, new_m):
                reassigned += 1


    for mn in ("Mesh_BalanceRollerPair", "Mesh_SliderRollerPair", "Mesh_SyncRoller"):
        if _swap_mesh_mat_slot(mn, COLOR_ID_SPECS[MAT_STAND_STEEL][0], m_id27b):
            reassigned += 1


    if _swap_mesh_mat_slot("Interior_Louvered_Corrugated_Liner", COLOR_ID_SPECS[MAT_NOZ_ACT_AB][0], m_id24b):
        reassigned += 1
    for mn in ("Interior_Louvered_Corrugated_Liner", "Interior_Stator_TailCone_And_GuideVanes", "Interior_VGutter_Flameholders_Detailed"):
        if _swap_mesh_mat_slot(mn, COLOR_ID_SPECS[MAT_FASTENER][0], m_id15b):
            reassigned += 1
    for mn in ("Interior_Stator_TailCone_And_GuideVanes", "Interior_VGutter_Flameholders_Detailed"):
        if _swap_mesh_mat_slot(mn, COLOR_ID_SPECS[MAT_INCONEL_CASE][0], m_id02b):
            reassigned += 1
    for old_n, new_m in [
        (COLOR_ID_SPECS[MAT_SPINNER_IGV][0], m_id04b),
        (COLOR_ID_SPECS[MAT_RIGID_TUBE][0],  m_id09b),
    ]:
        if _swap_mesh_mat_slot("Interior_VGutter_Flameholders_Detailed", old_n, new_m):
            reassigned += 1


    if _swap_mesh_mat_slot("Mesh_LPT_Rotor_Shaft", COLOR_ID_SPECS[MAT_ROTARY_BLADES][0], m_id05b):
        reassigned += 1
    if _swap_mesh_mat_slot("Mesh_LPT_Rotor_Shaft", COLOR_ID_SPECS[MAT_BRASS_FITTING][0], m_id05c):
        reassigned += 1
    if _swap_mesh_mat_slot("Mesh_LPT_Rotor_Shaft", COLOR_ID_SPECS[MAT_FASTENER][0], m_id15b):
        reassigned += 1


    if _swap_mesh_mat_slot("Static_Maintenance_Hoists_And_DataPlate", COLOR_ID_SPECS[MAT_STAND_ORANGE][0], m_id26b):
        reassigned += 1


    if _swap_mesh_mat_slot("Stand_Trailer_06_Pneumatic_GroundSupport_Wheels", COLOR_ID_SPECS[MAT_3D_TEXT][0], m_id19c):
        reassigned += 1
    for mn in (
        "Stand_Cradle_02_Pedestals_Rollers_And_Handwheels",
        "Stand_Cradle_03_RollOver_Arches_And_MountPickups",
        "Stand_Cradle_04_TopTieBar_And_DiagonalBraces",
        "Stand_Trailer_05_M3000_Steerable_Axles_And_TowBar",
    ):
        if _swap_mesh_mat_slot(mn, COLOR_ID_SPECS[MAT_STAND_STEEL][0], m_id27c):
            reassigned += 1

    if _swap_mesh_mat_slot("Stand_Trailer_06_Pneumatic_GroundSupport_Wheels", COLOR_ID_SPECS[MAT_STAND_STEEL][0], m_id27e):
        reassigned += 1

    if _swap_mesh_mat_slot("Stand_Cradle_01_Longitudinal_Rails_And_Crossbeams", COLOR_ID_SPECS[MAT_MACHINED_ALU][0], m_id27c):
        reassigned += 1

    for mn_ind in (
        "Stand_Cradle_01_Longitudinal_Rails_And_Crossbeams",
        "Stand_Cradle_02_Pedestals_Rollers_And_Handwheels",
        "Stand_Cradle_03_RollOver_Arches_And_MountPickups",
        "Stand_Cradle_04_TopTieBar_And_DiagonalBraces",
        "Stand_Trailer_05_M3000_Steerable_Axles_And_TowBar",
    ):
        for old_n, new_m in [
            (COLOR_ID_SPECS[MAT_AVIONICS_BOX][0],  m_id28b),
            (COLOR_ID_SPECS[MAT_PUMPS_OILTANK][0], m_id27d),
        ]:
            if _swap_mesh_mat_slot(mn_ind, old_n, new_m):
                reassigned += 1
    for mn in (
        "Stand_Cradle_01_Longitudinal_Rails_And_Crossbeams",
        "Stand_Cradle_03_RollOver_Arches_And_MountPickups",
        "Stand_Trailer_05_M3000_Steerable_Axles_And_TowBar",
    ):
        for old_n, new_m in [
            (COLOR_ID_SPECS[MAT_PLATE_BASE][0], m_id18d),
            (COLOR_ID_SPECS[MAT_3D_TEXT][0],    m_id19e),
        ]:
            if _swap_mesh_mat_slot(mn, old_n, new_m):
                reassigned += 1

    print(f"[Materials] Built procedural PBR material library (reassigned {reassigned} mesh slots)")
    return ng_global, ng_masks, ng_wrap


def _load_and_pack_hdri(filepath: str, img_name: str):
    if not os.path.exists(filepath):
        print(f"[Warning] HDRI file not found: {filepath}")
        return None
    img = bpy.data.images.get(img_name)
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.load(filepath, check_existing=False)
    img.name = img_name
    try:
        img.pack()
    except Exception as e:
        print(f"[Warning] Failed to pack HDRI ({img_name}): {e}")
    return img


def _create_studio_scrim_light(
    coll, name, loc, target, size_x, size_y, energy, col_rgb,
    spread_deg=78.0, shape='RECTANGLE', up_axis='Z', vis_glossy=True, hangar_ratio=0.0,
):

    old = bpy.data.objects.get(name)
    if old is not None:
        ld_old = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if ld_old and ld_old.users == 0:
            bpy.data.lights.remove(ld_old)

    la = bpy.data.lights.new(name, type='AREA')
    la.shape = shape
    la.size = size_x
    if shape == 'RECTANGLE':
        la.size_y = size_y
    la.energy = float(energy)
    la.color = tuple(col_rgb)
    if hasattr(la, "spread"):
        la.spread = math.radians(spread_deg)


    la.use_nodes = True
    lnt = la.node_tree
    lnt.nodes.clear()
    geo = _new_node(lnt, "ShaderNodeNewGeometry", (-960, 0))
    sep = _new_node(lnt, "ShaderNodeSeparateXYZ", (-760, 0))
    lnt.links.new(geo.outputs["Parametric"], sep.inputs["Vector"])


    u_c = _math(lnt, 'MULTIPLY_ADD', (-560, 120), v1=2.0, v2=-1.0)
    lnt.links.new(sep.outputs["X"], u_c.inputs[0])
    u_sq = _math(lnt, 'MULTIPLY', (-380, 120))
    lnt.links.new(u_c.outputs[0], u_sq.inputs[0])
    lnt.links.new(u_c.outputs[0], u_sq.inputs[1])

    v_c = _math(lnt, 'MULTIPLY_ADD', (-560, -80), v1=2.0, v2=-1.0)
    lnt.links.new(sep.outputs["Y"], v_c.inputs[0])
    v_sq = _math(lnt, 'MULTIPLY', (-380, -80))
    lnt.links.new(v_c.outputs[0], v_sq.inputs[0])
    lnt.links.new(v_c.outputs[0], v_sq.inputs[1])

    r_sq = _math(lnt, 'ADD', (-200, 20))
    lnt.links.new(u_sq.outputs[0], r_sq.inputs[0])
    lnt.links.new(v_sq.outputs[0], r_sq.inputs[1])
    r_val = _math(lnt, 'SQRT', (-40, 20))
    lnt.links.new(r_sq.outputs[0], r_val.inputs[0])


    fall = _map_range(lnt, 0.22, 1.04, 1.0, 0.04, (120, 20), clamp=True, interp='SMOOTHERSTEP')
    lnt.links.new(r_val.outputs[0], fall.inputs[0])
    em_str = _math(lnt, 'MULTIPLY', (300, 20), v1=1.25)
    lnt.links.new(fall.outputs[0], em_str.inputs[0])

    em = _new_node(lnt, "ShaderNodeEmission", (480, 20))
    em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    lnt.links.new(em_str.outputs[0], em.inputs["Strength"])

    lout = _new_node(lnt, "ShaderNodeOutputLight", (680, 20))
    lnt.links.new(em.outputs["Emission"], lout.inputs["Surface"])

    la["f15_studio_energy"] = float(energy)
    la["f15_hangar_ratio"] = float(hangar_ratio)

    lo = bpy.data.objects.new(name, la)
    lo.location = Vector(loc)

    fwd = (Vector(target) - Vector(loc)).normalized()
    z_ax = -fwd
    ref_up = Vector((0.0, 1.0, 0.0)) if up_axis == 'Y' else Vector((0.0, 0.0, 1.0))
    if abs(z_ax.dot(ref_up)) > 0.999:
        ref_up = Vector((1.0, 0.0, 0.0))
    y_ax = (ref_up - z_ax * ref_up.dot(z_ax)).normalized()
    x_ax = y_ax.cross(z_ax).normalized()
    rot_mat = Matrix((x_ax, y_ax, z_ax)).transposed()
    lo.rotation_euler = rot_mat.to_euler('XYZ')
    if hasattr(lo, "visible_camera"):
        lo.visible_camera = False
    if hasattr(lo, "visible_glossy"):
        lo.visible_glossy = bool(vis_glossy)
    coll.objects.link(lo)
    return lo


def setup_studio_compositor(scn):

    scn.render.use_compositing = True
    try:
        if hasattr(scn, "compositing_node_group"):
            old_ng = bpy.data.node_groups.get("Comp_F15_StudioOptics")
            if old_ng is not None:
                bpy.data.node_groups.remove(old_ng, do_unlink=True)
            ng = bpy.data.node_groups.new("Comp_F15_StudioOptics", 'CompositorNodeTree')
            ng.interface.new_socket(name="Image", in_out='OUTPUT', socket_type='NodeSocketColor')
            rl = ng.nodes.new("CompositorNodeRLayers")
            rl.location = (-360, 0)
            gl = ng.nodes.new("CompositorNodeGlare")
            gl.location = (0, 0)

            if "Type" in gl.inputs:
                for cand in ("Fog Glow", "Bloom", "FOG_GLOW", "BLOOM"):
                    try:
                        gl.inputs["Type"].default_value = cand
                        break
                    except Exception:
                        continue
            if "Quality" in gl.inputs:
                for cand in ("High", "HIGH"):
                    try:
                        gl.inputs["Quality"].default_value = cand
                        break
                    except Exception:
                        continue
            if "Threshold" in gl.inputs:
                gl.inputs["Threshold"].default_value = 1.90
            if "Clamp" in gl.inputs:
                gl.inputs["Clamp"].default_value = True
            if "Maximum" in gl.inputs:
                gl.inputs["Maximum"].default_value = 10.0
            if "Smoothness" in gl.inputs:
                gl.inputs["Smoothness"].default_value = 0.30
            if "Strength" in gl.inputs:
                gl.inputs["Strength"].default_value = 0.045
            if "Size" in gl.inputs:
                gl.inputs["Size"].default_value = 0.48
            if "Saturation" in gl.inputs:
                gl.inputs["Saturation"].default_value = 0.85
            out_n = ng.nodes.new("NodeGroupOutput")
            out_n.location = (360, 0)
            ng.links.new(rl.outputs["Image"], gl.inputs["Image"])
            ng.links.new(gl.outputs["Image"], out_n.inputs["Image"])
            scn.compositing_node_group = ng
            print("[Compositor] Fog Glow glare node enabled (Threshold=1.90, Strength=0.045, Clamp=10.0)")
    except Exception as e:
        print(f"[Warning] Compositor setup skipped: {e}")


def setup_camera_dof():

    fstop_map = {
        "Cam_03_Oblique_Main":              8.0,
        "Cam_01_Rear_IrisCheck":            8.0,
        "Cam_02_Side_LinkageCheck":         9.0,
        "Cam_08_Macro_Inner_ThermalTiles":  8.0,
        "Cam_10_Front_Inlet_Fan":           8.0,
        "Cam_11_Macro_Shroud_Nozzle":       5.6,
        "Cam_12_Macro_Core_Stand":          5.6,
        "Cam_13_Macro_Stand_TK9_Decals":    5.6,
        "Cam_14_Macro_Shroud_Decals":       5.0,
        "Cam_15_Macro_Actuator_DualBand":   4.5,
        "Cam_Show_01_PitotInlet":           4.5,
        "Cam_Show_02_OilSight":             4.5,
        "Cam_Show_03_NozzleBronze":         4.5,
        "Cam_Show_04_CoreLeft":             5.6,
        "Cam_Show_05_BellowsWaffle":        4.5,
    }

    surface_dist_override = {
        "Cam_14_Macro_Shroud_Decals":     1.33,
        "Cam_13_Macro_Stand_TK9_Decals":  1.46,
        "Cam_11_Macro_Shroud_Nozzle":     1.90,
        "Cam_12_Macro_Core_Stand":        2.65,
        "Cam_Show_04_CoreLeft":           2.02,
    }
    for obj in bpy.data.objects:
        if obj.type != 'CAMERA':
            continue
        tgt = bpy.data.objects.get(f"{obj.name}_Target")
        if tgt is None:
            for c in obj.constraints:
                if c.type == 'TRACK_TO' and c.target is not None:
                    tgt = c.target
                    break
        if obj.name in surface_dist_override:
            obj.data.dof.use_dof = True
            obj.data.dof.focus_object = None
            obj.data.dof.focus_distance = surface_dist_override[obj.name]
            obj.data.dof.aperture_fstop = fstop_map.get(obj.name, 6.3)
            obj.data.dof.aperture_blades = 7
            obj.data.dof.aperture_ratio = 1.0
        elif tgt is not None:
            obj.data.dof.use_dof = True
            obj.data.dof.focus_object = tgt
            obj.data.dof.aperture_fstop = fstop_map.get(obj.name, 6.3)
            obj.data.dof.aperture_blades = 7
            obj.data.dof.aperture_ratio = 1.0


def setup_pbr_world_hdri_and_studio_floor(hdri_dir: str = None):

    if hdri_dir is None:
        hdri_dir = os.path.join(_THIS_DIR, "assets_hdri")

    img_hangar = _load_and_pack_hdri(os.path.join(hdri_dir, "aircraft_workshop_01_4k.hdr"), "HDRI_Aircraft_Workshop_4K")
    img_studio = _load_and_pack_hdri(os.path.join(hdri_dir, "studio_small_09_4k.hdr"), "HDRI_Studio_Softbox_4K")

    scn = bpy.context.scene
    world = bpy.data.worlds.get("World_F15_PBR_IBL")
    if world is None:
        world = bpy.data.worlds.new("World_F15_PBR_IBL")
    scn.world = world
    wnt = world.node_tree
    wnt.nodes.clear()

    tc = _new_node(wnt, "ShaderNodeTexCoord", (-1280, 120))
    mp = _new_node(wnt, "ShaderNodeMapping", (-1060, 120), name="HDRI_Mapping")
    mp.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(165.0))
    wnt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])

    env_h = _new_node(wnt, "ShaderNodeTexEnvironment", (-780, 280), name="Env_Hangar")
    if img_hangar is not None:
        env_h.image = img_hangar
    wnt.links.new(mp.outputs["Vector"], env_h.inputs["Vector"])

    env_s = _new_node(wnt, "ShaderNodeTexEnvironment", (-780, -40), name="Env_Studio")
    if img_studio is not None:
        env_s.image = img_studio
    wnt.links.new(mp.outputs["Vector"], env_s.inputs["Vector"])


    mix_env = _mix_rgba(wnt, 'MIX', (-460, 160), fac=1.0)
    mix_env.name = "HDRI_Select_Mix"
    wnt.links.new(env_h.outputs["Color"], mix_env.inputs[6])
    wnt.links.new(env_s.outputs["Color"], mix_env.inputs[7])

    lp = _new_node(wnt, "ShaderNodeLightPath", (-460, 420))


    gl_gam = _new_node(wnt, "ShaderNodeGamma", (-300, 320), name="HDRI_Glossy_Contrast")
    gl_gam.inputs["Gamma"].default_value = 1.12
    wnt.links.new(mix_env.outputs[2], gl_gam.inputs["Color"])
    gl_cap = _vmath(wnt, 'MINIMUM', (-160, 320))
    gl_cap.name = "HDRI_Glossy_Clamp"
    gl_cap.inputs[1].default_value = (22.0, 22.0, 22.0)
    wnt.links.new(gl_gam.outputs["Color"], gl_cap.inputs[0])
    mix_gl_env = _mix_rgba(wnt, 'MIX', (-40, 220))
    mix_gl_env.clamp_result = False
    wnt.links.new(lp.outputs["Is Glossy Ray"], mix_gl_env.inputs[0])
    wnt.links.new(mix_env.outputs[2], mix_gl_env.inputs[6])
    wnt.links.new(gl_cap.outputs["Vector"], mix_gl_env.inputs[7])


    bg_ibl = _new_node(wnt, "ShaderNodeBackground", (140, 220), name="BG_IBL_Lighting")
    bg_ibl.inputs["Strength"].default_value = 0.18
    wnt.links.new(mix_gl_env.outputs[2], bg_ibl.inputs["Color"])


    sep_z = _new_node(wnt, "ShaderNodeSeparateXYZ", (-960, -380))
    wnt.links.new(tc.outputs["Generated"], sep_z.inputs["Vector"])
    bg_zmap = _map_range(wnt, -0.42, 0.42, 0.0, 1.0, (-740, -380), clamp=True)
    wnt.links.new(sep_z.outputs["Z"], bg_zmap.inputs[0])
    ramp_bg = _ramp(wnt, [
        (0.00, rgba(0x0A0D14)),
        (0.28, rgba(0x161D2A)),
        (0.45, rgba(0x232D3F)),
        (0.65, rgba(0x121723)),
        (1.00, rgba(0x070A10)),
    ], loc=(-500, -380))
    wnt.links.new(bg_zmap.outputs[0], ramp_bg.inputs["Fac"])

    bg_clean = _new_node(wnt, "ShaderNodeBackground", (-180, -320), name="BG_Clean_Backdrop")
    bg_clean.inputs["Strength"].default_value = 0.96
    wnt.links.new(ramp_bg.outputs["Color"], bg_clean.inputs["Color"])


    bg_vis_hdri = _new_node(wnt, "ShaderNodeBackground", (-180, -80), name="BG_Visible_HDRI")
    bg_vis_hdri.inputs["Strength"].default_value = 0.65
    wnt.links.new(mix_env.outputs[2], bg_vis_hdri.inputs["Color"])

    mix_cam_bg = _new_node(wnt, "ShaderNodeMixShader", (120, -200), name="BG_ShowHDRI_Switch")
    mix_cam_bg.inputs["Fac"].default_value = 0.0
    wnt.links.new(bg_clean.outputs["Background"], mix_cam_bg.inputs[1])
    wnt.links.new(bg_vis_hdri.outputs["Background"], mix_cam_bg.inputs[2])

    mix_out = _new_node(wnt, "ShaderNodeMixShader", (420, 40))
    wnt.links.new(lp.outputs["Is Camera Ray"], mix_out.inputs["Fac"])
    wnt.links.new(bg_ibl.outputs["Background"], mix_out.inputs[1])
    wnt.links.new(mix_cam_bg.outputs["Shader"], mix_out.inputs[2])

    w_out = _new_node(wnt, "ShaderNodeOutputWorld", (660, 40))
    wnt.links.new(mix_out.outputs["Shader"], w_out.inputs["Surface"])


    coll_lights = bpy.data.collections.get("05_Studio_And_Cameras")
    if coll_lights is None:
        coll_lights = scn.collection
    for lo in list(bpy.data.objects):
        if lo.type != 'LIGHT':
            continue
        if lo.name.startswith("Studio_"):
            ld_old = lo.data
            bpy.data.objects.remove(lo, do_unlink=True)
            if ld_old and ld_old.users == 0:
                bpy.data.lights.remove(ld_old)
        elif lo.name.startswith("Clay_"):
            if "f15_clay_energy" not in lo.data:
                lo.data["f15_clay_energy"] = float(lo.data.energy)
            lo.data["f15_pbr_ratio"] = 0.0
            lo.data.energy = 0.0
            if hasattr(lo, "visible_camera"):
                lo.visible_camera = False


    _create_studio_scrim_light(
        coll_lights, "Studio_Key_OverheadBlade",
        loc=(-0.35, -2.00, 4.35), target=(0.0, -1.95, -0.28),
        size_x=0.58, size_y=6.00, energy=560.0, col_rgb=(1.00, 0.978, 0.942),
        spread_deg=50.0, shape='RECTANGLE', up_axis='Y', vis_glossy=True, hangar_ratio=0.48,
    )

    _create_studio_scrim_light(
        coll_lights, "Studio_Rim_AftCoolKnife",
        loc=(-2.50, 2.35, 1.75), target=(0.0, -0.05, 0.12),
        size_x=1.80, size_y=0.42, energy=350.0, col_rgb=(0.72, 0.86, 1.00),
        spread_deg=48.0, shape='RECTANGLE', up_axis='Z', vis_glossy=True, hangar_ratio=0.42,
    )

    _create_studio_scrim_light(
        coll_lights, "Studio_Key_InletSpot",
        loc=(1.55, -6.15, 1.95), target=(0.0, -4.15, -0.10),
        size_x=1.40, size_y=0.90, energy=200.0, col_rgb=(1.00, 0.975, 0.935),
        spread_deg=52.0, shape='RECTANGLE', up_axis='Z', vis_glossy=True, hangar_ratio=0.40,
    )

    _create_studio_scrim_light(
        coll_lights, "Studio_Fill_ShadowLift",
        loc=(4.20, -2.00, 0.25), target=(0.0, -2.20, -0.55),
        size_x=5.20, size_y=1.40, energy=72.0, col_rgb=(0.76, 0.86, 1.00),
        spread_deg=102.0, shape='RECTANGLE', up_axis='Z', vis_glossy=False, hangar_ratio=0.15,
    )

    _create_studio_scrim_light(
        coll_lights, "Studio_Fill_RearCavity",
        loc=(0.45, 3.25, 0.35), target=(0.0, -0.55, 0.0),
        size_x=0.65, size_y=0.65, energy=26.0, col_rgb=(0.98, 0.98, 1.00),
        spread_deg=46.0, shape='DISK', up_axis='Z', vis_glossy=True, hangar_ratio=0.35,
    )

    _create_studio_scrim_light(
        coll_lights, "Studio_Fill_OilSightBounce",
        loc=(-1.85, -4.60, -0.10), target=(-0.45, -3.95, -0.25),
        size_x=1.40, size_y=0.85, energy=48.0, col_rgb=(0.94, 0.96, 1.00),
        spread_deg=72.0, shape='RECTANGLE', up_axis='Z', vis_glossy=False, hangar_ratio=0.15,
    )

    _create_studio_scrim_light(
        coll_lights, "Studio_Rim_AftShroudGraze",
        loc=(2.40, 2.85, 0.95), target=(0.25, 0.45, 0.00),
        size_x=0.48, size_y=2.00, energy=62.0, col_rgb=(1.00, 0.93, 0.82),
        spread_deg=48.0, shape='RECTANGLE', up_axis='Z', vis_glossy=True, hangar_ratio=0.25,
    )


    ab_boost_map = {
        "Light_AB_Nozzle_Throat": 10.0,
        "Light_AB_TailCone_Core": 22.0,
        "Light_AB_Turbine_Deep":  48.0,
    }
    ab_warm_rgb = (1.00, 0.44, 0.14)
    ab_blue_rgb = (0.45, 0.72, 1.00)
    for ab_name, ab_w in ab_boost_map.items():
        ab_obj = bpy.data.objects.get(ab_name)
        if ab_obj and ab_obj.type == 'LIGHT' and ab_obj.data.animation_data:
            for fc in ab_obj.data.animation_data.drivers:
                if fc.data_path == "energy":
                    fc.driver.expression = f"glow_on * {ab_w:.1f}"
                elif fc.data_path == "color" and 0 <= fc.array_index < 3:
                    cw = ab_warm_rgb[fc.array_index]
                    cb = ab_blue_rgb[fc.array_index]
                    fc.driver.expression = f"{cw:.4f} + is_blue * ({cb - cw:.4f})"


    floor_name = "Studio_Ground_Floor"
    old_fl = bpy.data.objects.get(floor_name)
    if old_fl is not None:
        me_old = old_fl.data
        bpy.data.objects.remove(old_fl, do_unlink=True)
        if me_old and me_old.users == 0:
            bpy.data.meshes.remove(me_old)

    bm_fl = bmesh.new()

    nx_f, ny_f = 16, 16
    half_s = 18.0
    yc_f = -2.25
    zf = -1.6095
    grid = []
    for ix in range(nx_f + 1):
        row = []
        xv = -half_s + (2.0 * half_s) * (ix / float(nx_f))
        for iy in range(ny_f + 1):
            yv = (yc_f - half_s) + (2.0 * half_s) * (iy / float(ny_f))
            row.append(bm_fl.verts.new((xv, yv, zf)))
        grid.append(row)
    for ix in range(nx_f):
        for iy in range(ny_f):
            bm_fl.faces.new((grid[ix][iy], grid[ix + 1][iy], grid[ix + 1][iy + 1], grid[ix][iy + 1]))
    bm_apply_hard_surface_smoothing(bm_fl, angle_deg=42.0)
    me_fl = bpy.data.meshes.new(floor_name)
    bm_fl.to_mesh(me_fl)
    bm_fl.free()


    m_fl = _get_or_new_mat("Mat_Studio_HangarFloor")
    ntf = m_fl.node_tree
    ntf.nodes.clear()
    tc_f = _new_node(ntf, "ShaderNodeTexCoord", (-1380, 0))
    ng_g_fl = bpy.data.node_groups.get("NG_F15_GlobalControls")
    ctrl_fl = None
    if ng_g_fl is not None:
        ctrl_fl = _new_node(ntf, "ShaderNodeGroup", (-1380, -480))
        ctrl_fl.node_tree = ng_g_fl

    sub_c = _vmath(ntf, 'SUBTRACT', (-1060, -240))
    sub_c.inputs[1].default_value = (0.0, yc_f, zf)
    ntf.links.new(tc_f.outputs["Object"], sub_c.inputs[0])
    len_c = _vmath(ntf, 'LENGTH', (-840, -240))
    ntf.links.new(sub_c.outputs["Vector"], len_c.inputs[0])

    alpha_lin = _map_range(ntf, 7.5, 14.5, 1.0, 0.0, (-620, -240), clamp=True, interp='SMOOTHERSTEP')
    ntf.links.new(len_c.outputs["Value"], alpha_lin.inputs[0])
    alpha_fade = _math(ntf, 'POWER', (-400, -240), clamp=True, v1=1.45)
    ntf.links.new(alpha_lin.outputs[0], alpha_fade.inputs[0])

    n_fl1 = _noise(ntf, scale=1.1, detail=6.0, roughness=0.58, distortion=0.25, loc=(-1040, 180))
    ntf.links.new(tc_f.outputs["Object"], n_fl1.inputs["Vector"])
    n_fl2 = _noise(ntf, scale=32.0, detail=7.0, roughness=0.68, loc=(-1040, -20))
    ntf.links.new(tc_f.outputs["Object"], n_fl2.inputs["Vector"])
    vor_agg = _voronoi(ntf, scale=160.0, feature='F1', randomness=0.95, loc=(-1040, -420))
    ntf.links.new(tc_f.outputs["Object"], vor_agg.inputs["Vector"])


    sep_fl = _new_node(ntf, "ShaderNodeSeparateXYZ", (-1220, 560))
    ntf.links.new(tc_f.outputs["Object"], sep_fl.inputs["Vector"])


    tire_png = os.path.join(_THIS_DIR, "assets_masks_4k", "mask_4k_floor_tiretracks.png")
    img_ftire = bpy.data.images.get("IMG_Mask4K_FloorTireTracks")
    if os.path.exists(tire_png):
        if img_ftire is not None:
            bpy.data.images.remove(img_ftire)
        img_ftire = bpy.data.images.load(tire_png, check_existing=False)
        img_ftire.name = "IMG_Mask4K_FloorTireTracks"
        try:
            img_ftire.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass
        try:
            img_ftire.pack()
        except Exception:
            pass

    tr_r_out, tr_g_out, tr_b_out = None, None, None
    if img_ftire is not None:
        mp_ft = _new_node(ntf, "ShaderNodeMapping", (-1220, 420))
        mp_ft.vector_type = 'POINT'

        mp_ft.inputs["Location"].default_value = (0.50, 0.725, 0.0)
        mp_ft.inputs["Scale"].default_value = (0.10, 0.10, 1.0)
        ntf.links.new(tc_f.outputs["Object"], mp_ft.inputs["Vector"])

        t_ft = _new_node(ntf, "ShaderNodeTexImage", (-980, 420))
        t_ft.image = img_ftire
        t_ft.interpolation = 'Cubic'
        t_ft.extension = 'CLIP'
        ntf.links.new(mp_ft.outputs["Vector"], t_ft.inputs["Vector"])

        sep_tr = _new_node(ntf, "ShaderNodeSeparateColor", (-720, 420))
        ntf.links.new(t_ft.outputs["Color"], sep_tr.inputs["Color"])


        if ctrl_fl is not None:
            tr_wmap = _map_range(ntf, 0.15, 1.45, 0.16, 1.28, (-720, 260), clamp=True)
            ntf.links.new(ctrl_fl.outputs["Weathering"], tr_wmap.inputs[0])
            tr_gate = _math(ntf, 'MULTIPLY', (-540, 260), clamp=True)
            ntf.links.new(tr_wmap.outputs[0], tr_gate.inputs[0])
            ntf.links.new(ctrl_fl.outputs["Show_PBR"], tr_gate.inputs[1])
        else:
            tr_gate = _math(ntf, 'ADD', (-540, 260), v0=1.0, v1=0.0, clamp=True)

        tr_r_n = _math(ntf, 'MULTIPLY', (-480, 520), clamp=True)
        ntf.links.new(sep_tr.outputs["Red"], tr_r_n.inputs[0])
        ntf.links.new(tr_gate.outputs[0], tr_r_n.inputs[1])
        tr_r_out = tr_r_n.outputs[0]

        tr_g_n = _math(ntf, 'MULTIPLY', (-480, 420), clamp=True)
        ntf.links.new(sep_tr.outputs["Green"], tr_g_n.inputs[0])
        ntf.links.new(tr_gate.outputs[0], tr_g_n.inputs[1])
        tr_g_out = tr_g_n.outputs[0]

        tr_b_n = _math(ntf, 'MULTIPLY', (-480, 320), clamp=True)
        ntf.links.new(sep_tr.outputs["Blue"], tr_b_n.inputs[0])
        ntf.links.new(tr_gate.outputs[0], tr_b_n.inputs[1])
        tr_b_out = tr_b_n.outputs[0]


    bay_png = os.path.join(_THIS_DIR, "assets_masks_4k", "mask_4k_floor_baymarkings.png")
    img_fbay = bpy.data.images.get("IMG_Mask4K_FloorBayMarkings")
    if os.path.exists(bay_png):
        if img_fbay is not None:
            bpy.data.images.remove(img_fbay)
        img_fbay = bpy.data.images.load(bay_png, check_existing=False)
        img_fbay.name = "IMG_Mask4K_FloorBayMarkings"
        img_fbay.alpha_mode = 'CHANNEL_PACKED'
        try:
            img_fbay.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass
        try:
            img_fbay.pack()
        except Exception:
            pass

    bay_r_out, bay_g_out, bay_b_out, bay_a_core, bay_a_spall = None, None, None, None, None
    if img_fbay is not None:
        mp_fb = _new_node(ntf, "ShaderNodeMapping", (-1220, 680))
        mp_fb.vector_type = 'POINT'
        mp_fb.inputs["Location"].default_value = (0.50, 0.725, 0.0)
        mp_fb.inputs["Scale"].default_value = (0.10, 0.10, 1.0)
        ntf.links.new(tc_f.outputs["Object"], mp_fb.inputs["Vector"])

        t_fb = _new_node(ntf, "ShaderNodeTexImage", (-980, 680))
        t_fb.image = img_fbay
        t_fb.interpolation = 'Cubic'
        t_fb.extension = 'CLIP'
        ntf.links.new(mp_fb.outputs["Vector"], t_fb.inputs["Vector"])

        sep_fb = _new_node(ntf, "ShaderNodeSeparateColor", (-720, 680))
        ntf.links.new(t_fb.outputs["Color"], sep_fb.inputs["Color"])

        if ctrl_fl is not None:
            pbr_gate = ctrl_fl.outputs["Show_PBR"]

            bay_wfade = _map_range(ntf, 0.15, 1.45, 1.00, 0.72, (-640, 820), clamp=True)
            ntf.links.new(ctrl_fl.outputs["Weathering"], bay_wfade.inputs[0])
            bay_rgate = _math(ntf, 'MULTIPLY', (-480, 820), clamp=True)
            ntf.links.new(pbr_gate, bay_rgate.inputs[0])
            ntf.links.new(bay_wfade.outputs[0], bay_rgate.inputs[1])


            bay_obd = _map_range(ntf, 0.10, 1.40, 0.42, 1.15, (-640, 640), clamp=True)
            ntf.links.new(ctrl_fl.outputs["SurfaceErosion"], bay_obd.inputs[0])
            bay_ow0 = _math(ntf, 'MULTIPLY', (-480, 640), clamp=True)
            ntf.links.new(ctrl_fl.outputs["Weathering"], bay_ow0.inputs[0])
            ntf.links.new(bay_obd.outputs[0], bay_ow0.inputs[1])
            bay_ggate = _math(ntf, 'MULTIPLY', (-340, 640), clamp=True)
            ntf.links.new(bay_ow0.outputs[0], bay_ggate.inputs[0])
            ntf.links.new(pbr_gate, bay_ggate.inputs[1])


            bay_jw = _map_range(ntf, 0.10, 1.45, 0.25, 1.15, (-640, 540), clamp=True)
            ntf.links.new(ctrl_fl.outputs["Weathering"], bay_jw.inputs[0])
            bay_bgate = _math(ntf, 'MULTIPLY', (-480, 540), clamp=True)
            ntf.links.new(bay_jw.outputs[0], bay_bgate.inputs[0])
            ntf.links.new(pbr_gate, bay_bgate.inputs[1])
        else:
            bay_rgate = _math(ntf, 'ADD', (-480, 820), v0=0.90, v1=0.0, clamp=True)
            bay_ggate = _math(ntf, 'ADD', (-340, 640), v0=0.65, v1=0.0, clamp=True)
            bay_bgate = _math(ntf, 'ADD', (-480, 540), v0=0.80, v1=0.0, clamp=True)


        n_bay_med = _noise(ntf, scale=38.0, detail=9.0, roughness=0.74, distortion=0.45, loc=(-640, 960))
        ntf.links.new(tc_f.outputs["Object"], n_bay_med.inputs["Vector"])
        n_bay_fine = _noise(ntf, scale=260.0, detail=10.0, roughness=0.78, distortion=0.22, loc=(-640, 1080))
        ntf.links.new(tc_f.outputs["Object"], n_bay_fine.inputs["Vector"])
        n_bay_mix = _mix_float(ntf, (-480, 1000), fac=0.44)
        ntf.links.new(n_bay_med.outputs["Factor"], n_bay_mix.inputs[2])
        ntf.links.new(n_bay_fine.outputs["Factor"], n_bay_mix.inputs[3])
        bay_micro = _map_range(ntf, 0.24, 0.76, -0.30, +0.30, (-360, 1000), clamp=True)
        ntf.links.new(n_bay_mix.outputs[0], bay_micro.inputs[0])


        n_bay_pore = _noise(ntf, scale=620.0, detail=8.0, roughness=0.80, distortion=0.12, loc=(-640, 1200))
        ntf.links.new(tc_f.outputs["Object"], n_bay_pore.inputs["Vector"])
        bay_pore_k = _map_range(ntf, 0.24, 0.42, 0.72, 1.00, (-360, 1200), clamp=True)
        ntf.links.new(n_bay_pore.outputs["Factor"], bay_pore_k.inputs[0])


        if ctrl_fl is not None:
            bay_w_sub = _map_range(ntf, 0.15, 1.45, +0.10, -0.10, (-480, 900), clamp=True)
            ntf.links.new(ctrl_fl.outputs["Weathering"], bay_w_sub.inputs[0])
            bay_r_w = _math(ntf, 'ADD', (-360, 900), clamp=False)
            ntf.links.new(sep_fb.outputs["Red"], bay_r_w.inputs[0])
            ntf.links.new(bay_w_sub.outputs[0], bay_r_w.inputs[1])
            bay_r_base_sig = bay_r_w.outputs[0]
        else:
            bay_r_base_sig = sep_fb.outputs["Red"]

        bay_r_mod = _math(ntf, 'ADD', (-280, 920), clamp=True)
        ntf.links.new(bay_r_base_sig, bay_r_mod.inputs[0])
        ntf.links.new(bay_micro.outputs[0], bay_r_mod.inputs[1])

        bay_r_crisp = _map_range(ntf, 0.45, 0.49, 0.0, 1.0, (-220, 920), clamp=True, interp='LINEAR')
        ntf.links.new(bay_r_mod.outputs[0], bay_r_crisp.inputs[0])


        bay_geom_gate = _map_range(ntf, 0.20, 0.32, 0.0, 1.0, (-280, 820), clamp=True, interp='LINEAR')
        ntf.links.new(sep_fb.outputs["Red"], bay_geom_gate.inputs[0])

        br_sharp0 = _math(ntf, 'MULTIPLY', (-180, 860), clamp=True)
        ntf.links.new(bay_r_crisp.outputs[0], br_sharp0.inputs[0])
        ntf.links.new(bay_geom_gate.outputs[0], br_sharp0.inputs[1])

        br_sharp = _math(ntf, 'MULTIPLY', (-150, 860), clamp=True)
        ntf.links.new(br_sharp0.outputs[0], br_sharp.inputs[0])
        ntf.links.new(bay_pore_k.outputs[0], br_sharp.inputs[1])

        br_n = _math(ntf, 'MULTIPLY', (-120, 820), clamp=True)
        ntf.links.new(br_sharp.outputs[0], br_n.inputs[0])
        if ctrl_fl is not None:
            ntf.links.new(pbr_gate, br_n.inputs[1])
        else:
            br_n.inputs[1].default_value = 1.0
        bay_r_out = br_n.outputs[0]

        bg_n = _math(ntf, 'MULTIPLY', (-200, 680), clamp=True)
        ntf.links.new(sep_fb.outputs["Green"], bg_n.inputs[0])
        ntf.links.new(bay_ggate.outputs[0], bg_n.inputs[1])
        bay_g_out = bg_n.outputs[0]

        bb_n = _math(ntf, 'MULTIPLY', (-320, 580), clamp=True)
        ntf.links.new(sep_fb.outputs["Blue"], bb_n.inputs[0])
        ntf.links.new(bay_bgate.outputs[0], bb_n.inputs[1])
        bay_b_out = bb_n.outputs[0]


        ba_c = _map_range(ntf, 0.56, 0.90, 0.0, 0.85, (-520, 960), clamp=True)
        ntf.links.new(t_fb.outputs["Alpha"], ba_c.inputs[0])
        ba_sp0 = _map_range(ntf, 0.14, 0.56, 0.0, 0.72, (-520, 1080), clamp=True)
        ntf.links.new(t_fb.outputs["Alpha"], ba_sp0.inputs[0])
        ba_inv = _math(ntf, 'SUBTRACT', (-340, 960), clamp=True, v0=1.0)
        ntf.links.new(ba_c.outputs[0], ba_inv.inputs[1])
        ba_sp = _math(ntf, 'MULTIPLY', (-200, 1040), clamp=True)
        ntf.links.new(ba_sp0.outputs[0], ba_sp.inputs[0])
        ntf.links.new(ba_inv.outputs[0], ba_sp.inputs[1])
        bay_a_core = ba_c.outputs[0]
        bay_a_spall = ba_sp.outputs[0]

    mp_sc = _new_node(ntf, "ShaderNodeMapping", (-1040, 140))
    mp_sc.inputs["Scale"].default_value = (4.5, 0.85, 4.5)
    ntf.links.new(tc_f.outputs["Object"], mp_sc.inputs["Vector"])
    n_sc = _noise(ntf, scale=1.0, detail=5.0, roughness=0.62, distortion=0.35, loc=(-820, 140))
    ntf.links.new(mp_sc.outputs["Vector"], n_sc.inputs["Vector"])


    oil_sub = _vmath(ntf, 'SUBTRACT', (-1040, -640))
    oil_sub.inputs[1].default_value = (0.04, -2.55, zf)
    ntf.links.new(tc_f.outputs["Object"], oil_sub.inputs[0])
    oil_sc = _vmath(ntf, 'MULTIPLY', (-880, -640))
    oil_sc.inputs[1].default_value = (1.7, 1.1, 1.0)
    ntf.links.new(oil_sub.outputs["Vector"], oil_sc.inputs[0])
    oil_d = _vmath(ntf, 'LENGTH', (-720, -640))
    ntf.links.new(oil_sc.outputs["Vector"], oil_d.inputs[0])
    n_oil_fl = _noise(ntf, scale=3.4, detail=5.0, roughness=0.58, distortion=0.65, loc=(-880, -780))
    ntf.links.new(tc_f.outputs["Object"], n_oil_fl.inputs["Vector"])
    oil_dj = _math(ntf, 'MULTIPLY_ADD', (-560, -660), v1=0.35)
    ntf.links.new(n_oil_fl.outputs["Factor"], oil_dj.inputs[0])
    ntf.links.new(oil_d.outputs["Value"], oil_dj.inputs[2])
    oil_rad = _map_range(ntf, 0.42, 0.88, 1.0, 0.0, (-400, -660), clamp=True)
    ntf.links.new(oil_dj.outputs[0], oil_rad.inputs[0])
    if ctrl_fl is not None:
        oil_bd = _map_range(ntf, 0.20, 1.40, 0.18, 0.72, (-640, -820), clamp=True)
        ntf.links.new(ctrl_fl.outputs["SurfaceErosion"], oil_bd.inputs[0])
        oil_w0 = _math(ntf, 'MULTIPLY', (-420, -820), clamp=True)
        ntf.links.new(ctrl_fl.outputs["Weathering"], oil_w0.inputs[0])
        ntf.links.new(oil_bd.outputs[0], oil_w0.inputs[1])
        oil_eff = _math(ntf, 'MULTIPLY', (-220, -680), clamp=True)
        ntf.links.new(oil_rad.outputs[0], oil_eff.inputs[0])
        ntf.links.new(oil_w0.outputs[0], oil_eff.inputs[1])
    else:
        oil_eff = _math(ntf, 'MULTIPLY', (-220, -680), clamp=True, v1=0.25)
        ntf.links.new(oil_rad.outputs[0], oil_eff.inputs[0])


    img_fl_smg = bpy.data.images.get("IMG_Mask4K_SmudgesStains")
    img_fl_sc  = bpy.data.images.get("IMG_Mask4K_ScratchesFine")
    img_fl_af  = bpy.data.images.get("IMG_Mask4K_AirflowGrime")


    fl_sx = _math(ntf, 'MULTIPLY_ADD', (-1040, 820), v1=1.0 / 1.80, v2=0.50)
    ntf.links.new(sep_fl.outputs["X"], fl_sx.inputs[0])
    fl_fx = _math(ntf, 'FRACT', (-880, 820))
    ntf.links.new(fl_sx.outputs[0], fl_fx.inputs[0])
    fl_dx = _math(ntf, 'SUBTRACT', (-720, 820), v1=0.5)
    ntf.links.new(fl_fx.outputs[0], fl_dx.inputs[0])
    fl_adx = _math(ntf, 'ABSOLUTE', (-560, 820))
    ntf.links.new(fl_dx.outputs[0], fl_adx.inputs[0])
    fl_gx = _map_range(ntf, 0.494, 0.4985, 0.0, 0.78, (-400, 820), clamp=True)
    ntf.links.new(fl_adx.outputs[0], fl_gx.inputs[0])

    fl_sy = _math(ntf, 'MULTIPLY_ADD', (-1040, 940), v1=1.0 / 1.80, v2=0.25)
    ntf.links.new(sep_fl.outputs["Y"], fl_sy.inputs[0])
    fl_fy = _math(ntf, 'FRACT', (-880, 940))
    ntf.links.new(fl_sy.outputs[0], fl_fy.inputs[0])
    fl_dy = _math(ntf, 'SUBTRACT', (-720, 940), v1=0.5)
    ntf.links.new(fl_fy.outputs[0], fl_dy.inputs[0])
    fl_ady = _math(ntf, 'ABSOLUTE', (-560, 940))
    ntf.links.new(fl_dy.outputs[0], fl_ady.inputs[0])
    fl_gy = _map_range(ntf, 0.494, 0.4985, 0.0, 0.78, (-400, 940), clamp=True)
    ntf.links.new(fl_ady.outputs[0], fl_gy.inputs[0])

    fl_joint = _math(ntf, 'MAXIMUM', (-220, 880), clamp=True)
    ntf.links.new(fl_gx.outputs[0], fl_joint.inputs[0])
    ntf.links.new(fl_gy.outputs[0], fl_joint.inputs[1])


    fl_smg_out = None
    if img_fl_smg is not None:
        mp_fsmg1 = _new_node(ntf, "ShaderNodeMapping", (-1040, -960))
        mp_fsmg1.inputs["Scale"].default_value = (0.31, 0.31, 0.31)
        mp_fsmg1.inputs["Rotation"].default_value = (0.0, 0.0, 0.42)
        ntf.links.new(tc_f.outputs["Object"], mp_fsmg1.inputs["Vector"])
        t_fsmg1 = _new_node(ntf, "ShaderNodeTexImage", (-820, -960))
        t_fsmg1.image = img_fl_smg
        ntf.links.new(mp_fsmg1.outputs["Vector"], t_fsmg1.inputs["Vector"])

        mp_fsmg2 = _new_node(ntf, "ShaderNodeMapping", (-1040, -1160))
        mp_fsmg2.inputs["Scale"].default_value = (0.19, 0.19, 0.19)
        mp_fsmg2.inputs["Rotation"].default_value = (0.0, 0.0, 1.85)
        ntf.links.new(tc_f.outputs["Object"], mp_fsmg2.inputs["Vector"])
        t_fsmg2 = _new_node(ntf, "ShaderNodeTexImage", (-820, -1160))
        t_fsmg2.image = img_fl_smg
        ntf.links.new(mp_fsmg2.outputs["Vector"], t_fsmg2.inputs["Vector"])

        mx_fsmg = _mix_float(ntf, (-560, -1040))
        ntf.links.new(n_fl1.outputs["Factor"], mx_fsmg.inputs[0])
        ntf.links.new(t_fsmg1.outputs["Color"], mx_fsmg.inputs[2])
        ntf.links.new(t_fsmg2.outputs["Color"], mx_fsmg.inputs[3])
        fl_smg_out = mx_fsmg.outputs[0]

    fl_sc_out = None
    if img_fl_sc is not None:
        mp_fsc = _new_node(ntf, "ShaderNodeMapping", (-1040, -1360))
        mp_fsc.inputs["Scale"].default_value = (0.58, 0.58, 0.58)
        mp_fsc.inputs["Rotation"].default_value = (0.0, 0.0, 0.72)
        ntf.links.new(tc_f.outputs["Object"], mp_fsc.inputs["Vector"])
        t_fsc = _new_node(ntf, "ShaderNodeTexImage", (-820, -1360))
        t_fsc.image = img_fl_sc
        ntf.links.new(mp_fsc.outputs["Vector"], t_fsc.inputs["Vector"])
        fsc_m = _map_range(ntf, 0.32, 0.86, 0.0, 1.0, (-560, -1360), clamp=True)
        ntf.links.new(t_fsc.outputs["Color"], fsc_m.inputs[0])
        fl_sc_out = fsc_m.outputs[0]

    fl_af_out = None
    if img_fl_af is not None:
        mp_faf = _new_node(ntf, "ShaderNodeMapping", (-1040, -1560))
        mp_faf.inputs["Scale"].default_value = (0.45, 0.24, 0.45)
        ntf.links.new(tc_f.outputs["Object"], mp_faf.inputs["Vector"])
        t_faf = _new_node(ntf, "ShaderNodeTexImage", (-820, -1560))
        t_faf.image = img_fl_af
        ntf.links.new(mp_faf.outputs["Vector"], t_faf.inputs["Vector"])
        faf_y = _map_range(ntf, -0.60, 1.40, 0.0, 0.55, (-560, -1500), clamp=True)
        ntf.links.new(sep_fl.outputs["Y"], faf_y.inputs[0])
        faf_m = _math(ntf, 'MULTIPLY', (-380, -1540), clamp=True)
        ntf.links.new(t_faf.outputs["Color"], faf_m.inputs[0])
        ntf.links.new(faf_y.outputs[0], faf_m.inputs[1])
        fl_af_out = faf_m.outputs[0]


    col_fl0 = _mix_rgba(ntf, 'MIX', (-540, 220), col_a=rgba(0x262B33), col_b=rgba(0x1B1F26))
    ntf.links.new(n_fl1.outputs["Factor"], col_fl0.inputs[0])
    agg_s = _map_range(ntf, 0.0, 0.25, 0.10, 0.0, (-680, -420), clamp=True)
    ntf.links.new(vor_agg.outputs["Distance"], agg_s.inputs[0])
    col_fl1 = _mix_rgba(ntf, 'MIX', (-340, 220), col_b=rgba(0x323842))
    ntf.links.new(agg_s.outputs[0], col_fl1.inputs[0])
    ntf.links.new(col_fl0.outputs[2], col_fl1.inputs[6])

    cur_fl_col = col_fl1.outputs[2]
    if fl_smg_out is not None:
        smg_cf = _math(ntf, 'MULTIPLY', (-260, -960), clamp=True, v1=0.12)
        ntf.links.new(fl_smg_out, smg_cf.inputs[0])
        mix_fl_smg = _mix_rgba(ntf, 'MIX', (-240, 220), col_b=rgba(0x16171B))
        ntf.links.new(smg_cf.outputs[0], mix_fl_smg.inputs[0])
        ntf.links.new(cur_fl_col, mix_fl_smg.inputs[6])
        cur_fl_col = mix_fl_smg.outputs[2]
    if fl_af_out is not None:
        af_cf = _math(ntf, 'MULTIPLY', (-220, -1540), clamp=True, v1=0.35)
        ntf.links.new(fl_af_out, af_cf.inputs[0])
        mix_fl_af = _mix_rgba(ntf, 'MIX', (-210, 240), col_b=rgba(0x060608))
        ntf.links.new(af_cf.outputs[0], mix_fl_af.inputs[0])
        ntf.links.new(cur_fl_col, mix_fl_af.inputs[6])
        cur_fl_col = mix_fl_af.outputs[2]


    if bay_r_out is not None:
        bay_col_var = _mix_float(ntf, (-255, 740), fac=0.45)
        ntf.links.new(n_fl1.outputs["Factor"], bay_col_var.inputs[2])
        ntf.links.new(n_fl2.outputs["Factor"], bay_col_var.inputs[3])
        bay_yel = _mix_rgba(ntf, 'MIX', (-230, 740), col_a=rgba(0x884402), col_b=rgba(0x562801))
        ntf.links.new(bay_col_var.outputs[0], bay_yel.inputs[0])
        bay_cf = _math(ntf, 'MULTIPLY', (-210, 660), clamp=True, v1=0.92)
        ntf.links.new(bay_r_out, bay_cf.inputs[0])
        mix_fl_bay = _mix_rgba(ntf, 'MIX', (-205, 340))
        ntf.links.new(bay_cf.outputs[0], mix_fl_bay.inputs[0])
        ntf.links.new(cur_fl_col, mix_fl_bay.inputs[6])
        ntf.links.new(bay_yel.outputs[2], mix_fl_bay.inputs[7])
        cur_fl_col = mix_fl_bay.outputs[2]


    if bay_b_out is not None:
        jk_cf = _math(ntf, 'MULTIPLY', (-220, 580), clamp=True, v1=0.62)
        ntf.links.new(bay_b_out, jk_cf.inputs[0])
        mix_fl_jk = _mix_rgba(ntf, 'MIX', (-195, 310), col_b=rgba(0x4C4B48))
        ntf.links.new(jk_cf.outputs[0], mix_fl_jk.inputs[0])
        ntf.links.new(cur_fl_col, mix_fl_jk.inputs[6])
        cur_fl_col = mix_fl_jk.outputs[2]


    if tr_r_out is not None and tr_g_out is not None:
        tr_r_cf = _math(ntf, 'MULTIPLY', (-260, 540), clamp=True, v1=0.58)
        ntf.links.new(tr_r_out, tr_r_cf.inputs[0])
        col_tr_r = _mix_rgba(ntf, 'MIX', (-200, 280), col_b=rgba(0x050607))
        ntf.links.new(tr_r_cf.outputs[0], col_tr_r.inputs[0])
        ntf.links.new(cur_fl_col, col_tr_r.inputs[6])

        tr_g_cf = _math(ntf, 'MULTIPLY', (-260, 440), clamp=True, v1=0.48)
        ntf.links.new(tr_g_out, tr_g_cf.inputs[0])
        col_fl2 = _mix_rgba(ntf, 'MIX', (-160, 260), col_b=rgba(0x565A60))
        ntf.links.new(tr_g_cf.outputs[0], col_fl2.inputs[0])
        ntf.links.new(col_tr_r.outputs[2], col_fl2.inputs[6])
    else:
        col_fl2 = _mix_rgba(ntf, 'MIX', (-180, 260), fac=0.0, col_b=rgba(0x07080A))
        ntf.links.new(cur_fl_col, col_fl2.inputs[6])


    joint_eff_out = bay_a_core if bay_a_core is not None else fl_joint.outputs[0]
    col_fl_prej = col_fl2.outputs[2]
    if bay_a_spall is not None:
        sp_cf = _math(ntf, 'MULTIPLY', (-140, 960), clamp=True, v1=0.42)
        ntf.links.new(bay_a_spall, sp_cf.inputs[0])
        col_fl_sp = _mix_rgba(ntf, 'MIX', (-130, 260), col_b=rgba(0x2E3138))
        ntf.links.new(sp_cf.outputs[0], col_fl_sp.inputs[0])
        ntf.links.new(col_fl_prej, col_fl_sp.inputs[6])
        col_fl_prej = col_fl_sp.outputs[2]

    col_fl3 = _mix_rgba(ntf, 'MIX', (-90, 260), col_b=rgba(0x050608))
    ntf.links.new(joint_eff_out, col_fl3.inputs[0])
    ntf.links.new(col_fl_prej, col_fl3.inputs[6])


    oil_eff_out = bay_g_out if bay_g_out is not None else oil_eff.outputs[0]
    col_fl = _mix_rgba(ntf, 'MIX', (-20, 220), col_b=rgba(0x050403))
    ntf.links.new(oil_eff_out, col_fl.inputs[0])
    ntf.links.new(col_fl3.outputs[2], col_fl.inputs[6])

    r_fl0 = _map_range(ntf, 0.20, 0.80, 0.36, 0.56, (-540, 40), clamp=True)
    ntf.links.new(n_fl1.outputs["Factor"], r_fl0.inputs[0])
    r_sc = _map_range(ntf, 0.45, 0.75, -0.06, +0.10, (-540, 360), clamp=True)
    ntf.links.new(n_sc.outputs["Factor"], r_sc.inputs[0])
    r_fl1 = _math(ntf, 'ADD', (-340, 40), clamp=True)
    ntf.links.new(r_fl0.outputs[0], r_fl1.inputs[0])
    ntf.links.new(r_sc.outputs[0], r_fl1.inputs[1])
    cur_fl_r = r_fl1.outputs[0]
    if fl_smg_out is not None:
        smg_rf = _map_range(ntf, 0.0, 1.0, -0.075, +0.135, (-340, -1040), clamp=True)
        ntf.links.new(fl_smg_out, smg_rf.inputs[0])
        r_add_fs = _math(ntf, 'ADD', (-280, 40), clamp=True)
        ntf.links.new(cur_fl_r, r_add_fs.inputs[0])
        ntf.links.new(smg_rf.outputs[0], r_add_fs.inputs[1])
        cur_fl_r = r_add_fs.outputs[0]
    if fl_sc_out is not None:
        fsc_rf = _math(ntf, 'MULTIPLY', (-340, -1360), clamp=True, v1=0.14)
        ntf.links.new(fl_sc_out, fsc_rf.inputs[0])
        r_add_fsc = _math(ntf, 'ADD', (-240, 40), clamp=True)
        ntf.links.new(cur_fl_r, r_add_fsc.inputs[0])
        ntf.links.new(fsc_rf.outputs[0], r_add_fsc.inputs[1])
        cur_fl_r = r_add_fsc.outputs[0]
    if bay_r_out is not None:
        r_fl_bay = _mix_float(ntf, (-225, 40), b=0.58)
        ntf.links.new(bay_r_out, r_fl_bay.inputs[0])
        ntf.links.new(cur_fl_r, r_fl_bay.inputs[2])
        cur_fl_r = r_fl_bay.outputs[0]
    if bay_b_out is not None:
        r_jk = _math(ntf, 'MULTIPLY', (-225, -20), clamp=True, v1=0.28)
        ntf.links.new(bay_b_out, r_jk.inputs[0])
        r_add_jk = _math(ntf, 'ADD', (-205, 40), clamp=True)
        ntf.links.new(cur_fl_r, r_add_jk.inputs[0])
        ntf.links.new(r_jk.outputs[0], r_add_jk.inputs[1])
        cur_fl_r = r_add_jk.outputs[0]


    tr_coat_kill = None
    if tr_r_out is not None and tr_g_out is not None and tr_b_out is not None:
        r_tr_b = _math(ntf, 'MULTIPLY', (-320, 320), clamp=True, v1=0.18)
        ntf.links.new(tr_b_out, r_tr_b.inputs[0])
        r_tr_g = _math(ntf, 'MULTIPLY', (-320, 220), clamp=True, v1=0.34)
        ntf.links.new(tr_g_out, r_tr_g.inputs[0])
        r_tr_r = _math(ntf, 'MULTIPLY', (-320, 120), clamp=True, v1=0.16)
        ntf.links.new(tr_r_out, r_tr_r.inputs[0])
        r_tr_sum0 = _math(ntf, 'ADD', (-250, 260), clamp=True)
        ntf.links.new(r_tr_b.outputs[0], r_tr_sum0.inputs[0])
        ntf.links.new(r_tr_g.outputs[0], r_tr_sum0.inputs[1])
        r_tr_sum = _math(ntf, 'ADD', (-210, 200), clamp=True)
        ntf.links.new(r_tr_sum0.outputs[0], r_tr_sum.inputs[0])
        ntf.links.new(r_tr_r.outputs[0], r_tr_sum.inputs[1])

        r_fl2 = _math(ntf, 'ADD', (-180, 40), clamp=True)
        ntf.links.new(cur_fl_r, r_fl2.inputs[0])
        ntf.links.new(r_tr_sum.outputs[0], r_fl2.inputs[1])

        ck_rg = _math(ntf, 'MAXIMUM', (-240, -140), clamp=True)
        ntf.links.new(tr_r_out, ck_rg.inputs[0])
        ntf.links.new(tr_g_out, ck_rg.inputs[1])
        ck_b = _math(ntf, 'MULTIPLY', (-240, -220), clamp=True, v1=0.65)
        ntf.links.new(tr_b_out, ck_b.inputs[0])
        tr_coat_kill = _math(ntf, 'MAXIMUM', (-160, -180), clamp=True)
        ntf.links.new(ck_rg.outputs[0], tr_coat_kill.inputs[0])
        ntf.links.new(ck_b.outputs[0], tr_coat_kill.inputs[1])
    else:
        r_fl2 = _math(ntf, 'ADD', (-200, 40), clamp=True, v1=0.0)
        ntf.links.new(cur_fl_r, r_fl2.inputs[0])

    r_fl_prej = r_fl2.outputs[0]
    if bay_a_spall is not None:
        r_fl_sp = _mix_float(ntf, (-150, 40), b=0.86)
        ntf.links.new(bay_a_spall, r_fl_sp.inputs[0])
        ntf.links.new(r_fl_prej, r_fl_sp.inputs[2])
        r_fl_prej = r_fl_sp.outputs[0]

    r_fl3 = _mix_float(ntf, (-110, 40), b=0.94)
    ntf.links.new(joint_eff_out, r_fl3.inputs[0])
    ntf.links.new(r_fl_prej, r_fl3.inputs[2])
    r_fl = _mix_float(ntf, (-20, 40), b=0.16)
    ntf.links.new(oil_eff_out, r_fl.inputs[0])
    ntf.links.new(r_fl3.outputs[0], r_fl.inputs[2])

    coat_kill_eff = tr_coat_kill.outputs[0] if tr_coat_kill is not None else None
    if bay_b_out is not None and coat_kill_eff is not None:
        ck_jk = _math(ntf, 'MAXIMUM', (-130, -180), clamp=True)
        ntf.links.new(coat_kill_eff, ck_jk.inputs[0])
        ntf.links.new(bay_b_out, ck_jk.inputs[1])
        coat_kill_eff = ck_jk.outputs[0]
    if bay_r_out is not None and coat_kill_eff is not None:
        ck_br = _math(ntf, 'MULTIPLY', (-130, -240), clamp=True, v1=0.92)
        ntf.links.new(bay_r_out, ck_br.inputs[0])
        ck_all = _math(ntf, 'MAXIMUM', (-110, -200), clamp=True)
        ntf.links.new(coat_kill_eff, ck_all.inputs[0])
        ntf.links.new(ck_br.outputs[0], ck_all.inputs[1])
        coat_kill_eff = ck_all.outputs[0]


    coat_base = _mix_float(ntf, (-90, -140), a=0.24, b=0.00)
    if coat_kill_eff is not None:
        ntf.links.new(coat_kill_eff, coat_base.inputs[0])
    else:
        coat_base.inputs[0].default_value = 0.0
    coat_fl = _mix_float(ntf, (-20, -140), b=0.42)
    ntf.links.new(oil_eff_out, coat_fl.inputs[0])
    ntf.links.new(coat_base.outputs[0], coat_fl.inputs[2])
    coat_r_fl = _mix_float(ntf, (-40, -260), a=0.24, b=0.14)
    ntf.links.new(oil_eff_out, coat_r_fl.inputs[0])


    fl_macro_terms = []
    fl_micro_terms = [
        (n_fl2.outputs["Factor"], 0.022, 0.0010),
        (vor_agg.outputs["Distance"], 0.022, 0.0011),
    ]
    if fl_sc_out is not None:
        fsc_inv = _math(ntf, 'SUBTRACT', (-200, -1360), clamp=True, v0=1.0)
        ntf.links.new(fl_sc_out, fsc_inv.inputs[1])
        fl_micro_terms.append((fsc_inv.outputs[0], 0.024, 0.0005))
    if bay_r_out is not None:
        fl_micro_terms.append((bay_r_out, 0.030, 0.00055))
    if tr_g_out is not None:
        fl_micro_terms.append((tr_g_out, 0.015, 0.0004))
    if bay_a_spall is not None:
        sp_inv = _math(ntf, 'SUBTRACT', (-95, 1020), clamp=True, v0=1.0)
        ntf.links.new(bay_a_spall, sp_inv.inputs[1])
        fl_macro_terms.append((sp_inv.outputs[0], 0.065, 0.0012))
    fl_j_inv = _math(ntf, 'SUBTRACT', (-80, 880), clamp=True, v0=1.0)
    ntf.links.new(joint_eff_out, fl_j_inv.inputs[1])
    fl_macro_terms.append((fl_j_inv.outputs[0], 0.12, 0.0024))

    cur_fl_norm = _combine_scalar_heights(ntf, fl_macro_terms, None, ref_distance=0.0020, loc=(-220, -100))
    cur_fl_norm = _combine_scalar_heights(ntf, fl_micro_terms, cur_fl_norm, ref_distance=0.0008, loc=(-40, -100))

    bsdf_fl = _new_node(ntf, "ShaderNodeBsdfPrincipled", (120, 120))
    ntf.links.new(col_fl.outputs[2], bsdf_fl.inputs["Base Color"])
    bsdf_fl.inputs["Metallic"].default_value = 0.0
    ntf.links.new(r_fl.outputs[0], bsdf_fl.inputs["Roughness"])
    bsdf_fl.inputs["Specular IOR Level"].default_value = 0.42
    ntf.links.new(coat_fl.outputs[0], bsdf_fl.inputs["Coat Weight"])
    ntf.links.new(coat_r_fl.outputs[0], bsdf_fl.inputs["Coat Roughness"])
    ntf.links.new(cur_fl_norm, bsdf_fl.inputs["Normal"])
    if "Coat Normal" in bsdf_fl.inputs:
        ntf.links.new(cur_fl_norm, bsdf_fl.inputs["Coat Normal"])

    trans_fl = _new_node(ntf, "ShaderNodeBsdfTransparent", (60, -180))
    mix_fl = _new_node(ntf, "ShaderNodeMixShader", (360, 40))
    ntf.links.new(alpha_fade.outputs[0], mix_fl.inputs["Fac"])
    ntf.links.new(trans_fl.outputs["BSDF"], mix_fl.inputs[1])
    ntf.links.new(bsdf_fl.outputs["BSDF"], mix_fl.inputs[2])

    out_fl = _new_node(ntf, "ShaderNodeOutputMaterial", (580, 40))
    ntf.links.new(mix_fl.outputs["Shader"], out_fl.inputs["Surface"])
    m_fl.diffuse_color = rgba(0x1E222A)

    me_fl.materials.append(m_fl)
    obj_fl = bpy.data.objects.new(floor_name, me_fl)
    coll_lights.objects.link(obj_fl)
    attach_weighted_normal_modifier(obj_fl)


    scn.render.engine = 'CYCLES'
    scn.cycles.use_denoising = True
    scn.cycles.max_bounces = 10
    scn.cycles.diffuse_bounces = 4
    scn.cycles.glossy_bounces = 8
    scn.cycles.transmission_bounces = 6
    scn.cycles.transparent_max_bounces = 8
    scn.cycles.sample_clamp_direct = 18.0
    scn.cycles.sample_clamp_indirect = 28.0
    scn.view_settings.view_transform = 'AgX'
    try:
        scn.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass
    scn.view_settings.exposure = 0.0

    setup_studio_compositor(scn)
    setup_camera_dof()


    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for dev_type in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI'):
            try:
                prefs.compute_device_type = dev_type
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type != 'CPU']
                if gpus:
                    for d in prefs.devices:
                        d.use = (d.type != 'CPU')
                    scn.cycles.device = 'GPU'
                    break
            except Exception:
                continue
    except Exception:
        pass

    print("[Lighting] Configured studio area light array, dual 4K HDRI world, ground plane, and AgX color management")


ENHANCED_UI_PANEL_SCRIPT_SOURCE = '''import math
import bpy

def _update_nozzle_props(self, context):
    if context and context.view_layer:
        context.view_layer.update()

def _srgb_to_lin_ui(c):
    return (c / 12.92) if c <= 0.04045 else (((c + 0.055) / 1.055) ** 2.4)

def _hex_rgba_ui(h):
    return (
        _srgb_to_lin_ui(((h >> 16) & 0xFF) / 255.0),
        _srgb_to_lin_ui(((h >> 8) & 0xFF) / 255.0),
        _srgb_to_lin_ui((h & 0xFF) / 255.0),
        1.0,
    )

def _sync_decal_visibility(scn):
    mode = getattr(scn, "f15_shade_mode", 'PBR')
    show_dec = getattr(scn, "f15_show_decals", True) and (mode == 'PBR')
    show_sh = getattr(scn, "f15_show_shroud", True)
    show_st = getattr(scn, "f15_show_stand", True)
    show_tr = getattr(scn, "f15_show_trailer", True)
    rules = {
        "Decals_Engine_Shroud":    show_dec and show_sh,
        "Decals_Engine_Core":      show_dec,
        "Decals_Nozzle_PetalNumbers": show_dec,
        "Decals_Stand_Cradle":     show_dec and show_st,
        "Decals_Stand_Trailer":    show_dec and show_st and show_tr,
        "Decals_Stand_Wheels":     show_dec and show_st and show_tr,
        "Decals_TorqueSeal_Engine": show_dec,
        "Decals_TorqueSeal_Stand":  show_dec and show_st,
    }
    for oname, vis in rules.items():
        obj = bpy.data.objects.get(oname)
        if obj is not None:
            obj.hide_viewport = not vis
            obj.hide_render = not vis
    fl = bpy.data.objects.get("Studio_Ground_Floor")
    if fl is not None:
        vis_fl = getattr(scn, "f15_show_floor", True) and (mode == 'PBR')
        fl.hide_viewport = not vis_fl
        fl.hide_render = not vis_fl

def _update_shader_and_lookdev(self, context):
    mode = getattr(self, "f15_shade_mode", 'PBR')
    ng = bpy.data.node_groups.get("NG_F15_GlobalControls")
    if ng is not None:
        is_pbr = 1.0 if mode == 'PBR' else 0.0
        is_cid = 1.0 if mode == 'COLOR_ID' else 0.0
        show_dec = 1.0 if getattr(self, "f15_show_decals", True) else 0.0
        vals = {
            "Val_Show_PBR":       is_pbr,
            "Val_Show_ColorID":   is_cid,
            "Val_Weathering":     float(getattr(self, "f15_weathering", 0.95)),
            "Val_HeatTint":       float(getattr(self, "f15_heat_tint", 1.0)),
            "Val_Soot":           float(getattr(self, "f15_soot", 1.0)),
            "Val_EdgeWear":       float(getattr(self, "f15_edge_wear", 0.90)),
            "Val_SurfaceErosion": float(getattr(self, "f15_surface_erosion", 0.45)),
            "Val_Show_Decals":    show_dec,
        }
        for nname, val in vals.items():
            nd = ng.nodes.get(nname)
            if nd is not None:
                nd.outputs[0].default_value = val

    _sync_decal_visibility(self)

    for mat in bpy.data.materials:
        if "f15_id_hex" in mat:
            if mode == 'PBR' and "f15_pbr_hex" in mat:
                mat.diffuse_color = _hex_rgba_ui(int(mat["f15_pbr_hex"]))
            elif mode == 'COLOR_ID':
                mat.diffuse_color = _hex_rgba_ui(int(mat["f15_id_hex"]))
            else:
                mat.diffuse_color = (0.86, 0.87, 0.89, 1.0)

    if context and context.screen:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for sp in area.spaces:
                    if sp.type == 'VIEW_3D':
                        sp.shading.color_type = 'OBJECT' if mode == 'CLAY' else 'MATERIAL'
    if context and context.view_layer:
        context.view_layer.update()

def _update_lighting_preset(self, context):
    preset = getattr(self, "f15_light_preset", 'STUDIO')
    rot_deg = float(getattr(self, "f15_hdri_rot", 165.0))
    show_bg = getattr(self, "f15_show_hdri_bg", False)
    show_fl = getattr(self, "f15_show_floor", True)

    scn = context.scene if context and context.scene else bpy.context.scene
    world = scn.world if scn else None
    if world and world.node_tree:
        wnt = world.node_tree
        mp = wnt.nodes.get("HDRI_Mapping")
        if mp is not None:
            mp.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rot_deg))
        mx = wnt.nodes.get("HDRI_Select_Mix")
        bg_ibl = wnt.nodes.get("BG_IBL_Lighting")
        bg_cln = wnt.nodes.get("BG_Clean_Backdrop")
        sw = wnt.nodes.get("BG_ShowHDRI_Switch")
        if preset == 'STUDIO':
            if mx: mx.inputs[0].default_value = 1.0
            if bg_ibl: bg_ibl.inputs["Strength"].default_value = 0.18
            if bg_cln: bg_cln.inputs["Strength"].default_value = 0.96
            if sw: sw.inputs["Fac"].default_value = 1.0 if show_bg else 0.0
            if scn: scn.view_settings.exposure = 0.0
            for lo in bpy.data.objects:
                if lo.type != 'LIGHT':
                    continue
                if "f15_studio_energy" in lo.data:
                    lo.data.energy = float(lo.data["f15_studio_energy"])
                elif "f15_clay_energy" in lo.data:
                    lo.data.energy = 0.0
        elif preset == 'HANGAR':
            if mx: mx.inputs[0].default_value = 0.0
            if bg_ibl: bg_ibl.inputs["Strength"].default_value = 0.88
            if bg_cln: bg_cln.inputs["Strength"].default_value = 0.55
            if sw: sw.inputs["Fac"].default_value = 1.0 if show_bg else 0.0
            if scn: scn.view_settings.exposure = 0.10
            for lo in bpy.data.objects:
                if lo.type != 'LIGHT':
                    continue
                if "f15_studio_energy" in lo.data:
                    hr = float(lo.data.get("f15_hangar_ratio", 0.25))
                    lo.data.energy = float(lo.data["f15_studio_energy"]) * hr
                elif "f15_clay_energy" in lo.data:
                    lo.data.energy = 0.0
        else:
            if mx: mx.inputs[0].default_value = 1.0
            if bg_ibl: bg_ibl.inputs["Strength"].default_value = 0.22
            if bg_cln: bg_cln.inputs["Strength"].default_value = 0.96
            if sw: sw.inputs["Fac"].default_value = 0.0
            if scn: scn.view_settings.exposure = 0.0
            for lo in bpy.data.objects:
                if lo.type != 'LIGHT':
                    continue
                if "f15_studio_energy" in lo.data:
                    lo.data.energy = 0.0
                elif "f15_clay_energy" in lo.data:
                    lo.data.energy = float(lo.data["f15_clay_energy"]) * 0.95

    fl = bpy.data.objects.get("Studio_Ground_Floor")
    if fl is not None:
        vis_fl = show_fl and (getattr(self, "f15_shade_mode", 'PBR') == 'PBR')
        fl.hide_viewport = not vis_fl
        fl.hide_render = not vis_fl

    if context and context.view_layer:
        context.view_layer.update()

def _update_seal_style(self, context):
    style = getattr(self, "f15_seal_style", 'DARK_OXIDE')
    target_name = {
        'DARK_OXIDE':     "Mat_ID22b_Nozzle_DivSealBlueMetal",
        'TEMPERED_STEEL': "Mat_ID22c_Nozzle_SealTempered",
        'BASE_ALLOY':     "Mat_ID22_Nozzle_ExternalLinkages",
    }.get(style, "Mat_ID22b_Nozzle_DivSealBlueMetal")
    target_mat = bpy.data.materials.get(target_name)
    if target_mat is None:
        return
    valid_names = {
        "Mat_ID22b_Nozzle_DivSealBlueMetal",
        "Mat_ID22c_Nozzle_SealTempered",
        "Mat_ID22_Nozzle_ExternalLinkages",
    }
    for mname in ("Mesh_DivSeal", "Mesh_ConvSeal"):
        me = bpy.data.meshes.get(mname)
        if me is None:
            continue
        for i, m in enumerate(me.materials):
            if m and m.name in valid_names:
                me.materials[i] = target_mat
    if context and context.view_layer:
        context.view_layer.update()

def _update_shroud_vis(self, context):
    show = getattr(self, "f15_show_shroud", True)
    sh_obj = bpy.data.objects.get("Static_Nozzle_Outer_Shroud")
    if sh_obj is not None:
        sh_obj.hide_viewport = not show
        sh_obj.hide_render = not show
    show_st = getattr(self, "f15_show_stand", True)
    show_tr = getattr(self, "f15_show_trailer", True)
    for o in bpy.data.objects:
        if o.name.startswith("Stand_Cradle_"):
            o.hide_viewport = not show_st
            o.hide_render = not show_st
        elif o.name.startswith("Stand_Trailer_"):
            vis_t = show_st and show_tr
            o.hide_viewport = not vis_t
            o.hide_render = not vis_t
    _sync_decal_visibility(self)
    if context and context.view_layer:
        context.view_layer.update()

class F15_OT_SetNozzlePreset(bpy.types.Operator):
    bl_idname = "f15_nozzle.set_preset"
    bl_label = "Set Nozzle Aperture Preset"
    bl_description = "Apply a predefined nozzle throat and exit area configuration"
    bl_options = {'REGISTER', 'UNDO'}

    target_open: bpy.props.FloatProperty(default=0.0, min=0.0, max=1.0)
    target_slider: bpy.props.FloatProperty(default=0.0, min=-1.0, max=1.0)

    def execute(self, context):
        scn = context.scene
        if getattr(scn, "f15_auto_demo", False):
            scn.f15_auto_demo = False
        scn.f15_nozzle_open = self.target_open
        scn.f15_slider_mode = self.target_slider
        return {'FINISHED'}

class F15_OT_SetWeatheringPreset(bpy.types.Operator):
    bl_idname = "f15_nozzle.set_weathering_preset"
    bl_label = "Set Surface Condition Preset"
    bl_description = "Apply a predefined surface weathering and oxidation level"
    bl_options = {'REGISTER', 'UNDO'}

    w_val: bpy.props.FloatProperty(default=0.95)
    h_val: bpy.props.FloatProperty(default=1.0)
    s_val: bpy.props.FloatProperty(default=1.0)
    e_val: bpy.props.FloatProperty(default=0.90)
    bd_val: bpy.props.FloatProperty(default=0.45)

    def execute(self, context):
        scn = context.scene
        scn.f15_weathering = self.w_val
        scn.f15_heat_tint = self.h_val
        scn.f15_soot = self.s_val
        scn.f15_edge_wear = self.e_val
        scn.f15_surface_erosion = self.bd_val
        return {'FINISHED'}

class F15_OT_ToggleGroundTestDemo(bpy.types.Operator):
    bl_idname = "f15_nozzle.toggle_demo"
    bl_label = "Toggle Actuation Cycle"
    bl_description = "Start or stop timeline playback of the nozzle actuation test cycle"

    def execute(self, context):
        scn = context.scene
        scn.f15_auto_demo = not scn.f15_auto_demo
        act = bpy.data.actions.get("Action_F15_GroundTest")
        if scn.f15_auto_demo:
            if act is not None:
                if scn.animation_data is None:
                    scn.animation_data_create()
                scn.animation_data.action = act
                if hasattr(scn.animation_data, "action_slot") and hasattr(act, "slots") and len(act.slots) > 0:
                    try:
                        scn.animation_data.action_slot = act.slots[0]
                    except Exception:
                        pass
            if context.screen and not context.screen.is_animation_playing:
                bpy.ops.screen.animation_play()
        else:
            if context.screen and context.screen.is_animation_playing:
                bpy.ops.screen.animation_play()
            if scn.animation_data is not None:
                scn.animation_data.action = None
        return {'FINISHED'}

class F15_PT_NozzleControlPanel(bpy.types.Panel):
    bl_label = "F100-PW-220 Control"
    bl_idname = "F15_PT_nozzle_control"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "F100 Nozzle"

    def draw(self, context):
        layout = self.layout
        scn = context.scene

        box_mat = layout.box()
        box_mat.label(text="Shading & Surface Condition", icon='MATERIAL')
        row_m = box_mat.row(align=True)
        row_m.scale_y = 1.25
        row_m.prop(scn, "f15_shade_mode", expand=True)

        if getattr(scn, "f15_shade_mode", 'PBR') == 'PBR':
            col_w = box_mat.column(align=True)
            col_w.prop(scn, "f15_weathering",      text="Service Weathering", slider=True)
            col_w.prop(scn, "f15_heat_tint",       text="Thermal Oxidation", slider=True)
            col_w.prop(scn, "f15_soot",            text="Exhaust Carbon", slider=True)
            col_w.prop(scn, "f15_edge_wear",       text="Edge Abrasion", slider=True)
            col_w.prop(scn, "f15_surface_erosion", text="Surface Erosion", slider=True)
            col_w.prop(scn, "f15_show_decals",     text="Show Stencils & Placards", toggle=True, icon='IMAGE_DATA')
            row_wp = box_mat.row(align=True)
            p_new = row_wp.operator("f15_nozzle.set_weathering_preset", text="Baseline")
            p_new.w_val, p_new.h_val, p_new.s_val, p_new.e_val, p_new.bd_val = 0.20, 0.30, 0.15, 0.12, 0.00
            p_std = row_wp.operator("f15_nozzle.set_weathering_preset", text="In-Service")
            p_std.w_val, p_std.h_val, p_std.s_val, p_std.e_val, p_std.bd_val = 0.95, 1.00, 1.00, 0.90, 0.45
            p_hvy = row_wp.operator("f15_nozzle.set_weathering_preset", text="Heavy Wear")
            p_hvy.w_val, p_hvy.h_val, p_hvy.s_val, p_hvy.e_val, p_hvy.bd_val = 1.45, 1.40, 1.45, 1.45, 1.45
            box_mat.label(text="Inner Seal Finish (DivSeal / ConvSeal)", icon='SHADING_RENDERED')
            row_sl = box_mat.row(align=True)
            row_sl.prop(scn, "f15_seal_style", expand=True)

        box_lt = layout.box()
        box_lt.label(text="Lighting & Environment", icon='WORLD')
        row_l = box_lt.row(align=True)
        row_l.prop(scn, "f15_light_preset", expand=True)
        box_lt.prop(scn, "f15_hdri_rot", text="Environment Rotation (deg)", slider=True)
        row_bg = box_lt.row(align=True)
        row_bg.prop(scn, "f15_show_hdri_bg", text="Environment Background", toggle=True)
        row_bg.prop(scn, "f15_show_floor",   text="Ground Plane", toggle=True)

        box_main = layout.box()
        box_main.label(text="Nozzle Kinematics", icon='ARMATURE_DATA')
        col = box_main.column(align=True)
        col.scale_y = 1.25
        col.prop(scn, "f15_nozzle_open", text="Nozzle Aperture (A8)", slider=True)
        col.prop(scn, "f15_slider_mode", text="Exit Area Offset (A9)", slider=True)

        box_pre = layout.box()
        box_pre.label(text="Aperture Presets", icon='PRESET')
        row = box_pre.row(align=True)
        row.scale_y = 1.20
        op0 = row.operator("f15_nozzle.set_preset", text="Closed")
        op0.target_open, op0.target_slider = 0.0, 0.0
        op1 = row.operator("f15_nozzle.set_preset", text="Intermediate")
        op1.target_open, op1.target_slider = 0.5, 0.0
        op2 = row.operator("f15_nozzle.set_preset", text="Full Open")
        op2.target_open, op2.target_slider = 1.0, 0.0

        box_rot = layout.box()
        box_rot.label(text="Rotor Assembly", icon='FORCE_VORTEX')
        box_rot.prop(scn, "f15_turbine_spin", text="Drive Rotor on Playback", toggle=True)
        box_rot.prop(scn, "f15_rotor_manual", text="Manual Rotor Angle (deg)", slider=True)

        box_demo = layout.box()
        box_demo.label(text="Actuation Cycle", icon='TIME')
        is_demo = getattr(scn, "f15_auto_demo", False)
        btn_txt = "Stop Actuation Cycle" if is_demo else "Run Actuation Cycle"
        btn_ico = 'PAUSE' if is_demo else 'PLAY'
        row_d = box_demo.row()
        row_d.scale_y = 1.25
        row_d.operator("f15_nozzle.toggle_demo", text=btn_txt, icon=btn_ico, depress=is_demo)

        box_ab = layout.box()
        box_ab.label(text="Visibility & Inspection", icon='LIGHT_SUN')
        box_ab.prop(scn, "f15_show_shroud", text="Outer Shroud", toggle=True, icon='MOD_SOLIDIFY')
        box_ab.prop(scn, "f15_show_stand", text="Maintenance Stand", toggle=True, icon='OBJECT_DATAMODE')
        row_tr = box_ab.row(align=True)
        row_tr.enabled = getattr(scn, "f15_show_stand", True)
        row_tr.prop(scn, "f15_show_trailer", text="Transport Trailer", toggle=True, icon='AUTO')
        box_ab.prop(scn, "f15_ab_glow", text="Internal Inspection Light", toggle=True, icon='SHADING_SOLID')
        sub = box_ab.row(align=True)
        sub.enabled = getattr(scn, "f15_ab_glow", False)
        sub.prop(scn, "f15_ab_glow_blue", text="Cool Inspection Tint", toggle=True)

_CLASSES = (
    F15_OT_SetNozzlePreset,
    F15_OT_SetWeatheringPreset,
    F15_OT_ToggleGroundTestDemo,
    F15_PT_NozzleControlPanel,
)

def register_f15_ui():
    bpy.types.Scene.f15_shade_mode = bpy.props.EnumProperty(
        name="Shading Mode",
        description="Select viewport and render shading mode",
        items=[
            ('PBR',      "PBR",          "Procedural metallic, coated, ceramic, and elastomer shading"),
            ('COLOR_ID', "Component ID", "Color-coded shading by component group"),
            ('CLAY',     "Clay",         "Neutral matte clay shading for geometry inspection"),
        ],
        default='PBR',
        update=_update_shader_and_lookdev,
    )
    bpy.types.Scene.f15_weathering = bpy.props.FloatProperty(
        name="Service Weathering",
        description="Global surface grime and operational aging intensity",
        default=1.0, min=0.0, max=1.8, precision=2,
        update=_update_shader_and_lookdev,
    )
    bpy.types.Scene.f15_heat_tint = bpy.props.FloatProperty(
        name="Thermal Oxidation",
        description="High-temperature discoloration and oxide tint intensity",
        default=1.0, min=0.0, max=1.8, precision=2,
        update=_update_shader_and_lookdev,
    )
    bpy.types.Scene.f15_soot = bpy.props.FloatProperty(
        name="Exhaust Carbon",
        description="Internal and aft carbon deposit intensity",
        default=1.0, min=0.0, max=1.8, precision=2,
        update=_update_shader_and_lookdev,
    )
    bpy.types.Scene.f15_edge_wear = bpy.props.FloatProperty(
        name="Edge Abrasion",
        description="Geometric edge chipping and metallic wear intensity",
        default=0.90, min=0.0, max=1.8, precision=2,
        update=_update_shader_and_lookdev,
    )
    bpy.types.Scene.f15_surface_erosion = bpy.props.FloatProperty(
        name="Surface Erosion",
        description="Localized scratches, corrosion, oil residue, and coating loss intensity",
        default=0.45, min=0.0, max=1.8, precision=2,
        update=_update_shader_and_lookdev,
    )
    bpy.types.Scene.f15_show_decals = bpy.props.BoolProperty(
        name="Show Stencils & Placards",
        description="Toggle conformal technical placards and markings",
        default=True,
        update=_update_shader_and_lookdev,
    )
    bpy.types.Scene.f15_seal_style = bpy.props.EnumProperty(
        name="Inner Seal Finish",
        description="Material finish assigned to inner convergent and divergent seal flaps",
        items=[
            ('DARK_OXIDE',     "Dark Oxide", "Low-saturation dark thermally oxidized alloy finish"),
            ('TEMPERED_STEEL', "Tempered",   "Moderate heat-tinted steel finish"),
            ('BASE_ALLOY',     "Base Alloy", "Neutral grey metallic finish matching linkage hardware"),
        ],
        default='DARK_OXIDE',
        update=_update_seal_style,
    )
    bpy.types.Scene.f15_light_preset = bpy.props.EnumProperty(
        name="Lighting Preset",
        description="Select lighting rig and environment illumination profile",
        items=[
            ('STUDIO',      "Studio",     "Controlled multi-panel studio area lighting with neutral backdrop"),
            ('HANGAR',      "Workshop",   "4K workshop environment HDRI with overhead fill lighting"),
            ('DARK_STUDIO', "Inspection", "Neutral area lighting configured for clay and component ID inspection"),
        ],
        default='STUDIO',
        update=_update_lighting_preset,
    )
    bpy.types.Scene.f15_hdri_rot = bpy.props.FloatProperty(
        name="Environment Rotation",
        description="Azimuth rotation angle of the environment HDRI in degrees",
        default=165.0, min=0.0, max=360.0, precision=1,
        update=_update_lighting_preset,
    )
    bpy.types.Scene.f15_show_hdri_bg = bpy.props.BoolProperty(
        name="Environment Background",
        description="Display the environment HDRI directly to camera rays",
        default=False, update=_update_lighting_preset,
    )
    bpy.types.Scene.f15_show_floor = bpy.props.BoolProperty(
        name="Ground Plane",
        description="Toggle visibility of the workshop ground plane in PBR mode",
        default=True, update=_update_lighting_preset,
    )
    bpy.types.Scene.f15_color_id_mode = bpy.props.BoolProperty(
        name="Component ID Shading", default=False, update=_update_shader_and_lookdev,
    )
    bpy.types.Scene.f15_nozzle_open = bpy.props.FloatProperty(
        name="Nozzle Aperture (A8)",
        description="Continuous throat aperture control from minimum (0.0) to maximum (1.0)",
        default=0.0, min=0.0, max=1.0, precision=3, subtype='FACTOR',
        update=_update_nozzle_props,
    )
    bpy.types.Scene.f15_slider_mode = bpy.props.FloatProperty(
        name="Exit Area Offset (A9)",
        description="Secondary carriage offset along the divergent guide rails",
        default=0.0, min=-1.0, max=1.0, precision=3,
        update=_update_nozzle_props,
    )
    bpy.types.Scene.f15_turbine_spin = bpy.props.BoolProperty(
        name="Drive Rotor on Playback",
        description="Rotate the low-pressure turbine rotor and spline shaft during timeline playback",
        default=True, update=_update_nozzle_props,
    )
    bpy.types.Scene.f15_rotor_manual = bpy.props.FloatProperty(
        name="Manual Rotor Angle",
        description="Manual angular offset for static rotor inspection (0 to 360 deg)",
        default=0.0, min=0.0, max=360.0, precision=1,
        update=_update_nozzle_props,
    )
    bpy.types.Scene.f15_auto_demo = bpy.props.BoolProperty(
        name="Actuation Cycle Active", default=False,
    )
    bpy.types.Scene.f15_show_shroud = bpy.props.BoolProperty(
        name="Outer Shroud",
        description="Toggle visibility of the outer fairing shroud",
        default=True, update=_update_shroud_vis,
    )
    bpy.types.Scene.f15_show_stand = bpy.props.BoolProperty(
        name="Maintenance Stand",
        description="Toggle visibility of the rollover maintenance stand and guide rails",
        default=True, update=_update_shroud_vis,
    )
    bpy.types.Scene.f15_show_trailer = bpy.props.BoolProperty(
        name="Transport Trailer",
        description="Toggle visibility of the four-wheel steerable transport chassis",
        default=True, update=_update_shroud_vis,
    )
    bpy.types.Scene.f15_ab_glow = bpy.props.BoolProperty(
        name="Internal Inspection Light",
        description="Toggle internal cavity illumination for interior inspection",
        default=True, update=_update_nozzle_props,
    )
    bpy.types.Scene.f15_ab_glow_blue = bpy.props.BoolProperty(
        name="Cool Inspection Tint",
        description="Switch internal illumination from neutral warm to cool blue",
        default=False, update=_update_nozzle_props,
    )
    for cls in _CLASSES:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
        bpy.utils.register_class(cls)

if __name__ == "__main__":
    register_f15_ui()
'''


def install_enhanced_ui_panel():
    txt_name = "f15_nozzle_ui.py"
    if txt_name in bpy.data.texts:
        bpy.data.texts.remove(bpy.data.texts[txt_name])
    txt = bpy.data.texts.new(txt_name)
    txt.write(ENHANCED_UI_PANEL_SCRIPT_SOURCE)
    txt.use_module = True

    ns = {}
    exec(compile(ENHANCED_UI_PANEL_SCRIPT_SOURCE, txt_name, 'exec'), ns)
    ns["register_f15_ui"]()

    scn = bpy.context.scene
    scn.f15_color_id_mode = False
    scn.f15_shade_mode = 'PBR'
    scn.f15_weathering = 0.95
    scn.f15_heat_tint = 1.0
    scn.f15_soot = 1.0
    scn.f15_edge_wear = 0.90
    scn.f15_surface_erosion = 0.45
    scn.f15_show_decals = True
    scn.f15_seal_style = 'DARK_OXIDE'
    scn.f15_light_preset = 'STUDIO'
    scn.f15_hdri_rot = 165.0
    scn.f15_show_hdri_bg = False
    scn.f15_show_floor = True


    m_glow = bpy.data.materials.get("Mat_Clay_AB_Accent")
    if m_glow and m_glow.node_tree:
        glow_bsdf = m_glow.node_tree.nodes.get("AB_Glow_BSDF")
        if glow_bsdf and "Emission Strength" in glow_bsdf.inputs:
            try:
                glow_bsdf.inputs["Emission Strength"].driver_remove("default_value")
            except Exception:
                pass
            fc_em = glow_bsdf.inputs["Emission Strength"].driver_add("default_value")
            dem = fc_em.driver
            dem.type = 'SCRIPTED'
            vg = dem.variables.new()
            vg.name = "glow_on"
            vg.type = 'SINGLE_PROP'
            vg.targets[0].id_type = 'SCENE'
            vg.targets[0].id = scn
            vg.targets[0].data_path = "f15_ab_glow"
            dem.expression = "glow_on * 0.20"


    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for sp in area.spaces:
                    if sp.type == 'VIEW_3D':
                        sp.shading.color_type = 'MATERIAL'


def apply_procedural_pbr_to_current_scene(save_blend_path: str = None, hdri_dir: str = None):

    ng_global, ng_masks, ng_wrap = build_all_f100_procedural_materials()
    from f100_service_detailing import build_service_details
    build_service_details(ng_global, ng_masks, ng_wrap)
    bake_all_mesh_geometry_attributes()
    refine_existing_special_face_materials(ng_global, ng_masks, ng_wrap)
    build_conformal_aviation_decals(ng_global)
    setup_pbr_world_hdri_and_studio_floor(hdri_dir=hdri_dir)
    install_enhanced_ui_panel()

    bpy.context.scene.f15_nozzle_open = 0.22
    bpy.context.scene.f15_slider_mode = 0.0
    bpy.context.scene.f15_ab_glow = True
    bpy.context.view_layer.update()

    if save_blend_path:
        bpy.ops.wm.save_as_mainfile(filepath=save_blend_path)
        print(f"[Saved] {save_blend_path}")


if __name__ == "__main__":
    blend_file = os.path.join(_THIS_DIR, "f15_exhaust_nozzle.blend")
    save_path = None if "--no-save" in sys.argv else blend_file
    apply_procedural_pbr_to_current_scene(save_blend_path=save_path)
