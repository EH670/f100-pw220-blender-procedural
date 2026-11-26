import math
from collections import deque
from dataclasses import dataclass
import bpy
import bmesh
from mathutils import Vector, Matrix, Euler


def srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def rgba(hex_rgb: int, alpha: float = 1.0):
    r = ((hex_rgb >> 16) & 0xFF) / 255.0
    g = ((hex_rgb >> 8) & 0xFF) / 255.0
    b = (hex_rgb & 0xFF) / 255.0
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), alpha)


def circle_intersect_2d(p1, r1: float, p2, r2: float, side: int = 1):
    dy = p2[0] - p1[0]
    dr = p2[1] - p1[1]
    d = math.hypot(dy, dr)
    if d <= 1e-9 or d >= (r1 + r2) or d <= abs(r1 - r2):
        raise ValueError(f"Four-bar linkage geometry unreachable: d={d:.5f}, r1={r1:.5f}, r2={r2:.5f}")
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h = math.sqrt(max(0.0, r1 * r1 - a * a))
    my = p1[0] + a * dy / d
    mr = p1[1] + a * dr / d
    return (my - side * h * dr / d, mr + side * h * dy / d)


@dataclass
class NozzleSpec:
    n_sectors: int = 16
    n_actuators: int = 8

    y_case_fwd: float = -0.520
    y_case_aft: float = 0.015
    r_liner_base: float = 0.520
    r_case_body: float = 0.546
    r_case_flange: float = 0.634

    p0_y: float = 0.000
    p0_r: float = 0.545
    seal_radial_offset: float = 0.0046
    L_conv: float = 0.410
    L_div: float = 0.545

    conv_lug_a: float = 0.248
    conv_lug_h: float = 0.036
    div_lug_a: float = 0.228
    div_lug_h: float = 0.038
    slider_auto_stroke: float = 0.032
    slider_mode_stroke: float = 0.016

    fork_half_span: float = 0.023

    pb_y: float = -0.038
    pb_r: float = 0.642
    r_bell_in: float = 0.092
    r_bell_out: float = 0.115
    bell_spread_deg: float = 118.0
    r_sync_ring: float = 0.575
    L_short: float = 0.165
    L_pull: float = 0.264
    y_ring_closed: float = -0.2205
    ring_stroke: float = 0.0445

    ps_y: float = 0.018
    ps_r: float = 0.660

    def __post_init__(self):
        self.r0c = math.hypot(self.conv_lug_a, self.conv_lug_h)
        self.alpha_conv = math.atan2(self.conv_lug_h, self.conv_lug_a)
        self.r1d = math.hypot(self.div_lug_a, self.div_lug_h)
        self.alpha_div = math.atan2(self.div_lug_h, self.div_lug_a)
        self.bell_spread = math.radians(self.bell_spread_deg)

        def _pd_at(th1_d: float, th2_d: float, a_val: float):
            t1, t2 = math.radians(th1_d), math.radians(th2_d)
            r1d_v = math.hypot(a_val, self.div_lug_h)
            alp_v = math.atan2(self.div_lug_h, a_val)
            p1 = (self.p0_y + self.L_conv * math.cos(t1), self.p0_r + self.L_conv * math.sin(t1))
            return (p1[0] + r1d_v * math.cos(t2 + alp_v),
                    p1[1] + r1d_v * math.sin(t2 + alp_v))

        pd0 = _pd_at(-26.0, -2.5, self.div_lug_a)
        self.L_strut = math.hypot(pd0[0] - self.ps_y, pd0[1] - self.ps_r)

        self.state_closed = self.solve_planar(0.0, 0.0)
        self.state_open = self.solve_planar(1.0, 0.0)

    def solve_planar(self, u: float, slider_mode: float = 0.0) -> dict:
        u_c = max(0.0, min(1.0, u))
        m_c = max(-1.0, min(1.0, slider_mode))
        m_eff = m_c if m_c >= 0.0 else (m_c * min(1.0, u_c / 0.28))
        delta_a = u_c * self.slider_auto_stroke + m_eff * self.slider_mode_stroke
        a_cur = self.div_lug_a + delta_a
        r1d_cur = math.hypot(a_cur, self.div_lug_h)
        alpha_div_cur = math.atan2(self.div_lug_h, a_cur)

        y_ring = self.y_ring_closed + u_c * self.ring_stroke
        p_ring = (y_ring, self.r_sync_ring)
        pb = (self.pb_y, self.pb_r)
        p0 = (self.p0_y, self.p0_r)
        ps = (self.ps_y, self.ps_r)

        a_in = circle_intersect_2d(pb, self.r_bell_in, p_ring, self.L_short, side=+1)
        phi_in = math.atan2(a_in[1] - pb[1], a_in[0] - pb[0])
        phi_out = phi_in + self.bell_spread
        a_out = (pb[0] + self.r_bell_out * math.cos(phi_out),
                 pb[1] + self.r_bell_out * math.sin(phi_out))

        pc = circle_intersect_2d(p0, self.r0c, a_out, self.L_pull, side=-1)
        theta1 = math.atan2(pc[1] - p0[1], pc[0] - p0[0]) - self.alpha_conv
        p1 = (p0[0] + self.L_conv * math.cos(theta1),
              p0[1] + self.L_conv * math.sin(theta1))

        pd = circle_intersect_2d(p1, r1d_cur, ps, self.L_strut, side=-1)
        theta2 = math.atan2(pd[1] - p1[1], pd[0] - p1[0]) - alpha_div_cur
        p2 = (p1[0] + self.L_div * math.cos(theta2),
              p1[1] + self.L_div * math.sin(theta2))

        return {
            "u": u_c, "slider_mode": m_c, "delta_a": delta_a, "a_cur": a_cur,
            "P_ring": p_ring, "PB": pb, "A_in": a_in, "A_out": a_out,
            "P0": p0, "PC": pc, "P1": p1, "PD": pd, "P2": p2, "PS": ps,
            "theta1": theta1, "theta2": theta2,
        }


MAT_TI_CASE         = 1
MAT_INCONEL_CASE    = 2
MAT_WAFFLE_DUCT     = 3
MAT_SPINNER_IGV     = 4
MAT_ROTARY_BLADES   = 5
MAT_MACHINED_ALU    = 6
MAT_AVIONICS_BOX    = 7
MAT_PUMPS_OILTANK   = 8
MAT_RIGID_TUBE      = 9
MAT_BRAID_HOSE      = 10
MAT_YELLOW_HARNESS  = 11
MAT_BRASS_FITTING   = 12
MAT_ADEL_CUSHION    = 13
MAT_CONNECTOR_38999 = 14
MAT_FASTENER        = 15
MAT_THREAD          = 16
MAT_LOCKWIRE        = 17
MAT_PLATE_BASE      = 18
MAT_3D_TEXT         = 19
MAT_SHROUD_GOLD     = 20
MAT_SHROUD_GREY     = 21
MAT_NOZ_EXT_LINK    = 22
MAT_NOZ_FLAP_BODY   = 23
MAT_NOZ_ACT_AB      = 24
MAT_HOTGAS_TILE     = 25
MAT_STAND_ORANGE    = 26
MAT_STAND_STEEL     = 27
MAT_TRAILER_YELLOW  = 28
MAT_RUBBER_DARK     = 29

COLOR_ID_SPECS = {
    MAT_TI_CASE:         ("Mat_ID01_Ti_FanAndCoreCase",        0x6C8EAD),
    MAT_INCONEL_CASE:    ("Mat_ID02_Inconel_CombustorAndTEC",  0x8D7AA8),
    MAT_WAFFLE_DUCT:     ("Mat_ID03_TiWaffle_AugmentorCase",   0x4F969C),
    MAT_SPINNER_IGV:     ("Mat_ID04_InletSpinner_And_Stators", 0x7BAFD4),
    MAT_ROTARY_BLADES:   ("Mat_ID05_Rotary_FanAndTurbine",     0x3B689B),
    MAT_MACHINED_ALU:    ("Mat_ID06_Machined_MountsAndTracks", 0xB4C7D2),
    MAT_AVIONICS_BOX:    ("Mat_ID07_Avionics_ControlBoxes",    0x3B4351),
    MAT_PUMPS_OILTANK:   ("Mat_ID08_Cast_AGB_Pumps_OilTank",   0x6D9C62),
    MAT_RIGID_TUBE:      ("Mat_ID09_Stainless_RigidTubing",    0xDF8A46),
    MAT_BRAID_HOSE:      ("Mat_ID10_Braided_FlexHoses",        0xCA5246),
    MAT_YELLOW_HARNESS:  ("Mat_ID11_HighTemp_WireHarnesses",   0xE5BA38),
    MAT_BRASS_FITTING:   ("Mat_ID12_BNuts_Ferrules_Fittings",  0xD49B35),
    MAT_ADEL_CUSHION:    ("Mat_ID13_Adel_Fluorosilicone_Blue", 0x2563EB),
    MAT_CONNECTOR_38999: ("Mat_ID14_MIL38999_And_KnurledKnobs",0x8A9D3F),
    MAT_FASTENER:        ("Mat_ID15_Aerospace_BoltsAndRivets", 0xDCD4E8),
    MAT_THREAD:          ("Mat_ID16_Helical_ExposedThreads",   0xE26464),
    MAT_LOCKWIRE:        ("Mat_ID17_Safety_Lockwire_Lanyards", 0xFFE066),
    MAT_PLATE_BASE:      ("Mat_ID18_DataPlate_Backing",        0x263849),
    MAT_3D_TEXT:         ("Mat_ID19_3D_Typography_And_Scales", 0xFAF0CA),
    MAT_SHROUD_GOLD:     ("Mat_ID20_OuterShroud_TiGold",       0xC49A45),
    MAT_SHROUD_GREY:     ("Mat_ID21_OuterShroud_BellyBand",    0x50565E),
    MAT_NOZ_EXT_LINK:    ("Mat_ID22_Nozzle_ExternalLinkages",  0x468896),
    MAT_NOZ_FLAP_BODY:   ("Mat_ID23_Nozzle_PrimaryFlaps",      0x79A6B8),
    MAT_NOZ_ACT_AB:      ("Mat_ID24_Actuators_Sync_Augmentor", 0xB97A9C),
    MAT_HOTGAS_TILE:     ("Mat_ID25_Ceramic_RefractoryTiles",  0xDEC0CB),
    MAT_STAND_ORANGE:    ("Mat_ID26_Stand_M4_SafetyOrange",    0xE36B37),
    MAT_STAND_STEEL:     ("Mat_ID27_Stand_TK9_SteelRails",     0x8DA9DC),
    MAT_TRAILER_YELLOW:  ("Mat_ID28_Trailer_M3000_GSEYellow",  0xE8B436),
    MAT_RUBBER_DARK:     ("Mat_ID29_Rubber_Tires_And_Chocks",  0x2B2F36),
}


def get_or_create_color_id_mat(mid: int):

    m_name, hex_rgb = COLOR_ID_SPECS.get(mid, COLOR_ID_SPECS[MAT_TI_CASE])
    existing = bpy.data.materials.get(m_name)
    if existing is not None:
        return existing

    mat = bpy.data.materials.new(m_name)
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.52
    for spec_name in ("Specular IOR Level", "Specular"):
        if spec_name in bsdf.inputs:
            bsdf.inputs[spec_name].default_value = 0.38
            break


    mix_mode = nt.nodes.new("ShaderNodeMix")
    mix_mode.name = "ColorID_Mode_Mix"
    mix_mode.data_type = 'RGBA'
    mix_mode.blend_type = 'MIX'
    mix_mode.inputs[0].default_value = 1.0
    mix_mode.inputs[6].default_value = rgba(0xC2C6CC)
    mix_mode.inputs[7].default_value = rgba(hex_rgb)

    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 16
    ao.inputs["Distance"].default_value = 0.055
    ao.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    ramp_ao = nt.nodes.new("ShaderNodeValToRGB")
    ramp_ao.color_ramp.elements[0].position = 0.12
    ramp_ao.color_ramp.elements[0].color = rgba(0x525760)
    ramp_ao.color_ramp.elements[1].position = 0.90
    ramp_ao.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    mix_ao = nt.nodes.new("ShaderNodeMix")
    mix_ao.data_type = 'RGBA'
    mix_ao.blend_type = 'MULTIPLY'
    mix_ao.inputs[0].default_value = 1.0

    bev = nt.nodes.new("ShaderNodeBevel")
    bev.samples = 6
    bev.inputs["Radius"].default_value = 0.00032

    nt.links.new(ao.outputs["AO"], ramp_ao.inputs["Fac"])
    nt.links.new(mix_mode.outputs[2], mix_ao.inputs[6])
    nt.links.new(ramp_ao.outputs["Color"], mix_ao.inputs[7])
    nt.links.new(mix_ao.outputs[2], bsdf.inputs["Base Color"])
    nt.links.new(bev.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    mat["f15_id_hex"] = int(hex_rgb)
    mat.diffuse_color = rgba(hex_rgb)
    mat.roughness = 0.52
    return mat


def _infer_base_mat_id_by_name(name: str) -> int:

    exact_map = {
        "Fwd_01_Inlet_Spinner_Pitot_And_Sensors": MAT_SPINNER_IGV,
        "Fwd_02_Fan_Case_VaneLinkage_And_DEEC": MAT_TI_CASE,
        "Fwd_03_Core_Isogrid_Duct_And_Splitters": MAT_TI_CASE,
        "Fwd_04_Core_Left_SBleed_ServoValve_And_Tubes": MAT_NOZ_ACT_AB,
        "Fwd_05_Core_Right_TwinUBleed_RCVV_And_Valves": MAT_NOZ_ACT_AB,
        "Fwd_06_Right_Stamped_Saddle_OilTank": MAT_PUMPS_OILTANK,
        "Fwd_07_Underslung_AGB_FCOC_And_PumpCluster": MAT_PUMPS_OILTANK,
        "Fwd_08_Combustor_Turbine_Case_Fuel_And_Igniters": MAT_INCONEL_CASE,
        "Fwd_09_Ventral_AugmentorFuel_Metering_And_BellyTrunks": MAT_PUMPS_OILTANK,
        "Fwd_10_Dorsal_BleedManifolds_ACC_And_CoolingRings": MAT_RIGID_TUBE,
        "Static_Forward_Waffle_Duct": MAT_WAFFLE_DUCT,
        "Static_Base_Casing_Ribbed": MAT_WAFFLE_DUCT,
        "Static_Aerospace_Bolts_And_Rivets": MAT_FASTENER,
        "Static_Stanchion_Trunnion_Assemblies": MAT_MACHINED_ALU,
        "Static_Hydraulic_System_Detailed": MAT_NOZ_ACT_AB,
        "Static_Sync_FlexShaft_System": MAT_MACHINED_ALU,
        "Static_Asymmetric_Plumbing_And_Sensors": MAT_RIGID_TUBE,
        "Static_Maintenance_Hoists_And_DataPlate": MAT_MACHINED_ALU,
        "Static_Nozzle_Outer_Shroud": MAT_SHROUD_GOLD,
        "Static_Forward_TEC_And_Mount": MAT_INCONEL_CASE,
        "Static_Balance_Chamber_Manifold": MAT_RIGID_TUBE,
        "Interior_Louvered_Corrugated_Liner": MAT_NOZ_ACT_AB,
        "Interior_VGutter_Flameholders_Detailed": MAT_INCONEL_CASE,
        "Interior_Stator_TailCone_And_GuideVanes": MAT_INCONEL_CASE,
        "Mesh_LPT_Rotor_Shaft": MAT_ROTARY_BLADES,
        "Mesh_Fan_Rotor_3Stage": MAT_ROTARY_BLADES,
        "Mesh_ConvMaster": MAT_NOZ_FLAP_BODY,
        "Mesh_ConvSeal": MAT_NOZ_EXT_LINK,
        "Mesh_SliderCarriage": MAT_MACHINED_ALU,
        "Mesh_SliderRollerPair": MAT_STAND_STEEL,
        "Mesh_DivMaster": MAT_NOZ_FLAP_BODY,
        "Mesh_DivSeal": MAT_NOZ_EXT_LINK,
        "Mesh_RingKeeperSaddle": MAT_MACHINED_ALU,
        "Mesh_Bellcrank": MAT_MACHINED_ALU,
        "Mesh_ShortLink": MAT_MACHINED_ALU,
        "Mesh_TwinFork_PullRod": MAT_NOZ_EXT_LINK,
        "Mesh_Diagonal_Strut": MAT_NOZ_EXT_LINK,
        "Mesh_Boomerang": MAT_MACHINED_ALU,
        "Mesh_TransRodA": MAT_RIGID_TUBE,
        "Mesh_TransRodB": MAT_RIGID_TUBE,
        "Mesh_BoomFwdRod": MAT_NOZ_ACT_AB,
        "Mesh_BoomGuideCyl": MAT_MACHINED_ALU,
        "Mesh_RingStrap_R": MAT_MACHINED_ALU,
        "Mesh_RingStrap_L": MAT_MACHINED_ALU,
        "Mesh_BalanceFlap": MAT_NOZ_EXT_LINK,
        "Mesh_BalanceRollerPair": MAT_STAND_STEEL,
        "Mesh_SyncRoller": MAT_STAND_STEEL,
        "Mesh_SyncRing_Pistons": MAT_NOZ_ACT_AB,
        "Stand_Cradle_01_Longitudinal_Rails_And_Crossbeams": MAT_STAND_STEEL,
        "Stand_Cradle_02_Pedestals_Rollers_And_Handwheels": MAT_STAND_ORANGE,
        "Stand_Cradle_03_RollOver_Arches_And_MountPickups": MAT_STAND_ORANGE,
        "Stand_Cradle_04_TopTieBar_And_DiagonalBraces": MAT_STAND_ORANGE,
        "Stand_Cradle_05_RBF_Streamers": MAT_STAND_ORANGE,
        "Stand_Trailer_05_M3000_Steerable_Axles_And_TowBar": MAT_TRAILER_YELLOW,
        "Stand_Trailer_06_Pneumatic_GroundSupport_Wheels": MAT_TRAILER_YELLOW,
        "Stand_Trailer_07_Deployed_GSE_Props": MAT_RUBBER_DARK,
    }
    return exact_map.get(name, MAT_TI_CASE)


def bm_fill_unset_from(bm: bmesh.types.BMesh, f_start: int, mat_id: int):

    bm.faces.ensure_lookup_table()
    for i in range(f_start, len(bm.faces)):
        f = bm.faces[i]
        if f.material_index == 0:
            f.material_index = mat_id


def bm_fill_unset_mat(bm: bmesh.types.BMesh, mat_id: int):

    for f in bm.faces:
        if f.material_index == 0:
            f.material_index = mat_id


def build_clay_materials() -> dict:

    m_glow = bpy.data.materials.new("Mat_Clay_AB_Accent")
    ntg = m_glow.node_tree
    ntg.nodes.clear()
    out_g = ntg.nodes.new("ShaderNodeOutputMaterial")
    bsdf_g = ntg.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf_g.name = "AB_Glow_BSDF"
    mix_g = ntg.nodes.new("ShaderNodeMix")
    mix_g.name = "ColorID_Mode_Mix"
    mix_g.data_type = 'RGBA'
    mix_g.inputs[0].default_value = 1.0
    mix_g.inputs[6].default_value = rgba(0xD8DCE2)
    mix_g.inputs[7].default_value = rgba(0xFFE066)
    bsdf_g.inputs["Roughness"].default_value = 0.50
    for em_c in ("Emission Color", "Emission"):
        if em_c in bsdf_g.inputs:
            bsdf_g.inputs[em_c].default_value = rgba(0xFFF3E2)
            break
    if "Emission Strength" in bsdf_g.inputs:
        bsdf_g.inputs["Emission Strength"].default_value = 0.0
    ntg.links.new(mix_g.outputs[2], bsdf_g.inputs["Base Color"])
    ntg.links.new(bsdf_g.outputs["BSDF"], out_g.inputs["Surface"])
    m_glow["f15_id_hex"] = int(0xFFE066)
    m_glow.diffuse_color = rgba(0xFFE066)

    return {
        "clay": "AUTO_COLOR_ID",
        "ab_glow_ring": m_glow,
    }


def bm_merge(dst_bm: bmesh.types.BMesh, src_bm: bmesh.types.BMesh, transform: Matrix = None, default_mat_id: int = 0):
    if default_mat_id > 0:
        bm_fill_unset_mat(src_bm, default_mat_id)
    if transform is not None:
        bmesh.ops.transform(src_bm, matrix=transform, verts=src_bm.verts)
        if transform.determinant() < 0.0 and len(src_bm.faces) > 0:
            bmesh.ops.reverse_faces(src_bm, faces=src_bm.faces)
    me_t = bpy.data.meshes.new("_tmp_merge")
    src_bm.to_mesh(me_t)
    src_bm.free()
    dst_bm.from_mesh(me_t)
    bpy.data.meshes.remove(me_t)


def bm_ensure_outward_shell_normals(bm: bmesh.types.BMesh):

    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    n_f = len(bm.faces)
    if n_f == 0:
        return
    visited = bytearray(n_f)
    for f0 in bm.faces:
        if visited[f0.index]:
            continue
        shell_faces = []
        q = deque([f0])
        visited[f0.index] = 1
        c0 = f0.verts[0].co
        vol6 = 0.0
        has_open = False
        while q:
            fc = q.popleft()
            shell_faces.append(fc)
            vs = fc.verts
            if len(vs) >= 3:
                a = vs[0].co - c0
                b = vs[1].co - c0
                c = vs[2].co - c0
                vol6 += a.dot(b.cross(c))
                if len(vs) == 4:
                    d = vs[3].co - c0
                    vol6 += a.dot(c.cross(d))
            for ed in fc.edges:
                lf = ed.link_faces
                if len(lf) < 2:
                    has_open = True
                for fn in lf:
                    if not visited[fn.index]:
                        visited[fn.index] = 1
                        q.append(fn)
        if (not has_open) and vol6 < -1e-15:
            for sf in shell_faces:
                sf.normal_flip()


def bm_apply_hard_surface_smoothing(bm: bmesh.types.BMesh, angle_deg: float = 36.5):

    bm.normal_update()
    cos_thresh = math.cos(math.radians(angle_deg))
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        if len(e.link_faces) == 2:
            n0 = e.link_faces[0].normal
            n1 = e.link_faces[1].normal
            if n0.length_squared > 1e-12 and n1.length_squared > 1e-12:
                e.smooth = (n0.dot(n1) >= cos_thresh)
            else:
                e.smooth = False
        else:
            e.smooth = False


def attach_weighted_normal_modifier(obj):

    mod = obj.modifiers.new(name="HardSurface_WeightedNormal", type='WEIGHTED_NORMAL')
    mod.mode = 'FACE_AREA_WITH_ANGLE'
    mod.weight = 85
    mod.keep_sharp = True
    return mod


def _assign_compact_color_id_slots(name: str, bm: bmesh.types.BMesh, mesh, material, base_mat_id: int = None):

    if material is not None and material != "AUTO_COLOR_ID" and hasattr(material, "name"):
        for f in bm.faces:
            f.material_index = 0
        bm.to_mesh(mesh)
        bm.free()
        mesh.materials.append(material)
        return

    fallback_id = base_mat_id if (base_mat_id is not None and base_mat_id > 0) else _infer_base_mat_id_by_name(name)
    used_set = set()
    for f in bm.faces:
        mid = f.material_index if f.material_index in COLOR_ID_SPECS else fallback_id
        f.material_index = mid
        used_set.add(mid)
    used_mids = sorted(used_set) if used_set else [fallback_id]
    slot_map = {mid: idx for idx, mid in enumerate(used_mids)}
    for f in bm.faces:
        f.material_index = slot_map[f.material_index]
    bm.to_mesh(mesh)
    bm.free()
    for mid in used_mids:
        mesh.materials.append(get_or_create_color_id_mat(mid))


def create_mesh_obj(name: str, bm: bmesh.types.BMesh, collection, material, smooth=True, base_mat_id: int = None):
    loose_vs = [v for v in bm.verts if len(v.link_edges) == 0]
    if loose_vs:
        bmesh.ops.delete(bm, geom=loose_vs, context='VERTS')
    bm_ensure_outward_shell_normals(bm)
    mesh = bpy.data.meshes.new(name)
    if smooth:
        bm_apply_hard_surface_smoothing(bm, angle_deg=36.5)
    else:
        bm.normal_update()
    _assign_compact_color_id_slots(name, bm, mesh, material, base_mat_id=base_mat_id)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    if smooth:
        attach_weighted_normal_modifier(obj)
    return obj


def finalize_shared_mesh(name: str, bm: bmesh.types.BMesh, material, smooth: bool = True, base_mat_id: int = None):
    loose_vs = [v for v in bm.verts if len(v.link_edges) == 0]
    if loose_vs:
        bmesh.ops.delete(bm, geom=loose_vs, context='VERTS')
    bm_ensure_outward_shell_normals(bm)
    mesh = bpy.data.meshes.new(name)
    if smooth:
        bm_apply_hard_surface_smoothing(bm, angle_deg=36.5)
    else:
        bm.normal_update()
    _assign_compact_color_id_slots(name, bm, mesh, material, base_mat_id=base_mat_id)
    return mesh


def _bm_cap_ring_quads(bm: bmesh.types.BMesh, ring: list, reverse: bool = False, inset_frac: float = 0.76):

    n = len(ring)
    if n < 4:
        return
    ctr = Vector((0.0, 0.0, 0.0))
    for v in ring:
        ctr += v.co
    ctr /= float(n)

    r_in = [bm.verts.new(ctr.lerp(v.co, inset_frac)) for v in ring]
    for s in range(n):
        sn = (s + 1) % n
        try:
            if reverse:
                bm.faces.new((ring[sn], ring[s], r_in[s], r_in[sn]))
            else:
                bm.faces.new((ring[s], ring[sn], r_in[sn], r_in[s]))
        except ValueError:
            pass

    if n % 2 == 0:
        k_half = n // 2
        for m in range(k_half - 1):
            try:
                if reverse:
                    bm.faces.new((r_in[n - 1 - m], r_in[n - 2 - m], r_in[m + 1], r_in[m]))
                else:
                    bm.faces.new((r_in[m], r_in[m + 1], r_in[n - 2 - m], r_in[n - 1 - m]))
            except ValueError:
                pass
    else:
        try:
            bm.faces.new(list(reversed(r_in)) if reverse else r_in)
        except ValueError:
            pass


def bevel_2d_profile(profile_pts: list, bevel_r: float = 0.00045, close_loop: bool = True) -> list:

    m = len(profile_pts)
    if m < 3 or bevel_r <= 1e-6:
        return list(profile_pts)
    out = []
    for i in range(m):
        if not close_loop and (i == 0 or i == m - 1):
            out.append(profile_pts[i])
            continue
        p_prev = profile_pts[(i - 1) % m]
        p_cur  = profile_pts[i]
        p_next = profile_pts[(i + 1) % m]
        vin_x, vin_y = p_prev[0] - p_cur[0], p_prev[1] - p_cur[1]
        vout_x, vout_y = p_next[0] - p_cur[0], p_next[1] - p_cur[1]
        l_in = math.hypot(vin_x, vin_y)
        l_out = math.hypot(vout_x, vout_y)
        if l_in < 1e-6 or l_out < 1e-6:
            out.append(p_cur)
            continue
        uin_x, uin_y = vin_x / l_in, vin_y / l_in
        uout_x, uout_y = vout_x / l_out, vout_y / l_out
        cos_turn = (-uin_x) * uout_x + (-uin_y) * uout_y
        if cos_turn > math.cos(math.radians(20.0)):
            out.append(p_cur)
            continue
        d_b = min(bevel_r, 0.20 * l_in, 0.20 * l_out)
        d_s = min(d_b * 1.85, 0.36 * l_in, 0.36 * l_out)
        if d_b < 1e-5:
            out.append(p_cur)
            continue
        cx, cy = p_cur
        out.extend([
            (cx + uin_x * d_s,  cy + uin_y * d_s),
            (cx + uin_x * d_b,  cy + uin_y * d_b),
            (cx + (0.500 * uin_x + 0.134 * uout_x) * d_b, cy + (0.500 * uin_y + 0.134 * uout_y) * d_b),
            (cx + (0.134 * uin_x + 0.500 * uout_x) * d_b, cy + (0.134 * uin_y + 0.500 * uout_y) * d_b),
            (cx + uout_x * d_b, cy + uout_y * d_b),
            (cx + uout_x * d_s, cy + uout_y * d_s),
        ])
    return out


def subdivide_1d_with_support_loops(n_interior: int, sup1: float = 0.005, sup2: float = 0.016) -> list:

    ni = max(2, n_interior)
    s2 = max(0.0025, min(0.18, sup2))
    s1 = max(0.0008, min(s2 * 0.42, sup1))
    mid = [s2 + (1.0 - 2.0 * s2) * (k / ni) for k in range(ni + 1)]
    return [0.0, s1] + mid + [1.0 - s1, 1.0]


def bm_build_beveled_quad_shell(bm: bmesh.types.BMesh, bot_grid: list, top_grid: list,
                                bulge_scale: float = 0.20, max_bevel: float = 0.0022,
                                inset_bevel: bool = False):

    ic = max(0, (len(top_grid) - 1) // 2)
    jc = max(0, (len(top_grid[0]) - 1) // 2)
    ei = top_grid[ic + 1][jc].co - top_grid[ic][jc].co
    ej = top_grid[ic][jc + 1].co - top_grid[ic][jc].co
    eth = top_grid[ic][jc].co - bot_grid[ic][jc].co
    if ei.cross(ej).dot(eth) < 0.0:
        bot_grid = [list(col) for col in zip(*bot_grid)]
        top_grid = [list(col) for col in zip(*top_grid)]

    n_u = len(top_grid) - 1
    n_v = len(top_grid[0]) - 1
    for i in range(n_u):
        for j in range(n_v):
            bm.faces.new((top_grid[i][j], top_grid[i + 1][j], top_grid[i + 1][j + 1], top_grid[i][j + 1]))
            bm.faces.new((bot_grid[i][j], bot_grid[i][j + 1], bot_grid[i + 1][j + 1], bot_grid[i + 1][j]))

    perim = (
        [(i, 0) for i in range(n_u)]
        + [(n_u, j) for j in range(n_v)]
        + [(i, n_v) for i in range(n_u, 0, -1)]
        + [(0, j) for j in range(n_v, 0, -1)]
    )


    bot_nom = {(i, j): bot_grid[i][j].co.copy() for (i, j) in perim}
    top_nom = {(i, j): top_grid[i][j].co.copy() for (i, j) in perim}
    for (i, j) in perim:
        i_in = 1 if i == 0 else (n_u - 1 if i == n_u else i)
        j_in = 1 if j == 0 else (n_v - 1 if j == n_v else j)
        for pt in ((i_in, j), (i, j_in), (i_in, j_in)):
            if pt not in bot_nom:
                bot_nom[pt] = bot_grid[pt[0]][pt[1]].co.copy()
                top_nom[pt] = top_grid[pt[0]][pt[1]].co.copy()

    vb_ring, vr1_ring, vr2_ring, vt_ring = [], [], [], []
    for (i, j) in perim:
        vb = bot_grid[i][j]
        vt = top_grid[i][j]
        pb0 = bot_nom[(i, j)]
        pt0 = top_nom[(i, j)]
        d_th = pt0 - pb0
        th = d_th.length
        n_th = d_th / max(th, 1e-8)

        on_u = (i == 0 or i == n_u)
        on_v = (j == 0 or j == n_v)
        i_in = 1 if i == 0 else (n_u - 1 if i == n_u else i)
        j_in = 1 if j == 0 else (n_v - 1 if j == n_v else j)

        c_bev = min(th * bulge_scale, max_bevel, 0.24 * th) if inset_bevel else min(th * bulge_scale, max_bevel)
        t_sh = min(0.24, max(0.015, (c_bev if inset_bevel else 1.5 * c_bev) / max(th, 1e-6)))

        if inset_bevel and c_bev > 1e-7:
            sh_b = Vector((0.0, 0.0, 0.0))
            sh_t = Vector((0.0, 0.0, 0.0))
            if on_u:
                du_b = bot_nom[(i_in, j)] - pb0
                du_t = top_nom[(i_in, j)] - pt0
                lu_b, lu_t = du_b.length, du_t.length
                if lu_b > 1e-7:
                    sh_b += du_b * (min(c_bev, 0.38 * lu_b) / lu_b)
                if lu_t > 1e-7:
                    sh_t += du_t * (min(c_bev, 0.38 * lu_t) / lu_t)
            if on_v:
                dv_b = bot_nom[(i, j_in)] - pb0
                dv_t = top_nom[(i, j_in)] - pt0
                lv_b, lv_t = dv_b.length, dv_t.length
                if lv_b > 1e-7:
                    sh_b += dv_b * (min(c_bev, 0.38 * lv_b) / lv_b)
                if lv_t > 1e-7:
                    sh_t += dv_t * (min(c_bev, 0.38 * lv_t) / lv_t)
            vb.co = pb0 + sh_b
            vt.co = pt0 + sh_t
            vr1_ring.append(bm.verts.new(pb0.lerp(pt0, t_sh)))
            vr2_ring.append(bm.verts.new(pb0.lerp(pt0, 1.0 - t_sh)))
        else:
            mid_edge = 0.5 * (pb0 + pt0)
            out_vec = Vector((0.0, 0.0, 0.0))
            if on_u:
                mid_u = 0.5 * (bot_nom[(i_in, j)] + top_nom[(i_in, j)])
                ou = mid_edge - mid_u
                ou = ou - n_th * ou.dot(n_th)
                if ou.length > 1e-8:
                    out_vec += ou.normalized()
            if on_v:
                mid_v = 0.5 * (bot_nom[(i, j_in)] + top_nom[(i, j_in)])
                ov = mid_edge - mid_v
                ov = ov - n_th * ov.dot(n_th)
                if ov.length > 1e-8:
                    out_vec += ov.normalized()
            bulge = out_vec * c_bev
            vr1_ring.append(bm.verts.new(pb0.lerp(pt0, t_sh) + bulge))
            vr2_ring.append(bm.verts.new(pb0.lerp(pt0, 1.0 - t_sh) + bulge))

        vb_ring.append(vb)
        vt_ring.append(vt)

    m_p = len(perim)
    for k in range(m_p):
        kn = (k + 1) % m_p
        bm.faces.new((vr2_ring[k], vr2_ring[kn], vt_ring[kn], vt_ring[k]))
        bm.faces.new((vr1_ring[k], vr1_ring[kn], vr2_ring[kn], vr2_ring[k]))
        bm.faces.new((vb_ring[k],  vb_ring[kn],  vr1_ring[kn], vr1_ring[k]))


def bm_add_revolved_profile(bm: bmesh.types.BMesh, profile_yr: list, segs: int = 128,
                            phi_start: float = 0.0, phi_end: float = 2.0 * math.pi,
                            close_loop: bool = True, bevel_r: float = 0.0):
    prof_use = bevel_2d_profile(profile_yr, bevel_r=bevel_r, close_loop=close_loop) if bevel_r > 1e-6 else profile_yr
    n_prof = len(prof_use)
    full_circle = abs((phi_end - phi_start) - 2.0 * math.pi) < 1e-5
    rings = []
    n_steps = segs if full_circle else (segs + 1)
    for s in range(n_steps):
        phi = phi_start + (phi_end - phi_start) * (s / segs)
        cp, sp = math.cos(phi), math.sin(phi)
        rings.append([bm.verts.new((r * sp, y, r * cp)) for (y, r) in prof_use])

    rcnt = len(rings)
    for s in range(segs):
        r_curr = rings[s]
        r_next = rings[(s + 1) % rcnt] if full_circle else rings[s + 1]
        m = n_prof if close_loop else (n_prof - 1)
        for j in range(m):
            jn = (j + 1) % n_prof
            try:
                bm.faces.new((r_curr[j], r_next[j], r_next[jn], r_curr[jn]))
            except ValueError:
                pass
    return rings


def bm_add_notched_revolved_profile(bm: bmesh.types.BMesh, profile_yr: list,
                                    notches: list, bevel_r: float = 0.00045,
                                    base_segs: int = 256):

    prof_use = bevel_2d_profile(profile_yr, bevel_r=bevel_r, close_loop=True) if bevel_r > 1e-6 else profile_yr
    n_prof = len(prof_use)
    r_min_p = min(r for _, r in prof_use)
    r_max_p = max(r for _, r in prof_use)
    r_span_p = max(1e-6, r_max_p - r_min_p)
    r_ref = 0.5 * (r_min_p + r_max_p)
    two_pi = 2.0 * math.pi

    def _wrap_phi(p: float) -> float:
        w = math.fmod(p, two_pi)
        return w + two_pi if w < 0.0 else w

    def _ang_dist(a: float, b: float) -> float:
        d = abs(_wrap_phi(a - b))
        return two_pi - d if d > math.pi else d

    notch_stations = []
    parsed_notches = []
    for (phi_c, half_arc, ramp_arc, r_bot_n, r_top_n) in notches:
        pc = _wrap_phi(phi_c)
        d_flat = half_arc / r_ref
        d_ramp = max(0.0015, ramp_arc / r_ref)
        d_out = d_flat + d_ramp
        parsed_notches.append((pc, d_flat, d_ramp, r_bot_n, r_top_n))
        notch_stations.append(pc)
        for sgn in (-1.0, +1.0):
            for off in (
                0.50 * d_flat,
                max(0.0006, d_flat - 0.0012),
                d_flat,
                d_flat + 0.18 * d_ramp,
                d_flat + 0.50 * d_ramp,
                d_out - 0.18 * d_ramp,
                d_out,
                d_out + 0.0012,
                d_out + 0.0038,
            ):
                notch_stations.append(_wrap_phi(pc + sgn * off))

    notch_stations.sort()
    dedup_stations = []
    for p in notch_stations:
        if not dedup_stations or _ang_dist(p, dedup_stations[-1]) > 0.00055:
            dedup_stations.append(p)
    if len(dedup_stations) >= 2 and _ang_dist(dedup_stations[0], dedup_stations[-1]) <= 0.00055:
        dedup_stations.pop()

    all_phis = list(dedup_stations)
    for s in range(base_segs):
        pu = two_pi * (s / base_segs)
        if all(_ang_dist(pu, ps) > 0.0011 for ps in dedup_stations):
            all_phis.append(pu)
    all_phis.sort()

    rings = []
    for phi in all_phis:
        cp, sp = math.cos(phi), math.sin(phi)
        w_best = 0.0
        rb_use, rt_use = r_min_p, r_max_p
        for (pc, d_flat, d_ramp, r_bot_n, r_top_n) in parsed_notches:
            da = _ang_dist(phi, pc)
            if da <= d_flat:
                w = 1.0
            elif da < d_flat + d_ramp:
                u = (da - d_flat) / d_ramp
                w = 1.0 - (u * u * (3.0 - 2.0 * u))
            else:
                w = 0.0
            if w > w_best:
                w_best = w
                rb_use, rt_use = r_bot_n, r_top_n

        ring_v = []
        for (yv, rv) in prof_use:
            if w_best > 1e-6:
                tr = (rv - r_min_p) / r_span_p
                r_tgt = rb_use + tr * (rt_use - rb_use)
                r_cur = rv * (1.0 - w_best) + r_tgt * w_best
            else:
                r_cur = rv
            ring_v.append(bm.verts.new((r_cur * sp, yv, r_cur * cp)))
        rings.append(ring_v)

    rcnt = len(rings)
    for s in range(rcnt):
        r_curr = rings[s]
        r_next = rings[(s + 1) % rcnt]
        for j in range(n_prof):
            jn = (j + 1) % n_prof
            bm.faces.new((r_curr[j], r_next[j], r_next[jn], r_curr[jn]))
    return rings


def bm_add_segmented_tube(bm: bmesh.types.BMesh, p_a: Vector, p_b: Vector,
                          stations: list, segs: int = 24, cap_ends: bool = True):

    axis = p_b - p_a
    length = axis.length
    if length < 1e-7 or len(stations) < 2:
        return
    t_dir = axis.normalized()
    ref = Vector((1, 0, 0)) if abs(t_dir.x) < 0.9 else Vector((0, 1, 0))
    u_dir = t_dir.cross(ref).normalized()
    v_dir = t_dir.cross(u_dir).normalized()

    rings = []
    for frac, rad in stations:
        center = p_a + axis * frac
        ring = []
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs
            off = u_dir * (rad * math.cos(ang)) + v_dir * (rad * math.sin(ang))
            ring.append(bm.verts.new(center + off))
        rings.append(ring)

    for k in range(len(rings) - 1):
        r0, r1 = rings[k], rings[k + 1]
        for s in range(segs):
            sn = (s + 1) % segs
            try:
                bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))
            except ValueError:
                pass

    if cap_ends:
        _bm_cap_ring_quads(bm, rings[0],  reverse=True,  inset_frac=0.76)
        _bm_cap_ring_quads(bm, rings[-1], reverse=False, inset_frac=0.76)


def bm_add_bevel_locked_hex(bm: bmesh.types.BMesh, p_a: Vector, p_b: Vector,
                            stations: list, corner_bevel: float = 0.14, cap_ends: bool = True):

    axis = p_b - p_a
    if axis.length < 1e-7 or len(stations) < 2:
        return
    t_dir = axis.normalized()
    ref = Vector((1, 0, 0)) if abs(t_dir.x) < 0.9 else Vector((0, 1, 0))
    u_dir = t_dir.cross(ref).normalized()
    v_dir = t_dir.cross(u_dir).normalized()

    cb = max(0.05, min(0.24, corner_bevel))
    s_bev = 1.0 - cb
    s_sup = max(0.25, 1.0 - cb * 1.95)
    unit_hex_2d = []
    for k in range(6):
        th = k * (math.pi / 3.0)
        nx, ny = math.cos(th), math.sin(th)
        tx, ty = -ny, nx
        ap = 0.8660254
        hw = 0.5000000
        unit_hex_2d.extend([
            (nx * ap - tx * hw * s_bev, ny * ap - ty * hw * s_bev),
            (nx * ap - tx * hw * s_sup, ny * ap - ty * hw * s_sup),
            (nx * ap + tx * hw * s_sup, ny * ap + ty * hw * s_sup),
            (nx * ap + tx * hw * s_bev, ny * ap + ty * hw * s_bev),
        ])
        th_c = th + (math.pi / 6.0)
        rc_mid = 0.75 + 0.25 * s_bev + 0.11 * cb
        unit_hex_2d.append((rc_mid * math.cos(th_c), rc_mid * math.sin(th_c)))

    rings = []
    for frac, rad in stations:
        ctr = p_a + axis * frac
        rings.append([
            bm.verts.new(ctr + u_dir * (rad * ux) + v_dir * (rad * uy))
            for (ux, uy) in unit_hex_2d
        ])

    n_v = len(unit_hex_2d)
    for k in range(len(rings) - 1):
        r0, r1 = rings[k], rings[k + 1]
        for s in range(n_v):
            sn = (s + 1) % n_v
            bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))

    if cap_ends:
        _bm_cap_ring_quads(bm, rings[0],  reverse=True,  inset_frac=0.74)
        _bm_cap_ring_quads(bm, rings[-1], reverse=False, inset_frac=0.74)


def bm_add_chamfered_cyl_x(bm: bmesh.types.BMesh, y: float, z: float,
                           x0: float, x1: float, radius: float, segs: int = 32,
                           chamfer: float = 0.0008, hollow_r: float = 0.0):

    xl, xr = (x0, x1) if x0 <= x1 else (x1, x0)
    p_a = Vector((xl, y, z))
    p_b = Vector((xr, y, z))
    L = max(1e-6, xr - xl)
    ch = min(chamfer, 0.20 * L, 0.35 * max(0.0004, radius - hollow_r))
    f_ch = ch / L
    f_sup = min(0.42, f_ch * 1.80)
    segs_even = max(12, segs + (segs % 2))

    if hollow_r <= 1e-5:
        r_end = max(0.0002, radius - ch)
        st = [
            (0.0,               r_end),
            (0.134 * f_ch,      radius - 0.500 * ch),
            (0.500 * f_ch,      radius - 0.134 * ch),
            (f_ch,              radius),
            (f_sup,             radius),
            (1.0 - f_sup,       radius),
            (1.0 - f_ch,        radius),
            (1.0 - 0.500 * f_ch, radius - 0.134 * ch),
            (1.0 - 0.134 * f_ch, radius - 0.500 * ch),
            (1.0,               r_end),
        ]
        bm_add_segmented_tube(bm, p_a, p_b, st, segs=segs_even, cap_ends=True)
    else:
        ch_in = min(ch * 0.55, 0.18 * L)
        raw_prof = [
            (xl, hollow_r),
            (xl, radius),
            (xr, radius),
            (xr, hollow_r),
        ]

        prof = bevel_2d_profile(raw_prof, bevel_r=max(ch_in, ch * 0.85), close_loop=True)
        rings = []
        for s in range(segs_even):
            ang = 2.0 * math.pi * s / segs_even
            ca, sa = math.cos(ang), math.sin(ang)
            rings.append([bm.verts.new((xp, y + rp * ca, z + rp * sa)) for (xp, rp) in prof])
        np = len(prof)
        for s in range(segs_even):
            r0, r1 = rings[s], rings[(s + 1) % segs_even]
            for j in range(np):
                jn = (j + 1) % np
                bm.faces.new((r0[j], r1[j], r1[jn], r0[jn]))


def bm_add_aerospace_bolt(bm: bmesh.types.BMesh, pos: Vector, normal: Vector,
                          head_r: float = 0.0055, height: float = 0.0050,
                          with_12pt_nut_depth: float = 0.0):

    f0_bolt = len(bm.faces)
    n = normal.normalized()

    p0 = pos - n * 0.0003
    p_w = pos + n * (height * 0.24)
    bm_add_segmented_tube(bm, p0, p_w, [
        (0.00, head_r * 1.24),
        (0.15, head_r * 1.33),
        (0.30, head_r * 1.33),
        (0.72, head_r * 1.33),
        (0.88, head_r * 1.30),
        (1.00, head_r * 1.20),
    ], segs=24, cap_ends=True)


    p_h = pos + n * height
    bm_add_bevel_locked_hex(bm, p_w, p_h, [
        (0.00, head_r * 0.94),
        (0.07, head_r),
        (0.16, head_r),
        (0.74, head_r),
        (0.83, head_r),
        (0.92, head_r * 0.88),
        (0.97, head_r * 0.68),
        (1.00, head_r * 0.46),
    ], corner_bevel=0.14, cap_ends=True)


    if with_12pt_nut_depth > 1e-4:
        p_back0 = pos - n * with_12pt_nut_depth
        p_w_nut = p_back0 - n * (height * 0.24)
        bm_add_segmented_tube(bm, p_back0, p_w_nut, [
            (0.00, head_r * 1.24), (0.18, head_r * 1.32), (0.82, head_r * 1.32), (1.00, head_r * 1.16),
        ], segs=24, cap_ends=True)

        p_back1 = p_back0 - n * (height * 1.15)
        axis_n = p_back1 - p_w_nut
        t_n = axis_n.normalized()
        ref_n = Vector((1, 0, 0)) if abs(t_n.x) < 0.9 else Vector((0, 1, 0))
        u_n = t_n.cross(ref_n).normalized()
        v_n = t_n.cross(u_n).normalized()

        spline_2d = []
        for kt in range(12):
            base_a = 2.0 * math.pi * kt / 12.0
            for sub_da, r_mul in [(-0.075, 1.05), (+0.075, 1.05), (+0.2618, 0.85)]:
                aa = base_a + sub_da
                spline_2d.append((r_mul * math.cos(aa), r_mul * math.sin(aa)))
        nut_rings = []
        for fn, rn in [(0.00, head_r * 0.94), (0.09, head_r), (0.20, head_r), (0.72, head_r), (0.82, head_r * 0.96), (1.00, head_r * 0.74)]:
            cn = p_w_nut + axis_n * fn
            nut_rings.append([bm.verts.new(cn + u_n * (rn * sx) + v_n * (rn * sy)) for (sx, sy) in spline_2d])
        for kr in range(len(nut_rings) - 1):
            ra_n, rb_n = nut_rings[kr], nut_rings[kr + 1]
            for s in range(36):
                sn = (s + 1) % 36
                bm.faces.new((ra_n[s], ra_n[sn], rb_n[sn], rb_n[s]))
        _bm_cap_ring_quads(bm, nut_rings[0],  reverse=True,  inset_frac=0.75)
        _bm_cap_ring_quads(bm, nut_rings[-1], reverse=False, inset_frac=0.75)


        p_tip = p_back1 - n * (height * 0.82)
        bm_add_helical_thread_cylinder(
            bm, p_back1 + n * (height * 0.04), p_tip,
            r_core=head_r * 0.50, thread_h=head_r * 0.145,
            n_turns=5, segs_turn=16, cap_core=True,
        )
    bm_fill_unset_from(bm, f0_bolt, MAT_FASTENER)


def bm_add_dome_rivet(bm: bmesh.types.BMesh, pos: Vector, normal: Vector, radius: float = 0.0032):

    f0_rv = len(bm.faces)
    n = normal.normalized()
    p_base = pos - n * 0.0003
    p_top = pos + n * (radius * 0.68)
    bm_add_segmented_tube(bm, p_base, p_top, [
        (0.00, radius * 1.18),
        (0.08, radius * 1.18),
        (0.18, radius * 1.00),
        (0.46, radius * 0.92),
        (0.72, radius * 0.72),
        (0.90, radius * 0.46),
        (1.00, radius * 0.22),
    ], segs=20, cap_ends=True)
    bm_fill_unset_from(bm, f0_rv, MAT_FASTENER)


def _eval_catmull_rom_frames(ctrl_pts: list, n_samples: int):

    pts = [Vector(p) for p in ctrl_pts]
    m = len(pts)
    if m < 2:
        return []
    padded = [pts[0] * 2.0 - pts[1]] + pts + [pts[-1] * 2.0 - pts[-2]]
    n_segs = m - 1

    sampled_pos = []
    for k in range(n_samples):
        u_glob = (k / max(1, n_samples - 1)) * n_segs
        seg = min(n_segs - 1, int(math.floor(u_glob)))
        t = u_glob - seg
        p0, p1, p2, p3 = padded[seg], padded[seg + 1], padded[seg + 2], padded[seg + 3]
        t2 = t * t
        t3 = t2 * t
        pos = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        sampled_pos.append(pos)

    frames = []
    arc = 0.0
    prev_u = None
    for k in range(n_samples):
        if k == 0:
            tang = (sampled_pos[1] - sampled_pos[0])
        elif k == n_samples - 1:
            tang = (sampled_pos[-1] - sampled_pos[-2])
        else:
            tang = (sampled_pos[k + 1] - sampled_pos[k - 1])
        t_dir = tang.normalized() if tang.length > 1e-8 else Vector((0, 1, 0))
        if k > 0:
            arc += (sampled_pos[k] - sampled_pos[k - 1]).length

        if prev_u is None:
            ref = Vector((1, 0, 0)) if abs(t_dir.x) < 0.85 else Vector((0, 0, 1))
            u_dir = t_dir.cross(ref).normalized()
        else:
            u_proj = prev_u - t_dir * prev_u.dot(t_dir)
            if u_proj.length < 1e-6:
                ref = Vector((1, 0, 0)) if abs(t_dir.x) < 0.85 else Vector((0, 0, 1))
                u_dir = t_dir.cross(ref).normalized()
            else:
                u_dir = u_proj.normalized()
        v_dir = t_dir.cross(u_dir).normalized()
        prev_u = u_dir
        frames.append((sampled_pos[k], t_dir, u_dir, v_dir, arc))
    return frames


def bm_add_spline_tube(bm: bmesh.types.BMesh, ctrl_pts: list, radius: float,
                       n_samples: int = 36, segs: int = 18, cap_ends: bool = True,
                       corrugations: int = 0, corr_amp: float = 0.22,
                       closed: bool = False):

    if closed and len(ctrl_pts) >= 3:
        pts_c = [Vector(p) for p in ctrl_pts]
        if (pts_c[-1] - pts_c[0]).length < 1e-4:
            pts_c = pts_c[:-1]
        m_c = len(pts_c)
        if m_c >= 3:
            n_samp = max(12, int(n_samples))
            sampled_pos = []
            for k in range(n_samp):
                u_glob = (k / float(n_samp)) * m_c
                seg = int(math.floor(u_glob)) % m_c
                t = u_glob - math.floor(u_glob)
                p0 = pts_c[(seg - 1) % m_c]
                p1 = pts_c[seg]
                p2 = pts_c[(seg + 1) % m_c]
                p3 = pts_c[(seg + 2) % m_c]
                t2 = t * t
                t3 = t2 * t
                pos = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                sampled_pos.append(pos)

            frames = []
            arc = 0.0
            prev_u = None
            for k in range(n_samp):
                tang = sampled_pos[(k + 1) % n_samp] - sampled_pos[(k - 1) % n_samp]
                t_dir = tang.normalized() if tang.length > 1e-8 else Vector((0, 1, 0))
                if k > 0:
                    arc += (sampled_pos[k] - sampled_pos[k - 1]).length
                if prev_u is None:
                    ref = Vector((1, 0, 0)) if abs(t_dir.x) < 0.85 else Vector((0, 0, 1))
                    u_dir = t_dir.cross(ref).normalized()
                else:
                    u_proj = prev_u - t_dir * prev_u.dot(t_dir)
                    u_dir = u_proj.normalized() if u_proj.length >= 1e-6 else t_dir.cross(Vector((0, 0, 1))).normalized()
                v_dir = t_dir.cross(u_dir).normalized()
                prev_u = u_dir
                frames.append((sampled_pos[k], t_dir, u_dir, v_dir, arc))

            t0, u0, v0 = frames[0][1], frames[0][2], frames[0][3]
            u_end = frames[-1][2] - t0 * frames[-1][2].dot(t0)
            twist_err = 0.0
            if u_end.length > 1e-6:
                u_end.normalize()
                twist_err = math.atan2(u_end.dot(v0), u_end.dot(u0))
            segs_even = max(12, segs + (segs % 2))
            rings = []
            for idx, (pos, _t, u_dir, v_dir, _arc) in enumerate(frames):
                d_ang = -twist_err * (idx / float(n_samp))
                cd, sd = math.cos(d_ang), math.sin(d_ang)
                u_c = u_dir * cd + v_dir * sd
                v_c = -u_dir * sd + v_dir * cd
                ring = []
                for s in range(segs_even):
                    ang = 2.0 * math.pi * s / segs_even
                    ring.append(bm.verts.new(pos + u_c * (radius * math.cos(ang)) + v_c * (radius * math.sin(ang))))
                rings.append(ring)
            for k in range(n_samp):
                r0, r1 = rings[k], rings[(k + 1) % n_samp]
                for s in range(segs_even):
                    sn = (s + 1) % segs_even
                    bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))
            return frames

    frames = _eval_catmull_rom_frames(ctrl_pts, n_samples)
    if len(frames) < 2:
        return frames
    if corrugations >= 12:
        bm_add_braided_hose_from_frames(bm, frames, radius=radius, n_half=5, cap_ends=cap_ends)
        return frames
    segs_even = max(12, segs + (segs % 2))
    rings = []
    for idx, (pos, _t, u_dir, v_dir, _arc) in enumerate(frames):
        frac = idx / max(1, len(frames) - 1)
        r_cur = radius
        if corrugations > 0 and 0.04 < frac < 0.96:
            f_inner = (frac - 0.04) / 0.92
            wave = math.sin(2.0 * math.pi * corrugations * f_inner)
            r_cur = radius * (1.0 + corr_amp * wave)
        ring = []
        for s in range(segs_even):
            ang = 2.0 * math.pi * s / segs_even
            ring.append(bm.verts.new(pos + u_dir * (r_cur * math.cos(ang)) + v_dir * (r_cur * math.sin(ang))))
        rings.append(ring)

    for k in range(len(rings) - 1):
        r0, r1 = rings[k], rings[k + 1]
        for s in range(segs_even):
            sn = (s + 1) % segs_even
            bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))
    if cap_ends:
        _bm_cap_ring_quads(bm, rings[0],  reverse=True,  inset_frac=0.74)
        _bm_cap_ring_quads(bm, rings[-1], reverse=False, inset_frac=0.74)
    return frames


def bm_sweep_centerline_tube(bm: bmesh.types.BMesh, pts_3d: list, radius: float,
                             segs: int = 20, cap_ends: bool = True,
                             corrugations: int = 0, corr_amp: float = 0.12):

    clean_pts = [pts_3d[0]]
    for p in pts_3d[1:]:
        if (p - clean_pts[-1]).length > 1e-5:
            clean_pts.append(p)
    n_p = len(clean_pts)
    if n_p < 2:
        return []

    tangents = []
    arc_lens = [0.0]
    for k in range(n_p):
        if k > 0:
            arc_lens.append(arc_lens[-1] + (clean_pts[k] - clean_pts[k - 1]).length)
        if k == 0:
            td = clean_pts[1] - clean_pts[0]
        elif k == n_p - 1:
            td = clean_pts[-1] - clean_pts[-2]
        else:
            v1 = (clean_pts[k] - clean_pts[k - 1]).normalized()
            v2 = (clean_pts[k + 1] - clean_pts[k]).normalized()
            td = v1 + v2
            if td.length < 1e-5:
                td = v2
        tangents.append(td.normalized())

    t0 = tangents[0]
    ref = Vector((1.0, 0.0, 0.0)) if abs(t0.x) < 0.85 else Vector((0.0, 0.0, 1.0))
    u_prev = (ref - t0 * ref.dot(t0)).normalized()

    frames = []
    segs_even = max(12, segs + (segs % 2))
    rings = []
    total_len = max(1e-6, arc_lens[-1])

    for k in range(n_p):
        tk = tangents[k]
        if k > 0:
            t_prev = tangents[k - 1]
            axis = t_prev.cross(tk)
            if axis.length > 1e-6:
                ang_r = math.acos(max(-1.0, min(1.0, t_prev.dot(tk))))
                u_cur = (Matrix.Rotation(ang_r, 3, axis.normalized()) @ u_prev)
                u_cur = (u_cur - tk * u_cur.dot(tk)).normalized()
            else:
                u_cur = (u_prev - tk * u_prev.dot(tk)).normalized()
        else:
            u_cur = u_prev
        v_cur = tk.cross(u_cur).normalized()
        u_prev = u_cur
        frames.append((clean_pts[k], tk, u_cur, v_cur, arc_lens[k]))

    if corrugations >= 12:
        bm_add_braided_hose_from_frames(bm, frames, radius=radius, n_half=5, cap_ends=cap_ends)
        return frames

    for k, (pt_k, _tk, u_cur, v_cur, arc_k) in enumerate(frames):
        frac = arc_k / total_len
        r_cur = radius
        if corrugations > 0 and 0.03 < frac < 0.97:
            f_in = (frac - 0.03) / 0.94
            r_cur = radius * (1.0 + corr_amp * math.sin(2.0 * math.pi * corrugations * f_in))
        ring = []
        for s in range(segs_even):
            ang = 2.0 * math.pi * s / segs_even
            ring.append(bm.verts.new(pt_k + u_cur * (r_cur * math.cos(ang)) + v_cur * (r_cur * math.sin(ang))))
        rings.append(ring)

    for k in range(len(rings) - 1):
        r0, r1 = rings[k], rings[k + 1]
        for s in range(segs_even):
            sn = (s + 1) % segs_even
            bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))
    if cap_ends:
        _bm_cap_ring_quads(bm, rings[0],  reverse=True,  inset_frac=0.74)
        _bm_cap_ring_quads(bm, rings[-1], reverse=False, inset_frac=0.74)
    return frames


def build_cyl_mandrel_pts(waypoints_deg_yr: list, bend_r: float = 0.036,
                          r_unwrap: float = 0.535, arc_segs: int = 16, max_step: float = 0.012,
                          corner_radii: dict = None) -> list:

    u_pts = [Vector((r_unwrap * math.radians(d), y, r)) for (d, y, r) in waypoints_deg_yr]

    pts = [u_pts[0]]
    for up in u_pts[1:]:
        if (up - pts[-1]).length > 1e-5:
            pts.append(up)
    m = len(pts)
    if m < 2:
        return []
    if m == 2:
        n_s = max(2, int(math.ceil((pts[1] - pts[0]).length / max_step)))
        out = []
        for k in range(n_s + 1):
            u = pts[0].lerp(pts[1], k / float(n_s))
            ph = u.x / r_unwrap
            out.append(Vector((u.z * math.sin(ph), u.y, u.z * math.cos(ph))))
        return out


    seg_vecs = [pts[i + 1] - pts[i] for i in range(m - 1)]
    seg_lens = [v.length for v in seg_vecs]
    seg_dirs = [v / max(1e-8, l) for v, l in zip(seg_vecs, seg_lens)]

    corner_t = [0.0] * m
    corner_theta = [0.0] * m
    for i in range(1, m - 1):
        e_in = seg_dirs[i - 1]
        e_out = seg_dirs[i]
        dot_v = max(-0.999999, min(0.999999, e_in.dot(e_out)))
        th = math.acos(dot_v)
        corner_theta[i] = th
        if th > 1e-3 and th < math.radians(178.5):
            r_b_cur = corner_radii.get(i, bend_r) if corner_radii else bend_r
            t_req = r_b_cur * math.tan(0.5 * th)
            t_lim = 0.494 * min(seg_lens[i - 1], seg_lens[i])
            corner_t[i] = min(t_req, t_lim)


    for i in range(m - 1):
        s_t = corner_t[i] + corner_t[i + 1]
        if s_t > 0.985 * seg_lens[i] and s_t > 1e-7:
            scale = (0.985 * seg_lens[i]) / s_t
            corner_t[i] *= scale
            corner_t[i + 1] *= scale

    sampled_u = []

    def _append_straight(ua: Vector, ub: Vector):
        dist = (ub - ua).length
        if dist < 1e-6:
            if not sampled_u:
                sampled_u.append(ua)
            return
        n_st = max(2, int(math.ceil(dist / max_step)))
        k0 = 0 if not sampled_u else 1
        for k in range(k0, n_st + 1):
            sampled_u.append(ua.lerp(ub, k / float(n_st)))

    cur_start = pts[0]
    for i in range(1, m - 1):
        th = corner_theta[i]
        ti = corner_t[i]
        if ti < 1e-5 or th <= 1e-3:
            _append_straight(cur_start, pts[i])
            cur_start = pts[i]
            continue
        e_in = seg_dirs[i - 1]
        e_out = seg_dirs[i]
        a_pt = pts[i] - e_in * ti
        b_pt = pts[i] + e_out * ti
        _append_straight(cur_start, a_pt)

        r_eff = ti / math.tan(0.5 * th)
        n_turn_raw = e_out - e_in * e_in.dot(e_out)
        if n_turn_raw.length < 1e-7:
            _append_straight(a_pt, b_pt)
            cur_start = b_pt
            continue
        n_turn = n_turn_raw.normalized()
        c_pt = a_pt + n_turn * r_eff
        v0 = -n_turn * r_eff
        v90 = e_in * r_eff
        n_a = max(6, int(math.ceil(arc_segs * (th / (0.5 * math.pi)))))
        for ka in range(1, n_a + 1):
            alp = th * (ka / float(n_a))
            sampled_u.append(c_pt + v0 * math.cos(alp) + v90 * math.sin(alp))
        cur_start = b_pt

    _append_straight(cur_start, pts[-1])

    out_3d = []
    for u in sampled_u:
        ph = u.x / r_unwrap
        out_3d.append(Vector((u.z * math.sin(ph), u.y, u.z * math.cos(ph))))
    return out_3d


def bm_add_bnut_flare_fitting(bm: bmesh.types.BMesh, pos: Vector, tangent: Vector, tube_r: float = 0.0038):

    f0_bn = len(bm.faces)
    t = tangent.normalized()
    p0 = pos - t * (tube_r * 2.5)
    p1 = pos + t * (tube_r * 2.5)
    bm_add_segmented_tube(bm, p0, p1, [
        (0.00, tube_r * 1.20), (0.04, tube_r * 1.30), (0.18, tube_r * 1.30),
        (0.22, tube_r * 1.12), (0.78, tube_r * 1.12),
        (0.82, tube_r * 1.32), (0.96, tube_r * 1.32), (1.00, tube_r * 1.20),
    ], segs=20, cap_ends=True)

    bm_add_bevel_locked_hex(bm, pos - t * (tube_r * 1.42), pos - t * (tube_r * 0.12), [
        (0.00, tube_r * 1.62), (0.10, tube_r * 1.94), (0.20, tube_r * 1.94),
        (0.80, tube_r * 1.94), (0.90, tube_r * 1.94), (1.00, tube_r * 1.62),
    ], corner_bevel=0.14, cap_ends=True)
    bm_add_bevel_locked_hex(bm, pos + t * (tube_r * 0.12), pos + t * (tube_r * 1.48), [
        (0.00, tube_r * 1.54), (0.10, tube_r * 1.82), (0.20, tube_r * 1.82),
        (0.80, tube_r * 1.82), (0.90, tube_r * 1.82), (1.00, tube_r * 1.52),
    ], corner_bevel=0.14, cap_ends=True)
    bm_fill_unset_from(bm, f0_bn, MAT_BRASS_FITTING)


def bm_add_adel_loop_clamp(bm: bmesh.types.BMesh, p_tube: Vector, t_tube: Vector,
                           p_foot_base: Vector, bolt_normal: Vector, tube_r: float = 0.0038):

    t = t_tube.normalized()
    w_half = max(0.0042, tube_r * 1.35)

    f0_ad = len(bm.faces)
    bm_add_segmented_tube(bm, p_tube - t * (w_half * 1.18), p_tube + t * (w_half * 1.18), [
        (0.00, tube_r * 1.10), (0.06, tube_r * 1.38), (0.16, tube_r * 1.38),
        (0.20, tube_r * 1.28), (0.80, tube_r * 1.28),
        (0.84, tube_r * 1.38), (0.94, tube_r * 1.38), (1.00, tube_r * 1.10),
    ], segs=24, cap_ends=True)
    bm_fill_unset_from(bm, f0_ad, MAT_ADEL_CUSHION)

    f1_ad = len(bm.faces)
    bm_add_segmented_tube(bm, p_tube - t * w_half, p_tube + t * w_half, [
        (0.00, tube_r * 1.32), (0.08, tube_r * 1.54), (0.18, tube_r * 1.54),
        (0.82, tube_r * 1.54), (0.92, tube_r * 1.54), (1.00, tube_r * 1.32),
    ], segs=24, cap_ends=True)

    span = p_tube - p_foot_base
    if span.length > 1e-5:
        r_dir = span.normalized()
        n_bn = bolt_normal.normalized()
        s_dir = t.cross(r_dir)
        if s_dir.length < 1e-5:
            s_dir = t.cross(n_bn)
        s_dir = s_dir.normalized()

        bm_add_segmented_tube(bm, p_foot_base - r_dir * 0.0012, p_tube - r_dir * (tube_r * 1.08), [
            (0.00, max(0.0046, tube_r * 1.25)), (0.20, max(0.0046, tube_r * 1.25)),
            (0.82, max(0.0042, tube_r * 1.15)), (1.00, max(0.0038, tube_r * 1.05)),
        ], segs=18, cap_ends=True)

        lat_off = tube_r * 1.65 + 0.0052
        tab_h = 0.0030
        p_foot_lat = p_foot_base + s_dir * lat_off
        p_arm_a = p_tube - r_dir * (tube_r * 0.85) + s_dir * (tube_r * 0.85)
        p_arm_b = p_foot_lat + r_dir * (tab_h * 0.55)
        bm_add_segmented_tube(bm, p_arm_a, p_arm_b, [
            (0.00, max(0.0030, w_half * 0.72)), (0.18, max(0.0032, w_half * 0.78)),
            (0.82, max(0.0032, w_half * 0.78)), (1.00, max(0.0030, w_half * 0.72)),
        ], segs=16, cap_ends=True)
        bm_add_segmented_tube(bm, p_foot_lat - r_dir * 0.0012, p_foot_lat + r_dir * tab_h, [
            (0.00, 0.0052), (0.18, 0.0056), (0.82, 0.0056), (1.00, 0.0050),
        ], segs=20, cap_ends=True)
        bm_add_aerospace_bolt(bm, p_foot_lat + r_dir * (tab_h - 0.0002), r_dir, head_r=0.0026, height=0.0026)
    bm_fill_unset_from(bm, f1_ad, MAT_MACHINED_ALU)


def bm_add_twisted_lockwire(bm: bmesh.types.BMesh, ctrl_pts: list,
                            strand_r: float = 0.00048, twists_per_m: float = 210.0,
                            n_samples: int = 28, segs: int = 12):

    frames = _eval_catmull_rom_frames(ctrl_pts, n_samples)
    if len(frames) < 2:
        return
    f0_lw = len(bm.faces)
    segs_use = max(12, segs + (segs % 2))
    sep = strand_r * 0.96
    for strand_idx in (0, 1):
        phase0 = strand_idx * math.pi
        rings = []
        for idx, (pos, _t, u_dir, v_dir, arc) in enumerate(frames):
            psi = 2.0 * math.pi * twists_per_m * arc + phase0
            c_s = pos + u_dir * (sep * math.cos(psi)) + v_dir * (sep * math.sin(psi))
            r_s = strand_r * (0.65 if idx == len(frames) - 1 else 1.0)
            ring = []
            for s in range(segs_use):
                ang = 2.0 * math.pi * s / segs_use
                ring.append(bm.verts.new(c_s + u_dir * (r_s * math.cos(ang)) + v_dir * (r_s * math.sin(ang))))
            rings.append(ring)
        for k in range(len(rings) - 1):
            r0, r1 = rings[k], rings[k + 1]
            for s in range(segs_use):
                sn = (s + 1) % segs_use
                bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))
        _bm_cap_ring_quads(bm, rings[0],  reverse=True,  inset_frac=0.70)
        _bm_cap_ring_quads(bm, rings[-1], reverse=False, inset_frac=0.70)
    bm_fill_unset_from(bm, f0_lw, MAT_LOCKWIRE)


def bm_add_bolt_pair_lockwire(bm: bmesh.types.BMesh, p1: Vector, p2: Vector,
                              normal: Vector, head_r: float = 0.0050, strand_r: float = 0.00048):

    n = normal.normalized()
    span_v = p2 - p1
    dist = span_v.length
    if dist < 1e-5:
        return
    t_dir = span_v / dist
    s_dir = n.cross(t_dir).normalized()
    h_lift = n * (head_r * 0.62)

    c1 = p1 + h_lift
    c2 = p2 + h_lift
    mid_pts = [(c1 + c2) * 0.5]
    if dist > 4.2 * head_r:
        mid_pts = [
            c1 + t_dir * (head_r * 1.85) + s_dir * (head_r * 0.22),
            (c1 + c2) * 0.5,
            c2 - t_dir * (head_r * 1.85) - s_dir * (head_r * 0.22),
        ]
    pts_main = [
        c1 - t_dir * (head_r * 0.85) + s_dir * (head_r * 0.45),
        c1 + s_dir * (head_r * 0.92),
        c1 + t_dir * (head_r * 0.95) + s_dir * (head_r * 0.55),
    ] + mid_pts + [
        c2 - t_dir * (head_r * 0.95) - s_dir * (head_r * 0.55),
        c2 - s_dir * (head_r * 0.92),
        c2 + t_dir * (head_r * 0.88) - s_dir * (head_r * 0.35),
        c2 + t_dir * (head_r * 1.65) + s_dir * (head_r * 0.30) - n * (head_r * 0.18),
    ]
    bm_add_twisted_lockwire(bm, pts_main, strand_r=strand_r, twists_per_m=220.0, n_samples=28, segs=12)


def bm_add_clevis_pin_assembly_x(bm: bmesh.types.BMesh, y: float, z: float,
                                 x_left: float, x_right: float, pin_r: float = 0.0050,
                                 segs: int = 28, with_cotter: bool = True,
                                 compact_ends: bool = False, flip_x: bool = False):

    if flip_x:
        bm_tmp = bmesh.new()
        half_sp = 0.5 * abs(x_right - x_left)
        x_mid = 0.5 * (x_left + x_right)
        bm_add_clevis_pin_assembly_x(
            bm_tmp, y, z, -half_sp, +half_sp,
            pin_r=pin_r, segs=segs, with_cotter=with_cotter,
            compact_ends=compact_ends, flip_x=False,
        )
        bmesh.ops.transform(bm_tmp, matrix=Matrix.Scale(-1.0, 4, Vector((1.0, 0.0, 0.0))), verts=bm_tmp.verts)
        bmesh.ops.reverse_faces(bm_tmp, faces=bm_tmp.faces)
        bm_merge(bm, bm_tmp, Matrix.Translation((x_mid, 0.0, 0.0)))
        return

    f0_cpin = len(bm.faces)
    if compact_ends:
        l_ext = 0.0023
        r_ext = 0.0052 if with_cotter else 0.0026
        hd_r = pin_r * 1.40
        ws_r = pin_r * 1.56
    else:
        l_ext = 0.0046
        r_ext = 0.0074
        hd_r = pin_r * 1.56
        ws_r = pin_r * 1.82

    p_l = Vector((x_left - l_ext, y, z))
    p_r = Vector((x_right + r_ext, y, z))
    L = (p_r - p_l).length
    def _fx(xv):
        return (xv - p_l.x) / L

    f0_cpin = len(bm.faces)
    if compact_ends:
        stations = [
            (0.0,                        pin_r * 1.18),
            (_fx(x_left - l_ext + 0.0003), hd_r),
            (_fx(x_left - 0.0008),       hd_r),
            (_fx(x_left - 0.0006),       ws_r),
            (_fx(x_left - 0.0001),       ws_r),
            (_fx(x_left + 0.0002),       pin_r),
            (_fx(x_right - 0.0002),      pin_r),
            (_fx(x_right + 0.0001),      ws_r),
            (_fx(x_right + 0.0007),      ws_r),
            (_fx(x_right + 0.0009),      pin_r * 1.02),
            (_fx(x_right + r_ext - 0.0006), pin_r * 1.02),
            (1.0,                        pin_r * 0.78),
        ]
    else:
        stations = [
            (0.0,                        pin_r * 1.25),
            (_fx(x_left - 0.0042),       hd_r),
            (_fx(x_left - 0.0038),       hd_r),
            (_fx(x_left - 0.0012),       hd_r),
            (_fx(x_left - 0.0009),       ws_r),
            (_fx(x_left - 0.0001),       ws_r),
            (_fx(x_left + 0.0002),       pin_r),
            (_fx(x_right - 0.0002),      pin_r),
            (_fx(x_right + 0.0001),      ws_r),
            (_fx(x_right + 0.0010),      ws_r),
            (_fx(x_right + 0.0012),      pin_r * 1.04),
            (_fx(x_right + 0.0052),      pin_r * 1.04),
            (_fx(x_right + 0.0056),      pin_r * 0.92),
            (1.0,                        pin_r * 0.78),
        ]
    bm_add_segmented_tube(bm, p_l, p_r, stations, segs=max(24, segs + (segs % 2)), cap_ends=True)


    hx0 = x_right + (0.0007 if compact_ends else 0.0010)
    hx1 = x_right + (0.0024 if compact_ends else 0.0033)
    nut_s = 1.46 if compact_ends else 1.58
    bm_add_bevel_locked_hex(
        bm, Vector((hx0, y, z)), Vector((hx1, y, z)),
        [(0.0, pin_r * 1.30), (0.14, pin_r * nut_s), (0.28, pin_r * nut_s), (0.76, pin_r * nut_s), (0.90, pin_r * (nut_s - 0.04)), (1.0, pin_r * 1.34)],
        corner_bevel=0.14, cap_ends=True,
    )

    if with_cotter:

        x_t_arr = (
            [x_right + 0.0023, x_right + 0.0025, x_right + 0.0037, x_right + 0.0039]
            if compact_ends else
            [x_right + 0.0031, x_right + 0.0033, x_right + 0.0048, x_right + 0.0050]
        )
        r_in_t = pin_r * 0.98
        r_out_t = pin_r * (1.38 if compact_ends else 1.48)
        for ti in range(6):
            a0 = 2.0 * math.pi * (ti + 0.17) / 6.0
            a1 = 2.0 * math.pi * (ti + 0.67) / 6.0
            a_arr = [a0, a0 + 0.12 * (a1 - a0), a0 + 0.50 * (a1 - a0), a1 - 0.12 * (a1 - a0), a1]
            g_tin, g_tout = [], []
            for xv_t in x_t_arr:
                ri_row, ro_row = [], []
                for aa in a_arr:
                    ca_t, sa_t = math.cos(aa), math.sin(aa)
                    ri_row.append(bm.verts.new((xv_t, y + r_in_t * ca_t,  z + r_in_t * sa_t)))
                    ro_row.append(bm.verts.new((xv_t, y + r_out_t * ca_t, z + r_out_t * sa_t)))
                g_tin.append(ri_row)
                g_tout.append(ro_row)
            bm_build_beveled_quad_shell(bm, g_tin, g_tout, bulge_scale=0.15)


        xc = x_right + (0.0032 if compact_ends else 0.0041)
        wr = max(0.00040, pin_r * 0.140)
        c_scale = 0.84 if compact_ends else 1.00
        x_tip = x_right + (0.0054 if compact_ends else 0.0076)
        leg1 = [
            Vector((xc, y - wr * 0.8, z + pin_r * 1.12 * c_scale)),
            Vector((xc, y - pin_r * 0.50 * c_scale, z + pin_r * 1.80 * c_scale)),
            Vector((xc, y + pin_r * 0.50 * c_scale, z + pin_r * 1.80 * c_scale)),
            Vector((xc, y + wr * 0.7, z + pin_r * 1.08 * c_scale)),
            Vector((xc, y + wr * 0.6, z - pin_r * 1.15 * c_scale)),
            Vector((xc + pin_r * 0.45 * c_scale, y + wr * 0.6, z - pin_r * 1.45 * c_scale)),
            Vector((x_tip, y + wr * 0.4, z - pin_r * 0.82 * c_scale)),
        ]
        leg2 = [
            Vector((xc, y - wr * 0.6, z + pin_r * 1.10 * c_scale)),
            Vector((xc, y - wr * 0.6, z - pin_r * 1.18 * c_scale)),
            Vector((xc - pin_r * 0.30 * c_scale, y - pin_r * 0.52 * c_scale, z - pin_r * 1.55 * c_scale)),
            Vector((xc - pin_r * 0.48 * c_scale, y - pin_r * 1.18 * c_scale, z - pin_r * 1.30 * c_scale)),
        ]
        f0_cot = len(bm.faces)
        bm_add_spline_tube(bm, leg1, radius=wr, n_samples=20, segs=12, cap_ends=True)
        bm_add_spline_tube(bm, leg2, radius=wr, n_samples=14, segs=12, cap_ends=True)
        bm_fill_unset_from(bm, f0_cot, MAT_LOCKWIRE)


        if pin_r >= 0.0038 and not compact_ends:
            yk_t, zk_t = y - pin_r * 1.90, z + pin_r * 1.90
            bm_add_clevis_plate(
                bm, (y, z), (yk_t, zk_t),
                pin_r * 1.62, pin_r * 0.92,
                x_left - 0.0014, x_left - 0.0002, arc_segs=11, bevel_inset=0.00022,
            )
            bm_add_aerospace_bolt(
                bm, Vector((x_left - 0.0014, yk_t, zk_t)),
                Vector((-1.0, 0.0, 0.0)), head_r=pin_r * 0.42, height=pin_r * 0.38,
            )
    bm_fill_unset_from(bm, f0_cpin, MAT_FASTENER)


def bm_add_grease_zerk(bm: bmesh.types.BMesh, pos: Vector, normal: Vector, scale: float = 1.0):

    f0_gz = len(bm.faces)
    n = normal.normalized()
    s = scale
    p0 = pos - n * (0.0003 * s)
    p_hex = pos + n * (0.0019 * s)
    bm_add_bevel_locked_hex(bm, p0, p_hex, [
        (0.00, 0.0018 * s), (0.14, 0.0021 * s), (0.28, 0.0021 * s),
        (0.72, 0.0021 * s), (0.86, 0.0021 * s), (1.00, 0.0016 * s),
    ], corner_bevel=0.14, cap_ends=True)
    bm_add_segmented_tube(bm, p_hex - n * (0.0001 * s), pos + n * (0.0044 * s), [
        (0.00, 0.00125 * s), (0.28, 0.00092 * s), (0.56, 0.00142 * s),
        (0.78, 0.00136 * s), (0.92, 0.00098 * s), (1.00, 0.00048 * s),
    ], segs=16, cap_ends=True)
    bm_fill_unset_from(bm, f0_gz, MAT_BRASS_FITTING)


def bm_add_hat_stiffener_rib(bm: bmesh.types.BMesh, p_a: Vector, p_b: Vector,
                             u_dir: Vector, n_dir: Vector,
                             w_foot: float = 0.0066, w_top: float = 0.0026,
                             h_hat: float = 0.0036, n_steps: int = 5,
                             rivet_r: float = 0.0013):

    u = u_dir.normalized()
    n = n_dir.normalized()
    if n.cross(p_b - p_a).dot(u) > 0.0:
        u = -u
    w_web = w_top + 0.36 * (w_foot - w_top)
    h_fl = max(0.0008, 0.28 * h_hat)
    th = max(0.0006, 0.24 * h_hat)
    prof_hat = [
        (-w_foot, 0.0000), (-w_foot, h_fl), (-w_web, h_fl), (-w_top, h_hat),
        (+w_top, h_hat),   (+w_web, h_fl),  (+w_foot, h_fl), (+w_foot, 0.0000),
        (+w_web - th * 0.85, 0.0001), (+w_top - th * 0.55, h_hat - th),
        (-w_top + th * 0.55, h_hat - th), (-w_web + th * 0.85, 0.0001),
    ]
    hat_caps = [(0, 1, 2, 11), (11, 2, 3, 10), (10, 3, 4, 9), (9, 4, 5, 8), (8, 5, 6, 7)]
    bm_add_swept_quad_profile(bm, p_a, p_b, u, n, prof_hat, n_steps=n_steps, cap_quad_pairs=hat_caps)
    if rivet_r > 0.0:
        w_rv = 0.5 * (w_foot + w_web)
        for sf_r in (0.22, 0.78):
            p_m = p_a.lerp(p_b, sf_r)
            for sx_r in (-1.0, +1.0):
                bm_add_dome_rivet(bm, p_m + u * (sx_r * w_rv) + n * h_fl, n, radius=rivet_r)


def bm_add_turnbuckle_rod(bm: bmesh.types.BMesh, p_a: Vector, p_b: Vector,
                          rod_r: float = 0.0055, eye_r: float = 0.0105,
                          eye_half_w: float = 0.0048, n_threads: int = 12, segs: int = 32,
                          with_lockwire: bool = True):

    axis = p_b - p_a
    L = axis.length
    f_eye_a = (eye_r * 1.25) / L
    f_eye_b = 1.0 - f_eye_a

    f_m0 = f_eye_a + 0.112
    f_m1 = f_eye_b - 0.112
    st = [
        (0.0, rod_r * 1.28), (f_eye_a, rod_r * 1.32), (f_eye_a + 0.015, rod_r * 0.90),
        (f_m0 - 0.004, rod_r * 0.90),
        (f_m0,          rod_r * 1.50),
        (f_m0 + 0.004,  rod_r * 1.52),
        (f_m0 + 0.024,  rod_r * 1.52),
        (f_m0 + 0.030,  rod_r * 1.26),
        (f_m0 + 0.036,  rod_r * 1.26),
        (f_m0 + 0.094,  rod_r * 1.26),
        (f_m0 + 0.104,  rod_r * 1.02),
        (0.376, rod_r * 1.02), (0.388, rod_r * 1.18), (0.393, rod_r * 1.18), (0.407, rod_r * 1.18), (0.412, rod_r * 1.18), (0.424, rod_r * 1.02),
        (0.486, rod_r * 1.02), (0.496, rod_r * 1.22), (0.500, rod_r * 1.22), (0.510, rod_r * 1.22), (0.514, rod_r * 1.22), (0.524, rod_r * 1.02),
        (0.576, rod_r * 1.02), (0.588, rod_r * 1.18), (0.593, rod_r * 1.18), (0.607, rod_r * 1.18), (0.612, rod_r * 1.18), (0.624, rod_r * 1.02),
        (f_m1 - 0.104,  rod_r * 1.02),
        (f_m1 - 0.094,  rod_r * 1.26),
        (f_m1 - 0.036,  rod_r * 1.26),
        (f_m1 - 0.030,  rod_r * 1.26),
        (f_m1 - 0.024,  rod_r * 1.52),
        (f_m1 - 0.004,  rod_r * 1.52),
        (f_m1,          rod_r * 1.50),
        (f_m1 + 0.004,  rod_r * 0.90),
        (f_eye_b - 0.015, rod_r * 0.90), (f_eye_b, rod_r * 1.32), (1.0, rod_r * 1.28),
    ]
    bm_add_segmented_tube(bm, p_a, p_b, st, segs=max(24, segs + (segs % 2)), cap_ends=True)

    bm_add_helical_thread_cylinder(
        bm, p_a + axis * (f_eye_a + 0.014), p_a + axis * (f_m0 - 0.002),
        r_core=rod_r * 0.90, thread_h=rod_r * 0.25, n_turns=max(4, n_threads), segs_turn=18, cap_core=False,
    )
    bm_add_helical_thread_cylinder(
        bm, p_a + axis * (f_m1 + 0.002), p_a + axis * (f_eye_b - 0.014),
        r_core=rod_r * 0.90, thread_h=rod_r * 0.25, n_turns=max(4, n_threads), segs_turn=18, cap_core=False,
    )


    t_dir = axis.normalized()
    u_perp = t_dir.cross(Vector((1, 0, 0))).normalized() if abs(t_dir.x) < 0.9 else t_dir.cross(Vector((0, 0, 1))).normalized()
    v_perp = t_dir.cross(u_perp).normalized()
    for f_jam0, f_jam1, f_eye_neck, f_barrel_hole in [
        (f_m0 - 0.002, f_m0 + 0.028, f_eye_a + 0.008, f_m0 + 0.072),
        (f_m1 - 0.028, f_m1 + 0.002, f_eye_b - 0.008, f_m1 - 0.072),
    ]:
        f0_jam = len(bm.faces)
        bm_add_bevel_locked_hex(
            bm, p_a + axis * f_jam0, p_a + axis * f_jam1,
            [(0.0, rod_r * 1.58), (0.12, rod_r * 1.80), (0.24, rod_r * 1.80),
             (0.76, rod_r * 1.80), (0.88, rod_r * 1.80), (1.0, rod_r * 1.58)],
            corner_bevel=0.14, cap_ends=True,
        )
        bm_fill_unset_from(bm, f0_jam, MAT_FASTENER)
        if with_lockwire:
            f_jm = 0.5 * (f_jam0 + f_jam1)
            lw_pts = [
                p_a + axis * f_eye_neck + u_perp * (rod_r * 1.35),
                p_a + axis * (0.5 * (f_eye_neck + f_jm)) + u_perp * (rod_r * 1.48) + v_perp * (rod_r * 0.45),
                p_a + axis * f_jm + u_perp * (rod_r * 1.84),
                p_a + axis * (0.5 * (f_jm + f_barrel_hole)) + u_perp * (rod_r * 1.45) - v_perp * (rod_r * 0.45),
                p_a + axis * f_barrel_hole + u_perp * (rod_r * 1.28),
                p_a + axis * f_barrel_hole + u_perp * (rod_r * 1.75) - v_perp * (rod_r * 0.70),
            ]
            bm_add_twisted_lockwire(bm, lw_pts, strand_r=max(0.00038, rod_r * 0.082), twists_per_m=240.0, n_samples=24, segs=12)


    bm_add_bevel_locked_hex(
        bm, p_a + axis * 0.476, p_a + axis * 0.524,
        [(0.0, rod_r * 1.20), (0.14, rod_r * 1.44), (0.26, rod_r * 1.44),
         (0.74, rod_r * 1.44), (0.86, rod_r * 1.44), (1.0, rod_r * 1.20)],
        corner_bevel=0.14, cap_ends=True,
    )
    for f_wit in (0.400, 0.600):
        p_w_ctr = p_a + axis * f_wit
        for s_w in (-1.0, +1.0):
            bm_add_segmented_tube(
                bm, p_w_ctr + v_perp * (s_w * rod_r * 1.05), p_w_ctr + v_perp * (s_w * rod_r * 1.30),
                [(0.0, rod_r * 0.34), (0.45, rod_r * 0.34), (0.65, rod_r * 0.20), (1.0, rod_r * 0.16)],
                segs=14, cap_ends=True,
            )


    for pt in (p_a, p_b):
        f0_hm = len(bm.faces)
        bm_add_chamfered_cyl_x(
            bm, pt.y, pt.z,
            pt.x - eye_half_w, pt.x + eye_half_w,
            radius=eye_r, segs=36, chamfer=0.0010, hollow_r=eye_r * 0.46,
        )
        bm_fill_unset_from(bm, f0_hm, MAT_MACHINED_ALU)
        f1_hm = len(bm.faces)
        for sx_cr in (-1.0, +1.0):
            bm_add_chamfered_cyl_x(
                bm, pt.y, pt.z,
                pt.x + sx_cr * eye_half_w * 0.96, pt.x + sx_cr * eye_half_w * 1.10,
                radius=eye_r * 0.78, segs=28, chamfer=0.00022, hollow_r=eye_r * 0.64,
            )
        bm_fill_unset_from(bm, f1_hm, MAT_ADEL_CUSHION)
        f2_hm = len(bm.faces)
        bm_add_chamfered_cyl_x(
            bm, pt.y, pt.z,
            pt.x - eye_half_w * 1.25, pt.x + eye_half_w * 1.25,
            radius=eye_r * 0.66, segs=28, chamfer=0.0006, hollow_r=eye_r * 0.40,
        )
        bm_fill_unset_from(bm, f2_hm, MAT_BRASS_FITTING)


def bm_add_clevis_plate(bm: bmesh.types.BMesh, p_a: tuple, p_b: tuple,
                        r_a: float, r_b: float, x0: float, x1: float,
                        arc_segs: int = 21, bevel_inset: float = 0.00075):

    xl, xr = (x0, x1) if x0 <= x1 else (x1, x0)
    dy = p_b[0] - p_a[0]
    dz = p_b[1] - p_a[1]
    base_ang = math.atan2(dz, dy)
    a_segs = max(9, int(arc_segs) | 1)

    def _contour(ra, rb):
        pts = []
        for i in range(a_segs + 1):
            ang = base_ang + 0.5 * math.pi + math.pi * (i / a_segs)
            pts.append((p_a[0] + ra * math.cos(ang), p_a[1] + ra * math.sin(ang)))
        for i in range(a_segs + 1):
            ang = base_ang - 0.5 * math.pi + math.pi * (i / a_segs)
            pts.append((p_b[0] + rb * math.cos(ang), p_b[1] + rb * math.sin(ang)))
        return pts

    w = max(1e-6, xr - xl)
    r_min = min(r_a, r_b)
    bv = min(bevel_inset, 0.22 * w, 0.20 * r_min)
    bv_in_sup = min(bv * 1.70, 0.36 * r_min)
    x_sup = min(0.38 * w, bv * 1.75)

    ring_specs = [
        (xl,                 bv_in_sup),
        (xl,                 bv),
        (xl + 0.134 * bv,    0.500 * bv),
        (xl + 0.500 * bv,    0.134 * bv),
        (xl + bv,            0.0),
        (xl + x_sup,         0.0),
        (xr - x_sup,         0.0),
        (xr - bv,            0.0),
        (xr - 0.500 * bv,    0.134 * bv),
        (xr - 0.134 * bv,    0.500 * bv),
        (xr,                 bv),
        (xr,                 bv_in_sup),
    ]
    rings = []
    for xv, inset_r in ring_specs:
        pts_c = _contour(max(0.0002, r_a - inset_r), max(0.0002, r_b - inset_r))
        rings.append([bm.verts.new((xv, y, z)) for (y, z) in pts_c])

    n_c = len(rings[0])
    for kr in range(len(rings) - 1):
        r_cur, r_nxt = rings[kr], rings[kr + 1]
        for i in range(n_c):
            inext = (i + 1) % n_c
            bm.faces.new((r_cur[i], r_cur[inext], r_nxt[inext], r_nxt[i]))


    h_idx = a_segs // 2
    bot_chain = list(range(h_idx + 1, a_segs + 1 + h_idx + 1))
    top_chain = list(range(h_idx, -1, -1)) + list(range(n_c - 1, a_segs + 1 + h_idx, -1))
    r_left_cap, r_right_cap = rings[0], rings[-1]
    for m in range(a_segs):
        b0, b1 = bot_chain[m], bot_chain[m + 1]
        t0, t1 = top_chain[m], top_chain[m + 1]
        bm.faces.new((r_left_cap[b0],  r_left_cap[t0],  r_left_cap[t1],  r_left_cap[b1]))
        bm.faces.new((r_right_cap[t0], r_right_cap[b0], r_right_cap[b1], r_right_cap[t1]))


def bm_add_spoked_roller_x(bm: bmesh.types.BMesh, y: float, z: float,
                           x0: float, x1: float, radius: float,
                           axle_r: float = 0.0018, n_spokes: int = 5):

    f0_rl = len(bm.faces)
    xl, xr = (x0, x1) if x0 <= x1 else (x1, x0)
    w = max(0.0016, xr - xl)
    r_dish = radius * 0.60
    bm_add_chamfered_cyl_x(bm, y, z, xl, xr, radius=radius, segs=28, chamfer=min(0.00055, 0.16 * w), hollow_r=r_dish)
    bm_add_chamfered_cyl_x(bm, y, z, xl + 0.24 * w, xr - 0.24 * w, radius=r_dish * 1.05, segs=24, chamfer=0.00025)
    r_sp_end = r_dish * 0.96
    sp_r0 = max(0.00050, radius * 0.145)
    sp_r1 = max(0.00042, radius * 0.120)
    for ks in range(n_spokes):
        ang_s = 2.0 * math.pi * ks / n_spokes
        bm_add_clevis_plate(
            bm,
            (y, z),
            (y + r_sp_end * math.cos(ang_s), z + r_sp_end * math.sin(ang_s)),
            sp_r0, sp_r1,
            xl + 0.07 * w, xr - 0.07 * w,
            arc_segs=9, bevel_inset=0.00018,
        )
    bm_add_bevel_locked_hex(
        bm, Vector((xl - 0.00025, y, z)), Vector((xr + 0.00025, y, z)),
        [(0.0, axle_r * 0.92), (0.14, axle_r * 1.12), (0.86, axle_r * 1.12), (1.0, axle_r * 0.92)],
        corner_bevel=0.14, cap_ends=True,
    )
    bm_fill_unset_from(bm, f0_rl, MAT_STAND_STEEL)


def bm_add_swept_quad_profile(bm: bmesh.types.BMesh, p_a: Vector, p_b: Vector,
                              u_dir: Vector, v_dir: Vector, profile_uv: list,
                              n_steps: int = 18, cap_quad_pairs: list = None,
                              bevel_r: float = 0.00028):

    prof_use = bevel_2d_profile(profile_uv, bevel_r=bevel_r, close_loop=True) if (not cap_quad_pairs and bevel_r > 1e-6) else profile_uv
    t_vals = subdivide_1d_with_support_loops(max(4, n_steps), sup1=0.008, sup2=0.026)
    rings = []
    for t in t_vals:
        ctr = p_a.lerp(p_b, t)
        rings.append([bm.verts.new(ctr + u_dir * uv[0] + v_dir * uv[1]) for uv in prof_use])

    m = len(prof_use)
    for k in range(len(rings) - 1):
        r0, r1 = rings[k], rings[k + 1]
        for j in range(m):
            jn = (j + 1) % m
            bm.faces.new((r0[j], r0[jn], r1[jn], r1[j]))

    if cap_quad_pairs:
        for (i0, i1, i2, i3) in cap_quad_pairs:
            bm.faces.new((rings[0][i3], rings[0][i2], rings[0][i1], rings[0][i0]))
            bm.faces.new((rings[-1][i0], rings[-1][i1], rings[-1][i2], rings[-1][i3]))


def bm_add_curved_cyl_panel(bm: bmesh.types.BMesh, phi0_deg: float, phi1_deg: float,
                            y0: float, y1: float, r_in: float, r_out: float,
                            n_phi: int = 16, n_y: int = 12, r_surf_fn=None):

    ph0, ph1 = math.radians(phi0_deg), math.radians(phi1_deg)
    t_phi = subdivide_1d_with_support_loops(max(2, n_phi), sup1=0.012, sup2=0.038)
    t_y   = subdivide_1d_with_support_loops(max(2, n_y),   sup1=0.012, sup2=0.038)

    g_in, g_out = [], []
    for fp in t_phi:
        ph = ph0 + (ph1 - ph0) * fp
        sp, cp = math.sin(ph), math.cos(ph)
        ri_col, ro_col = [], []
        for fy in t_y:
            yv = y0 + (y1 - y0) * fy
            if r_surf_fn is not None:
                r_base = r_surf_fn(yv)
                ri_v, ro_v = r_base + r_in, r_base + r_out
            else:
                ri_v, ro_v = r_in, r_out
            ri_col.append(bm.verts.new((ri_v * sp, yv, ri_v * cp)))
            ro_col.append(bm.verts.new((ro_v * sp, yv, ro_v * cp)))
        g_in.append(ri_col)
        g_out.append(ro_col)

    bm_build_beveled_quad_shell(bm, g_in, g_out, bulge_scale=0.16)


def bm_add_flattened_aero_strut(bm: bmesh.types.BMesh, p_a: Vector, p_b: Vector,
                                half_w_mid: float = 0.0156, half_h_mid: float = 0.0048,
                                end_half_w: float = 0.0044, end_half_h: float = 0.0048,
                                eye_r: float = 0.0102, eye_half_w: float = 0.0045,
                                taper_frac: float = 0.21, pin_along_x: bool = True,
                                up_hint: Vector = None, with_flutes: bool = True,
                                ribbed_pockets: int = 0):

    axis = p_b - p_a
    L = axis.length
    if L < 1e-5:
        return
    e_s = axis / L
    if pin_along_x:
        e_u = Vector((1.0, 0.0, 0.0))
        if abs(e_s.dot(e_u)) > 0.92:
            e_u = Vector((0.0, 0.0, 1.0))
        e_u = (e_u - e_s * e_u.dot(e_s)).normalized()
        e_v = e_s.cross(e_u).normalized()
    else:
        up_ref = up_hint if up_hint is not None else Vector((0.0, 0.0, 1.0))
        if abs(e_s.dot(up_ref)) > 0.95:
            up_ref = Vector((1.0, 0.0, 0.0))
        e_v = (up_ref - e_s * up_ref.dot(e_s)).normalized()
        e_u = e_v.cross(e_s).normalized()

    s0 = min(0.16, (eye_r * 0.66) / L)
    s1 = 1.0 - s0
    p_lo = taper_frac + 0.015
    p_hi = 1.0 - taper_frac - 0.015
    bay_ranges = []
    if ribbed_pockets > 0 and p_hi > p_lo + 0.08:
        span_p = p_hi - p_lo
        rib_gap = 0.024
        bay_len = (span_p - rib_gap * (ribbed_pockets - 1)) / ribbed_pockets
        pts_mid = []
        for kb in range(ribbed_pockets):
            ba = p_lo + kb * (bay_len + rib_gap)
            bb = ba + bay_len
            bay_ranges.append((ba, bb))
            pts_mid.extend([ba - 0.005, ba, ba + 0.005, 0.5 * (ba + bb), bb - 0.005, bb, bb + 0.005])
        s_vals = sorted(set([
            0.0, 0.010, 0.028, 0.08, 0.14,
            taper_frac - 0.016, taper_frac,
            1.0 - taper_frac, 1.0 - taper_frac + 0.016,
            0.86, 0.92, 0.972, 0.990, 1.0,
        ] + pts_mid))
    else:
        s_vals = sorted([
            0.0, 0.010, 0.028,
            0.08, 0.14,
            taper_frac - 0.018, taper_frac, taper_frac + 0.020,
            0.36, 0.50, 0.64,
            1.0 - taper_frac - 0.020, 1.0 - taper_frac, 1.0 - taper_frac + 0.018,
            0.86, 0.92,
            0.972, 0.990, 1.0,
        ])

    u_eta = [
        -1.00, -0.95, -0.86,
        -0.64, -0.58, -0.52,
        -0.46, -0.42,  0.00, +0.42, +0.46,
        +0.52, +0.58, +0.64,
        +0.86, +0.95, +1.00,
    ]

    g_bot, g_top = [], []
    for fs in s_vals:
        s_real = s0 + (s1 - s0) * fs
        ctr = p_a + axis * s_real
        d_end = min(fs, 1.0 - fs)
        t_raw = min(1.0, max(0.0, d_end / max(0.05, taper_frac)))
        f_wide = t_raw * t_raw * (3.0 - 2.0 * t_raw)
        hw = end_half_w + (half_w_mid - end_half_w) * f_wide
        hh = end_half_h + (half_h_mid - end_half_h) * f_wide

        in_bay_s = 0.0
        if bay_ranges:
            for (ba, bb) in bay_ranges:
                if ba < fs < bb:
                    in_bay_s = max(in_bay_s, min(1.0, min(fs - ba, bb - fs) / 0.005))
        flute_env = min(1.0, max(0.0, (d_end - taper_frac * 0.85) / 0.06)) if with_flutes else 0.0

        rb_row, rt_row = [], []
        for eu in u_eta:
            au = abs(eu)
            if au <= 0.54:
                fac = 1.0
            elif au <= 0.58:
                tf_step = (au - 0.54) / 0.04
                fac = 1.0 - 0.24 * tf_step
            else:
                tf = (au - 0.58) / 0.42
                fac = 0.76 - 0.28 * (tf * tf * (3.0 - 2.0 * tf))

            if bay_ranges:
                lat_pkt = min(1.0, max(0.0, (0.46 - au) / 0.04))
                flute_dip = 0.54 * hh * in_bay_s * lat_pkt
            else:
                flute_dip = (0.18 * hh * flute_env * (1.0 - min(1.0, (au / 0.42) ** 4))) if au < 0.44 else 0.0

            h_cur = max(0.0007, hh * fac - flute_dip)
            u_off = e_u * (eu * hw)
            rb_row.append(bm.verts.new(ctr + u_off - e_v * h_cur))
            rt_row.append(bm.verts.new(ctr + u_off + e_v * h_cur))
        g_bot.append(rb_row)
        g_top.append(rt_row)

    is_thick_strut = max(half_h_mid, end_half_h) > 0.008
    bm_build_beveled_quad_shell(
        bm, g_bot, g_top,
        bulge_scale=0.06 if is_thick_strut else 0.16,
        max_bevel=0.0020,
        inset_bevel=is_thick_strut,
    )


    r_neck = min(end_half_w, end_half_h) * 0.98
    for pt, sgn in ((p_a, +1.0), (p_b, -1.0)):
        bm_add_segmented_tube(
            bm, pt + e_s * (sgn * eye_r * 0.70), pt + e_s * (sgn * eye_r * 1.24),
            [(0.0, r_neck * 0.92), (0.24, r_neck * 1.04), (0.76, r_neck * 1.04), (1.0, r_neck * 0.88)],
            segs=20, cap_ends=True,
        )


    for pt in (p_a, p_b):
        if pin_along_x:
            bm_add_chamfered_cyl_x(
                bm, pt.y, pt.z,
                pt.x - eye_half_w, pt.x + eye_half_w,
                radius=eye_r, segs=32, chamfer=0.00085, hollow_r=eye_r * 0.46,
            )
            for sx_cr in (-1.0, +1.0):
                bm_add_chamfered_cyl_x(
                    bm, pt.y, pt.z,
                    pt.x + sx_cr * eye_half_w * 0.96, pt.x + sx_cr * eye_half_w * 1.10,
                    radius=eye_r * 0.76, segs=24, chamfer=0.00020, hollow_r=eye_r * 0.62,
                )
            bm_add_chamfered_cyl_x(
                bm, pt.y, pt.z,
                pt.x - eye_half_w * 1.20, pt.x + eye_half_w * 1.20,
                radius=eye_r * 0.65, segs=24, chamfer=0.00055, hollow_r=eye_r * 0.40,
            )
        else:
            bm_add_segmented_tube(
                bm, pt - e_v * eye_half_w, pt + e_v * eye_half_w,
                [
                    (0.0, eye_r * 0.84), (0.10, eye_r), (0.90, eye_r), (1.0, eye_r * 0.84),
                ],
                segs=26, cap_ends=True,
            )
            bm_add_segmented_tube(
                bm, pt - e_v * (eye_half_w * 1.32), pt + e_v * (eye_half_w * 1.32),
                [
                    (0.0, eye_r * 0.44), (0.12, eye_r * 0.56), (0.88, eye_r * 0.56), (1.0, eye_r * 0.44),
                ],
                segs=22, cap_ends=True,
            )


def bm_add_machined_clevis_trunnion(bm: bmesh.types.BMesh, pin_pos: Vector,
                                    open_dir: Vector, n_dir: Vector,
                                    slot_half_h: float = 0.0023,
                                    base_drop: float = 0.0068):

    u_dir = (open_dir - n_dir * open_dir.dot(n_dir)).normalized()
    v_dir = n_dir.cross(u_dir).normalized()

    u_grid = [-0.0138, -0.0130, -0.0108, -0.0076, -0.0056, -0.0028, 0.0000, +0.0028, +0.0056, +0.0076, +0.0094, +0.0102]
    v_eta = [-1.00, -0.92, -0.76, -0.62, -0.32, 0.00, +0.32, +0.62, +0.76, +0.92, +1.00]

    def _hw_profile(uv: float) -> float:
        hw = 0.0066
        if uv < -0.0068:
            hw -= 0.0013 * min(1.0, (-0.0068 - uv) / 0.0070)
        elif uv > +0.0072:
            hw -= 0.0023 * min(1.0, (uv - 0.0072) / 0.0030)
        return hw


    gb_top, gt_top = [], []
    gb_bot, gt_bot = [], []
    jaw_thick = 0.0031
    for uv in u_grid:
        hw_u = _hw_profile(uv)
        p_u_env = min(1.0, max(0.0, min((uv - (-0.0114)) / 0.0018, (0.0076 - uv) / 0.0018)))
        r_tb, r_tt = [], []
        r_bb, r_bt = [], []
        for ev in v_eta:
            av = abs(ev)
            p_v_env = min(1.0, max(0.0, (0.68 - av) / 0.12))

            rad_pin = math.hypot(uv, ev * hw_u)
            boss_keep = min(1.0, max(0.0, (rad_pin - 0.0040) / 0.0014))
            pocket_dip = 0.00085 * p_u_env * p_v_env * boss_keep
            p_xy = pin_pos + u_dir * uv + v_dir * (ev * hw_u)
            r_tb.append(bm.verts.new(p_xy + n_dir * slot_half_h))
            r_tt.append(bm.verts.new(p_xy + n_dir * (slot_half_h + jaw_thick - pocket_dip)))
            r_bb.append(bm.verts.new(p_xy + n_dir * (-slot_half_h - jaw_thick + 0.5 * pocket_dip)))
            r_bt.append(bm.verts.new(p_xy + n_dir * (-slot_half_h)))
        gb_top.append(r_tb)
        gt_top.append(r_tt)
        gb_bot.append(r_bb)
        gt_bot.append(r_bt)
    bm_build_beveled_quad_shell(bm, gb_top, gt_top, bulge_scale=0.14)
    bm_build_beveled_quad_shell(bm, gb_bot, gt_bot, bulge_scale=0.14)


    t_s5 = subdivide_1d_with_support_loops(5, sup1=0.04, sup2=0.14)
    gb_nk, gt_nk = [], []
    for fu in t_s5:
        uv = -0.0136 + 0.0080 * fu
        hw_nk = _hw_profile(uv) * 0.88
        rb, rt = [], []
        for fv in t_s5:
            vv = (-1.0 + 2.0 * fv) * hw_nk
            p_xy = pin_pos + u_dir * uv + v_dir * vv
            rb.append(bm.verts.new(p_xy - n_dir * (slot_half_h + 0.0004)))
            rt.append(bm.verts.new(p_xy + n_dir * (slot_half_h + 0.0004)))
        gb_nk.append(rb)
        gt_nk.append(rt)
    bm_build_beveled_quad_shell(bm, gb_nk, gt_nk, bulge_scale=0.16)


    gb_ft, gt_ft = [], []
    for fu in t_s5:
        uv = -0.0160 + 0.0092 * fu
        rb, rt = [], []
        for fv in t_s5:
            vv = (-1.0 + 2.0 * fv) * 0.0066
            p_xy = pin_pos + u_dir * uv + v_dir * vv
            rb.append(bm.verts.new(p_xy - n_dir * base_drop))
            rt.append(bm.verts.new(p_xy - n_dir * (slot_half_h + jaw_thick - 0.0006)))
        gb_ft.append(rb)
        gt_ft.append(rt)
    bm_build_beveled_quad_shell(bm, gb_ft, gt_ft, bulge_scale=0.15)

    for side_v in (-1.0, +1.0):
        gb_gs, gt_gs = [], []
        for fu in subdivide_1d_with_support_loops(4, sup1=0.05, sup2=0.16):
            uv = -0.0154 + 0.0086 * fu
            z_top_gs = (-slot_half_h - jaw_thick + 0.0004) + (jaw_thick + 2.0 * slot_half_h + 0.0018) * fu
            rb, rt = [], []
            for fv in subdivide_1d_with_support_loops(4, sup1=0.06, sup2=0.18):
                vv = side_v * (0.0036 + 0.0020 * fv)
                p_xy = pin_pos + u_dir * uv + v_dir * vv
                rb.append(bm.verts.new(p_xy - n_dir * (base_drop - 0.0004)))
                rt.append(bm.verts.new(p_xy + n_dir * z_top_gs))
            gb_gs.append(rb)
            gt_gs.append(rt)
        bm_build_beveled_quad_shell(bm, gb_gs, gt_gs, bulge_scale=0.16)
        p_rbolt = pin_pos + u_dir * (-0.0148) + v_dir * (side_v * 0.0042) - n_dir * (slot_half_h + jaw_thick - 0.0005)
        bm_add_aerospace_bolt(bm, p_rbolt, n_dir, head_r=0.0014, height=0.0015)

    p_rbolt_l = pin_pos + u_dir * (-0.0148) + v_dir * (-0.0042) - n_dir * (slot_half_h + jaw_thick - 0.0005)
    p_rbolt_r = pin_pos + u_dir * (-0.0148) + v_dir * (+0.0042) - n_dir * (slot_half_h + jaw_thick - 0.0005)
    bm_add_bolt_pair_lockwire(bm, p_rbolt_l, p_rbolt_r, n_dir, head_r=0.0014, strand_r=0.00023)


    z_top_face = slot_half_h + jaw_thick
    z_bot_face = -slot_half_h - jaw_thick
    bm_add_segmented_tube(
        bm, pin_pos + n_dir * (z_bot_face - 0.0028), pin_pos + n_dir * (z_top_face + 0.0006),
        [(0.0, 0.0018), (0.12, 0.0023), (0.88, 0.0023), (1.0, 0.0020)],
        segs=20, cap_ends=True,
    )
    bm_add_segmented_tube(
        bm, pin_pos + n_dir * (z_top_face - 0.0003), pin_pos + n_dir * (z_top_face + 0.0008),
        [(0.0, 0.0041), (0.45, 0.0041), (0.85, 0.0036), (1.0, 0.0028)],
        segs=20, cap_ends=True,
    )
    p_cbolt = pin_pos + n_dir * (z_top_face + 0.0006)
    bm_add_aerospace_bolt(bm, p_cbolt, n_dir, head_r=0.0028, height=0.0024)

    p_kbolt = pin_pos + u_dir * (-0.0114) + v_dir * (+0.0024) + n_dir * z_top_face
    bm_add_aerospace_bolt(bm, p_kbolt, n_dir, head_r=0.00155, height=0.00155)
    t_ck = (u_dir * (-0.0114) + v_dir * (+0.0024)).normalized()
    s_ck = n_dir.cross(t_ck).normalized()
    c1_lw = pin_pos + n_dir * (z_top_face + 0.00185)
    c2_lw = pin_pos + u_dir * (-0.0114) + v_dir * (+0.0024) + n_dir * (z_top_face + 0.00095)
    bm_add_twisted_lockwire(
        bm,
        [
            c1_lw - t_ck * (0.0028 * 0.86) + s_ck * (0.0028 * 0.48),
            c1_lw + s_ck * (0.0028 * 0.96),
            c1_lw + t_ck * (0.0028 * 0.98) + s_ck * (0.0028 * 0.52),
            c1_lw.lerp(c2_lw, 0.52) + n_dir * 0.00025,
            c2_lw - t_ck * (0.00155 * 0.98) - s_ck * (0.00155 * 0.55),
            c2_lw - s_ck * (0.00155 * 0.96),
            c2_lw + t_ck * (0.00155 * 0.88) - s_ck * (0.00155 * 0.35),
            c2_lw + t_ck * (0.00155 * 1.90) + s_ck * (0.00155 * 0.45) - n_dir * 0.00022,
        ],
        strand_r=0.00028, twists_per_m=245.0, n_samples=28, segs=12,
    )


    bm_add_segmented_tube(
        bm, pin_pos + n_dir * (z_bot_face + 0.0002), pin_pos + n_dir * (z_bot_face - 0.0007),
        [(0.0, 0.0040), (0.55, 0.0040), (1.0, 0.0032)],
        segs=20, cap_ends=True,
    )
    bm_add_bevel_locked_hex(
        bm, pin_pos + n_dir * (z_bot_face - 0.0006), pin_pos + n_dir * (z_bot_face - 0.0024),
        [(0.0, 0.0031), (0.18, 0.0033), (0.82, 0.0033), (1.0, 0.0029)],
        corner_bevel=0.14, cap_ends=True,
    )
    p_cot = pin_pos + n_dir * (z_bot_face - 0.0021)
    bm_add_segmented_tube(
        bm, p_cot - v_dir * 0.0042, p_cot + v_dir * 0.0042,
        [(0.0, 0.00045), (0.15, 0.00058), (0.85, 0.00058), (1.0, 0.00078)],
        segs=12, cap_ends=True,
    )


def bm_add_slotted_ring_strap(bm: bmesh.types.BMesh, y_start: float = 0.0120, y_end: float = 0.1055,
                              half_w: float = 0.0086, z_center: float = 0.0, half_th: float = 0.00085):

    span = y_end - y_start
    y0 = y_start
    y1 = y_start + 0.115 * span
    y2 = y_start + 0.375 * span
    y3 = y_start + 0.635 * span
    y4 = y_end

    fx_top = [-1.00, -0.72, -0.36, 0.00, +0.36, +0.72, +1.00]
    fy_side = [0.00, 0.14, 0.38, 0.50, 0.62, 0.86, 1.00]

    def _edge_z(ae_v: float):
        ec = 0.30 * half_th * max(0.0, (ae_v - 0.82) / 0.18)
        return (z_center - half_th + 0.5 * ec, z_center + half_th - ec)

    v_cache = {}
    def _get_grid_pair(xv: float, yv: float, ae_v: float):
        key = (round(xv * 1e6), round(yv * 1e6))
        if key not in v_cache:
            zb, zt = _edge_z(ae_v)
            v_cache[key] = (bm.verts.new((xv, yv, zb)), bm.verts.new((xv, yv, zt)))
        return v_cache[key]


    perim_loops_bot = []
    perim_loops_top = []

    y_root_rows = [y0 + (y1 - y0) * f for f in [0.00, 0.08, 0.24, 0.55, 0.84, 1.00]]
    gb_rt, gt_rt = [], []
    for yr in y_root_rows:
        rb, rt = [], []
        for fx in fx_top:
            vb, vt = _get_grid_pair(fx * half_w, yr, abs(fx))
            rb.append(vb)
            rt.append(vt)
        gb_rt.append(rb)
        gt_rt.append(rt)
    for r in range(len(y_root_rows) - 1):
        for c in range(6):
            bm.faces.new((gt_rt[r][c], gt_rt[r][c + 1], gt_rt[r + 1][c + 1], gt_rt[r + 1][c]))
            bm.faces.new((gb_rt[r][c], gb_rt[r + 1][c], gb_rt[r + 1][c + 1], gb_rt[r][c + 1]))


    bay_sides_right = []
    bay_sides_left = []
    for ya_b, yb_b in ((y1, y2), (y2, y3)):
        yc_b = 0.5 * (ya_b + yb_b)
        hy_b = 0.5 * (yb_b - ya_b)

        out_pts = []
        for c in range(6):
            out_pts.append((fx_top[c] * half_w, ya_b, abs(fx_top[c])))
        for c in range(6):
            out_pts.append((+half_w, ya_b + (yb_b - ya_b) * fy_side[c], 1.0))
        for c in range(6):
            fx_rev = fx_top[6 - c]
            out_pts.append((fx_rev * half_w, yb_b, abs(fx_rev)))
        for c in range(6):
            fy_rev = fy_side[6 - c]
            out_pts.append((-half_w, ya_b + (yb_b - ya_b) * fy_rev, 1.0))


        sw_s = 0.44 * half_w
        sl_s = 0.40 * hy_b
        in_pts = []
        for (xo, yo, _) in out_pts:
            ang = math.atan2(yo - yc_b, xo)
            ca, sa = math.cos(ang), math.sin(ang)

            xi = sw_s * ca
            yi_cap = sw_s * sa
            if sa > 0.0:
                yi = yc_b + sl_s * (abs(sa) ** 0.55) + 0.82 * yi_cap
            elif sa < 0.0:
                yi = yc_b - sl_s * (abs(sa) ** 0.55) + 0.82 * yi_cap
            else:
                yi = yc_b
            in_pts.append((xi, yi))


        rad_specs = [
            (0.00, 0.00, 0.00),
            (0.10, 0.00, 0.00),
            (0.46, 0.00, 0.00),
            (0.56, +1.25 * half_th, +0.35 * half_th),
            (0.86, +1.25 * half_th, +0.35 * half_th),
            (0.95, +0.85 * half_th, +0.45 * half_th),
            (1.00, +0.42 * half_th, +0.58 * half_th),
        ]
        rings_b, rings_t = [], []
        for r_idx, (fr, dz_t, dz_b) in enumerate(rad_specs):
            rb_ring, rt_ring = [], []
            for k in range(24):
                xo, yo, aeo = out_pts[k]
                if r_idx == 0:
                    vb, vt = _get_grid_pair(xo, yo, aeo)
                else:
                    xi, yi = in_pts[k]
                    xk = xo + (xi - xo) * fr
                    yk = yo + (yi - yo) * fr
                    zb0, zt0 = _edge_z(aeo * (1.0 - fr))
                    vb = bm.verts.new((xk, yk, zb0 + dz_b))
                    vt = bm.verts.new((xk, yk, zt0 + dz_t))
                rb_ring.append(vb)
                rt_ring.append(vt)
            rings_b.append(rb_ring)
            rings_t.append(rt_ring)

        for r_idx in range(len(rad_specs) - 1):
            for k in range(24):
                kn = (k + 1) % 24
                bm.faces.new((rings_t[r_idx][k], rings_t[r_idx][kn], rings_t[r_idx + 1][kn], rings_t[r_idx + 1][k]))
                bm.faces.new((rings_b[r_idx][k], rings_b[r_idx + 1][k], rings_b[r_idx + 1][kn], rings_b[r_idx][kn]))


        in_mid_ring = []
        for k in range(24):
            xi, yi = in_pts[k]
            in_mid_ring.append(bm.verts.new((xi * 0.985, yi, z_center + 0.45 * half_th)))
        for k in range(24):
            kn = (k + 1) % 24
            bm.faces.new((rings_t[-1][k], rings_t[-1][kn], in_mid_ring[kn], in_mid_ring[k]))
            bm.faces.new((in_mid_ring[k], in_mid_ring[kn], rings_b[-1][kn], rings_b[-1][k]))


        for c in range(1, 6):
            bay_sides_right.append((rings_b[0][6 + c], rings_t[0][6 + c]))
        if yb_b < y3 - 1e-6:
            bay_sides_right.append((rings_b[0][12], rings_t[0][12]))
        for c in range(1, 6):
            bay_sides_left.append((rings_b[0][24 - c], rings_t[0][24 - c]))
        if yb_b < y3 - 1e-6:
            bay_sides_left.append((rings_b[0][18], rings_t[0][18]))


    y_tail_rows = [y3 + (y4 - y3) * f for f in [0.00, 0.16, 0.48, 0.78, 0.92, 1.00]]
    gb_tl, gt_tl = [], []
    for idx_t, yt in enumerate(y_tail_rows):
        fy_gl = (yt - y0) / span
        tp = 1.0 - 0.20 * max(0.0, (fy_gl - 0.88) / 0.12) if idx_t > 0 else 1.0
        rb, rt = [], []
        for fx in fx_top:
            vb, vt = _get_grid_pair(fx * half_w * tp, yt, abs(fx))
            rb.append(vb)
            rt.append(vt)
        gb_tl.append(rb)
        gt_tl.append(rt)
    for r in range(len(y_tail_rows) - 1):
        for c in range(6):
            bm.faces.new((gt_tl[r][c], gt_tl[r][c + 1], gt_tl[r + 1][c + 1], gt_tl[r + 1][c]))
            bm.faces.new((gb_tl[r][c], gb_tl[r + 1][c], gb_tl[r + 1][c + 1], gb_tl[r][c + 1]))


    for c in range(6):
        perim_loops_bot.append(gb_rt[0][c])
        perim_loops_top.append(gt_rt[0][c])
    for r in range(len(y_root_rows)):
        perim_loops_bot.append(gb_rt[r][6])
        perim_loops_top.append(gt_rt[r][6])
    for vb_s, vt_s in bay_sides_right:
        perim_loops_bot.append(vb_s)
        perim_loops_top.append(vt_s)
    for r in range(len(y_tail_rows)):
        perim_loops_bot.append(gb_tl[r][6])
        perim_loops_top.append(gt_tl[r][6])
    for c in range(5, 0, -1):
        perim_loops_bot.append(gb_tl[-1][c])
        perim_loops_top.append(gt_tl[-1][c])
    for r in range(len(y_tail_rows) - 1, -1, -1):
        perim_loops_bot.append(gb_tl[r][0])
        perim_loops_top.append(gt_tl[r][0])
    for vb_s, vt_s in reversed(bay_sides_left):
        perim_loops_bot.append(vb_s)
        perim_loops_top.append(vt_s)
    for r in range(len(y_root_rows) - 1, 0, -1):
        perim_loops_bot.append(gb_rt[r][0])
        perim_loops_top.append(gt_rt[r][0])

    n_per = len(perim_loops_bot)
    perim_mid = []
    for k in range(n_per):
        cb = perim_loops_bot[k].co
        ct = perim_loops_top[k].co
        nx_p = 1.0 if cb.x > 0.002 else (-1.0 if cb.x < -0.002 else 0.0)
        ny_p = -1.0 if cb.y < y0 + 1e-4 else (+1.0 if cb.y > y4 - 1e-4 else 0.0)
        out_n = Vector((nx_p, ny_p, 0.0))
        if out_n.length > 1e-5:
            out_n.normalize()
        perim_mid.append(bm.verts.new(0.5 * (cb + ct) + out_n * (0.16 * half_th)))

    for k in range(n_per):
        kn = (k + 1) % n_per
        bm.faces.new((perim_loops_bot[k], perim_loops_bot[kn], perim_mid[kn], perim_mid[k]))
        bm.faces.new((perim_mid[k], perim_mid[kn], perim_loops_top[kn], perim_loops_top[k]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)


def bm_add_ring_channel_saddle(bm: bmesh.types.BMesh, center_yz: tuple,
                               t_dir: Vector, n_dir: Vector,
                               half_span_x: float = 0.0365, half_chord_a: float = 0.0160,
                               h_foot: float = -0.0148, h_tunnel_bot: float = -0.0032,
                               h_tunnel_top: float = +0.0032, h_roof: float = +0.0068,
                               rho_curve: float = 0.50, is_through_tunnel: bool = True):

    x_dir = Vector((1.0, 0.0, 0.0))
    c_pos = Vector((0.0, center_yz[0], center_yz[1]))

    def _pt_sad(xv: float, av: float, hv: float) -> Vector:
        drop = -(xv * xv) / (2.0 * rho_curve)
        return c_pos + x_dir * xv + t_dir * av + n_dir * (hv + drop)


    fx_list = [
        -1.00, -0.95, -0.82, -0.74, -0.68, -0.48, -0.28, -0.22, -0.14, 0.00,
        +0.14, +0.22, +0.28, +0.48, +0.68, +0.74, +0.82, +0.95, +1.00,
    ]
    fa_list = [-1.00, -0.94, -0.80, -0.58, -0.50, -0.38, 0.00, +0.38, +0.50, +0.58, +0.80, +0.94, +1.00]
    gb_roof, gt_roof = [], []
    for fx in fx_list:
        ax = abs(fx)
        chord_scale = 1.0 - 0.44 * max(0.0, (ax - 0.68) / 0.32)
        xv = fx * half_span_x
        w_pkt_x = min(1.0, max(0.0, min((ax - 0.24) / 0.05, (0.72 - ax) / 0.05)))
        rb, rt = [], []
        for fa in fa_list:
            aa = abs(fa)
            rib_h = 0.0032 * min(1.0, max(0.0, (aa - 0.50) / 0.07)) * (1.0 - 0.30 * max(0.0, (ax - 0.78) / 0.22))
            w_pkt_a = min(1.0, max(0.0, (0.42 - aa) / 0.08))
            pkt_dip = 0.00145 * w_pkt_x * w_pkt_a
            av = fa * half_chord_a * chord_scale
            h_b = h_tunnel_top if is_through_tunnel else (h_foot + 0.0028 * min(1.0, ax / 0.45))
            rb.append(bm.verts.new(_pt_sad(xv, av, h_b)))
            rt.append(bm.verts.new(_pt_sad(xv, av, h_roof + rib_h - pkt_dip)))
        gb_roof.append(rb)
        gt_roof.append(rt)
    bm_build_beveled_quad_shell(bm, gb_roof, gt_roof, bulge_scale=0.12)

    if is_through_tunnel:

        t_s6 = subdivide_1d_with_support_loops(6, sup1=0.03, sup2=0.10)
        gb_bot, gt_bot = [], []
        for fx in t_s6:
            xv = (-0.72 + 1.44 * fx) * half_span_x
            rb, rt = [], []
            for fa in t_s6:
                av = (-0.96 + 1.92 * fa) * half_chord_a
                rb.append(bm.verts.new(_pt_sad(xv, av, h_tunnel_bot - 0.0026)))
                rt.append(bm.verts.new(_pt_sad(xv, av, h_tunnel_bot)))
            gb_bot.append(rb)
            gt_bot.append(rt)
        bm_build_beveled_quad_shell(bm, gb_bot, gt_bot, bulge_scale=0.15)

        for side_a in (-1.0, +1.0):
            gb_w, gt_w = [], []
            for fx in t_s6:
                xv = (-0.78 + 1.56 * fx) * half_span_x
                flare = 0.0014 * max(0.0, (abs(-0.78 + 1.56 * fx) - 0.56) / 0.22)
                rb, rt = [], []
                for fa in subdivide_1d_with_support_loops(4, sup1=0.05, sup2=0.16):
                    av = side_a * ((0.70 + 0.24 * fa) * half_chord_a + flare)
                    rb.append(bm.verts.new(_pt_sad(xv, av, h_tunnel_bot - 0.0004)))
                    rt.append(bm.verts.new(_pt_sad(xv, av, h_tunnel_top + 0.0004)))
                gb_w.append(rb)
                gt_w.append(rt)
            bm_build_beveled_quad_shell(bm, gb_w, gt_w, bulge_scale=0.15)


        for sx_r in (-1.0, +1.0):
            for sa_r in (-1.0, +1.0):
                xr_p = sx_r * 0.64 * half_span_x
                ar_p = sa_r * 0.72 * half_chord_a
                bm_add_segmented_tube(
                    bm, _pt_sad(xr_p, ar_p, h_tunnel_bot - 0.0005), _pt_sad(xr_p, ar_p, h_tunnel_top + 0.0005),
                    [(0.0, 0.0020), (0.18, 0.0025), (0.82, 0.0025), (1.0, 0.0020)],
                    segs=16, cap_ends=True,
                )
                bm_add_aerospace_bolt(bm, _pt_sad(xr_p, ar_p, h_roof + 0.0026), n_dir, head_r=0.0016, height=0.0016)


        t_s4 = subdivide_1d_with_support_loops(4, sup1=0.05, sup2=0.16)
        for sx_c in (-1.0, +1.0):
            gb_cw, gt_cw = [], []
            for fx in t_s4:
                xv = sx_c * (0.00090 + 0.00280 * fx)
                rb, rt = [], []
                for fa in t_s6:
                    av = (-0.88 + 1.76 * fa) * half_chord_a
                    rb.append(bm.verts.new(_pt_sad(xv, av, h_foot)))
                    rt.append(bm.verts.new(_pt_sad(xv, av, h_tunnel_bot - 0.0018)))
                gb_cw.append(rb)
                gt_cw.append(rt)
            bm_build_beveled_quad_shell(bm, gb_cw, gt_cw, bulge_scale=0.14)


        for sx_bw in (-1.0, +1.0):
            xb_w = sx_bw * 0.36 * half_span_x
            pb_s0 = _pt_sad(xb_w, -0.26 * half_chord_a, h_roof - 0.0013)
            pb_s1 = _pt_sad(xb_w, +0.26 * half_chord_a, h_roof - 0.0013)
            bm_add_aerospace_bolt(bm, pb_s0, n_dir, head_r=0.0017, height=0.0018)
            bm_add_aerospace_bolt(bm, pb_s1, n_dir, head_r=0.0017, height=0.0018)
            bm_add_bolt_pair_lockwire(bm, pb_s0, pb_s1, n_dir, head_r=0.0017, strand_r=0.00026)
        bm_add_grease_zerk(bm, _pt_sad(0.0, -0.36 * half_chord_a, h_roof), n_dir, scale=0.68)


    p_bolt = _pt_sad(0.0, 0.0, h_roof)
    bm_add_segmented_tube(
        bm, p_bolt - n_dir * 0.0004, p_bolt + n_dir * 0.0014,
        [(0.0, 0.0058), (0.40, 0.0058), (0.72, 0.0046), (1.0, 0.0038)],
        segs=22, cap_ends=True,
    )
    bm_add_aerospace_bolt(bm, p_bolt + n_dir * 0.0011, n_dir, head_r=0.0036, height=0.0034)


def build_isogrid_cased_cylinder(bm: bmesh.types.BMesh, y0: float, y1: float,
                                 r_inner: float, r_outer: float,
                                 n_circ: int = 24, n_rows: int = 6,
                                 pocket_depth: float = 0.0050,
                                 radial_cap_fn=None,
                                 r_inner_end: float = None,
                                 r_outer_end: float = None,
                                 axial_ribs: bool = True):

    assert n_rows >= 2, "n_rows must be >= 2"
    if axial_ribs:
        assert n_circ >= 4 and (n_circ % 2 == 0), "n_circ must be a positive even integer in axial_ribs mode"
    r_in1  = r_inner if r_inner_end is None else r_inner_end
    r_out1 = r_outer if r_outer_end is None else r_outer_end
    two_pi = 2.0 * math.pi
    dphi = two_pi / float(n_circ)
    dy = (y1 - y0) / float(n_rows)
    span_y = max(1e-6, y1 - y0)

    def _pt3d(u_phi: float, yv: float, dr_p: float) -> Vector:
        pw = math.fmod(u_phi, two_pi)
        if pw < 0.0:
            pw += two_pi
        ty = max(0.0, min(1.0, (yv - y0) / span_y))
        rc = (r_outer + (r_out1 - r_outer) * ty) - dr_p
        if radial_cap_fn is not None:
            rc = min(rc, radial_cap_fn(pw, yv))
        return Vector((rc * math.sin(pw), yv, rc * math.cos(pw)))

    corner_verts = {}
    corner_uv = {}
    if axial_ribs:
        for i in range(n_circ):
            u_val = i * dphi
            if i % 2 == 0:
                for m in range(n_rows + 1):
                    yv = y0 + m * dy
                    corner_uv[(i, m)] = (u_val, yv)
                    corner_verts[(i, m)] = bm.verts.new(_pt3d(u_val, yv, 0.0))
            else:
                y_list = [y0] + [y0 + (m + 0.5) * dy for m in range(n_rows)] + [y1]
                for m, yv in enumerate(y_list):
                    corner_uv[(i, m)] = (u_val, yv)
                    corner_verts[(i, m)] = bm.verts.new(_pt3d(u_val, yv, 0.0))
    else:
        for j in range(n_rows + 1):
            yj = y0 + j * dy
            off = 0.5 * dphi if (j % 2 == 1) else 0.0
            for i in range(n_circ):
                u_val = off + i * dphi
                corner_uv[(j, i)] = (u_val, yj)
                corner_verts[(j, i)] = bm.verts.new(_pt3d(u_val, yj, 0.0))

    edge_cache = {}
    def _get_edge_mids(k1: tuple, u1: float, v1: float, k2: tuple, u2: float, v2: float) -> list:
        ka, kb = (k1, k2) if k1 < k2 else (k2, k1)
        key = (ka, kb)
        if key not in edge_cache:
            ua, va = corner_uv[ka]
            ub, vb = corner_uv[kb]
            if ub - ua > math.pi:
                ub -= two_pi
            elif ub - ua < -math.pi:
                ub += two_pi
            mids = [
                bm.verts.new(_pt3d(ua + (ub - ua) * t, va + (vb - va) * t, 0.0))
                for t in (0.25, 0.50, 0.75)
            ]
            edge_cache[key] = mids
        mids = edge_cache[key]
        return mids if k1 == ka else [mids[2], mids[1], mids[0]]

    inset_specs_full = [
        (0.110, 0.00, 0.02),
        (0.142, 0.00, 0.05),
        (0.222, 1.00, 0.09),
        (0.275, 1.00, 0.10),
        (0.540, 1.00, 0.11),
    ]
    inset_specs_half = [
        (0.165, 0.00, 0.02),
        (0.205, 0.00, 0.05),
        (0.300, 0.85, 0.09),
        (0.360, 0.85, 0.10),
        (0.580, 0.85, 0.11),
    ]

    all_tris = []
    if axial_ribs:
        for i in range(n_circ):
            inext = (i + 1) % n_circ
            u0, u1 = i * dphi, (i + 1.0) * dphi
            if i % 2 == 0:
                all_tris.append(((i, 0), u0, y0, (inext, 0), u1, y0, (inext, 1), u1, y0 + 0.5 * dy, True))
                for m in range(n_rows):
                    all_tris.append(((i, m), u0, y0 + m * dy,
                                     (inext, m + 1), u1, y0 + (m + 0.5) * dy,
                                     (i, m + 1), u0, y0 + (m + 1) * dy, False))
                for m in range(1, n_rows):
                    all_tris.append(((i, m), u0, y0 + m * dy,
                                     (inext, m), u1, y0 + (m - 0.5) * dy,
                                     (inext, m + 1), u1, y0 + (m + 0.5) * dy, False))
                all_tris.append(((i, n_rows), u0, y1, (inext, n_rows), u1, y1 - 0.5 * dy, (inext, n_rows + 1), u1, y1, True))
            else:
                all_tris.append(((i, 0), u0, y0, (inext, 0), u1, y0, (i, 1), u0, y0 + 0.5 * dy, True))
                for m in range(n_rows):
                    all_tris.append(((inext, m), u1, y0 + m * dy,
                                     (inext, m + 1), u1, y0 + (m + 1) * dy,
                                     (i, m + 1), u0, y0 + (m + 0.5) * dy, False))
                for m in range(1, n_rows):
                    all_tris.append(((i, m), u0, y0 + (m - 0.5) * dy,
                                     (inext, m), u1, y0 + m * dy,
                                     (i, m + 1), u0, y0 + (m + 0.5) * dy, False))
                all_tris.append(((i, n_rows), u0, y1 - 0.5 * dy, (inext, n_rows), u1, y1, (i, n_rows + 1), u0, y1, True))
    else:
        for j in range(n_rows):
            yj0, yj1 = y0 + j * dy, y0 + (j + 1) * dy
            even = (j % 2 == 0)
            for i in range(n_circ):
                if even:
                    all_tris.extend([
                        ((j, i), i * dphi, yj0, (j, (i + 1) % n_circ), (i + 1.0) * dphi, yj0, (j + 1, i), (i + 0.5) * dphi, yj1, False),
                        ((j, (i + 1) % n_circ), (i + 1.0) * dphi, yj0, (j + 1, (i + 1) % n_circ), (i + 1.5) * dphi, yj1, (j + 1, i), (i + 0.5) * dphi, yj1, False),
                    ])
                else:
                    all_tris.extend([
                        ((j, i), (i + 0.5) * dphi, yj0, (j, (i + 1) % n_circ), (i + 1.5) * dphi, yj0, (j + 1, (i + 1) % n_circ), (i + 1.0) * dphi, yj1, False),
                        ((j, (i + 1) % n_circ), (i + 1.5) * dphi, yj0, (j + 1, (i + 2) % n_circ), (i + 2.0) * dphi, yj1, (j + 1, (i + 1) % n_circ), (i + 1.0) * dphi, yj1, False),
                    ])

    for (ka, ua, va, kb, ub, vb, kc, uc, vc_y, is_half) in all_tris:
        cur_specs = inset_specs_half if is_half else inset_specs_full
        e_ab = _get_edge_mids(ka, ua, va, kb, ub, vb)
        e_bc = _get_edge_mids(kb, ub, vb, kc, uc, vc_y)
        e_ca = _get_edge_mids(kc, uc, vc_y, ka, ua, va)
        r0 = [corner_verts[ka]] + e_ab + [corner_verts[kb]] + e_bc + [corner_verts[kc]] + e_ca

        p_perim = [
            (ua, va),
            (0.75 * ua + 0.25 * ub, 0.75 * va + 0.25 * vb),
            (0.50 * ua + 0.50 * ub, 0.50 * va + 0.50 * vb),
            (0.25 * ua + 0.75 * ub, 0.25 * va + 0.75 * vb),
            (ub, vb),
            (0.75 * ub + 0.25 * uc, 0.75 * vb + 0.25 * vc_y),
            (0.50 * ub + 0.50 * uc, 0.50 * vb + 0.50 * vc_y),
            (0.25 * ub + 0.75 * uc, 0.25 * vb + 0.75 * vc_y),
            (uc, vc_y),
            (0.75 * uc + 0.25 * ua, 0.75 * vc_y + 0.25 * va),
            (0.50 * uc + 0.50 * ua, 0.50 * vc_y + 0.50 * va),
            (0.25 * uc + 0.75 * ua, 0.25 * vc_y + 0.75 * va),
        ]
        gu = (ua + ub + uc) / 3.0
        gv = (va + vb + vc_y) / 3.0

        rings_t = [r0]
        for (tau, dr_f, c_rnd) in cur_specs:
            dr_cur = pocket_depth * dr_f
            rk = []
            for idx_p, (pu, pv) in enumerate(p_perim):
                t_eff = tau + (c_rnd * (1.0 - tau) if idx_p in (0, 4, 8) else 0.0)
                rk.append(bm.verts.new(_pt3d(pu + (gu - pu) * t_eff, pv + (gv - pv) * t_eff, dr_cur)))
            rings_t.append(rk)

        for lr in range(len(rings_t) - 1):
            ra_l, rb_l = rings_t[lr], rings_t[lr + 1]
            for m in range(12):
                mn = (m + 1) % 12
                bm.faces.new((ra_l[m], ra_l[mn], rb_l[mn], rb_l[m]))

        r5 = rings_t[-1]
        u_p5 = []
        tau5, dr5_f, cr5 = cur_specs[-1]
        dr_bot = pocket_depth * dr5_f
        for idx_p, (pu, pv) in enumerate(p_perim):
            t_eff = tau5 + (cr5 * (1.0 - tau5) if idx_p in (0, 4, 8) else 0.0)
            u_p5.append((pu + (gu - pu) * t_eff, pv + (gv - pv) * t_eff))

        v_ctr = bm.verts.new(_pt3d(gu, gv, dr_bot))
        u_sab = (0.5 * (gu + u_p5[2][0]),  0.5 * (gv + u_p5[2][1]))
        u_sbc = (0.5 * (gu + u_p5[6][0]),  0.5 * (gv + u_p5[6][1]))
        u_sca = (0.5 * (gu + u_p5[10][0]), 0.5 * (gv + u_p5[10][1]))
        v_sab = bm.verts.new(_pt3d(u_sab[0], u_sab[1], dr_bot))
        v_sbc = bm.verts.new(_pt3d(u_sbc[0], u_sbc[1], dr_bot))
        v_sca = bm.verts.new(_pt3d(u_sca[0], u_sca[1], dr_bot))

        u_pa = (0.25 * (gu + u_sca[0] + u_p5[0][0] + u_sab[0]), 0.25 * (gv + u_sca[1] + u_p5[0][1] + u_sab[1]))
        u_pb = (0.25 * (gu + u_sab[0] + u_p5[4][0] + u_sbc[0]), 0.25 * (gv + u_sab[1] + u_p5[4][1] + u_sbc[1]))
        u_pc = (0.25 * (gu + u_sbc[0] + u_p5[8][0] + u_sca[0]), 0.25 * (gv + u_sbc[1] + u_p5[8][1] + u_sca[1]))
        v_pa = bm.verts.new(_pt3d(u_pa[0], u_pa[1], dr_bot))
        v_pb = bm.verts.new(_pt3d(u_pb[0], u_pb[1], dr_bot))
        v_pc = bm.verts.new(_pt3d(u_pc[0], u_pc[1], dr_bot))

        bm.faces.new((v_ctr, v_sca, v_pa, v_sab))
        bm.faces.new((v_sca, r5[10], r5[11], v_pa))
        bm.faces.new((v_pa, r5[11], r5[0], r5[1]))
        bm.faces.new((v_sab, v_pa, r5[1], r5[2]))

        bm.faces.new((v_ctr, v_sab, v_pb, v_sbc))
        bm.faces.new((v_sab, r5[2], r5[3], v_pb))
        bm.faces.new((v_pb, r5[3], r5[4], r5[5]))
        bm.faces.new((v_sbc, v_pb, r5[5], r5[6]))

        bm.faces.new((v_ctr, v_sbc, v_pc, v_sca))
        bm.faces.new((v_sbc, r5[6], r5[7], v_pc))
        bm.faces.new((v_pc, r5[7], r5[8], r5[9]))
        bm.faces.new((v_sca, v_pc, r5[9], r5[10]))

    if axial_ribs:
        out_y0, out_y1 = [], []
        for i in range(n_circ):
            inext = (i + 1) % n_circ
            ka0, kb0 = (i, 0), (inext, 0)
            out_y0.append(corner_verts[ka0])
            out_y0.extend(_get_edge_mids(ka0, i * dphi, y0, kb0, (i + 1) * dphi, y0))

            m_top_a = n_rows if (i % 2 == 0) else (n_rows + 1)
            m_top_b = n_rows if (inext % 2 == 0) else (n_rows + 1)
            ka1, kb1 = (i, m_top_a), (inext, m_top_b)
            out_y1.append(corner_verts[ka1])
            out_y1.extend(_get_edge_mids(ka1, i * dphi, y1, kb1, (i + 1) * dphi, y1))
    else:
        def _boundary_ring(row_idx: int) -> list:
            yv_b = y0 + row_idx * dy
            ring = []
            for i in range(n_circ):
                ka, kb = (row_idx, i), (row_idx, (i + 1) % n_circ)
                ring.append(corner_verts[ka])
                ring.extend(_get_edge_mids(ka, i * dphi, yv_b, kb, (i + 1) * dphi, yv_b))
            return ring
        out_y0 = _boundary_ring(0)
        out_y1 = _boundary_ring(n_rows)
        if n_rows % 2 == 1:
            out_y1 = out_y1[-2:] + out_y1[:-2]

    n_seg_tot = len(out_y0)
    n_in_y = 2 * n_rows
    in_rings = []
    for ky in range(n_in_y + 1):
        ty_in = ky / float(n_in_y)
        yv_in = y0 + (y1 - y0) * ty_in
        ri_cur = r_inner + (r_in1 - r_inner) * ty_in
        r_row = []
        for ip in range(n_seg_tot):
            ph = two_pi * ip / float(n_seg_tot)
            r_row.append(bm.verts.new((ri_cur * math.sin(ph), yv_in, ri_cur * math.cos(ph))))
        in_rings.append(r_row)

    for ky in range(n_in_y):
        ri0, ri1 = in_rings[ky], in_rings[ky + 1]
        for ip in range(n_seg_tot):
            ipn = (ip + 1) % n_seg_tot
            bm.faces.new((ri0[ip], ri1[ip], ri1[ipn], ri0[ipn]))

    for ip in range(n_seg_tot):
        ipn = (ip + 1) % n_seg_tot
        bm.faces.new((in_rings[0][ip], out_y0[ip], out_y0[ipn], in_rings[0][ipn]))
        bm.faces.new((out_y1[ip], in_rings[-1][ip], in_rings[-1][ipn], out_y1[ipn]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)


def build_waffle_cased_cylinder(bm: bmesh.types.BMesh, y0: float, y1: float,
                                r_inner: float, r_outer: float,
                                n_phi: int, n_y: int,
                                pocket_bands_y: list, n_pockets_phi: int = 32,
                                pocket_depth: float = 0.0042,
                                radial_cap_fn=None, y_grid_custom: list = None):

    outer_rings = []
    inner_rings = []
    y_vals = y_grid_custom if y_grid_custom is not None else [y0 + (y1 - y0) * (iy / n_y) for iy in range(n_y + 1)]
    for yv in y_vals:

        env_y = 0.0
        for (py0, py1) in pocket_bands_y:
            if py0 < yv < py1:
                span = py1 - py0
                d_edge = min(yv - py0, py1 - yv)
                ramp = min(1.0, d_edge / max(0.006, 0.14 * span))
                env_y = max(env_y, ramp)

        ring_o, ring_i = [], []
        for ip in range(n_phi):
            phi = 2.0 * math.pi * ip / n_phi

            cell_phase = (ip / n_phi) * n_pockets_phi
            frac_c = cell_phase - math.floor(cell_phase)
            d_c = min(frac_c, 1.0 - frac_c)
            env_p = min(1.0, max(0.0, (d_c - 0.11) / 0.10))

            r_cur = r_outer - pocket_depth * env_y * env_p
            if radial_cap_fn is not None:
                r_cur = min(r_cur, radial_cap_fn(phi, yv))
            sp, cp = math.sin(phi), math.cos(phi)
            ring_o.append(bm.verts.new((r_cur * sp, yv, r_cur * cp)))
            ring_i.append(bm.verts.new((r_inner * sp, yv, r_inner * cp)))
        outer_rings.append(ring_o)
        inner_rings.append(ring_i)

    n_rings_y = len(y_vals) - 1
    for iy in range(n_rings_y):
        ro0, ro1 = outer_rings[iy], outer_rings[iy + 1]
        ri0, ri1 = inner_rings[iy], inner_rings[iy + 1]
        for ip in range(n_phi):
            ipn = (ip + 1) % n_phi
            bm.faces.new((ro0[ip], ro0[ipn], ro1[ipn], ro1[ip]))
            bm.faces.new((ri0[ip], ri1[ip], ri1[ipn], ri0[ipn]))


    for ip in range(n_phi):
        ipn = (ip + 1) % n_phi
        bm.faces.new((inner_rings[0][ip], inner_rings[0][ipn], outer_rings[0][ipn], outer_rings[0][ip]))
        bm.faces.new((inner_rings[-1][ip], outer_rings[-1][ip], outer_rings[-1][ipn], inner_rings[-1][ipn]))


def bm_add_articulated_igv_stage(dst_bm: bmesh.types.BMesh, n_vanes: int = 24,
                                 y_center: float = -4.202, chord: float = 0.064,
                                 r_hub: float = 0.174, r_tip: float = 0.484):

    b_one = bmesh.new()
    r_steps = 18
    c_pts = 16
    rings_b = []
    for rs in range(r_steps + 1):
        fr = rs / float(r_steps)
        rr = r_hub + fr * (r_tip - r_hub)

        tw_le = math.radians(6.0 + 7.0 * fr)
        flap_def = math.radians(18.0 + 11.5 * fr)
        c_cur = chord * (0.86 + 0.32 * fr)

        u_samples = [
            0.000, 0.012, 0.040, 0.100, 0.200, 0.320,
            0.426, 0.442, 0.450, 0.458, 0.474,
            0.580, 0.700, 0.820, 0.920, 0.975, 1.000,
        ]
        n_u = len(u_samples) - 1
        prof_pts = []
        for ip in range(2 * n_u):
            if ip <= n_u:
                u = u_samples[ip]
                side = +1.0
            else:
                u = u_samples[2 * n_u - ip]
                side = -1.0
            yt = 0.136 * c_cur * (2.969 * math.sqrt(max(0.0, u)) - 1.260 * u - 3.516 * u * u + 2.843 * (u ** 3) - 1.015 * (u ** 4))
            hinge_pinch = 1.0 - 0.52 * math.exp(-((u - 0.450) / 0.014) ** 2)
            yt *= hinge_pinch

            s_loc = (u - 0.5) * c_cur
            if u <= 0.45:
                ang_u = tw_le
                x_c = s_loc * math.sin(ang_u)
                y_c = y_center + s_loc * math.cos(ang_u)
            else:
                s_h = (0.45 - 0.5) * c_cur
                xh = s_h * math.sin(tw_le)
                yh = y_center + s_h * math.cos(tw_le)
                ds = (u - 0.45) * c_cur
                ang_u = tw_le + flap_def * ((u - 0.45) / 0.55) ** 0.85
                x_c = xh + ds * math.sin(ang_u)
                y_c = yh + ds * math.cos(ang_u)

            ctw, stw = math.cos(ang_u), math.sin(ang_u)
            x_3d = x_c + (side * yt) * ctw
            y_3d = y_c - (side * yt) * stw
            prof_pts.append(b_one.verts.new((x_3d, y_3d, rr)))
        rings_b.append(prof_pts)

    n_ring = len(rings_b[0])
    for rs in range(r_steps):
        ra, rb = rings_b[rs], rings_b[rs + 1]
        for j in range(n_ring):
            jn = (j + 1) % n_ring
            b_one.faces.new((ra[j], rb[j], rb[jn], ra[jn]))
    _bm_cap_ring_quads(b_one, rings_b[-1], reverse=False, inset_frac=0.72)
    _bm_cap_ring_quads(b_one, rings_b[0],  reverse=True,  inset_frac=0.72)
    bmesh.ops.recalc_face_normals(b_one, faces=b_one.faces)


    bm_fill_unset_mat(b_one, MAT_SPINNER_IGV)
    f0_igv_hw = len(b_one.faces)
    bm_add_segmented_tube(
        b_one, Vector((0.0, y_center + 0.004, r_hub - 0.0025)), Vector((0.0, y_center + 0.004, r_hub + 0.0032)),
        [(0.0, 0.0185), (0.62, 0.0185), (0.74, 0.0155), (1.0, 0.0145)], segs=22, cap_ends=True,
    )
    bm_add_segmented_tube(
        b_one, Vector((0.0, y_center + 0.004, r_tip + 0.0025)), Vector((0.0, y_center + 0.004, r_tip - 0.0034)),
        [(0.0, 0.0225), (0.62, 0.0225), (0.74, 0.0195), (1.0, 0.0182)], segs=22, cap_ends=True,
    )

    prof_igv_lv = [
        (-0.0036, 0.0000), (-0.0036, 0.0014), (-0.0024, 0.0023), (+0.0024, 0.0023), (+0.0036, 0.0014), (+0.0036, 0.0000),
        (+0.0025, 0.0000), (+0.0025, 0.0009), (+0.0017, 0.0015), (-0.0017, 0.0015), (-0.0025, 0.0009), (-0.0025, 0.0000),
    ]
    igv_lv_caps = [(0, 1, 10, 11), (1, 2, 9, 10), (2, 3, 8, 9), (3, 4, 7, 8), (4, 5, 6, 7)]
    for ks_ai in range(14):
        fr_ai = 0.08 + 0.84 * (ks_ai / 13.0)
        rz_ai = r_hub + fr_ai * (r_tip - r_hub)
        tw_ai = math.radians(6.0 + 7.0 * fr_ai)
        c_ai  = chord * (0.86 + 0.32 * fr_ai)
        s_ai  = (0.43 - 0.50) * c_ai
        x_ai  = s_ai * math.sin(tw_ai) + 0.0026 * math.cos(tw_ai)
        y_ai  = y_center + s_ai * math.cos(tw_ai) - 0.0026 * math.sin(tw_ai)
        n_ai  = Vector((math.cos(tw_ai), -math.sin(tw_ai), 0.0))
        t_ai  = Vector((math.sin(tw_ai),  math.cos(tw_ai), 0.0))
        p_ai  = Vector((x_ai, y_ai, rz_ai))
        bm_add_swept_quad_profile(
            b_one, p_ai - t_ai * 0.0024, p_ai + t_ai * 0.0024,
            Vector((0.0, 0.0, 1.0)), n_ai, prof_igv_lv, n_steps=2, cap_quad_pairs=igv_lv_caps,
        )
    for ks_te in range(16):
        fr_te = 0.07 + 0.78 * (ks_te / 15.0)
        rz_te = r_hub + fr_te * (r_tip - r_hub)
        tw_l  = math.radians(6.0 + 7.0 * fr_te)
        fl_d  = math.radians(18.0 + 11.5 * fr_te)
        c_te  = chord * (0.86 + 0.32 * fr_te)
        sh_te = -0.05 * c_te
        ds_te = 0.43 * c_te
        ang_t = tw_l + fl_d * (0.43 / 0.55) ** 0.85
        x_te  = sh_te * math.sin(tw_l) + ds_te * math.sin(ang_t)
        y_te  = y_center + sh_te * math.cos(tw_l) + ds_te * math.cos(ang_t)
        n_te  = Vector((math.cos(ang_t), -math.sin(ang_t), 0.0))
        p_te  = Vector((x_te, y_te, rz_te))

        bm_add_segmented_tube(
            b_one, p_te - n_te * 0.0018, p_te + n_te * 0.0024,
            [(0.0, 0.0016), (0.35, 0.0022), (0.86, 0.0021), (1.0, 0.0019), (0.94, 0.0011), (0.22, 0.00095)],
            segs=12, cap_ends=True,
        )
    bm_fill_unset_from(b_one, f0_igv_hw, MAT_MACHINED_ALU)

    for bi in range(n_vanes):
        bm_merge(dst_bm, b_one.copy(), Matrix.Rotation(2.0 * math.pi * bi / float(n_vanes), 4, 'Y'))
    b_one.free()


def _bm_add_annular_quad_extrusion(bm_dst: bmesh.types.BMesh, pt_fn, n_loop: int, t_rad: list, t_thick: list):

    b_sub = bmesh.new()
    nr = len(t_rad)
    nt = len(t_thick)
    grid = []
    for vt in t_thick:
        layer_r = []
        for vr in t_rad:
            ring_k = [b_sub.verts.new(pt_fn(k, vr, vt)) for k in range(n_loop)]
            layer_r.append(ring_k)
        grid.append(layer_r)

    for ir in range(nr - 1):
        for k in range(n_loop):
            kn = (k + 1) % n_loop
            b_sub.faces.new((grid[0][ir][k], grid[0][ir][kn], grid[0][ir + 1][kn], grid[0][ir + 1][k]))
            b_sub.faces.new((grid[-1][ir][k], grid[-1][ir + 1][k], grid[-1][ir + 1][kn], grid[-1][ir][kn]))

    for it in range(nt - 1):
        for k in range(n_loop):
            kn = (k + 1) % n_loop
            b_sub.faces.new((grid[it][-1][k], grid[it][-1][kn], grid[it + 1][-1][kn], grid[it + 1][-1][k]))
            b_sub.faces.new((grid[it][0][k], grid[it + 1][0][k], grid[it + 1][0][kn], grid[it][0][kn]))

    bmesh.ops.recalc_face_normals(b_sub, faces=b_sub.faces)
    bm_merge(bm_dst, b_sub)


def bm_add_chamfered_box(bm_dst: bmesh.types.BMesh, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float,
                         nx: int = 2, ny: int = 2, nz: int = 2, chamfer: float = 0.0016, k_crest: float = 0.25):

    xa, xb = min(x0, x1), max(x0, x1)
    ya, yb = min(y0, y1), max(y0, y1)
    za, zb = min(z0, z1), max(z0, z1)
    dx, dy, dz = max(xb - xa, 1e-5), max(yb - ya, 1e-5), max(zb - za, 1e-5)
    c = min(chamfer, 0.22 * dx, 0.22 * dy, 0.22 * dz)

    def _axis_coords(a: float, b: float, span: float, n_int: int) -> list:
        flat = span - 2.0 * c
        if flat <= 1e-6:
            return [a, 0.5 * (a + b), b]
        ni = max(1, int(n_int))
        if flat <= 2.4 * c or (ni == 1 and flat <= 5.0 * c):
            mids = [a + c + flat * (k / ni) for k in range(1, ni)]
            return [a, a + c] + mids + [b - c, b]
        sup = min(0.18 * flat, 2.0 * c)
        inner = flat - 2.0 * sup
        mids = [a + c + sup + inner * (k / ni) for k in range(ni + 1)]
        return [a, a + c] + mids + [b - c, b]

    xs = _axis_coords(xa, xb, dx, nx)
    ys = _axis_coords(ya, yb, dy, ny)
    zs = _axis_coords(za, zb, dz, nz)
    Ix, Iy, Iz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    k3 = min(0.46, 1.44 * k_crest)

    b_tmp = bmesh.new()
    vmap = {}

    def _get_v(ix: int, iy: int, iz: int):
        key = (ix, iy, iz)
        v = vmap.get(key)
        if v is not None:
            return v
        bx = -1 if ix == 0 else (1 if ix == Ix else 0)
        by = -1 if iy == 0 else (1 if iy == Iy else 0)
        bz = -1 if iz == 0 else (1 if iz == Iz else 0)
        nb = (bx != 0) + (by != 0) + (bz != 0)
        x, y, z = xs[ix], ys[iy], zs[iz]
        if nb == 2:
            x -= bx * k_crest * c
            y -= by * k_crest * c
            z -= bz * k_crest * c
        elif nb == 3:
            x -= bx * k3 * c
            y -= by * k3 * c
            z -= bz * k3 * c
        v = b_tmp.verts.new((x, y, z))
        vmap[key] = v
        return v

    for iy in range(Iy):
        for iz in range(Iz):
            b_tmp.faces.new((_get_v(Ix, iy, iz), _get_v(Ix, iy + 1, iz), _get_v(Ix, iy + 1, iz + 1), _get_v(Ix, iy, iz + 1)))
            b_tmp.faces.new((_get_v(0,  iy, iz), _get_v(0,  iy, iz + 1), _get_v(0,  iy + 1, iz + 1), _get_v(0,  iy + 1, iz)))
    for ix in range(Ix):
        for iz in range(Iz):
            b_tmp.faces.new((_get_v(ix, Iy, iz), _get_v(ix, Iy, iz + 1), _get_v(ix + 1, Iy, iz + 1), _get_v(ix + 1, Iy, iz)))
            b_tmp.faces.new((_get_v(ix, 0,  iz), _get_v(ix + 1, 0,  iz), _get_v(ix + 1, 0,  iz + 1), _get_v(ix, 0,  iz + 1)))
    for ix in range(Ix):
        for iy in range(Iy):
            b_tmp.faces.new((_get_v(ix, iy, Iz), _get_v(ix + 1, iy, Iz), _get_v(ix + 1, iy + 1, Iz), _get_v(ix, iy + 1, Iz)))
            b_tmp.faces.new((_get_v(ix, iy, 0),  _get_v(ix, iy + 1, 0),  _get_v(ix + 1, iy + 1, 0),  _get_v(ix + 1, iy, 0)))

    bmesh.ops.recalc_face_normals(b_tmp, faces=b_tmp.faces)
    bm_merge(bm_dst, b_tmp)


def _bm_add_box_x_yz(bm_dst: bmesh.types.BMesh, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float,
                     nx: int = 2, ny: int = 4, nz: int = 4, sup: float = 0.04, chamfer: float = 0.0014):

    bm_add_chamfered_box(bm_dst, x0, x1, y0, y1, z0, z1, nx=max(1, nx), ny=max(1, ny), nz=max(1, nz), chamfer=chamfer)


def bm_add_airfoil_stage(dst_bm: bmesh.types.BMesh, n_blades: int, y_center: float, chord: float,
                         r_hub: float, r_tip: float, twist_root_deg: float,
                         twist_tip_deg: float, camber: float, flare_tip: bool = False):

    b_one = bmesh.new()
    r_steps = 14
    c_pts = 10
    rings_b = []
    for rs in range(r_steps + 1):
        fr = rs / r_steps
        rr = r_hub + fr * (r_tip - r_hub)
        tw = math.radians(twist_root_deg + fr * (twist_tip_deg - twist_root_deg))
        ctw, stw = math.cos(tw), math.sin(tw)
        c_cur = chord * ((0.62 + 0.48 * fr) if flare_tip else (1.05 - 0.18 * fr))

        prof_pts = []
        for ip in range(2 * c_pts):
            if ip <= c_pts:
                u = ip / c_pts
                side = +1.0
            else:
                u = (2 * c_pts - ip) / c_pts
                side = -1.0
            yt = 0.13 * c_cur * (2.969 * math.sqrt(max(0.0, u)) - 1.260 * u - 3.516 * u * u + 2.843 * u ** 3 - 1.015 * u ** 4)
            yc = camber * c_cur * 4.0 * u * (1.0 - u)
            s_loc = (u - 0.5) * c_cur
            n_loc = yc + side * yt
            x_3d = s_loc * stw + n_loc * ctw
            y_3d = y_center + s_loc * ctw - n_loc * stw
            prof_pts.append(b_one.verts.new((x_3d, y_3d, rr)))
        rings_b.append(prof_pts)

    n_ring = len(rings_b[0])
    for rs in range(r_steps):
        ra, rb = rings_b[rs], rings_b[rs + 1]
        for j in range(n_ring):
            jn = (j + 1) % n_ring
            b_one.faces.new((ra[j], rb[j], rb[jn], ra[jn]))
    _bm_cap_ring_quads(b_one, rings_b[-1], reverse=False, inset_frac=0.72)
    _bm_cap_ring_quads(b_one, rings_b[0],  reverse=True,  inset_frac=0.72)
    bmesh.ops.recalc_face_normals(b_one, faces=b_one.faces)

    for bi in range(n_blades):
        bm_merge(dst_bm, b_one.copy(), Matrix.Rotation(2.0 * math.pi * bi / n_blades, 4, 'Y'))
    b_one.free()


def bm_add_cyl_plain_patch(bm: bmesh.types.BMesh, phi0: float, phi1: float, y0: float, y1: float,
                           n_phi: int, n_y: int, r_outer_fn, reverse_normals: bool = False):

    grid_v = []
    for i in range(n_phi + 1):
        ph = phi0 + (phi1 - phi0) * (i / float(n_phi))
        pr = math.radians(ph)
        sp, cp = math.sin(pr), math.cos(pr)
        col = []
        for j in range(n_y + 1):
            yv = y0 + (y1 - y0) * (j / float(n_y))
            rv = r_outer_fn(ph, yv)
            col.append(bm.verts.new((rv * sp, yv, rv * cp)))
        grid_v.append(col)
    for i in range(n_phi):
        for j in range(n_y):
            v00, v10, v11, v01 = grid_v[i][j], grid_v[i + 1][j], grid_v[i + 1][j + 1], grid_v[i][j + 1]
            if reverse_normals:
                bm.faces.new((v00, v01, v11, v10))
            else:
                bm.faces.new((v00, v10, v11, v01))
    return grid_v


def bm_add_cyl_conical_pocket_patch(bm: bmesh.types.BMesh, phi0: float, phi1: float, y0: float, y1: float,
                                    n_phi: int, n_y: int, r_outer_fn,
                                    half_arc_w: float, half_y_h: float, depth_profile: list,
                                    stadium_axis: str = 'ELLIPSE', phi_c: float = None, y_c: float = None,
                                    t_skin_rings: tuple = (0.0, 0.48, 0.80, 0.955, 1.00)):

    pc = 0.5 * (phi0 + phi1) if phi_c is None else phi_c
    yc = 0.5 * (y0 + y1) if y_c is None else y_c

    rect_uv = []
    edge_tags = []
    for i in range(n_phi):
        fu = i / float(n_phi)
        rect_uv.append((phi0 + (phi1 - phi0) * fu, y0))
        edge_tags.append((0, fu))
    for j in range(n_y):
        fv = j / float(n_y)
        rect_uv.append((phi1, y0 + (y1 - y0) * fv))
        edge_tags.append((1, fv))
    for i in range(n_phi):
        fu = i / float(n_phi)
        rect_uv.append((phi1 - (phi1 - phi0) * fu, y1))
        edge_tags.append((2, fu))
    for j in range(n_y):
        fv = j / float(n_y)
        rect_uv.append((phi0, y1 - (y1 - y0) * fv))
        edge_tags.append((3, fv))

    n_tot = len(rect_uv)

    def _eval_side_arc(s_frac: float, rc_v: float, lv_v: float):

        a_arc = 0.25 * math.pi * rc_v
        tot_l = 2.0 * a_arc + 2.0 * lv_v
        dist = s_frac * tot_l
        if dist <= a_arc:
            ang = -0.25 * math.pi + (dist / max(1e-9, a_arc)) * (0.25 * math.pi)
            return (rc_v * math.cos(ang), -lv_v + rc_v * math.sin(ang))
        elif dist <= a_arc + 2.0 * lv_v:
            return (rc_v, -lv_v + (dist - a_arc))
        else:
            ang = ((dist - a_arc - 2.0 * lv_v) / max(1e-9, a_arc)) * (0.25 * math.pi)
            return (rc_v * math.cos(ang), +lv_v + rc_v * math.sin(ang))

    def _curve_uv(tag: tuple, shrink_m: float):
        eid, fr = tag
        hw = max(0.0012, half_arc_w - shrink_m)
        hy = max(0.0012, half_y_h - shrink_m)
        if stadium_axis == 'Y':
            rc = min(hw, hy)
            lv = max(0.0, hy - rc)
            if eid == 0:
                ang = -0.75 * math.pi + 0.50 * math.pi * fr
                sx, sy = rc * math.cos(ang), -lv + rc * math.sin(ang)
            elif eid == 1:
                sx, sy = _eval_side_arc(fr, rc, lv)
            elif eid == 2:
                ang = +0.25 * math.pi + 0.50 * math.pi * fr
                sx, sy = rc * math.cos(ang), +lv + rc * math.sin(ang)
            else:
                sx_p, sy_p = _eval_side_arc(1.0 - fr, rc, lv)
                sx, sy = -sx_p, sy_p
        elif stadium_axis == 'PHI':
            rc = min(hw, hy)
            lu = max(0.0, hw - rc)
            if eid == 0:
                sy_p, sx_p = _eval_side_arc(fr, rc, lu)
                sx, sy = sx_p, -sy_p
            elif eid == 1:
                ang = -0.25 * math.pi + 0.50 * math.pi * fr
                sx, sy = +lu + rc * math.cos(ang), rc * math.sin(ang)
            elif eid == 2:
                sy_p, sx_p = _eval_side_arc(1.0 - fr, rc, lu)
                sx, sy = sx_p, +sy_p
            else:
                ang = +0.75 * math.pi + 0.50 * math.pi * fr
                sx, sy = -lu + rc * math.cos(ang), rc * math.sin(ang)
        else:
            if eid == 0:
                psi = math.atan2(-1.0, -1.0 + 2.0 * fr)
            elif eid == 1:
                psi = math.atan2(-1.0 + 2.0 * fr, +1.0)
            elif eid == 2:
                psi = math.atan2(+1.0, +1.0 - 2.0 * fr)
            else:
                psi = math.atan2(+1.0 - 2.0 * fr, -1.0)
            sx, sy = hw * math.cos(psi), hy * math.sin(psi)
        yc_pt = yc + sy
        r_loc = max(0.010, r_outer_fn(pc, yc_pt))
        ph_pt = pc + math.degrees(sx / r_loc)
        return (ph_pt, yc_pt)

    rim_uv = [_curve_uv(tg, 0.0) for tg in edge_tags]

    def _to_3d(ph_deg: float, yv: float, dep_m: float) -> Vector:
        pr = math.radians(ph_deg)
        rv = max(0.004, r_outer_fn(ph_deg, yv) - dep_m)
        return Vector((rv * math.sin(pr), yv, rv * math.cos(pr)))

    all_rings = []

    for tr in t_skin_rings:
        ring = []
        for k in range(n_tot):
            pu = rect_uv[k][0] + (rim_uv[k][0] - rect_uv[k][0]) * tr
            pv = rect_uv[k][1] + (rim_uv[k][1] - rect_uv[k][1]) * tr
            ring.append(bm.verts.new(_to_3d(pu, pv, 0.0)))
        all_rings.append(ring)


    floor_uv = rim_uv
    final_dep = 0.0
    for idx_d, (sh_m, dep_m) in enumerate(depth_profile):
        if idx_d == 0 and abs(sh_m) < 1e-7 and abs(dep_m) < 1e-7:
            continue
        cur_uv = [_curve_uv(tg, sh_m) for tg in edge_tags]
        all_rings.append([bm.verts.new(_to_3d(u_p, v_p, dep_m)) for (u_p, v_p) in cur_uv])
        floor_uv = cur_uv
        final_dep = dep_m

    for r_i in range(len(all_rings) - 1):
        ra, rb = all_rings[r_i], all_rings[r_i + 1]
        for k in range(n_tot):
            kn = (k + 1) % n_tot
            bm.faces.new((ra[k], ra[kn], rb[kn], rb[k]))


    in_ring = all_rings[-1]
    b_bot_uv   = [floor_uv[i] for i in range(n_phi + 1)]
    b_right_uv = [floor_uv[n_phi + j] for j in range(n_y + 1)]
    b_top_uv   = [floor_uv[2 * n_phi + n_y - i] for i in range(n_phi + 1)]
    b_left_uv  = [floor_uv[0]] + [floor_uv[n_tot - j] for j in range(1, n_y)] + [floor_uv[2 * n_phi + n_y]]

    c00, c10, c11, c01 = b_bot_uv[0], b_bot_uv[n_phi], b_top_uv[n_phi], b_top_uv[0]
    cap_grid = [[None] * (n_y + 1) for _ in range(n_phi + 1)]
    for i in range(n_phi + 1):
        cap_grid[i][0] = in_ring[i]
        cap_grid[i][n_y] = in_ring[2 * n_phi + n_y - i]
    for j in range(1, n_y):
        cap_grid[n_phi][j] = in_ring[n_phi + j]
        cap_grid[0][j] = in_ring[n_tot - j]

    for i in range(1, n_phi):
        xi = i / float(n_phi)
        for j in range(1, n_y):
            eta = j / float(n_y)
            u_c = ((1.0 - eta) * b_bot_uv[i][0] + eta * b_top_uv[i][0]
                   + (1.0 - xi) * b_left_uv[j][0] + xi * b_right_uv[j][0]
                   - ((1.0 - xi) * (1.0 - eta) * c00[0] + xi * (1.0 - eta) * c10[0]
                      + xi * eta * c11[0] + (1.0 - xi) * eta * c01[0]))
            v_c = ((1.0 - eta) * b_bot_uv[i][1] + eta * b_top_uv[i][1]
                   + (1.0 - xi) * b_left_uv[j][1] + xi * b_right_uv[j][1]
                   - ((1.0 - xi) * (1.0 - eta) * c00[1] + xi * (1.0 - eta) * c10[1]
                      + xi * eta * c11[1] + (1.0 - xi) * eta * c01[1]))
            cap_grid[i][j] = bm.verts.new(_to_3d(u_c, v_c, final_dep))

    for i in range(n_phi):
        for j in range(n_y):
            bm.faces.new((cap_grid[i][j], cap_grid[i + 1][j], cap_grid[i + 1][j + 1], cap_grid[i][j + 1]))

    p_ctr = _to_3d(pc, yc, final_dep)
    n_ctr = Vector((math.sin(math.radians(pc)), 0.0, math.cos(math.radians(pc))))
    return p_ctr, n_ctr


def bm_add_helical_thread_cylinder(bm: bmesh.types.BMesh, p_a: Vector, p_b: Vector,
                                   r_core: float, thread_h: float,
                                   n_turns: int = 6, segs_turn: int = 16,
                                   cap_core: bool = True):

    axis = p_b - p_a
    L = axis.length
    if L < 1e-6 or r_core < 1e-5:
        return
    t_dir = axis / L
    ref = Vector((1.0, 0.0, 0.0)) if abs(t_dir.x) < 0.85 else Vector((0.0, 0.0, 1.0))
    u_dir = t_dir.cross(ref).normalized()
    v_dir = t_dir.cross(u_dir).normalized()

    segs_even = max(14, segs_turn + (segs_turn % 2))
    if cap_core:
        f0_thc = len(bm.faces)
        bm_add_segmented_tube(
            bm, p_a, p_b,
            [(0.00, r_core * 0.84), (0.05, r_core), (0.95, r_core), (1.00, r_core * 0.76)],
            segs=segs_even, cap_ends=True,
        )
        bm_fill_unset_from(bm, f0_thc, MAT_THREAD)

    turns = max(1.5, float(n_turns))
    n_steps = max(16, int(round(turns * segs_turn)))
    L_span = 0.90 * L
    pitch = L_span / (turns + 0.65)
    hw_root = 0.42 * pitch
    hw_crest = 0.11 * pitch
    r_base = r_core * 0.91

    b_th = bmesh.new()
    rings = []
    for k in range(n_steps + 1):
        tau = k / float(n_steps)
        ang = 2.0 * math.pi * turns * tau
        rad_dir = u_dir * math.cos(ang) + v_dir * math.sin(ang)
        taper_turns = min(tau * turns, (1.0 - tau) * turns)
        env = math.sin(0.5 * math.pi * min(1.0, max(0.05, taper_turns / 0.55)))
        r_cr = r_base + (r_core * 0.09 + thread_h) * env
        s_c = 0.05 * L + 0.45 * pitch + (L_span - 0.90 * pitch) * tau
        c_pt = p_a + t_dir * s_c
        rings.append([
            b_th.verts.new(c_pt - t_dir * hw_root + rad_dir * r_base),
            b_th.verts.new(c_pt - t_dir * hw_crest + rad_dir * r_cr),
            b_th.verts.new(c_pt + t_dir * hw_crest + rad_dir * r_cr),
            b_th.verts.new(c_pt + t_dir * hw_root + rad_dir * r_base),
        ])

    for k in range(n_steps):
        ra, rb = rings[k], rings[k + 1]
        for s in range(4):
            sn = (s + 1) % 4
            b_th.faces.new((ra[s], ra[sn], rb[sn], rb[s]))
    b_th.faces.new((rings[0][3], rings[0][2], rings[0][1], rings[0][0]))
    b_th.faces.new((rings[-1][0], rings[-1][1], rings[-1][2], rings[-1][3]))
    bmesh.ops.recalc_face_normals(b_th, faces=b_th.faces)
    bm_merge(bm, b_th, default_mat_id=MAT_THREAD)


def _resample_frames_by_arclen(frames: list, target_arcs: list) -> list:

    out = []
    n_f = len(frames)
    idx = 0
    for s_val in target_arcs:
        while idx < n_f - 2 and frames[idx + 1][4] < s_val:
            idx += 1
        p0, t0, u0, _v0, a0 = frames[idx]
        p1, t1, u1, _v1, a1 = frames[idx + 1]
        ds = max(1e-8, a1 - a0)
        w = max(0.0, min(1.0, (s_val - a0) / ds))
        pos = p0.lerp(p1, w)
        t_m = t0.lerp(t1, w)
        t_dir = t_m.normalized() if t_m.length > 1e-7 else t0
        u_m = u0.lerp(u1, w)
        u_proj = u_m - t_dir * u_m.dot(t_dir)
        if u_proj.length < 1e-6:
            ref = Vector((1.0, 0.0, 0.0)) if abs(t_dir.x) < 0.85 else Vector((0.0, 0.0, 1.0))
            u_dir = t_dir.cross(ref).normalized()
        else:
            u_dir = u_proj.normalized()
        v_dir = t_dir.cross(u_dir).normalized()
        out.append((pos, t_dir, u_dir, v_dir, s_val))
    return out


def bm_add_braided_hose_from_frames(bm: bmesh.types.BMesh, frames: list, radius: float,
                                    n_half: int = 5, cap_ends: bool = True):

    if len(frames) < 2:
        return
    L_tot = frames[-1][4]
    if L_tot < 2.6 * radius:
        bm_add_segmented_tube(bm, frames[0][0], frames[-1][0], [(0.0, radius), (1.0, radius)], segs=16, cap_ends=cap_ends)
        return

    b_hose = bmesh.new()
    f_sw = min(0.085, max(0.022, (2.2 * radius) / L_tot))


    core_fracs = [
        (0.000,            radius * 1.04),
        (0.18 * f_sw,      radius * 1.14),
        (0.42 * f_sw,      radius * 1.14),
        (0.50 * f_sw,      radius * 1.05),
        (0.58 * f_sw,      radius * 1.14),
        (0.92 * f_sw,      radius * 1.14),
        (f_sw,             radius * 0.85),
    ]
    n_mid_core = max(14, min(72, int(round(L_tot / max(0.014, 3.2 * radius)))))
    for km in range(1, n_mid_core):
        core_fracs.append((f_sw + (1.0 - 2.0 * f_sw) * (km / float(n_mid_core)), radius * 0.85))
    core_fracs.extend([
        (1.0 - f_sw,        radius * 0.85),
        (1.0 - 0.92 * f_sw, radius * 1.14),
        (1.0 - 0.58 * f_sw, radius * 1.14),
        (1.0 - 0.50 * f_sw, radius * 1.05),
        (1.0 - 0.42 * f_sw, radius * 1.14),
        (1.0 - 0.18 * f_sw, radius * 1.14),
        (1.000,             radius * 1.04),
    ])
    core_frames = _resample_frames_by_arclen(frames, [cf * L_tot for cf, _ in core_fracs])
    segs_core = 16
    core_rings = []
    for (_, r_c), (pos, _t, u_d, v_d, _a) in zip(core_fracs, core_frames):
        ring = []
        for s in range(segs_core):
            ang = 2.0 * math.pi * s / segs_core
            ring.append(b_hose.verts.new(pos + u_d * (r_c * math.cos(ang)) + v_d * (r_c * math.sin(ang))))
        core_rings.append(ring)
    n_cr = len(core_rings)
    for k in range(n_cr - 1):
        r0, r1 = core_rings[k], core_rings[k + 1]
        is_ferrule = (k < 6) or (k >= n_cr - 7)
        for s in range(segs_core):
            sn = (s + 1) % segs_core
            fc = b_hose.faces.new((r0[s], r0[sn], r1[sn], r1[s]))
            fc.material_index = MAT_BRASS_FITTING if is_ferrule else MAT_BRAID_HOSE
    if cap_ends:
        f0_cap_h = len(b_hose.faces)
        _bm_cap_ring_quads(b_hose, core_rings[0],  reverse=True,  inset_frac=0.74)
        _bm_cap_ring_quads(b_hose, core_rings[-1], reverse=False, inset_frac=0.74)
        bm_fill_unset_from(b_hose, f0_cap_h, MAT_BRASS_FITTING)


    s_b0 = 0.65 * f_sw * L_tot
    s_b1 = (1.0 - 0.65 * f_sw) * L_tot
    L_braid = max(1e-5, s_b1 - s_b0)
    n_turns = max(1.5, min(16.0, L_braid / max(0.016, 6.4 * radius)))
    m_steps = max(48, min(224, int(round(28.0 * n_turns))))
    braid_arcs = [s_b0 + L_braid * (j / float(m_steps)) for j in range(m_steps + 1)]
    braid_frames = _resample_frames_by_arclen(frames, braid_arcs)

    n_carr = 4
    hd_ang = 0.66 * (math.pi / float(n_carr))
    ht_rad = radius * 0.020
    for sgn_h in (+1.0, -1.0):
        for c_idx in range(n_carr):
            rib_rings = []
            for j, (pos, _t, u_d, v_d, _a) in enumerate(braid_frames):
                u_f = j / float(m_steps)
                weave = sgn_h * math.cos(math.pi * 2.0 * n_carr * n_turns * u_f)
                r_mid = radius * (0.945 + 0.015 * weave)
                ang_c = 2.0 * math.pi * ((c_idx / float(n_carr)) + sgn_h * n_turns * u_f)
                a_lo, a_hi = ang_c - hd_ang, ang_c + hd_ang
                d_lo = u_d * math.cos(a_lo) + v_d * math.sin(a_lo)
                d_hi = u_d * math.cos(a_hi) + v_d * math.sin(a_hi)
                rib_rings.append([
                    b_hose.verts.new(pos + d_lo * (r_mid - ht_rad)),
                    b_hose.verts.new(pos + d_lo * (r_mid + ht_rad)),
                    b_hose.verts.new(pos + d_hi * (r_mid + ht_rad)),
                    b_hose.verts.new(pos + d_hi * (r_mid - ht_rad)),
                ])
            for j in range(m_steps):
                ra, rb = rib_rings[j], rib_rings[j + 1]
                for s in range(4):
                    sn = (s + 1) % 4
                    b_hose.faces.new((ra[s], ra[sn], rb[sn], rb[s]))
            b_hose.faces.new((rib_rings[0][3], rib_rings[0][2], rib_rings[0][1], rib_rings[0][0]))
            b_hose.faces.new((rib_rings[-1][0], rib_rings[-1][1], rib_rings[-1][2], rib_rings[-1][3]))

    bmesh.ops.recalc_face_normals(b_hose, faces=b_hose.faces)
    bm_merge(bm, b_hose, default_mat_id=MAT_BRAID_HOSE)


def bm_add_harness_ziptie(bm: bmesh.types.BMesh, pos: Vector, tang: Vector,
                          out_nrm: Vector, bundle_r: float, width: float = 0.0026):

    f0_zt = len(bm.faces)
    t_d = tang.normalized()
    n_proj = out_nrm - t_d * out_nrm.dot(t_d)
    n_d = n_proj.normalized() if n_proj.length > 1e-6 else (
        t_d.cross(Vector((1, 0, 0))).normalized() if abs(t_d.x) < 0.85 else t_d.cross(Vector((0, 0, 1))).normalized()
    )
    u_d = n_d.cross(t_d).normalized()

    hw = 0.5 * width
    r_in = bundle_r * 0.96
    r_out = bundle_r * 1.18
    bm_add_segmented_tube(
        bm, pos - t_d * hw, pos + t_d * hw,
        [(0.0, r_in), (0.12, r_out), (0.88, r_out), (1.0, r_in)],
        segs=18, cap_ends=True,
    )

    b_bk = bmesh.new()
    bk_w = max(0.0018, bundle_r * 0.56)
    bk_l = max(0.0020, width * 0.92)
    bk_h = max(0.0022, bundle_r * 0.68)
    _bm_add_box_x_yz(b_bk, -bk_w, +bk_w, -bk_l, +bk_l, bundle_r * 0.98, bundle_r * 1.02 + bk_h,
                     nx=1, ny=1, nz=1, chamfer=min(0.00035, bk_w * 0.22))

    _bm_add_box_x_yz(b_bk, bk_w * 0.70, bk_w * 1.95, -bk_l * 0.55, +bk_l * 0.55,
                     bundle_r * 1.02 + 0.25 * bk_h, bundle_r * 1.02 + 0.62 * bk_h,
                     nx=1, ny=1, nz=1, chamfer=0.00018)
    m_tf = Matrix.Identity(4)
    m_tf[0][0], m_tf[1][0], m_tf[2][0] = u_d.x, u_d.y, u_d.z
    m_tf[0][1], m_tf[1][1], m_tf[2][1] = t_d.x, t_d.y, t_d.z
    m_tf[0][2], m_tf[1][2], m_tf[2][2] = n_d.x, n_d.y, n_d.z
    m_tf[0][3], m_tf[1][3], m_tf[2][3] = pos.x, pos.y, pos.z
    bm_merge(bm, b_bk, m_tf, default_mat_id=MAT_LOCKWIRE)
    bm_fill_unset_from(bm, f0_zt, MAT_LOCKWIRE)


def bm_add_wire_id_sleeve(bm: bmesh.types.BMesh, pos: Vector, tang: Vector,
                          wire_r: float, length: float = 0.013):

    f0_ws = len(bm.faces)
    t_d = tang.normalized()
    hl = 0.5 * length
    bm_add_segmented_tube(
        bm, pos - t_d * hl, pos + t_d * hl,
        [
            (0.00, wire_r * 0.96), (0.08, wire_r * 1.28), (0.22, wire_r * 1.28),
            (0.26, wire_r * 1.15), (0.30, wire_r * 1.28), (0.70, wire_r * 1.28),
            (0.74, wire_r * 1.15), (0.78, wire_r * 1.28), (0.92, wire_r * 1.28),
            (1.00, wire_r * 0.96),
        ],
        segs=16, cap_ends=True,
    )
    bm_fill_unset_from(bm, f0_ws, MAT_BRASS_FITTING)


def bm_add_mil38999_connector(bm: bmesh.types.BMesh, p_base: Vector, axis_dir: Vector,
                              shell_r: float = 0.0110, with_flange: bool = True,
                              cable_pts: list = None, cable_braided: bool = False,
                              n_zipties: int = 2):

    t_ax = axis_dir.normalized()
    ref = Vector((1.0, 0.0, 0.0)) if abs(t_ax.x) < 0.85 else Vector((0.0, 0.0, 1.0))
    u_x = t_ax.cross(ref).normalized()
    v_y = t_ax.cross(u_x).normalized()

    b_cn = bmesh.new()
    sr = shell_r


    if with_flange:
        fw = sr * 1.38
        _bm_add_box_x_yz(b_cn, -fw, +fw, -fw, +fw, -0.05 * sr, 0.24 * sr,
                         nx=2, ny=2, nz=1, chamfer=sr * 0.08)
        fb = sr * 1.04
        for sx_b in (-fb, +fb):
            for sy_b in (-fb, +fb):
                bm_add_aerospace_bolt(b_cn, Vector((sx_b, sy_b, 0.24 * sr)), Vector((0, 0, 1)),
                                      head_r=sr * 0.16, height=sr * 0.15)


    bm_add_segmented_tube(
        b_cn, Vector((0, 0, -0.02 * sr)), Vector((0, 0, 2.88 * sr)),
        [
            (0.00, sr * 0.96), (0.14, sr * 0.96), (0.17, sr * 1.04), (0.28, sr * 1.04),
            (0.31, sr * 0.92), (0.64, sr * 0.92), (0.68, sr * 0.88), (0.82, sr * 0.78),
            (0.86, sr * 0.84), (0.94, sr * 0.84), (1.00, sr * 0.62),
        ],
        segs=24, cap_ends=True,
    )
    for kb_p in range(3):
        abp = 2.0 * math.pi * kb_p / 3.0 + 0.26
        n_bp = Vector((math.cos(abp), math.sin(abp), 0.0))
        p_bp = n_bp * (sr * 0.98) + Vector((0, 0, 0.66 * sr))
        bm_add_segmented_tube(
            b_cn, p_bp, p_bp + n_bp * (sr * 0.16),
            [(0.0, sr * 0.11), (0.75, sr * 0.11), (1.0, sr * 0.07)], segs=12, cap_ends=True,
        )


    b_kn = bmesh.new()
    n_c = 18
    n_ring = 2 * n_c
    m_lv = 4
    z_k0, z_k1 = 0.88 * sr, 1.84 * sr
    r_val = sr * 1.08
    r_crs = sr * 1.19
    g_v = []
    for m in range(m_lv + 1):
        zm_base = z_k0 + (z_k1 - z_k0) * (m / float(m_lv))
        row_v = []
        for idx in range(n_ring):
            ang = 2.0 * math.pi * idx / float(n_ring)
            z_off = 0.0
            if m == 0 and (idx % 2 == 1):
                z_off = 0.06 * (z_k1 - z_k0)
            elif m == m_lv and (idx % 2 == (m_lv % 2)):
                z_off = -0.06 * (z_k1 - z_k0)
            row_v.append(b_kn.verts.new((r_val * math.cos(ang), r_val * math.sin(ang), zm_base + z_off)))
        g_v.append(row_v)

    def _add_knurl_diamond(v_b, v_r, v_t, v_l):
        ctr = 0.25 * (v_b.co + v_r.co + v_t.co + v_l.co)
        ang_c = math.atan2(ctr.y, ctr.x)
        n_rad = Vector((math.cos(ang_c), math.sin(ang_c), 0.0))
        h_bump = r_crs - ctr.xy.length
        in_pts = []
        for v_out in (v_b, v_r, v_t, v_l):
            p_in = ctr.lerp(v_out.co, 0.38) + n_rad * h_bump
            in_pts.append(b_kn.verts.new(p_in))
        b_kn.faces.new((v_b, v_r, in_pts[1], in_pts[0]))
        b_kn.faces.new((v_r, v_t, in_pts[2], in_pts[1]))
        b_kn.faces.new((v_t, v_l, in_pts[3], in_pts[2]))
        b_kn.faces.new((v_l, v_b, in_pts[0], in_pts[3]))
        b_kn.faces.new((in_pts[0], in_pts[1], in_pts[2], in_pts[3]))

    for m in range(1, m_lv):
        parity = (m - 1) % 2
        for k in range(n_c):
            ic = (2 * k + parity) % n_ring
            v_b = g_v[m - 1][ic]
            v_r = g_v[m][(ic + 1) % n_ring]
            v_t = g_v[m + 1][ic]
            v_l = g_v[m][(ic - 1) % n_ring]
            _add_knurl_diamond(v_b, v_r, v_t, v_l)


    for k in range(n_c):
        i0 = (2 * k) % n_ring
        i1 = (2 * k + 1) % n_ring
        i2 = (2 * k + 2) % n_ring
        b_kn.faces.new((g_v[0][i0], g_v[0][i1], g_v[0][i2], g_v[1][i1]))
        top_par = (m_lv - 2) % 2
        j0 = (2 * k + top_par) % n_ring
        j1 = (j0 + 1) % n_ring
        j2 = (j0 + 2) % n_ring
        b_kn.faces.new((g_v[m_lv - 1][j1], g_v[m_lv][j2], g_v[m_lv][j1], g_v[m_lv][j0]))

    _bm_cap_ring_quads(b_kn, g_v[0],    reverse=True,  inset_frac=0.82)
    _bm_cap_ring_quads(b_kn, g_v[m_lv], reverse=False, inset_frac=0.82)
    bmesh.ops.recalc_face_normals(b_kn, faces=b_kn.faces)
    bm_merge(b_cn, b_kn)


    z_cl0, z_cl1 = 2.40 * sr, 2.76 * sr
    for sy_cl in (-1.0, +1.0):
        yc_a = min(sy_cl * 0.42 * sr, sy_cl * 0.88 * sr)
        yc_b = max(sy_cl * 0.42 * sr, sy_cl * 0.88 * sr)
        _bm_add_box_x_yz(b_cn, -1.24 * sr, +1.24 * sr, yc_a, yc_b, z_cl0, z_cl1,
                         nx=2, ny=1, nz=1, chamfer=sr * 0.06)
    for sx_cs in (-1.02 * sr, +1.02 * sr):
        bm_add_aerospace_bolt(
            b_cn, Vector((sx_cs, +0.88 * sr, 0.5 * (z_cl0 + z_cl1))), Vector((0, 1, 0)),
            head_r=sr * 0.16, height=sr * 0.15, with_12pt_nut_depth=sr * 1.72,
        )

    m_cn = Matrix.Identity(4)
    m_cn[0][0], m_cn[1][0], m_cn[2][0] = u_x.x, u_x.y, u_x.z
    m_cn[0][1], m_cn[1][1], m_cn[2][1] = v_y.x, v_y.y, v_y.z
    m_cn[0][2], m_cn[1][2], m_cn[2][2] = t_ax.x, t_ax.y, t_ax.z
    m_cn[0][3], m_cn[1][3], m_cn[2][3] = p_base.x, p_base.y, p_base.z
    bm_merge(bm, b_cn, m_cn, default_mat_id=MAT_CONNECTOR_38999)


    if cable_pts and len(cable_pts) >= 2:
        r_cab = sr * 0.44
        pts_full = [p_base + t_ax * (2.78 * sr)] + [Vector(p) for p in cable_pts]
        f0_cb = len(bm.faces)
        c_frames = bm_add_spline_tube(
            bm, pts_full, radius=r_cab, n_samples=max(24, len(pts_full) * 8),
            segs=14, cap_ends=True, corrugations=(20 if cable_braided else 0),
        )
        bm_fill_unset_from(bm, f0_cb, MAT_YELLOW_HARNESS)
        if len(c_frames) >= 8:
            for f_id in (0.14, 0.86):
                idx_i = min(len(c_frames) - 2, max(1, int(round(f_id * (len(c_frames) - 1)))))
                bm_add_wire_id_sleeve(bm, c_frames[idx_i][0], c_frames[idx_i][1], wire_r=r_cab, length=max(0.010, sr * 1.05))
            for kz in range(n_zipties):
                fz = (kz + 1.0) / float(n_zipties + 1)
                idx_z = min(len(c_frames) - 2, max(1, int(round(fz * (len(c_frames) - 1)))))
                bm_add_harness_ziptie(bm, c_frames[idx_z][0], c_frames[idx_z][1], c_frames[idx_z][2],
                                      bundle_r=r_cab * (1.06 if cable_braided else 1.0), width=max(0.0020, sr * 0.22))


AERO_GLYPH_STROKES = {
    'A': [(0.00, 0.22, 0.16, 1.00), (0.78, 1.00, 0.16, 1.00), (0.18, 0.82, 0.00, 0.18), (0.22, 0.78, 0.46, 0.62)],
    'B': [(0.00, 0.22, 0.00, 1.00), (0.22, 0.78, 0.00, 0.16), (0.22, 0.76, 0.42, 0.58), (0.22, 0.78, 0.84, 1.00), (0.76, 0.98, 0.12, 0.44), (0.76, 1.00, 0.56, 0.88)],
    'C': [(0.00, 0.22, 0.12, 0.88), (0.20, 0.96, 0.00, 0.18), (0.20, 0.96, 0.82, 1.00)],
    'D': [(0.00, 0.22, 0.00, 1.00), (0.22, 0.76, 0.00, 0.18), (0.22, 0.76, 0.82, 1.00), (0.76, 1.00, 0.14, 0.86)],
    'E': [(0.00, 0.22, 0.00, 1.00), (0.22, 0.98, 0.00, 0.18), (0.22, 0.82, 0.42, 0.58), (0.22, 0.98, 0.82, 1.00)],
    'F': [(0.00, 0.22, 0.00, 1.00), (0.22, 0.98, 0.00, 0.18), (0.22, 0.82, 0.42, 0.58)],
    'G': [(0.00, 0.22, 0.12, 0.88), (0.20, 0.94, 0.00, 0.18), (0.20, 0.94, 0.82, 1.00), (0.76, 1.00, 0.48, 0.86), (0.48, 1.00, 0.46, 0.62)],
    'H': [(0.00, 0.22, 0.00, 1.00), (0.78, 1.00, 0.00, 1.00), (0.22, 0.78, 0.42, 0.58)],
    'I': [(0.16, 0.84, 0.00, 0.18), (0.38, 0.62, 0.18, 0.82), (0.16, 0.84, 0.82, 1.00)],
    'J': [(0.68, 0.92, 0.00, 0.84), (0.16, 0.74, 0.82, 1.00), (0.04, 0.26, 0.62, 0.86)],
    'K': [(0.00, 0.22, 0.00, 1.00), (0.22, 0.56, 0.40, 0.60), (0.54, 0.96, 0.00, 0.42), (0.54, 0.98, 0.58, 1.00)],
    'L': [(0.00, 0.22, 0.00, 1.00), (0.22, 0.98, 0.82, 1.00)],
    'M': [(0.00, 0.20, 0.00, 1.00), (0.80, 1.00, 0.00, 1.00), (0.20, 0.40, 0.06, 0.40), (0.60, 0.80, 0.06, 0.40), (0.40, 0.60, 0.24, 0.64)],
    'N': [(0.00, 0.22, 0.00, 1.00), (0.78, 1.00, 0.00, 1.00), (0.22, 0.42, 0.16, 0.46), (0.42, 0.60, 0.36, 0.66), (0.60, 0.78, 0.54, 0.86)],
    'O': [(0.00, 0.22, 0.10, 0.90), (0.78, 1.00, 0.10, 0.90), (0.20, 0.80, 0.00, 0.18), (0.20, 0.80, 0.82, 1.00)],
    'P': [(0.00, 0.22, 0.00, 1.00), (0.22, 0.84, 0.00, 0.18), (0.22, 0.84, 0.44, 0.60), (0.76, 1.00, 0.12, 0.48)],
    'Q': [(0.00, 0.22, 0.10, 0.86), (0.78, 1.00, 0.10, 0.82), (0.20, 0.80, 0.00, 0.18), (0.20, 0.76, 0.78, 0.96), (0.58, 1.00, 0.68, 1.00)],
    'R': [(0.00, 0.22, 0.00, 1.00), (0.22, 0.84, 0.00, 0.18), (0.22, 0.82, 0.44, 0.60), (0.76, 1.00, 0.12, 0.46), (0.58, 0.98, 0.60, 1.00)],
    'S': [(0.04, 0.96, 0.00, 0.18), (0.00, 0.24, 0.16, 0.46), (0.08, 0.92, 0.42, 0.58), (0.76, 1.00, 0.54, 0.84), (0.04, 0.96, 0.82, 1.00)],
    'T': [(0.00, 1.00, 0.00, 0.18), (0.38, 0.62, 0.18, 1.00)],
    'U': [(0.00, 0.22, 0.00, 0.84), (0.78, 1.00, 0.00, 0.84), (0.18, 0.82, 0.82, 1.00)],
    'V': [(0.00, 0.22, 0.00, 0.64), (0.78, 1.00, 0.00, 0.64), (0.16, 0.42, 0.58, 0.88), (0.58, 0.84, 0.58, 0.88), (0.36, 0.64, 0.80, 1.00)],
    'W': [(0.00, 0.20, 0.00, 1.00), (0.80, 1.00, 0.00, 1.00), (0.40, 0.60, 0.36, 0.84), (0.20, 0.42, 0.78, 1.00), (0.58, 0.80, 0.78, 1.00)],
    'X': [(0.04, 0.34, 0.00, 0.38), (0.66, 0.96, 0.00, 0.38), (0.32, 0.68, 0.34, 0.66), (0.04, 0.34, 0.62, 1.00), (0.66, 0.96, 0.62, 1.00)],
    'Y': [(0.04, 0.34, 0.00, 0.44), (0.66, 0.96, 0.00, 0.44), (0.38, 0.62, 0.40, 1.00)],
    'Z': [(0.00, 1.00, 0.00, 0.18), (0.62, 0.96, 0.18, 0.44), (0.34, 0.66, 0.40, 0.62), (0.04, 0.38, 0.58, 0.82), (0.00, 1.00, 0.82, 1.00)],
    '0': [(0.00, 0.22, 0.10, 0.90), (0.78, 1.00, 0.10, 0.90), (0.20, 0.80, 0.00, 0.18), (0.20, 0.80, 0.82, 1.00)],
    '1': [(0.38, 0.62, 0.00, 1.00), (0.14, 0.38, 0.06, 0.24), (0.16, 0.84, 0.82, 1.00)],
    '2': [(0.04, 0.96, 0.00, 0.18), (0.76, 1.00, 0.18, 0.46), (0.06, 0.94, 0.42, 0.58), (0.00, 0.24, 0.56, 0.84), (0.00, 1.00, 0.82, 1.00)],
    '3': [(0.04, 0.94, 0.00, 0.18), (0.76, 1.00, 0.16, 0.84), (0.20, 0.84, 0.42, 0.58), (0.04, 0.94, 0.82, 1.00)],
    '4': [(0.00, 0.22, 0.00, 0.58), (0.00, 1.00, 0.44, 0.60), (0.68, 0.90, 0.00, 1.00)],
    '5': [(0.00, 0.96, 0.00, 0.18), (0.00, 0.22, 0.18, 0.52), (0.06, 0.92, 0.42, 0.58), (0.76, 1.00, 0.54, 0.86), (0.04, 0.92, 0.82, 1.00)],
    '6': [(0.00, 0.22, 0.12, 0.88), (0.18, 0.92, 0.00, 0.18), (0.20, 0.92, 0.42, 0.58), (0.76, 1.00, 0.54, 0.88), (0.18, 0.92, 0.82, 1.00)],
    '7': [(0.00, 1.00, 0.00, 0.18), (0.72, 0.96, 0.18, 0.50), (0.48, 0.74, 0.48, 1.00)],
    '8': [(0.00, 0.22, 0.12, 0.88), (0.78, 1.00, 0.12, 0.88), (0.18, 0.82, 0.00, 0.18), (0.18, 0.82, 0.42, 0.58), (0.18, 0.82, 0.82, 1.00)],
    '9': [(0.00, 0.22, 0.12, 0.50), (0.78, 1.00, 0.12, 0.88), (0.18, 0.82, 0.00, 0.18), (0.18, 0.82, 0.42, 0.58), (0.12, 0.82, 0.82, 1.00)],
    '-': [(0.14, 0.86, 0.42, 0.58)],
    '/': [(0.64, 0.92, 0.04, 0.36), (0.36, 0.64, 0.34, 0.66), (0.08, 0.36, 0.64, 0.96)],
    '.': [(0.34, 0.66, 0.74, 1.00)],
    ':': [(0.36, 0.64, 0.20, 0.42), (0.36, 0.64, 0.64, 0.86)],
    '°': [(0.24, 0.76, 0.00, 0.38)],
}


def _bm_emit_bevel_stroke_quad_prism(bm: bmesh.types.BMesh, pt_eval_fn,
                                     ua: float, ub: float, va: float, vb: float,
                                     z_embed: float, z_top: float):

    u0, u1 = min(ua, ub), max(ua, ub)
    v0, v1 = min(va, vb), max(va, vb)
    du = max(1e-6, u1 - u0)
    dv = max(1e-6, v1 - v0)
    ch = min(0.24 * du, 0.24 * dv, 0.36 * max(1e-5, z_top))
    zm = max(0.0, z_top - ch)

    b0 = bm.verts.new(pt_eval_fn(u0, v0, -z_embed))
    b1 = bm.verts.new(pt_eval_fn(u1, v0, -z_embed))
    b2 = bm.verts.new(pt_eval_fn(u1, v1, -z_embed))
    b3 = bm.verts.new(pt_eval_fn(u0, v1, -z_embed))

    m0 = bm.verts.new(pt_eval_fn(u0, v0, zm))
    m1 = bm.verts.new(pt_eval_fn(u1, v0, zm))
    m2 = bm.verts.new(pt_eval_fn(u1, v1, zm))
    m3 = bm.verts.new(pt_eval_fn(u0, v1, zm))

    t0 = bm.verts.new(pt_eval_fn(u0 + ch, v0 + ch, z_top))
    t1 = bm.verts.new(pt_eval_fn(u1 - ch, v0 + ch, z_top))
    t2 = bm.verts.new(pt_eval_fn(u1 - ch, v1 - ch, z_top))
    t3 = bm.verts.new(pt_eval_fn(u0 + ch, v1 - ch, z_top))

    for vs_st in (
        (b3, b2, b1, b0),
        (b0, b1, m1, m0), (b1, b2, m2, m1), (b2, b3, m3, m2), (b3, b0, m0, m3),
        (m0, m1, t1, t0), (m1, m2, t2, t1), (m2, m3, t3, t2), (m3, m0, t0, t3),
        (t0, t1, t2, t3),
    ):
        fc_st = bm.faces.new(vs_st)
        fc_st.material_index = MAT_3D_TEXT


def bm_add_3d_plate_text(bm: bmesh.types.BMesh, text_lines: list,
                         center: Vector, u_right: Vector, v_up: Vector, n_out: Vector,
                         width: float, height: float, relief: float = 0.00065,
                         with_border: bool = False, with_dividers: bool = False):

    if not text_lines:
        return
    R = u_right.normalized()
    U = v_up.normalized()
    N = n_out.normalized()
    if width >= 0.014 and height >= 0.005:
        area_lim = max(0.0008, width * height * 0.85)
        for f_bg in bm.faces:
            if f_bg.material_index == 0 and f_bg.calc_area() < area_lim:
                dv_bg = f_bg.calc_center_median() - center
                if abs(dv_bg.dot(R)) <= width * 0.72 and abs(dv_bg.dot(U)) <= height * 0.76 and -0.0075 <= dv_bg.dot(N) <= 0.0015:
                    f_bg.material_index = MAT_PLATE_BASE

    b_txt = bmesh.new()
    z_emb = relief * 0.45

    def _pt_flat(xu: float, yv: float, zn: float) -> Vector:
        return center + R * xu + U * yv + N * zn

    hw, hh = 0.5 * width, 0.5 * height
    if with_border:
        bw = max(0.00055, min(width, height) * 0.032)
        _bm_emit_bevel_stroke_quad_prism(b_txt, _pt_flat, -hw, +hw, +hh - bw, +hh, z_emb, relief * 0.85)
        _bm_emit_bevel_stroke_quad_prism(b_txt, _pt_flat, -hw, +hw, -hh, -hh + bw, z_emb, relief * 0.85)
        _bm_emit_bevel_stroke_quad_prism(b_txt, _pt_flat, -hw, -hw + bw, -hh + bw, +hh - bw, z_emb, relief * 0.85)
        _bm_emit_bevel_stroke_quad_prism(b_txt, _pt_flat, +hw - bw, +hw, -hh + bw, +hh - bw, z_emb, relief * 0.85)

    n_rows = len(text_lines)
    pad_x = width * 0.08
    pad_y = height * 0.11
    usable_w = max(0.002, width - 2.0 * pad_x)
    usable_h = max(0.002, height - 2.0 * pad_y)
    row_step = usable_h / float(n_rows)

    for r_idx, line in enumerate(text_lines):
        vy_top = +0.5 * usable_h - r_idx * row_step - row_step * 0.10
        vy_bot = +0.5 * usable_h - (r_idx + 1) * row_step + row_step * 0.12
        if with_dividers and r_idx > 0:
            y_div = +0.5 * usable_h - r_idx * row_step
            _bm_emit_bevel_stroke_quad_prism(
                b_txt, _pt_flat, -0.46 * width, +0.46 * width,
                y_div - 0.00028, y_div + 0.00028, z_emb, relief * 0.65,
            )
        n_ch = max(1, len(line))
        slot_w = usable_w / float(n_ch)
        ch_h = max(0.0008, vy_top - vy_bot)

        eff_slot = min(slot_w, ch_h * 0.82)
        tot_line_w = eff_slot * n_ch
        x_start = -0.5 * tot_line_w
        for c_idx, ch in enumerate(line.upper()):
            strokes = AERO_GLYPH_STROKES.get(ch)
            if not strokes:
                continue
            xa = x_start + c_idx * eff_slot + eff_slot * 0.11
            xb = x_start + (c_idx + 1) * eff_slot - eff_slot * 0.11
            for (su0, su1, sv0, sv1) in strokes:
                ux0 = xa + su0 * (xb - xa)
                ux1 = xa + su1 * (xb - xa)
                vy1 = vy_top - sv0 * (vy_top - vy_bot)
                vy0 = vy_top - sv1 * (vy_top - vy_bot)
                _bm_emit_bevel_stroke_quad_prism(b_txt, _pt_flat, ux0, ux1, vy0, vy1, z_emb, relief)

    bmesh.ops.recalc_face_normals(b_txt, faces=b_txt.faces)
    bm_merge(bm, b_txt)


def bm_add_conformal_cyl_text(bm: bmesh.types.BMesh, text_lines: list,
                              phi_deg_0: float, phi_deg_1: float,
                              y_0: float, y_1: float, r_surf: float,
                              relief: float = 0.00060,
                              center_offset: Vector = Vector((0.0, 0.0, 0.0)),
                              read_mode: str = 'CIRC', with_dividers: bool = False,
                              r_surf_fn=None):

    if not text_lines:
        return
    b_ct = bmesh.new()
    z_emb = relief * 0.45
    r_mid = (r_surf_fn(0.5 * (y_0 + y_1)) + r_surf) if r_surf_fn is not None else r_surf

    def _cyl_eval(ph_deg: float, yv: float, dr: float) -> Vector:
        pr = math.radians(ph_deg)
        rr = (r_surf_fn(yv) + r_surf if r_surf_fn is not None else r_surf) + dr
        return center_offset + Vector((rr * math.sin(pr), yv, rr * math.cos(pr)))

    p_lo, p_hi = min(phi_deg_0, phi_deg_1), max(phi_deg_0, phi_deg_1)
    y_lo, y_hi = min(y_0, y_1), max(y_0, y_1)
    if (p_hi - p_lo) >= 3.0 and (y_hi - y_lo) >= 0.015:
        plq_area_lim = max(0.0012, r_mid * math.radians(p_hi - p_lo) * (y_hi - y_lo) * 0.55)
        for f_cp in bm.faces:
            if f_cp.material_index == 0 and f_cp.calc_area() < plq_area_lim:
                cc = f_cp.calc_center_median() - center_offset
                rc_f = math.hypot(cc.x, cc.z)
                r_exp = (r_surf_fn(cc.y) + r_surf) if r_surf_fn is not None else r_surf
                if (y_lo - 0.010) <= cc.y <= (y_hi + 0.010) and (r_exp - 0.0055) <= rc_f <= (r_exp + 0.0025):
                    ph_f = math.degrees(math.atan2(cc.x, cc.z)) % 360.0
                    d_lo = (ph_f - (p_lo - 1.5)) % 360.0
                    if d_lo <= (p_hi - p_lo + 3.0):
                        f_cp.material_index = MAT_PLATE_BASE
    n_rows = len(text_lines)

    if read_mode == 'CIRC':
        arc_span = max(1e-4, p_hi - p_lo) * 0.86
        p_start_all = 0.5 * (p_lo + p_hi) - 0.5 * arc_span
        y_span = max(1e-4, y_hi - y_lo) * 0.80
        y_top_all = 0.5 * (y_lo + y_hi) + 0.5 * y_span
        row_h = y_span / float(n_rows)

        def _pt_circ(ph_u: float, y_v: float, dr: float) -> Vector:
            return _cyl_eval(ph_u, y_v, dr)

        for r_idx, line in enumerate(text_lines):
            yt = y_top_all - r_idx * row_h - row_h * 0.10
            yb = y_top_all - (r_idx + 1) * row_h + row_h * 0.12
            if with_dividers and r_idx > 0:
                yd = y_top_all - r_idx * row_h
                _bm_emit_bevel_stroke_quad_prism(
                    b_ct, _pt_circ, p_start_all, p_start_all + arc_span,
                    yd - 0.0003, yd + 0.0003, z_emb, relief * 0.65,
                )
            n_ch = max(1, len(line))
            slot_deg = arc_span / float(n_ch)
            max_deg = math.degrees(max(0.0008, (yt - yb) * 0.82) / max(0.01, r_mid))
            eff_deg = min(slot_deg, max_deg)
            p0_line = 0.5 * (p_lo + p_hi) - 0.5 * (eff_deg * n_ch)
            for c_idx, ch in enumerate(line.upper()):
                strokes = AERO_GLYPH_STROKES.get(ch)
                if not strokes:
                    continue
                pa = p0_line + c_idx * eff_deg + eff_deg * 0.11
                pb = p0_line + (c_idx + 1) * eff_deg - eff_deg * 0.11
                for (su0, su1, sv0, sv1) in strokes:
                    pu0 = pa + su0 * (pb - pa)
                    pu1 = pa + su1 * (pb - pa)
                    yv1 = yt - sv0 * (yt - yb)
                    yv0 = yt - sv1 * (yt - yb)
                    _bm_emit_bevel_stroke_quad_prism(b_ct, _pt_circ, pu0, pu1, yv0, yv1, z_emb, relief)
    else:
        is_stbd = (read_mode == 'AXIAL_STBD')
        y_span = max(1e-4, y_hi - y_lo) * 0.84
        p_span = max(1e-4, p_hi - p_lo) * 0.80
        row_p = p_span / float(n_rows)

        def _pt_ax(yv_u: float, ph_v: float, dr: float) -> Vector:
            return _cyl_eval(ph_v, yv_u, dr)

        for r_idx, line in enumerate(text_lines):
            if is_stbd:
                pt = (0.5 * (p_lo + p_hi) + 0.5 * p_span) - r_idx * row_p - row_p * 0.10
                pb = (0.5 * (p_lo + p_hi) + 0.5 * p_span) - (r_idx + 1) * row_p + row_p * 0.12
            else:
                pt = (0.5 * (p_lo + p_hi) - 0.5 * p_span) + r_idx * row_p + row_p * 0.10
                pb = (0.5 * (p_lo + p_hi) - 0.5 * p_span) + (r_idx + 1) * row_p - row_p * 0.12
            if with_dividers and r_idx > 0:
                pd = (0.5 * (p_lo + p_hi) + (0.5 * p_span - r_idx * row_p if is_stbd else -0.5 * p_span + r_idx * row_p))
                _bm_emit_bevel_stroke_quad_prism(
                    b_ct, _pt_ax, 0.5 * (y_lo + y_hi) - 0.5 * y_span, 0.5 * (y_lo + y_hi) + 0.5 * y_span,
                    pd - 0.12, pd + 0.12, z_emb, relief * 0.65,
                )
            n_ch = max(1, len(line))
            slot_y = y_span / float(n_ch)
            ch_arc_m = r_mid * math.radians(abs(pt - pb))
            eff_y = min(slot_y, max(0.0010, ch_arc_m * 0.85))
            tot_yw = eff_y * n_ch
            for c_idx, ch in enumerate(line.upper()):
                strokes = AERO_GLYPH_STROKES.get(ch)
                if not strokes:
                    continue
                if is_stbd:
                    y_left = 0.5 * (y_lo + y_hi) + 0.5 * tot_yw - c_idx * eff_y - eff_y * 0.11
                    y_right = 0.5 * (y_lo + y_hi) + 0.5 * tot_yw - (c_idx + 1) * eff_y + eff_y * 0.11
                else:
                    y_left = 0.5 * (y_lo + y_hi) - 0.5 * tot_yw + c_idx * eff_y + eff_y * 0.11
                    y_right = 0.5 * (y_lo + y_hi) - 0.5 * tot_yw + (c_idx + 1) * eff_y - eff_y * 0.11
                for (su0, su1, sv0, sv1) in strokes:
                    yu0 = y_left + su0 * (y_right - y_left)
                    yu1 = y_left + su1 * (y_right - y_left)
                    pv0 = pt + sv0 * (pb - pt)
                    pv1 = pt + sv1 * (pb - pt)
                    _bm_emit_bevel_stroke_quad_prism(b_ct, _pt_ax, yu0, yu1, pv0, pv1, z_emb, relief)

    bmesh.ops.recalc_face_normals(b_ct, faces=b_ct.faces)
    bm_merge(bm, b_ct)


def bm_add_recessed_micro_port(bm: bmesh.types.BMesh, pos: Vector, n_gas: Vector,
                               r_outer: float = 0.0028, depth: float = 0.0026, segs: int = 12):

    n_d = n_gas.normalized()
    p_back = pos - n_d * depth
    p_lip = pos + n_d * 0.00075
    bm_add_segmented_tube(
        bm, p_back, p_lip,
        [
            (0.00, r_outer * 0.92),
            (0.76, r_outer * 1.05),
            (1.00, r_outer * 0.88),
            (0.92, r_outer * 0.52),
            (0.14, r_outer * 0.48),
        ],
        segs=max(12, segs + (segs % 2)), cap_ends=True,
    )


def bm_add_hollow_rect_duct(
    bm: bmesh.types.BMesh,
    p_start: Vector,
    p_end: Vector,
    u_axis: Vector,
    v_axis: Vector,
    hw_out: float,
    hh_out: float,
    hw_in: float,
    hh_in: float,
    n_steps: int = 4,
):

    u_n = u_axis.normalized()
    v_n = v_axis.normalized()

    t_ax = p_end - p_start
    if u_n.cross(v_n).dot(t_ax) < 0.0:
        u_n = -u_n


    c_u = [
        -1.00, -0.86, +0.86, +1.00,
        +1.00, +1.00, +1.00, +0.86,
        -0.86, -1.00, -1.00, -1.00,
    ]
    c_v = [
        -1.00, -1.00, -1.00, -0.86,
        -0.70, +0.70, +0.86, +1.00,
        +1.00, +0.86, +0.70, -0.70,
    ]
    n_loop = len(c_u)
    t_rad = [0.00, 0.16, 0.84, 1.00]
    t_ax_vals = subdivide_1d_with_support_loops(max(2, n_steps), sup1=0.06, sup2=0.18)

    def _duct_pt(k_idx: int, vr: float, vt: float) -> Vector:
        cu = c_u[k_idx % n_loop]
        cv = c_v[k_idx % n_loop]
        hw_cur = hw_in + (hw_out - hw_in) * vr
        hh_cur = hh_in + (hh_out - hh_in) * vr
        p_c = p_start.lerp(p_end, vt)
        return p_c + u_n * (cu * hw_cur) + v_n * (cv * hh_cur)

    _bm_add_annular_quad_extrusion(bm, _duct_pt, n_loop, t_rad, t_ax_vals)


def bm_add_knurled_vernier_knob(
    bm: bmesh.types.BMesh,
    p_base: Vector,
    axis_out: Vector,
    knob_r: float = 0.0095,
    knob_h: float = 0.0135,
    n_knurls: int = 16,
    with_lockwire_lug: bool = True,
):

    ax = axis_out.normalized()
    ref = Vector((0.0, 1.0, 0.0)) if abs(ax.y) < 0.85 else Vector((1.0, 0.0, 0.0))
    u_r = ref.cross(ax).normalized()
    v_u = ax.cross(u_r).normalized()

    b_kn = bmesh.new()

    bm_add_segmented_tube(
        b_kn, p_base - ax * 0.0008, p_base + ax * knob_h,
        [
            (0.00, knob_r * 1.16),
            (0.22, knob_r * 1.16),
            (0.26, knob_r * 0.86),
            (0.34, knob_r * 0.98),
            (0.90, knob_r * 0.98),
            (0.98, knob_r * 0.84),
            (1.00, knob_r * 0.56),
            (0.88, knob_r * 0.52),
        ],
        segs=max(16, n_knurls + (n_knurls % 2)), cap_ends=True,
    )

    for kt in range(12):
        ang_t = 2.0 * math.pi * kt / 12.0
        d_rad = u_r * math.cos(ang_t) + v_u * math.sin(ang_t)
        d_tan = -u_r * math.sin(ang_t) + v_u * math.cos(ang_t)
        h_tk = knob_h * (0.20 if kt % 3 == 0 else 0.14)
        p_tk = p_base + ax * (0.0006 + 0.5 * h_tk) + d_rad * (knob_r * 1.165)
        bm_add_3d_plate_text(
            b_kn, ["I"], p_tk, d_tan, ax, d_rad,
            width=knob_r * 0.12, height=h_tk, relief=knob_r * 0.045,
        )

    z_k0, z_k1 = knob_h * 0.36, knob_h * 0.88
    dz_row = (z_k1 - z_k0) / 2.0
    d_th = 2.0 * math.pi / float(n_knurls)
    r_kb = knob_r * 0.960
    r_kp = knob_r * 1.058
    for r_idx in range(2):
        za = z_k0 + r_idx * dz_row + dz_row * 0.04
        zb = z_k0 + (r_idx + 1) * dz_row - dz_row * 0.04
        zm = 0.5 * (za + zb)
        ph_sh = 0.5 * d_th if (r_idx % 2 == 1) else 0.0
        for kc in range(n_knurls):
            tc = ph_sh + (kc + 0.5) * d_th
            ta = tc - 0.45 * d_th
            tb = tc + 0.45 * d_th
            d_c = u_r * math.cos(tc) + v_u * math.sin(tc)
            p_ctr = p_base + ax * zm + d_c * r_kb
            base_pts = [
                p_base + ax * za + d_c * r_kb,
                p_base + ax * zm + (u_r * math.cos(tb) + v_u * math.sin(tb)) * r_kb,
                p_base + ax * zb + d_c * r_kb,
                p_base + ax * zm + (u_r * math.cos(ta) + v_u * math.sin(ta)) * r_kb,
            ]
            vb = [b_kn.verts.new(pt_b) for pt_b in base_pts]
            vt = [b_kn.verts.new(p_ctr.lerp(pt_b, 0.36) + d_c * (r_kp - r_kb)) for pt_b in base_pts]
            q_faces = [
                b_kn.faces.new((vb[0], vb[3], vb[2], vb[1])),
                b_kn.faces.new((vt[0], vt[1], vt[2], vt[3])),
            ]
            for s_q in range(4):
                sn_q = (s_q + 1) % 4
                q_faces.append(b_kn.faces.new((vb[s_q], vb[sn_q], vt[sn_q], vt[s_q])))
            for fq in q_faces:
                for ed in fq.edges:
                    ed.smooth = False

    bm_add_bevel_locked_hex(
        b_kn, p_base + ax * (knob_h * 0.86), p_base + ax * (knob_h * 1.08),
        [(0.0, knob_r * 0.36), (0.25, knob_r * 0.44), (0.80, knob_r * 0.44), (1.0, knob_r * 0.34)],
        corner_bevel=0.14, cap_ends=True,
    )
    if with_lockwire_lug:
        p_lg = p_base + ax * (knob_h * 0.14) + u_r * (knob_r * 1.32)
        bm_add_aerospace_bolt(b_kn, p_lg, ax, head_r=knob_r * 0.20, height=knob_r * 0.20)
        bm_add_twisted_lockwire(
            b_kn,
            [
                p_base + ax * (knob_h * 0.98) + u_r * (knob_r * 0.38),
                p_base + ax * (knob_h * 0.92) + u_r * (knob_r * 1.08),
                p_lg + ax * (knob_r * 0.16),
            ],
            strand_r=max(0.00022, knob_r * 0.026), twists_per_m=260.0, n_samples=16, segs=10,
        )
    bmesh.ops.recalc_face_normals(b_kn, faces=b_kn.faces)
    bm_merge(bm, b_kn, default_mat_id=MAT_CONNECTOR_38999)


def bm_add_mil38999_dustcap_receptacle(
    bm: bmesh.types.BMesh,
    p_mount: Vector,
    axis_out: Vector,
    shell_r: float = 0.0086,
    lanyard_dir: Vector = None,
):

    ax = axis_out.normalized()
    ref = Vector((0.0, 1.0, 0.0)) if abs(ax.y) < 0.85 else Vector((1.0, 0.0, 0.0))
    u_r = ref.cross(ax).normalized()
    v_u = ax.cross(u_r).normalized()
    lan_d = lanyard_dir.normalized() if lanyard_dir is not None else v_u
    lan_d = (lan_d - ax * lan_d.dot(ax)).normalized()

    b_rc = bmesh.new()

    fl_hw = shell_r * 1.52
    m_fl = Matrix.Identity(4)
    m_fl[0][0], m_fl[1][0], m_fl[2][0] = u_r.x, u_r.y, u_r.z
    m_fl[0][1], m_fl[1][1], m_fl[2][1] = v_u.x, v_u.y, v_u.z
    m_fl[0][2], m_fl[1][2], m_fl[2][2] = ax.x,  ax.y,  ax.z
    m_fl[0][3], m_fl[1][3], m_fl[2][3] = p_mount.x, p_mount.y, p_mount.z
    b_fl = bmesh.new()
    _bm_add_box_x_yz(b_fl, -fl_hw, +fl_hw, -fl_hw, +fl_hw, -0.0012, +0.0024, nx=2, ny=2, nz=1, chamfer=0.0006)
    bm_merge(b_rc, b_fl, m_fl)

    fb_pts = []
    for su in (-1.0, +1.0):
        for sv in (-1.0, +1.0):
            p_fb = p_mount + u_r * (su * fl_hw * 0.74) + v_u * (sv * fl_hw * 0.74) + ax * 0.0024
            fb_pts.append(p_fb)
            bm_add_aerospace_bolt(b_rc, p_fb, ax, head_r=shell_r * 0.21, height=shell_r * 0.20)
    bm_add_bolt_pair_lockwire(b_rc, fb_pts[0], fb_pts[1], ax, head_r=shell_r * 0.21, strand_r=max(0.00022, shell_r * 0.028))
    bm_add_bolt_pair_lockwire(b_rc, fb_pts[2], fb_pts[3], ax, head_r=shell_r * 0.21, strand_r=max(0.00022, shell_r * 0.028))


    bm_add_segmented_tube(
        b_rc, p_mount + ax * 0.0018, p_mount + ax * (shell_r * 1.15),
        [(0.0, shell_r * 1.06), (0.35, shell_r * 0.96), (1.0, shell_r * 0.96)],
        segs=24, cap_ends=True,
    )
    bm_add_knurled_vernier_knob(
        b_rc, p_mount + ax * (shell_r * 0.95), ax,
        knob_r=shell_r * 1.08, knob_h=shell_r * 1.18, n_knurls=18, with_lockwire_lug=False,
    )

    p_cap_ear = p_mount + ax * (shell_r * 1.10) + lan_d * (shell_r * 1.30)
    p_case_anc = p_mount + ax * 0.0010 + lan_d * (shell_r * 2.55)
    bm_add_segmented_tube(
        b_rc, p_cap_ear - ax * 0.0012, p_cap_ear + ax * 0.0012,
        [(0.0, shell_r * 0.22), (1.0, shell_r * 0.22)], segs=12, cap_ends=True,
    )
    bm_add_dome_rivet(b_rc, p_case_anc, ax, radius=shell_r * 0.22)
    f0_lan = len(b_rc.faces)
    bm_add_spline_tube(
        b_rc,
        [
            p_cap_ear,
            0.5 * (p_cap_ear + p_case_anc) + ax * (shell_r * 0.42) + ax.cross(lan_d) * (shell_r * 0.45),
            p_case_anc + ax * 0.0012,
        ],
        radius=max(0.00045, shell_r * 0.065), n_samples=18, segs=10, cap_ends=True,
    )
    bm_fill_unset_from(b_rc, f0_lan, MAT_LOCKWIRE)
    bmesh.ops.recalc_face_normals(b_rc, faces=b_rc.faces)
    bm_merge(bm, b_rc, default_mat_id=MAT_CONNECTOR_38999)
