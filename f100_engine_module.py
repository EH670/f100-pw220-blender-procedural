import math
import bpy
import bmesh
from mathutils import Vector, Matrix, Euler

import f100_geom_primitives as _gp
from f100_geom_primitives import *
import f100_hardware_detailing as _hd_detail


_bm_cap_ring_quads = _gp._bm_cap_ring_quads
_eval_catmull_rom_frames = _gp._eval_catmull_rom_frames
_bm_add_annular_quad_extrusion = _gp._bm_add_annular_quad_extrusion
_bm_add_box_x_yz = _gp._bm_add_box_x_yz

def build_forward_turbofan_core_and_fan(spec: NozzleSpec, mats: dict, coll_fwd):

    clay = mats["clay"]

    def _cp(phi_deg: float, yv: float, rv: float) -> Vector:
        pr = math.radians(phi_deg)
        return Vector((rv * math.sin(pr), yv, rv * math.cos(pr)))

    def _cn(phi_deg: float) -> Vector:
        pr = math.radians(phi_deg)
        return Vector((math.sin(pr), 0.0, math.cos(pr)))

    def _add_fwd_pipe(bm_dst: bmesh.types.BMesh, waypoints: list, tube_r: float,
                      clamp_indices: tuple = (), union_indices: tuple = (),
                      r_case_fn=None, r_case_ref: float = 0.494, segs: int = 20,
                      corrugations: int = 0, corr_amp: float = 0.12,
                      bend_r: float = None, bnut_ends: bool = True):
        br = bend_r if bend_r is not None else max(5.6 * tube_r, 0.032)
        f0_fp = len(bm_dst.faces)
        pts_m = build_cyl_mandrel_pts(waypoints, bend_r=br, r_unwrap=0.535, arc_segs=16, max_step=0.011)
        frames = bm_sweep_centerline_tube(bm_dst, pts_m, radius=tube_r, segs=segs,
                                          cap_ends=True, corrugations=corrugations, corr_amp=corr_amp)
        bm_fill_unset_from(bm_dst, f0_fp, MAT_YELLOW_HARNESS if (0 < corrugations < 12) else MAT_RIGID_TUBE)
        if bnut_ends and len(frames) >= 6:
            bm_add_bnut_flare_fitting(bm_dst, frames[1][0], frames[1][1], tube_r=tube_r)
            bm_add_bnut_flare_fitting(bm_dst, frames[-2][0], frames[-2][1], tube_r=tube_r)
        for ui in union_indices:
            if 0 < ui < len(waypoints) - 1 and len(frames) >= 4:
                du, yu, ru = waypoints[ui]

                d_prev, y_prev, r_prev = waypoints[ui - 1]
                ptu = _cp(0.35 * d_prev + 0.65 * du, 0.35 * y_prev + 0.65 * yu, 0.35 * r_prev + 0.65 * ru)
                bf_u = min(frames, key=lambda fr: (fr[0] - ptu).length_squared)
                bm_add_bnut_flare_fitting(bm_dst, bf_u[0], bf_u[1], tube_r=tube_r)
        for ci in clamp_indices:
            if 0 <= ci < len(waypoints) and len(frames) >= 2:
                dc, yc, rc = waypoints[ci]
                if 0 < ci < len(waypoints) - 1:
                    dp, yp, rp = waypoints[ci - 1]
                    dc, yc, rc = (0.28 * dp + 0.72 * dc, 0.28 * yp + 0.72 * yc, 0.28 * rp + 0.72 * rc)
                pt = _cp(dc, yc, rc)
                best_f = min(frames, key=lambda fr: (fr[0] - pt).length_squared)
                ph_f = math.degrees(math.atan2(best_f[0].x, best_f[0].z)) % 360.0
                yf = best_f[0].y
                r_surf = r_case_fn(yf) if r_case_fn is not None else r_case_ref
                p_root = _cp(ph_f, yf, r_surf)
                nc = _cn(ph_f)
                bn = Vector((0.0, 1.0, 0.0)) if abs(best_f[1].y) < 0.72 else nc.cross(best_f[1]).normalized()
                bm_add_adel_loop_clamp(bm_dst, best_f[0], best_f[1], p_root, bn, tube_r=tube_r)
        return frames

    def _add_hex_tee_block(bm_dst: bmesh.types.BMesh, phi_deg: float, y_p: float,
                           r_surf: float, r_mid: float, branch_sgn: float = +1.0, tube_r: float = 0.0044):

        b_tee = bmesh.new()
        hw = max(0.0085, tube_r * 2.15)
        hy = max(0.0110, tube_r * 2.65)
        _bm_add_box_x_yz(b_tee, -hw, +hw, y_p - hy, y_p + hy, r_mid - hw, r_mid + hw, nx=2, ny=2, nz=2, sup=0.10)
        bm_add_segmented_tube(b_tee, Vector((0.0, y_p, r_surf - 0.002)), Vector((0.0, y_p, r_mid - hw + 0.001)),
                              [(0.0, hw * 0.88), (0.25, hw * 0.65), (1.0, hw * 0.65)], segs=16, cap_ends=True)
        for sy_t in (-1.0, +1.0):
            bm_add_bevel_locked_hex(
                b_tee, Vector((0.0, y_p + sy_t * (hy - 0.001), r_mid)),
                Vector((0.0, y_p + sy_t * (hy + 0.010), r_mid)),
                [(0.0, tube_r * 1.65), (0.25, tube_r * 1.95), (0.80, tube_r * 1.95), (1.0, tube_r * 1.45)],
                corner_bevel=0.14, cap_ends=True,
            )
        if abs(branch_sgn) > 0.1:
            sx_b = 1.0 if branch_sgn > 0 else -1.0
            bm_add_bevel_locked_hex(
                b_tee, Vector((sx_b * (hw - 0.001), y_p, r_mid)),
                Vector((sx_b * (hw + 0.010), y_p, r_mid)),
                [(0.0, tube_r * 1.65), (0.25, tube_r * 1.95), (0.80, tube_r * 1.95), (1.0, tube_r * 1.45)],
                corner_bevel=0.14, cap_ends=True,
            )
        bm_add_aerospace_bolt(b_tee, Vector((0.0, y_p, r_mid + hw)), Vector((0, 0, 1)), head_r=0.0022, height=0.0020)
        bm_merge(bm_dst, b_tee, Matrix.Rotation(math.radians(phi_deg), 4, 'Y'), default_mat_id=MAT_BRASS_FITTING)

    def _add_transducer_capsule(bm_dst: bmesh.types.BMesh, phi_deg: float, y_p: float,
                                r_surf: float, r_can: float = 0.0145, len_can: float = 0.056):

        b_tc = bmesh.new()
        r_ctr = r_surf + r_can + 0.008
        tc_bpts = []
        for sy_c in (-0.32 * len_can, +0.32 * len_can):
            bm_add_clevis_plate(b_tc, (y_p + sy_c - 0.005, r_surf - 0.001), (y_p + sy_c + 0.005, r_ctr - r_can * 0.35),
                                0.005, r_can * 0.65, -0.005, +0.005, arc_segs=11)
            for sx_tb in (-0.0082, +0.0082):
                _bm_add_box_x_yz(b_tc, sx_tb - 0.0032, sx_tb + 0.0032, y_p + sy_c - 0.005, y_p + sy_c + 0.005,
                                 r_surf - 0.001, r_surf + 0.0028, nx=1, ny=1, nz=1, chamfer=0.0004)
                p_tcb = Vector((sx_tb, y_p + sy_c, r_surf + 0.0028))
                tc_bpts.append(p_tcb)
                bm_add_aerospace_bolt(b_tc, p_tcb, Vector((0, 0, 1)), head_r=0.0017, height=0.0016)
        bm_add_bolt_pair_lockwire(b_tc, tc_bpts[0], tc_bpts[2], Vector((0, 0, 1)), head_r=0.0017, strand_r=0.00024)
        bm_add_bolt_pair_lockwire(b_tc, tc_bpts[1], tc_bpts[3], Vector((0, 0, 1)), head_r=0.0017, strand_r=0.00024)
        bm_add_segmented_tube(
            b_tc, Vector((0.0, y_p - 0.5 * len_can, r_ctr)), Vector((0.0, y_p + 0.5 * len_can, r_ctr)),
            [
                (0.000, r_can * 0.88), (0.028, r_can), (0.440, r_can),
                (0.446, r_can * 1.14), (0.554, r_can * 1.14), (0.560, r_can),
                (0.972, r_can), (1.000, r_can * 0.88),
            ],
            segs=28, cap_ends=True,
        )

        z_vb = r_ctr + r_can * 1.16
        bm_add_chamfered_cyl_x(b_tc, y_p, z_vb, -0.0085, -0.0042, radius=0.0024, segs=14)
        bm_add_chamfered_cyl_x(b_tc, y_p, z_vb, +0.0042, +0.0085, radius=0.0024, segs=14)
        bm_add_helical_thread_cylinder(
            b_tc, Vector((-0.0095, y_p, z_vb)), Vector((+0.0125, y_p, z_vb)),
            r_core=0.0012, thread_h=0.00035, n_turns=6, segs_turn=14, cap_core=True,
        )
        bm_add_bevel_locked_hex(
            b_tc, Vector((+0.0084, y_p, z_vb)), Vector((+0.0110, y_p, z_vb)),
            [(0.0, 0.0020), (0.25, 0.0024), (0.80, 0.0024), (1.0, 0.0019)],
            corner_bevel=0.14, cap_ends=True,
        )

        sgn_tc_u = +1.0 if (phi_deg < 180.0) else -1.0
        _bm_add_box_x_yz(b_tc, -0.0042, +0.0042, y_p - 0.022, y_p - 0.005, r_ctr + r_can - 0.0008, r_ctr + r_can + 0.0009,
                         nx=1, ny=2, nz=1, chamfer=0.00025)
        bm_add_3d_plate_text(
            b_tc, ["P-XDCR"], Vector((0.0, y_p - 0.0135, r_ctr + r_can + 0.0009)),
            Vector((0.0, sgn_tc_u, 0.0)), Vector((-sgn_tc_u, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)),
            width=0.0145, height=0.0055, relief=0.00038,
        )

        bm_add_bevel_locked_hex(
            b_tc, Vector((0.0, y_p - 0.5 * len_can - 0.011, r_ctr)), Vector((0.0, y_p - 0.5 * len_can + 0.001, r_ctr)),
            [(0.0, r_can * 0.48), (0.25, r_can * 0.62), (0.80, r_can * 0.62), (1.0, r_can * 0.52)],
            corner_bevel=0.14, cap_ends=True,
        )

        sx_lead = -1.0 if (phi_deg < 180.0) else +1.0
        p_lead_end = Vector((sx_lead * 0.018, y_p + 0.5 * len_can + 0.056, r_surf + 0.0085))
        bm_add_mil38999_connector(
            b_tc, Vector((0.0, y_p + 0.5 * len_can - 0.001, r_ctr)), Vector((0.0, 1.0, 0.0)),
            shell_r=max(0.0072, r_can * 0.56), with_flange=False,
            cable_pts=[
                Vector((sx_lead * 0.008, y_p + 0.5 * len_can + 0.034, r_ctr - 0.006)),
                p_lead_end,
            ],
            cable_braided=True, n_zipties=1,
        )
        bm_add_adel_loop_clamp(
            b_tc, p_lead_end, Vector((0.0, 1.0, 0.0)),
            Vector((sx_lead * 0.018, y_p + 0.5 * len_can + 0.056, r_surf)),
            Vector((1.0, 0.0, 0.0)), tube_r=max(0.0028, r_can * 0.25),
        )
        bm_merge(bm_dst, b_tc, Matrix.Rotation(math.radians(phi_deg), 4, 'Y'), default_mat_id=MAT_PUMPS_OILTANK)


    bm_inlet = bmesh.new()


    prof_pitot = [
        (-4.670, 0.0016), (-4.695, 0.0022), (-4.695, 0.0044), (-4.682, 0.0054),
        (-4.618, 0.0078), (-4.612, 0.0092), (-4.598, 0.0092), (-4.592, 0.0086),
        (-4.538, 0.0145), (-4.520, 0.0242), (-4.518, 0.0242), (-4.518, 0.0118), (-4.668, 0.0016),
    ]
    bm_add_revolved_profile(bm_inlet, prof_pitot, segs=144, close_loop=True, bevel_r=0.00030)


    b_sp_cone = bmesh.new()
    y_sp0, y_sp1 = -4.520, -4.392
    r_sp0, r_sp1 = 0.0242, 0.0916

    def _r_sp_exact(_ph_d: float, y_val: float) -> float:
        return r_sp0 + (r_sp1 - r_sp0) * ((y_val - y_sp0) / (y_sp1 - y_sp0))

    def _add_cone_collar_band(bm_c: bmesh.types.BMesh, prof_yr: list, n_seg: int = 128, phi_start_deg: float = -22.5):
        rings_c = []
        for y_c_p, r_c_p in prof_yr:
            row_c = []
            for k_c in range(n_seg):
                ar_c = math.radians(phi_start_deg + 360.0 * k_c / float(n_seg))
                row_c.append(bm_c.verts.new((r_c_p * math.sin(ar_c), y_c_p, r_c_p * math.cos(ar_c))))
            rings_c.append(row_c)
        for ir_c in range(len(rings_c) - 1):
            ra_c, rb_c = rings_c[ir_c], rings_c[ir_c + 1]
            for k_c in range(n_seg):
                kn_c = (k_c + 1) % n_seg
                bm_c.faces.new((ra_c[k_c], ra_c[kn_c], rb_c[kn_c], rb_c[k_c]))


    _add_cone_collar_band(b_sp_cone, [
        (-4.5200, 0.0137),
        (-4.5200, _r_sp_exact(0.0, -4.5200) - 0.0004),
        (-4.5194, _r_sp_exact(0.0, -4.5194)),
        (-4.5120, _r_sp_exact(0.0, -4.5120)),
    ])


    sp_band_specs = [
        (-4.5120, -4.4740,  0.0, 0.0043, 0.0078, [(0.0, 0.0), (0.00045, 0.00052), (0.00055, 0.0032), (0.00065, 0.0056), (0.00110, 0.0060)]),
        (-4.4740, -4.4350, 22.5, 0.0056, 0.0092, [(0.0, 0.0), (0.00048, 0.00055), (0.00058, 0.0035), (0.00068, 0.0060), (0.00118, 0.0064)]),
        (-4.4350, -4.3960,  0.0, 0.0068, 0.0102, [(0.0, 0.0), (0.00050, 0.00058), (0.00060, 0.0038), (0.00072, 0.0064), (0.00125, 0.0068)]),
    ]
    for yb_a, yb_b, ph_off, hw_sp, hy_sp, dprof_sp in sp_band_specs:
        for k_sec in range(8):
            ph_sec0 = ph_off - 22.5 + k_sec * 45.0
            ph_sec1 = ph_sec0 + 45.0
            bm_add_cyl_conical_pocket_patch(
                b_sp_cone, ph_sec0, ph_sec1, yb_a, yb_b,
                n_phi=16, n_y=10, r_outer_fn=_r_sp_exact,
                half_arc_w=hw_sp, half_y_h=hy_sp, depth_profile=dprof_sp,
                stadium_axis='Y', t_skin_rings=(0.0, 0.46, 0.78, 0.955, 1.00),
            )


    _add_cone_collar_band(b_sp_cone, [
        (-4.3960, _r_sp_exact(0.0, -4.3960)),
        (-4.3952, _r_sp_exact(0.0, -4.3952)),
        (-4.3949, _r_sp_exact(0.0, -4.3949) - 0.0015),
        (-4.3933, _r_sp_exact(0.0, -4.3933) - 0.0015),
        (-4.3930, _r_sp_exact(0.0, -4.3930)),
        (-4.3920, _r_sp_exact(0.0, -4.3920)),
        (-4.3920, 0.0811),
        (-4.4560, 0.0474),
        (-4.5200, 0.0137),
    ])
    bmesh.ops.remove_doubles(b_sp_cone, verts=[v for v in b_sp_cone.verts if v.is_boundary], dist=1e-4)
    bmesh.ops.recalc_face_normals(b_sp_cone, faces=b_sp_cone.faces)
    bm_merge(bm_inlet, b_sp_cone)

    prof_aft_cone = [
        (-4.392, 0.0811), (-4.392, 0.0916),
        (-4.258, 0.1592), (-4.255, 0.1575), (-4.240, 0.1680),
        (-4.240, 0.0880), (-4.325, 0.0880),
    ]
    bm_add_revolved_profile(bm_inlet, prof_aft_cone, segs=160, close_loop=True, bevel_r=0.00035)


    bm_add_bevel_locked_hex(
        bm_inlet, Vector((0, -4.534, 0)), Vector((0, -4.521, 0)),
        [(0.0, 0.0155), (0.15, 0.0192), (0.85, 0.0192), (1.0, 0.0165)], corner_bevel=0.14, cap_ends=True,
    )
    for isp in range(8):
        asp = 2.0 * math.pi * isp / 8.0
        p_st = Vector((0.0086 * math.sin(asp), -4.605, 0.0086 * math.cos(asp)))
        n_st = Vector((math.sin(asp), 0.0, math.cos(asp)))
        bm_add_recessed_micro_port(bm_inlet, p_st, n_st, r_outer=0.00125, depth=0.0018, segs=12)
    bm_add_recessed_micro_port(bm_inlet, Vector((0.0, -4.568, -0.0112)), Vector((0.0, -0.22, -0.975)).normalized(), r_outer=0.00135, depth=0.0020, segs=12)


    for im_sp in range(18):
        am = 2.0 * math.pi * im_sp / 18.0
        n_m = Vector((math.sin(am) * 0.89, -0.46, math.cos(am) * 0.89)).normalized()
        p_m = Vector((0.0942 * math.sin(am), -4.387, 0.0942 * math.cos(am)))
        bm_add_recessed_micro_port(bm_inlet, p_m, n_m, r_outer=0.0022, depth=0.0018, segs=12)
        bm_add_aerospace_bolt(bm_inlet, p_m - n_m * 0.0009, n_m, head_r=0.00135, height=0.0011)

    for idb_sp in range(12):
        adb_s = 2.0 * math.pi * idb_sp / 12.0
        n_dbs = Vector((math.sin(adb_s) * 0.89, -0.46, math.cos(adb_s) * 0.89)).normalized()
        p_dbs = Vector((0.1298 * math.sin(adb_s), -4.316, 0.1298 * math.cos(adb_s)))
        bm_add_recessed_micro_port(bm_inlet, p_dbs, n_dbs, r_outer=0.0034, depth=0.0024, segs=14)
        bm_add_bevel_locked_hex(
            bm_inlet, p_dbs - n_dbs * 0.0018, p_dbs - n_dbs * 0.0002,
            [(0.0, 0.0019), (0.30, 0.0023), (0.85, 0.0023), (1.0, 0.0018)], corner_bevel=0.14, cap_ends=True,
        )
    bm_add_3d_plate_text(
        bm_inlet, ["INDEX 0"],
        Vector((0.0, -4.286, 0.1455)),
        Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.89, 0.46)).normalized(), Vector((0.0, -0.46, 0.89)).normalized(),
        width=0.026, height=0.0065, relief=0.00045,
    )


    b_base_lug = bmesh.new()
    for sx_bl in (-0.0086, +0.0086):
        _bm_add_box_x_yz(b_base_lug, sx_bl - 0.0024, sx_bl + 0.0024, -4.257, -4.240, 0.1565, 0.1642, nx=1, ny=2, nz=1, chamfer=0.0005)
    _bm_add_box_x_yz(b_base_lug, -0.0065, +0.0065, -4.257, -4.253, 0.1565, 0.1635, nx=2, ny=1, nz=1, chamfer=0.0005)
    bm_add_aerospace_bolt(b_base_lug, Vector((0.0, -4.247, 0.1618)), Vector((0.0, -0.46, 0.89)).normalized(), head_r=0.0024, height=0.0020)
    for ib_sp in range(24):
        bm_merge(bm_inlet, b_base_lug.copy(), Matrix.Rotation(2.0 * math.pi * (ib_sp + 0.5) / 24.0, 4, 'Y'))
    b_base_lug.free()


    prof_fan_hub = [
        (-4.240, 0.088), (-4.240, 0.168), (-4.148, 0.186), (-4.144, 0.176),
        (-4.070, 0.176), (-4.066, 0.204), (-3.992, 0.210), (-3.988, 0.200),
        (-3.916, 0.200), (-3.912, 0.226), (-3.852, 0.232), (-3.848, 0.222),
        (-3.786, 0.222), (-3.773, 0.244), (-3.773, 0.088),
    ]
    bm_add_revolved_profile(bm_inlet, prof_fan_hub, segs=160, close_loop=True, bevel_r=0.00040)


    bm_add_articulated_igv_stage(bm_inlet, n_vanes=24, y_center=-4.212, chord=0.096, r_hub=0.174, r_tip=0.483)
    bm_add_airfoil_stage(bm_inlet, 52, y_center=-4.028, chord=0.042, r_hub=0.206, r_tip=0.475,
                         twist_root_deg=-18.0, twist_tip_deg=-30.0, camber=-0.07)
    bm_add_airfoil_stage(bm_inlet, 56, y_center=-3.882, chord=0.038, r_hub=0.228, r_tip=0.467,
                         twist_root_deg=-16.0, twist_tip_deg=-28.0, camber=-0.06)


    for phi_pt2, is_tt2_cage in [(142.5, False), (172.5, False), (187.5, True), (217.5, False)]:
        b_pt2 = bmesh.new()
        bm_add_clevis_plate(b_pt2, (-4.267, 0.4855), (-4.247, 0.4855), 0.0115, 0.0115, -0.0038, +0.0038, arc_segs=11)
        pt2_bpts = []
        for sx_pf, sy_pf in ((-0.0082, -4.264), (+0.0082, -4.264), (-0.0082, -4.250), (+0.0082, -4.250)):
            p_pfb = Vector((sx_pf, sy_pf, 0.4820))
            pt2_bpts.append(p_pfb)
            bm_add_aerospace_bolt(b_pt2, p_pfb, Vector((0.0, 0.0, -1.0)), head_r=0.0017, height=0.0016)
        bm_add_bolt_pair_lockwire(b_pt2, pt2_bpts[0], pt2_bpts[2], Vector((0.0, 0.0, -1.0)), head_r=0.0017, strand_r=0.00024)
        bm_add_bolt_pair_lockwire(b_pt2, pt2_bpts[1], pt2_bpts[3], Vector((0.0, 0.0, -1.0)), head_r=0.0017, strand_r=0.00024)
        if is_tt2_cage:
            bm_add_segmented_tube(
                b_pt2, Vector((0.0, -4.257, 0.486)), Vector((0.0, -4.257, 0.398)),
                [(0.0, 0.0122), (0.14, 0.0122), (0.20, 0.0094), (0.94, 0.0090), (1.0, 0.0066)], segs=24, cap_ends=True,
            )
            for kz_p in range(4):
                zp = 0.412 + kz_p * 0.014
                bm_add_chamfered_cyl_x(b_pt2, -4.257, zp, -0.0098, +0.0098, radius=0.0024, segs=12, hollow_r=0.0012)
        else:
            bm_add_segmented_tube(
                b_pt2, Vector((0.0, -4.257, 0.486)), Vector((0.0, -4.257, 0.412)),
                [(0.0, 0.0092), (0.16, 0.0092), (0.24, 0.0050), (1.0, 0.0039)], segs=18, cap_ends=True,
            )
            b_kiel = bmesh.new()
            bm_add_chamfered_cyl_x(b_kiel, 0.0, 0.0, -0.014, +0.014, radius=0.0068, segs=20, chamfer=0.0008, hollow_r=0.0040)
            bm_merge(b_pt2, b_kiel, Matrix.Translation((0.0, -4.257, 0.412)) @ Matrix.Rotation(0.5 * math.pi, 4, 'Z'))
        bm_merge(bm_inlet, b_pt2, Matrix.Rotation(math.radians(phi_pt2), 4, 'Y'))

    create_mesh_obj("Fwd_01_Inlet_Spinner_Pitot_And_Sensors", bm_inlet, coll_fwd, clay)


    bm_fcase = bmesh.new()


    prof_fan_case = [
        (-4.273, 0.486), (-4.273, 0.501), (-4.269, 0.501), (-4.269, 0.507), (-4.273, 0.507),
        (-4.273, 0.517), (-4.270, 0.517), (-4.270, 0.521), (-4.273, 0.521), (-4.273, 0.527),
        (-4.264, 0.531), (-4.256, 0.531), (-4.253, 0.523), (-4.244, 0.523), (-4.240, 0.529), (-4.235, 0.512),
        (-4.226, 0.512), (-4.220, 0.5165), (-4.174, 0.5155), (-4.168, 0.5075),
        (-4.155, 0.5065), (-4.150, 0.5185), (-4.134, 0.5185), (-4.128, 0.505),
        (-4.085, 0.503), (-4.080, 0.5005), (-4.012, 0.4995), (-4.006, 0.512), (-3.990, 0.512), (-3.984, 0.498),
        (-3.940, 0.496), (-3.935, 0.4935), (-3.876, 0.4930), (-3.870, 0.505), (-3.852, 0.505), (-3.846, 0.494),
        (-3.788, 0.494), (-3.782, 0.518), (-3.773, 0.518),
        (-3.773, 0.466), (-3.860, 0.468), (-3.990, 0.476),
        (-4.076, 0.4835), (-4.080, 0.4805), (-4.136, 0.4805), (-4.140, 0.4845),
        (-4.164, 0.4850), (-4.168, 0.4828), (-4.175, 0.4828), (-4.179, 0.4855),
        (-4.254, 0.4860),
    ]
    bm_add_revolved_profile(bm_fcase, prof_fan_case, segs=256, close_loop=True, bevel_r=0.00038)


    for k_rub in range(24):
        ph_r0 = k_rub * 15.0 + 0.45
        ph_r1 = (k_rub + 1) * 15.0 - 0.45
        bm_add_curved_cyl_panel(bm_fcase, ph_r0, ph_r1, -4.134, -4.082, 0.4795, 0.4812, n_phi=6, n_y=4)
        bm_add_curved_cyl_panel(bm_fcase, ph_r0, ph_r1, -4.264, -4.242, 0.4842, 0.4862, n_phi=6, n_y=3)


    b_rim_rib = bmesh.new()
    for sx_rr in (-0.0068, +0.0068):
        bm_add_clevis_plate(b_rim_rib, (-4.258, 0.5220), (-4.240, 0.5220), 0.0072, 0.0072, sx_rr - 0.0024, sx_rr + 0.0024, arc_segs=9)
    bm_add_aerospace_bolt(b_rim_rib, Vector((0.0, -4.249, 0.5255)), Vector((0.0, 0.0, 1.0)), head_r=0.0022, height=0.0020)
    for ir_r in range(36):
        bm_merge(bm_fcase, b_rim_rib.copy(), Matrix.Rotation(2.0 * math.pi * ir_r / 36.0, 4, 'Y'))
    b_rim_rib.free()

    inlet_bolts = []
    for if_in in range(72):
        af = 2.0 * math.pi * if_in / 72.0
        p_ib = Vector((0.520 * math.sin(af), -4.235, 0.520 * math.cos(af)))
        inlet_bolts.append(p_ib)
        bm_add_aerospace_bolt(bm_fcase, p_ib, Vector((0.0, 1.0, 0.0)), head_r=0.0030, height=0.0030)
    for if_in in range(0, 72, 2):
        bm_add_bolt_pair_lockwire(bm_fcase, inlet_bolts[if_in], inlet_bolts[if_in + 1], Vector((0.0, 1.0, 0.0)), head_r=0.0030, strand_r=0.00034)
    for ang_dw in (45.0, 135.0, 225.0, 315.0):
        adw = math.radians(ang_dw)
        bm_add_segmented_tube(
            bm_fcase, Vector((0.516 * math.sin(adw), -4.273, 0.516 * math.cos(adw))),
            Vector((0.516 * math.sin(adw), -4.286, 0.516 * math.cos(adw))),
            [(0.0, 0.0046), (0.70, 0.0046), (1.0, 0.0030)], segs=16, cap_ends=True,
        )


    for y_lring, r_lring in ((-4.212, 0.5205), (-4.172, 0.5210)):
        for d_seg_a, d_seg_b in ((6.0, 126.0), (146.0, 200.0), (286.0, 354.0)):
            pts_lr = [_cp(d_seg_a + (d_seg_b - d_seg_a) * (ks / 24.0), y_lring, r_lring) for ks in range(25)]
            fr_lr = bm_add_spline_tube(bm_fcase, pts_lr, radius=0.0038, n_samples=48, segs=14, cap_ends=True)
            bm_add_bnut_flare_fitting(bm_fcase, fr_lr[2][0], fr_lr[2][1], tube_r=0.0038)
            bm_add_bnut_flare_fitting(bm_fcase, fr_lr[-3][0], fr_lr[-3][1], tube_r=0.0038)
            for k_cl in range(1, 5):
                idx_c = int(round(k_cl * (len(fr_lr) - 1) / 5.0))
                ph_cl = math.degrees(math.atan2(fr_lr[idx_c][0].x, fr_lr[idx_c][0].z)) % 360.0
                bm_add_adel_loop_clamp(bm_fcase, fr_lr[idx_c][0], fr_lr[idx_c][1], _cp(ph_cl, y_lring, 0.515), Vector((0, 1, 0)), tube_r=0.0038)
    for deg_s_boss in (22.0, 42.0, 62.0, 82.0, 102.0, 302.0, 318.0, 334.0, 346.0):
        b_sboss = bmesh.new()
        bm_add_segmented_tube(b_sboss, Vector((0, -4.198, 0.514)), Vector((0, -4.198, 0.527)),
                              [(0.0, 0.0115), (0.55, 0.0115), (0.68, 0.0078), (1.0, 0.0072)], segs=18, cap_ends=True)
        bm_add_bevel_locked_hex(
            b_sboss, Vector((0, -4.198, 0.526)), Vector((0, -4.198, 0.5315)),
            [(0.0, 0.0046), (0.25, 0.0056), (0.80, 0.0056), (1.0, 0.0044)], corner_bevel=0.14, cap_ends=True,
        )
        sb_pts = []
        for kb_sb in range(4):
            asb = 0.25 * math.pi + 0.5 * math.pi * kb_sb
            p_sbb = Vector((0.0086 * math.cos(asb), -4.198 + 0.0086 * math.sin(asb), 0.521))
            sb_pts.append(p_sbb)
            bm_add_aerospace_bolt(b_sboss, p_sbb, Vector((0, 0, 1)), head_r=0.0016, height=0.0015)
        bm_add_bolt_pair_lockwire(b_sboss, sb_pts[0], sb_pts[1], Vector((0, 0, 1)), head_r=0.0016, strand_r=0.00022)
        bm_add_bolt_pair_lockwire(b_sboss, sb_pts[2], sb_pts[3], Vector((0, 0, 1)), head_r=0.0016, strand_r=0.00022)
        bm_add_bolt_pair_lockwire(b_sboss, Vector((0, -4.198, 0.5295)), sb_pts[0], Vector((0, 0, 1)), head_r=0.0028, strand_r=0.00024)
        bm_merge(bm_fcase, b_sboss, Matrix.Rotation(math.radians(deg_s_boss), 4, 'Y'))

    fan_aft_bolts = []
    for if_af in range(80):
        af = 2.0 * math.pi * if_af / 80.0
        pb_fa = Vector((0.507 * math.sin(af), -3.783, 0.507 * math.cos(af)))
        fan_aft_bolts.append(pb_fa)
        bm_add_aerospace_bolt(bm_fcase, pb_fa, Vector((0.0, -1.0, 0.0)), head_r=0.0032, height=0.0030, with_12pt_nut_depth=0.016)
    for if_af in range(0, 80, 2):
        bm_add_bolt_pair_lockwire(bm_fcase, fan_aft_bolts[if_af], fan_aft_bolts[if_af + 1], Vector((0.0, -1.0, 0.0)), head_r=0.0032, strand_r=0.00038)


    b_hsplit = bmesh.new()
    _bm_add_box_x_yz(b_hsplit, -0.014, +0.014, -4.236, -3.790, 0.494, 0.515, nx=2, ny=10, nz=2)
    for ks_b in range(9):
        ys_b = -4.215 + ks_b * 0.050
        bm_add_aerospace_bolt(b_hsplit, Vector((-0.015, ys_b, 0.508)), Vector((-1, 0, 0)), head_r=0.0028, height=0.0026)
    bm_merge(bm_fcase, b_hsplit, Matrix.Rotation(math.radians(90.0), 4, 'Y'))


    vane_stages = [

        (24, -4.206, -4.132, 0.515, 0.532, -0.040, 0.0078, False, False, True),
        (52, -4.028, -3.966, 0.500, 0.520, -0.044, 0.0080, True,  False, False),
        (56, -3.882, -3.802, 0.494, 0.512, -0.046, 0.0078, True,  True,  False),
    ]
    for (nv_s, y_tr, y_rg, r_cb, r_rg, dx_sw, hw_a, sk_oil, sk_pad, is_igv_h) in vane_stages:
        prof_uni = [
            (y_rg - 0.0092, r_rg - 0.0042), (y_rg + 0.0092, r_rg - 0.0042),
            (y_rg + 0.0092, r_rg + 0.0048), (y_rg + 0.0054, r_rg + 0.0048),
            (y_rg + 0.0050, r_rg + 0.0018), (y_rg - 0.0050, r_rg + 0.0018),
            (y_rg - 0.0054, r_rg + 0.0048), (y_rg - 0.0092, r_rg + 0.0048),
        ]
        bm_add_revolved_profile(bm_fcase, prof_uni, segs=224, close_loop=True, bevel_r=0.00030)

        b_rguide = bmesh.new()
        bm_add_clevis_plate(b_rguide, (y_rg - 0.015, r_cb - 0.001), (y_rg + 0.015, r_cb - 0.001), 0.005, 0.005, -0.008, +0.008, arc_segs=9)
        bm_add_chamfered_cyl_x(b_rguide, y_rg, r_rg - 0.006, -0.006, +0.006, radius=0.0055, segs=18)
        for ig_r in range(8):
            dg_r = ig_r * 45.0 + 18.0
            if sk_oil and (200.0 <= dg_r <= 286.0):
                continue
            bm_merge(bm_fcase, b_rguide.copy(), Matrix.Rotation(math.radians(dg_r), 4, 'Y'))
        b_rguide.free()

        b_varm = bmesh.new()
        r_tr_top = r_cb + (0.0175 if is_igv_h else 0.0150)
        bm_add_segmented_tube(
            b_varm, Vector((0.0, y_tr, r_cb - 0.004)), Vector((0.0, y_tr, r_tr_top)),
            [(0.0, 0.0135 if is_igv_h else 0.0094), (0.48, 0.0135 if is_igv_h else 0.0094),
             (0.58, 0.0082 if is_igv_h else 0.0066), (1.0, 0.0074 if is_igv_h else 0.0060)],
            segs=20, cap_ends=True,
        )
        bm_add_bevel_locked_hex(
            b_varm, Vector((0.0, y_tr, r_cb + 0.006)), Vector((0.0, y_tr, r_cb + 0.012)),
            [(0.0, 0.0096 if is_igv_h else 0.0076), (0.20, 0.0110 if is_igv_h else 0.0086),
             (0.80, 0.0110 if is_igv_h else 0.0086), (1.0, 0.0092 if is_igv_h else 0.0074)],
            corner_bevel=0.14, cap_ends=True,
        )
        p_la = Vector((0.0, y_tr, r_tr_top - 0.0015))
        p_lb = Vector((dx_sw, y_rg, r_rg + 0.0044))
        bm_add_flattened_aero_strut(
            b_varm, p_la, p_lb,
            half_w_mid=hw_a, half_h_mid=0.0024 if is_igv_h else 0.0020,
            end_half_w=0.0048 if is_igv_h else 0.0040, end_half_h=0.0026 if is_igv_h else 0.0022,
            eye_r=0.0068 if is_igv_h else 0.0056, eye_half_w=0.0028 if is_igv_h else 0.0024,
            taper_frac=0.20, pin_along_x=False, ribbed_pockets=1,
        )
        if is_igv_h:

            pm_tb = p_la.lerp(p_lb, 0.52)
            dir_tb = (p_lb - p_la).normalized()
            bm_add_bevel_locked_hex(
                b_varm, pm_tb - dir_tb * 0.012, pm_tb + dir_tb * 0.012,
                [(0.0, 0.0048), (0.20, 0.0062), (0.80, 0.0062), (1.0, 0.0048)], corner_bevel=0.14, cap_ends=True,
            )
            bm_add_helical_thread_cylinder(b_varm, pm_tb - dir_tb * 0.022, pm_tb - dir_tb * 0.012,
                                           r_core=0.0020, thread_h=0.00052, n_turns=5, segs_turn=16, cap_core=True)
            bm_add_helical_thread_cylinder(b_varm, pm_tb + dir_tb * 0.012, pm_tb + dir_tb * 0.022,
                                           r_core=0.0020, thread_h=0.00052, n_turns=5, segs_turn=16, cap_core=True)
        p_vb_a = p_la + Vector((0, 0, 0.0028))
        p_vb_b = p_lb + Vector((0, 0, 0.0028))
        bm_add_aerospace_bolt(b_varm, p_vb_a, Vector((0, 0, 1)), head_r=0.0028, height=0.0024)
        bm_add_aerospace_bolt(b_varm, p_vb_b, Vector((0, 0, 1)), head_r=0.0026, height=0.0024)
        bm_add_bolt_pair_lockwire(b_varm, p_vb_a, p_vb_b, Vector((0, 0, 1)), head_r=0.0026, strand_r=0.00024)

        for iv in range(nv_s):
            deg_v = 360.0 * iv / float(nv_s)
            if sk_oil and (200.0 <= deg_v <= 286.0):
                continue
            if sk_pad and ((33.0 <= deg_v <= 47.0) or (313.0 <= deg_v <= 327.0)):
                continue
            bm_merge(bm_fcase, b_varm.copy(), Matrix.Rotation(math.radians(deg_v), 4, 'Y'), default_mat_id=MAT_MACHINED_ALU)
        b_varm.free()


    for ph_igv_act in (62.0, 338.0):
        rot_iga = Matrix.Rotation(math.radians(ph_igv_act), 4, 'Y')
        b_iga = bmesh.new()
        bm_add_segmented_tube(
            b_iga, Vector((0.0, -4.025, 0.526)), Vector((0.0, -3.800, 0.520)),
            [(0.0, 0.0078), (0.18, 0.0095), (0.82, 0.0095), (1.0, 0.0078)], segs=22, cap_ends=True,
        )
        bm_add_segmented_tube(
            b_iga, Vector((-0.024, -4.010, 0.528)), Vector((-0.024, -3.855, 0.524)),
            [(0.0, 0.0145), (0.12, 0.0168), (0.68, 0.0168), (0.74, 0.0080), (1.0, 0.0080)], segs=26, cap_ends=True,
        )
        _bm_add_box_x_yz(b_iga, -0.038, -0.012, -3.965, -3.895, 0.536, 0.554, nx=2, ny=2, nz=2)
        for y_br_i, r_br_i in [(-3.966, 0.500), (-3.802, 0.494)]:
            bm_add_clevis_plate(b_iga, (y_br_i - 0.010, r_br_i), (y_br_i + 0.010, r_br_i + 0.026), 0.0070, 0.010, -0.032, +0.006, arc_segs=13)
        bm_merge(bm_fcase, b_iga, rot_iga, default_mat_id=MAT_NOZ_ACT_AB)


    p_hub_c = _cp(106.5, -3.868, 0.544)
    n_hub_c = _cn(106.5)
    bm_add_segmented_tube(bm_fcase, _cp(106.5, -3.868, 0.493), p_hub_c + n_hub_c * 0.020,
                          [(0.0, 0.028), (0.35, 0.026), (0.65, 0.030), (0.88, 0.026), (1.0, 0.018)], segs=28, cap_ends=True)
    bm_add_segmented_tube(bm_fcase, _cp(106.5, -3.902, 0.544), _cp(106.5, -3.834, 0.544),
                          [(0.0, 0.018), (0.22, 0.026), (0.50, 0.029), (0.78, 0.026), (1.0, 0.018)], segs=28, cap_ends=True)
    hub_bpts = []
    for kb_hc in range(6):
        ahc = 2.0 * math.pi * kb_hc / 6.0
        t_tan = Vector((math.cos(math.radians(106.5)), 0.0, -math.sin(math.radians(106.5))))
        p_hcb = p_hub_c + n_hub_c * 0.019 + (t_tan * math.cos(ahc) + Vector((0, 1, 0)) * math.sin(ahc)) * 0.019
        hub_bpts.append(p_hcb)
        bm_add_aerospace_bolt(bm_fcase, p_hcb, n_hub_c, head_r=0.0022, height=0.0020)
    for kb_hc in range(0, 6, 2):
        bm_add_bolt_pair_lockwire(bm_fcase, hub_bpts[kb_hc], hub_bpts[kb_hc + 1], n_hub_c, head_r=0.0022, strand_r=0.00028)


    diag_ai_fr = _add_fwd_pipe(
        bm_fcase,
        [(106.5, -3.868, 0.546), (82.0, -3.972, 0.547), (52.0, -4.094, 0.543), (10.0, -4.104, 0.541)],
        tube_r=0.0135, clamp_indices=(2, 3), r_case_ref=0.504, segs=26, bend_r=0.076, bnut_ends=False,
    )
    for f_vbf in (0.14, 0.44, 0.72, 0.90):
        idx_va = min(len(diag_ai_fr) - 2, max(1, int(round(f_vbf * (len(diag_ai_fr) - 1)))))
        pv_a, tv_a = diag_ai_fr[idx_va][0], diag_ai_fr[idx_va][1]
        bm_add_segmented_tube(bm_fcase, pv_a - tv_a * 0.009, pv_a + tv_a * 0.009,
                              [(0.0, 0.0150), (0.22, 0.0192), (0.78, 0.0192), (1.0, 0.0150)], segs=24, cap_ends=True)

    for f_ws in (0.27, 0.56):
        idx_ws = min(len(diag_ai_fr) - 2, max(1, int(round(f_ws * (len(diag_ai_fr) - 1)))))
        pw_s, tw_s = diag_ai_fr[idx_ws][0], diag_ai_fr[idx_ws][1]
        bm_add_segmented_tube(bm_fcase, pw_s - tw_s * 0.022, pw_s + tw_s * 0.022,
                              [(0.0, 0.0146), (0.08, 0.0172), (0.92, 0.0172), (1.0, 0.0146)], segs=26, cap_ends=True)
        ph_ws = math.degrees(math.atan2(pw_s.x, pw_s.z)) % 360.0
        bm_add_adel_loop_clamp(bm_fcase, pw_s, tw_s, _cp(ph_ws, pw_s.y, 0.502), Vector((0, 1, 0)), tube_r=0.0168)


    vert_ai_fr = _add_fwd_pipe(
        bm_fcase,
        [(106.5, -3.868, 0.538), (106.5, -4.038, 0.538), (22.0, -4.038, 0.538)],
        tube_r=0.0096, clamp_indices=(1, 2), r_case_ref=0.502, segs=22, bend_r=0.054, bnut_ends=False,
    )
    for f_vv in (0.18, 0.42, 0.68, 0.88):
        idx_vv = min(len(vert_ai_fr) - 2, max(1, int(round(f_vv * (len(vert_ai_fr) - 1)))))
        pv_v, tv_v = vert_ai_fr[idx_vv][0], vert_ai_fr[idx_vv][1]
        bm_add_segmented_tube(bm_fcase, pv_v - tv_v * 0.007, pv_v + tv_v * 0.007,
                              [(0.0, 0.0110), (0.22, 0.0142), (0.78, 0.0142), (1.0, 0.0110)], segs=22, cap_ends=True)


    s_arch_fr = _add_fwd_pipe(
        bm_fcase,
        [(106.5, -3.868, 0.538), (119.5, -3.992, 0.540), (102.0, -4.118, 0.5515),
         (102.0, -4.174, 0.5515), (120.5, -4.174, 0.544), (120.5, -4.234, 0.508)],
        tube_r=0.0090, clamp_indices=(1, 4), union_indices=(4,), r_case_ref=0.508, segs=22, bend_r=0.032, bnut_ends=False,
    )
    p_u_jb = _cp(120.5, -4.234, 0.507)
    bm_add_segmented_tube(bm_fcase, p_u_jb, p_u_jb + _cn(120.5) * 0.014,
                          [(0.0, 0.016), (0.65, 0.016), (1.0, 0.0115)], segs=20, cap_ends=True)


    p_tee_s = _cp(113.0, -3.930, 0.539)
    bf_ts = min(s_arch_fr, key=lambda fr: (fr[0] - p_tee_s).length_squared)
    bm_add_segmented_tube(bm_fcase, bf_ts[0] - bf_ts[1] * 0.012, bf_ts[0] + bf_ts[1] * 0.012,
                          [(0.0, 0.0102), (0.20, 0.0132), (0.80, 0.0132), (1.0, 0.0102)], segs=20, cap_ends=True)
    _add_fwd_pipe(
        bm_fcase,
        [(113.0, -3.930, 0.539), (126.5, -4.025, 0.534), (134.0, -4.065, 0.528), (134.0, -4.065, 0.504)],
        tube_r=0.0074, clamp_indices=(1,), union_indices=(2,), r_case_ref=0.502, segs=20, bend_r=0.034, bnut_ends=False,
    )


    _add_fwd_pipe(
        bm_fcase,
        [(106.5, -3.868, 0.542), (102.5, -3.795, 0.534), (100.0, -3.720, 0.526), (100.0, -3.720, 0.490)],
        tube_r=0.0155, clamp_indices=(1,), r_case_ref=0.496, segs=24, bend_r=0.052, bnut_ends=False,
    )


    for deg_j, y_src_j, r_src_j in (
        (12.0, -4.103, 0.541), (28.0, -4.100, 0.542), (44.0, -4.096, 0.543),
        (60.0, -4.062, 0.544), (76.0, -4.038, 0.538), (95.5, -4.038, 0.538),
    ):
        j_wps = [
            (deg_j,       y_src_j, r_src_j),
            (deg_j,       -4.158,  0.537),
            (deg_j - 4.5, -4.158,  0.534),
            (deg_j - 4.5, -4.234,  0.528),
            (deg_j - 4.5, -4.234,  0.508),
        ]
        _add_fwd_pipe(bm_fcase, j_wps, tube_r=0.0056, clamp_indices=(1,), union_indices=(3,),
                      r_case_ref=0.512, segs=18, bend_r=0.026)
        p_jb = _cp(deg_j - 4.5, -4.234, 0.510)
        bm_add_segmented_tube(bm_fcase, p_jb, p_jb + _cn(deg_j - 4.5) * 0.012,
                              [(0.0, 0.0125), (0.65, 0.0125), (1.0, 0.0090)], segs=18, cap_ends=True)


    b_deec = bmesh.new()
    for dy_sh in (-4.215, -4.015):
        bm_add_clevis_plate(b_deec, (dy_sh - 0.012, 0.506), (dy_sh + 0.012, 0.528), 0.008, 0.010, -0.090, +0.090, arc_segs=11)
        for sx_sh in (-0.074, +0.074):
            bm_add_clevis_plate(b_deec, (dy_sh - 0.022, 0.510), (dy_sh + 0.022, 0.558), 0.007, 0.007, sx_sh - 0.004, sx_sh + 0.004, arc_segs=9)
            bm_add_segmented_tube(
                b_deec, Vector((sx_sh, dy_sh, 0.514)), Vector((sx_sh, dy_sh, 0.540)),
                [(0.0, 0.015), (0.22, 0.017), (0.35, 0.0125), (0.65, 0.0125), (0.78, 0.017), (1.0, 0.014)], segs=20, cap_ends=True,
            )
            bm_add_aerospace_bolt(b_deec, Vector((sx_sh, dy_sh, 0.541)), Vector((0, 0, 1)), head_r=0.0032, height=0.0030)
    _bm_add_box_x_yz(b_deec, -0.086, +0.086, -4.228, -4.002, 0.535, 0.578, nx=4, ny=5, nz=3, chamfer=0.0016)
    _bm_add_box_x_yz(b_deec, -0.090, +0.090, -4.232, -3.998, 0.575, 0.583, nx=4, ny=5, nz=2, chamfer=0.0012)
    for kf_d in range(13):
        xf_d = -0.078 + kf_d * 0.013
        _bm_add_box_x_yz(b_deec, xf_d - 0.0020, xf_d + 0.0020, -4.220, -4.070, 0.582, 0.593, nx=1, ny=3, nz=1, chamfer=0.0005)
    _bm_add_box_x_yz(b_deec, +0.062, +0.074, -4.244, -4.224, 0.514, 0.542, nx=1, ny=2, nz=2, chamfer=0.0006)
    bm_add_aerospace_bolt(b_deec, Vector((+0.068, -4.238, 0.516)), Vector((0, -1, 0)), head_r=0.0024, height=0.0022)
    _bm_add_box_x_yz(b_deec, -0.040, +0.040, -4.064, -4.006, 0.582, 0.5855, nx=3, ny=3, nz=1, chamfer=0.0006)
    bm_add_3d_plate_text(
        b_deec, ["CAUTION", "DEEC-220", "ESD SENS"],
        Vector((0.0, -4.035, 0.5855)), Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)),
        width=0.068, height=0.048, relief=0.00075, with_border=True, with_dividers=True,
    )
    for sx_pl in (-0.034, +0.034):
        for sy_pl in (-4.058, -4.012):
            bm_add_dome_rivet(b_deec, Vector((sx_pl, sy_pl, 0.5865)), Vector((0, 0, 1)), radius=0.0014)
    deec_cov_bpts = []
    for sx_db in (-0.080, 0.0, +0.080):
        for sy_db in (-4.224, -4.115, -4.006):
            p_dcb = Vector((sx_db, sy_db, 0.583))
            deec_cov_bpts.append(p_dcb)
            bm_add_aerospace_bolt(b_deec, p_dcb, Vector((0, 0, 1)), head_r=0.0024, height=0.0022)
    for i_lw_d, j_lw_d in ((0, 1), (1, 2), (3, 4), (6, 7), (7, 8)):
        bm_add_bolt_pair_lockwire(b_deec, deec_cov_bpts[i_lw_d], deec_cov_bpts[j_lw_d], Vector((0, 0, 1)), head_r=0.0024, strand_r=0.00028)

    for kx_cp, xc_p in enumerate((-0.058, -0.019, +0.019, +0.058)):
        zc_p = 0.556 + (0.008 if kx_cp % 2 == 0 else -0.006)
        bm_add_mil38999_connector(
            b_deec, Vector((xc_p, -4.002, zc_p)), Vector((0.0, 1.0, 0.0)),
            shell_r=0.0112, with_flange=True,
        )
    for kz_idx, kz_sp in enumerate((0.546, 0.568)):
        y_sp_c = -4.162 + kz_idx * 0.042
        bm_add_mil38999_connector(
            b_deec, Vector((-0.086, y_sp_c, kz_sp)), Vector((-1.0, 0.0, 0.0)),
            shell_r=0.0094, with_flange=True,
            cable_pts=[
                Vector((-0.128, y_sp_c - 0.014, kz_sp - 0.010)),
                Vector((-0.146, -4.212, 0.526 - kz_idx * 0.008)),
                Vector((-0.158, -4.236, 0.514)),
            ],
            cable_braided=True, n_zipties=2,
        )
    bm_merge(bm_fcase, b_deec, Matrix.Rotation(math.radians(132.0), 4, 'Y'), default_mat_id=MAT_AVIONICS_BOX)


    for d_off_h, r_h, y_end_c in ((-5.5, 0.0064, -3.620), (-1.8, 0.0070, -3.580), (+2.0, 0.0062, -3.600), (+5.8, 0.0058, -3.640)):
        dh = 132.0 + d_off_h
        h_wps = [
            (dh, -3.968, 0.554), (dh + 2.5, -3.922, 0.532),
            (dh + 5.0, -3.845, 0.524), (dh + 6.5, y_end_c, 0.518),
        ]
        h_fr = _add_fwd_pipe(bm_fcase, h_wps, tube_r=r_h, clamp_indices=(1, 2), r_case_ref=0.495, segs=16, corrugations=28, corr_amp=0.10, bnut_ends=False)
        if len(h_fr) >= 10:
            bm_add_wire_id_sleeve(bm_fcase, h_fr[2][0], h_fr[2][1], wire_r=r_h * 1.08, length=0.015)
            for f_zt in (0.28, 0.68):
                iz = min(len(h_fr) - 2, max(1, int(round(f_zt * (len(h_fr) - 1)))))
                bm_add_harness_ziptie(bm_fcase, h_fr[iz][0], h_fr[iz][1], _cn(dh + 3.0), bundle_r=r_h * 1.08)


    b_fvblk = bmesh.new()
    _bm_add_box_x_yz(b_fvblk, -0.056, +0.056, -4.002, -3.816, 0.494, 0.532, nx=3, ny=4, nz=2)
    _bm_add_box_x_yz(b_fvblk, -0.042, +0.042, -3.982, -3.836, 0.530, 0.556, nx=3, ny=4, nz=2)
    for y_rib_fv in (-3.968, -3.909, -3.850):
        _bm_add_box_x_yz(b_fvblk, -0.052, +0.052, y_rib_fv - 0.003, y_rib_fv + 0.003, 0.512, 0.554, nx=2, ny=1, nz=1, chamfer=0.0006)

    for x_fvc, y_fvc, r_fvc in ((-0.018, -3.956, 0.018), (+0.020, -3.908, 0.020), (-0.016, -3.860, 0.017)):
        bm_add_segmented_tube(b_fvblk, Vector((x_fvc, y_fvc, 0.552)), Vector((x_fvc, y_fvc, 0.574)),
                              [(0.0, r_fvc), (0.30, r_fvc * 1.08), (0.82, r_fvc * 0.96), (1.0, r_fvc * 0.68)], segs=22, cap_ends=True)
        fv_cbolts = []
        for kb_fv in range(6):
            afv = 2.0 * math.pi * kb_fv / 6.0
            pfvb = Vector((x_fvc + r_fvc * 0.82 * math.cos(afv), y_fvc + r_fvc * 0.82 * math.sin(afv), 0.556))
            fv_cbolts.append(pfvb)
            bm_add_aerospace_bolt(b_fvblk, pfvb, Vector((0, 0, 1)), head_r=0.0018, height=0.0017)
        for kb_fv in range(0, 6, 2):
            bm_add_bolt_pair_lockwire(b_fvblk, fv_cbolts[kb_fv], fv_cbolts[kb_fv + 1], Vector((0, 0, 1)), head_r=0.0018, strand_r=0.00025)
    bm_add_segmented_tube(b_fvblk, Vector((+0.028, -3.968, 0.542)), Vector((+0.056, -3.992, 0.562)),
                          [(0.0, 0.015), (0.75, 0.016), (1.0, 0.011)], segs=20, cap_ends=True)
    bm_add_mil38999_connector(
        b_fvblk, Vector((+0.052, -3.989, 0.559)), Vector((+0.58, -0.50, +0.42)),
        shell_r=0.0078, with_flange=False,
        cable_pts=[Vector((+0.072, -3.998, 0.562)), Vector((+0.082, -3.965, 0.544))],
        cable_braided=True, n_zipties=1,
    )
    fvp_l_pts, fvp_r_pts = [], []
    for y_fvp in (-3.972, -3.930, -3.888, -3.846):
        bm_add_bevel_locked_hex(b_fvblk, Vector((-0.068, y_fvp, 0.518)), Vector((-0.052, y_fvp, 0.518)),
                                [(0.0, 0.0072), (0.30, 0.0090), (0.80, 0.0090), (1.0, 0.0070)], corner_bevel=0.14, cap_ends=True)
        bm_add_bevel_locked_hex(b_fvblk, Vector((+0.052, y_fvp, 0.518)), Vector((+0.066, y_fvp, 0.518)),
                                [(0.0, 0.0068), (0.30, 0.0084), (0.80, 0.0084), (1.0, 0.0066)], corner_bevel=0.14, cap_ends=True)
        fvp_l_pts.append(Vector((-0.066, y_fvp, 0.518)))
        fvp_r_pts.append(Vector((+0.064, y_fvp, 0.518)))
    for kp_lw in (0, 2):
        bm_add_bolt_pair_lockwire(b_fvblk, fvp_l_pts[kp_lw], fvp_l_pts[kp_lw + 1], Vector((-1, 0, 0)), head_r=0.0075, strand_r=0.00030)
        bm_add_bolt_pair_lockwire(b_fvblk, fvp_r_pts[kp_lw], fvp_r_pts[kp_lw + 1], Vector((+1, 0, 0)), head_r=0.0070, strand_r=0.00030)
    bm_merge(bm_fcase, b_fvblk, Matrix.Rotation(math.radians(123.5), 4, 'Y'))


    b_arch_port = bmesh.new()
    bm_add_segmented_tube(b_arch_port, Vector((0, -3.835, 0.492)), Vector((0, -3.835, 0.516)),
                          [(0.0, 0.024), (0.60, 0.024), (0.72, 0.016), (1.0, 0.015)], segs=24, cap_ends=True)
    ap_bpts = []
    for kb_ap in range(6):
        aap = 2.0 * math.pi * kb_ap / 6.0
        p_apb = Vector((0.019 * math.cos(aap), -3.835 + 0.019 * math.sin(aap), 0.508))
        ap_bpts.append(p_apb)
        bm_add_aerospace_bolt(b_arch_port, p_apb, Vector((0, 0, 1)), head_r=0.0020, height=0.0018)
    for kb_ap in range(0, 6, 2):
        bm_add_bolt_pair_lockwire(b_arch_port, ap_bpts[kb_ap], ap_bpts[kb_ap + 1], Vector((0, 0, 1)), head_r=0.0020, strand_r=0.00026)
    bm_merge(bm_fcase, b_arch_port, Matrix.Rotation(math.radians(90.5), 4, 'Y'))


    for idx_fr3, y_fr3 in enumerate((-3.944, -3.922, -3.900)):
        tr_fr3 = 0.0046 if idx_fr3 == 1 else 0.0040
        _add_fwd_pipe(bm_fcase, [(14.0, -3.790, 0.510), (14.0, y_fr3, 0.514), (32.5, y_fr3, 0.514), (32.5, -3.952, 0.506)],
                      tube_r=tr_fr3, clamp_indices=(1, 2), r_case_ref=0.496, segs=16, bend_r=0.022)
        _add_fwd_pipe(bm_fcase, [(69.0, -3.748, 0.509), (69.0, y_fr3, 0.514), (116.5, y_fr3, 0.516)],
                      tube_r=tr_fr3, clamp_indices=(1, 2), union_indices=(1,), r_case_ref=0.496, segs=16, bend_r=0.026)


    for deg_lip_h, y_turn_h, d_end_h, y_end_h in [
        (108.5, -4.196, 121.0, -3.995),
        (114.5, -4.210, 124.5, -3.995),
        (127.5, -4.236, 126.5, -3.998),
        (143.5, -4.205, 139.5, -4.005),
        (151.5, -4.192, 160.0, -4.025),
    ]:
        p_lh = _cp(deg_lip_h, -4.242, 0.510)
        bm_add_bevel_locked_hex(bm_fcase, p_lh, p_lh + _cn(deg_lip_h) * 0.015,
                                [(0.0, 0.0095), (0.30, 0.0112), (0.80, 0.0112), (1.0, 0.0080)], corner_bevel=0.14, cap_ends=True)
        _add_fwd_pipe(
            bm_fcase,
            [(deg_lip_h, -4.242, 0.520), (deg_lip_h, y_turn_h, 0.524), (d_end_h, y_turn_h, 0.524), (d_end_h, y_end_h, 0.522)],
            tube_r=0.0045, clamp_indices=(2,), r_case_ref=0.508, segs=16, corrugations=22, corr_amp=0.09, bend_r=0.024,
        )


    arch_bridge_tubes = [
        ([(18.0, -4.085, 0.514), (18.0, -3.856, 0.512), (18.0, -3.742, 0.510)], 0.0054, (1,)),
        ([(27.5, -4.065, 0.512), (27.5, -3.856, 0.509), (27.5, -3.748, 0.506)], 0.0040, (1,)),
        ([(54.5, -4.075, 0.514), (54.5, -3.856, 0.511), (54.5, -3.752, 0.508)], 0.0046, (1,)),
        ([(64.0, -4.035, 0.514), (64.0, -3.856, 0.511), (64.0, -3.752, 0.509)], 0.0042, (1,)),
        ([(72.0, -4.035, 0.515), (72.0, -3.856, 0.513), (72.0, -3.752, 0.511)], 0.0043, (1,)),
        ([(79.0, -4.035, 0.514), (79.0, -3.856, 0.511), (79.0, -3.752, 0.508)], 0.0046, (1,)),
        ([(90.5, -3.835, 0.514), (90.5, -3.775, 0.515), (75.5, -3.752, 0.513)], 0.0038, (1,)),
        ([(116.0, -3.965, 0.516), (116.0, -3.835, 0.512), (116.0, -3.752, 0.508)], 0.0044, (1,)),
    ]
    for wps_ab, tr_ab, cl_ab in arch_bridge_tubes:
        _add_fwd_pipe(bm_fcase, wps_ab, tube_r=tr_ab, clamp_indices=cl_ab, r_case_ref=0.495, segs=16, bend_r=0.032)


    for k_tst, d_lat_f in enumerate((137.5, 143.0)):
        b_dl = bmesh.new()
        _bm_add_box_x_yz(b_dl, -0.008, +0.008, -4.228, -4.186, 0.510, 0.532, nx=2, ny=2, nz=2, sup=0.10)
        bm_add_chamfered_cyl_x(b_dl, -4.218, 0.534, -0.011, +0.011, radius=0.0046, segs=16)
        bm_add_segmented_tube(b_dl, Vector((0.0, -4.236, 0.542)), Vector((0.0, -4.218, 0.534)),
                              [(0.0, 0.0028), (0.50, 0.0034), (1.0, 0.0028)], segs=14, cap_ends=True)
        bm_add_mil38999_dustcap_receptacle(
            b_dl, Vector((0.0, -4.152, 0.5175)), Vector((0.0, 0.0, 1.0)),
            shell_r=0.0078, lanyard_dir=Vector((0.0, 1.0, 0.0)),
        )
        _bm_add_box_x_yz(b_dl, -0.0048, +0.0048, -4.124, -4.102, 0.503, 0.5055, nx=1, ny=2, nz=1, chamfer=0.0003)
        bm_add_3d_plate_text(
            b_dl, [f"TEST J{k_tst + 1}"], Vector((0.0, -4.113, 0.5055)),
            Vector((0.0, 1.0, 0.0)), Vector((-1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)),
            width=0.016, height=0.0052, relief=0.00038,
        )
        bm_merge(bm_fcase, b_dl, Matrix.Rotation(math.radians(d_lat_f), 4, 'Y'), default_mat_id=MAT_MACHINED_ALU)

    _hd_detail.add_fwd02_inlet_and_deec_detailing(bm_fcase, _cp, _cn)
    create_mesh_obj("Fwd_02_Fan_Case_VaneLinkage_And_DEEC", bm_fcase, coll_fwd, clay)


    bm_ciso = bmesh.new()
    build_isogrid_cased_cylinder(
        bm_ciso,
        y0=-3.773, y1=-3.051,
        r_inner=0.466, r_outer=0.494,
        n_circ=24, n_rows=5,
        pocket_depth=0.0052,
        r_inner_end=0.466, r_outer_end=0.494,
        axial_ribs=True,
    )
    prof_hpc_inner = [
        (-3.773, 0.244), (-3.773, 0.376), (-3.730, 0.382), (-3.051, 0.370),
        (-3.051, 0.256), (-3.410, 0.262), (-3.730, 0.244),
    ]
    bm_add_revolved_profile(bm_ciso, prof_hpc_inner, segs=128, close_loop=True, bevel_r=0.00045)
    bm_add_airfoil_stage(bm_ciso, 36, y_center=-3.728, chord=0.044, r_hub=0.244, r_tip=0.378,
                         twist_root_deg=12.0, twist_tip_deg=24.0, camber=0.05)

    for y_cf, r_cf in [(-3.765, 0.518), (-3.059, 0.518)]:
        prof_cf = [
            (y_cf - 0.008, 0.490), (y_cf - 0.008, r_cf), (y_cf + 0.008, r_cf), (y_cf + 0.008, 0.490),
        ]
        bm_add_revolved_profile(bm_ciso, prof_cf, segs=240, close_loop=True, bevel_r=0.00040)

    c_fwd_bolts = []
    for icf in range(72):
        acf = 2.0 * math.pi * icf / 72.0
        pcf = Vector((0.506 * math.sin(acf), -3.757, 0.506 * math.cos(acf)))
        c_fwd_bolts.append(pcf)
        bm_add_aerospace_bolt(bm_ciso, pcf, Vector((0.0, 1.0, 0.0)), head_r=0.0031, height=0.0029)
    for icf in range(0, 72, 2):
        bm_add_bolt_pair_lockwire(bm_ciso, c_fwd_bolts[icf], c_fwd_bolts[icf + 1], Vector((0.0, 1.0, 0.0)), head_r=0.0031, strand_r=0.00036)

    c_aft_bolts = []
    for icb in range(80):
        acb = 2.0 * math.pi * icb / 80.0
        pcb = Vector((0.506 * math.sin(acb), -3.050, 0.506 * math.cos(acb)))
        c_aft_bolts.append(pcb)
        bm_add_aerospace_bolt(bm_ciso, pcb, Vector((0.0, 1.0, 0.0)), head_r=0.0032, height=0.0030, with_12pt_nut_depth=0.016)
    for icb in range(0, 80, 2):
        bm_add_bolt_pair_lockwire(bm_ciso, c_aft_bolts[icb], c_aft_bolts[icb + 1], Vector((0.0, 1.0, 0.0)), head_r=0.0032, strand_r=0.00038)

    for k_bs, (deg_bs, y_bs) in enumerate([
        (28.0, -3.48), (56.0, -3.28), (78.0, -3.58), (104.0, -3.21),
        (254.0, -3.26), (256.0, -3.56), (288.0, -3.22), (322.0, -3.62),
    ]):
        b_cbs = bmesh.new()
        bm_add_segmented_tube(
            b_cbs, Vector((0.0, y_bs, 0.490)), Vector((0.0, y_bs, 0.505)),
            [(0.0, 0.022), (0.55, 0.022), (0.68, 0.015), (1.0, 0.014)], segs=24, cap_ends=True,
        )
        p_plg = Vector((0.0, y_bs, 0.505))
        p_anc = Vector((0.014, y_bs + 0.010, 0.498))
        bm_add_aerospace_bolt(b_cbs, p_plg, Vector((0, 0, 1)), head_r=0.0062, height=0.0055)
        bm_add_aerospace_bolt(b_cbs, p_anc, Vector((0, 0, 1)), head_r=0.0025, height=0.0024)
        bm_add_bolt_pair_lockwire(b_cbs, p_plg, p_anc, Vector((0, 0, 1)), head_r=0.0040, strand_r=0.00038)
        sgn_bs = +1.0 if deg_bs < 180.0 else -1.0
        _bm_add_box_x_yz(b_cbs, -0.0055, +0.0055, y_bs - 0.043, y_bs - 0.017, 0.4935, 0.4958, nx=1, ny=2, nz=1, chamfer=0.0003)
        bm_add_3d_plate_text(
            b_cbs, [f"HPC BS-{k_bs + 1}"], Vector((0.0, y_bs - 0.030, 0.4958)),
            Vector((0.0, sgn_bs, 0.0)),
            Vector((-sgn_bs, 0.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
            width=0.022, height=0.0068, relief=0.00042,
        )
        bm_merge(bm_ciso, b_cbs, Matrix.Rotation(math.radians(deg_bs), 4, 'Y'), default_mat_id=MAT_MACHINED_ALU)

    create_mesh_obj("Fwd_03_Core_Isogrid_Duct_And_Splitters", bm_ciso, coll_fwd, clay)


    bm_cleft = bmesh.new()

    def _add_port_flange(bm_t: bmesh.types.BMesh, phi_deg: float, y_p: float, r_surf: float, r_fl: float = 0.032, n_bolts: int = 8):
        b_pf = bmesh.new()
        bm_add_segmented_tube(
            b_pf, Vector((0.0, y_p, r_surf - 0.008)), Vector((0.0, y_p, r_surf + 0.014)),
            [(0.0, r_fl), (0.62, r_fl), (0.72, r_fl * 0.78), (1.0, r_fl * 0.78)], segs=28, cap_ends=True,
        )
        pts_fb = []
        for kb in range(n_bolts):
            ab = 2.0 * math.pi * kb / float(n_bolts)
            pb = Vector((r_fl * 0.87 * math.cos(ab), y_p + r_fl * 0.87 * math.sin(ab), r_surf + 0.0085))
            pts_fb.append(pb)
            bm_add_aerospace_bolt(b_pf, pb, Vector((0, 0, 1)), head_r=0.0025, height=0.0024)
        for kb in range(0, n_bolts - 1, 2):
            bm_add_bolt_pair_lockwire(b_pf, pts_fb[kb], pts_fb[kb + 1], Vector((0, 0, 1)), head_r=0.0025, strand_r=0.00032)
        bm_merge(bm_t, b_pf, Matrix.Rotation(math.radians(phi_deg), 4, 'Y'))

    def _add_gang_clamp_bar(bm_t: bmesh.types.BMesh, phi_a: float, phi_b: float, y_cl: float, r_surf: float, r_bar: float):
        b_gc = bmesh.new()
        arc_len = r_bar * math.radians(abs(phi_b - phi_a))
        _bm_add_box_x_yz(b_gc, -0.5 * arc_len - 0.008, +0.5 * arc_len + 0.008, y_cl - 0.0045, y_cl + 0.0045, r_bar - 0.006, r_bar + 0.005, nx=4, ny=1, nz=1, sup=0.08)
        for sx_g in (-0.5 * arc_len, 0.0, +0.5 * arc_len):
            bm_add_segmented_tube(b_gc, Vector((sx_g, y_cl, r_surf - 0.002)), Vector((sx_g, y_cl, r_bar + 0.006)),
                                  [(0.0, 0.0042), (0.85, 0.0032), (1.0, 0.0032)], segs=12, cap_ends=True)
            bm_add_aerospace_bolt(b_gc, Vector((sx_g, y_cl, r_bar + 0.006)), Vector((0, 0, 1)), head_r=0.0024, height=0.0022)
        bm_merge(bm_t, b_gc, Matrix.Rotation(math.radians(0.5 * (phi_a + phi_b)), 4, 'Y'))


    s_bleed_wps = [
        (113.0, -3.298, 0.534),
        (113.0, -3.255, 0.538),
        (80.5,  -3.344, 0.556),
        (80.5,  -3.136, 0.556),
        (115.5, -3.148, 0.528),
        (122.0, -3.168, 0.486),
    ]
    sb_pts_3d = build_cyl_mandrel_pts(
        s_bleed_wps, bend_r=0.060, r_unwrap=0.545, arc_segs=22, max_step=0.010,
        corner_radii={1: 0.034, 2: 0.064, 3: 0.064, 4: 0.044},
    )
    sb_fr = bm_sweep_centerline_tube(bm_cleft, sb_pts_3d, radius=0.0235, segs=28, cap_ends=True)
    _add_port_flange(bm_cleft, 122.0, -3.168, 0.494, r_fl=0.037, n_bolts=8)


    idx_wye = max(0, len(sb_fr) - 10)
    p_wye, t_wye = sb_fr[idx_wye][0], sb_fr[idx_wye][1]
    bm_add_segmented_tube(
        bm_cleft, p_wye - t_wye * 0.024, p_wye + t_wye * 0.024,
        [(0.0, 0.0252), (0.15, 0.0300), (0.85, 0.0300), (1.0, 0.0252)], segs=28, cap_ends=True,
    )
    _add_fwd_pipe(bm_cleft, [(115.0, -3.142, 0.530), (118.0, -3.082, 0.520), (120.0, -3.072, 0.488)],
                  tube_r=0.0175, r_case_ref=0.494, segs=24, bend_r=0.045, bnut_ends=False)
    _add_port_flange(bm_cleft, 120.0, -3.072, 0.494, r_fl=0.029, n_bolts=6)


    for f_frac in (0.21, 0.74):
        idx_vf = min(len(sb_fr) - 2, max(1, int(round(f_frac * (len(sb_fr) - 1)))))
        pf_v, tf_v, uf_v, vf_v, _ = sb_fr[idx_vf]
        bm_add_segmented_tube(
            bm_cleft, pf_v - tf_v * 0.011, pf_v + tf_v * 0.011,
            [(0.0, 0.0250), (0.18, 0.0350), (0.45, 0.0350), (0.50, 0.0300), (0.55, 0.0350), (0.82, 0.0350), (1.0, 0.0250)],
            segs=28, cap_ends=True,
        )
        sb_fl_pts = []
        for kb_s in range(8):
            ang_bs = 2.0 * math.pi * kb_s / 8.0
            off_bs = (uf_v * math.cos(ang_bs) + vf_v * math.sin(ang_bs)) * 0.0305
            p_sfb = pf_v + off_bs + tf_v * 0.010
            sb_fl_pts.append(p_sfb)
            bm_add_aerospace_bolt(bm_cleft, p_sfb, tf_v, head_r=0.0025, height=0.0024, with_12pt_nut_depth=0.020)
        for kb_s in range(0, 8, 2):
            bm_add_bolt_pair_lockwire(bm_cleft, sb_fl_pts[kb_s], sb_fl_pts[kb_s + 1], tf_v, head_r=0.0025, strand_r=0.00030)


    b_bval = bmesh.new()
    bm_add_segmented_tube(
        b_bval, Vector((0.0, -3.456, 0.534)), Vector((0.0, -3.292, 0.534)),
        [
            (0.000, 0.036), (0.015, 0.040), (0.080, 0.040), (0.088, 0.048),
            (0.455, 0.048), (0.462, 0.054), (0.538, 0.054), (0.545, 0.048),
            (0.912, 0.048), (0.920, 0.040), (0.985, 0.040), (1.000, 0.036),
        ],
        segs=32, cap_ends=True,
    )
    bm_add_segmented_tube(
        b_bval, Vector((+0.024, -3.374, 0.526)), Vector((+0.052, -3.374, 0.538)),
        [(0.0, 0.022), (0.65, 0.024), (1.0, 0.016)], segs=22, cap_ends=True,
    )
    bv_bpts = []
    for kb_v in range(10):
        av = 2.0 * math.pi * kb_v / 10.0
        p_bvb = Vector((0.049 * math.cos(av), -3.374, 0.534 + 0.049 * math.sin(av)))
        bv_bpts.append(p_bvb)
        bm_add_aerospace_bolt(b_bval, p_bvb, Vector((0, 1, 0)), head_r=0.0025, height=0.0023)
    for kb_v in range(0, 10, 2):
        bm_add_bolt_pair_lockwire(b_bval, bv_bpts[kb_v], bv_bpts[kb_v + 1], Vector((0, 1, 0)), head_r=0.0025, strand_r=0.00028)
    bm_add_clevis_plate(b_bval, (-3.430, 0.492), (-3.318, 0.534), 0.015, 0.017, -0.024, +0.024, arc_segs=13)
    bm_merge(bm_cleft, b_bval, Matrix.Rotation(math.radians(113.0), 4, 'Y'))

    _add_fwd_pipe(
        bm_cleft,
        [(113.0, -3.452, 0.534), (113.0, -3.555, 0.532), (101.0, -3.665, 0.524), (100.0, -3.720, 0.522), (100.0, -3.720, 0.488)],
        tube_r=0.0172, r_case_ref=0.494, segs=24, bend_r=0.072, bnut_ends=False,
    )
    _add_port_flange(bm_cleft, 100.0, -3.720, 0.494, r_fl=0.030, n_bolts=6)


    b_servo = bmesh.new()
    _bm_add_box_x_yz(b_servo, -0.050, +0.050, -3.558, -3.372, 0.493, 0.514, nx=3, ny=4, nz=2)
    _bm_add_box_x_yz(b_servo, -0.038, +0.038, -3.542, -3.382, 0.512, 0.542, nx=3, ny=4, nz=2)
    for y_wrib in (-3.518, -3.462, -3.406):
        _bm_add_box_x_yz(b_servo, -0.046, +0.046, y_wrib - 0.0035, y_wrib + 0.0035, 0.510, 0.548, nx=2, ny=1, nz=1, chamfer=0.0006)

    bm_add_segmented_tube(
        b_servo, Vector((-0.008, -3.548, 0.540)), Vector((-0.008, -3.376, 0.538)),
        [
            (0.000, 0.0285), (0.014, 0.0315), (0.415, 0.0315),
            (0.425, 0.0235), (0.870, 0.0235), (0.878, 0.0260),
            (0.986, 0.0260), (1.000, 0.0225),
        ],
        segs=28, cap_ends=True,
    )

    _bm_add_box_x_yz(b_servo, -0.058, -0.014, -3.462, -3.386, 0.520, 0.556, nx=2, ny=3, nz=2, chamfer=0.0010)
    _bm_add_box_x_yz(b_servo, -0.070, -0.056, -3.452, -3.396, 0.526, 0.550, nx=1, ny=2, nz=2, chamfer=0.0008)
    for sy_pt, sz_pt in ((-3.448, 0.530), (-3.448, 0.546), (-3.400, 0.530), (-3.400, 0.546)):
        bm_add_aerospace_bolt(b_servo, Vector((-0.070, sy_pt, sz_pt)), Vector((-1, 0, 0)), head_r=0.0020, height=0.0019)
    bm_add_mil38999_connector(
        b_servo, Vector((-0.068, -3.422, 0.538)), Vector((-0.020, +0.016, +0.002)),
        shell_r=0.0086, with_flange=False,
    )

    _bm_add_box_x_yz(b_servo, -0.022, +0.012, -3.485, -3.408, 0.552, 0.562, nx=2, ny=2, nz=1, chamfer=0.0007)
    bm_add_3d_plate_text(
        b_servo, ["SERVO REG", "P/N 7642"],
        Vector((-0.005, -3.4465, 0.562)), Vector((0.0, 1.0, 0.0)), Vector((-1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)),
        width=0.064, height=0.026, relief=0.00060, with_dividers=True,
    )
    for sx_np, sy_np in ((-0.018, -3.480), (+0.008, -3.480), (-0.018, -3.413), (+0.008, -3.413)):
        bm_add_dome_rivet(b_servo, Vector((sx_np, sy_np, 0.562)), Vector((0, 0, 1)), radius=0.0014)
    sv_f_pts, sv_a_pts = [], []
    for k_ec in range(6):
        aec = 2.0 * math.pi * k_ec / 6.0
        pf_sv = Vector((-0.008 + 0.024 * math.cos(aec), -3.549, 0.540 + 0.024 * math.sin(aec)))
        pa_sv = Vector((-0.008 + 0.018 * math.cos(aec), -3.375, 0.538 + 0.018 * math.sin(aec)))
        sv_f_pts.append(pf_sv)
        sv_a_pts.append(pa_sv)
        bm_add_aerospace_bolt(b_servo, pf_sv, Vector((0, -1, 0)), head_r=0.0022, height=0.0020)
        bm_add_aerospace_bolt(b_servo, pa_sv, Vector((0, +1, 0)), head_r=0.0020, height=0.0020)
    for k_ec in range(0, 6, 2):
        bm_add_bolt_pair_lockwire(b_servo, sv_f_pts[k_ec], sv_f_pts[k_ec + 1], Vector((0, -1, 0)), head_r=0.0022, strand_r=0.00026)
        bm_add_bolt_pair_lockwire(b_servo, sv_a_pts[k_ec], sv_a_pts[k_ec + 1], Vector((0, +1, 0)), head_r=0.0020, strand_r=0.00026)

    p_ang0 = Vector((+0.018, -3.492, 0.534))
    p_ang1 = Vector((+0.058, -3.516, 0.544))
    bm_add_segmented_tube(b_servo, p_ang0, p_ang1,
                          [(0.0, 0.020), (0.20, 0.022), (0.75, 0.021), (0.85, 0.014), (1.0, 0.012)], segs=24, cap_ends=True)
    bm_add_bevel_locked_hex(b_servo, p_ang1, p_ang1 + Vector((+0.016, -0.008, +0.003)),
                            [(0.0, 0.0090), (0.30, 0.0112), (0.80, 0.0112), (1.0, 0.0080)], corner_bevel=0.14, cap_ends=True)
    for sx_sv in (-0.042, +0.042):
        p0_sv = Vector((sx_sv, -3.546, 0.514))
        p1_sv = Vector((sx_sv, -3.382, 0.514))
        bm_add_aerospace_bolt(b_servo, p0_sv, Vector((0, 0, 1)), head_r=0.0032, height=0.0030)
        bm_add_aerospace_bolt(b_servo, p1_sv, Vector((0, 0, 1)), head_r=0.0032, height=0.0030)
        bm_add_bolt_pair_lockwire(b_servo, p0_sv, p1_sv, Vector((0, 0, 1)), head_r=0.0032, strand_r=0.00036)

    bm_add_segmented_tube(b_servo, Vector((-0.006, -3.655, 0.526)), Vector((-0.006, -3.546, 0.532)),
                          [(0.0, 0.0065), (0.22, 0.0090), (0.35, 0.0062), (0.82, 0.0062), (1.0, 0.0095)], segs=18, cap_ends=True)
    bm_add_segmented_tube(b_servo, Vector((0.0, -3.655, 0.492)), Vector((0.0, -3.655, 0.538)),
                          [(0.0, 0.026), (0.42, 0.026), (0.52, 0.016), (1.0, 0.014)], segs=24, cap_ends=True)
    for ang_sp4 in (25.0, 115.0, 205.0, 295.0):
        ar4 = math.radians(ang_sp4)
        p_arm4 = Vector((0.044 * math.cos(ar4), -3.655 + 0.044 * math.sin(ar4), 0.524))
        bm_add_flattened_aero_strut(
            b_servo, Vector((0.0, -3.655, 0.524)), p_arm4,
            half_w_mid=0.0065, half_h_mid=0.0030, end_half_w=0.0048, end_half_h=0.0030,
            eye_r=0.0072, eye_half_w=0.0032, taper_frac=0.22, pin_along_x=False, ribbed_pockets=1,
        )
        bm_add_aerospace_bolt(b_servo, p_arm4 + Vector((0, 0, 0.0032)), Vector((0, 0, 1)), head_r=0.0028, height=0.0024)
    bm_merge(bm_cleft, b_servo, Matrix.Rotation(math.radians(91.0), 4, 'Y'), default_mat_id=MAT_PUMPS_OILTANK)


    b_lhood = bmesh.new()
    _bm_add_box_x_yz(b_lhood, -0.052, +0.052, -3.495, -3.275, 0.492, 0.514, nx=3, ny=4, nz=2)
    _bm_add_box_x_yz(b_lhood, -0.036, +0.036, -3.465, -3.305, 0.512, 0.534, nx=2, ny=3, nz=2)
    bm_add_segmented_tube(
        b_lhood, Vector((-0.006, -3.476, 0.530)), Vector((+0.004, -3.296, 0.542)),
        [
            (0.000, 0.0220), (0.015, 0.0255), (0.145, 0.0255),
            (0.155, 0.0305), (0.245, 0.0305), (0.255, 0.0268),
            (0.495, 0.0268), (0.505, 0.0235), (0.555, 0.0235), (0.565, 0.0268),
            (0.755, 0.0268), (0.765, 0.0300), (0.845, 0.0300), (0.855, 0.0240),
            (0.985, 0.0240), (1.000, 0.0205),
        ],
        segs=28, cap_ends=True,
    )
    bm_add_bevel_locked_hex(
        b_lhood, Vector((+0.004, -3.296, 0.542)), Vector((+0.005, -3.282, 0.543)),
        [(0.0, 0.011), (0.20, 0.013), (0.82, 0.013), (1.0, 0.010)], corner_bevel=0.14, cap_ends=True,
    )
    bm_add_segmented_tube(
        b_lhood, Vector((+0.024, -3.430, 0.526)), Vector((+0.032, -3.332, 0.534)),
        [
            (0.000, 0.0128), (0.020, 0.0155), (0.280, 0.0155),
            (0.295, 0.0182), (0.420, 0.0182), (0.435, 0.0155),
            (0.980, 0.0155), (1.000, 0.0128),
        ],
        segs=22, cap_ends=True,
    )
    for sx_lh in (-0.042, 0.0, +0.042):
        for sy_lh in (-3.480, -3.290):
            bm_add_aerospace_bolt(b_lhood, Vector((sx_lh, sy_lh, 0.514)), Vector((0, 0, 1)), head_r=0.0025, height=0.0022)
    bm_merge(bm_cleft, b_lhood, Matrix.Rotation(math.radians(44.0), 4, 'Y'))


    b_c_can = bmesh.new()
    bm_add_clevis_plate(b_c_can, (-3.575, 0.493), (-3.415, 0.528), 0.010, 0.012, -0.018, +0.018, arc_segs=11)
    bm_add_segmented_tube(
        b_c_can, Vector((0.0, -3.595, 0.534)), Vector((0.0, -3.395, 0.534)),
        [
            (0.000, 0.0230), (0.014, 0.0265), (0.285, 0.0265),
            (0.292, 0.0295), (0.358, 0.0295), (0.365, 0.0265),
            (0.555, 0.0265), (0.562, 0.0225), (0.618, 0.0225), (0.625, 0.0265),
            (0.695, 0.0265), (0.702, 0.0225), (0.758, 0.0225), (0.765, 0.0265),
            (0.986, 0.0265), (1.000, 0.0230),
        ],
        segs=26, cap_ends=True,
    )
    bm_add_bevel_locked_hex(b_c_can, Vector((0.0, -3.595, 0.534)), Vector((0.0, -3.608, 0.534)),
                            [(0.0, 0.008), (0.30, 0.010), (0.80, 0.010), (1.0, 0.0075)], corner_bevel=0.14, cap_ends=True)
    bm_merge(bm_cleft, b_c_can, Matrix.Rotation(math.radians(127.5), 4, 'Y'))

    b_c_mfd = bmesh.new()
    _bm_add_box_x_yz(b_c_mfd, -0.034, +0.034, -3.365, -3.215, 0.494, 0.528, nx=2, ny=3, nz=2)
    for y_cm_p in (-3.335, -3.285, -3.240):
        bm_add_segmented_tube(b_c_mfd, Vector((0.0, y_cm_p, 0.526)), Vector((0.0, y_cm_p, 0.546)),
                              [(0.0, 0.012), (0.45, 0.014), (1.0, 0.0095)], segs=18, cap_ends=True)
    bm_merge(bm_cleft, b_c_mfd, Matrix.Rotation(math.radians(136.5), 4, 'Y'))


    left_core_tubes = [
        ([(79.0, -3.752, 0.508), (79.0, -3.535, 0.510), (67.0, -3.415, 0.512), (67.0, -3.068, 0.510)], 0.0046, (1, 3), (1,), 0.046),
        ([(72.0, -3.752, 0.511), (72.0, -3.505, 0.512), (61.5, -3.395, 0.514), (61.5, -3.068, 0.512)], 0.0043, (1, 3), (1,), 0.046),
        ([(64.0, -3.752, 0.509), (64.0, -3.465, 0.511), (55.5, -3.380, 0.512), (55.5, -3.068, 0.510)], 0.0042, (1, 3), (1,), 0.044),
        ([(54.5, -3.752, 0.508), (54.5, -3.535, 0.509), (45.0, -3.485, 0.516), (45.0, -3.455, 0.520)], 0.0048, (1,), (1,), 0.038),
        ([(44.0, -3.295, 0.542), (44.0, -3.215, 0.518), (39.5, -3.165, 0.512), (39.5, -3.068, 0.510)], 0.0054, (3,), (1,), 0.042),
        ([(49.5, -3.335, 0.536), (49.5, -3.245, 0.516), (50.5, -3.068, 0.511)], 0.0044, (2,), (1,), 0.034),
        ([(35.0, -3.748, 0.508), (35.0, -3.068, 0.508)], 0.0045, (1,), (), 0.040),
        ([(27.5, -3.748, 0.506), (27.5, -3.068, 0.506)], 0.0040, (1,), (), 0.040),
        ([(18.0, -3.742, 0.510), (18.0, -3.072, 0.509)], 0.0054, (1,), (), 0.045),

        ([(90.0, -3.376, 0.536), (90.0, -3.346, 0.518), (85.0, -3.346, 0.512), (85.0, -3.068, 0.511)], 0.0038, (3,), (1,), 0.022),

        ([(98.5, -3.520, 0.544), (105.5, -3.520, 0.540), (107.0, -3.395, 0.534)], 0.0052, (1,), (), 0.026),
    ]
    bm_add_bevel_locked_hex(bm_cleft, _cp(107.0, -3.395, 0.518), _cp(107.0, -3.395, 0.536),
                            [(0.0, 0.0085), (0.25, 0.0105), (0.80, 0.0105), (1.0, 0.0078)], corner_bevel=0.14, cap_ends=True)
    for wps_l, tr_l, cl_l, un_l, br_l in left_core_tubes:
        _add_fwd_pipe(bm_cleft, wps_l, tube_r=tr_l, clamp_indices=cl_l, union_indices=un_l, r_case_ref=0.494, segs=18, bend_r=br_l)


    for wps_yh, tr_yh, cl_yh in [
        ([(81.75, -3.408, 0.547), (76.5, -3.392, 0.534), (68.0, -3.375, 0.522), (68.0, -3.112, 0.520), (88.0, -3.084, 0.522), (112.0, -3.084, 0.522), (131.0, -3.055, 0.528)], 0.0054, (2, 3, 5)),
        ([(33.5, -3.748, 0.512), (48.0, -3.445, 0.520), (68.0, -3.375, 0.522)], 0.0044, (1,)),
    ]:
        yh_fr = _add_fwd_pipe(bm_cleft, wps_yh, tube_r=tr_yh, clamp_indices=cl_yh, r_case_ref=0.494, segs=16, bend_r=0.052, bnut_ends=False)
        if len(yh_fr) >= 12:
            bm_add_wire_id_sleeve(bm_cleft, yh_fr[3][0], yh_fr[3][1], wire_r=tr_yh, length=0.014)
            bm_add_wire_id_sleeve(bm_cleft, yh_fr[-4][0], yh_fr[-4][1], wire_r=tr_yh, length=0.014)
            for f_yz in (0.24, 0.52, 0.78):
                izy = min(len(yh_fr) - 2, max(1, int(round(f_yz * (len(yh_fr) - 1)))))
                ph_zy = math.degrees(math.atan2(yh_fr[izy][0].x, yh_fr[izy][0].z)) % 360.0
                bm_add_harness_ziptie(bm_cleft, yh_fr[izy][0], yh_fr[izy][1], _cn(ph_zy), bundle_r=tr_yh)
    bm_add_segmented_tube(bm_cleft, _cp(68.0, -3.392, 0.522), _cp(68.0, -3.356, 0.522),
                          [(0.0, 0.0068), (0.35, 0.0086), (0.75, 0.0086), (1.0, 0.0068)], segs=18, cap_ends=True)


    for d_lk_end in (71.0, 118.0):
        b_lkb = bmesh.new()
        bm_add_clevis_plate(b_lkb, (-3.666, 0.493), (-3.644, 0.524), 0.007, 0.008, -0.009, +0.009, arc_segs=9)
        bm_add_chamfered_cyl_x(b_lkb, -3.655, 0.522, -0.011, +0.011, radius=0.0060, segs=16)
        bm_merge(bm_cleft, b_lkb, Matrix.Rotation(math.radians(d_lk_end), 4, 'Y'))
    _add_fwd_pipe(bm_cleft, [(71.5, -3.655, 0.522), (87.0, -3.655, 0.522)], tube_r=0.0042, r_case_ref=0.494, segs=14, bend_r=0.020)
    _add_fwd_pipe(bm_cleft, [(95.0, -3.655, 0.522), (117.5, -3.655, 0.522)], tube_r=0.0042, r_case_ref=0.494, segs=14, bend_r=0.020)


    _add_hex_tee_block(bm_cleft, 46.0, -3.640, 0.494, 0.510, branch_sgn=-1.0, tube_r=0.0044)
    _add_hex_tee_block(bm_cleft, 68.0, -3.245, 0.494, 0.512, branch_sgn=+1.0, tube_r=0.0040)
    _add_transducer_capsule(bm_cleft, 33.5, -3.165, 0.494, r_can=0.014, len_can=0.054)
    _add_transducer_capsule(bm_cleft, 79.5, -3.615, 0.494, r_can=0.0145, len_can=0.056)
    _add_port_flange(bm_cleft, 96.0, -3.235, 0.494, r_fl=0.022, n_bolts=6)
    _add_port_flange(bm_cleft, 128.0, -3.145, 0.494, r_fl=0.026, n_bolts=8)
    _add_transducer_capsule(bm_cleft, 120.5, -3.575, 0.494, r_can=0.015, len_can=0.058)

    left_dense_extra_tubes = [
        ([(46.0, -3.640, 0.510), (46.0, -3.695, 0.512), (23.0, -3.695, 0.510), (23.0, -3.070, 0.509)], 0.0044, (2, 3), (2,), 0.032),
        ([(44.0, -3.305, 0.518), (44.0, -3.225, 0.518), (33.5, -3.225, 0.516), (33.5, -3.195, 0.516)], 0.0042, (1,), (), 0.024),
        ([(85.5, -3.505, 0.536), (75.5, -3.505, 0.516), (75.5, -3.752, 0.513)], 0.0036, (2,), (1,), 0.022),
        ([(85.5, -3.425, 0.536), (68.0, -3.425, 0.513), (68.0, -3.245, 0.512)], 0.0036, (1,), (), 0.022),
        ([(96.0, -3.235, 0.504), (96.0, -3.235, 0.512), (86.0, -3.235, 0.510), (86.0, -3.068, 0.510)], 0.0036, (3,), (2,), 0.022),
        ([(99.5, -3.235, 0.504), (99.5, -3.235, 0.510), (91.0, -3.235, 0.508), (91.0, -3.068, 0.508)], 0.0034, (3,), (2,), 0.020),

        ([(116.0, -3.752, 0.508), (116.0, -3.560, 0.509), (121.0, -3.475, 0.510), (121.0, -3.320, 0.510), (114.0, -3.230, 0.510), (114.0, -3.068, 0.510)], 0.0042, (1, 3, 5), (2,), 0.038),
        ([(120.5, -3.752, 0.510), (120.5, -3.610, 0.512), (132.0, -3.485, 0.512), (133.0, -3.195, 0.512), (133.0, -3.068, 0.512)], 0.0040, (1, 3), (2,), 0.040),
        ([(125.0, -3.752, 0.512), (125.0, -3.625, 0.514), (139.5, -3.495, 0.514), (140.5, -3.185, 0.514), (140.5, -3.068, 0.514)], 0.0044, (1, 3), (2,), 0.042),
        ([(129.5, -3.752, 0.514), (129.5, -3.640, 0.516), (143.5, -3.505, 0.516), (144.5, -3.175, 0.516), (144.5, -3.068, 0.516)], 0.0038, (1, 3), (2,), 0.042),

        ([(118.5, -3.745, 0.524), (127.5, -3.608, 0.534)], 0.0052, (0,), (), 0.032),
        ([(127.5, -3.395, 0.534), (136.5, -3.350, 0.532), (136.5, -3.225, 0.532), (129.5, -3.068, 0.526)], 0.0050, (3,), (1,), 0.028),
    ]
    for wps_le, tr_le, cl_le, un_le, br_le in left_dense_extra_tubes:
        _add_fwd_pipe(bm_cleft, wps_le, tube_r=tr_le, clamp_indices=cl_le, union_indices=un_le, r_case_ref=0.494, segs=18, bend_r=br_le)

    for wps_flx in [
        [(113.0, -3.374, 0.550), (122.0, -3.335, 0.542), (136.5, -3.285, 0.542)],
        [(124.0, -3.740, 0.528), (134.5, -3.560, 0.538), (136.5, -3.360, 0.532)],
    ]:
        _add_fwd_pipe(bm_cleft, wps_flx, tube_r=0.0046, clamp_indices=(1,), r_case_ref=0.494, segs=16, corrugations=24, corr_amp=0.10, bend_r=0.032)

    _add_gang_clamp_bar(bm_cleft, 54.0, 80.0, -3.600, 0.494, 0.511)
    _add_gang_clamp_bar(bm_cleft, 55.0, 80.0, -3.445, 0.494, 0.513)
    _add_gang_clamp_bar(bm_cleft, 22.0, 40.0, -3.390, 0.494, 0.509)
    _add_gang_clamp_bar(bm_cleft, 55.0, 92.0, -3.135, 0.494, 0.512)
    _add_gang_clamp_bar(bm_cleft, 115.0, 131.0, -3.655, 0.494, 0.513)
    _add_gang_clamp_bar(bm_cleft, 131.0, 145.0, -3.420, 0.494, 0.515)

    create_mesh_obj("Fwd_04_Core_Left_SBleed_ServoValve_And_Tubes", bm_cleft, coll_fwd, clay)


    bm_cright = bmesh.new()


    u_configs = [
        (-3.668, -3.418, 273.5, 0.066),
        (-3.355, -3.148, 274.5, 0.102),
    ]
    for y_ufwd, y_uaft, phi_utop, br_u in u_configs:
        u_wps = [
            (231.0, y_ufwd, 0.486),
            (231.0, y_ufwd, 0.538),
            (phi_utop, y_ufwd, 0.546),
            (phi_utop, y_uaft, 0.546),
            (231.0, y_uaft, 0.538),
            (231.0, y_uaft, 0.486),
        ]
        u_pts = build_cyl_mandrel_pts(u_wps, bend_r=br_u, r_unwrap=0.542, arc_segs=20, max_step=0.010)
        ul_fr = bm_sweep_centerline_tube(bm_cright, u_pts, radius=0.0172, segs=28, cap_ends=True)
        _add_port_flange(bm_cright, 231.0, y_ufwd, 0.494, r_fl=0.029, n_bolts=6)
        _add_port_flange(bm_cright, 231.0, y_uaft, 0.494, r_fl=0.029, n_bolts=6)

        for f_ucol in (0.14, 0.86):
            idx_uf = min(len(ul_fr) - 2, max(1, int(round(f_ucol * (len(ul_fr) - 1)))))
            pf_u, tf_u = ul_fr[idx_uf][0], ul_fr[idx_uf][1]
            bm_add_segmented_tube(
                bm_cright, pf_u - tf_u * 0.008, pf_u + tf_u * 0.008,
                [(0.0, 0.0185), (0.20, 0.0238), (0.80, 0.0238), (1.0, 0.0185)], segs=26, cap_ends=True,
            )


    b_mvblk = bmesh.new()
    _bm_add_box_x_yz(b_mvblk, -0.044, +0.044, -3.612, -3.474, 0.493, 0.526, nx=3, ny=3, nz=2)
    _bm_add_box_x_yz(b_mvblk, -0.032, +0.032, -3.592, -3.494, 0.524, 0.548, nx=2, ny=3, nz=2)
    for y_vc in (-3.580, -3.543, -3.506):
        bm_add_chamfered_cyl_x(b_mvblk, y_vc, 0.536, -0.048, +0.048, radius=0.0115, segs=20)
    bm_add_chamfered_cyl_x(b_mvblk, -3.543, 0.514, -0.068, -0.042, radius=0.0185, segs=24)
    mv_bolts = []
    for x_mv, y_mv in ((-0.018, -3.576), (+0.018, -3.576), (-0.018, -3.510), (+0.018, -3.510)):
        pmv = Vector((x_mv, y_mv, 0.548))
        mv_bolts.append(pmv)
        bm_add_bevel_locked_hex(b_mvblk, pmv - Vector((0, 0, 0.002)), pmv + Vector((0, 0, 0.010)),
                                [(0.0, 0.0075), (0.20, 0.0092), (0.80, 0.0092), (1.0, 0.0070)], corner_bevel=0.14, cap_ends=True)
    bm_add_bolt_pair_lockwire(b_mvblk, mv_bolts[0], mv_bolts[1], Vector((0, 0, 1)), head_r=0.0075, strand_r=0.00036)
    bm_add_bolt_pair_lockwire(b_mvblk, mv_bolts[2], mv_bolts[3], Vector((0, 0, 1)), head_r=0.0075, strand_r=0.00036)
    bm_merge(bm_cright, b_mvblk, Matrix.Rotation(math.radians(241.0), 4, 'Y'))


    b_rcvv = bmesh.new()
    _bm_add_box_x_yz(b_rcvv, -0.028, +0.028, -3.605, -3.425, 0.493, 0.512, nx=2, ny=3, nz=2)
    bm_add_segmented_tube(
        b_rcvv, Vector((0.0, -3.615, 0.528)), Vector((0.0, -3.415, 0.528)),
        [(0.000, 0.0190), (0.014, 0.0220), (0.680, 0.0220), (0.692, 0.0140), (0.986, 0.0140), (1.000, 0.0115)],
        segs=26, cap_ends=True,
    )
    bm_add_segmented_tube(
        b_rcvv, Vector((-0.022, -3.595, 0.538)), Vector((-0.022, -3.445, 0.538)),
        [(0.0, 0.0085), (0.20, 0.0105), (0.80, 0.0105), (1.0, 0.0085)], segs=20, cap_ends=True,
    )
    bm_add_segmented_tube(
        b_rcvv, Vector((0.0, -3.702, 0.526)), Vector((0.0, -3.615, 0.528)),
        [(0.0, 0.0075), (0.20, 0.0055), (0.80, 0.0055), (1.0, 0.0085)], segs=18, cap_ends=True,
    )
    bm_add_flattened_aero_strut(
        b_rcvv, Vector((0.0, -3.702, 0.526)), Vector((-0.038, -3.702, 0.506)),
        half_w_mid=0.0065, half_h_mid=0.0032, end_half_w=0.0048, end_half_h=0.0032,
        eye_r=0.0075, eye_half_w=0.0035, taper_frac=0.22, pin_along_x=False, ribbed_pockets=1,
    )
    bm_merge(bm_cright, b_rcvv, Matrix.Rotation(math.radians(287.0), 4, 'Y'), default_mat_id=MAT_MACHINED_ALU)


    b_rjbox = bmesh.new()
    for y_rjb in (-3.530, -3.380):
        bm_add_clevis_plate(b_rjbox, (y_rjb - 0.008, 0.492), (y_rjb + 0.008, 0.506), 0.005, 0.006, -0.048, +0.048, arc_segs=9)
    _bm_add_box_x_yz(b_rjbox, -0.040, +0.040, -3.535, -3.375, 0.502, 0.524, nx=3, ny=3, nz=2)
    _bm_add_box_x_yz(b_rjbox, -0.044, +0.044, -3.539, -3.371, 0.522, 0.528, nx=3, ny=3, nz=1)
    _bm_add_box_x_yz(b_rjbox, -0.022, +0.022, -3.500, -3.410, 0.527, 0.533, nx=2, ny=2, nz=1, sup=0.08)
    bm_add_3d_plate_text(
        b_rjbox, ["J-BOX 2B", "28V DC"],
        Vector((0.0, -3.455, 0.533)), Vector((0.0, -1.0, 0.0)), Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)),
        width=0.076, height=0.036, relief=0.00060, with_dividers=True,
    )
    rjb_bpts = []
    for sy_rj in (-3.531, -3.379):
        for sx_rj in (-0.036, 0.0, +0.036):
            p_rjb = Vector((sx_rj, sy_rj, 0.528))
            rjb_bpts.append(p_rjb)
            bm_add_aerospace_bolt(b_rjbox, p_rjb, Vector((0, 0, 1)), head_r=0.0022, height=0.0020)
    for kb_rj in (0, 1, 3, 4):
        bm_add_bolt_pair_lockwire(b_rjbox, rjb_bpts[kb_rj], rjb_bpts[kb_rj + 1], Vector((0, 0, 1)), head_r=0.0022, strand_r=0.00026)
    for sx_rc in (-0.018, +0.018):
        bm_add_mil38999_connector(
            b_rjbox, Vector((sx_rc, -3.375, 0.514)), Vector((0.0, 1.0, 0.0)),
            shell_r=0.0080, with_flange=True,
        )
    bm_merge(bm_cright, b_rjbox, Matrix.Rotation(math.radians(310.0), 4, 'Y'), default_mat_id=MAT_AVIONICS_BOX)


    p_rcvv_cn0 = _cp(291.5, -3.455, 0.532)
    p_rcvv_cn1 = _cp(294.5, -3.442, 0.544)
    bm_add_mil38999_connector(bm_cright, p_rcvv_cn0, p_rcvv_cn1 - p_rcvv_cn0, shell_r=0.0082, with_flange=False)
    _add_port_flange(bm_cright, 235.0, -3.714, 0.494, r_fl=0.020, n_bolts=4)
    _add_port_flange(bm_cright, 245.0, -3.748, 0.494, r_fl=0.019, n_bolts=4)

    right_core_tubes = [

        ([(306.0, -3.455, 0.516), (280.5, -3.455, 0.520), (280.5, -3.714, 0.520), (235.0, -3.714, 0.514), (235.0, -3.714, 0.490)], 0.0072, (1, 3), (2,), 0.044),

        ([(324.0, -3.695, 0.514), (300.0, -3.695, 0.516), (290.0, -3.748, 0.516), (245.0, -3.748, 0.514), (245.0, -3.748, 0.490)], 0.0062, (1, 3), (1,), 0.040),

        ([(332.0, -3.092, 0.516), (242.0, -3.092, 0.520), (242.0, -3.065, 0.516)], 0.0074, (1,), (), 0.036),

        ([(244.0, -3.474, 0.532), (244.0, -3.386, 0.516), (259.0, -3.386, 0.512), (259.0, -3.068, 0.512)], 0.0034, (3,), (1,), 0.022),
        ([(238.0, -3.474, 0.530), (238.0, -3.400, 0.514), (255.0, -3.400, 0.510), (255.0, -3.068, 0.510)], 0.0034, (3,), (1,), 0.022),

        ([(290.0, -3.415, 0.534), (290.0, -3.335, 0.516), (304.0, -3.335, 0.512), (304.0, -3.070, 0.510)], 0.0036, (3,), (1,), 0.024),
        ([(285.0, -3.415, 0.528), (285.0, -3.350, 0.514), (299.0, -3.350, 0.510), (299.0, -3.070, 0.508)], 0.0036, (3,), (1,), 0.024),

        ([(342.0, -3.742, 0.509), (342.0, -3.072, 0.508)], 0.0052, (1,), (), 0.045),
        ([(328.0, -3.742, 0.507), (328.0, -3.460, 0.508), (320.0, -3.380, 0.508), (320.0, -3.072, 0.507)], 0.0042, (1, 3), (1,), 0.048),
        ([(249.0, -3.710, 0.510), (249.0, -3.068, 0.510)], 0.0046, (1,), (), 0.040),
        ([(294.5, -3.442, 0.542), (306.0, -3.415, 0.522), (306.0, -3.245, 0.518), (330.0, -3.105, 0.516)], 0.0046, (2,), (), 0.076),
    ]
    for wps_r, tr_r, cl_r, un_r, br_r in right_core_tubes:
        _add_fwd_pipe(bm_cright, wps_r, tube_r=tr_r, clamp_indices=cl_r, union_indices=un_r, r_case_ref=0.494, segs=18, bend_r=br_r)


    b_rlow = bmesh.new()
    _bm_add_box_x_yz(b_rlow, -0.032, +0.032, -3.615, -3.475, 0.494, 0.522, nx=2, ny=3, nz=2)
    bm_add_segmented_tube(
        b_rlow, Vector((0.0, -3.625, 0.532)), Vector((0.0, -3.465, 0.532)),
        [(0.000, 0.0175), (0.016, 0.0205), (0.455, 0.0205), (0.465, 0.0225),
         (0.535, 0.0225), (0.545, 0.0205), (0.984, 0.0205), (1.000, 0.0175)],
        segs=24, cap_ends=True,
    )
    for sx_rl in (-0.024, +0.024):
        for sy_rl in (-3.600, -3.490):
            bm_add_aerospace_bolt(b_rlow, Vector((sx_rl, sy_rl, 0.522)), Vector((0, 0, 1)), head_r=0.0021, height=0.0019)
    bm_merge(bm_cright, b_rlow, Matrix.Rotation(math.radians(221.5), 4, 'Y'))


    _add_hex_tee_block(bm_cright, 281.5, -3.225, 0.494, 0.512, branch_sgn=+1.0, tube_r=0.0042)
    _add_hex_tee_block(bm_cright, 313.0, -3.225, 0.494, 0.511, branch_sgn=-1.0, tube_r=0.0042)
    _add_transducer_capsule(bm_cright, 286.0, -3.225, 0.494, r_can=0.0135, len_can=0.052)

    right_dense_extra_tubes = [

        ([(246.5, -3.705, 0.508), (246.5, -3.068, 0.508)], 0.0038, (1,), (), 0.036),
        ([(252.0, -3.705, 0.509), (252.0, -3.068, 0.509)], 0.0040, (1,), (), 0.036),
        ([(262.5, -3.695, 0.509), (262.5, -3.068, 0.509)], 0.0038, (1,), (), 0.036),
        ([(266.5, -3.695, 0.508), (266.5, -3.068, 0.508)], 0.0036, (1,), (), 0.036),

        ([(246.0, -3.595, 0.526), (269.5, -3.595, 0.511), (269.5, -3.068, 0.510)], 0.0032, (2,), (1,), 0.020),
        ([(246.0, -3.555, 0.526), (264.5, -3.555, 0.511), (264.5, -3.068, 0.510)], 0.0032, (2,), (1,), 0.020),

        ([(236.0, -3.575, 0.522), (225.5, -3.575, 0.526), (225.5, -3.715, 0.518), (234.0, -3.715, 0.512)], 0.0042, (2,), (), 0.024),
        ([(221.5, -3.465, 0.530), (221.5, -3.320, 0.520), (226.0, -3.220, 0.516), (226.0, -3.068, 0.516)], 0.0046, (2, 3), (1,), 0.034),
        ([(217.5, -3.625, 0.526), (217.5, -3.735, 0.520), (217.5, -3.068, 0.518)], 0.0042, (2,), (), 0.034),

        ([(242.0, -3.092, 0.518), (281.5, -3.092, 0.514), (281.5, -3.455, 0.512), (285.5, -3.455, 0.518)], 0.0040, (2,), (1,), 0.024),
        ([(310.0, -3.375, 0.514), (313.0, -3.225, 0.511), (313.0, -3.070, 0.510)], 0.0042, (2,), (), 0.028),
        ([(290.5, -3.605, 0.528), (290.5, -3.658, 0.518), (307.5, -3.658, 0.516), (307.5, -3.535, 0.514)], 0.0038, (2,), (1,), 0.022),

        ([(335.0, -3.742, 0.512), (335.0, -3.560, 0.514), (325.0, -3.440, 0.512), (325.0, -3.070, 0.511)], 0.0054, (1, 3), (2,), 0.046),
    ]
    for wps_re, tr_re, cl_re, un_re, br_re in right_dense_extra_tubes:
        _add_fwd_pipe(bm_cright, wps_re, tube_r=tr_re, clamp_indices=cl_re, union_indices=un_re, r_case_ref=0.494, segs=18, bend_r=br_re)

    _add_gang_clamp_bar(bm_cright, 245.5, 270.0, -3.535, 0.494, 0.509)
    _add_gang_clamp_bar(bm_cright, 245.5, 270.0, -3.255, 0.494, 0.509)
    _add_gang_clamp_bar(bm_cright, 280.0, 305.0, -3.210, 0.494, 0.511)
    _add_gang_clamp_bar(bm_cright, 312.0, 343.0, -3.260, 0.494, 0.509)

    create_mesh_obj("Fwd_05_Core_Right_TwinUBleed_RCVV_And_Valves", bm_cright, coll_fwd, clay)


    bm_otank = bmesh.new()

    def _build_stamped_saddle_chamber(bm_dst_ch: bmesh.types.BMesh, bm_patches: bmesh.types.BMesh,
                                      phi_in0: float, phi_in1: float, y_in0: float, y_in1: float,
                                      n_phi_in: int, n_y_in: int, p_out_loop: list,
                                      r_deck: float, r_flange: float = 0.5065, r_back: float = 0.4995):

        b_ch = bm_patches.copy()
        bm_patches.free()

        p_in_loop = []
        for i in range(n_phi_in):
            p_in_loop.append((phi_in0 + (phi_in1 - phi_in0) * (i / float(n_phi_in)), y_in0))
        for j in range(n_y_in):
            p_in_loop.append((phi_in1, y_in0 + (y_in1 - y_in0) * (j / float(n_y_in))))
        for i in range(n_phi_in):
            p_in_loop.append((phi_in1 - (phi_in1 - phi_in0) * (i / float(n_phi_in)), y_in1))
        for j in range(n_y_in):
            p_in_loop.append((phi_in0, y_in1 - (y_in1 - y_in0) * (j / float(n_y_in))))

        n_loop = len(p_in_loop)
        assert len(p_out_loop) == n_loop, f"Loop mismatch: {len(p_out_loop)} vs {n_loop}"


        stamp_steps = [
            (0.00, r_deck),
            (0.50, r_deck),
            (0.66, r_deck),
            (0.70, r_deck - 0.0016),
            (0.82, r_flange + 0.0016),
            (0.86, r_flange),
            (0.97, r_flange),
            (1.00, r_flange - 0.0008),
            (1.00, r_back),
        ]
        border_rings = []
        for s_f, r_st in stamp_steps:
            ring_v = []
            for k in range(n_loop):
                ph_k = p_in_loop[k][0] + (p_out_loop[k][0] - p_in_loop[k][0]) * s_f
                yv_k = p_in_loop[k][1] + (p_out_loop[k][1] - p_in_loop[k][1]) * s_f
                pr_k = math.radians(ph_k)
                ring_v.append(b_ch.verts.new((r_st * math.sin(pr_k), yv_k, r_st * math.cos(pr_k))))
            border_rings.append(ring_v)

        for ir_b in range(len(border_rings) - 1):
            ra_b, rb_b = border_rings[ir_b], border_rings[ir_b + 1]
            for k in range(n_loop):
                kn = (k + 1) % n_loop
                b_ch.faces.new((ra_b[k], rb_b[k], rb_b[kn], ra_b[kn]))


        back_ring = border_rings[-1]
        b_bot_uv   = [p_out_loop[i] for i in range(n_phi_in + 1)]
        b_right_uv = [p_out_loop[n_phi_in + j] for j in range(n_y_in + 1)]
        b_top_uv   = [p_out_loop[2 * n_phi_in + n_y_in - i] for i in range(n_phi_in + 1)]
        b_left_uv  = [p_out_loop[0]] + [p_out_loop[n_loop - j] for j in range(1, n_y_in)] + [p_out_loop[2 * n_phi_in + n_y_in]]
        c00, c10, c11, c01 = b_bot_uv[0], b_bot_uv[n_phi_in], b_top_uv[n_phi_in], b_top_uv[0]

        bg_grid = [[None] * (n_y_in + 1) for _ in range(n_phi_in + 1)]
        for i in range(n_phi_in + 1):
            bg_grid[i][0] = back_ring[i]
            bg_grid[i][n_y_in] = back_ring[2 * n_phi_in + n_y_in - i]
        for j in range(1, n_y_in):
            bg_grid[n_phi_in][j] = back_ring[n_phi_in + j]
            bg_grid[0][j] = back_ring[n_loop - j]
        for i in range(1, n_phi_in):
            xi = i / float(n_phi_in)
            for j in range(1, n_y_in):
                eta = j / float(n_y_in)
                u_c = ((1.0 - eta) * b_bot_uv[i][0] + eta * b_top_uv[i][0]
                       + (1.0 - xi) * b_left_uv[j][0] + xi * b_right_uv[j][0]
                       - ((1.0 - xi) * (1.0 - eta) * c00[0] + xi * (1.0 - eta) * c10[0]
                          + xi * eta * c11[0] + (1.0 - xi) * eta * c01[0]))
                v_c = ((1.0 - eta) * b_bot_uv[i][1] + eta * b_top_uv[i][1]
                       + (1.0 - xi) * b_left_uv[j][1] + xi * b_right_uv[j][1]
                       - ((1.0 - xi) * (1.0 - eta) * c00[1] + xi * (1.0 - eta) * c10[1]
                          + xi * eta * c11[1] + (1.0 - xi) * eta * c01[1]))
                pr_c = math.radians(u_c)
                bg_grid[i][j] = b_ch.verts.new((r_back * math.sin(pr_c), v_c, r_back * math.cos(pr_c)))
        for i in range(n_phi_in):
            for j in range(n_y_in):
                b_ch.faces.new((bg_grid[i][j], bg_grid[i][j + 1], bg_grid[i + 1][j + 1], bg_grid[i + 1][j]))

        bmesh.ops.remove_doubles(b_ch, verts=[v for v in b_ch.verts if v.is_boundary], dist=1e-4)
        bmesh.ops.recalc_face_normals(b_ch, faces=b_ch.faces)
        bm_merge(bm_dst_ch, b_ch)

    def _add_tie_bolt_hardware(bm_dst_tb: bmesh.types.BMesh, phi_tb: float, y_tb: float, r_floor: float):

        b_tb = bmesh.new()
        bm_add_segmented_tube(
            b_tb, Vector((0.0, y_tb, r_floor - 0.0010)), Vector((0.0, y_tb, r_floor + 0.0046)),
            [(0.0, 0.0112), (0.45, 0.0112), (0.55, 0.0086), (1.0, 0.0082)], segs=24, cap_ends=True,
        )
        bm_add_aerospace_bolt(b_tb, Vector((0.0, y_tb, r_floor + 0.0046)), Vector((0, 0, 1)),
                              head_r=0.0046, height=0.0034, with_12pt_nut_depth=0.0030)
        bm_merge(bm_dst_tb, b_tb, Matrix.Rotation(math.radians(phi_tb), 4, 'Y'))

    tie_cup_prof = [(0.0, 0.0), (0.00075, 0.00085), (0.0010, 0.0048), (0.0013, 0.0082), (0.0022, 0.0088)]


    r_up_deck = 0.5530
    b_up_p = bmesh.new()
    for ya_u, yb_u in ((-4.052, -3.972), (-3.972, -3.892)):
        bm_add_cyl_conical_pocket_patch(
            b_up_p, 260.0, 268.2, ya_u, yb_u, n_phi=12, n_y=12,
            r_outer_fn=(lambda _p, _y: r_up_deck), half_arc_w=0.0155, half_y_h=0.0155,
            depth_profile=tie_cup_prof, stadium_axis='ELLIPSE',
        )
        _add_tie_bolt_hardware(bm_otank, 264.1, 0.5 * (ya_u + yb_u), r_up_deck - 0.0088)

    up_out_loop = []

    for i in range(12):
        if i <= 2:
            up_out_loop.append((258.0 + 3.2 * (i / 2.0), -4.152))
        else:
            fu = (i - 2) / 10.0
            up_out_loop.append((261.2 + (271.8 - 261.2) * fu, -4.152 + (-4.042 - (-4.152)) * fu))

    for j in range(24):
        if j <= 8:
            fv = j / 8.0
            up_out_loop.append((271.8 + (279.5 - 271.8) * fv, -4.042 + (-3.962 - (-4.042)) * fv))
        else:
            fv = (j - 8) / 16.0
            up_out_loop.append((279.5 + (274.2 - 279.5) * fv, -3.962 + (-3.800 - (-3.962)) * fv))

    for i in range(12):
        up_out_loop.append((274.2 - (274.2 - 258.0) * (i / 12.0), -3.800))

    for j in range(24):
        up_out_loop.append((258.0, -3.800 + (-4.152 - (-3.800)) * (j / 24.0)))

    _build_stamped_saddle_chamber(bm_otank, b_up_p, 260.0, 268.2, -4.052, -3.892, 12, 24, up_out_loop, r_deck=r_up_deck)


    r_lo_deck = 0.5540
    b_lo_p = bmesh.new()
    for ph_r0, ph_r1 in ((215.0, 227.2), (227.2, 239.4), (239.4, 251.6)):
        bm_add_cyl_conical_pocket_patch(
            b_lo_p, ph_r0, ph_r1, -4.016, -3.930, n_phi=12, n_y=12,
            r_outer_fn=(lambda _p, _y: r_lo_deck), half_arc_w=0.0170, half_y_h=0.0170,
            depth_profile=tie_cup_prof, stadium_axis='ELLIPSE',
        )
        _add_tie_bolt_hardware(bm_otank, 0.5 * (ph_r0 + ph_r1), -3.973, r_lo_deck - 0.0088)


    sg_pocket_prof = [(0.0, 0.0), (0.00080, 0.00090), (0.0011, 0.0052), (0.0014, 0.0092), (0.0022, 0.0098)]
    bm_add_cyl_conical_pocket_patch(
        b_lo_p, 227.2, 251.6, -4.102, -4.016, n_phi=24, n_y=12,
        r_outer_fn=(lambda _p, _y: r_lo_deck), half_arc_w=0.0820, half_y_h=0.0235,
        depth_profile=sg_pocket_prof, stadium_axis='PHI',
    )
    bm_add_cyl_plain_patch(b_lo_p, 215.0, 227.2, -4.102, -4.016, n_phi=12, n_y=12, r_outer_fn=(lambda _p, _y: r_lo_deck))

    lo_out_loop = []

    for i in range(36):
        if i <= 14:
            fu = i / 14.0
            lo_out_loop.append((211.2 + (231.5 - 211.2) * fu, -4.028 + (-4.164 - (-4.028)) * fu))
        else:
            fu = (i - 14) / 22.0
            lo_out_loop.append((231.5 + (256.0 - 231.5) * fu, -4.164))

    for j in range(24):
        lo_out_loop.append((256.0, -4.164 + (-3.800 - (-4.164)) * (j / 24.0)))

    for i in range(36):
        lo_out_loop.append((256.0 - (256.0 - 211.2) * (i / 36.0), -3.800))

    for j in range(24):
        lo_out_loop.append((211.2, -3.800 + (-4.028 - (-3.800)) * (j / 24.0)))

    _build_stamped_saddle_chamber(bm_otank, b_lo_p, 215.0, 251.6, -4.102, -3.930, 36, 24, lo_out_loop, r_deck=r_lo_deck)


    b_sump_p = bmesh.new()
    bm_add_cyl_plain_patch(b_sump_p, 204.2, 207.2, -3.985, -3.840, n_phi=6, n_y=12, r_outer_fn=(lambda _p, _y: 0.5475))
    sump_out_loop = (
        [(202.0 + 7.4 * (i / 6.0), -4.024) for i in range(6)]
        + [(209.4, -4.024 + 0.222 * (j / 12.0)) for j in range(12)]
        + [(209.4 - 7.4 * (i / 6.0), -3.802) for i in range(6)]
        + [(202.0, -3.802 - 0.222 * (j / 12.0)) for j in range(12)]
    )
    _build_stamped_saddle_chamber(bm_otank, b_sump_p, 204.2, 207.2, -3.985, -3.840, 6, 12, sump_out_loop, r_deck=0.5475)

    def _add_chamfered_cyl_band(bm_t: bmesh.types.BMesh, pa: float, pb: float, ya: float, yb: float,
                                r_in_b: float, r_out_b: float, np_seg: int = 6, ny_seg: int = 12,
                                taper_p: float = 0.08, taper_y: float = 0.04):
        tp_b = subdivide_1d_with_support_loops(np_seg, sup1=0.04, sup2=max(0.08, taper_p))
        ty_b = subdivide_1d_with_support_loops(ny_seg, sup1=0.02, sup2=max(0.05, taper_y))
        gi_b, go_b = [], []
        for fp_b in tp_b:
            ar_b = math.radians(pa + (pb - pa) * fp_b)
            sb_v, cb_v = math.sin(ar_b), math.cos(ar_b)
            ep_b = min(1.0, min(fp_b, 1.0 - fp_b) / max(1e-4, taper_p))
            ci_b, co_b = [], []
            for fy_b in ty_b:
                yv_b = ya + (yb - ya) * fy_b
                ey_b = min(1.0, min(fy_b, 1.0 - fy_b) / max(1e-4, taper_y))
                ro_cur = r_in_b + (r_out_b - r_in_b) * (0.08 + 0.92 * min(ep_b, ey_b))
                ci_b.append(bm_t.verts.new((r_in_b * sb_v, yv_b, r_in_b * cb_v)))
                co_b.append(bm_t.verts.new((ro_cur * sb_v, yv_b, ro_cur * cb_v)))
            gi_b.append(ci_b)
            go_b.append(co_b)
        bm_build_beveled_quad_shell(bm_t, gi_b, go_b, bulge_scale=0.04)


    for ph_st_c, ya_st, yb_st, r_st_b in ((257.0, -4.156, -3.799, 0.5495), (210.3, -4.026, -3.799, 0.5465)):
        _add_chamfered_cyl_band(bm_otank, ph_st_c - 1.12, ph_st_c + 1.12, ya_st, yb_st, 0.5045, r_st_b, np_seg=4, ny_seg=14, taper_p=0.12, taper_y=0.025)
        _add_chamfered_cyl_band(bm_otank, ph_st_c - 0.65, ph_st_c + 0.65, ya_st - 0.002, yb_st + 0.002, r_st_b - 0.0005, r_st_b + 0.0038, np_seg=4, ny_seg=14, taper_p=0.14, taper_y=0.025)
        for y_tb_end, sgn_tb in ((ya_st - 0.003, -1.0), (yb_st + 0.003, +1.0)):
            p_tbe = _cp(ph_st_c, y_tb_end, r_st_b - 0.005)
            bm_add_bevel_locked_hex(bm_otank, p_tbe, p_tbe + Vector((0, sgn_tb * 0.011, 0)),
                                    [(0.0, 0.0048), (0.30, 0.0062), (0.80, 0.0062), (1.0, 0.0045)], corner_bevel=0.14, cap_ends=True)


    for ph_rb0, ph_rb1, y_rb_c in ((233.5, 249.5, -4.122), (218.0, 238.0, -3.868)):
        _add_chamfered_cyl_band(bm_otank, ph_rb0, ph_rb1, y_rb_c - 0.0060, y_rb_c + 0.0060,
                                r_lo_deck - 0.0006, r_lo_deck + 0.0036, np_seg=10, ny_seg=4, taper_p=0.14, taper_y=0.28)


    for k_pb in range(8):
        y_pb = -4.010 + k_pb * 0.030
        bm_add_aerospace_bolt(bm_otank, _cp(202.8, y_pb, 0.5070), _cn(202.8), head_r=0.0022, height=0.0020)
    for k_pba in range(9):
        deg_pba = 212.0 + k_pba * 7.5
        bm_add_aerospace_bolt(bm_otank, _cp(deg_pba, -3.804, 0.5070), _cn(deg_pba), head_r=0.0022, height=0.0020)
    for k_pbf in range(5):
        deg_pbf = 234.0 + k_pbf * 5.2
        bm_add_aerospace_bolt(bm_otank, _cp(deg_pbf, -4.160, 0.5070), _cn(deg_pbf), head_r=0.0022, height=0.0020)


    b_sg = bmesh.new()
    r_sg_f = r_lo_deck - 0.0098
    _bm_add_box_x_yz(b_sg, -0.070, +0.070, -4.076, -4.042, r_sg_f - 0.0015, r_sg_f + 0.0014, nx=3, ny=2, nz=1, chamfer=0.0005)
    for sy_rail0, sy_rail1 in ((-4.076, -4.0655), (-4.0525, -4.042)):
        _bm_add_box_x_yz(b_sg, -0.070, +0.070, sy_rail0, sy_rail1, r_sg_f + 0.0010, r_sg_f + 0.0055, nx=3, ny=1, nz=1, chamfer=0.0005)
    for sx_end0, sx_end1 in ((-0.070, -0.053), (+0.053, +0.070)):
        _bm_add_box_x_yz(b_sg, sx_end0, sx_end1, -4.066, -4.052, r_sg_f + 0.0010, r_sg_f + 0.0055, nx=1, ny=2, nz=1, chamfer=0.0005)

    f0_sg_ref = len(b_sg.faces)
    _bm_add_box_x_yz(b_sg, -0.053, +0.053, -4.0652, -4.0528, r_sg_f + 0.0009, r_sg_f + 0.0017, nx=4, ny=2, nz=1, chamfer=0.0002)
    bm_fill_unset_from(b_sg, f0_sg_ref, MAT_MACHINED_ALU)
    for sy_lip in (-4.072, -4.046):
        _bm_add_box_x_yz(b_sg, -0.066, +0.066, sy_lip - 0.0038, sy_lip + 0.0038, r_sg_f + 0.0050, r_sg_f + 0.0096, nx=3, ny=1, nz=1, chamfer=0.0006)
    for sx_lip in (-0.060, +0.060):
        _bm_add_box_x_yz(b_sg, sx_lip - 0.0050, sx_lip + 0.0050, -4.069, -4.049, r_sg_f + 0.0050, r_sg_f + 0.0096, nx=1, ny=2, nz=1, chamfer=0.0006)
    bm_add_chamfered_cyl_x(b_sg, -4.059, r_sg_f + 0.0072, -0.052, +0.052, radius=0.0054, segs=22)

    bm_add_chamfered_cyl_x(b_sg, -4.059, r_sg_f + 0.0074, +0.010, +0.015, radius=0.0064, segs=18)
    for k_tk in range(11):
        xt_k = -0.045 + k_tk * 0.009
        tk_len = 0.0028 if (k_tk % 5 == 0) else 0.0016
        _bm_add_box_x_yz(b_sg, xt_k - 0.0005, xt_k + 0.0005, -4.049, -4.049 + tk_len, r_sg_f + 0.0092, r_sg_f + 0.0103, nx=1, ny=1, nz=1, chamfer=0.00015)
        _bm_add_box_x_yz(b_sg, xt_k - 0.0005, xt_k + 0.0005, -4.069 - tk_len, -4.069, r_sg_f + 0.0092, r_sg_f + 0.0103, nx=1, ny=1, nz=1, chamfer=0.00015)

    bm_add_3d_plate_text(
        b_sg, ["FULL"], Vector((+0.060, -4.059, r_sg_f + 0.0096)),
        Vector((0.0, -1.0, 0.0)), Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)),
        width=0.016, height=0.0075, relief=0.00055,
    )
    bm_add_3d_plate_text(
        b_sg, ["ADD"], Vector((-0.060, -4.059, r_sg_f + 0.0096)),
        Vector((0.0, -1.0, 0.0)), Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)),
        width=0.016, height=0.0075, relief=0.00055,
    )
    for k_fl in (-0.036, -0.018, 0.0, +0.018, +0.036):
        bm_add_chamfered_cyl_x(b_sg, -4.059, r_sg_f + 0.0078, k_fl - 0.0020, k_fl + 0.0020, radius=0.0068, segs=18)
    sg_top_b, sg_bot_b = [], []
    for sx_sg in (-0.058, -0.035, -0.012, +0.012, +0.035, +0.058):
        p_s1 = Vector((sx_sg, -4.072, r_sg_f + 0.0096))
        p_s2 = Vector((sx_sg, -4.046, r_sg_f + 0.0096))
        sg_bot_b.append(p_s1)
        sg_top_b.append(p_s2)
        bm_add_aerospace_bolt(b_sg, p_s1, Vector((0, 0, 1)), head_r=0.0018, height=0.0017)
        bm_add_aerospace_bolt(b_sg, p_s2, Vector((0, 0, 1)), head_r=0.0018, height=0.0017)
    for k_sg_lw in (0, 2, 4):
        bm_add_bolt_pair_lockwire(b_sg, sg_bot_b[k_sg_lw], sg_bot_b[k_sg_lw + 1], Vector((0, 0, 1)), head_r=0.0018, strand_r=0.00024)
        bm_add_bolt_pair_lockwire(b_sg, sg_top_b[k_sg_lw], sg_top_b[k_sg_lw + 1], Vector((0, 0, 1)), head_r=0.0018, strand_r=0.00024)
    bm_merge(bm_otank, b_sg, Matrix.Rotation(math.radians(239.4), 4, 'Y'), default_mat_id=MAT_PLATE_BASE)


    _add_chamfered_cyl_band(bm_otank, 217.8, 225.2, -4.090, -4.028, r_lo_deck - 0.0004, r_lo_deck + 0.0016, np_seg=6, ny_seg=6, taper_p=0.08, taper_y=0.06)
    for dp_or in (218.6, 224.4):
        for dy_or in (-4.084, -4.034):
            bm_add_dome_rivet(bm_otank, _cp(dp_or, dy_or, r_lo_deck + 0.0016), _cn(dp_or), radius=0.0011)
    bm_add_conformal_cyl_text(
        bm_otank, ["MIL-L-7808", "OIL 14.5QT"],
        218.6, 224.4, -4.084, -4.034, r_lo_deck + 0.0016,
        relief=0.00055, read_mode='AXIAL_STBD', with_dividers=True,
    )


    p_fn0 = _cp(222.5, -4.098, 0.526)
    p_fn1 = _cp(225.5, -4.132, 0.552)
    p_fn_lug = _cp(218.5, -4.088, 0.528)
    bm_add_segmented_tube(bm_otank, p_fn0, p_fn1,
                          [(0.0, 0.015), (0.62, 0.015), (0.70, 0.020), (0.92, 0.020), (1.0, 0.017)], segs=24, cap_ends=True)
    bm_add_dome_rivet(bm_otank, p_fn_lug, _cn(218.5), radius=0.0030)
    bm_add_spline_tube(bm_otank, [_cp(224.8, -4.126, 0.548), _cp(221.5, -4.118, 0.540), p_fn_lug], radius=0.0013, n_samples=18, segs=12, cap_ends=True)


    for d_ot0, d_ot_top, d_ot_mid, d_ot1, y_ot_l, tr_ot in [
        (289.2, 285.0, 266.8, 248.0, -4.234, 0.0056),
        (292.6, 288.4, 270.2, 253.5, -4.234, 0.0044),
    ]:
        _add_fwd_pipe(
            bm_otank,
            [(d_ot0, -3.782, 0.516), (d_ot0, -3.868, 0.528), (d_ot_top, -3.942, 0.544),
             (d_ot_mid, -4.150, 0.540), (d_ot1, -4.205, 0.524), (d_ot1, y_ot_l, 0.508)],
            tube_r=tr_ot, clamp_indices=(1, 3, 4), union_indices=(2, 3), r_case_ref=0.504, segs=18, bend_r=0.042,
        )
        p_otb = _cp(d_ot1, y_ot_l, 0.507)
        bm_add_segmented_tube(bm_otank, p_otb, p_otb + _cn(d_ot1) * 0.012,
                              [(0.0, 0.012), (0.65, 0.012), (1.0, 0.0085)], segs=18, cap_ends=True)


    b_odrn = bmesh.new()
    bm_add_segmented_tube(b_odrn, Vector((0.0, -4.172, 0.520)), Vector((0.0, -4.172, 0.568)),
                          [(0.0, 0.022), (0.35, 0.022), (0.45, 0.016), (0.82, 0.018), (1.0, 0.014)], segs=24, cap_ends=True)
    bm_add_bevel_locked_hex(b_odrn, Vector((0.0, -4.172, 0.566)), Vector((0.0, -4.172, 0.578)),
                            [(0.0, 0.012), (0.25, 0.0145), (0.80, 0.0145), (1.0, 0.011)], corner_bevel=0.14, cap_ends=True)
    bm_add_spline_tube(b_odrn, [Vector((0.012, -4.172, 0.572)), Vector((0.022, -4.160, 0.558)), Vector((0.016, -4.150, 0.542))],
                       radius=0.0012, n_samples=18, segs=12, cap_ends=True)
    bm_merge(bm_otank, b_odrn, Matrix.Rotation(math.radians(213.5), 4, 'Y'))

    create_mesh_obj("Fwd_06_Right_Stamped_Saddle_OilTank", bm_otank, coll_fwd, clay)


    bm_agb = bmesh.new()


    b_ufc = bmesh.new()
    _bm_add_box_x_yz(b_ufc, -0.068, +0.068, -4.195, -3.795, 0.500, 0.562, nx=3, ny=5, nz=2)
    _bm_add_box_x_yz(b_ufc, -0.054, +0.054, -4.170, -3.820, 0.558, 0.606, nx=3, ny=4, nz=2)
    for y_uf_r in (-4.140, -4.055, -3.970, -3.885):
        _bm_add_box_x_yz(b_ufc, -0.060, +0.060, y_uf_r - 0.0035, y_uf_r + 0.0035, 0.520, 0.602, nx=2, ny=1, nz=1, sup=0.08)
    for x_ud, y_ud, r_ud, h_ud in [(-0.026, -4.115, 0.022, 0.630), (+0.026, -4.015, 0.024, 0.632), (-0.020, -3.895, 0.020, 0.626)]:
        bm_add_segmented_tube(b_ufc, Vector((x_ud, y_ud, 0.600)), Vector((x_ud, y_ud, h_ud)),
                              [(0.0, r_ud), (0.20, r_ud * 1.08), (0.82, r_ud * 1.04), (1.0, r_ud * 0.72)], segs=24, cap_ends=True)
        ud_bpts = []
        for kb_u in range(6):
            au = 2.0 * math.pi * kb_u / 6.0
            p_udb = Vector((x_ud + r_ud * 0.85 * math.cos(au), y_ud + r_ud * 0.85 * math.sin(au), 0.606))
            ud_bpts.append(p_udb)
            bm_add_aerospace_bolt(b_ufc, p_udb, Vector((0, 0, 1)), head_r=0.0020, height=0.0018)
        for kb_u in range(0, 6, 2):
            bm_add_bolt_pair_lockwire(b_ufc, ud_bpts[kb_u], ud_bpts[kb_u + 1], Vector((0, 0, 1)), head_r=0.0020, strand_r=0.00025)
    bm_add_segmented_tube(b_ufc, Vector((+0.030, -4.185, 0.582)), Vector((+0.030, -3.835, 0.582)),
                          [(0.0, 0.018), (0.15, 0.023), (0.85, 0.023), (1.0, 0.017)], segs=24, cap_ends=True)
    for y_trm in (-4.135, -3.945):
        bm_add_knurled_vernier_knob(b_ufc, Vector((-0.054, y_trm, 0.582)), Vector((-1.0, 0.0, 0.0)),
                                    knob_r=0.0125, knob_h=0.0185, n_knurls=18, with_lockwire_lug=True)
        bm_add_bevel_locked_hex(b_ufc, Vector((+0.052, y_trm, 0.544)), Vector((+0.068, y_trm, 0.544)),
                                [(0.0, 0.008), (0.25, 0.0095), (0.80, 0.0095), (1.0, 0.0075)], corner_bevel=0.14, cap_ends=True)

    _bm_add_box_x_yz(b_ufc, -0.070, -0.0675, -4.085, -4.000, 0.520, 0.554, nx=1, ny=2, nz=2, chamfer=0.0004)
    bm_add_3d_plate_text(
        b_ufc, ["JFC-90 UFC", "HAMILTON"],
        Vector((-0.070, -4.0425, 0.537)),
        Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, -1.0)), Vector((-1.0, 0.0, 0.0)),
        width=0.074, height=0.026, relief=0.00055, with_border=True, with_dividers=True,
    )
    for sy_up in (-4.079, -4.006):
        for sz_up in (0.525, 0.549):
            bm_add_dome_rivet(b_ufc, Vector((-0.070, sy_up, sz_up)), Vector((-1, 0, 0)), radius=0.0011)
    bm_add_mil38999_connector(
        b_ufc, Vector((-0.032, -3.820, 0.582)), Vector((0.0, 1.0, 0.0)),
        shell_r=0.0102, with_flange=True,
        cable_pts=[Vector((-0.044, -3.770, 0.568)), Vector((-0.052, -3.725, 0.548))],
        cable_braided=True, n_zipties=2,
    )
    bm_merge(bm_agb, b_ufc, Matrix.Rotation(math.radians(156.0), 4, 'Y'), default_mat_id=MAT_AVIONICS_BOX)


    b_fsump = bmesh.new()
    _bm_add_box_x_yz(b_fsump, -0.056, +0.056, -4.238, -3.775, 0.498, 0.566, nx=3, ny=5, nz=2)
    bm_add_segmented_tube(
        b_fsump, Vector((0.0, -4.235, 0.588)), Vector((0.0, -3.775, 0.596)),
        [(0.0, 0.026), (0.10, 0.038), (0.48, 0.040), (0.54, 0.044), (0.90, 0.040), (1.0, 0.034)], segs=28, cap_ends=True,
    )

    bm_add_segmented_tube(
        b_fsump, Vector((+0.016, -4.095, 0.562)), Vector((+0.016, -4.095, 0.634)),
        [(0.0, 0.032), (0.25, 0.034), (0.78, 0.030), (1.0, 0.022)], segs=26, cap_ends=True,
    )
    bm_add_bevel_locked_hex(b_fsump, Vector((+0.016, -4.095, 0.632)), Vector((+0.016, -4.095, 0.641)),
                            [(0.0, 0.009), (0.25, 0.011), (0.80, 0.011), (1.0, 0.008)], corner_bevel=0.14, cap_ends=True)
    bm_add_aerospace_bolt(b_fsump, Vector((-0.010, -4.095, 0.631)), Vector((0, 0, 1)), head_r=0.0022, height=0.0020)
    bm_add_bolt_pair_lockwire(b_fsump, Vector((+0.016, -4.095, 0.638)), Vector((-0.010, -4.095, 0.631)), Vector((0, 0, 1)), head_r=0.0038, strand_r=0.00030)

    bm_add_clevis_plate(b_fsump, (-4.260, 0.518), (-4.190, 0.598), 0.012, 0.018, -0.028, +0.028, arc_segs=13)
    bm_add_segmented_tube(
        b_fsump, Vector((-0.018, -4.256, 0.576)), Vector((-0.018, -4.165, 0.614)),
        [(0.0, 0.018), (0.25, 0.024), (0.78, 0.024), (1.0, 0.018)], segs=22, cap_ends=True,
    )
    bm_merge(bm_agb, b_fsump, Matrix.Rotation(math.radians(184.0), 4, 'Y'))

    b_aiv = bmesh.new()

    _bm_add_box_x_yz(b_aiv, -0.044, +0.044, -4.242, -4.088, 0.516, 0.576, nx=3, ny=3, nz=2)
    _bm_add_box_x_yz(b_aiv, -0.036, +0.036, -4.228, -4.112, 0.572, 0.618, nx=2, ny=3, nz=2)
    bm_add_segmented_tube(
        b_aiv, Vector((0.0, -4.248, 0.548)), Vector((0.0, -4.015, 0.552)),
        [(0.000, 0.0290), (0.016, 0.0340), (0.560, 0.0340), (0.575, 0.0260), (0.984, 0.0260), (1.000, 0.0220)],
        segs=24, cap_ends=True,
    )
    bm_add_segmented_tube(b_aiv, Vector((-0.014, -4.165, 0.562)), Vector((-0.014, -4.165, 0.632)),
                          [(0.0, 0.026), (0.50, 0.032), (0.85, 0.030), (1.0, 0.020)], segs=24, cap_ends=True)
    aiv_bpts = []
    for sy_ab_f, sz_ab_f in ((-4.205, 0.585), (-4.170, 0.608), (-4.135, 0.608), (-4.120, 0.585), (-4.148, 0.576), (-4.192, 0.576)):
        p_aivb = Vector((+0.037, sy_ab_f, sz_ab_f))
        aiv_bpts.append(p_aivb)
        bm_add_aerospace_bolt(b_aiv, p_aivb, Vector((1, 0, 0)), head_r=0.0021, height=0.0020)
    for k_aiv in (0, 2, 4):
        bm_add_bolt_pair_lockwire(b_aiv, aiv_bpts[k_aiv], aiv_bpts[k_aiv + 1], Vector((1, 0, 0)), head_r=0.0021, strand_r=0.00026)
    bm_add_bevel_locked_hex(b_aiv, Vector((-0.044, -4.175, 0.596)), Vector((-0.058, -4.175, 0.596)),
                            [(0.0, 0.009), (0.30, 0.0115), (0.80, 0.0115), (1.0, 0.0085)], corner_bevel=0.14, cap_ends=True)
    bm_merge(bm_agb, b_aiv, Matrix.Rotation(math.radians(204.0), 4, 'Y'))


    for wps_chin, tr_ch, cr_ch in [
        ([(213.5, -4.172, 0.556), (206.0, -4.215, 0.622), (196.5, -4.185, 0.614), (186.0, -4.155, 0.592)], 0.0072, 26),
        ([(208.0, -4.236, 0.526), (200.5, -4.242, 0.586), (190.0, -4.212, 0.584)], 0.0058, 0),
        ([(206.0, -4.222, 0.528), (198.5, -4.228, 0.572), (188.0, -4.198, 0.572)], 0.0052, 0),
        ([(178.0, -4.242, 0.578), (166.5, -4.225, 0.588), (156.0, -4.175, 0.576)], 0.0056, 22),
        ([(174.0, -4.228, 0.566), (161.0, -4.205, 0.568), (146.0, -4.188, 0.548)], 0.0048, 0),
        ([(148.0, -4.236, 0.538), (141.5, -4.215, 0.562), (135.5, -4.155, 0.556)], 0.0052, 22),
    ]:
        _add_fwd_pipe(bm_agb, wps_chin, tube_r=tr_ch, clamp_indices=(1,), r_case_ref=0.512, segs=16, corrugations=cr_ch, corr_amp=0.10, bend_r=0.028)


    t_gb_p = sorted(set(
        subdivide_1d_with_support_loops(20, sup1=0.008, sup2=0.026)
        + [0.225, 0.245, 0.755, 0.775]
    ))
    t_gb_y = subdivide_1d_with_support_loops(18, sup1=0.008, sup2=0.026)
    g_gbi, g_gbo = [], []
    for fp_g in t_gb_p:
        ang_g = math.radians(143.0 + 72.0 * fp_g)
        sg, cg = math.sin(ang_g), math.cos(ang_g)
        ci_g, co_g = [], []
        bev_p = min(1.0, min(fp_g, 1.0 - fp_g) / 0.026)

        if 0.245 <= fp_g <= 0.755:
            step_h = 0.106
        elif fp_g < 0.225 or fp_g > 0.775:
            step_h = 0.082
        else:
            u_st = (fp_g - 0.225) / 0.020 if fp_g < 0.5 else (0.775 - fp_g) / 0.020
            step_h = 0.082 + 0.024 * max(0.0, min(1.0, u_st))
        for fy_g in t_gb_y:
            yg = -3.768 + 0.636 * fy_g
            bev_y = min(1.0, min(fy_g, 1.0 - fy_g) / 0.026)
            r_ig = 0.496
            r_og = 0.512 + step_h * min(bev_p, bev_y)
            ci_g.append(bm_agb.verts.new((r_ig * sg, yg, r_ig * cg)))
            co_g.append(bm_agb.verts.new((r_og * sg, yg, r_og * cg)))
        g_gbi.append(ci_g)
        g_gbo.append(co_g)
    bm_build_beveled_quad_shell(bm_agb, g_gbi, g_gbo, bulge_scale=0.04, max_bevel=0.0020, inset_bevel=True)

    def _agb_step_r(ph_deg_v: float) -> float:
        fp_v = max(0.0, min(1.0, (ph_deg_v - 143.0) / 72.0))
        if 0.245 <= fp_v <= 0.755:
            sh_v = 0.106
        elif fp_v < 0.225 or fp_v > 0.775:
            sh_v = 0.082
        else:
            u_v = (fp_v - 0.225) / 0.020 if fp_v < 0.5 else (0.775 - fp_v) / 0.020
            sh_v = 0.082 + 0.024 * max(0.0, min(1.0, u_v))
        return 0.512 + sh_v

    def _add_agb_contour_rib(ya_r: float, yb_r: float, protrude: float = 0.0055):
        ty_r = subdivide_1d_with_support_loops(2, sup1=0.14, sup2=0.30)
        gi_r, go_r = [], []
        for fp_r in t_gb_p:
            ph_r = 143.0 + 72.0 * fp_r
            ar_r = math.radians(ph_r)
            sr_r, cr_r = math.sin(ar_r), math.cos(ar_r)
            ep_r = min(1.0, min(fp_r, 1.0 - fp_r) / 0.026)
            r_top_r = 0.512 + (_agb_step_r(ph_r) - 0.512 + protrude) * (0.08 + 0.92 * ep_r)
            ci_r, co_r = [], []
            for fy_r in ty_r:
                yv_r = ya_r + (yb_r - ya_r) * fy_r
                ey_r = min(1.0, min(fy_r, 1.0 - fy_r) / 0.22)
                ro_r = 0.5112 + (r_top_r - 0.5112) * ey_r
                ci_r.append(bm_agb.verts.new((0.510 * sr_r, yv_r, 0.510 * cr_r)))
                co_r.append(bm_agb.verts.new((ro_r * sr_r, yv_r, ro_r * cr_r)))
            gi_r.append(ci_r)
            go_r.append(co_r)
        bm_build_beveled_quad_shell(bm_agb, gi_r, go_r, bulge_scale=0.04, max_bevel=0.0020, inset_bevel=True)


    _add_agb_contour_rib(-3.461, -3.439, protrude=0.0075)
    for yg_cr in (-3.705, -3.620, -3.535, -3.365, -3.280, -3.195):
        _add_agb_contour_rib(yg_cr - 0.0055, yg_cr + 0.0055, protrude=0.0046)
    agb_f_bpts, agb_a_bpts = [], []
    for k_agb_b in range(9):
        ph_ab = 148.0 + k_agb_b * 7.6
        r_ab = _agb_step_r(ph_ab) + 0.0040
        p_af0 = _cp(ph_ab, -3.461, r_ab)
        p_af1 = _cp(ph_ab, -3.439, r_ab)
        agb_f_bpts.append(p_af0)
        agb_a_bpts.append(p_af1)
        bm_add_aerospace_bolt(bm_agb, p_af0, Vector((0, -1, 0)), head_r=0.0026, height=0.0024)
        bm_add_aerospace_bolt(bm_agb, p_af1, Vector((0, +1, 0)), head_r=0.0026, height=0.0024)
    for k_agb_b in range(0, 8, 2):
        bm_add_bolt_pair_lockwire(bm_agb, agb_f_bpts[k_agb_b], agb_f_bpts[k_agb_b + 1], Vector((0, -1, 0)), head_r=0.0026, strand_r=0.00028)
        bm_add_bolt_pair_lockwire(bm_agb, agb_a_bpts[k_agb_b], agb_a_bpts[k_agb_b + 1], Vector((0, +1, 0)), head_r=0.0026, strand_r=0.00028)
    for ph_ax_r in (152.0, 165.0, 179.0, 193.0, 206.0):
        _add_chamfered_cyl_band(bm_agb, ph_ax_r - 0.65, ph_ax_r + 0.65, -3.748, -3.152,
                                0.512, _agb_step_r(ph_ax_r) + 0.0044, np_seg=3, ny_seg=12, taper_p=0.22, taper_y=0.03)

    bm_add_segmented_tube(
        bm_agb, Vector((0.0, -3.670, -0.484)), Vector((0.0, -3.670, -0.636)),
        [(0.0, 0.052), (0.35, 0.044), (0.82, 0.048), (1.0, 0.038)], segs=32, cap_ends=True,
    )
    for ky_gr in range(8):
        yg_r = -3.720 + ky_gr * 0.078
        for ph_sb in (145.0, 213.0):
            bm_add_aerospace_bolt(bm_agb, _cp(ph_sb, yg_r, 0.556), _cn(ph_sb), head_r=0.0030, height=0.0028)

    for ph_gcv, y_gcv, r_top_g in [
        (148.0, -3.600, 0.608), (148.0, -3.390, 0.608), (154.0, -3.220, 0.606),
        (178.0, -3.510, 0.638), (178.0, -3.310, 0.638),
        (204.0, -3.580, 0.612), (204.0, -3.360, 0.612),
    ]:
        b_gcv = bmesh.new()
        bm_add_segmented_tube(
            b_gcv, Vector((0.0, y_gcv, r_top_g - 0.032)), Vector((0.0, y_gcv, r_top_g)),
            [(0.0, 0.036), (0.55, 0.036), (0.70, 0.025), (1.0, 0.020)], segs=24, cap_ends=True,
        )
        bm_add_bevel_locked_hex(b_gcv, Vector((0.0, y_gcv, r_top_g - 0.002)), Vector((0.0, y_gcv, r_top_g + 0.007)),
                                [(0.0, 0.0085), (0.20, 0.010), (0.80, 0.010), (1.0, 0.008)], corner_bevel=0.14, cap_ends=True)
        gcv_bpts = []
        for ibg in range(6):
            abg = 2.0 * math.pi * ibg / 6.0
            p_gcvb = Vector((0.029 * math.cos(abg), y_gcv + 0.029 * math.sin(abg), r_top_g - 0.014))
            gcv_bpts.append(p_gcvb)
            bm_add_aerospace_bolt(b_gcv, p_gcvb, Vector((0, 0, 1)), head_r=0.0022, height=0.0020)
        for ibg in range(0, 6, 2):
            bm_add_bolt_pair_lockwire(b_gcv, gcv_bpts[ibg], gcv_bpts[ibg + 1], Vector((0, 0, 1)), head_r=0.0022, strand_r=0.00026)
        bm_add_bolt_pair_lockwire(b_gcv, Vector((0.0, y_gcv, r_top_g + 0.004)), gcv_bpts[0], Vector((0, 0, 1)), head_r=0.0042, strand_r=0.00026)
        bm_merge(bm_agb, b_gcv, Matrix.Rotation(math.radians(ph_gcv), 4, 'Y'))


    for ph_pmp, y_pm0, y_pm1, r_pm, rad_pm, has_fins, pmp_lbl in [
        (196.0, -3.995, -3.745, 0.576, 0.044, False, ["SCAV PUMP", "LUBE #1"]),
        (166.0, -4.015, -3.745, 0.582, 0.046, True,  ["AUG PUMP",  "CENTRIF"]),
        (184.0, -3.165, -2.835, 0.588, 0.050, True,  ["JFS-190",   "STARTER"]),
        (160.0, -3.165, -2.865, 0.584, 0.048, True,  ["MFP-220",   "MAIN FUEL"]),
        (206.0, -3.165, -2.895, 0.578, 0.044, False, ["HYD PUMP",  "3000 PSI"]),
        (140.0, -3.720, -3.520, 0.562, 0.036, False, ["IDG-40KVA", "BACKUP"]),
    ]:
        b_pm = bmesh.new()
        is_fwd_pm = (y_pm0 < -3.5)
        len_pm = y_pm1 - y_pm0
        st_pm = [
            (0.000, rad_pm * 0.96), (0.012, rad_pm * 1.08), (0.085, rad_pm * 1.08),
            (0.092, rad_pm * 1.00), (0.520, rad_pm * 1.00), (0.530, rad_pm * 0.88),
        ]
        if has_fins:
            for kf_p in range(6):
                f0_p = 0.565 + kf_p * 0.052
                st_pm.extend([
                    (f0_p,         rad_pm * 0.84),
                    (f0_p + 0.005, rad_pm * 0.99),
                    (f0_p + 0.022, rad_pm * 0.99),
                    (f0_p + 0.027, rad_pm * 0.84),
                ])
        st_pm.extend([
            (0.895, rad_pm * 0.88), (0.903, rad_pm * 1.04),
            (0.988, rad_pm * 1.04), (1.000, rad_pm * 0.94),
        ])
        bm_add_segmented_tube(b_pm, Vector((0.0, y_pm0, r_pm)), Vector((0.0, y_pm1, r_pm)), st_pm, segs=36, cap_ends=True)


        ya_rib = y_pm0 + 0.14 * len_pm
        yb_rib = y_pm0 + 0.50 * len_pm
        for kr_pm in range(8):
            ang_rpm = kr_pm * (math.pi / 4.0) + 0.3927
            b_prib = bmesh.new()
            _bm_add_box_x_yz(b_prib, -0.0024, +0.0024, ya_rib, yb_rib, rad_pm * 0.94, rad_pm * 1.055, nx=1, ny=3, nz=1, chamfer=0.0005)
            bm_merge(b_pm, b_prib, Matrix.Translation((0.0, 0.0, r_pm)) @ Matrix.Rotation(ang_rpm, 4, 'Y'))

        y_qad = (y_pm1 - 0.026) if is_fwd_pm else (y_pm0 + 0.026)
        bm_add_segmented_tube(
            b_pm, Vector((0.0, y_qad - 0.007, r_pm)), Vector((0.0, y_qad + 0.007, r_pm)),
            [(0.0, rad_pm * 1.01), (0.25, rad_pm * 1.11), (0.75, rad_pm * 1.11), (1.0, rad_pm * 1.01)], segs=36, cap_ends=True,
        )
        sgn_px = -1.0 if (ph_pmp <= 180.0) else +1.0
        x_qtr = sgn_px * (rad_pm * 1.12)
        bm_add_segmented_tube(
            b_pm, Vector((x_qtr, y_qad, r_pm - 0.016)), Vector((x_qtr, y_qad, r_pm + 0.016)),
            [(0.0, 0.0044), (0.28, 0.0060), (0.42, 0.0032), (0.58, 0.0032), (0.72, 0.0060), (1.0, 0.0044)], segs=16, cap_ends=True,
        )
        bm_add_helical_thread_cylinder(
            b_pm, Vector((x_qtr, y_qad, r_pm + 0.015)), Vector((x_qtr, y_qad, r_pm + 0.029)),
            r_core=0.0024, thread_h=0.00055, n_turns=6, segs_turn=14, cap_core=True,
        )
        bm_add_bevel_locked_hex(
            b_pm, Vector((x_qtr, y_qad, r_pm + 0.016)), Vector((x_qtr, y_qad, r_pm + 0.023)),
            [(0.0, 0.0046), (0.25, 0.0056), (0.80, 0.0056), (1.0, 0.0042)], corner_bevel=0.14, cap_ends=True,
        )


        y_cap_end = y_pm0 if is_fwd_pm else y_pm1
        s_cap_dir = -1.0 if is_fwd_pm else +1.0
        bm_add_bevel_locked_hex(b_pm, Vector((0.0, y_cap_end, r_pm)), Vector((0.0, y_cap_end + s_cap_dir * 0.011, r_pm)),
                                [(0.0, 0.0085), (0.30, 0.0108), (0.80, 0.0108), (1.0, 0.0080)], corner_bevel=0.14, cap_ends=True)
        ec_pm_pts = []
        for ike in range(8):
            ake = 2.0 * math.pi * ike / 8.0
            p_ecb = Vector((rad_pm * 0.52 * math.cos(ake), y_cap_end + s_cap_dir * 0.001, r_pm + rad_pm * 0.52 * math.sin(ake)))
            ec_pm_pts.append(p_ecb)
            bm_add_aerospace_bolt(b_pm, p_ecb, Vector((0, s_cap_dir, 0)), head_r=0.0020, height=0.0019)
        for ike in range(0, 8, 2):
            bm_add_bolt_pair_lockwire(b_pm, ec_pm_pts[ike], ec_pm_pts[ike + 1], Vector((0, s_cap_dir, 0)), head_r=0.0020, strand_r=0.00025)

        pm_bpts = []
        for ibm in range(10):
            abm = 2.0 * math.pi * ibm / 10.0
            y_fl = (y_pm1 - 0.010) if is_fwd_pm else (y_pm0 + 0.010)
            n_fl = Vector((0, -1 if is_fwd_pm else 1, 0))
            p_b = Vector((rad_pm * 0.92 * math.cos(abm), y_fl, r_pm + rad_pm * 0.92 * math.sin(abm)))
            pm_bpts.append((p_b, n_fl))
            bm_add_aerospace_bolt(b_pm, p_b, n_fl, head_r=0.0026, height=0.0024)
        for ibm in range(0, 10, 2):
            bm_add_bolt_pair_lockwire(b_pm, pm_bpts[ibm][0], pm_bpts[ibm + 1][0], pm_bpts[ibm][1], head_r=0.0026, strand_r=0.00030)


        ym_pl = y_pm0 + 0.32 * len_pm
        x_pl_in, x_pl_out = min(sgn_px * (rad_pm - 0.001), sgn_px * (rad_pm + 0.0022)), max(sgn_px * (rad_pm - 0.001), sgn_px * (rad_pm + 0.0022))
        _bm_add_box_x_yz(b_pm, x_pl_in, x_pl_out, ym_pl - 0.032, ym_pl + 0.032, r_pm - 0.013, r_pm + 0.013, nx=1, ny=2, nz=2, chamfer=0.00035)
        bm_add_3d_plate_text(
            b_pm, pmp_lbl,
            Vector((sgn_px * (rad_pm + 0.0022), ym_pl, r_pm)),
            Vector((0.0, -sgn_px, 0.0)), Vector((0.0, 0.0, -1.0)), Vector((sgn_px, 0.0, 0.0)),
            width=0.054, height=0.020, relief=0.00048, with_border=True, with_dividers=True,
        )
        for sy_pmr in (ym_pl - 0.027, ym_pl + 0.027):
            for sz_pmr in (r_pm - 0.0095, r_pm + 0.0095):
                bm_add_dome_rivet(b_pm, Vector((sgn_px * (rad_pm + 0.0022), sy_pmr, sz_pmr)), Vector((sgn_px, 0, 0)), radius=0.0010)
        y_cn_pm = y_pm0 + (0.76 if not is_fwd_pm else 0.24) * len_pm
        bm_add_mil38999_connector(
            b_pm, Vector((sgn_px * (rad_pm * 0.84), y_cn_pm, r_pm - rad_pm * 0.36)),
            Vector((sgn_px * 0.78, 0.0, -0.62)).normalized(),
            shell_r=0.0080, with_flange=False,
            cable_pts=[
                Vector((sgn_px * (rad_pm + 0.020), y_cn_pm - s_cap_dir * 0.026, r_pm - rad_pm * 0.72)),
                Vector((sgn_px * (rad_pm * 0.75),  y_cn_pm - s_cap_dir * 0.052, 0.522)),
            ],
            cable_braided=True, n_zipties=1,
        )
        bm_merge(
            bm_agb, b_pm, Matrix.Rotation(math.radians(ph_pmp), 4, 'Y'),
            default_mat_id=(MAT_MACHINED_ALU if ("JFS" in pmp_lbl[0] or "IDG" in pmp_lbl[0]) else MAT_AVIONICS_BOX if "HYD" in pmp_lbl[0] else MAT_PUMPS_OILTANK),
        )


    b_fcoc = bmesh.new()
    bm_add_segmented_tube(
        b_fcoc, Vector((0.0, -3.600, 0.550)), Vector((0.0, -3.210, 0.550)),
        [
            (0.000, 0.034), (0.010, 0.041), (0.072, 0.041), (0.078, 0.038),
            (0.922, 0.038), (0.928, 0.041), (0.990, 0.041), (1.000, 0.034),
        ],
        segs=32, cap_ends=True,
    )
    bm_add_segmented_tube(
        b_fcoc, Vector((0.0, -3.665, 0.616)), Vector((0.0, -3.145, 0.616)),
        [
            (0.000, 0.039), (0.009, 0.047), (0.056, 0.047), (0.062, 0.044),
            (0.938, 0.044), (0.944, 0.047), (0.991, 0.047), (1.000, 0.039),
        ],
        segs=36, cap_ends=True,
    )

    for z_fc_c, r_fc_c, ya_fc, yb_fc in ((0.550, 0.0395, -3.600, -3.210), (0.616, 0.0455, -3.665, -3.145)):
        for y_vb_f in (ya_fc + 0.078, yb_fc - 0.078):
            bm_add_segmented_tube(b_fcoc, Vector((0.0, y_vb_f - 0.007, z_fc_c)), Vector((0.0, y_vb_f + 0.007, z_fc_c)),
                                  [(0.0, r_fc_c), (0.25, r_fc_c + 0.0035), (0.75, r_fc_c + 0.0035), (1.0, r_fc_c)], segs=32, cap_ends=True)
            bm_add_chamfered_cyl_x(b_fcoc, y_vb_f, z_fc_c + r_fc_c + 0.004, -0.020, +0.018, radius=0.0036, segs=14)
            bm_add_helical_thread_cylinder(
                b_fcoc, Vector((+0.016, y_vb_f, z_fc_c + r_fc_c + 0.004)), Vector((+0.033, y_vb_f, z_fc_c + r_fc_c + 0.004)),
                r_core=0.0024, thread_h=0.00055, n_turns=6, segs_turn=14, cap_core=True,
            )
            bm_add_bevel_locked_hex(b_fcoc, Vector((+0.020, y_vb_f, z_fc_c + r_fc_c + 0.004)), Vector((+0.028, y_vb_f, z_fc_c + r_fc_c + 0.004)),
                                    [(0.0, 0.0045), (0.30, 0.0058), (0.80, 0.0058), (1.0, 0.0042)], corner_bevel=0.14, cap_ends=True)
        for y_ec_f, s_ec_f in ((ya_fc, -1.0), (yb_fc, +1.0)):
            bm_add_bevel_locked_hex(b_fcoc, Vector((0.0, y_ec_f, z_fc_c)), Vector((0.0, y_ec_f + s_ec_f * 0.010, z_fc_c)),
                                    [(0.0, 0.0085), (0.25, 0.0105), (0.80, 0.0105), (1.0, 0.0080)], corner_bevel=0.14, cap_ends=True)
            ec_pts = []
            for kb_fc in range(12):
                afc = 2.0 * math.pi * kb_fc / 12.0
                p_fc_b = Vector(((r_fc_c - 0.003) * math.cos(afc), y_ec_f - s_ec_f * 0.026, z_fc_c + (r_fc_c - 0.003) * math.sin(afc)))
                ec_pts.append(p_fc_b)
                bm_add_aerospace_bolt(b_fcoc, p_fc_b, Vector((0, s_ec_f, 0)), head_r=0.0021, height=0.0020)
            for kb_fc in range(0, 12, 2):
                bm_add_bolt_pair_lockwire(b_fcoc, ec_pts[kb_fc], ec_pts[kb_fc + 1], Vector((0, s_ec_f, 0)), head_r=0.0021, strand_r=0.00028)

    bm_add_chamfered_cyl_x(b_fcoc, -3.405, 0.616, +0.042, +0.062, radius=0.011, segs=20)
    bm_add_chamfered_cyl_x(b_fcoc, -3.405, 0.616, +0.060, +0.068, radius=0.0045, segs=14)
    _bm_add_box_x_yz(b_fcoc, +0.0442, +0.0462, -3.515, -3.435, 0.602, 0.630, nx=1, ny=2, nz=2, chamfer=0.0004)
    bm_add_3d_plate_text(
        b_fcoc, ["FCOC-220", "FUEL/OIL"],
        Vector((+0.0462, -3.475, 0.616)), Vector((0.0, -1.0, 0.0)), Vector((0.0, 0.0, -1.0)), Vector((+1.0, 0.0, 0.0)),
        width=0.070, height=0.024, relief=0.00055, with_border=True, with_dividers=True,
    )
    for sy_fcr in (-3.508, -3.442):
        for sz_fcr in (0.606, 0.626):
            bm_add_dome_rivet(b_fcoc, Vector((+0.0462, sy_fcr, sz_fcr)), Vector((+1, 0, 0)), radius=0.0011)
    for y_fc_b in (-3.535, -3.275):
        bm_add_clevis_plate(b_fcoc, (y_fc_b - 0.012, 0.494), (y_fc_b + 0.012, 0.600), 0.012, 0.030, -0.016, +0.016, arc_segs=15)
        for sx_fc in (-0.034, +0.034):
            bm_add_clevis_plate(b_fcoc, (y_fc_b, 0.496), (y_fc_b, 0.600), 0.008, 0.009, sx_fc - 0.004, sx_fc + 0.004, arc_segs=9)
    for y_fcp in (-3.595, -3.215):
        _bm_add_box_x_yz(b_fcoc, -0.024, +0.024, y_fcp - 0.022, y_fcp + 0.022, 0.648, 0.660, nx=2, ny=2, nz=1)
        bm_add_segmented_tube(b_fcoc, Vector((0, y_fcp, 0.658)), Vector((0, y_fcp, 0.674)), [(0.0, 0.014), (1.0, 0.012)], segs=20, cap_ends=True)
        fcp_bpts = []
        for sx_fb in (-0.016, +0.016):
            for sy_fb in (-0.016, +0.016):
                pfcp = Vector((sx_fb, y_fcp + sy_fb, 0.660))
                fcp_bpts.append(pfcp)
                bm_add_aerospace_bolt(b_fcoc, pfcp, Vector((0, 0, 1)), head_r=0.0022, height=0.0020)
        bm_add_bolt_pair_lockwire(b_fcoc, fcp_bpts[0], fcp_bpts[1], Vector((0, 0, 1)), head_r=0.0022, strand_r=0.00026)
        bm_add_bolt_pair_lockwire(b_fcoc, fcp_bpts[2], fcp_bpts[3], Vector((0, 0, 1)), head_r=0.0022, strand_r=0.00026)
    bm_merge(bm_agb, b_fcoc, Matrix.Rotation(math.radians(218.0), 4, 'Y'), default_mat_id=MAT_MACHINED_ALU)


    b_lsump = bmesh.new()
    _bm_add_box_x_yz(b_lsump, -0.042, +0.042, -3.525, -3.155, 0.526, 0.576, nx=3, ny=5, nz=2, chamfer=0.0014)
    _bm_add_box_x_yz(b_lsump, -0.045, +0.045, -3.529, -3.151, 0.573, 0.580, nx=3, ny=5, nz=1, chamfer=0.0010)
    for kf_ls in range(9):
        yf_ls = -3.495 + kf_ls * 0.039
        if -3.385 <= yf_ls <= -3.295:
            _bm_add_box_x_yz(b_lsump, -0.036, +0.036, yf_ls - 0.0022, yf_ls + 0.0022, 0.579, 0.585, nx=2, ny=1, nz=1, chamfer=0.0004)
        else:
            _bm_add_box_x_yz(b_lsump, -0.044, +0.044, yf_ls - 0.0022, yf_ls + 0.0022, 0.530, 0.585, nx=2, ny=1, nz=2, chamfer=0.0004)
    ls_bpts = []
    for sy_lsb in (-3.518, -3.398, -3.282, -3.162):
        for sx_lsb in (-0.038, +0.038):
            p_lsb = Vector((sx_lsb, sy_lsb, 0.580))
            ls_bpts.append(p_lsb)
            bm_add_aerospace_bolt(b_lsump, p_lsb, Vector((0, 0, 1)), head_r=0.0021, height=0.0019)
    for k_lsb in range(0, len(ls_bpts), 2):
        bm_add_bolt_pair_lockwire(b_lsump, ls_bpts[k_lsb], ls_bpts[k_lsb + 1], Vector((0, 0, 1)), head_r=0.0021, strand_r=0.00026)
    _bm_add_box_x_yz(b_lsump, -0.0442, -0.0418, -3.382, -3.298, 0.536, 0.568, nx=1, ny=2, nz=2, chamfer=0.0004)
    bm_add_3d_plate_text(
        b_lsump, ["AUX CTRL", "VIB-220"],
        Vector((-0.0442, -3.340, 0.552)),
        Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, -1.0)), Vector((-1.0, 0.0, 0.0)),
        width=0.072, height=0.024, relief=0.00050, with_border=True, with_dividers=True,
    )
    for sy_lsr in (-3.376, -3.304):
        for sz_lsr in (0.540, 0.564):
            bm_add_dome_rivet(b_lsump, Vector((-0.0442, sy_lsr, sz_lsr)), Vector((-1, 0, 0)), radius=0.0010)
    for sx_lcn, y_lcn_end in ((-0.018, -3.575), (+0.018, -3.568)):
        bm_add_mil38999_connector(
            b_lsump, Vector((sx_lcn, -3.525, 0.552)), Vector((0.0, -1.0, 0.0)),
            shell_r=0.0086, with_flange=True,
            cable_pts=[Vector((sx_lcn - 0.010, y_lcn_end, 0.542)), Vector((sx_lcn - 0.022, y_lcn_end - 0.036, 0.528))],
            cable_braided=True, n_zipties=1,
        )
    for y_spg in (-3.485, -3.195):
        bm_add_clevis_plate(b_lsump, (y_spg - 0.012, 0.492), (y_spg + 0.012, 0.526), 0.009, 0.011, -0.034, +0.034, arc_segs=11)
        for sx_sp_d in (-0.024, +0.024):
            bm_add_segmented_tube(
                b_lsump, Vector((sx_sp_d, y_spg, 0.502)), Vector((sx_sp_d, y_spg, 0.532)),
                [(0.0, 0.0085), (0.25, 0.0052), (0.75, 0.0052), (1.0, 0.0085)], segs=16, cap_ends=True,
            )
            coil_pts = [
                Vector((
                    sx_sp_d + 0.0092 * math.cos(2.0 * math.pi * 4.5 * (ks / 44.0)),
                    y_spg + 0.0092 * math.sin(2.0 * math.pi * 4.5 * (ks / 44.0)),
                    0.505 + 0.023 * (ks / 44.0),
                ))
                for ks in range(45)
            ]
            f0_spg = len(b_lsump.faces)
            bm_add_spline_tube(b_lsump, coil_pts, radius=0.0016, n_samples=56, segs=10, cap_ends=True)
            bm_fill_unset_from(b_lsump, f0_spg, MAT_YELLOW_HARNESS)
    bm_merge(bm_agb, b_lsump, Matrix.Rotation(math.radians(136.0), 4, 'Y'), default_mat_id=MAT_AVIONICS_BOX)


    agb_fluid_tubes = [
        ([(234.0, -3.825, 0.514), (234.0, -3.710, 0.556), (218.0, -3.680, 0.658), (218.0, -3.595, 0.664)], 0.0108, 0.052),
        ([(218.0, -3.215, 0.664), (218.0, -3.130, 0.625), (198.0, -3.130, 0.596), (198.0, -3.220, 0.588)], 0.0108, 0.048),
        ([(218.0, -3.665, 0.616), (218.0, -3.760, 0.602), (196.0, -3.760, 0.584)], 0.0088, 0.044),
        ([(208.0, -4.165, 0.552), (208.0, -4.060, 0.588), (188.0, -4.060, 0.596)], 0.0080, 0.040),
        ([(156.0, -4.165, 0.584), (142.0, -4.165, 0.556), (142.0, -3.995, 0.548)], 0.0076, 0.040),
        ([(160.0, -2.920, 0.586), (144.0, -2.920, 0.554), (144.0, -2.998, 0.526)], 0.0088, 0.042),
    ]
    for agb_wps, tr_g, br_g in agb_fluid_tubes:
        _add_fwd_pipe(bm_agb, agb_wps, tube_r=tr_g, clamp_indices=(1,), r_case_ref=0.495, segs=20, bend_r=br_g)

    for base_deg in (140.0, 212.0):
        for k_cb, d_cb in enumerate((-2.4, 0.0, +2.4)):
            dc = base_deg + d_cb
            cb_wps = [
                (dc, -4.160, 0.538), (dc, -3.856, 0.536),
                (dc, -3.520, 0.568), (dc, -3.220, 0.568), (dc, -2.980, 0.548),
            ]
            _add_fwd_pipe(bm_agb, cb_wps, tube_r=0.0056 + 0.0010 * (k_cb == 1), clamp_indices=(1, 2, 3),
                          r_case_ref=0.496, segs=16, corrugations=44, corr_amp=0.08)

    create_mesh_obj("Fwd_07_Underslung_AGB_FCOC_And_PumpCluster", bm_agb, coll_fwd, clay)


    bm_comb = bmesh.new()

    def _r_comb_surf(yv: float) -> float:
        if yv < -2.756:
            return 0.494 + 0.006 * max(0.0, (yv - (-3.051)) / 0.295)
        return 0.502 + 0.040 * max(0.0, min(1.0, (yv - (-2.756)) / 0.511))

    prof_comb_case = [
        (-3.051, 0.456), (-3.051, 0.518), (-3.041, 0.518), (-3.033, 0.494),
        (-2.770, 0.500), (-2.764, 0.525), (-2.748, 0.525), (-2.742, 0.502),
        (-2.524, 0.519), (-2.518, 0.540), (-2.504, 0.540), (-2.498, 0.521),
        (-2.265, 0.542), (-2.257, 0.566), (-2.245, 0.566),
        (-2.245, 0.498), (-2.495, 0.478), (-2.760, 0.458),
    ]
    bm_add_revolved_profile(bm_comb, prof_comb_case, segs=240, close_loop=True, bevel_r=0.00045)


    for n_mfl, y_mfl, r_mfl, nut_d in [(76, -2.764, 0.516, 0.015), (80, -2.518, 0.531, 0.015), (88, -2.257, 0.554, 0.018)]:
        mfl_pts = []
        for icm in range(n_mfl):
            acm = 2.0 * math.pi * icm / float(n_mfl)
            pcm = Vector((r_mfl * math.sin(acm), y_mfl, r_mfl * math.cos(acm)))
            mfl_pts.append(pcm)
            bm_add_aerospace_bolt(bm_comb, pcm, Vector((0.0, -1.0, 0.0)), head_r=0.0031, height=0.0029, with_12pt_nut_depth=nut_d)
        for icm in range(0, n_mfl, 2):
            bm_add_bolt_pair_lockwire(bm_comb, mfl_pts[icm], mfl_pts[icm + 1], Vector((0.0, -1.0, 0.0)), head_r=0.0031, strand_r=0.00035)


    for deg_bs_c, y_bs_c in [
        (44.0, -2.66), (58.0, -2.43), (96.0, -2.62), (118.0, -2.44), (146.0, -2.62),
        (214.0, -2.62), (242.0, -2.44), (264.0, -2.62), (302.0, -2.43), (316.0, -2.66),
    ]:
        rc_b = _r_comb_surf(y_bs_c)
        b_bs_c = bmesh.new()
        bm_add_segmented_tube(b_bs_c, Vector((0, y_bs_c, rc_b - 0.004)), Vector((0, y_bs_c, rc_b + 0.018)),
                              [(0.0, 0.019), (0.45, 0.019), (0.55, 0.013), (1.0, 0.0115)], segs=22, cap_ends=True)
        p_cp0 = Vector((0.0, y_bs_c, rc_b + 0.018))
        p_cp1 = Vector((0.013, y_bs_c + 0.009, rc_b + 0.009))
        bm_add_aerospace_bolt(b_bs_c, p_cp0, Vector((0, 0, 1)), head_r=0.0056, height=0.0052)
        bm_add_aerospace_bolt(b_bs_c, p_cp1, Vector((0, 0, 1)), head_r=0.0024, height=0.0022)
        bm_add_bolt_pair_lockwire(b_bs_c, p_cp0, p_cp1, Vector((0, 0, 1)), head_r=0.0036, strand_r=0.00036)
        bm_merge(bm_comb, b_bs_c, Matrix.Rotation(math.radians(deg_bs_c), 4, 'Y'))

    for deg_cl_c, y_cl_c in [
        (32.0, -2.62), (76.0, -2.86), (82.0, -2.46), (118.0, -2.64),
        (242.0, -2.64), (278.0, -2.46), (284.0, -2.86), (328.0, -2.62),
    ]:
        rc_cl = _r_comb_surf(y_cl_c)
        b_clp_c = bmesh.new()
        bm_add_clevis_plate(b_clp_c, (y_cl_c - 0.018, rc_cl), (y_cl_c + 0.018, rc_cl + 0.022), 0.005, 0.007, -0.004, +0.004, arc_segs=11)
        bm_add_chamfered_cyl_x(b_clp_c, y_cl_c + 0.014, rc_cl + 0.020, -0.008, +0.008, radius=0.0048, segs=16, hollow_r=0.0024)
        bm_add_aerospace_bolt(b_clp_c, Vector((+0.005, y_cl_c - 0.012, rc_cl + 0.004)), Vector((1, 0, 0)), head_r=0.0020, height=0.0018)
        bm_merge(bm_comb, b_clp_c, Matrix.Rotation(math.radians(deg_cl_c), 4, 'Y'))


    for y_fm, r_fm in [(-2.976, 0.522), (-2.998, 0.522)]:
        prof_fm = [
            (y_fm + 0.0050 * math.cos(2 * math.pi * s / 20), r_fm + 0.0050 * math.sin(2 * math.pi * s / 20))
            for s in range(20)
        ]
        bm_add_revolved_profile(bm_comb, prof_fm, segs=208, close_loop=True)

    b_fnz = bmesh.new()
    bm_add_clevis_plate(b_fnz, (-2.912, 0.495), (-2.878, 0.496), 0.018, 0.018, -0.004, +0.008, arc_segs=13)
    bm_add_segmented_tube(
        b_fnz, Vector((0.0, -2.895, 0.496)), Vector((0.0, -2.895, 0.530)),
        [(0.0, 0.020), (0.30, 0.020), (0.38, 0.0125), (0.78, 0.0125), (1.0, 0.0095)], segs=24, cap_ends=True,
    )
    bm_add_bevel_locked_hex(b_fnz, Vector((0, -2.895, 0.515)), Vector((0, -2.895, 0.525)),
                            [(0.0, 0.013), (0.20, 0.015), (0.80, 0.015), (1.0, 0.013)], corner_bevel=0.14, cap_ends=True)
    fn_bpts = [Vector((-0.015, -2.895, 0.504)), Vector((+0.015, -2.895, 0.504)), Vector((0.0, -2.876, 0.504))]
    for pfb in fn_bpts:
        bm_add_aerospace_bolt(b_fnz, pfb, Vector((0, 0, 1)), head_r=0.0023, height=0.0022)
    bm_add_bolt_pair_lockwire(b_fnz, fn_bpts[0], fn_bpts[2], Vector((0, 0, 1)), head_r=0.0023, strand_r=0.00030)
    bm_add_bolt_pair_lockwire(b_fnz, fn_bpts[1], fn_bpts[2], Vector((0, 0, 1)), head_r=0.0023, strand_r=0.00030)
    for x_pg, y_man_t, r_pg in ((+0.009, -2.976, 0.0032), (-0.009, -2.998, 0.0028)):
        pg_pts = [
            Vector((x_pg * 0.7, -2.895, 0.526)),
            Vector((x_pg * 1.9, -2.918, 0.536)),
            Vector((x_pg * 2.1, -2.952, 0.533)),
            Vector((x_pg * 1.2, y_man_t, 0.522)),
        ]
        pg_fr = bm_add_spline_tube(b_fnz, pg_pts, radius=r_pg, n_samples=24, segs=14, cap_ends=True)
        bm_add_bnut_flare_fitting(b_fnz, pg_fr[2][0], pg_fr[2][1], tube_r=r_pg)
        bm_add_bnut_flare_fitting(b_fnz, pg_fr[-2][0], pg_fr[-2][1], tube_r=r_pg)

        bm_add_bevel_locked_hex(
            b_fnz, Vector((x_pg * 1.2 - 0.007, y_man_t, 0.522)), Vector((x_pg * 1.2 + 0.007, y_man_t, 0.522)),
            [(0.0, 0.0068), (0.22, 0.0082), (0.78, 0.0082), (1.0, 0.0068)], corner_bevel=0.14, cap_ends=True,
        )
    bm_add_clevis_plate(b_fnz, (-3.004, 0.494), (-2.970, 0.524), 0.005, 0.006, +0.024, +0.030, arc_segs=9)
    for y_mcl in (-2.998, -2.976):
        bm_add_chamfered_cyl_x(b_fnz, y_mcl, 0.522, +0.022, +0.032, radius=0.0068, segs=16, hollow_r=0.0050)
    for ifn in range(16):
        bm_merge(bm_comb, b_fnz.copy(), Matrix.Rotation(2.0 * math.pi * ifn / 16.0, 4, 'Y'))
    b_fnz.free()


    for ph_cdi, up_sgn in ((132.0, +1.0), (228.0, -1.0)):
        b_cdi = bmesh.new()
        for yc_sh in (-2.936, -2.744):
            bm_add_clevis_plate(b_cdi, (yc_sh - 0.010, 0.498), (yc_sh + 0.010, 0.520), 0.007, 0.009, -0.060, +0.060, arc_segs=11)
            for sx_cm in (-0.048, +0.048):
                bm_add_segmented_tube(
                    b_cdi, Vector((sx_cm, yc_sh, 0.506)), Vector((sx_cm, yc_sh, 0.524)),
                    [(0.0, 0.011), (0.30, 0.013), (0.50, 0.0095), (0.70, 0.013), (1.0, 0.010)], segs=18, cap_ends=True,
                )
                bm_add_aerospace_bolt(b_cdi, Vector((sx_cm, yc_sh, 0.525)), Vector((0, 0, 1)), head_r=0.0024, height=0.0022)
        _bm_add_box_x_yz(b_cdi, -0.054, +0.054, -2.945, -2.735, 0.522, 0.566, nx=4, ny=5, nz=2, chamfer=0.0014)
        _bm_add_box_x_yz(b_cdi, -0.058, +0.058, -2.949, -2.731, 0.564, 0.571, nx=4, ny=5, nz=1, chamfer=0.0012)

        for kf_c in range(11):
            yf_c = -2.930 + kf_c * 0.018
            if -2.870 <= yf_c <= -2.810:
                for x_fl, x_fr in ((-0.056, -0.031), (+0.031, +0.056)):
                    _bm_add_box_x_yz(b_cdi, x_fl, x_fr, yf_c - 0.0020, yf_c + 0.0020, 0.570, 0.5765, nx=1, ny=1, nz=1, chamfer=0.0005)
                _bm_add_box_x_yz(b_cdi, -0.056, +0.056, yf_c - 0.0020, yf_c + 0.0020, 0.525, 0.538, nx=2, ny=1, nz=1, chamfer=0.0005)
            else:
                _bm_add_box_x_yz(b_cdi, -0.056, +0.056, yf_c - 0.0020, yf_c + 0.0020, 0.525, 0.5765, nx=2, ny=1, nz=2, chamfer=0.0005)
        _bm_add_box_x_yz(b_cdi, -0.028, +0.028, -2.868, -2.812, 0.570, 0.5745, nx=2, ny=2, nz=1, chamfer=0.0005)
        bm_add_3d_plate_text(
            b_cdi, ["DANGER", "HIGH VOLT", "CDI-220"],
            Vector((0.0, -2.840, 0.5745)),
            Vector((0.0, up_sgn * 1.0, 0.0)), Vector((-up_sgn * 1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)),
            width=0.048, height=0.046, relief=0.00065, with_border=True, with_dividers=True,
        )
        for sx_cp in (-0.023, +0.023):
            for sy_cp in (-2.862, -2.818):
                bm_add_dome_rivet(b_cdi, Vector((sx_cp, sy_cp, 0.5745)), Vector((0, 0, 1)), radius=0.0012)

        cdi_bpts = []
        for sy_cb in (-2.941, -2.874, -2.806, -2.739):
            for sx_cb in (-0.048, +0.048):
                pcb_i = Vector((sx_cb, sy_cb, 0.571))
                cdi_bpts.append(pcb_i)
                bm_add_aerospace_bolt(b_cdi, pcb_i, Vector((0, 0, 1)), head_r=0.0022, height=0.0020)
        for kb_ci in range(0, len(cdi_bpts), 2):
            bm_add_bolt_pair_lockwire(b_cdi, cdi_bpts[kb_ci], cdi_bpts[kb_ci + 1], Vector((0, 0, 1)), head_r=0.0022, strand_r=0.00028)
        bm_add_mil38999_connector(
            b_cdi, Vector((0.0, -2.945, 0.546)), Vector((0.0, -1.0, 0.0)),
            shell_r=0.0108, with_flange=True,
            cable_pts=[
                Vector((up_sgn * 0.012, -3.005, 0.538)),
                Vector((up_sgn * 0.026, -3.045, 0.528)),
            ],
            cable_braided=True, n_zipties=2,
        )

        x_top_cdi = -up_sgn * 0.054
        _bm_add_box_x_yz(b_cdi, x_top_cdi - up_sgn * 0.006, x_top_cdi + up_sgn * 0.002, -2.905, -2.775, 0.533, 0.563, nx=1, ny=3, nz=2, chamfer=0.0008)
        for sy_ob in (-2.897, -2.840, -2.783):
            for sz_ob in (0.537, 0.559):
                bm_add_aerospace_bolt(b_cdi, Vector((x_top_cdi - up_sgn * 0.006, sy_ob, sz_ob)), Vector((-up_sgn, 0, 0)), head_r=0.0019, height=0.0018)
        bm_merge(bm_comb, b_cdi, Matrix.Rotation(math.radians(ph_cdi), 4, 'Y'), default_mat_id=MAT_AVIONICS_BOX)

        for k_ig, (dy_ig, d_phi_plg, y_plg) in enumerate([(-2.875, 25.0, -2.715), (-2.805, 33.0, -2.665)]):
            phi_exit = ph_cdi - up_sgn * 5.9
            phi_mid0 = ph_cdi - up_sgn * 8.5
            phi_mid1 = ph_cdi - up_sgn * 12.2
            phi_plg  = ph_cdi - up_sgn * d_phi_plg

            p_ex0 = _cp(phi_exit, dy_ig, 0.546)
            p_exm = _cp(phi_mid0, dy_ig + 0.004, 0.535)
            p_ex1 = _cp(phi_mid1, dy_ig + 0.012, 0.527)
            bm_add_segmented_tube(bm_comb, p_ex0, p_exm,
                                  [(0.0, 0.0132), (0.30, 0.0132), (0.45, 0.0105), (1.0, 0.0095)], segs=20, cap_ends=True)
            bm_add_bevel_locked_hex(bm_comb, p_exm, p_ex1, [(0.0, 0.0092), (0.25, 0.0116), (0.80, 0.0116), (1.0, 0.0082)], corner_bevel=0.14, cap_ends=True)
            ig_wps = [
                (phi_mid1, dy_ig + 0.012, 0.527),
                (phi_plg,  dy_ig + 0.024, 0.526),
                (phi_plg,  y_plg, 0.526),
                (phi_plg,  y_plg, 0.506),
            ]
            _add_fwd_pipe(bm_comb, ig_wps, tube_r=0.0050, clamp_indices=(2,), r_case_fn=_r_comb_surf, segs=16,
                          corrugations=22, corr_amp=0.10, bend_r=0.036)
            rc_plg = _r_comb_surf(y_plg)
            b_ig_pad = bmesh.new()
            bm_add_clevis_plate(b_ig_pad, (y_plg - 0.018, rc_plg - 0.002), (y_plg + 0.018, rc_plg - 0.002),
                                0.015, 0.015, -0.002, +0.006, arc_segs=13)
            bm_add_segmented_tube(
                b_ig_pad, Vector((0.0, y_plg, rc_plg - 0.002)), Vector((0.0, y_plg, rc_plg + 0.026)),
                [(0.0, 0.0168), (0.32, 0.0168), (0.40, 0.0132), (0.55, 0.0148), (0.65, 0.0112), (0.85, 0.0112), (1.0, 0.0085)],
                segs=24, cap_ends=True,
            )
            bm_add_bevel_locked_hex(
                b_ig_pad, Vector((0.0, y_plg, rc_plg + 0.016)), Vector((0.0, y_plg, rc_plg + 0.023)),
                [(0.0, 0.0102), (0.22, 0.0124), (0.78, 0.0124), (1.0, 0.0100)], corner_bevel=0.14, cap_ends=True,
            )
            ig_bpts = []
            for sx_igb in (-0.011, +0.011):
                for sy_igb in (y_plg - 0.015, y_plg + 0.015):
                    p_igb = Vector((sx_igb, sy_igb, rc_plg + 0.004))
                    ig_bpts.append(p_igb)
                    bm_add_aerospace_bolt(b_ig_pad, p_igb, Vector((0, 0, 1)), head_r=0.0020, height=0.0019)
            bm_add_bolt_pair_lockwire(b_ig_pad, ig_bpts[0], ig_bpts[1], Vector((0, 0, 1)), head_r=0.0020, strand_r=0.00028)
            bm_add_bolt_pair_lockwire(b_ig_pad, ig_bpts[2], ig_bpts[3], Vector((0, 0, 1)), head_r=0.0020, strand_r=0.00028)
            bm_merge(bm_comb, b_ig_pad, Matrix.Rotation(math.radians(phi_plg), 4, 'Y'), default_mat_id=MAT_MACHINED_ALU)


    ftit_pts = [_cp(k * 10.0, -2.355, 0.552) for k in range(37)]
    ftit_fr = bm_add_spline_tube(bm_comb, ftit_pts, radius=0.0054, n_samples=144, segs=16, cap_ends=True, corrugations=68, corr_amp=0.10)
    for kc_f in range(12):
        dc_f = kc_f * 30.0 + 15.0
        pc_f = _cp(dc_f, -2.355, 0.552)
        bf_f = min(ftit_fr, key=lambda fr: (fr[0] - pc_f).length_squared)
        bm_add_adel_loop_clamp(bm_comb, bf_f[0], bf_f[1], _cp(dc_f, -2.355, _r_comb_surf(-2.355)), Vector((0, 1, 0)), tube_r=0.0054)

    f0_flp = len(bm_comb.faces)
    fire_loop_pts = [_cp(k * 10.0, -2.288, 0.555) for k in range(37)]
    fire_fr = bm_add_spline_tube(bm_comb, fire_loop_pts, radius=0.0034, n_samples=144, segs=14, cap_ends=True)
    bm_fill_unset_from(bm_comb, f0_flp, MAT_THREAD)
    for kc_fl in range(12):
        dc_fl = kc_fl * 30.0 + 5.0
        pc_fl = _cp(dc_fl, -2.288, 0.555)
        bf_fl = min(fire_fr, key=lambda fr: (fr[0] - pc_fl).length_squared)
        bm_add_adel_loop_clamp(bm_comb, bf_fl[0], bf_fl[1], _cp(dc_fl, -2.288, _r_comb_surf(-2.288)), Vector((0, 1, 0)), tube_r=0.0034)


    for y_arp in (-2.692, -2.462):
        rc_arp = _r_comb_surf(y_arp)
        for k_arp in range(12):
            deg_arp = k_arp * 30.0 + (18.0 if y_arp < -2.55 else 6.0)
            if 152.0 <= deg_arp <= 208.0:
                continue
            b_arp = bmesh.new()
            bm_add_segmented_tube(b_arp, Vector((0.0, y_arp, rc_arp - 0.002)), Vector((0.0, y_arp, rc_arp + 0.009)),
                                  [(0.0, 0.0105), (0.55, 0.0105), (0.72, 0.0072), (1.0, 0.0068)], segs=18, cap_ends=True)
            bm_add_bevel_locked_hex(b_arp, Vector((0.0, y_arp, rc_arp + 0.008)), Vector((0.0, y_arp, rc_arp + 0.0125)),
                                    [(0.0, 0.0046), (0.30, 0.0056), (0.80, 0.0056), (1.0, 0.0044)], corner_bevel=0.14, cap_ends=True)
            bm_add_aerospace_bolt(b_arp, Vector((0.0082, y_arp + 0.0065, rc_arp + 0.0035)), Vector((0, 0, 1)), head_r=0.0017, height=0.0016)
            bm_add_bolt_pair_lockwire(b_arp, Vector((0.0, y_arp, rc_arp + 0.0105)), Vector((0.0082, y_arp + 0.0065, rc_arp + 0.0035)),
                                      Vector((0, 0, 1)), head_r=0.0028, strand_r=0.00024)
            bm_merge(bm_comb, b_arp, Matrix.Rotation(math.radians(deg_arp), 4, 'Y'), default_mat_id=MAT_MACHINED_ALU)

    for k_ft in range(10):
        deg_ft = 360.0 * (k_ft + 0.3) / 10.0
        b_ftp = bmesh.new()
        rc_ft = _r_comb_surf(-2.405)
        bm_add_clevis_plate(b_ftp, (-2.418, rc_ft - 0.001), (-2.392, rc_ft - 0.001), 0.011, 0.011, -0.003, +0.005, arc_segs=11)
        bm_add_mil38999_connector(
            b_ftp, Vector((0.0, -2.405, rc_ft + 0.004)), Vector((0.0, 0.0, 1.0)),
            shell_r=0.0068, with_flange=True,
            cable_pts=[Vector((0.010, -2.382, 0.556)), Vector((0.016, -2.355, 0.552))],
            cable_braided=False, n_zipties=1,
        )
        bm_merge(bm_comb, b_ftp, Matrix.Rotation(math.radians(deg_ft), 4, 'Y'), default_mat_id=MAT_AVIONICS_BOX)


    flank_bank_degs = [

        68.0, 73.0, 78.0, 99.0, 105.0,
        255.0, 261.0, 282.0, 287.0, 292.0,
    ]
    for idx_fb, deg_fb in enumerate(flank_bank_degs):
        tr_fb = 0.0044 if (idx_fb % 2 == 0) else 0.0038
        fb_wps = [
            (deg_fb, -3.065, 0.522),
            (deg_fb, -2.920, 0.518),
            (deg_fb, -2.756, 0.532),
            (deg_fb, -2.630, 0.528),
            (deg_fb, -2.512, 0.546),
            (deg_fb, -2.365, 0.548),
            (deg_fb, -2.238, 0.568),
        ]
        _add_fwd_pipe(bm_comb, fb_wps, tube_r=tr_fb, clamp_indices=(1, 3), union_indices=((3,) if idx_fb % 2 == 0 else (1,)),
                      r_case_fn=_r_comb_surf, segs=16)
    for p_ga, p_gb in ((67.0, 79.0), (98.0, 106.0), (254.0, 262.0), (281.0, 293.0)):
        _add_gang_clamp_bar(bm_comb, p_ga, p_gb, -2.860, _r_comb_surf(-2.860), 0.520)
        _add_gang_clamp_bar(bm_comb, p_ga, p_gb, -2.635, _r_comb_surf(-2.635), 0.530)
        _add_gang_clamp_bar(bm_comb, p_ga, p_gb, -2.415, _r_comb_surf(-2.415), 0.546)


    for sgn_wf, p_base_wf, p_low_wf in ((-1.0, 246.5, 220.0), (+1.0, 113.5, 140.0)):
        for k_wf in range(5):
            d_hi_w = p_base_wf - sgn_wf * (k_wf * 2.8)
            d_lo_w = p_low_wf  - sgn_wf * (k_wf * 2.4)
            y_turn1 = -2.715 + k_wf * 0.018
            y_turn2 = -2.415 + k_wf * 0.016
            tr_wf = 0.0042 if (k_wf % 2 == 0) else 0.0036
            wf_wps = [
                (d_hi_w, -3.062, 0.520),
                (d_hi_w, -2.756, 0.530),
                (d_hi_w, y_turn1, 0.532),
                (d_lo_w, y_turn2, 0.552),
                (d_lo_w, -2.242, 0.566),
            ]
            _add_fwd_pipe(bm_comb, wf_wps, tube_r=tr_wf, clamp_indices=(1, 3), union_indices=(2,),
                          r_case_fn=_r_comb_surf, segs=16, bend_r=0.046)

        pa_h = min(p_base_wf, p_base_wf - sgn_wf * 11.2) - 1.0
        pb_h = max(p_base_wf, p_base_wf - sgn_wf * 11.2) + 1.0
        _add_gang_clamp_bar(bm_comb, pa_h, pb_h, -2.885, _r_comb_surf(-2.885), 0.523)
        pa_l = min(p_low_wf, p_low_wf - sgn_wf * 9.6) - 1.0
        pb_l = max(p_low_wf, p_low_wf - sgn_wf * 9.6) + 1.0
        _add_gang_clamp_bar(bm_comb, pa_l, pb_l, -2.310, _r_comb_surf(-2.310), 0.558)


    for sgn_ca, d_bot_ca, d_top_ca in ((-1.0, 212.0, 241.5), (+1.0, 148.0, 118.5)):
        _add_port_flange(bm_comb, d_bot_ca, -2.445, _r_comb_surf(-2.445), r_fl=0.022, n_bolts=6)
        ca_wps = [
            (d_bot_ca, -2.992, 0.548),
            (d_top_ca, -2.992, 0.542),
            (d_top_ca, -2.675, 0.546),
            (d_bot_ca, -2.445, 0.554),
            (d_bot_ca, -2.445, _r_comb_surf(-2.445) - 0.004),
        ]
        _add_fwd_pipe(bm_comb, ca_wps, tube_r=0.0108, clamp_indices=(2,), union_indices=(1,),
                      r_case_fn=_r_comb_surf, segs=22, bend_r=0.052, bnut_ends=False)


    ovh_pts = [_cp(k * 10.0, -2.638, _r_comb_surf(-2.638) + 0.014) for k in range(37)]
    ovh_fr = bm_add_spline_tube(bm_comb, ovh_pts, radius=0.0032, n_samples=144, segs=14, cap_ends=True, corrugations=56, corr_amp=0.09)
    for kc_o in range(12):
        dc_o = kc_o * 30.0 + 10.0
        pc_o = _cp(dc_o, -2.638, _r_comb_surf(-2.638) + 0.014)
        bf_o = min(ovh_fr, key=lambda fr: (fr[0] - pc_o).length_squared)
        bm_add_adel_loop_clamp(bm_comb, bf_o[0], bf_o[1], _cp(dc_o, -2.638, _r_comb_surf(-2.638)), Vector((0, 1, 0)), tube_r=0.0032)

    for deg_bs_j, y_bs_j, d_dst_j in [
        (44.0, -2.66, 54.0), (58.0, -2.43, 66.0), (96.0, -2.62, 88.0),
        (264.0, -2.62, 272.0), (302.0, -2.43, 294.0), (316.0, -2.66, 306.0),
    ]:
        rc_j = _r_comb_surf(y_bs_j)
        _add_fwd_pipe(
            bm_comb,
            [(deg_bs_j, y_bs_j, rc_j + 0.012), (d_dst_j, y_bs_j, rc_j + 0.016), (d_dst_j, -2.355, 0.552)],
            tube_r=0.0032, clamp_indices=(2,), r_case_fn=_r_comb_surf, segs=14, bend_r=0.022,
        )
        pt_ft_j = _cp(d_dst_j, -2.355, 0.552)
        t_ft_c = Vector((math.cos(math.radians(d_dst_j)), 0.0, -math.sin(math.radians(d_dst_j))))
        bm_add_bevel_locked_hex(bm_comb, pt_ft_j - t_ft_c * 0.007, pt_ft_j + t_ft_c * 0.007,
                                [(0.0, 0.0058), (0.25, 0.0072), (0.75, 0.0072), (1.0, 0.0058)], corner_bevel=0.14, cap_ends=True)

    create_mesh_obj("Fwd_08_Combustor_Turbine_Case_Fuel_And_Igniters", bm_comb, coll_fwd, clay)


    bm_vent = bmesh.new()

    def _r_aft_waist(yv: float) -> float:
        if yv < -2.245:
            return _r_comb_surf(yv)
        return 0.542 + 0.003 * max(0.0, min(1.0, (yv - (-2.245)) / 0.365))


    b_afc = bmesh.new()
    for y_ah, r_ah in ((-2.845, 0.502), (-2.465, 0.522)):
        bm_add_clevis_plate(b_afc, (y_ah - 0.014, r_ah), (y_ah + 0.014, 0.558), 0.010, 0.012, -0.088, +0.088, arc_segs=11)
        for sx_ah in (-0.072, +0.072):
            bm_add_clevis_plate(b_afc, (y_ah - 0.018, r_ah + 0.004), (y_ah + 0.018, 0.588), 0.008, 0.008, sx_ah - 0.0045, sx_ah + 0.0045, arc_segs=9)
    _bm_add_box_x_yz(b_afc, -0.084, +0.084, -2.875, -2.435, 0.552, 0.602, nx=4, ny=6, nz=2, chamfer=0.0014)
    _bm_add_box_x_yz(b_afc, -0.062, +0.062, -2.835, -2.475, 0.598, 0.626, nx=3, ny=5, nz=2, chamfer=0.0012)
    for yf_afc in (-2.810, -2.735, -2.660, -2.585, -2.510):
        _bm_add_box_x_yz(b_afc, -0.0865, +0.0865, yf_afc - 0.0024, yf_afc + 0.0024, 0.556, 0.598, nx=2, ny=1, nz=1, chamfer=0.0005)

    for kz_v in range(5):
        yz_v = -2.805 + kz_v * 0.075
        sgn_zv = -1.0 if (kz_v % 2 == 0) else +1.0
        xz_v = sgn_zv * 0.030
        bm_add_segmented_tube(
            b_afc, Vector((xz_v, yz_v, 0.622)), Vector((xz_v, yz_v, 0.637)),
            [(0.0, 0.019), (0.35, 0.021), (0.80, 0.018), (1.0, 0.013)], segs=22, cap_ends=True,
        )
        bm_add_bevel_locked_hex(
            b_afc, Vector((xz_v, yz_v, 0.636)), Vector((xz_v, yz_v, 0.641)),
            [(0.0, 0.0075), (0.30, 0.0092), (0.80, 0.0092), (1.0, 0.0070)], corner_bevel=0.14, cap_ends=True,
        )
        zv_bpts = []
        for kb_zv in range(4):
            azv = 0.25 * math.pi + 0.5 * math.pi * kb_zv
            p_zvb = Vector((xz_v + 0.0165 * math.cos(azv), yz_v + 0.0165 * math.sin(azv), 0.626))
            zv_bpts.append(p_zvb)
            bm_add_aerospace_bolt(b_afc, p_zvb, Vector((0, 0, 1)), head_r=0.0017, height=0.0016)
        bm_add_bolt_pair_lockwire(b_afc, zv_bpts[0], zv_bpts[1], Vector((0, 0, 1)), head_r=0.0017, strand_r=0.00022)
        bm_add_bolt_pair_lockwire(b_afc, zv_bpts[2], zv_bpts[3], Vector((0, 0, 1)), head_r=0.0017, strand_r=0.00022)
        bm_add_mil38999_connector(
            b_afc, Vector((sgn_zv * 0.061, yz_v, 0.613)), Vector((sgn_zv, 0.0, 0.0)),
            shell_r=0.0070, with_flange=False,
            cable_pts=[
                Vector((sgn_zv * 0.088, yz_v + 0.014, 0.584)),
                Vector((sgn_zv * 0.092, yz_v + 0.036, 0.562)),
            ],
            cable_braided=True, n_zipties=1,
        )
    bm_add_segmented_tube(
        b_afc, Vector((0.0, -2.865, 0.614)), Vector((0.0, -2.445, 0.614)),
        [(0.000, 0.0185), (0.008, 0.0220), (0.992, 0.0220), (1.000, 0.0185)],
        segs=24, cap_ends=True,
    )

    _bm_add_box_x_yz(b_afc, -0.024, +0.024, -2.695, -2.615, 0.634, 0.6375, nx=2, ny=2, nz=1, chamfer=0.0004)
    bm_add_3d_plate_text(
        b_afc, ["AFC 5-ZONE", "AUGMENTER"],
        Vector((0.0, -2.655, 0.6375)),
        Vector((0.0, 1.0, 0.0)), Vector((-1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)),
        width=0.068, height=0.038, relief=0.00050, with_border=True, with_dividers=True,
    )
    afc_bpts = []
    for sy_ab in (-2.860, -2.655, -2.450):
        for sx_ab in (-0.074, +0.074):
            p_ab_i = Vector((sx_ab, sy_ab, 0.602))
            afc_bpts.append(p_ab_i)
            bm_add_aerospace_bolt(b_afc, p_ab_i, Vector((0, 0, 1)), head_r=0.0024, height=0.0022)
    for kb_af in range(0, len(afc_bpts), 2):
        bm_add_bolt_pair_lockwire(b_afc, afc_bpts[kb_af], afc_bpts[kb_af + 1], Vector((0, 0, 1)), head_r=0.0024, strand_r=0.00028)

    for sx_af_c in (-0.106, +0.106):
        sgn_afc = 1.0 if sx_af_c > 0 else -1.0
        bm_add_segmented_tube(
            b_afc, Vector((sx_af_c, -2.865, 0.592)), Vector((sx_af_c, -2.645, 0.592)),
            [
                (0.000, 0.0265), (0.014, 0.0300), (0.060, 0.0300), (0.068, 0.0280),
                (0.470, 0.0280), (0.476, 0.0330), (0.554, 0.0330), (0.560, 0.0280),
                (0.932, 0.0280), (0.940, 0.0300), (0.986, 0.0300), (1.000, 0.0265),
            ],
            segs=28, cap_ends=True,
        )
        bm_add_segmented_tube(
            b_afc, Vector((sx_af_c + sgn_afc * 0.031, -2.752, 0.576)), Vector((sx_af_c + sgn_afc * 0.031, -2.752, 0.608)),
            [(0.0, 0.0038), (0.25, 0.0052), (0.75, 0.0052), (1.0, 0.0032)], segs=14, cap_ends=True,
        )
        bm_add_helical_thread_cylinder(
            b_afc, Vector((sx_af_c + sgn_afc * 0.031, -2.752, 0.606)), Vector((sx_af_c + sgn_afc * 0.031, -2.752, 0.620)),
            r_core=0.0022, thread_h=0.00050, n_turns=6, segs_turn=14, cap_core=True,
        )
        bm_add_bevel_locked_hex(
            b_afc, Vector((sx_af_c + sgn_afc * 0.031, -2.752, 0.608)), Vector((sx_af_c + sgn_afc * 0.031, -2.752, 0.614)),
            [(0.0, 0.0040), (0.30, 0.0048), (0.80, 0.0048), (1.0, 0.0038)], corner_bevel=0.14, cap_ends=True,
        )
        af_fb_pts = []
        for kb_fb in range(8):
            afb = 2.0 * math.pi * kb_fb / 8.0
            p_affb = Vector((sx_af_c + 0.022 * math.cos(afb), -2.865, 0.592 + 0.022 * math.sin(afb)))
            af_fb_pts.append(p_affb)
            bm_add_aerospace_bolt(b_afc, p_affb, Vector((0, -1, 0)), head_r=0.0018, height=0.0018)
        for kb_fb in range(0, 8, 2):
            bm_add_bolt_pair_lockwire(b_afc, af_fb_pts[kb_fb], af_fb_pts[kb_fb + 1], Vector((0, -1, 0)), head_r=0.0018, strand_r=0.00024)
        bm_add_bevel_locked_hex(
            b_afc, Vector((sx_af_c, -2.864, 0.592)), Vector((sx_af_c, -2.874, 0.592)),
            [(0.0, 0.0075), (0.30, 0.0092), (0.80, 0.0092), (1.0, 0.0068)], corner_bevel=0.14, cap_ends=True,
        )
        bm_add_segmented_tube(
            b_afc, Vector((sx_af_c + sgn_afc * 0.024, -2.815, 0.606)), Vector((sx_af_c + sgn_afc * 0.034, -2.815, 0.614)),
            [(0.0, 0.0065), (0.55, 0.0065), (0.65, 0.0036), (1.0, 0.0034)], segs=16, cap_ends=True,
        )
        bm_add_segmented_tube(
            b_afc, Vector((sx_af_c * 0.92, -2.625, 0.602)), Vector((sx_af_c * 0.92, -2.435, 0.602)),
            [
                (0.000, 0.0225), (0.015, 0.0268), (0.068, 0.0268), (0.076, 0.0245),
                (0.455, 0.0245), (0.462, 0.0290), (0.538, 0.0290), (0.545, 0.0245),
                (0.924, 0.0245), (0.932, 0.0268), (0.985, 0.0268), (1.000, 0.0225),
            ],
            segs=26, cap_ends=True,
        )
        bm_add_bevel_locked_hex(
            b_afc, Vector((sx_af_c * 0.92, -2.436, 0.602)), Vector((sx_af_c * 0.92, -2.426, 0.602)),
            [(0.0, 0.0070), (0.30, 0.0085), (0.80, 0.0085), (1.0, 0.0062)], corner_bevel=0.14, cap_ends=True,
        )
    bm_merge(bm_vent, b_afc, Matrix.Rotation(math.radians(180.0), 4, 'Y'), default_mat_id=MAT_AVIONICS_BOX)


    for ph_vu, y_v0, y_v1, r_vc, rad_vc, has_rib, sgn_vout, vu_lbl in [
        (156.0, -2.445, -2.015, 0.586, 0.036, True,  -1.0, ["NAC-HYD",   "P/N 402"]),
        (180.0, -2.395, -1.915, 0.600, 0.035, False, -1.0, ["TURB SCAV", "FILTER #5"]),
        (204.0, -2.485, -1.945, 0.590, 0.038, True,  +1.0, ["P&D VALVE", "COOLER"]),
    ]:
        b_vu = bmesh.new()
        st_vu = [
            (0.000, rad_vc * 0.94), (0.006, rad_vc * 1.06), (0.052, rad_vc * 1.06),
            (0.055, rad_vc * 1.00), (0.472, rad_vc * 1.00),
            (0.476, rad_vc * 1.13), (0.534, rad_vc * 1.13), (0.538, rad_vc * 1.00),
        ]
        if has_rib:
            for kr_v in range(4):
                fv = 0.585 + kr_v * 0.072
                st_vu.extend([
                    (fv,         rad_vc * 0.92),
                    (fv + 0.004, rad_vc * 1.06),
                    (fv + 0.032, rad_vc * 1.06),
                    (fv + 0.036, rad_vc * 0.92),
                ])
        st_vu.extend([
            (0.945, rad_vc * 1.00), (0.948, rad_vc * 1.06),
            (0.994, rad_vc * 1.06), (1.000, rad_vc * 0.94),
        ])
        bm_add_segmented_tube(b_vu, Vector((0.0, y_v0, r_vc)), Vector((0.0, y_v1, r_vc)), st_vu, segs=32, cap_ends=True)


        y_vmid = y_v0 + 0.505 * (y_v1 - y_v0)
        x_vtrun = sgn_vout * (rad_vc * 1.10)
        bm_add_segmented_tube(
            b_vu, Vector((x_vtrun, y_vmid, r_vc - 0.018)), Vector((x_vtrun, y_vmid, r_vc + 0.018)),
            [(0.0, 0.0042), (0.22, 0.0062), (0.38, 0.0032), (0.62, 0.0032), (0.78, 0.0062), (1.0, 0.0040)], segs=16, cap_ends=True,
        )
        bm_add_helical_thread_cylinder(
            b_vu, Vector((x_vtrun, y_vmid, r_vc + 0.017)), Vector((x_vtrun, y_vmid, r_vc + 0.032)),
            r_core=0.0026, thread_h=0.00055, n_turns=6, segs_turn=14, cap_core=True,
        )
        bm_add_bevel_locked_hex(
            b_vu, Vector((x_vtrun, y_vmid, r_vc + 0.018)), Vector((x_vtrun, y_vmid, r_vc + 0.025)),
            [(0.0, 0.0048), (0.30, 0.0060), (0.80, 0.0060), (1.0, 0.0044)], corner_bevel=0.14, cap_ends=True,
        )


        for is_aft_c, y_face_c, ny_dir in ((False, y_v0, -1.0), (True, y_v1, +1.0)):
            cap_bpts = []
            for kb_vc in range(8):
                avc = 2.0 * math.pi * kb_vc / 8.0
                p_vcb = Vector((rad_vc * 0.80 * math.cos(avc), y_face_c, r_vc + rad_vc * 0.80 * math.sin(avc)))
                cap_bpts.append(p_vcb)
                bm_add_aerospace_bolt(b_vu, p_vcb, Vector((0, ny_dir, 0)), head_r=0.0020, height=0.0020)
            for kb_vc in range(0, 8, 2):
                bm_add_bolt_pair_lockwire(b_vu, cap_bpts[kb_vc], cap_bpts[kb_vc + 1], Vector((0, ny_dir, 0)), head_r=0.0020, strand_r=0.00026)
        bm_add_bevel_locked_hex(
            b_vu, Vector((0.0, y_v1 - 0.002, r_vc)), Vector((0.0, y_v1 + 0.012, r_vc)),
            [(0.0, 0.0095), (0.28, 0.0118), (0.80, 0.0118), (1.0, 0.0085)], corner_bevel=0.14, cap_ends=True,
        )
        bm_add_segmented_tube(
            b_vu, Vector((sgn_vout * rad_vc * 0.78, y_v0 + 0.090, r_vc + rad_vc * 0.52)),
            Vector((sgn_vout * (rad_vc * 0.78 + 0.011), y_v0 + 0.090, r_vc + rad_vc * 0.52 + 0.008)),
            [(0.0, 0.0078), (0.55, 0.0078), (0.65, 0.0042), (1.0, 0.0040)], segs=16, cap_ends=True,
        )
        _bm_add_box_x_yz(
            b_vu, min(sgn_vout * (rad_vc - 0.001), sgn_vout * (rad_vc + 0.0022)),
            max(sgn_vout * (rad_vc - 0.001), sgn_vout * (rad_vc + 0.0022)),
            y_v0 + 0.120, y_v0 + 0.190, r_vc - 0.014, r_vc + 0.014, nx=1, ny=2, nz=2, chamfer=0.0004,
        )
        bm_add_3d_plate_text(
            b_vu, vu_lbl,
            Vector((sgn_vout * (rad_vc + 0.0022), y_v0 + 0.155, r_vc)),
            Vector((0.0, -sgn_vout * 1.0, 0.0)), Vector((0.0, 0.0, -1.0)), Vector((sgn_vout, 0.0, 0.0)),
            width=0.056, height=0.020, relief=0.00050, with_border=True, with_dividers=True,
        )
        for sy_vp in (y_v0 + 0.126, y_v0 + 0.184):
            for sz_vp in (r_vc - 0.010, r_vc + 0.010):
                bm_add_dome_rivet(b_vu, Vector((sgn_vout * (rad_vc + 0.0022), sy_vp, sz_vp)), Vector((sgn_vout, 0, 0)), radius=0.0011)

        for y_vs in (y_v0 + 0.085, y_v1 - 0.085):
            r_in_vs = _r_aft_waist(y_vs) - 0.002
            bm_add_clevis_plate(
                b_vu, (y_vs - 0.008, r_in_vs), (y_vs + 0.004, r_vc - rad_vc * 0.36),
                0.009, rad_vc * 0.60, -0.012, +0.012, arc_segs=13,
            )
        bm_merge(bm_vent, b_vu, Matrix.Rotation(math.radians(ph_vu), 4, 'Y'), default_mat_id=MAT_PUMPS_OILTANK)


    for wps_jmp in [
        [(170.5, -2.505, 0.588), (161.0, -2.455, 0.584), (156.5, -2.385, 0.582)],
        [(189.5, -2.505, 0.588), (199.0, -2.475, 0.586), (203.5, -2.405, 0.584)],
    ]:
        _add_fwd_pipe(bm_vent, wps_jmp, tube_r=0.0046, clamp_indices=(), r_case_fn=_r_aft_waist, segs=16, bend_r=0.028)


    for sgn_m, d_run in ((+1.0, 151.0), (-1.0, 209.0)):
        ab_main_wps = [
            (d_run + sgn_m * 12.0, -2.985, 0.584),
            (d_run,                -2.855, 0.566),
            (d_run,                -2.425, 0.568),
            (d_run - sgn_m * 6.0,  -2.265, 0.578),
            (d_run - sgn_m * 6.0,  -1.905, 0.578),
        ]
        b_abm = bmesh.new()
        abm_pts = build_cyl_mandrel_pts(ab_main_wps, bend_r=0.068, r_unwrap=0.565, arc_segs=18, max_step=0.012)
        abm_fr = bm_sweep_centerline_tube(b_abm, abm_pts, radius=0.0140, segs=24, cap_ends=True)
        for y_mf_t in (-2.78, -2.22):
            bf_mf = min(abm_fr, key=lambda fr: abs(fr[0].y - y_mf_t))
            pm_f, tm_f, um_f, vm_f, _ = bf_mf
            bm_add_segmented_tube(
                b_abm, pm_f - tm_f * 0.008, pm_f + tm_f * 0.008,
                [(0.00, 0.0155), (0.10, 0.0225), (0.90, 0.0225), (1.00, 0.0155)], segs=24, cap_ends=True,
            )
            for kb_m in range(6):
                ang_m = 2.0 * math.pi * kb_m / 6.0
                off_m = (um_f * math.cos(ang_m) + vm_f * math.sin(ang_m)) * 0.019
                bm_add_aerospace_bolt(b_abm, pm_f + off_m + tm_f * 0.007, tm_f, head_r=0.0020, height=0.0020)
        for y_mc in (-2.72, -2.58, -2.10):
            bf_mc = min(abm_fr, key=lambda fr: abs(fr[0].y - y_mc))
            ph_mc = math.degrees(math.atan2(bf_mc[0].x, bf_mc[0].z)) % 360.0
            bm_add_adel_loop_clamp(b_abm, bf_mc[0], bf_mc[1],
                                   _cp(ph_mc, bf_mc[0].y, _r_aft_waist(bf_mc[0].y)), Vector((0, 1, 0)), tube_r=0.0140)
        bm_merge(bm_vent, b_abm, Matrix.Identity(4), default_mat_id=MAT_RIGID_TUBE)


    ventral_long_tubes = [
        ([(142.0, -3.020, 0.556), (142.0, -2.756, 0.556), (142.0, -2.512, 0.562), (142.0, -2.245, 0.574), (143.0, -1.895, 0.574)], 0.0055, (1, 3), 0),
        ([(146.5, -3.020, 0.562), (146.5, -2.756, 0.562), (146.5, -2.512, 0.568), (146.5, -2.245, 0.580), (147.0, -2.065, 0.576)], 0.0048, (1, 2), 0),
        ([(161.2, -2.980, 0.562), (161.2, -2.650, 0.566), (162.5, -2.245, 0.576), (163.5, -1.905, 0.576)], 0.0060, (1, 2), 0),
        ([(198.8, -2.980, 0.562), (198.8, -2.650, 0.566), (197.5, -2.245, 0.576), (196.5, -1.905, 0.576)], 0.0060, (1, 2), 0),
        ([(213.5, -3.020, 0.562), (213.5, -2.756, 0.562), (213.5, -2.512, 0.568), (213.5, -2.245, 0.580), (213.0, -1.895, 0.576)], 0.0048, (1, 3), 0),
        ([(218.0, -3.020, 0.556), (218.0, -2.756, 0.556), (218.0, -2.512, 0.562), (218.0, -2.245, 0.574), (218.0, -1.895, 0.574)], 0.0055, (1, 3), 0),

        ([(137.0, -3.020, 0.548), (137.0, -2.680, 0.552), (137.0, -2.245, 0.570), (138.0, -1.895, 0.570)], 0.0062, (1, 2), 42),
        ([(139.5, -3.020, 0.552), (139.5, -2.680, 0.556), (139.5, -2.245, 0.574), (140.5, -1.895, 0.574)], 0.0054, (1, 2), 42),
        ([(220.5, -3.020, 0.552), (220.5, -2.680, 0.556), (220.5, -2.245, 0.574), (220.5, -1.895, 0.574)], 0.0054, (1, 2), 42),
        ([(223.0, -3.020, 0.548), (223.0, -2.680, 0.552), (223.0, -2.245, 0.570), (223.0, -1.895, 0.570)], 0.0062, (1, 2), 42),
    ]
    for wps_v, tr_v, cl_v, cr_v in ventral_long_tubes:
        _add_fwd_pipe(bm_vent, wps_v, tube_r=tr_v, clamp_indices=cl_v, r_case_fn=_r_aft_waist, segs=16, corrugations=cr_v, corr_amp=0.09)

    create_mesh_obj("Fwd_09_Ventral_AugmentorFuel_Metering_And_BellyTrunks", bm_vent, coll_fwd, clay)


    bm_dors = bmesh.new()


    b_fmount = bmesh.new()
    _bm_add_box_x_yz(b_fmount, -0.056, +0.056, -4.065, -3.915, 0.502, 0.536, nx=3, ny=4, nz=2)
    for sx_fm in (-0.024, +0.024):
        bm_add_clevis_plate(b_fmount, (-4.045, 0.530), (-3.985, 0.576), 0.016, 0.020, sx_fm - 0.0055, sx_fm + 0.0055, arc_segs=17)
        bm_add_clevis_plate(b_fmount, (-3.935, 0.530), (-3.985, 0.576), 0.016, 0.020, sx_fm - 0.0055, sx_fm + 0.0055, arc_segs=17)
    bm_add_chamfered_cyl_x(b_fmount, -3.985, 0.572, -0.036, +0.036, radius=0.015, segs=24, hollow_r=0.008)
    fm_bpts = []
    for sx_mb in (-0.044, +0.044):
        for sy_mb in (-4.050, -3.990, -3.930):
            pfmb = Vector((sx_mb, sy_mb, 0.536))
            fm_bpts.append(pfmb)
            bm_add_aerospace_bolt(b_fmount, pfmb, Vector((0, 0, 1)), head_r=0.0028, height=0.0026)
    for k_fmb in (0, 1, 3, 4):
        bm_add_bolt_pair_lockwire(b_fmount, fm_bpts[k_fmb], fm_bpts[k_fmb + 1], Vector((0, 0, 1)), head_r=0.0028, strand_r=0.00032)
    bm_merge(bm_dors, b_fmount, Matrix.Identity(4), default_mat_id=MAT_MACHINED_ALU)


    bm_add_segmented_tube(bm_dors, _cp(288.0, -3.868, 0.493), _cp(288.0, -3.868, 0.558),
                          [(0.0, 0.024), (0.50, 0.026), (0.80, 0.024), (1.0, 0.016)], segs=24, cap_ends=True)
    stbd_ai_fr = _add_fwd_pipe(
        bm_dors,
        [(288.0, -3.725, 0.492), (288.0, -3.868, 0.544), (306.0, -3.988, 0.544), (326.0, -4.088, 0.542), (352.0, -4.098, 0.541)],
        tube_r=0.0122, clamp_indices=(2, 3), r_case_ref=0.504, segs=24, bend_r=0.066, bnut_ends=False,
    )
    for f_svb in (0.24, 0.52, 0.78):
        idx_sv = min(len(stbd_ai_fr) - 2, max(1, int(round(f_svb * (len(stbd_ai_fr) - 1)))))
        pv_s, tv_s = stbd_ai_fr[idx_sv][0], stbd_ai_fr[idx_sv][1]
        bm_add_segmented_tube(bm_dors, pv_s - tv_s * 0.008, pv_s + tv_s * 0.008,
                              [(0.0, 0.0138), (0.22, 0.0176), (0.78, 0.0176), (1.0, 0.0138)], segs=22, cap_ends=True)
    _add_fwd_pipe(bm_dors, [(288.0, -3.868, 0.538), (288.0, -4.042, 0.538), (350.0, -4.042, 0.538)],
                  tube_r=0.0088, clamp_indices=(1, 2), r_case_ref=0.502, segs=20, bend_r=0.052, bnut_ends=False)
    for deg_rj, y_src_r, r_src_r in (
        (298.5, -4.042, 0.538), (311.0, -4.042, 0.538), (323.5, -4.042, 0.538),
        (336.5, -4.092, 0.542), (350.0, -4.096, 0.541),
    ):
        rj_wps = [
            (deg_rj,       y_src_r, r_src_r),
            (deg_rj,       -4.156,  0.537),
            (deg_rj + 4.5, -4.156,  0.534),
            (deg_rj + 4.5, -4.234,  0.528),
            (deg_rj + 4.5, -4.234,  0.508),
        ]
        _add_fwd_pipe(bm_dors, rj_wps, tube_r=0.0054, clamp_indices=(1,), union_indices=(3,), r_case_ref=0.512, segs=18, bend_r=0.026)
        p_rjb = _cp(deg_rj + 4.5, -4.234, 0.510)
        bm_add_segmented_tube(bm_dors, p_rjb, p_rjb + _cn(deg_rj + 4.5) * 0.011,
                              [(0.0, 0.0120), (0.65, 0.0120), (1.0, 0.0086)], segs=18, cap_ends=True)

    for idx_sr3, y_sr2 in enumerate((-3.944, -3.922, -3.900)):
        tr_sr = 0.0044 if idx_sr3 == 1 else 0.0038
        _add_fwd_pipe(bm_dors, [(286.0, -3.760, 0.509), (286.0, y_sr2, 0.514), (310.5, y_sr2, 0.514), (310.5, -3.956, 0.506)],
                      tube_r=tr_sr, clamp_indices=(1, 2), r_case_ref=0.496, segs=16, bend_r=0.022)
        _add_fwd_pipe(bm_dors, [(331.0, -3.760, 0.509), (331.0, y_sr2, 0.514), (354.5, y_sr2, 0.514), (354.5, -3.788, 0.510)],
                      tube_r=tr_sr, clamp_indices=(2,), r_case_ref=0.496, segs=16, bend_r=0.022)
    for deg_sbr, y_end_sb, r_end_sb, tr_sb in [
        (292.5, -3.748, 0.516, 0.0046),
        (306.5, -3.742, 0.512, 0.0042),
        (328.0, -3.742, 0.507, 0.0042),
        (335.0, -3.742, 0.512, 0.0052),
        (342.0, -3.742, 0.509, 0.0050),
    ]:
        _add_fwd_pipe(bm_dors, [(deg_sbr, -4.038, 0.514), (deg_sbr, -3.856, 0.512), (deg_sbr, y_end_sb, r_end_sb)],
                      tube_r=tr_sb, clamp_indices=(1,), r_case_ref=0.495, segs=16, bend_r=0.032)


    b_dhood = bmesh.new()
    _bm_add_box_x_yz(b_dhood, -0.064, +0.064, -3.555, -3.255, 0.493, 0.520, nx=3, ny=5, nz=2)
    _bm_add_box_x_yz(b_dhood, -0.048, +0.048, -3.525, -3.285, 0.518, 0.556, nx=3, ny=4, nz=2)
    bm_add_segmented_tube(
        b_dhood, Vector((0.0, -3.535, 0.552)), Vector((0.0, -3.265, 0.562)),
        [(0.000, 0.026), (0.014, 0.033), (0.720, 0.033), (0.734, 0.026), (0.986, 0.026), (1.000, 0.022)],
        segs=28, cap_ends=True,
    )
    dh_bpts = []
    for sx_dh in (-0.054, 0.0, +0.054):
        for sy_dh in (-3.540, -3.405, -3.270):
            pdhb = Vector((sx_dh, sy_dh, 0.520))
            dh_bpts.append(pdhb)
            bm_add_aerospace_bolt(b_dhood, pdhb, Vector((0, 0, 1)), head_r=0.0026, height=0.0024)
    for i_dh, j_dh in ((0, 1), (1, 2), (6, 7), (7, 8)):
        bm_add_bolt_pair_lockwire(b_dhood, dh_bpts[i_dh], dh_bpts[j_dh], Vector((0, 0, 1)), head_r=0.0026, strand_r=0.00030)
    bm_merge(bm_dors, b_dhood, Matrix.Rotation(math.radians(4.0), 4, 'Y'))


    b_dsol = bmesh.new()
    _bm_add_box_x_yz(b_dsol, -0.038, +0.038, -3.660, -3.500, 0.493, 0.506, nx=2, ny=3, nz=1, chamfer=0.0010)
    _bm_add_box_x_yz(b_dsol, -0.030, +0.030, -3.648, -3.512, 0.504, 0.538, nx=2, ny=3, nz=2, chamfer=0.0012)
    for yf_ds in (-3.635, -3.580, -3.525):
        _bm_add_box_x_yz(b_dsol, -0.033, +0.033, yf_ds - 0.0022, yf_ds + 0.0022, 0.506, 0.536, nx=2, ny=1, nz=1, chamfer=0.0004)
    dsol_bpts = []
    for sy_dsb in (-3.650, -3.580, -3.510):
        for sx_dsb in (-0.032, +0.032):
            p_dsb = Vector((sx_dsb, sy_dsb, 0.506))
            dsol_bpts.append(p_dsb)
            bm_add_aerospace_bolt(b_dsol, p_dsb, Vector((0, 0, 1)), head_r=0.0021, height=0.0019)
    for k_dsb in (0, 2, 4):
        bm_add_bolt_pair_lockwire(b_dsol, dsol_bpts[k_dsb], dsol_bpts[k_dsb + 1], Vector((0, 0, 1)), head_r=0.0021, strand_r=0.00026)
    for y_sc_d, r_sc_d, h_sc_d in ((-3.618, 0.018, 0.568), (-3.542, 0.016, 0.564)):
        bm_add_segmented_tube(
            b_dsol, Vector((0.0, y_sc_d, 0.535)), Vector((0.0, y_sc_d, h_sc_d)),
            [(0.0, r_sc_d * 1.08), (0.18, r_sc_d), (0.48, r_sc_d), (0.52, r_sc_d * 1.12), (0.56, r_sc_d), (0.86, r_sc_d), (1.0, r_sc_d * 0.68)],
            segs=24, cap_ends=True,
        )
        bm_add_bevel_locked_hex(
            b_dsol, Vector((0.0, y_sc_d, h_sc_d - 0.001)), Vector((0.0, y_sc_d, h_sc_d + 0.0045)),
            [(0.0, r_sc_d * 0.42), (0.30, r_sc_d * 0.52), (0.80, r_sc_d * 0.52), (1.0, r_sc_d * 0.38)], corner_bevel=0.14, cap_ends=True,
        )
        bm_add_mil38999_connector(
            b_dsol, Vector((+0.029, y_sc_d, 0.523)), Vector((+1.0, 0.0, 0.0)),
            shell_r=0.0072, with_flange=False,
            cable_pts=[Vector((+0.052, y_sc_d + 0.016, 0.512)), Vector((+0.062, y_sc_d + 0.042, 0.500))],
            cable_braided=True, n_zipties=1,
        )

    _bm_add_box_x_yz(b_dsol, -0.0295, -0.0165, -3.606, -3.554, 0.537, 0.5392, nx=1, ny=2, nz=1, chamfer=0.0003)
    bm_add_3d_plate_text(
        b_dsol, ["BLEED SOL", "P3-VALVE"],
        Vector((-0.023, -3.580, 0.5392)),
        Vector((0.0, -1.0, 0.0)), Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)),
        width=0.046, height=0.011, relief=0.00040, with_dividers=True,
    )
    for y_ps_d in (-3.638, -3.522):
        ps_fr = bm_add_spline_tube(
            b_dsol,
            [Vector((-0.029, y_ps_d, 0.520)), Vector((-0.048, y_ps_d, 0.518)), Vector((-0.052, y_ps_d, 0.495))],
            radius=0.0032, n_samples=18, segs=14, cap_ends=True,
        )
        bm_add_bnut_flare_fitting(b_dsol, ps_fr[2][0], ps_fr[2][1], tube_r=0.0032)
        bm_add_bnut_flare_fitting(b_dsol, ps_fr[-2][0], ps_fr[-2][1], tube_r=0.0032)
    bm_merge(bm_dors, b_dsol, Matrix.Rotation(math.radians(338.0), 4, 'Y'), default_mat_id=MAT_AVIONICS_BOX)


    for deg_dd in (16.0, 344.0):
        dd_wps = [
            (deg_dd, -3.725, 0.490),
            (deg_dd, -3.725, 0.558),
            (deg_dd, -2.440, 0.562),
            (deg_dd, -2.333, 0.556),
            (deg_dd, -2.210, 0.564),
            (deg_dd, -1.905, 0.566),
        ]
        dd_pts_m = build_cyl_mandrel_pts(dd_wps, bend_r=0.048, r_unwrap=0.560, arc_segs=18, max_step=0.015)
        dd_fr = bm_sweep_centerline_tube(bm_dors, dd_pts_m, radius=0.0125, segs=24, cap_ends=True)
        _add_port_flange(bm_dors, deg_dd, -3.725, 0.494, r_fl=0.024, n_bolts=6)

        for f_vb in (0.20, 0.46, 0.70, 0.90):
            idx_vb = min(len(dd_fr) - 3, max(2, int(round(f_vb * (len(dd_fr) - 1)))))
            p_vb, t_vb = dd_fr[idx_vb][0], dd_fr[idx_vb][1]
            n_rad_vb = _cn(deg_dd)
            u_tan_vb = n_rad_vb.cross(t_vb).normalized()
            st_bel = [(0.0, 0.0138), (0.08, 0.0198), (0.14, 0.0198), (0.17, 0.0152)]
            for kb_w in range(5):
                fb0 = 0.20 + kb_w * 0.12
                st_bel.extend([(fb0, 0.0150), (fb0 + 0.04, 0.0178), (fb0 + 0.08, 0.0178), (fb0 + 0.11, 0.0150)])
            st_bel.extend([(0.83, 0.0152), (0.86, 0.0198), (0.92, 0.0198), (1.0, 0.0138)])
            bm_add_segmented_tube(
                bm_dors, p_vb - t_vb * 0.024, p_vb + t_vb * 0.024, st_bel, segs=26, cap_ends=True,
            )
            for sgn_tr in (-1.0, +1.0):
                p_tr_mid = p_vb + u_tan_vb * (sgn_tr * 0.0210)
                bm_add_segmented_tube(
                    bm_dors, p_tr_mid - t_vb * 0.026, p_tr_mid + t_vb * 0.026,
                    [(0.0, 0.0018), (0.12, 0.0032), (0.22, 0.0020), (0.78, 0.0020), (0.88, 0.0032), (1.0, 0.0018)],
                    segs=14, cap_ends=True,
                )
                for s_end_t in (-1.0, +1.0):
                    p_ear = p_vb + t_vb * (s_end_t * 0.020)
                    bm_add_segmented_tube(
                        bm_dors, p_ear + u_tan_vb * (sgn_tr * 0.015), p_ear + u_tan_vb * (sgn_tr * 0.0235),
                        [(0.0, 0.0042), (0.70, 0.0042), (1.0, 0.0032)], segs=14, cap_ends=True,
                    )
                    bm_add_bevel_locked_hex(
                        bm_dors, p_tr_mid + t_vb * (s_end_t * 0.022), p_tr_mid + t_vb * (s_end_t * 0.026),
                        [(0.0, 0.0032), (0.25, 0.0038), (0.75, 0.0038), (1.0, 0.0030)], corner_bevel=0.14, cap_ends=True,
                    )
        for y_dc in (-3.22, -2.64, -2.06):
            bf_dc = min(dd_fr, key=lambda fr: abs(fr[0].y - y_dc))
            bm_add_adel_loop_clamp(bm_dors, bf_dc[0], bf_dc[1],
                                   _cp(deg_dd, bf_dc[0].y, _r_aft_waist(bf_dc[0].y) if bf_dc[0].y > -3.051 else 0.494), Vector((0, 1, 0)), tube_r=0.0125)


    for deg_dt in (6.0, 25.0, 335.0, 354.0):
        dt_wps = [
            (deg_dt, -3.960, 0.528),
            (deg_dt, -3.856, 0.524),
            (deg_dt, -3.620, 0.514),
            (deg_dt, -3.120, 0.514),
            (deg_dt, -2.756, 0.534),
            (deg_dt, -2.512, 0.548),
            (deg_dt, -2.333, 0.552),
            (deg_dt, -1.882, 0.564),
        ]
        _add_fwd_pipe(bm_dors, dt_wps, tube_r=0.0040, clamp_indices=(2, 3, 5, 6),
                      r_case_fn=(lambda yv: 0.500 if yv < -3.773 else (0.494 if yv < -3.051 else _r_aft_waist(yv))), segs=16)


    for y_acc, r_acc, tr_acc, y_imp in [
        (-2.808, 0.542, 0.0092, -2.712),
        (-2.562, 0.556, 0.0088, -2.455),
    ]:
        prof_acc = [
            (y_acc + tr_acc * math.cos(2 * math.pi * s / 22), r_acc + tr_acc * math.sin(2 * math.pi * s / 22))
            for s in range(22)
        ]
        bm_add_revolved_profile(bm_dors, prof_acc, segs=224, close_loop=True)

        for deg_af in (16.0, 344.0):
            bm_add_segmented_tube(bm_dors, _cp(deg_af, y_acc, r_acc), _cp(deg_af, y_acc, 0.565),
                                  [(0.0, 0.0125), (0.50, 0.0145), (1.0, 0.0125)], segs=20, cap_ends=True)

        rc_st = _r_comb_surf(y_acc)
        rc_im = _r_comb_surf(y_imp)
        b_acc_u = bmesh.new()
        bm_add_clevis_plate(b_acc_u, (y_acc - 0.008, rc_st - 0.001), (y_acc + 0.008, r_acc), 0.005, tr_acc + 0.0035, -0.004, +0.004, arc_segs=11)


        imp_pts = build_cyl_mandrel_pts(
            [(0.0, y_acc, r_acc), (1.5, y_acc + 0.028, r_acc), (1.5, y_imp, r_acc - 0.004), (1.5, y_imp, rc_im + 0.002)],
            bend_r=0.022, r_unwrap=0.545, arc_segs=14, max_step=0.008,
        )
        im_fr = bm_sweep_centerline_tube(b_acc_u, imp_pts, radius=0.0035, segs=14, cap_ends=True)
        bm_add_bnut_flare_fitting(b_acc_u, im_fr[2][0], im_fr[2][1], tube_r=0.0036)
        bm_add_bnut_flare_fitting(b_acc_u, im_fr[-3][0], im_fr[-3][1], tube_r=0.0036)
        bm_add_segmented_tube(b_acc_u, Vector((0.014, y_imp, rc_im - 0.003)), Vector((0.014, y_imp, rc_im + 0.010)),
                              [(0.0, 0.0105), (0.65, 0.0105), (1.0, 0.0072)], segs=18, cap_ends=True)
        for k_au in range(16):
            deg_au = k_au * 22.5 + 11.25

            if 164.0 <= deg_au <= 196.0:
                continue
            bm_merge(bm_dors, b_acc_u.copy(), Matrix.Rotation(math.radians(deg_au), 4, 'Y'))
        b_acc_u.free()

    create_mesh_obj("Fwd_10_Dorsal_BleedManifolds_ACC_And_CoolingRings", bm_dors, coll_fwd, clay)


def build_static_base_and_interior(spec: NozzleSpec, mats: dict, coll_static, coll_interior):
    clay = mats["clay"]
    two_pi = 2.0 * math.pi
    fh_slot = spec.fork_half_span

    def _ang_diff(a: float, b: float) -> float:
        d = abs(math.fmod(a - b, two_pi))
        if d > two_pi:
            d -= two_pi
        return two_pi - d if d > math.pi else d

    def _smooth01(x: float) -> float:
        c = max(0.0, min(1.0, x))
        return c * c * (3.0 - 2.0 * c)

    act_phis = [two_pi * (2 * k) / spec.n_sectors for k in range(spec.n_actuators)]
    sec_phis = [two_pi * i / spec.n_sectors for i in range(spec.n_sectors)]
    rvdt_phi_rad = math.radians(56.25)


    bm_duct = bmesh.new()
    build_isogrid_cased_cylinder(
        bm_duct,
        y0=-1.880, y1=-0.855,
        r_inner=0.514, r_outer=0.5430,
        n_circ=24, n_rows=6,
        pocket_depth=0.0056,
        r_inner_end=0.517, r_outer_end=0.5448,
        axial_ribs=True,
    )
    prof_aug_fwd_flange = [
        (-1.880, 0.537), (-1.880, 0.568), (-1.868, 0.568), (-1.862, 0.5430),
    ]
    bm_add_revolved_profile(bm_duct, prof_aug_fwd_flange, segs=256, close_loop=True, bevel_r=0.00045)

    aug_fwd_bolts = []
    for iaf in range(96):
        aaf = 2.0 * math.pi * iaf / 96.0
        paf = Vector((0.5555 * math.sin(aaf), -1.868, 0.5555 * math.cos(aaf)))
        aug_fwd_bolts.append(paf)
        bm_add_aerospace_bolt(bm_duct, paf, Vector((0.0, 1.0, 0.0)), head_r=0.0028, height=0.0027, with_12pt_nut_depth=0.022)
    for iaf in range(0, 96, 2):
        bm_add_bolt_pair_lockwire(bm_duct, aug_fwd_bolts[iaf], aug_fwd_bolts[iaf + 1], Vector((0.0, 1.0, 0.0)), head_r=0.0028, strand_r=0.00033)
    create_mesh_obj("Static_Forward_Waffle_Duct", bm_duct, coll_static, clay)

    def _case_radial_cap(phi: float, yv: float) -> float:
        r_lim = 0.555
        if yv < -0.244:
            d_act = min(_ang_diff(phi, pa) for pa in act_phis) * 0.550
            if d_act < 0.056:
                r_lim = min(r_lim, 0.5490)
        if -0.131 < yv < 0.0025:
            d_sec_min = min(_ang_diff(phi, ps) for ps in sec_phis)
            x_loc = abs(d_sec_min * 0.550)
            if yv < -0.095:
                ey_s = _smooth01((yv - (-0.131)) / 0.0030)
                d_sl = abs(x_loc - fh_slot)
                if d_sl < 0.0092 + 0.0028:
                    ep_s = 1.0 - _smooth01((d_sl - 0.0092) / 0.0028)
                    r_lim = min(r_lim, 0.550 - (0.550 - 0.5482) * ey_s * ep_s)
            else:
                ey_aft = _smooth01(min((yv - (-0.098)) / 0.0045, (0.0020 - yv) / 0.0032))
                d_tr = abs(x_loc - 0.0213)
                if d_tr < 0.0141 + 0.0020:
                    ep_tr = 1.0 - _smooth01((d_tr - 0.0141) / 0.0020)
                    r_lim = min(r_lim, 0.550 - (0.550 - 0.5360) * ey_aft * ep_tr)
        return r_lim

    bm_case = bmesh.new()
    prof_flare_main = [
        (-0.855, 0.517), (-0.855, 0.5485), (-0.847, 0.5485), (-0.841, 0.5445),
        (-0.744, 0.5448), (-0.744, 0.522), (-0.805, 0.518),
    ]
    bm_add_revolved_profile(bm_case, prof_flare_main, segs=256, close_loop=True, bevel_r=0.00042)

    prof_flare_lip = [
        (-0.746, 0.522), (-0.746, 0.5448), (-0.726, 0.604), (-0.716, 0.608), (-0.716, 0.524),
    ]
    notches_flare_lip = (
        [(pa, 0.0560, 0.0190, 0.5440, 0.5495) for pa in act_phis]
        + [(rvdt_phi_rad, 0.0240, 0.0120, 0.5440, 0.5520)]
    )
    bm_add_notched_revolved_profile(bm_case, prof_flare_lip, notches_flare_lip, bevel_r=0.00042, base_segs=256)

    flare_lip_bolts = []
    for ifb in range(72):
        ph_fb = 2.0 * math.pi * (ifb + 0.5) / 72.0
        if abs(_ang_diff(ph_fb, rvdt_phi_rad)) < 0.055 or min(_ang_diff(ph_fb, pa) for pa in act_phis) < 0.105:
            continue
        p_flb = Vector((0.5965 * math.sin(ph_fb), -0.7255, 0.5965 * math.cos(ph_fb)))
        flare_lip_bolts.append(p_flb)
        bm_add_aerospace_bolt(bm_case, p_flb, Vector((0.0, -0.96, 0.28)).normalized(), head_r=0.0028, height=0.0028)
    for ifb in range(0, len(flare_lip_bolts) - 1, 2):
        if (flare_lip_bolts[ifb] - flare_lip_bolts[ifb + 1]).length < 0.062:
            bm_add_bolt_pair_lockwire(bm_case, flare_lip_bolts[ifb], flare_lip_bolts[ifb + 1],
                                      Vector((0.0, -0.96, 0.28)).normalized(), head_r=0.0028, strand_r=0.00032)


    b_fl_rib = bmesh.new()
    bm_add_clevis_plate(b_fl_rib, (-0.840, 0.5445), (-0.722, 0.598), 0.0034, 0.0048, -0.0020, +0.0020, arc_segs=11)
    for ir_f in range(24):
        ph_fr = 2.0 * math.pi * (ir_f + 0.5) / 24.0
        if abs(_ang_diff(ph_fr, rvdt_phi_rad)) < 0.08 or min(_ang_diff(ph_fr, pa) for pa in act_phis) < 0.11:
            continue
        bm_merge(bm_case, b_fl_rib.copy(), Matrix.Rotation(ph_fr, 4, 'Y'))
    b_fl_rib.free()

    for ir_d in range(24):
        ph_d0 = 2.0 * math.pi * ir_d / 24.0
        ph_dm = 2.0 * math.pi * (ir_d + 0.5) / 24.0
        ph_d1 = 2.0 * math.pi * (ir_d + 1.0) / 24.0
        p_da = Vector((0.5445 * math.sin(ph_d0), -0.842, 0.5445 * math.cos(ph_d0)))
        p_db = Vector((0.5448 * math.sin(ph_dm), -0.748, 0.5448 * math.cos(ph_dm)))
        p_dc = Vector((0.5445 * math.sin(ph_d1), -0.842, 0.5445 * math.cos(ph_d1)))
        for p0_d, p1_d in ((p_da, p_db), (p_dc, p_db)):
            pm_d = 0.5 * (p0_d + p1_d)
            n_d = Vector((pm_d.x, 0.0, pm_d.z)).normalized()
            u_d = n_d.cross(p1_d - p0_d).normalized()
            bm_add_hat_stiffener_rib(
                bm_case, p0_d, p1_d, u_d, n_d,
                w_foot=0.0040, w_top=0.0016, h_hat=0.0018, n_steps=6, rivet_r=0.0,
            )


    build_waffle_cased_cylinder(
        bm_case,
        y0=-0.716, y1=-0.256,
        r_inner=0.524, r_outer=0.548,
        n_phi=320, n_y=40,
        pocket_bands_y=[(-0.695, -0.515), (-0.450, -0.275)],
        n_pockets_phi=32, pocket_depth=0.0034,
    )
    prof_rod_brg_ring = [
        (-0.501, 0.546), (-0.497, 0.556), (-0.473, 0.556), (-0.469, 0.546),
    ]
    bm_add_revolved_profile(bm_case, prof_rod_brg_ring, segs=256, close_loop=True, bevel_r=0.00040)
    for k_rb in range(spec.n_actuators):
        b_rbr = bmesh.new()
        for sx_rb in (-0.022, +0.022):
            bm_add_clevis_plate(b_rbr, (-0.495, 0.553), (-0.475, spec.r_sync_ring), 0.006, 0.007, sx_rb - 0.0030, sx_rb + 0.0030, arc_segs=11)
        prof_bush = [
            (-0.501, 0.0168), (-0.501, 0.0232), (-0.469, 0.0232), (-0.469, 0.0168),
        ]
        b_bsh = bmesh.new()
        bm_add_revolved_profile(b_bsh, prof_bush, segs=28, close_loop=True, bevel_r=0.00035)
        bm_merge(b_rbr, b_bsh, Matrix.Translation((0.0, 0.0, spec.r_sync_ring)))
        bm_merge(bm_case, b_rbr, Matrix.Rotation(act_phis[k_rb], 4, 'Y'))


    y_case_grid = sorted(set(
        [-0.256 + 0.123 * (k / 24) for k in range(25)]
        + [-0.133 + 0.147 * (k / 56) for k in range(57)]
    ))
    build_waffle_cased_cylinder(
        bm_case,
        y0=-0.256, y1=0.014,
        r_inner=0.525, r_outer=0.550,
        n_phi=640, n_y=len(y_case_grid) - 1,
        pocket_bands_y=[(-0.244, -0.132)],
        n_pockets_phi=32, pocket_depth=0.0034,
        radial_cap_fn=_case_radial_cap, y_grid_custom=y_case_grid,
    )


    prof_ring3 = [
        (-0.128, 0.547), (-0.122, 0.570), (-0.119, 0.573),
        (-0.106, 0.573), (-0.103, 0.570), (-0.097, 0.547),
    ]
    dphi_sl = math.atan2(fh_slot, 0.558)
    notches_ring3 = []
    for ps in sec_phis:
        notches_ring3.append((ps - dphi_sl, 0.0092, 0.0028, 0.5462, 0.5504))
        notches_ring3.append((ps + dphi_sl, 0.0092, 0.0028, 0.5462, 0.5504))
    bm_add_notched_revolved_profile(bm_case, prof_ring3, notches_ring3, bevel_r=0.00045, base_segs=256)


    prof_aft_lip = [
        (0.0025, 0.548), (0.012, 0.550), (0.017, 0.536), (0.012, 0.525),
    ]
    bm_add_revolved_profile(bm_case, prof_aft_lip, segs=256, close_loop=True, bevel_r=0.00040)
    create_mesh_obj("Static_Base_Casing_Ribbed", bm_case, coll_static, clay)


    bm_fast = bmesh.new()
    n_main_bolts = 112
    main_bolt_pts = []
    for i in range(n_main_bolts):
        phi = 2.0 * math.pi * i / n_main_bolts
        sp, cp = math.sin(phi), math.cos(phi)
        r_b = 0.556
        p_b = Vector((r_b * sp, -1.867, r_b * cp))
        main_bolt_pts.append(p_b)
        bm_add_aerospace_bolt(
            bm_fast,
            pos=p_b,
            normal=Vector((0.0, 1.0, 0.0)),
            head_r=0.0044, height=0.0048,
            with_12pt_nut_depth=0.024,
        )
    for i in range(0, n_main_bolts, 2):
        bm_add_bolt_pair_lockwire(
            bm_fast, main_bolt_pts[i], main_bolt_pts[i + 1],
            normal=Vector((0.0, 1.0, 0.0)), head_r=0.0044, strand_r=0.00046,
        )

    mid_bolt_pts = []
    for i in range(64):
        phi = 2.0 * math.pi * i / 64
        if min(_ang_diff(phi, pa) for pa in act_phis) * 0.548 < 0.042:
            continue
        rad_v = Vector((math.sin(phi), 0.0, math.cos(phi)))
        p_mb = rad_v * 0.5485 + Vector((0.0, -0.851, 0.0))
        mid_bolt_pts.append((p_mb, rad_v))
        bm_add_aerospace_bolt(
            bm_fast,
            pos=p_mb,
            normal=rad_v, head_r=0.0034, height=0.0032,
        )
    for i in range(0, len(mid_bolt_pts) - 1, 2):
        if (mid_bolt_pts[i][0] - mid_bolt_pts[i + 1][0]).length < 0.095:
            n_avg = (mid_bolt_pts[i][1] + mid_bolt_pts[i + 1][1]).normalized()
            bm_add_bolt_pair_lockwire(
                bm_fast, mid_bolt_pts[i][0], mid_bolt_pts[i + 1][0],
                normal=n_avg, head_r=0.0034, strand_r=0.00040,
            )

    rivet_rings = [
        (-1.874, 0.5680), (-0.834, 0.5446), (-0.700, 0.548),
        (-0.136, 0.550), (-0.088, 0.550),
    ]
    for y_rv, r_rv in rivet_rings:
        for i in range(144):
            phi = 2.0 * math.pi * i / 144
            if y_rv == -0.136:
                if min(_ang_diff(phi, ps) for ps in sec_phis) * 0.550 < 0.017:
                    continue
            elif y_rv == -0.088:
                if min(_ang_diff(phi, ps) for ps in sec_phis) * 0.550 < 0.054:
                    continue
            rad_v = Vector((math.sin(phi), 0.0, math.cos(phi)))
            bm_add_dome_rivet(bm_fast, rad_v * r_rv + Vector((0.0, y_rv, 0.0)), rad_v, radius=0.0024)

    t_bj_u = subdivide_1d_with_support_loops(6, sup1=0.04, sup2=0.14)
    t_bj_s = subdivide_1d_with_support_loops(12, sup1=0.02, sup2=0.07)
    for y_ja, r_ja, y_jb, r_jb, r_jpeak, bj_angles in [
        (-1.912, 0.5445, -1.848, 0.5432, 0.580, (22.5, 112.5, 157.5, 202.5, 247.5, 292.5)),
        (-2.277, 0.5415, -2.213, 0.5425, 0.578, (15.0, 75.0, 165.0, 195.0, 285.0, 345.0)),
    ]:
        b_bj_unit = bmesh.new()
        bj_bpts = []
        for y_lug, r_lug in ((y_ja, r_ja), (y_jb, r_jb)):
            bm_add_clevis_plate(b_bj_unit, (y_lug - 0.0050, r_lug + 0.0008), (y_lug + 0.0050, r_lug + 0.0008), 0.0020, 0.0020, -0.0058, +0.0058, arc_segs=9)
            p_bjb = Vector((0.0, y_lug, r_lug + 0.0024))
            bj_bpts.append(p_bjb)
            bm_add_aerospace_bolt(b_bj_unit, p_bjb, Vector((0.0, 0.0, 1.0)), head_r=0.0023, height=0.0021)
            bm_add_aerospace_bolt(b_bj_unit, Vector((+0.0095, y_lug, r_lug + 0.0014)), Vector((0.0, 0.0, 1.0)), head_r=0.0015, height=0.0014)
            bm_add_bolt_pair_lockwire(b_bj_unit, p_bjb, Vector((+0.0095, y_lug, r_lug + 0.0014)), Vector((0.0, 0.0, 1.0)), head_r=0.0018, strand_r=0.00024)
        g_bj_b, g_bj_t = [], []
        for fs_j in t_bj_s:
            yj = y_ja + (y_jb - y_ja) * fs_j
            rj_base = r_ja + (r_jb - r_ja) * fs_j
            arch_h = (r_jpeak - 0.5 * (r_ja + r_jb)) * (math.sin(math.pi * fs_j) ** 0.78)
            rz_c = rj_base + 0.0014 + arch_h
            rb_j, rt_j = [], []
            for fu_j in t_bj_u:
                xj = (-0.5 + fu_j) * 0.0092
                braid_rip = 0.00028 * math.cos(2.0 * math.pi * 5.0 * fu_j + math.pi * 6.0 * fs_j)
                rb_j.append(b_bj_unit.verts.new((xj, yj, rz_c - 0.0008)))
                rt_j.append(b_bj_unit.verts.new((xj, yj, rz_c + 0.0009 + braid_rip)))
            g_bj_b.append(rb_j)
            g_bj_t.append(rt_j)
        bm_build_beveled_quad_shell(b_bj_unit, g_bj_b, g_bj_t, bulge_scale=0.16)
        for ang_bj in bj_angles:
            bm_merge(bm_fast, b_bj_unit.copy(), Matrix.Rotation(math.radians(ang_bj), 4, 'Y'))
        b_bj_unit.free()


    b_vb_seg = bmesh.new()
    _bm_add_box_x_yz(b_vb_seg, -0.0125, +0.0125, -1.891, -1.869, 0.564, 0.5755, nx=2, ny=2, nz=2, sup=0.12)
    bm_add_aerospace_bolt(b_vb_seg, Vector((0.0, -1.880, 0.5755)), Vector((0, 0, 1)), head_r=0.0024, height=0.0022)
    bm_add_aerospace_bolt(b_vb_seg, Vector((0.024, -1.872, 0.5615)), Vector((0, 1, 0)), head_r=0.0032, height=0.0034, with_12pt_nut_depth=0.018)
    for kvb in range(72):
        deg_vb = kvb * 5.0 + 2.5
        if any(abs((deg_vb - dt_l + 180.0) % 360.0 - 180.0) < 4.2 for dt_l in (42.0, 138.0, 222.0, 318.0)):
            continue
        bm_merge(bm_fast, b_vb_seg.copy(), Matrix.Rotation(math.radians(deg_vb), 4, 'Y'))
    b_vb_seg.free()

    for dt_lat in (42.0, 138.0, 222.0, 318.0):
        b_vbl = bmesh.new()
        for sx_vl in (-0.018, +0.018):
            bm_add_clevis_plate(b_vbl, (-1.892, 0.562), (-1.868, 0.580), 0.007, 0.008, sx_vl - 0.0035, sx_vl + 0.0035, arc_segs=11)
            bm_add_chamfered_cyl_x(b_vbl, -1.880, 0.576, sx_vl - 0.0055, sx_vl + 0.0055, radius=0.0068, segs=18)
        bm_add_chamfered_cyl_x(b_vbl, -1.880, 0.576, -0.028, +0.021, radius=0.0046, segs=18)
        bm_add_bevel_locked_hex(b_vbl, Vector((-0.032, -1.880, 0.576)), Vector((-0.024, -1.880, 0.576)),
                                [(0.0, 0.0060), (0.25, 0.0074), (0.80, 0.0074), (1.0, 0.0058)], corner_bevel=0.14, cap_ends=True)
        bm_add_helical_thread_cylinder(
            b_vbl, Vector((+0.019, -1.880, 0.576)), Vector((+0.044, -1.880, 0.576)),
            r_core=0.0034, thread_h=0.00080, n_turns=8, segs_turn=16, cap_core=True,
        )
        bm_add_bevel_locked_hex(b_vbl, Vector((+0.022, -1.880, 0.576)), Vector((+0.030, -1.880, 0.576)),
                                [(0.0, 0.0062), (0.25, 0.0076), (0.80, 0.0076), (1.0, 0.0058)], corner_bevel=0.14, cap_ends=True)
        bm_add_bevel_locked_hex(b_vbl, Vector((+0.031, -1.880, 0.576)), Vector((+0.037, -1.880, 0.576)),
                                [(0.0, 0.0058), (0.25, 0.0070), (0.80, 0.0070), (1.0, 0.0054)], corner_bevel=0.14, cap_ends=True)
        bm_add_twisted_lockwire(
            b_vbl,
            [Vector((+0.026, -1.880, 0.583)), Vector((+0.034, -1.880, 0.583)), Vector((+0.018, -1.889, 0.582))],
            strand_r=0.00032, twists_per_m=240.0, n_samples=16, segs=10,
        )
        bm_merge(bm_fast, b_vbl, Matrix.Rotation(math.radians(dt_lat), 4, 'Y'))

    create_mesh_obj("Static_Aerospace_Bolts_And_Rivets", bm_fast, coll_static, clay)


    bm_brkt = bmesh.new()
    fh = spec.fork_half_span
    bs_unit = bmesh.new()

    for side_s in (-1.0, 1.0):
        x_p0, x_p1 = side_s * 0.0360, side_s * 0.0505
        bm_add_clevis_plate(bs_unit, (-0.092, 0.546), (0.030, 0.546), 0.0062, 0.0062, x_p0, x_p1, arc_segs=11)
        bx = side_s * 0.0442
        p_bf = Vector((bx, -0.078, 0.5515))
        p_ba = Vector((bx, +0.016, 0.5515))
        bm_add_aerospace_bolt(bs_unit, p_bf, Vector((0, 0, 1)), head_r=0.0038, height=0.0036)
        bm_add_aerospace_bolt(bs_unit, p_ba, Vector((0, 0, 1)), head_r=0.0038, height=0.0036)
        bm_add_bolt_pair_lockwire(bs_unit, p_bf, p_ba, Vector((0, 0, 1)), head_r=0.0038, strand_r=0.00044)


    bm_add_clevis_plate(bs_unit, (0.003, 0.5465), (0.030, 0.5465), 0.0050, 0.0050, -0.0120, +0.0120, arc_segs=11)
    p_csb0 = Vector((-0.0074, 0.025, 0.5512))
    p_csb1 = Vector((+0.0074, 0.025, 0.5512))
    bm_add_aerospace_bolt(bs_unit, p_csb0, Vector((0, 0, 1)), head_r=0.0026, height=0.0025)
    bm_add_aerospace_bolt(bs_unit, p_csb1, Vector((0, 0, 1)), head_r=0.0026, height=0.0025)
    bm_add_bolt_pair_lockwire(bs_unit, p_csb0, p_csb1, Vector((0, 0, 1)), head_r=0.0026, strand_r=0.00034)

    for side_s in (-1.0, 1.0):

        xc = side_s * (fh + 0.0166)

        bm_add_clevis_plate(bs_unit, (-0.084, 0.545), (spec.pb_y, spec.pb_r), 0.0145, 0.0155, xc - 0.0030, xc + 0.0030, arc_segs=23)
        bm_add_clevis_plate(bs_unit, ( 0.004, 0.545), (spec.pb_y, spec.pb_r), 0.0125, 0.0145, xc - 0.0028, xc + 0.0028, arc_segs=21)
        bm_add_clevis_plate(bs_unit, (-0.084, 0.551), ( 0.004, 0.551), 0.0065, 0.0065, xc - 0.0028, xc + 0.0028, arc_segs=11)

        bm_add_clevis_plate(bs_unit, (-0.066, 0.556), (spec.pb_y, spec.pb_r - 0.016), 0.018, 0.015, xc - 0.00135, xc + 0.00135, arc_segs=15)
        bm_add_clevis_plate(bs_unit, (-0.014, 0.556), (spec.pb_y, spec.pb_r - 0.016), 0.016, 0.015, xc - 0.00135, xc + 0.00135, arc_segs=15)

        bm_add_clevis_plate(bs_unit, (-0.040, 0.547), (spec.pb_y, spec.pb_r - 0.018), 0.012, 0.008, xc + side_s * 0.0012, xc + side_s * 0.0062, arc_segs=13)
        for y_rib_s in (-0.064, -0.014):
            bm_add_clevis_plate(bs_unit, (y_rib_s, 0.552), (-0.040, 0.588), 0.0038, 0.0036, xc - 0.0024, xc + 0.0024, arc_segs=11)
            y_hb_s = 0.5 * (y_rib_s - 0.040)
            bm_add_chamfered_cyl_x(
                bs_unit, y_hb_s, 0.576,
                xc + side_s * 0.0011, xc + side_s * 0.0027,
                radius=0.0064, segs=22, chamfer=0.00035, hollow_r=0.0038,
            )

        bm_add_chamfered_cyl_x(bs_unit, spec.pb_y, spec.pb_r, xc - side_s * 0.0072, xc + side_s * 0.0046, radius=0.0108, segs=32, hollow_r=0.0060)
        bm_add_grease_zerk(bs_unit, Vector((xc, spec.pb_y, spec.pb_r + 0.0106)), Vector((0.0, 0.0, 1.0)), scale=0.95)


    for side_s in (-1.0, 1.0):
        xc = side_s * 0.0077
        bm_add_clevis_plate(bs_unit, (0.006, 0.546), (spec.ps_y, spec.ps_r), 0.0080, 0.0118, xc - 0.0020, xc + 0.0020, arc_segs=21)
        bm_add_chamfered_cyl_x(bs_unit, spec.ps_y, spec.ps_r, xc - 0.0021, xc + 0.0022, radius=0.0088, segs=32, hollow_r=0.0052)


    bm_add_clevis_pin_assembly_x(bs_unit, spec.pb_y, spec.pb_r, -(fh + 0.0216), +(fh + 0.0216), pin_r=0.0056, segs=28)
    bm_add_clevis_pin_assembly_x(bs_unit, spec.ps_y, spec.ps_r, -0.0100, +0.0100, pin_r=0.0050, segs=26, with_cotter=False, compact_ends=True)

    th1_mid = 0.5 * (spec.state_closed["theta1"] + spec.state_open["theta1"])
    p0_stat = (spec.p0_y - math.sin(th1_mid) * 0.0076, spec.p0_r + math.cos(th1_mid) * 0.0076)
    for x_st in (-0.033, -0.011, +0.011, +0.033):
        bm_add_clevis_plate(bs_unit, (0.0055, 0.545), p0_stat, 0.0048, 0.0052, x_st - 0.0038, x_st + 0.0038, arc_segs=15)
        bm_add_clevis_plate(bs_unit, (0.0020, 0.547), p0_stat, 0.0028, 0.0042, x_st - 0.0014, x_st + 0.0014, arc_segs=11)
        bm_add_chamfered_cyl_x(bs_unit, p0_stat[0], p0_stat[1], x_st - 0.0045, x_st + 0.0045, radius=0.0054, segs=24, hollow_r=0.0033)
    for x_p0w in (-0.0385, -0.0275, -0.0165, -0.0055, +0.0055, +0.0165, +0.0275, +0.0385):
        bm_add_chamfered_cyl_x(bs_unit, p0_stat[0], p0_stat[1], x_p0w - 0.0006, x_p0w + 0.0006, radius=0.0059, segs=24, chamfer=0.00018, hollow_r=0.0034)
    bm_add_clevis_pin_assembly_x(bs_unit, p0_stat[0], p0_stat[1], -0.051, +0.051, pin_r=0.0033, segs=24)


    bm_add_clevis_plate(bs_unit, (-0.251, 0.5530), (-0.138, 0.5530), 0.0042, 0.0042, -0.0055, +0.0055, arc_segs=11)
    for y_rl_pad in (-0.244, -0.146):
        bm_add_clevis_plate(bs_unit, (y_rl_pad - 0.006, 0.5505), (y_rl_pad + 0.006, 0.5505), 0.0030, 0.0030, -0.0135, +0.0135, arc_segs=9)
        p_rlb0 = Vector((-0.0098, y_rl_pad, 0.5532))
        p_rlb1 = Vector((+0.0098, y_rl_pad, 0.5532))
        bm_add_aerospace_bolt(bs_unit, p_rlb0, Vector((0, 0, 1)), head_r=0.0022, height=0.0022)
        bm_add_aerospace_bolt(bs_unit, p_rlb1, Vector((0, 0, 1)), head_r=0.0022, height=0.0022)
        bm_add_bolt_pair_lockwire(bs_unit, p_rlb0, p_rlb1, Vector((0, 0, 1)), head_r=0.0022, strand_r=0.00032)


    xg_c, yg_c, zg_c = +0.0510, -0.014, 0.588
    bm_add_clevis_plate(bs_unit, (yg_c - 0.0075, zg_c - 0.0035), (yg_c + 0.0055, zg_c + 0.0035), 0.0058, 0.0058, +0.0396, xg_c - 0.0036, arc_segs=13)
    for sub_g in (-0.0052, +0.0052):
        bm_add_clevis_plate(bs_unit, (yg_c - 0.0080, zg_c - 0.0030), (yg_c, zg_c), 0.0058, 0.0064, xg_c + sub_g - 0.0018, xg_c + sub_g + 0.0018, arc_segs=15)
        bm_add_chamfered_cyl_x(bs_unit, yg_c, zg_c, xg_c + sub_g - 0.0021, xg_c + sub_g + 0.0021, radius=0.0060, segs=22, hollow_r=0.0034)
    bm_add_clevis_pin_assembly_x(bs_unit, yg_c, zg_c, xg_c - 0.0078, xg_c + 0.0078, pin_r=0.0034, segs=20, compact_ends=True)

    for i in range(spec.n_sectors):
        b_copy = bs_unit.copy()
        bm_merge(bm_brkt, b_copy, Matrix.Rotation(2.0 * math.pi * i / spec.n_sectors, 4, 'Y'))
    bs_unit.free()
    create_mesh_obj("Static_Stanchion_Trunnion_Assemblies", bm_brkt, coll_static, clay)


    bm_act = bmesh.new()
    r_act = spec.r_sync_ring
    ba_unit = bmesh.new()
    y_trun = -0.916

    prof_act_hollow = [
        (-1.095, 0.0152), (-1.095, 0.0262), (-1.082, 0.0278), (-1.048, 0.0278), (-1.042, 0.0242),
        (-0.932, 0.0242), (-0.928, 0.0276), (-0.904, 0.0276), (-0.900, 0.0242),
        (-0.774, 0.0242), (-0.770, 0.0254), (-0.764, 0.0254), (-0.762, 0.0210),
    ]
    for ibw in range(7):
        yb0 = -0.761 + ibw * 0.0033
        prof_act_hollow.extend([
            (yb0,          0.0166),
            (yb0 + 0.0008, 0.0212),
            (yb0 + 0.0018, 0.0212),
            (yb0 + 0.0026, 0.0166),
        ])

    prof_act_hollow.extend([
        (-0.7375, 0.0174), (-0.7345, 0.0248), (-0.7295, 0.0264),
        (-0.7250, 0.0236), (-0.7170, 0.0228), (-0.7170, 0.0152),
    ])
    b_acyl = bmesh.new()
    bm_add_revolved_profile(b_acyl, prof_act_hollow, segs=48, close_loop=True, bevel_r=0.00022)
    bm_merge(ba_unit, b_acyl, Matrix.Translation((0.0, 0.0, r_act)))
    bm_add_segmented_tube(
        ba_unit, Vector((0.0, -1.104, r_act)), Vector((0.0, -1.095, r_act)),
        [(0.0, 0.020), (0.35, 0.0262), (1.0, 0.0262)], segs=48, cap_ends=True,
    )

    bm_add_clevis_plate(ba_unit, (-0.826, r_act + 0.020), (-0.806, r_act + 0.020), 0.0032, 0.0032, +0.020, +0.028, arc_segs=9)
    bm_add_aerospace_bolt(ba_unit, Vector((+0.026, -0.816, r_act + 0.023)), Vector((0, 0, 1)), head_r=0.0018, height=0.0018)

    for side_ar in (-1.0, +1.0):
        x_tr_eye = side_ar * 0.0365
        x_ft_eye = side_ar * 0.0435
        p_tr_pin = Vector((x_tr_eye, y_trun, r_act))
        p_ft_pin = Vector((x_ft_eye, -0.732, 0.5605))
        bm_add_chamfered_cyl_x(
            ba_unit, y_trun, r_act,
            side_ar * 0.0276, side_ar * 0.0325, radius=0.0092, segs=24, hollow_r=0.0048,
        )
        bm_add_clevis_pin_assembly_x(
            ba_unit, y_trun, r_act,
            side_ar * 0.0290, side_ar * 0.0425,
            pin_r=0.0048, segs=22, compact_ends=True, flip_x=(side_ar < 0.0),
        )
        bm_add_flattened_aero_strut(
            ba_unit, p_tr_pin, p_ft_pin,
            half_w_mid=0.0072, half_h_mid=0.0040,
            end_half_w=0.0036, end_half_h=0.0036,
            eye_r=0.0086, eye_half_w=0.0038, taper_frac=0.18,
            pin_along_x=True, with_flutes=True, ribbed_pockets=3,
        )
        x_pad_in, x_pad_out = side_ar * 0.0335, side_ar * 0.0535
        bm_add_clevis_plate(
            ba_unit, (-0.746, 0.5522), (-0.721, 0.5522), 0.0038, 0.0038,
            min(x_pad_in, x_pad_out), max(x_pad_in, x_pad_out), arc_segs=11,
        )
        for sub_af in (-0.0056, +0.0056):
            bm_add_clevis_plate(
                ba_unit, (-0.743, 0.5540), (-0.732, 0.5605), 0.0062, 0.0074,
                x_ft_eye + sub_af - 0.0018, x_ft_eye + sub_af + 0.0018, arc_segs=15,
            )
        bm_add_clevis_pin_assembly_x(
            ba_unit, -0.732, 0.5605, x_ft_eye - 0.0080, x_ft_eye + 0.0080,
            pin_r=0.0038, segs=20, compact_ends=True, flip_x=(side_ar < 0.0),
        )

        ft_bpts = []
        for dx_fb_a in (-0.0072, +0.0072):
            for y_fb_a in (-0.741, -0.725):
                p_fb_i = Vector((x_ft_eye + dx_fb_a, y_fb_a, 0.5558))
                ft_bpts.append(p_fb_i)
                bm_add_aerospace_bolt(ba_unit, p_fb_i, Vector((0, 0, 1)), head_r=0.0019, height=0.0019)
        bm_add_bolt_pair_lockwire(ba_unit, ft_bpts[0], ft_bpts[1], Vector((0, 0, 1)), head_r=0.0019, strand_r=0.00028)
        bm_add_bolt_pair_lockwire(ba_unit, ft_bpts[2], ft_bpts[3], Vector((0, 0, 1)), head_r=0.0019, strand_r=0.00028)


    gb_fl_pts = []
    for ang_deg_gb in (-32.0, +32.0, +82.0, +136.0, -136.0, -82.0):
        ang_gb = math.radians(ang_deg_gb)
        gx, gz = 0.0258 * math.sin(ang_gb), r_act + 0.0258 * math.cos(ang_gb)
        bm_add_segmented_tube(
            ba_unit, Vector((gx, -1.064, gz)), Vector((gx, -1.048, gz)),
            [(0.0, 0.0040), (0.25, 0.0052), (0.85, 0.0052), (1.0, 0.0044)], segs=16, cap_ends=True,
        )
        pgb = Vector((gx, -1.048, gz))
        gb_fl_pts.append(pgb)
        bm_add_aerospace_bolt(ba_unit, pgb, Vector((0, 1, 0)), head_r=0.0026, height=0.0025)
    for kb_g in (0, 2, 4):
        bm_add_bolt_pair_lockwire(ba_unit, gb_fl_pts[kb_g], gb_fl_pts[kb_g + 1], Vector((0, 1, 0)), head_r=0.0026, strand_r=0.00034)


    def _add_act_conformal_band(ya_b: float, yb_b: float, ang_deg_0: float, ang_deg_1: float, r_in_b: float, r_out_b: float, np_b: int = 12, ny_b: int = 4):
        tp_b = subdivide_1d_with_support_loops(np_b, sup1=0.03, sup2=0.10)
        ty_b = subdivide_1d_with_support_loops(ny_b, sup1=0.04, sup2=0.14)
        g_bi, g_bo = [], []
        for fp_i in tp_b:
            ab = math.radians(ang_deg_0 + (ang_deg_1 - ang_deg_0) * fp_i)
            sb_i, cb_i = math.sin(ab), math.cos(ab)
            ci_i, co_i = [], []
            for fy_i in ty_b:
                yv_i = ya_b + (yb_b - ya_b) * fy_i
                ci_i.append(ba_unit.verts.new((r_in_b * sb_i, yv_i, r_act + r_in_b * cb_i)))
                co_i.append(ba_unit.verts.new((r_out_b * sb_i, yv_i, r_act + r_out_b * cb_i)))
            g_bi.append(ci_i)
            g_bo.append(co_i)
        bm_build_beveled_quad_shell(ba_unit, g_bi, g_bo, bulge_scale=0.10)


    for ya_pl_a, yb_pl_a, ad_span in [
        (-1.038, -1.004, 34.0),
        (-0.998, -0.938, 46.0),
    ]:
        f0_act_pl = len(ba_unit.faces)
        _add_act_conformal_band(ya_pl_a, yb_pl_a, -ad_span, +ad_span, 0.0241, 0.0251, np_b=14, ny_b=5)
        bm_fill_unset_from(ba_unit, f0_act_pl, MAT_PLATE_BASE)
        for s_ang_r in (-ad_span + 6.0, +ad_span - 6.0):
            ar_rv = math.radians(s_ang_r)
            n_rv = Vector((math.sin(ar_rv), 0.0, math.cos(ar_rv)))
            for sy_pa in (ya_pl_a + 0.004, yb_pl_a - 0.004):
                bm_add_dome_rivet(ba_unit, Vector((0.0251 * n_rv.x, sy_pa, r_act + 0.0251 * n_rv.z)), n_rv, radius=0.00095)

    _add_act_conformal_band(-0.980, -0.954, +44.0, +58.0, 0.0239, 0.0258, np_b=6, ny_b=3)


    for side_ar in (-1.0, +1.0):
        bm_add_chamfered_cyl_x(ba_unit, y_trun, r_act, side_ar * 0.0396, side_ar * 0.0414, radius=0.0112, segs=24, hollow_r=0.0050)
        b_star_w = bmesh.new()
        for k_st in range(6):
            b_lobe = bmesh.new()
            _bm_add_box_x_yz(b_lobe, -0.0008, +0.0008, -0.0028, +0.0028, 0.0086, 0.0148, nx=1, ny=1, nz=1, chamfer=0.0003)
            bm_merge(b_star_w, b_lobe, Matrix.Rotation(k_st * math.pi / 3.0, 4, 'X'))
        bm_merge(ba_unit, b_star_w, Matrix.Translation((side_ar * 0.0405, y_trun, r_act)))
        bm_add_clevis_plate(ba_unit, (-0.862, 0.548), (-0.735, 0.556), 0.0046, 0.0126,
                            min(side_ar * 0.0385, side_ar * 0.0428), max(side_ar * 0.0385, side_ar * 0.0428), arc_segs=9)

    _bm_add_box_x_yz(ba_unit, -0.047, -0.032, -0.842, -0.806, 0.5444, 0.5568, nx=2, ny=2, nz=1, chamfer=0.0008)
    bm_add_aerospace_bolt(ba_unit, Vector((-0.047, -0.824, 0.5515)), Vector((-1, 0, 0)), head_r=0.0028, height=0.0026, with_12pt_nut_depth=0.012)

    for y_crd, r_foot in [(-1.068, 0.5442), (y_trun, 0.5445)]:
        bm_add_clevis_plate(ba_unit, (y_crd, r_foot), (y_crd, r_act), 0.013, 0.025, -0.014, +0.014, arc_segs=19)
        bm_add_clevis_plate(ba_unit, (y_crd - 0.011, r_foot + 0.001), (y_crd + 0.011, r_foot + 0.001), 0.0034, 0.0034, -0.026, +0.026, arc_segs=9)
        for sx_cf in (-0.020, +0.020):
            p_cf0 = Vector((sx_cf, y_crd - 0.0070, r_foot + 0.0036))
            p_cf1 = Vector((sx_cf, y_crd + 0.0070, r_foot + 0.0036))
            bm_add_aerospace_bolt(ba_unit, p_cf0, Vector((0, 0, 1)), head_r=0.0022, height=0.0022)
            bm_add_aerospace_bolt(ba_unit, p_cf1, Vector((0, 0, 1)), head_r=0.0022, height=0.0022)
            bm_add_bolt_pair_lockwire(ba_unit, p_cf0, p_cf1, Vector((0, 0, 1)), head_r=0.0022, strand_r=0.00032)


    b_prim_hdr = bmesh.new()
    _bm_add_box_x_yz(b_prim_hdr, -0.018, +0.018, -1.092, -1.052, r_act + 0.018, r_act + 0.033, nx=2, ny=2, nz=1, chamfer=0.0008)
    for sx_ph in (-0.009, +0.009):
        bm_add_bevel_locked_hex(b_prim_hdr, Vector((sx_ph, -1.092, r_act + 0.026)), Vector((sx_ph, -1.104, r_act + 0.026)),
                                [(0.0, 0.0046), (0.25, 0.0056), (0.80, 0.0056), (1.0, 0.0042)], corner_bevel=0.14, cap_ends=True)
    uloop_pts = [
        Vector((+0.016, -1.072, r_act + 0.025)),
        Vector((+0.030, -1.050, r_act + 0.032)),
        Vector((+0.032, -0.996, r_act + 0.034)),
        Vector((0.000,  -0.968, r_act + 0.035)),
        Vector((-0.032, -0.996, r_act + 0.034)),
        Vector((-0.030, -1.050, r_act + 0.032)),
        Vector((-0.016, -1.072, r_act + 0.025)),
    ]
    ul_fr = bm_add_spline_tube(b_prim_hdr, uloop_pts, radius=0.0031, n_samples=32, segs=16, cap_ends=True)
    bm_add_bnut_flare_fitting(b_prim_hdr, ul_fr[1][0], ul_fr[1][1], tube_r=0.0031)
    bm_add_bnut_flare_fitting(b_prim_hdr, ul_fr[-2][0], ul_fr[-2][1], tube_r=0.0031)
    for k in range(spec.n_actuators):
        phi = 2.0 * math.pi * (2 * k) / spec.n_sectors
        rot_act = Matrix.Rotation(phi, 4, 'Y')
        bm_merge(bm_act, ba_unit.copy(), rot_act)
        if k in (0, 3, 4, 5):
            bm_merge(bm_act, b_prim_hdr.copy(), rot_act)
    ba_unit.free()
    b_prim_hdr.free()


    for wps_jh in [
        [(0.8, -1.102, r_act + 0.026), (8.0, -1.175, r_act + 0.036), (34.0, -1.145, 0.562), (43.5, -1.096, r_act + 0.020)],
        [(136.0, -1.102, r_act + 0.026), (148.0, -1.172, r_act + 0.034), (168.0, -1.172, r_act + 0.034), (179.0, -1.102, r_act + 0.026)],
        [(181.0, -1.102, r_act + 0.026), (192.0, -1.172, r_act + 0.034), (212.0, -1.172, r_act + 0.034), (224.0, -1.102, r_act + 0.026)],
    ]:
        ah_pts = build_cyl_mandrel_pts(wps_jh, bend_r=0.038, r_unwrap=0.575, arc_segs=16, max_step=0.010)
        ah_fr = bm_sweep_centerline_tube(bm_act, ah_pts, radius=0.0044, segs=18, cap_ends=True, corrugations=22, corr_amp=0.09)
        bm_add_bnut_flare_fitting(bm_act, ah_fr[1][0], ah_fr[1][1], tube_r=0.0044)
        bm_add_bnut_flare_fitting(bm_act, ah_fr[-2][0], ah_fr[-2][1], tube_r=0.0044)

    create_mesh_obj("Static_Hydraulic_System_Detailed", bm_act, coll_static, clay)


    bm_flex = bmesh.new()
    y_fs = -1.106
    r_fs = r_act
    bg_unit = bmesh.new()
    bm_add_clevis_plate(bg_unit, (y_fs - 0.011, r_act - 0.008), (-1.094, r_fs + 0.010), 0.012, 0.012, -0.022, +0.022, arc_segs=15)

    bm_add_segmented_tube(
        bg_unit, Vector((0.0, -1.138, r_act)), Vector((0.0, -1.095, r_act)),
        [(0.0, 0.0126), (0.36, 0.0126), (0.40, 0.0218), (0.68, 0.0225), (0.76, 0.0268), (0.88, 0.0268), (1.0, 0.0258)],
        segs=36, cap_ends=True,
    )

    bm_add_chamfered_cyl_x(bg_unit, y_fs, r_fs, -0.032, +0.032, radius=0.0136, segs=28)
    for sx in (-1.0, +1.0):
        p_cp_a = Vector((sx * 0.028, y_fs, r_fs))
        p_cp_b = Vector((sx * 0.074, y_fs, r_fs))
        bm_add_segmented_tube(
            bg_unit, p_cp_a, p_cp_b,
            [
                (0.00, 0.0142), (0.12, 0.0168), (0.24, 0.0168), (0.28, 0.0146),
                (0.36, 0.0162), (0.74, 0.0162), (0.78, 0.0144), (0.84, 0.0166),
                (0.94, 0.0166), (1.00, 0.0138),
            ],
            segs=32, cap_ends=True,
        )

        for ka_sp in range(4):
            asp_c = 0.25 * math.pi + 0.5 * math.pi * ka_sp
            n_sp_r = Vector((0.0, math.cos(asp_c), math.sin(asp_c)))
            p_sp_h = Vector((sx * 0.052, y_fs, r_fs)) + n_sp_r * 0.0156
            bm_add_segmented_tube(bg_unit, p_sp_h - n_sp_r * 0.0012, p_sp_h + n_sp_r * 0.0010,
                                  [(0.0, 0.0024), (0.65, 0.0024), (1.0, 0.0014)], segs=12, cap_ends=True)
        bm_add_bevel_locked_hex(
            bg_unit, Vector((sx * 0.074, y_fs, r_fs)), Vector((sx * 0.088, y_fs, r_fs)),
            [(0.0, 0.0108), (0.20, 0.0132), (0.80, 0.0132), (1.0, 0.0098)], corner_bevel=0.14, cap_ends=True,
        )
        p_lw_a = Vector((sx * 0.033, y_fs - 0.004, r_fs + 0.0166))
        p_lw_b = Vector((sx * 0.072, y_fs + 0.004, r_fs + 0.0164))
        bm_add_aerospace_bolt(bg_unit, p_lw_a, Vector((0, 0, 1)), head_r=0.0021, height=0.0019)
        bm_add_aerospace_bolt(bg_unit, p_lw_b, Vector((0, 0, 1)), head_r=0.0019, height=0.0017)
        bm_add_bolt_pair_lockwire(bg_unit, p_lw_a, p_lw_b, Vector((0, 0, 1)), head_r=0.0020, strand_r=0.00034)

    for k in range(spec.n_actuators):
        phi_k = 2.0 * math.pi * (2 * k) / spec.n_sectors
        bm_merge(bm_flex, bg_unit.copy(), Matrix.Rotation(phi_k, 4, 'Y'))

        phi_next = 2.0 * math.pi * (2 * (k + 1)) / spec.n_sectors
        rot_k = Matrix.Rotation(phi_k, 4, 'Y')
        rot_n = Matrix.Rotation(phi_next, 4, 'Y')
        p_start_fs = rot_k @ Vector((+0.086, y_fs, r_fs))
        p_end_fs   = rot_n @ Vector((-0.086, y_fs, r_fs))
        arc_pts = [p_start_fs]
        for s_step in range(1, 14):
            fr = s_step / 14.0
            ang_m = phi_k + 0.148 + (phi_next - 0.148 - (phi_k + 0.148)) * fr
            r_mid = r_fs - 0.004 * math.sin(math.pi * fr)
            arc_pts.append(Vector((r_mid * math.sin(ang_m), y_fs, r_mid * math.cos(ang_m))))
        arc_pts.append(p_end_fs)
        bm_sweep_centerline_tube(bm_flex, arc_pts, radius=0.0068, segs=22, cap_ends=True)

        phi_mid = phi_k + math.pi / 8.0
        r_clamp_fs = r_fs - 0.004
        b_fs_clamp = bmesh.new()
        bm_add_clevis_plate(b_fs_clamp, (y_fs, 0.5444), (y_fs, r_clamp_fs), 0.0072, 0.0092, -0.0062, +0.0062, arc_segs=15)
        bm_add_chamfered_cyl_x(b_fs_clamp, y_fs, r_clamp_fs, -0.0095, +0.0095, radius=0.0092, segs=24, hollow_r=0.0070)
        bm_merge(bm_flex, b_fs_clamp, Matrix.Rotation(phi_mid, 4, 'Y'))
    bg_unit.free()
    create_mesh_obj("Static_Sync_FlexShaft_System", bm_flex, coll_static, clay)

    bm_asym = bmesh.new()

    def _cyl_p(phi_deg: float, yv: float, rv: float) -> Vector:
        ph = math.radians(phi_deg)
        return Vector((rv * math.sin(ph), yv, rv * math.cos(ph)))

    def _cyl_n(phi_deg: float) -> Vector:
        ph = math.radians(phi_deg)
        return Vector((math.sin(ph), 0.0, math.cos(ph)))

    def _r_iso_surf(yv: float) -> float:
        if yv < -1.880:
            return 0.542 + 0.003 * max(0.0, min(1.0, (yv - (-2.245)) / 0.365))
        return 0.5430 + 0.0018 * max(0.0, min(1.0, (yv - (-1.880)) / 1.025))

    def _add_routed_hard_tube(waypoints_deg_yr: list, tube_r: float = 0.0038, clamp_indices: tuple = (), bend_r: float = 0.042):
        pts_3d = build_cyl_mandrel_pts(waypoints_deg_yr, bend_r=bend_r, r_unwrap=0.558, arc_segs=16, max_step=0.012)
        frames = bm_sweep_centerline_tube(bm_asym, pts_3d, radius=tube_r, segs=20, cap_ends=True)
        if len(frames) >= 4:
            bm_add_bnut_flare_fitting(bm_asym, frames[1][0],  frames[1][1],  tube_r=tube_r)
            bm_add_bnut_flare_fitting(bm_asym, frames[-2][0], frames[-2][1], tube_r=tube_r)
        for ci in clamp_indices:
            if 0 <= ci < len(waypoints_deg_yr) and len(frames) >= 2:
                d_c, y_c, r_c = waypoints_deg_yr[ci]
                p_target = _cyl_p(d_c, y_c, r_c)
                best_fr = min(frames, key=lambda fr: (fr[0] - p_target).length_squared)
                p_fr, t_fr = best_fr[0], best_fr[1]
                p_root = _cyl_p(d_c, y_c, _r_iso_surf(y_c) - 0.0008)
                n_cl = _cyl_n(d_c)
                bolt_n = Vector((0.0, 1.0, 0.0)) if abs(t_fr.y) < 0.72 else n_cl.cross(t_fr).normalized()
                bm_add_adel_loop_clamp(bm_asym, p_fr, t_fr, p_root, bolt_n, tube_r=tube_r)

    rot_ncv = Matrix.Rotation(math.radians(148.0), 4, 'Y')
    b_ncv = bmesh.new()
    _bm_add_box_x_yz(b_ncv, -0.056, +0.056, -2.072, -1.898, 0.543, 0.556, nx=3, ny=4, nz=1, chamfer=0.0012)
    _bm_add_box_x_yz(b_ncv, -0.044, +0.044, -2.054, -1.916, 0.554, 0.588, nx=3, ny=4, nz=2, chamfer=0.0012)
    for yf_ncv in (-2.036, -2.002, -1.968, -1.934):
        _bm_add_box_x_yz(b_ncv, -0.048, +0.048, yf_ncv - 0.0022, yf_ncv + 0.0022, 0.556, 0.586, nx=2, ny=1, nz=1, chamfer=0.0004)
    ncv_bpts = []
    for sy_ncb in (-2.062, -2.010, -1.960, -1.908):
        for sx_ncb in (-0.048, +0.048):
            p_ncb = Vector((sx_ncb, sy_ncb, 0.556))
            ncv_bpts.append(p_ncb)
            bm_add_aerospace_bolt(b_ncv, p_ncb, Vector((0, 0, 1)), head_r=0.0022, height=0.0020)
    for k_ncb in range(0, len(ncv_bpts), 2):
        bm_add_bolt_pair_lockwire(b_ncv, ncv_bpts[k_ncb], ncv_bpts[k_ncb + 1], Vector((0, 0, 1)), head_r=0.0022, strand_r=0.00026)
    for y_tm_n, r_tm_n, h_tm_n in ((-2.018, 0.020, 0.620), (-1.952, 0.018, 0.616)):
        bm_add_segmented_tube(
            b_ncv, Vector((0.0, y_tm_n, 0.585)), Vector((0.0, y_tm_n, h_tm_n)),
            [(0.0, r_tm_n * 1.06), (0.20, r_tm_n), (0.82, r_tm_n), (1.0, r_tm_n * 0.65)], segs=24, cap_ends=True,
        )
        bm_add_bevel_locked_hex(
            b_ncv, Vector((0.0, y_tm_n, h_tm_n - 0.001)), Vector((0.0, y_tm_n, h_tm_n + 0.005)),
            [(0.0, 0.0068), (0.30, 0.0082), (0.80, 0.0082), (1.0, 0.0062)], corner_bevel=0.14, cap_ends=True,
        )
        bm_add_mil38999_connector(
            b_ncv, Vector((-0.043, y_tm_n, 0.572)), Vector((-1.0, 0.0, 0.0)),
            shell_r=0.0076, with_flange=False,
            cable_pts=[Vector((-0.068, y_tm_n - 0.016, 0.562)), Vector((-0.078, y_tm_n - 0.042, 0.548))],
            cable_braided=True, n_zipties=1,
        )

    _bm_add_box_x_yz(b_ncv, -0.0462, -0.0438, -2.012, -1.958, 0.558, 0.582, nx=1, ny=2, nz=2, chamfer=0.0004)
    bm_add_3d_plate_text(
        b_ncv, ["NOZ CTRL", "NCV-220"],
        Vector((-0.0462, -1.985, 0.570)),
        Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, -1.0)), Vector((-1.0, 0.0, 0.0)),
        width=0.048, height=0.018, relief=0.00045, with_border=True, with_dividers=True,
    )
    bm_merge(bm_asym, b_ncv, rot_ncv, default_mat_id=MAT_AVIONICS_BOX)


    phi_rvdt = 162.5
    rot_rvdt = Matrix.Rotation(math.radians(phi_rvdt), 4, 'Y')
    b_rvdt = bmesh.new()
    prof_rvdt_hollow = [
        (-1.078, 0.0078), (-1.078, 0.0154), (-1.042, 0.0154), (-1.036, 0.0135),
        (-0.768, 0.0135), (-0.762, 0.0158), (-0.734, 0.0158), (-0.727, 0.0110), (-0.727, 0.0078),
    ]
    b_rvh = bmesh.new()
    bm_add_revolved_profile(b_rvh, prof_rvdt_hollow, segs=28, close_loop=True, bevel_r=0.00025)
    bm_merge(b_rvdt, b_rvh, Matrix.Translation((0.0, 0.0, 0.575)))
    bm_add_segmented_tube(
        b_rvdt, Vector((0.0, -1.088, 0.575)), Vector((0.0, -1.078, 0.575)),
        [(0.0, 0.0102), (1.0, 0.0154)], segs=28, cap_ends=True,
    )
    bm_add_mil38999_connector(
        b_rvdt, Vector((0.0, -1.086, 0.575)), Vector((0.0, -1.0, 0.0)),
        shell_r=0.0086, with_flange=False,
        cable_pts=[Vector((-0.010, -1.125, 0.568)), Vector((-0.018, -1.168, 0.556))],
        cable_braided=True, n_zipties=1,
    )
    bm_add_conformal_cyl_text(
        b_rvdt, ["NOZ RVDT", "P/N 7220"], -42.0, +42.0, -0.985, -0.925,
        r_surf=0.0142, relief=0.00042, center_offset=Vector((0.0, 0.0, 0.575)),
        read_mode='AXIAL_PORT', with_dividers=True,
    )
    for y_sup, r_sup_b in [(-1.028, 0.5443), (-0.886, 0.5446)]:
        bm_add_clevis_plate(b_rvdt, (y_sup, r_sup_b), (y_sup, 0.575), 0.0085, 0.0148, -0.0075, +0.0075, arc_segs=15)
        bm_add_clevis_plate(b_rvdt, (y_sup - 0.008, r_sup_b + 0.001), (y_sup + 0.008, r_sup_b + 0.001), 0.0028, 0.0028, -0.018, +0.018, arc_segs=9)
        for sx_rvf in (-0.013, +0.013):
            p_rvf0 = Vector((sx_rvf, y_sup - 0.005, r_sup_b + 0.0032))
            p_rvf1 = Vector((sx_rvf, y_sup + 0.005, r_sup_b + 0.0032))
            bm_add_aerospace_bolt(b_rvdt, p_rvf0, Vector((0, 0, 1)), head_r=0.0018, height=0.0017)
            bm_add_aerospace_bolt(b_rvdt, p_rvf1, Vector((0, 0, 1)), head_r=0.0018, height=0.0017)
            bm_add_bolt_pair_lockwire(b_rvdt, p_rvf0, p_rvf1, Vector((0, 0, 1)), head_r=0.0018, strand_r=0.00024)
    bm_merge(bm_asym, b_rvdt, rot_rvdt, default_mat_id=MAT_AVIONICS_BOX)
    _add_routed_hard_tube([
        (162.5, -1.110, 0.575), (162.5, -1.240, 0.558), (160.5, -1.560, 0.556), (158.0, -1.875, 0.580), (156.0, -1.950, 0.568),
    ], tube_r=0.0032, clamp_indices=(1, 2))


    rot_zfm = Matrix.Rotation(math.radians(180.0), 4, 'Y')
    b_zfm = bmesh.new()
    _bm_add_box_x_yz(b_zfm, -0.044, +0.044, -1.860, -1.768, 0.543, 0.554, nx=3, ny=3, nz=1, chamfer=0.0010)
    _bm_add_box_x_yz(b_zfm, -0.032, +0.032, -1.848, -1.780, 0.552, 0.575, nx=2, ny=3, nz=2, chamfer=0.0010)
    zfm_bpts = []
    for sy_zfb in (-1.852, -1.814, -1.776):
        for sx_zfb in (-0.037, +0.037):
            p_zfb = Vector((sx_zfb, sy_zfb, 0.554))
            zfm_bpts.append(p_zfb)
            bm_add_aerospace_bolt(b_zfm, p_zfb, Vector((0, 0, 1)), head_r=0.0020, height=0.0018)
    for k_zfb in (0, 2, 4):
        bm_add_bolt_pair_lockwire(b_zfm, zfm_bpts[k_zfb], zfm_bpts[k_zfb + 1], Vector((0, 0, 1)), head_r=0.0020, strand_r=0.00024)
    zfm_cap_pts = []
    for kz in range(5):
        yz_s = -1.840 + kz * 0.013
        xz_s = -0.013 if (kz % 2 == 0) else +0.013
        bm_add_segmented_tube(
            b_zfm, Vector((xz_s, yz_s, 0.573)), Vector((xz_s, yz_s, 0.594)),
            [(0.0, 0.0074), (0.25, 0.0084), (0.78, 0.0082), (1.0, 0.0052)], segs=20, cap_ends=True,
        )
        bm_add_bevel_locked_hex(
            b_zfm, Vector((xz_s, yz_s, 0.592)), Vector((xz_s, yz_s, 0.598)),
            [(0.0, 0.0050), (0.30, 0.0062), (0.80, 0.0062), (1.0, 0.0046)], corner_bevel=0.14, cap_ends=True,
        )
        zfm_cap_pts.append(Vector((xz_s, yz_s, 0.596)))
    bm_add_bolt_pair_lockwire(b_zfm, zfm_cap_pts[0], zfm_cap_pts[2], Vector((0, 0, 1)), head_r=0.0030, strand_r=0.00024)
    bm_add_bolt_pair_lockwire(b_zfm, zfm_cap_pts[1], zfm_cap_pts[3], Vector((0, 0, 1)), head_r=0.0030, strand_r=0.00024)
    bm_add_mil38999_connector(
        b_zfm, Vector((-0.031, -1.814, 0.564)), Vector((-1.0, 0.0, 0.0)),
        shell_r=0.0072, with_flange=False,
        cable_pts=[Vector((-0.056, -1.828, 0.558)), Vector((-0.068, -1.855, 0.548))],
        cable_braided=True, n_zipties=1,
    )
    _bm_add_box_x_yz(b_zfm, +0.0315, +0.0338, -1.842, -1.786, 0.555, 0.573, nx=1, ny=2, nz=2, chamfer=0.0003)
    bm_add_3d_plate_text(
        b_zfm, ["5-ZONE", "MANIFOLD"],
        Vector((+0.0338, -1.814, 0.564)),
        Vector((0.0, -1.0, 0.0)), Vector((0.0, 0.0, -1.0)), Vector((+1.0, 0.0, 0.0)),
        width=0.048, height=0.015, relief=0.00042, with_dividers=True,
    )
    bm_merge(bm_asym, b_zfm, rot_zfm, default_mat_id=MAT_AVIONICS_BOX)

    for phi_fuel, y_inlet in [(152.0, -1.535), (175.0, -1.555), (202.0, -1.525)]:
        rot_f = Matrix.Rotation(math.radians(phi_fuel), 4, 'Y')
        b_fl = bmesh.new()
        r_fi = _r_iso_surf(y_inlet)
        bm_add_clevis_plate(b_fl, (y_inlet - 0.018, r_fi), (y_inlet + 0.018, r_fi), 0.010, 0.010, -0.018, +0.018, arc_segs=13)
        bm_merge(bm_asym, b_fl, rot_f)
        fuel_wps = [
            (phi_fuel - 3.5, -2.228, 0.554),
            (phi_fuel - 3.5, -1.935, 0.558),
            (phi_fuel - 2.0, -1.880, 0.581),
            (phi_fuel,       -1.810, 0.562),
            (phi_fuel,       y_inlet, 0.562),
            (phi_fuel,       y_inlet, r_fi + 0.004),
        ]
        fuel_pts_m = build_cyl_mandrel_pts(fuel_wps, bend_r=0.036, r_unwrap=0.560, arc_segs=16, max_step=0.011)
        f_fr = bm_sweep_centerline_tube(bm_asym, fuel_pts_m, radius=0.0060, segs=20, cap_ends=True, corrugations=18, corr_amp=0.10)
        bm_add_bnut_flare_fitting(bm_asym, f_fr[2][0], f_fr[2][1], tube_r=0.0060)
        bm_add_bnut_flare_fitting(bm_asym, f_fr[-2][0], f_fr[-2][1], tube_r=0.0060)


    for phi_ig, y_ig in [(156.0, -1.438), (204.0, -1.418)]:
        b_ig = bmesh.new()
        bm_add_segmented_tube(
            b_ig, Vector((0.0, y_ig, 0.5435)), Vector((0.0, y_ig - 0.010, 0.575)),
            [(0.0, 0.0115), (0.30, 0.0115), (0.40, 0.0086), (0.85, 0.0100), (1.0, 0.0058)], segs=22, cap_ends=True,
        )
        bm_merge(bm_asym, b_ig, Matrix.Rotation(math.radians(phi_ig), 4, 'Y'))


    for d_vh1, d_vh2, d_act_t in ((217.5, 221.5, 225.0), (142.5, 138.5, 135.0)):
        for idx_vh, (d_cur_vh, d_port_off) in enumerate(((d_vh1, -0.9), (d_vh2, +0.9))):
            _add_routed_hard_tube(
                [
                    (d_cur_vh, -1.985, 0.570),
                    (d_cur_vh, -1.880, 0.581),
                    (d_cur_vh, -1.785, 0.562),
                    (d_cur_vh, -1.165, 0.562),
                    (d_act_t + d_port_off, -1.125, r_act + 0.026),
                    (d_act_t + d_port_off, -1.098, r_act + 0.026),
                ],
                tube_r=0.0066, clamp_indices=(2, 3), bend_r=0.036,
            )
        for y_vcl in (-1.710, -1.500, -1.290):
            b_vhc = bmesh.new()
            _bm_add_box_x_yz(b_vhc, -0.028, +0.028, y_vcl - 0.006, y_vcl + 0.006, 0.5435, 0.570, nx=3, ny=1, nz=2, sup=0.10)
            for sx_vc in (-0.020, 0.0, +0.020):
                bm_add_aerospace_bolt(b_vhc, Vector((sx_vc, y_vcl, 0.570)), Vector((0, 0, 1)), head_r=0.0022, height=0.0020)
            bm_merge(bm_asym, b_vhc, Matrix.Rotation(math.radians(0.5 * (d_vh1 + d_vh2)), 4, 'Y'))


    for kz_spb, (d_spb, y_spb) in enumerate([
        (167.0, -1.725), (173.5, -1.655), (180.0, -1.595), (186.5, -1.655), (193.0, -1.725),
    ]):
        b_spb = bmesh.new()
        r_sp_s = _r_iso_surf(y_spb)
        bm_add_clevis_plate(b_spb, (y_spb - 0.015, r_sp_s), (y_spb + 0.015, r_sp_s), 0.012, 0.012, -0.015, +0.015, arc_segs=11)
        bm_add_segmented_tube(b_spb, Vector((0, y_spb, r_sp_s)), Vector((0, y_spb, r_sp_s + 0.020)),
                              [(0.0, 0.013), (0.45, 0.013), (0.55, 0.0085), (1.0, 0.0075)], segs=20, cap_ends=True)
        bm_merge(bm_asym, b_spb, Matrix.Rotation(math.radians(d_spb), 4, 'Y'))
        _add_routed_hard_tube(
            [(d_spb, -1.915, 0.581), (d_spb, -1.868, 0.581), (d_spb, y_spb, 0.568), (d_spb, y_spb, r_sp_s + 0.016)],
            tube_r=0.0046, clamp_indices=(), bend_r=0.028,
        )

    create_mesh_obj("Static_Asymmetric_Plumbing_And_Sensors", bm_asym, coll_static, clay)

    bm_maint = bmesh.new()
    for phi_ht in (30.0, 330.0):
        rot_ht = Matrix.Rotation(math.radians(phi_ht), 4, 'Y')
        b_ht = bmesh.new()
        r_hb_s = _r_iso_surf(-1.625)
        bm_add_clevis_plate(b_ht, (-1.670, r_hb_s), (-1.580, r_hb_s), 0.009, 0.009, -0.038, +0.038, arc_segs=11)
        for sx_h in (-0.028, +0.028):
            ht_bolts = []
            for y_hb in (-1.658, -1.625, -1.592):
                p_hb = Vector((sx_h, y_hb, r_hb_s + 0.007))
                ht_bolts.append(p_hb)
                bm_add_aerospace_bolt(b_ht, p_hb, Vector((0, 0, 1)), head_r=0.0032, height=0.0030)
            bm_add_bolt_pair_lockwire(b_ht, ht_bolts[0], ht_bolts[1], Vector((0, 0, 1)), head_r=0.0032, strand_r=0.00040)
            bm_add_bolt_pair_lockwire(b_ht, ht_bolts[1], ht_bolts[2], Vector((0, 0, 1)), head_r=0.0032, strand_r=0.00040)
        for sx_lug in (-0.014, +0.014):

            bm_add_clevis_plate(b_ht, (-1.654, r_hb_s + 0.006), (-1.625, r_hb_s + 0.054), 0.014, 0.0170, sx_lug - 0.00400, sx_lug + 0.00400, arc_segs=21)
            bm_add_clevis_plate(b_ht, (-1.596, r_hb_s + 0.006), (-1.625, r_hb_s + 0.054), 0.014, 0.01665, sx_lug - 0.00365, sx_lug + 0.00365, arc_segs=21)
        bm_add_chamfered_cyl_x(b_ht, -1.625, r_hb_s + 0.054, -0.023, +0.023, radius=0.0140, segs=32, chamfer=0.0012, hollow_r=0.0080)
        bm_add_segmented_tube(
            b_ht, Vector((-0.030, -1.625, r_hb_s + 0.054)), Vector((+0.042, -1.625, r_hb_s + 0.054)),
            [(0.00, 0.0054), (0.04, 0.0082), (0.08, 0.0074), (0.72, 0.0074), (0.74, 0.0108), (0.84, 0.0108), (0.86, 0.0056), (1.00, 0.0050)],
            segs=24, cap_ends=True,
        )
        bm_add_helical_thread_cylinder(
            b_ht, Vector((+0.028, -1.625, r_hb_s + 0.054)), Vector((+0.041, -1.625, r_hb_s + 0.054)),
            r_core=0.0052, thread_h=0.00095, n_turns=5, segs_turn=16, cap_core=False,
        )
        bm_add_bevel_locked_hex(
            b_ht, Vector((+0.028, -1.625, r_hb_s + 0.054)), Vector((+0.036, -1.625, r_hb_s + 0.054)),
            [(0.0, 0.0082), (0.25, 0.0098), (0.80, 0.0098), (1.0, 0.0080)], corner_bevel=0.14, cap_ends=True,
        )

        for sgn_hct in (-1.0, +1.0):
            bm_add_spline_tube(
                b_ht,
                [
                    Vector((+0.038, -1.625 - 0.007, r_hb_s + 0.054 + sgn_hct * 0.0006)),
                    Vector((+0.038, -1.625 + 0.005, r_hb_s + 0.054 + sgn_hct * 0.0006)),
                    Vector((+0.040, -1.625 + 0.010, r_hb_s + 0.054 + sgn_hct * 0.0042)),
                ],
                radius=0.00075, n_samples=14, segs=8, cap_ends=True,
            )
        bm_add_spline_tube(
            b_ht,
            [Vector((-0.029, -1.625, r_hb_s + 0.054)), Vector((-0.038, -1.648, r_hb_s + 0.034)), Vector((-0.032, -1.662, r_hb_s + 0.008))],
            radius=0.0011, n_samples=18, segs=10, cap_ends=True,
        )
        sgn_ht_side = +1.0 if (phi_ht < 180.0) else -1.0
        _bm_add_box_x_yz(b_ht, -0.020, +0.020, -1.708, -1.674, r_hb_s - 0.0005, r_hb_s + 0.0022, nx=2, ny=2, nz=1, chamfer=0.0003)
        bm_add_3d_plate_text(
            b_ht, ["HOIST PT", "4000 LBS"],
            Vector((0.0, -1.691, r_hb_s + 0.0022)),
            Vector((0.0, sgn_ht_side, 0.0)), Vector((-sgn_ht_side, 0.0, 0.0)), Vector((0.0, 0.0, 1.0)),
            width=0.030, height=0.024, relief=0.00045, with_dividers=True,
        )
        bm_merge(bm_maint, b_ht, rot_ht, default_mat_id=MAT_STAND_ORANGE)


    dp_p0, dp_p1, dp_y0, dp_y1 = 286.0, 297.2, -2.142, -2.056
    bm_add_curved_cyl_panel(bm_maint, dp_p0, dp_p1, dp_y0, dp_y1, 0.5432, 0.5462, n_phi=18, n_y=14)
    bm_add_curved_cyl_panel(bm_maint, dp_p0, dp_p1, dp_y0, dp_y0 + 0.0032, 0.5460, 0.5472, n_phi=12, n_y=2)
    bm_add_curved_cyl_panel(bm_maint, dp_p0, dp_p1, dp_y1 - 0.0032, dp_y1, 0.5460, 0.5472, n_phi=12, n_y=2)
    for ph_dpr in (dp_p0 + 0.9, dp_p1 - 0.9):
        for y_dpr in (dp_y0 + 0.006, dp_y1 - 0.006):
            bm_add_dome_rivet(bm_maint, _cyl_p(ph_dpr, y_dpr, 0.5468), _cyl_n(ph_dpr), radius=0.0013)
    bm_add_conformal_cyl_text(
        bm_maint, ["PRATT WHITNEY", "F100-PW-220", "SER P-701220"],
        dp_p0 + 1.1, dp_p1 - 1.1, dp_y0 + 0.006, dp_y1 - 0.006,
        r_surf=0.5462, relief=0.00065, read_mode='AXIAL_STBD', with_dividers=True,
    )
    create_mesh_obj("Static_Maintenance_Hoists_And_DataPlate", bm_maint, coll_static, clay)


    bm_shroud = bmesh.new()

    def _r_sh_out(yv: float) -> float:
        if yv <= -0.415:
            t_f = max(0.0, min(1.0, (-0.415 - yv) / 0.305))
            return 0.614 + 0.032 * (1.0 - (t_f ** 1.80))
        t_a = max(0.0, min(1.28, (yv - (-0.415)) / 0.360))
        return 0.646 - 0.0225 * (t_a ** 1.55)

    prof_sh_collar = [
        (-0.720, 0.5485), (-0.720, 0.6115), (-0.714, 0.6145),
        (-0.708, 0.6145), (-0.708, 0.6085), (-0.713, 0.6065), (-0.713, 0.5485),
    ]
    notches_sh_collar = (
        [(pa, 0.0595, 0.0055, 0.6055, 0.6145) for pa in act_phis]
        + [(rvdt_phi_rad, 0.0240, 0.0045, 0.5985, 0.6145)]
    )
    bm_add_notched_revolved_profile(bm_shroud, prof_sh_collar, notches_sh_collar, bevel_r=0.00038, base_segs=256)

    t_bz_p = subdivide_1d_with_support_loops(8, sup1=0.03, sup2=0.10)
    t_bz_y = subdivide_1d_with_support_loops(4, sup1=0.05, sup2=0.16)
    for k_bz in range(spec.n_actuators):
        ph_bz0 = k_bz * 45.0 - 5.1
        ph_bz1 = k_bz * 45.0 + 5.1
        g_bzi, g_bzo = [], []
        for fp_b in t_bz_p:
            ar_b = math.radians(ph_bz0 + (ph_bz1 - ph_bz0) * fp_b)
            sb, cb = math.sin(ar_b), math.cos(ar_b)
            ci_b, co_b = [], []
            for fy_b in t_bz_y:
                yb = -0.721 + 0.021 * fy_b
                ro_base = _r_sh_out(max(-0.714, yb))
                ci_b.append(bm_shroud.verts.new(((ro_base - 0.0005) * sb, yb, (ro_base - 0.0005) * cb)))
                co_b.append(bm_shroud.verts.new(((ro_base + 0.0022) * sb, yb, (ro_base + 0.0022) * cb)))
            g_bzi.append(ci_b)
            g_bzo.append(co_b)
        bm_build_beveled_quad_shell(bm_shroud, g_bzi, g_bzo, bulge_scale=0.16)
        for d_rv_b in (-4.1, +4.1):
            for y_rv_b in (-0.716, -0.704):
                bm_add_dome_rivet(bm_shroud, _cyl_p(k_bz * 45.0 + d_rv_b, y_rv_b, _r_sh_out(max(-0.714, y_rv_b)) + 0.0022), _cyl_n(k_bz * 45.0 + d_rv_b), radius=0.0016)


    prof_sh_barrel = [(-0.710, _r_sh_out(-0.710) - 0.0040), (-0.710, _r_sh_out(-0.710))]
    for ky_s in range(1, 23):
        ys = -0.710 + 0.228 * (ky_s / 22.0)
        prof_sh_barrel.append((ys, _r_sh_out(ys)))
    prof_sh_barrel.append((-0.480, _r_sh_out(-0.480) - 0.0016))
    for ky_c in range(1, 13):
        yc = -0.480 + 0.120 * (ky_c / 12.0)
        prof_sh_barrel.append((yc, _r_sh_out(yc) - 0.0016))
    prof_sh_barrel.append((-0.360, _r_sh_out(-0.360)))
    for ky_a in range(1, 25):
        ya = -0.360 + 0.281 * (ky_a / 24.0)
        prof_sh_barrel.append((ya, _r_sh_out(ya)))
    prof_sh_barrel.extend([
        (-0.079, _r_sh_out(-0.079) - 0.0018),
        (-0.072, _r_sh_out(-0.072) - 0.0018),
        (-0.072, _r_sh_out(-0.072) - 0.0042),
    ])
    for ky_i in range(32, 0, -1):
        yi = -0.710 + 0.638 * (ky_i / 33.0)
        prof_sh_barrel.append((yi, _r_sh_out(yi) - 0.0042))
    bm_add_revolved_profile(bm_shroud, prof_sh_barrel, segs=320, close_loop=True, bevel_r=0.00032)
    for f_shb in bm_shroud.faces:
        if f_shb.material_index == 0:
            c_shb = f_shb.calc_center_median()
            if -0.4805 <= c_shb.y <= -0.3595 and math.hypot(c_shb.x, c_shb.z) >= (_r_sh_out(c_shb.y) - 0.0028):
                f_shb.material_index = MAT_SHROUD_GREY


    f0_slp = len(bm_shroud.faces)
    for y_seal_lip in (-0.480, -0.360):
        r_slp = _r_sh_out(y_seal_lip)
        prof_slp = [
            (y_seal_lip - 0.0022, r_slp - 0.0012), (y_seal_lip - 0.0022, r_slp + 0.0014),
            (y_seal_lip + 0.0022, r_slp + 0.0014), (y_seal_lip + 0.0022, r_slp - 0.0012),
        ]
        bm_add_revolved_profile(bm_shroud, prof_slp, segs=320, close_loop=True, bevel_r=0.00025)
    bm_fill_unset_from(bm_shroud, f0_slp, MAT_RUBBER_DARK)

    for y_rpair in (-0.700, -0.682, -0.494, -0.464, -0.376, -0.344, -0.236, -0.220, -0.096, -0.082):
        r_rp = _r_sh_out(y_rpair) + (0.0025 if (-0.476 <= y_rpair <= -0.364) else 0.0)
        for kp in range(96):
            base_deg = kp * (360.0 / 96.0) + 1.875
            if -0.476 <= y_rpair <= -0.364:
                bm_add_dome_rivet(bm_shroud, _cyl_p(base_deg, y_rpair, r_rp), _cyl_n(base_deg), radius=0.00135)
            else:
                for d_pair in (-0.36, +0.36):
                    ph_d = base_deg + d_pair
                    bm_add_dome_rivet(bm_shroud, _cyl_p(ph_d, y_rpair, r_rp), _cyl_n(ph_d), radius=0.00145)

    for k_seam in range(8):
        ph_sm = k_seam * 45.0 + 11.25
        for ya_s, yb_s in [(-0.692, -0.486), (-0.354, -0.080)]:
            bm_add_curved_cyl_panel(bm_shroud, ph_sm - 0.85, ph_sm + 0.85, ya_s, yb_s, -0.0008, +0.0009, n_phi=4, n_y=12, r_surf_fn=_r_sh_out)
            for ir_s in range(7):
                yv_s = ya_s + 0.012 + (yb_s - ya_s - 0.024) * (ir_s / 6.0)
                rv_s = _r_sh_out(yv_s) + 0.0009
                for d_ps in (-0.42, +0.42):
                    bm_add_dome_rivet(bm_shroud, _cyl_p(ph_sm + d_ps, yv_s, rv_s), _cyl_n(ph_sm + d_ps), radius=0.00145)

    for k_np in range(8):
        ph_npc = k_np * 45.0 + 22.5
        ya_np, yb_np = -0.188, -0.090
        p0_np, p1_np = ph_npc - 3.60, ph_npc + 3.60
        f0_np = len(bm_shroud.faces)
        bm_add_curved_cyl_panel(bm_shroud, p0_np, p1_np, ya_np, yb_np, -0.0005, +0.0006, n_phi=8, n_y=8, r_surf_fn=_r_sh_out)
        bm_add_curved_cyl_panel(bm_shroud, p0_np, p1_np, ya_np, ya_np + 0.0032, -0.0002, +0.0012, n_phi=8, n_y=2, r_surf_fn=_r_sh_out)
        bm_add_curved_cyl_panel(bm_shroud, p0_np, p1_np, yb_np - 0.0032, yb_np, -0.0002, +0.0012, n_phi=8, n_y=2, r_surf_fn=_r_sh_out)
        bm_add_curved_cyl_panel(bm_shroud, p0_np, p0_np + 0.36, ya_np + 0.0032, yb_np - 0.0032, -0.0002, +0.0012, n_phi=2, n_y=6, r_surf_fn=_r_sh_out)
        bm_add_curved_cyl_panel(bm_shroud, p1_np - 0.36, p1_np, ya_np + 0.0032, yb_np - 0.0032, -0.0002, +0.0012, n_phi=2, n_y=6, r_surf_fn=_r_sh_out)
        bm_fill_unset_from(bm_shroud, f0_np, MAT_PLATE_BASE)
        for ph_npr in (p0_np + 0.65, p1_np - 0.65):
            for y_npr in (ya_np + 0.006, yb_np - 0.006):
                bm_add_dome_rivet(bm_shroud, _cyl_p(ph_npr, y_npr, _r_sh_out(y_npr) + 0.0007), _cyl_n(ph_npr), radius=0.00115)


    f0_sh_grey = len(bm_shroud.faces)
    t_st_y = subdivide_1d_with_support_loops(10, sup1=0.022, sup2=0.072)
    for ph_st0, ph_st1, n_st_seg in [(-88.2, +88.2, 64), (+91.8, +177.0, 32), (+183.0, +268.2, 32)]:
        t_st_p = subdivide_1d_with_support_loops(n_st_seg, sup1=0.005, sup2=0.016)
        g_sti, g_sto = [], []
        for fp_s in t_st_p:
            ar_s = math.radians(ph_st0 + (ph_st1 - ph_st0) * fp_s)
            ss, cs = math.sin(ar_s), math.cos(ar_s)
            ci_s, co_s = [], []
            for fy_s in t_st_y:
                ys_b = -0.478 + 0.116 * fy_s
                r_ch = _r_sh_out(ys_b) - 0.0016
                crown_b = 0.0006 * math.sin(math.pi * fy_s)
                ci_s.append(bm_shroud.verts.new(((r_ch + 0.0002) * ss, ys_b, (r_ch + 0.0002) * cs)))
                co_s.append(bm_shroud.verts.new(((r_ch + 0.0040 + crown_b) * ss, ys_b, (r_ch + 0.0040 + crown_b) * cs)))
            g_sti.append(ci_s)
            g_sto.append(co_s)
        bm_build_beveled_quad_shell(bm_shroud, g_sti, g_sto, bulge_scale=0.16)

    t_sp_p = subdivide_1d_with_support_loops(6, sup1=0.04, sup2=0.13)
    t_sp_y = subdivide_1d_with_support_loops(10, sup1=0.03, sup2=0.09)
    for ph_spl in (0.0, 45.0, 90.0, 135.0, 225.0, 270.0, 315.0):
        half_ang_sp = 3.40
        ya_sp, yb_sp = (-0.502, -0.338) if (ph_spl in (90.0, 270.0)) else (-0.476, -0.364)
        g_spi, g_spo = [], []
        for fp_p in t_sp_p:
            ar_p = math.radians(ph_spl - half_ang_sp + 2.0 * half_ang_sp * fp_p)
            sp_v, cp_v = math.sin(ar_p), math.cos(ar_p)
            ci_p, co_p = [], []
            for fy_p in t_sp_y:
                yp = ya_sp + (yb_sp - ya_sp) * fy_p
                r_sp_base = _r_sh_out(yp) + 0.0024
                ci_p.append(bm_shroud.verts.new((r_sp_base * sp_v, yp, r_sp_base * cp_v)))
                co_p.append(bm_shroud.verts.new(((r_sp_base + 0.0026) * sp_v, yp, (r_sp_base + 0.0026) * cp_v)))
            g_spi.append(ci_p)
            g_spo.append(co_p)
        bm_build_beveled_quad_shell(bm_shroud, g_spi, g_spo, bulge_scale=0.16)
        for d_sp_b in (-half_ang_sp * 0.66, +half_ang_sp * 0.66):
            for y_sp_b in (ya_sp + 0.014, yb_sp - 0.014):
                r_b_cur = _r_sh_out(y_sp_b) + 0.0050
                bm_add_aerospace_bolt(bm_shroud, _cyl_p(ph_spl + d_sp_b, y_sp_b, r_b_cur), _cyl_n(ph_spl + d_sp_b), head_r=0.0022, height=0.0020)

        for d_sp_rv in (-half_ang_sp * 0.36, +half_ang_sp * 0.36):
            for fy_spr in (0.28, 0.43, 0.57, 0.72):
                y_spr = ya_sp + (yb_sp - ya_sp) * fy_spr
                bm_add_dome_rivet(bm_shroud, _cyl_p(ph_spl + d_sp_rv, y_spr, _r_sh_out(y_spr) + 0.0050),
                                  _cyl_n(ph_spl + d_sp_rv), radius=0.00135)


    t_sk_r = [0.00, 0.08, 0.24, 0.76, 0.92, 1.00]
    t_sk_t = [0.00, 0.12, 0.88, 1.00]
    for k_sec_sh in range(8):
        for off_sk in (11.2, 18.7, 26.3, 33.8):
            ph_sk = k_sec_sh * 45.0 + off_sk
            if 173.0 <= ph_sk <= 187.0:
                continue
            def _sk_frame_pt(k_i: int, vr: float, vt: float, _pc=ph_sk):
                th_k = 2.0 * math.pi * k_i / 32.0
                ck, sk = math.cos(th_k), math.sin(th_k)
                m_inf = max(abs(ck), abs(sk), 1e-6)
                ux, uy = ck / m_inf, sk / m_inf
                dphi_v = (0.78 + (1.42 - 0.78) * vr) * ux
                dy_v   = (0.0058 + (0.0108 - 0.0058) * vr) * uy
                yc_v   = -0.420 + dy_v
                rc_v   = (_r_sh_out(yc_v) + 0.0004) + 0.0031 * vt
                ar_v   = math.radians(_pc + dphi_v)
                return Vector((rc_v * math.sin(ar_v), yc_v, rc_v * math.cos(ar_v)))
            _bm_add_annular_quad_extrusion(bm_shroud, _sk_frame_pt, 32, t_sk_r, t_sk_t)

            b_sk_tab = bmesh.new()
            r_sk_in = _r_sh_out(-0.420) + 0.0002
            _bm_add_box_x_yz(b_sk_tab, -0.0046, +0.0046, -0.4242, -0.4158, r_sk_in - 0.0010, r_sk_in + 0.0019, nx=2, ny=2, nz=1, chamfer=0.00035)
            bm_add_dome_rivet(b_sk_tab, Vector((0.0, -0.420, r_sk_in + 0.0019)), Vector((0, 0, 1)), radius=0.0012)
            bm_merge(bm_shroud, b_sk_tab, Matrix.Rotation(math.radians(ph_sk), 4, 'Y'), default_mat_id=MAT_MACHINED_ALU)
    bm_fill_unset_from(bm_shroud, f0_sh_grey, MAT_SHROUD_GREY)


    for k_ap in range(8):
        ph_ap = k_ap * 45.0 + 22.5
        b_aport = bmesh.new()
        r_ap_s = _r_sh_out(-0.268)

        prof_ap = [
            (-0.0032, 0.0038), (-0.0032, 0.0058), (+0.0004, 0.0058),
            (+0.0016, 0.0072), (+0.0016, 0.0112), (-0.0008, 0.0112), (-0.0012, 0.0038),
        ]
        b_ap_rev = bmesh.new()
        bm_add_revolved_profile(b_ap_rev, prof_ap, segs=28, close_loop=True, bevel_r=0.00020)
        bm_add_segmented_tube(
            b_ap_rev, Vector((0.0, -0.0030, 0.0)), Vector((0.0, -0.0010, 0.0)),
            [(0.0, 0.0039), (0.85, 0.0039), (1.0, 0.0034)], segs=20, cap_ends=True,
        )
        bm_merge(b_aport, b_ap_rev, Matrix.Translation((0.0, -0.268, r_ap_s)) @ Matrix.Rotation(0.5 * math.pi, 4, 'X'))
        for ka_rv in range(4):
            ang_arv = 0.25 * math.pi + 0.5 * math.pi * ka_rv
            bm_add_dome_rivet(
                b_aport, Vector((0.0090 * math.cos(ang_arv), -0.268 + 0.0090 * math.sin(ang_arv), r_ap_s + 0.0016)),
                Vector((0, 0, 1)), radius=0.0010,
            )
        bm_merge(bm_shroud, b_aport, Matrix.Rotation(math.radians(ph_ap), 4, 'Y'), default_mat_id=MAT_MACHINED_ALU)

    rot_btm_lch = Matrix.Rotation(math.pi, 4, 'Y')
    b_lch = bmesh.new()
    r_btm = _r_sh_out(-0.420)
    for sx_lc in (-0.026, +0.026):
        bm_add_clevis_plate(b_lch, (-0.457, r_btm + 0.002), (-0.383, r_btm + 0.018), 0.0075, 0.0085, sx_lc - 0.0048, sx_lc + 0.0048, arc_segs=13)
        bm_add_chamfered_cyl_x(b_lch, -0.420, r_btm + 0.017, sx_lc - 0.0064, sx_lc + 0.0064, radius=0.0076, segs=22)

    for sx_hl in (-0.014, +0.014):
        bm_add_clevis_plate(b_lch, (-0.420, r_btm + 0.017), (-0.386, r_btm + 0.028), 0.0062, 0.0050, sx_hl - 0.0022, sx_hl + 0.0022, arc_segs=11)
    _bm_add_box_x_yz(b_lch, -0.016, +0.016, -0.394, -0.372, r_btm + 0.025, r_btm + 0.031, nx=2, ny=2, nz=1, chamfer=0.0005)
    bm_add_segmented_tube(
        b_lch, Vector((-0.042, -0.420, r_btm + 0.017)), Vector((+0.048, -0.420, r_btm + 0.017)),
        [(0.0, 0.0060), (0.10, 0.0060), (0.12, 0.0038), (0.86, 0.0038), (1.0, 0.0032)], segs=18, cap_ends=True,
    )
    bm_add_bevel_locked_hex(
        b_lch, Vector((+0.035, -0.420, r_btm + 0.017)), Vector((+0.044, -0.420, r_btm + 0.017)),
        [(0.0, 0.0058), (0.16, 0.0070), (0.84, 0.0070), (1.0, 0.0054)], corner_bevel=0.14, cap_ends=True,
    )
    loop_wire_pts = [
        Vector((0.004 + 0.018 * math.sin(2.0 * math.pi * kw / 24.0), -0.420 + 0.004 * math.cos(2.0 * math.pi * kw / 24.0), r_btm + 0.029 - 0.015 * math.cos(2.0 * math.pi * kw / 24.0)))
        for kw in range(25)
    ]
    bm_add_spline_tube(b_lch, loop_wire_pts, radius=0.00090, n_samples=32, segs=12, cap_ends=True)
    bm_merge(bm_shroud, b_lch, rot_btm_lch, default_mat_id=MAT_SHROUD_GREY)

    r_lip_0 = _r_sh_out(-0.081)
    r_lip_1 = _r_sh_out(-0.072)
    prof_aft_cuff = [
        (-0.081, r_lip_0 - 0.0040),
        (-0.081, r_lip_0 + 0.0014),
        (-0.0725, r_lip_1 + 0.0014),
        (-0.0720, r_lip_1 + 0.0002),
        (-0.0720, r_lip_1 - 0.0040),
    ]
    bm_add_revolved_profile(bm_shroud, prof_aft_cuff, segs=320, close_loop=True, bevel_r=0.00028)
    for kp_aft in range(96):
        ph_af = kp_aft * (360.0 / 96.0) + 1.875
        bm_add_dome_rivet(bm_shroud, _cyl_p(ph_af, -0.0765, _r_sh_out(-0.0765) + 0.0014), _cyl_n(ph_af), radius=0.00140)

    _hd_detail.add_shroud_detailing(bm_shroud, _r_sh_out, _cyl_p, _cyl_n)
    create_mesh_obj("Static_Nozzle_Outer_Shroud", bm_shroud, coll_static, clay)


    bm_tec_front = bmesh.new()
    prof_tec_fwd = [
        (-2.245, 0.498), (-2.245, 0.566), (-2.235, 0.566), (-2.227, 0.542),
        (-2.180, 0.542), (-2.173, 0.560), (-2.157, 0.560), (-2.150, 0.5430),
        (-2.146, 0.5430), (-2.146, 0.506), (-2.200, 0.502),
    ]
    bm_add_revolved_profile(bm_tec_front, prof_tec_fwd, segs=240, close_loop=True, bevel_r=0.00045)
    prof_tec_aft = [
        (-1.902, 0.508), (-1.902, 0.5450), (-1.898, 0.5450),
        (-1.890, 0.568), (-1.880, 0.568), (-1.880, 0.512),
    ]
    bm_add_revolved_profile(bm_tec_front, prof_tec_aft, segs=240, close_loop=True, bevel_r=0.00045)


    def _r_tec_skin(_ph_deg: float, yv: float) -> float:
        return 0.5430 + 0.0020 * ((yv - (-2.148)) / 0.248)

    tec_pocket_cfg = {
        3:  (52.5,  -2.024, 0.042, 0.049, 'Y'),
        5:  (83.0,  -2.065, 0.043, 0.043, 'ELLIPSE'),
        8:  (127.5, -2.024, 0.042, 0.049, 'Y'),
        15: (232.5, -2.024, 0.042, 0.049, 'Y'),
        18: (277.0, -2.065, 0.043, 0.043, 'ELLIPSE'),
        20: (307.5, -2.024, 0.042, 0.049, 'Y'),
    }
    tec_pkt_prof = [(0.0, 0.0), (0.0007, 0.0008), (0.0038, 0.0082), (0.0052, 0.0086)]
    b_tec_shell = bmesh.new()
    for kt_p in range(24):
        ph_t0 = kt_p * 15.0
        ph_t1 = (kt_p + 1) * 15.0
        if kt_p in tec_pocket_cfg:
            pc_t, yc_t, hw_t, hy_t, ax_t = tec_pocket_cfg[kt_p]
            bm_add_cyl_conical_pocket_patch(
                b_tec_shell, ph_t0, ph_t1, -2.148, -1.900, 12, 14, _r_tec_skin,
                half_arc_w=hw_t, half_y_h=hy_t, depth_profile=tec_pkt_prof,
                stadium_axis=ax_t, phi_c=pc_t, y_c=yc_t,
            )
        else:
            bm_add_cyl_plain_patch(b_tec_shell, ph_t0, ph_t1, -2.148, -1.900, 12, 14, _r_tec_skin)
    bm_add_cyl_plain_patch(b_tec_shell, 0.0, 360.0, -2.148, -1.900, 288, 6, lambda _p, _y: 0.506, reverse_normals=True)
    for y_cap_t, rev_cap_t in ((-2.148, True), (-1.900, False)):
        ro_c = _r_tec_skin(0.0, y_cap_t)
        ri_c = 0.506
        v_o_ring, v_i_ring = [], []
        for ic_t in range(288 + 1):
            ang_ct = math.radians(ic_t * (360.0 / 288.0))
            st_c, ct_c = math.sin(ang_ct), math.cos(ang_ct)
            v_o_ring.append(b_tec_shell.verts.new((ro_c * st_c, y_cap_t, ro_c * ct_c)))
            v_i_ring.append(b_tec_shell.verts.new((ri_c * st_c, y_cap_t, ri_c * ct_c)))
        for ic_t in range(288):
            if rev_cap_t:
                b_tec_shell.faces.new((v_i_ring[ic_t], v_o_ring[ic_t], v_o_ring[ic_t + 1], v_i_ring[ic_t + 1]))
            else:
                b_tec_shell.faces.new((v_o_ring[ic_t], v_i_ring[ic_t], v_i_ring[ic_t + 1], v_o_ring[ic_t + 1]))
    bmesh.ops.remove_doubles(b_tec_shell, verts=[v for v in b_tec_shell.verts if v.is_boundary], dist=1e-4)
    bmesh.ops.recalc_face_normals(b_tec_shell, faces=b_tec_shell.faces)
    bm_merge(bm_tec_front, b_tec_shell)


    tec_fl_bolts = []
    for itf in range(80):
        deg_tf = 360.0 * itf / 80.0
        if (82.0 <= deg_tf <= 98.0) or (262.0 <= deg_tf <= 278.0):
            continue
        atf = math.radians(deg_tf)
        ptf = Vector((0.551 * math.sin(atf), -2.173, 0.551 * math.cos(atf)))
        tec_fl_bolts.append(ptf)
        bm_add_aerospace_bolt(bm_tec_front, ptf, Vector((0.0, -1.0, 0.0)), head_r=0.0029, height=0.0028, with_12pt_nut_depth=0.015)
    for itf in range(0, len(tec_fl_bolts) - 1, 2):
        if (tec_fl_bolts[itf] - tec_fl_bolts[itf + 1]).length < 0.065:
            bm_add_bolt_pair_lockwire(bm_tec_front, tec_fl_bolts[itf], tec_fl_bolts[itf + 1], Vector((0.0, -1.0, 0.0)), head_r=0.0029, strand_r=0.00034)


    b_tec_circ = bmesh.new()
    bm_add_segmented_tube(
        b_tec_circ, Vector((0.0, -2.024, 0.534)), Vector((0.0, -2.024, 0.5435)),
        [(0.0, 0.035), (0.62, 0.035), (0.78, 0.025), (1.0, 0.021)], segs=32, cap_ends=True,
    )
    tc_c_bpts = []
    for ibc in range(8):
        abc = 2.0 * math.pi * ibc / 8.0
        p_tcb_i = Vector((0.029 * math.cos(abc), -2.024 + 0.029 * math.sin(abc), 0.5398))
        tc_c_bpts.append(p_tcb_i)
        bm_add_aerospace_bolt(b_tec_circ, p_tcb_i, Vector((0, 0, 1)), head_r=0.0020, height=0.0019)
    for ibc in range(0, 8, 2):
        bm_add_bolt_pair_lockwire(b_tec_circ, tc_c_bpts[ibc], tc_c_bpts[ibc + 1], Vector((0, 0, 1)), head_r=0.0020, strand_r=0.00028)
    for deg_te in (52.5, 127.5, 232.5, 307.5):
        bm_merge(bm_tec_front, b_tec_circ.copy(), Matrix.Rotation(math.radians(deg_te), 4, 'Y'))
        te_pts = build_cyl_mandrel_pts(
            [(deg_te, -2.024, 0.542), (deg_te, -2.024, 0.574), (deg_te, -1.890, 0.573)],
            bend_r=0.032, r_unwrap=0.560, arc_segs=16, max_step=0.010,
        )
        te_fr = bm_sweep_centerline_tube(bm_tec_front, te_pts, radius=0.0088, segs=22, cap_ends=True)
        bm_add_bnut_flare_fitting(bm_tec_front, te_fr[1][0], te_fr[1][1], tube_r=0.0088)
        bm_add_adel_loop_clamp(bm_tec_front, te_fr[-3][0], te_fr[-3][1], _cyl_p(deg_te, -1.915, 0.545), Vector((0, 1, 0)), tube_r=0.0088)
    b_tec_circ.free()


    for deg_f8 in (96.5, 263.5):
        b_fig8 = bmesh.new()
        r_f8 = _r_tec_skin(deg_f8, -2.045)
        bm_add_clevis_plate(b_fig8, (-2.068, r_f8 - 0.001), (-2.022, r_f8 - 0.001), 0.017, 0.017, -0.002, +0.0055, arc_segs=15)
        for sy_p8 in (-2.058, -2.032):
            bm_add_segmented_tube(b_fig8, Vector((0.0, sy_p8, r_f8 + 0.004)), Vector((0.0, sy_p8, r_f8 + 0.013)),
                                  [(0.0, 0.0086), (0.60, 0.0086), (1.0, 0.0068)], segs=18, cap_ends=True)
            bm_add_bevel_locked_hex(b_fig8, Vector((0.0, sy_p8, r_f8 + 0.011)), Vector((0.0, sy_p8, r_f8 + 0.016)),
                                    [(0.0, 0.0062), (0.25, 0.0076), (0.75, 0.0076), (1.0, 0.0060)], corner_bevel=0.14, cap_ends=True)
        f8_bpts = []
        for sx_8 in (-0.013, +0.013):
            for sy_8 in (-2.066, -2.045, -2.024):
                p8_b = Vector((sx_8, sy_8, r_f8 + 0.0045))
                f8_bpts.append(p8_b)
                bm_add_aerospace_bolt(b_fig8, p8_b, Vector((0, 0, 1)), head_r=0.0020, height=0.0019)
        for kb_8 in (0, 3):
            bm_add_bolt_pair_lockwire(b_fig8, f8_bpts[kb_8], f8_bpts[kb_8 + 1], Vector((0, 0, 1)), head_r=0.0020, strand_r=0.00028)
            bm_add_bolt_pair_lockwire(b_fig8, f8_bpts[kb_8 + 1], f8_bpts[kb_8 + 2], Vector((0, 0, 1)), head_r=0.0020, strand_r=0.00028)
        bm_merge(bm_tec_front, b_fig8, Matrix.Rotation(math.radians(deg_f8), 4, 'Y'))


    b_tec_oval = bmesh.new()
    _bm_add_box_x_yz(b_tec_oval, -0.020, +0.020, -2.088, -2.050, 0.5425, 0.5480, nx=2, ny=2, nz=1, chamfer=0.0007)
    bm_add_segmented_tube(b_tec_oval, Vector((0.0, -2.069, 0.5465)), Vector((0.0, -2.069, 0.5515)),
                          [(0.0, 0.0105), (0.65, 0.0105), (1.0, 0.0078)], segs=20, cap_ends=True)
    ov_bpts = []
    for sx_ov in (-0.015, +0.015):
        for sy_ov in (-2.083, -2.055):
            pov = Vector((sx_ov, sy_ov, 0.5480))
            ov_bpts.append(pov)
            bm_add_aerospace_bolt(b_tec_oval, pov, Vector((0, 0, 1)), head_r=0.0020, height=0.0019)
    bm_add_bolt_pair_lockwire(b_tec_oval, ov_bpts[0], ov_bpts[1], Vector((0, 0, 1)), head_r=0.0020, strand_r=0.00028)
    bm_add_bolt_pair_lockwire(b_tec_oval, ov_bpts[2], ov_bpts[3], Vector((0, 0, 1)), head_r=0.0020, strand_r=0.00028)
    for deg_ov in (18.0, 104.5, 166.0, 194.0, 255.5, 342.0):
        bm_merge(bm_tec_front, b_tec_oval.copy(), Matrix.Rotation(math.radians(deg_ov), 4, 'Y'))
    b_tec_oval.free()

    for ph_mnt in (90.0, 270.0):
        b_mnt = bmesh.new()
        bm_add_clevis_plate(b_mnt, (-2.200, 0.542), (-2.130, 0.543), 0.012, 0.012, -0.036, +0.036, arc_segs=13)
        bm_add_segmented_tube(
            b_mnt, Vector((0.0, -2.165, 0.550)), Vector((0.0, -2.165, 0.592)),
            [(0.0, 0.026), (0.35, 0.026), (0.45, 0.019), (1.0, 0.017)], segs=28, cap_ends=True,
        )
        bm_merge(bm_tec_front, b_mnt, Matrix.Rotation(math.radians(ph_mnt), 4, 'Y'), default_mat_id=MAT_MACHINED_ALU)

    prof_mixer = [
        (-2.236, 0.390), (-2.236, 0.402), (-1.975, 0.418), (-1.975, 0.406),
    ]
    bm_add_revolved_profile(bm_tec_front, prof_mixer, segs=160, close_loop=True, bevel_r=0.00040)


    for sgn_tc, d_cov_tc, d_bot_tc in ((-1.0, 277.0, 210.0), (+1.0, 83.0, 150.0)):

        b_scov = bmesh.new()
        bm_add_segmented_tube(b_scov, Vector((0, -2.065, 0.534)), Vector((0, -2.065, 0.545)),
                              [(0.0, 0.038), (0.62, 0.038), (0.76, 0.028), (1.0, 0.025)], segs=32, cap_ends=True)
        sc_bpts = []
        for kb_sc in range(10):
            asc = 2.0 * math.pi * kb_sc / 10.0
            psc_i = Vector((0.032 * math.cos(asc), -2.065 + 0.032 * math.sin(asc), 0.5405))
            sc_bpts.append(psc_i)
            bm_add_aerospace_bolt(b_scov, psc_i, Vector((0, 0, 1)), head_r=0.0020, height=0.0019)
        for kb_sc in range(0, 10, 2):
            bm_add_bolt_pair_lockwire(b_scov, sc_bpts[kb_sc], sc_bpts[kb_sc + 1], Vector((0, 0, 1)), head_r=0.0020, strand_r=0.00028)
        bm_merge(bm_tec_front, b_scov, Matrix.Rotation(math.radians(d_cov_tc), 4, 'Y'))


        for idx_dp, (d_off_dp, y_vert_dp, br_dp) in enumerate([
            (-sgn_tc * 1.8, -1.940, 0.034),
            (+sgn_tc * 1.8, -1.964, 0.028),
        ]):
            d_st_dp = d_cov_tc + d_off_dp
            dp_pts = build_cyl_mandrel_pts(
                [
                    (d_st_dp, -2.065, 0.542),
                    (d_st_dp, -2.065, 0.566),
                    (d_st_dp, y_vert_dp, 0.566),
                    (d_bot_tc - sgn_tc * 5.0 + d_off_dp, y_vert_dp, 0.566),
                    (d_bot_tc - sgn_tc * 5.0 + d_off_dp, -2.015, 0.566),
                ],
                bend_r=br_dp, r_unwrap=0.562, arc_segs=16, max_step=0.010,
            )
            dp_fr = bm_sweep_centerline_tube(bm_tec_front, dp_pts, radius=0.0062, segs=20, cap_ends=True)
            bm_add_bnut_flare_fitting(bm_tec_front, dp_fr[2][0], dp_fr[2][1], tube_r=0.0062)
            bm_add_bnut_flare_fitting(bm_tec_front, dp_fr[-2][0], dp_fr[-2][1], tube_r=0.0062)


        for f_cl_t in (0.30, 0.58, 0.82):
            d_cl_m = d_cov_tc + (d_bot_tc - d_cov_tc) * f_cl_t
            b_tcl = bmesh.new()
            _bm_add_box_x_yz(b_tcl, -0.006, +0.006, -1.976, -1.928, 0.544, 0.573, nx=1, ny=3, nz=1, chamfer=0.0007)
            for y_tcb in (-1.971, -1.952, -1.933):
                bm_add_aerospace_bolt(b_tcl, Vector((0.0, y_tcb, 0.573)), Vector((0, 0, 1)), head_r=0.0020, height=0.0018)
            bm_merge(bm_tec_front, b_tcl, Matrix.Rotation(math.radians(d_cl_m), 4, 'Y'))


        d_mid_tc = 255.5 if sgn_tc < 0 else 104.5
        d_vblk_tc = d_bot_tc - sgn_tc * 5.0
        mp_pts = build_cyl_mandrel_pts(
            [
                (d_mid_tc, -2.069, 0.546),
                (d_mid_tc, -2.069, 0.562),
                (d_mid_tc, -2.002, 0.562),
                (d_vblk_tc, -2.002, 0.564),
                (d_vblk_tc, -2.035, 0.564),
            ],
            bend_r=0.024, r_unwrap=0.560, arc_segs=14, max_step=0.010,
        )
        mp_fr = bm_sweep_centerline_tube(bm_tec_front, mp_pts, radius=0.0048, segs=18, cap_ends=True)
        bm_add_bnut_flare_fitting(bm_tec_front, mp_fr[2][0], mp_fr[2][1], tube_r=0.0048)

        b_tvblk = bmesh.new()
        _bm_add_box_x_yz(b_tvblk, -0.042, +0.042, -2.115, -1.995, 0.544, 0.580, nx=2, ny=3, nz=2, chamfer=0.0010)
        for sx_tv in (-0.030, +0.030):
            for sy_tv in (-2.100, -2.010):
                bm_add_aerospace_bolt(b_tvblk, Vector((sx_tv, sy_tv, 0.580)), Vector((0, 0, 1)), head_r=0.0022, height=0.0020)
        bm_merge(bm_tec_front, b_tvblk, Matrix.Rotation(math.radians(d_vblk_tc), 4, 'Y'))


        for k_tbr in range(3):
            d_tbr = (d_bot_tc - sgn_tc * 6.0) + sgn_tc * (k_tbr * 3.2)
            tbr_pts = build_cyl_mandrel_pts(
                [(d_tbr, -2.238, 0.566), (d_tbr, -2.115, 0.562), (d_tbr, -1.925, 0.562), (d_tbr, -1.882, 0.568)],
                bend_r=0.040, r_unwrap=0.562, arc_segs=14, max_step=0.012,
            )
            bm_sweep_centerline_tube(bm_tec_front, tbr_pts, radius=0.0040, segs=16, cap_ends=True)

    create_mesh_obj("Static_Forward_TEC_And_Mount", bm_tec_front, coll_static, clay)


    bm_bal_stat = bmesh.new()
    b_bal_seg = bmesh.new()

    bm_add_curved_cyl_panel(b_bal_seg, -4.75, +4.75, -0.030, +0.008, 0.547, 0.564, n_phi=18, n_y=8)

    prof_scoop = [
        (-0.0068, 0.0000), (-0.0068, 0.0038), (-0.0046, 0.0056), (+0.0046, 0.0056), (+0.0068, 0.0038), (+0.0068, 0.0000),
        (+0.0050, 0.0000), (+0.0050, 0.0030), (+0.0034, 0.0042), (-0.0034, 0.0042), (-0.0050, 0.0030), (-0.0050, 0.0000),
    ]
    scoop_caps = [(0, 1, 10, 11), (1, 2, 9, 10), (2, 3, 8, 9), (3, 4, 7, 8), (4, 5, 6, 7)]
    for lv_d in (-3.1, -1.05, +1.05, +3.1):
        ph_sc = math.radians(lv_d)
        u_sc = Vector((math.cos(ph_sc), 0.0, -math.sin(ph_sc)))
        v_sc = Vector((math.sin(ph_sc), 0.0, math.cos(ph_sc)))
        p_sc_a = Vector((0.5632 * math.sin(ph_sc), -0.026, 0.5632 * math.cos(ph_sc)))
        p_sc_b = Vector((0.5632 * math.sin(ph_sc), +0.001, 0.5632 * math.cos(ph_sc)))
        bm_add_swept_quad_profile(b_bal_seg, p_sc_a, p_sc_b, u_sc, v_sc, prof_scoop, n_steps=4, cap_quad_pairs=scoop_caps)
    for sx_bs in (-0.0385, +0.0385):
        p_bs0 = Vector((sx_bs, -0.024, 0.563))
        p_bs1 = Vector((sx_bs, -0.002, 0.563))
        bm_add_aerospace_bolt(b_bal_seg, p_bs0, Vector((0, 0, 1)), head_r=0.0026, height=0.0026)
        bm_add_aerospace_bolt(b_bal_seg, p_bs1, Vector((0, 0, 1)), head_r=0.0026, height=0.0026)
        bm_add_bolt_pair_lockwire(b_bal_seg, p_bs0, p_bs1, Vector((0, 0, 1)), head_r=0.0026, strand_r=0.00036)

    for x_bkn in (-0.0315, 0.0, +0.0315):
        bm_add_clevis_plate(b_bal_seg, (-0.006, 0.563), (0.008, 0.562), 0.0036, 0.0054, x_bkn - 0.0038, x_bkn + 0.0038, arc_segs=13)
        bm_add_clevis_plate(b_bal_seg, (-0.010, 0.5635), (0.008, 0.562), 0.0024, 0.0044, x_bkn - 0.0013, x_bkn + 0.0013, arc_segs=11)
        bm_add_chamfered_cyl_x(b_bal_seg, 0.008, 0.562, x_bkn - 0.0044, x_bkn + 0.0044, radius=0.0058, segs=20, hollow_r=0.0034)
    for x_btw in (-0.0364, -0.0266, -0.0050, +0.0050, +0.0266, +0.0364):
        bm_add_chamfered_cyl_x(b_bal_seg, 0.008, 0.562, x_btw - 0.0005, x_btw + 0.0005, radius=0.0063, segs=20, chamfer=0.00015, hollow_r=0.0035)
    bm_add_clevis_pin_assembly_x(b_bal_seg, 0.008, 0.562, -0.0445, +0.0445, pin_r=0.0034, segs=20, compact_ends=True)

    for sx_st_sp in (-1.0, +1.0):
        x_anch = sx_st_sp * 0.0192
        st_sp_pts = []
        for ic_s in range(11):
            fc_s = ic_s / 10.0
            th_s = math.radians(295.0) - (2.0 * math.pi * 0.62) * fc_s
            st_sp_pts.append(Vector((x_anch, 0.008 + 0.0046 * math.cos(th_s), 0.562 + 0.0046 * math.sin(th_s))))
        st_sp_pts.extend([
            Vector((x_anch, +0.0020, 0.5646)),
            Vector((x_anch, -0.0075, 0.5646)),
            Vector((x_anch, -0.0170, 0.5645)),
        ])
        bm_add_spline_tube(b_bal_seg, st_sp_pts, radius=0.00046, n_samples=28, segs=12, cap_ends=True)
        bm_add_clevis_plate(b_bal_seg, (-0.0155, 0.5643), (-0.0085, 0.5643), 0.0014, 0.0014, x_anch - 0.0024, x_anch + 0.0024, arc_segs=9)
        bm_add_dome_rivet(b_bal_seg, Vector((x_anch + sx_st_sp * 0.0034, -0.0120, 0.5642)), Vector((0, 0, 1)), radius=0.0013)

    for x_sst in (-0.0205, 0.0, +0.0205):
        bm_add_clevis_plate(b_bal_seg, (-0.012, 0.5455), (-0.0015, 0.5420), 0.0032, 0.0032, x_sst - 0.0026, x_sst + 0.0026, arc_segs=11)
        bm_add_chamfered_cyl_x(b_bal_seg, -0.0015, 0.5420, x_sst - 0.0030, x_sst + 0.0030, radius=0.0032, segs=18, hollow_r=0.0018)

    for i_bs in range(spec.n_sectors):
        phi_s_rad = 2.0 * math.pi * (i_bs + 0.5) / spec.n_sectors
        bm_merge(bm_bal_stat, b_bal_seg.copy(), Matrix.Rotation(phi_s_rad, 4, 'Y'))
    b_bal_seg.free()
    create_mesh_obj("Static_Balance_Chamber_Manifold", bm_bal_stat, coll_static, clay)


    def _r_liner_env(yv: float) -> float:
        if yv <= -0.855:
            return 0.496 + 0.006 * max(0.0, min(1.0, (yv - (-1.862)) / 1.007))
        if yv <= -0.716:
            return 0.502 + 0.009 * ((yv - (-0.855)) / 0.139)
        return 0.511 + 0.005 * max(0.0, min(1.0, (yv - (-0.716)) / 0.716))

    def _smooth01(a: float, b: float, x: float) -> float:
        t_s = max(0.0, min(1.0, (x - a) / (b - a))) if abs(b - a) > 1e-9 else 0.0
        return t_s * t_s * (3.0 - 2.0 * t_s)


    bm_liner = bmesh.new()
    n_phi_ln = 378
    n_ax_ln  = 144
    t_y_ln   = [0.000, 0.004] + [0.010 + 0.980 * (k / (n_ax_ln - 4)) for k in range(n_ax_ln - 3)] + [0.996, 1.000]

    def _liner_shell_pt(k_idx: int, vr: float, vt: float) -> Vector:
        yv = -1.862 + vt * 1.838
        phi = 2.0 * math.pi * k_idx / float(n_phi_ln)
        rb_l = _r_liner_env(yv)
        env_flute = _smooth01(-1.856, -1.802, yv) * _smooth01(-0.028, -0.088, yv)

        flute = 0.0045 * env_flute * math.cos(54.0 * phi)

        u_tier = (yv - (-1.825)) / 0.242
        fr_u = u_tier - math.floor(u_tier)
        louver = 0.0024 * env_flute * (fr_u / 0.86 if fr_u < 0.86 else (1.0 - fr_u) / 0.14)

        env_aft_r = _smooth01(-0.45, -0.34, yv) * _smooth01(-0.026, -0.056, yv)
        aft_rings = 0.0016 * env_aft_r * math.sin(2.0 * math.pi * (yv - (-0.44)) / 0.0155)
        r_in_l = rb_l - 0.0064 + flute + louver + aft_rings
        r_cur  = r_in_l + vr * 0.0034
        return Vector((r_cur * math.sin(phi), yv, r_cur * math.cos(phi)))

    _bm_add_annular_quad_extrusion(bm_liner, _liner_shell_pt, n_phi_ln, [0.0, 0.18, 0.82, 1.0], t_y_ln)


    bl_strap = bmesh.new()
    for r_k in range(24):
        yr = -1.80 + r_k * 0.072
        rl_cur = _r_liner_env(yr) - 0.0022
        bm_add_dome_rivet(bl_strap, Vector((0.0, yr, rl_cur)), Vector((0, 0, -1)), radius=0.0021)
    rl_gt = _r_liner_env(-1.398) - 0.006
    for sx_gt in (-0.011, +0.011):
        bm_add_clevis_plate(bl_strap, (-1.414, rl_gt + 0.008), (-1.388, rl_gt - 0.009), 0.0055, 0.0055, sx_gt - 0.0022, sx_gt + 0.0022, arc_segs=12)
    bm_add_chamfered_cyl_x(bl_strap, -1.398, rl_gt - 0.0045, -0.015, +0.015, radius=0.0036, segs=16)
    for s_idx in range(18):
        bm_merge(bm_liner, bl_strap.copy(), Matrix.Rotation(2.0 * math.pi * s_idx / 18.0, 4, 'Y'))
    bl_strap.free()

    prof_lv_hood = [
        (-0.0076, 0.0000), (-0.0076, 0.0022), (-0.0052, 0.0038), (+0.0052, 0.0038), (+0.0076, 0.0022), (+0.0076, 0.0000),
        (+0.0060, 0.0000), (+0.0060, 0.0016), (+0.0040, 0.0027), (-0.0040, 0.0027), (-0.0060, 0.0016), (-0.0060, 0.0000),
    ]
    lv_hood_caps = [(0, 1, 10, 11), (1, 2, 9, 10), (2, 3, 8, 9), (3, 4, 7, 8), (4, 5, 6, 7)]
    for y_lvh in (-1.660, -1.360, -1.060, -0.760, -0.460, -0.180):
        rl_h = _r_liner_env(y_lvh) - 0.0028
        b_lvh_unit = bmesh.new()
        bm_add_swept_quad_profile(
            b_lvh_unit,
            Vector((0.0, y_lvh - 0.009, rl_h)),
            Vector((0.0, y_lvh + 0.009, rl_h)),
            Vector((-1.0, 0.0, 0.0)), Vector((0.0, 0.0, -1.0)),
            prof_lv_hood, n_steps=4, cap_quad_pairs=lv_hood_caps,
        )
        for ilv in range(36):
            bm_merge(bm_liner, b_lvh_unit.copy(), Matrix.Rotation(2.0 * math.pi * (ilv + 0.5) / 36.0, 4, 'Y'))
        b_lvh_unit.free()


    for idx_fc, y_fc_r in enumerate((-1.760, -1.600, -1.440, -1.280, -1.120, -0.960, -0.800, -0.640, -0.510, -0.390, -0.255, -0.115)):
        rl_fc = _r_liner_env(y_fc_r) - 0.0034
        b_fc_one = bmesh.new()
        bm_add_recessed_micro_port(b_fc_one, Vector((0.0, y_fc_r, rl_fc)), Vector((0.0, 0.0, -1.0)), r_outer=0.0035, depth=0.0034, segs=12)
        ph_off_fc = 2.5 * (idx_fc % 2)
        for ifc in range(72):
            bm_merge(bm_liner, b_fc_one.copy(), Matrix.Rotation(math.radians(ifc * 5.0 + ph_off_fc), 4, 'Y'))
        b_fc_one.free()

    for phi_ig, y_ig in [(115.0, -1.438), (122.0, -1.418)]:
        rl_ig = _r_liner_env(y_ig)
        b_isp = bmesh.new()
        bm_add_segmented_tube(
            b_isp, Vector((0.0, y_ig, rl_ig)), Vector((0.0, y_ig + 0.026, rl_ig - 0.068)),
            [(0.00, 0.0130), (0.20, 0.0100), (0.76, 0.0098), (0.82, 0.0058), (0.94, 0.0054), (1.00, 0.0022)],
            segs=22, cap_ends=True,
        )
        bm_merge(bm_liner, b_isp, Matrix.Rotation(math.radians(phi_ig), 4, 'Y'))

    for y_sd in (-0.340, -0.205, -0.075):
        prof_sd = [
            (y_sd - 0.009, 0.514), (y_sd - 0.007, 0.506), (y_sd + 0.007, 0.506), (y_sd + 0.009, 0.514),
        ]
        bm_add_revolved_profile(bm_liner, prof_sd, segs=192, close_loop=True, bevel_r=0.00035)

    prof_throat_seal = [
        (-0.016, 0.518), (-0.008, 0.513), (-0.0035, 0.519), (-0.0035, 0.522), (-0.016, 0.521),
    ]
    bm_add_revolved_profile(bm_liner, prof_throat_seal, segs=192, close_loop=True, bevel_r=0.00035)

    prof_tr_clamp = [
        (-0.038, 0.5148), (-0.038, 0.5112), (-0.021, 0.5112), (-0.019, 0.5148),
    ]
    bm_add_revolved_profile(bm_liner, prof_tr_clamp, segs=192, close_loop=True, bevel_r=0.00030)
    t_tr_p = [0.00, 0.06, 0.22, 0.34, 0.50, 0.66, 0.78, 0.94, 1.00]
    t_tr_s = [0.00, 0.05, 0.24, 0.34, 0.56, 0.82, 0.94, 1.00]
    for itr in range(32):
        ph_tc = 360.0 * (itr + 0.5) / 32.0
        sp_td = (360.0 / 32.0) * 0.82
        for d_ang, r_off_tr in ((0.0, 0.0), (5.625, 0.0018)):
            p0_tr = math.radians(ph_tc + d_ang - 0.5 * sp_td)
            p1_tr = math.radians(ph_tc + d_ang + 0.5 * sp_td)
            g_tri, g_tro = [], []
            for fp_t in t_tr_p:
                ang_t = p0_tr + (p1_tr - p0_tr) * fp_t
                st_a, ct_a = math.sin(ang_t), math.cos(ang_t)
                bead_p = 1.0 - min(1.0, abs(fp_t - 0.50) / 0.20)
                ci_r, co_r = [], []
                for fs_t in t_tr_s:
                    yt_cur = -0.036 + fs_t * 0.0375
                    z_step = 0.0014 * min(1.0, max(0.0, (fs_t - 0.24) / 0.10))
                    bead_h = -0.0010 * bead_p * math.sin(math.pi * min(1.0, max(0.0, (fs_t - 0.08) / 0.84)))
                    rt_in  = 0.5134 + r_off_tr + z_step + (fs_t ** 1.30) * 0.0052 + bead_h
                    rt_out = rt_in + 0.0015
                    ci_r.append(bm_liner.verts.new((rt_in * st_a,  yt_cur, rt_in * ct_a)))
                    co_r.append(bm_liner.verts.new((rt_out * st_a, yt_cur, rt_out * ct_a)))
                g_tri.append(ci_r)
                g_tro.append(co_r)
            bm_build_beveled_quad_shell(bm_liner, g_tri, g_tro, bulge_scale=0.16)

    for ang_kp in (14.0, 46.0, 84.0, 128.0, 172.0, 218.0, 268.0, 318.0):
        rl_kp = _r_liner_env(-1.485)
        b_kp = bmesh.new()
        bm_add_segmented_tube(
            b_kp, Vector((0.0, -1.485, rl_kp)), Vector((0.0, -1.485, rl_kp - 0.056)),
            [(0.0, 0.0105), (0.16, 0.0105), (0.22, 0.0054), (1.0, 0.0040)], segs=20, cap_ends=True,
        )
        b_ksh = bmesh.new()
        bm_add_chamfered_cyl_x(b_ksh, 0.0, 0.0, -0.015, +0.015, radius=0.0070, segs=22, chamfer=0.0008, hollow_r=0.0044)
        bm_merge(b_kp, b_ksh, Matrix.Translation((0.0, -1.485, rl_kp - 0.058)) @ Matrix.Rotation(0.5 * math.pi, 4, 'Z'))
        bm_merge(bm_liner, b_kp, Matrix.Rotation(math.radians(ang_kp), 4, 'Y'))

    create_mesh_obj("Interior_Louvered_Corrugated_Liner", bm_liner, coll_interior, clay)


    bm_fh = bmesh.new()
    gutter_specs = [
        (0.150, -1.504, 0.015, 0.028, 0.0032, 18),
        (0.258, -1.462, 0.025, 0.042, 0.0038, 36),
        (0.362, -1.430, 0.017, 0.032, 0.0033, 45),
        (0.452, -1.402, 0.015, 0.028, 0.0032, 54),
    ]
    gutter_radii = [(rg_i, yg_i) for rg_i, yg_i, _, _, _, _ in gutter_specs]
    for r_g, y_g, w_v, d_v, th_v, n_vtabs in gutter_specs:
        prof_v = [
            (y_g + 0.0035, r_g + w_v + 0.0028),
            (y_g,          r_g + w_v),
            (y_g - d_v,    r_g + w_v * 0.125),
            (y_g - d_v - 0.004, r_g),
            (y_g - d_v,    r_g - w_v * 0.125),
            (y_g,          r_g - w_v),
            (y_g + 0.0035, r_g - w_v - 0.0028),
            (y_g,          r_g - w_v + th_v),
            (y_g - d_v + th_v * 1.4, r_g),
            (y_g,          r_g + w_v - th_v),
        ]
        bm_add_revolved_profile(bm_fh, prof_v, segs=216, close_loop=True, bevel_r=0.00040)
        b_vtab = bmesh.new()
        bm_add_clevis_plate(b_vtab, (y_g - 0.002, r_g + w_v + 0.001), (y_g + 0.009, r_g + w_v + 0.005), 0.0024, 0.0020, -0.0032, +0.0032, arc_segs=9)
        bm_add_clevis_plate(b_vtab, (y_g - 0.002, r_g - w_v - 0.001), (y_g + 0.009, r_g - w_v - 0.005), 0.0024, 0.0020, -0.0032, +0.0032, arc_segs=9)
        for ivt in range(n_vtabs):
            bm_merge(bm_fh, b_vtab.copy(), Matrix.Rotation(2.0 * math.pi * (ivt + 0.5) / n_vtabs, 4, 'Y'))
        b_vtab.free()

    vg_cap_quads = [(0, 1, 8, 9), (1, 2, 3, 8), (3, 4, 5, 8), (5, 6, 7, 8)]


    wr_in, dr_in, thr_in = 0.0130, 0.028, 0.0028
    prof_in_vg = [
        (+wr_in + 0.0018, +0.0022), (+wr_in, 0.0000), (+0.0022, -dr_in), (0.0000, -dr_in - 0.0028),
        (-0.0022, -dr_in), (-wr_in, 0.0000), (-wr_in - 0.0018, +0.0022),
        (-wr_in + thr_in, 0.0000), (0.0000, -dr_in + thr_in * 1.35), (+wr_in - thr_in, 0.0000),
    ]
    b_in_spoke = bmesh.new()
    p_ina = Vector((0.0, -1.506, 0.143))
    p_inb = Vector((0.0, -1.462, 0.256))
    t_in  = (p_inb - p_ina).normalized()
    u_in  = Vector((1.0, 0.0, 0.0))
    v_in  = u_in.cross(t_in).normalized()
    if v_in.y < 0.0:
        v_in = -v_in
    bm_add_swept_quad_profile(b_in_spoke, p_ina, p_inb, u_in, v_in, prof_in_vg, n_steps=16, cap_quad_pairs=vg_cap_quads)
    for k_isp in range(9):
        bm_merge(bm_fh, b_in_spoke.copy(), Matrix.Rotation(math.radians(20.0 + k_isp * 40.0), 4, 'Y'))
    b_in_spoke.free()


    b_out_chute = bmesh.new()
    p_ga = Vector((0.0, -1.460, 0.262))
    p_gb = Vector((0.0, -1.402, 0.452))
    t_rg = (p_gb - p_ga).normalized()
    u_rg = Vector((1.0, 0.0, 0.0))
    v_rg = u_rg.cross(t_rg).normalized()
    if v_rg.y < 0.0:
        v_rg = -v_rg
    n_sc_steps = 44
    dr_sc, thr_sc = 0.032, 0.0030
    rings_vg = []
    for ks_c in range(n_sc_steps + 1):
        fc = ks_c / float(n_sc_steps)
        pc_cur = p_ga.lerp(p_gb, fc)
        root_fl = 0.0105 * math.exp(-((fc / 0.090) ** 2))
        tip_tap = 1.0 - 0.14 * _smooth01(0.88, 1.00, fc)
        lobes = 0.0
        for f_lc, rw_lc in ((0.24, 0.115), (0.53, 0.115), (0.81, 0.108)):
            du_l = abs(fc - f_lc) / rw_lc
            if du_l < 1.0:
                lobes += math.cos(0.5 * math.pi * du_l) ** 1.55
        wr_cur = (0.0130 + 0.0118 * min(1.0, lobes) + root_fl) * tip_tap
        lip_up = 0.0028 + 0.0022 * min(1.0, lobes)
        sec_pts_2d = [
            (+wr_cur + 0.0026, +lip_up),
            (+wr_cur,          0.0000),
            (+0.0028,          -dr_sc),
            (0.0000,           -dr_sc - 0.0032),
            (-0.0028,          -dr_sc),
            (-wr_cur,          0.0000),
            (-wr_cur - 0.0026, +lip_up),
            (-wr_cur + thr_sc, 0.0000),
            (0.0000,           -dr_sc + thr_sc * 1.35),
            (+wr_cur - thr_sc, 0.0000),
        ]
        rings_vg.append([b_out_chute.verts.new(pc_cur + u_rg * px_v + v_rg * py_v) for px_v, py_v in sec_pts_2d])
    for ks_c in range(n_sc_steps):
        ra_v, rb_v = rings_vg[ks_c], rings_vg[ks_c + 1]
        for j_v in range(10):
            jn_v = (j_v + 1) % 10
            b_out_chute.faces.new((ra_v[j_v], ra_v[jn_v], rb_v[jn_v], rb_v[j_v]))
    for ia_q, ib_q, ic_q, id_q in vg_cap_quads:
        b_out_chute.faces.new((rings_vg[0][id_q], rings_vg[0][ic_q], rings_vg[0][ib_q], rings_vg[0][ia_q]))
        b_out_chute.faces.new((rings_vg[-1][ia_q], rings_vg[-1][ib_q], rings_vg[-1][ic_q], rings_vg[-1][id_q]))
    bmesh.ops.recalc_face_normals(b_out_chute, faces=b_out_chute.faces)
    bm_add_segmented_tube(
        b_out_chute, Vector((0.0, -1.402, 0.452)), Vector((0.0, -1.398, 0.486)),
        [(0.0, 0.0072), (0.35, 0.0056), (1.0, 0.0052)], segs=16, cap_ends=True,
    )
    for i_rg in range(18):
        bm_merge(bm_fh, b_out_chute.copy(), Matrix.Rotation(2.0 * math.pi * i_rg / 18.0, 4, 'Y'))
    b_out_chute.free()


    b_torch = bmesh.new()
    prof_torch = [
        (-1.614, 0.0055), (-1.614, 0.0115), (-1.592, 0.0115), (-1.586, 0.0152),
        (-1.530, 0.0152), (-1.524, 0.0168), (-1.512, 0.0168), (-1.506, 0.0146),
        (-1.476, 0.0146), (-1.474, 0.0132), (-1.474, 0.0102), (-1.478, 0.0094),
        (-1.588, 0.0094), (-1.596, 0.0055),
    ]
    bm_add_revolved_profile(b_torch, prof_torch, segs=48, close_loop=True, bevel_r=0.00025)
    bm_merge(bm_fh, b_torch, Matrix.Rotation(math.radians(166.0), 4, 'Y') @ Matrix.Translation((0.0, 0.0, 0.204)))
    b_torch_mnt = bmesh.new()
    bm_add_clevis_plate(b_torch_mnt, (-1.532, 0.204), (-1.514, 0.254), 0.0040, 0.0040, -0.0035, +0.0035, arc_segs=9)
    bm_add_spline_tube(
        b_torch_mnt,
        [Vector((0.0, -1.614, 0.204)), Vector((0.016, -1.628, 0.236)), Vector((0.022, -1.608, 0.312)), Vector((0.018, -1.568, 0.438))],
        radius=0.0032, n_samples=24, segs=14, cap_ends=True,
    )
    bm_merge(bm_fh, b_torch_mnt, Matrix.Rotation(math.radians(166.0), 4, 'Y'))


    spray_ring_radii = [0.168, 0.220, 0.296, 0.366, 0.436]
    y_sp_ring = -1.569
    f0_spr_fh = len(bm_fh.faces)
    for r_sr in spray_ring_radii:
        prof_sr = [
            (y_sp_ring + 0.0044 * math.cos(2 * math.pi * s / 16), r_sr + 0.0044 * math.sin(2 * math.pi * s / 16))
            for s in range(16)
        ]
        bm_add_revolved_profile(bm_fh, prof_sr, segs=192, close_loop=True)
    bm_fill_unset_from(bm_fh, f0_spr_fh, MAT_RIGID_TUBE)

    bs_bar = bmesh.new()
    bm_add_segmented_tube(
        bs_bar, Vector((0, -1.550, 0.488)), Vector((0, -1.550, 0.146)),
        [(0.0, 0.0078), (0.06, 0.0045), (0.95, 0.0035), (1.0, 0.0020)], segs=18, cap_ends=True,
    )

    for r_sr_cl in spray_ring_radii:
        bm_add_clevis_plate(bs_bar, (-1.569, r_sr_cl), (-1.550, r_sr_cl), 0.0058, 0.0052, -0.0042, +0.0042, arc_segs=11)
        bm_add_chamfered_cyl_x(bs_bar, -1.569, r_sr_cl, -0.0068, +0.0068, radius=0.0060, segs=16, hollow_r=0.0043)
        bm_add_aerospace_bolt(bs_bar, Vector((0.0, -1.560, r_sr_cl + 0.0056)), Vector((0, 0, 1)), head_r=0.0016, height=0.0015)
    for nz_k in range(14):
        rz = 0.156 + nz_k * 0.0232
        side_x = 0.0058 if (nz_k % 2 == 0) else -0.0058
        bm_add_segmented_tube(
            bs_bar, Vector((0.0, -1.550, rz)), Vector((side_x, -1.541, rz)),
            [(0.0, 0.0025), (0.72, 0.0019), (1.0, 0.0008)], segs=14, cap_ends=True,
        )
    for i in range(18):
        bm_merge(bm_fh, bs_bar.copy(), Matrix.Rotation(2.0 * math.pi * (i + 0.5) / 18.0, 4, 'Y'), default_mat_id=MAT_RIGID_TUBE)
    bs_bar.free()


    f0_mxv = len(bm_fh.faces)
    bm_add_airfoil_stage(bm_fh, 36, y_center=-1.589, chord=0.032, r_hub=0.156, r_tip=0.254,
                         twist_root_deg=16.0, twist_tip_deg=28.0, camber=0.07)
    bm_fill_unset_from(bm_fh, f0_mxv, MAT_SPINNER_IGV)
    b_mix_tab = bmesh.new()
    for r_mt in (0.157, 0.252):
        _bm_add_box_x_yz(b_mix_tab, -0.0048, +0.0048, -1.606, -1.572, r_mt - 0.0015, r_mt + 0.0015, nx=1, ny=2, nz=1, chamfer=0.0003)
        bm_add_dome_rivet(b_mix_tab, Vector((0.0, -1.589, r_mt + (0.0016 if r_mt > 0.2 else -0.0016))),
                          Vector((0, 0, 1 if r_mt > 0.2 else -1)), radius=0.0011)
    for imx in range(36):
        bm_merge(bm_fh, b_mix_tab.copy(), Matrix.Rotation(2.0 * math.pi * imx / 36.0, 4, 'Y'))
    b_mix_tab.free()

    create_mesh_obj("Interior_VGutter_Flameholders_Detailed", bm_fh, coll_interior, clay)

    bm_gr = bmesh.new()
    for r_g, y_g in gutter_radii:
        prof_g = [
            (y_g - 0.022 + 0.0035 * math.cos(2 * math.pi * s / 12), r_g + 0.0035 * math.sin(2 * math.pi * s / 12))
            for s in range(12)
        ]
        bm_add_revolved_profile(bm_gr, prof_g, segs=96, close_loop=True)
    create_mesh_obj("Interior_Flameholder_PlasmaRings", bm_gr, coll_interior, mats["ab_glow_ring"])


    bm_cone = bmesh.new()
    prof_cone = [(-1.980, 0.188), (-1.980, 0.208)]
    n_ogive = 44
    for k in range(1, n_ogive + 1):
        t = k / float(n_ogive)
        yv = -1.980 + t * 0.445
        rv = 0.208 - (0.208 - 0.1140) * (t ** 0.88)
        if any(abs(t - seam_t) < 0.014 for seam_t in (0.26, 0.56, 0.82)):
            rv -= 0.0018
        prof_cone.append((yv, rv))

    prof_cone.extend([
        (-1.531, 0.1124), (-1.531, 0.1088), (-1.536, 0.1072),
        (-1.596, 0.1050), (-1.600, 0.0985), (-1.608, 0.0985), (-1.612, 0.1044),
        (-1.666, 0.1024), (-1.670, 0.0960), (-1.678, 0.0960), (-1.682, 0.1018),
        (-1.736, 0.0998), (-1.740, 0.0935), (-1.748, 0.0935), (-1.752, 0.0992),
        (-1.806, 0.0972), (-1.810, 0.0910), (-1.818, 0.0910), (-1.822, 0.0966),
        (-1.848, 0.0952), (-1.854, 0.0680), (-1.846, 0.0520), (-1.846, 0.0180),
        (-1.968, 0.0180), (-1.974, 0.1880),
    ])
    bm_add_revolved_profile(bm_cone, prof_cone, segs=180, close_loop=True, bevel_r=0.00030)


    cone_hb_pts = []
    for khb in range(12):
        ahb = 2.0 * math.pi * khb / 12.0
        p_chb = Vector((0.060 * math.sin(ahb), -1.848, 0.060 * math.cos(ahb)))
        cone_hb_pts.append(p_chb)
        bm_add_aerospace_bolt(bm_cone, p_chb, Vector((0, 1, 0)), head_r=0.0022, height=0.0022)
    for khb in range(0, 12, 2):
        bm_add_bolt_pair_lockwire(bm_cone, cone_hb_pts[khb], cone_hb_pts[khb + 1], Vector((0, 1, 0)), head_r=0.0022, strand_r=0.00028)


    u_edge_9 = [0.016, 0.055, 0.145, 0.285, 0.430, 0.570, 0.715, 0.855, 0.945, 0.984]
    n_loop_9 = 9 * len(u_edge_9)

    def _nonagon_shroud_pt(k_idx: int, vr: float, vt: float) -> Vector:
        s_side = k_idx // len(u_edge_9)
        u_loc  = u_edge_9[k_idx % len(u_edge_9)]
        ph_a   = math.radians(20.0 + s_side * 40.0)
        ph_b   = math.radians(20.0 + (s_side + 1) * 40.0)
        yv     = -1.582 + vt * 0.084
        r_vtx  = 0.1395 + 0.0055 * vt + vr * 0.0036
        xa, za = r_vtx * math.sin(ph_a), r_vtx * math.cos(ph_a)
        xb, zb = r_vtx * math.sin(ph_b), r_vtx * math.cos(ph_b)
        return Vector((xa + (xb - xa) * u_loc, yv, za + (zb - za) * u_loc))

    _bm_add_annular_quad_extrusion(bm_cone, _nonagon_shroud_pt, n_loop_9, [0.0, 0.20, 0.80, 1.0], [0.00, 0.05, 0.18, 0.50, 0.82, 0.95, 1.00])


    b_cstrut = bmesh.new()
    bm_add_clevis_plate(b_cstrut, (-1.574, 0.1155), (-1.514, 0.1415), 0.0080, 0.0080, -0.0024, +0.0024, arc_segs=9)
    for row_ac, y_ac in enumerate((-1.790, -1.725, -1.660, -1.595)):
        t_ac = (y_ac - (-1.980)) / 0.445
        r_ac = 0.208 - (0.208 - 0.1140) * (t_ac ** 0.88)
        for col_ac in range(7):
            ph_loc = math.radians(6.5 + col_ac * 4.5 + 1.5 * (row_ac % 2))
            n_ac = Vector((math.sin(ph_loc), 0.18, math.cos(ph_loc))).normalized()
            p_ac = Vector((r_ac * math.sin(ph_loc), y_ac, r_ac * math.cos(ph_loc)))
            bm_add_recessed_micro_port(b_cstrut, p_ac, n_ac, r_outer=0.0024, depth=0.0025, segs=12)
    for iv_c in range(9):
        bm_merge(bm_cone, b_cstrut.copy(), Matrix.Rotation(math.radians(20.0 + iv_c * 40.0), 4, 'Y'))
    b_cstrut.free()

    bm_add_airfoil_stage(bm_cone, 12, y_center=-1.920, chord=0.076, r_hub=0.188, r_tip=0.502,
                         twist_root_deg=-14.0, twist_tip_deg=-6.0, camber=-0.05)
    bm_add_airfoil_stage(bm_cone, 36, y_center=-2.120, chord=0.046, r_hub=0.205, r_tip=0.386,
                         twist_root_deg=-28.0, twist_tip_deg=-42.0, camber=-0.08)
    bm_add_airfoil_stage(bm_cone, 16, y_center=-2.207, chord=0.050, r_hub=0.194, r_tip=0.386,
                         twist_root_deg=12.0, twist_tip_deg=22.0, camber=0.05)

    prof_stator_brg = [
        (-2.080, 0.206), (-2.237, 0.194), (-2.237, 0.092), (-2.080, 0.092),
    ]
    bm_add_revolved_profile(bm_cone, prof_stator_brg, segs=128, close_loop=True, bevel_r=0.00045)
    create_mesh_obj("Interior_Stator_TailCone_And_GuideVanes", bm_cone, coll_interior, clay)


def build_lpt_rotor_mesh(mats: dict):

    clay = mats["clay"]
    bm_rot = bmesh.new()

    prof_rotor_shaft = [
        (-3.785, 0.044), (-3.785, 0.074), (-2.059, 0.074),
        (-2.051, 0.118), (-2.045, 0.184), (-2.061, 0.195), (-2.061, 0.206),
        (-2.007, 0.206), (-2.000, 0.206), (-2.000, 0.190), (-2.015, 0.182),
        (-2.010, 0.118), (-2.003, 0.074), (-1.999, 0.044),
    ]
    bm_add_revolved_profile(bm_rot, prof_rotor_shaft, segs=128, close_loop=True, bevel_r=0.00042)

    bm_add_airfoil_stage(bm_rot, 48, y_center=-2.033, chord=0.050, r_hub=0.205, r_tip=0.380,
                         twist_root_deg=24.0, twist_tip_deg=52.0, camber=0.09)
    prof_lpt_shroud = [
        (-2.057, 0.378), (-2.057, 0.383), (-2.041, 0.385), (-2.025, 0.385), (-2.009, 0.383), (-2.009, 0.378),
    ]
    bm_add_revolved_profile(bm_rot, prof_lpt_shroud, segs=192, close_loop=True, bevel_r=0.00030)

    disc_bolts = []
    for idb in range(24):
        adb = 2.0 * math.pi * idb / 24.0
        p_dcb = Vector((0.198 * math.sin(adb), -2.000, 0.198 * math.cos(adb)))
        disc_bolts.append(p_dcb)
        bm_add_aerospace_bolt(bm_rot, p_dcb, Vector((0, 1, 0)), head_r=0.0024, height=0.0024)
    for idb in range(0, 24, 2):
        bm_add_bolt_pair_lockwire(bm_rot, disc_bolts[idb], disc_bolts[idb + 1], Vector((0, 1, 0)), head_r=0.0024, strand_r=0.00030)


    f0_lnut = len(bm_rot.faces)
    bm_add_bevel_locked_hex(
        bm_rot, Vector((0.0, -2.002, 0.0)), Vector((0.0, -1.991, 0.0)),
        [(0.0, 0.072), (0.20, 0.086), (0.80, 0.086), (1.0, 0.070)], corner_bevel=0.14, cap_ends=True,
    )
    for kt_l in range(12):
        b_ltab = bmesh.new()
        _bm_add_box_x_yz(b_ltab, -0.0045, +0.0045, -2.004, -2.001, 0.078, 0.098, nx=1, ny=1, nz=1, chamfer=0.0004)
        bm_merge(bm_rot, b_ltab, Matrix.Rotation(2.0 * math.pi * kt_l / 12.0, 4, 'Y'))
    bm_fill_unset_from(bm_rot, f0_lnut, MAT_BRASS_FITTING)

    return finalize_shared_mesh("Mesh_LPT_Rotor_Shaft", bm_rot, clay)


def build_fan_rotor_3stage_mesh(mats: dict):

    clay = mats["clay"]
    bm_frot = bmesh.new()


    prof_fd1 = [
        (-4.137, 0.1830), (-4.137, 0.1915), (-4.134, 0.1915), (-4.133, 0.1895),
        (-4.130, 0.1895), (-4.129, 0.1925), (-4.079, 0.1925), (-4.079, 0.1830),
    ]
    bm_add_revolved_profile(bm_frot, prof_fd1, segs=136, close_loop=True, bevel_r=0.00032)
    frot_bolts = []
    for ib_f in range(24):
        abf = 2.0 * math.pi * ib_f / 24.0
        p_fb_r = Vector((0.1875 * math.sin(abf), -4.137, 0.1875 * math.cos(abf)))
        frot_bolts.append(p_fb_r)
        bm_add_aerospace_bolt(bm_frot, p_fb_r, Vector((0.0, -1.0, 0.0)), head_r=0.0020, height=0.0014)
    for ib_f in range(0, 24, 2):
        bm_add_bolt_pair_lockwire(bm_frot, frot_bolts[ib_f], frot_bolts[ib_f + 1], Vector((0.0, -1.0, 0.0)), head_r=0.0020, strand_r=0.00026)

    for y_d0, y_d1, r_din, r_dout in [
        (-3.980, -3.924, 0.206, 0.214),
        (-3.840, -3.792, 0.228, 0.236),
    ]:
        prof_fd = [(y_d0, r_din), (y_d0, r_dout), (y_d1, r_dout), (y_d1, r_din)]
        bm_add_revolved_profile(bm_frot, prof_fd, segs=128, close_loop=True, bevel_r=0.00035)


    bm_add_airfoil_stage(bm_frot, 34, y_center=-4.108, chord=0.104, r_hub=0.195, r_tip=0.473,
                         twist_root_deg=42.0, twist_tip_deg=66.0, camber=0.08, flare_tip=True)
    b_rplat = bmesh.new()
    b_rp_box = bmesh.new()
    _bm_add_box_x_yz(b_rp_box, -0.0072, +0.0072, -0.028, +0.028, 0.1916, 0.1962, nx=2, ny=3, nz=1, chamfer=0.0006)
    bm_merge(b_rplat, b_rp_box, Matrix.Translation((0.0, -4.108, 0.0)) @ Matrix.Rotation(math.radians(38.0), 4, 'Z'))
    b_clapper = bmesh.new()

    _bm_add_box_x_yz(b_clapper, -0.0292, +0.0292, -0.0082, +0.0082, -0.0046, +0.0046, nx=4, ny=2, nz=2, chamfer=0.0007)
    _bm_add_box_x_yz(b_clapper, -0.0298, -0.0175, -0.0138, +0.0032, -0.0052, +0.0052, nx=2, ny=2, nz=2, chamfer=0.0006)
    _bm_add_box_x_yz(b_clapper, +0.0175, +0.0298, -0.0032, +0.0138, -0.0052, +0.0052, nx=2, ny=2, nz=2, chamfer=0.0006)
    bm_merge(b_rplat, b_clapper, Matrix.Translation((0.0, -4.108, 0.352)) @ Matrix.Rotation(math.radians(56.0), 4, 'Z'))
    for isn in range(34):
        bm_merge(bm_frot, b_rplat.copy(), Matrix.Rotation(2.0 * math.pi * isn / 34.0, 4, 'Y'), default_mat_id=MAT_MACHINED_ALU)
    b_rplat.free()

    bm_add_airfoil_stage(bm_frot, 40, y_center=-3.952, chord=0.072, r_hub=0.214, r_tip=0.463,
                         twist_root_deg=38.0, twist_tip_deg=60.0, camber=0.07, flare_tip=True)
    bm_add_airfoil_stage(bm_frot, 46, y_center=-3.817, chord=0.062, r_hub=0.236, r_tip=0.454,
                         twist_root_deg=38.0, twist_tip_deg=58.0, camber=0.06, flare_tip=True)

    return finalize_shared_mesh("Mesh_Fan_Rotor_3Stage", bm_frot, clay)


def add_highres_flap_shell(
    bm: bmesh.types.BMesh,
    p_start: tuple,
    p_end: tuple,
    r_closed_start: float,
    r_closed_end: float,
    r_open_start: float,
    r_open_end: float,
    n_sectors: int,
    width_factor: float,
    thickness: float,
    is_outer_master: bool,
    flap_role: str,
    n_long: int = 80,
    n_lat: int = 48,
):

    dy = p_end[0] - p_start[0]
    dz = p_end[1] - p_start[1]
    length = math.hypot(dy, dz)
    ty, tz = dy / length, dz / length
    ny, nz = -tz, ty
    sin_hp = math.sin(math.pi / n_sectors)

    top_grid = []
    bot_grid = []
    base_s = subdivide_1d_with_support_loops(n_long, sup1=0.004, sup2=0.013)

    rib_cfg = {
        'DIV_MASTER':  (0.05, 0.85, 22, 0.25, 0.18),
        'CONV_MASTER': (0.06, 0.86, 12, 0.25, 0.18),
        'DIV_SEAL':    (0.05, 0.89, 26, 0.26, 0.19),
        'CONV_SEAL':   (0.05, 0.89, 14, 0.26, 0.19),
    }.get(flap_role)
    if rib_cfg is not None:
        rs0, rspan, rcnt, d_ft, d_cr = rib_cfg
        extra_s = []
        for rk in range(rcnt + 1):
            sc_r = rs0 + (rk / float(rcnt)) * rspan
            pitch_r = rspan / float(rcnt)
            for df_r in (-d_ft, -d_cr, +d_cr, +d_ft):
                sv_r = sc_r + df_r * pitch_r
                if rs0 < sv_r < rs0 + rspan:
                    extra_s.append(sv_r)
        merged_s = sorted(base_s + extra_s)
        s_vals = [merged_s[0]]
        for sv in merged_s[1:]:
            if sv - s_vals[-1] >= 0.00085 or sv > 0.9999:
                s_vals.append(sv)
        s_vals[-1] = 1.0
    else:
        s_vals = base_s
    u_vals = subdivide_1d_with_support_loops(n_lat,  sup1=0.005, sup2=0.016)


    if 'CONV' in flap_role:
        s_ext_0, s_ext_1 = (-0.002 if is_outer_master else 0.002), 0.993
    else:
        s_ext_0, s_ext_1 = 0.007, 1.006

    for s in s_vals:
        s_ext = s_ext_0 + s * (s_ext_1 - s_ext_0)
        s_c = max(0.0, min(1.0, s_ext))
        cy = p_start[0] + dy * s_ext
        cz = p_start[1] + dz * s_ext

        rc = (1.0 - s_c) * r_closed_start + s_c * r_closed_end
        ro = (1.0 - s_c) * r_open_start + s_c * r_open_end
        half_w = width_factor * rc * sin_hp
        if flap_role == 'DIV_MASTER' and s > 0.92:
            half_w *= (1.0 - 0.036 * ((s - 0.92) / 0.08))

        rho = (1.08 * ro) if is_outer_master else (0.93 * rc)

        lip_thin = 1.0 - (0.52 * (((s - 0.88) / 0.12) ** 1.35) if ('DIV' in flap_role and s > 0.88) else 0.0)
        th_local = thickness * lip_thin

        row_top, row_bot = [], []
        for u_lat in u_vals:
            eta = -1.0 + 2.0 * u_lat
            x = eta * half_w
            arch_drop = -(x * x) / (2.0 * rho)


            a_eta_in = abs(eta)
            lat_liner_env = max(0.0, min(1.0, (0.86 - a_eta_in) / 0.06)) if is_outer_master else max(0.0, min(1.0, (0.90 - a_eta_in) / 0.06))
            ax_liner_env = math.sin(math.pi * min(1.0, max(0.0, (s - 0.02) / 0.96))) ** 0.35
            n_corr_ch = 14.0 if is_outer_master else 12.0
            ch_wave = math.cos(2.0 * math.pi * n_corr_ch * u_lat)
            trap_corr = max(-1.0, min(1.0, 1.85 * ch_wave))
            seam_band = 0.0
            for s_seam in (0.34, 0.68):
                if abs(s - s_seam) < 0.014:
                    seam_band = -0.00075 * (1.0 - abs(s - s_seam) / 0.014)
            inner_groove = (-0.00078 * (0.5 + 0.5 * trap_corr) + seam_band) * lat_liner_env * ax_liner_env

            if is_outer_master:
                n_bot = arch_drop - inner_groove
                n_top = arch_drop + th_local
                spine_env = math.sin(math.pi * min(1.0, max(0.0, (s - 0.025) / 0.95))) ** 0.45
                ax = abs(x)
                a_eta = abs(eta)

                if flap_role == 'DIV_MASTER':

                    if ax < 0.0165:
                        plat = 1.0 if ax < 0.0068 else max(0.0, 1.0 - (ax - 0.0068) / 0.0097)
                        crown_dip = -0.0014 * max(0.0, 1.0 - (ax / 0.0038) ** 2)
                        n_top += (0.0150 * plat + crown_dip) * spine_env

                    elif 0.0165 <= ax and a_eta < 0.84 and 0.05 < s < 0.90:
                        rib_phase = (s - 0.05) / 0.85 * 22.0
                        d_rib = abs(rib_phase - round(rib_phase))
                        rib_trap = min(1.0, max(0.0, (0.25 - d_rib) / 0.07))
                        lat_fade = min(1.0, max(0.0, min((a_eta - 0.16) / 0.05, (0.84 - a_eta) / 0.05)))
                        n_top += 0.0024 * rib_trap * lat_fade

                    if a_eta > 0.85:
                        edge_step = min(1.0, (a_eta - 0.85) / 0.035)
                        n_top += 0.0019 * edge_step * spine_env

                elif flap_role == 'CONV_MASTER':

                    d_rail = abs(ax - 0.023)
                    if d_rail < 0.0085:
                        rail_p = 1.0 if d_rail < 0.0035 else max(0.0, 1.0 - (d_rail - 0.0035) / 0.0050)
                        n_top += 0.0052 * rail_p * spine_env
                    elif a_eta < 0.84 and 0.06 < s < 0.92:
                        rib_phase = (s - 0.06) / 0.86 * 12.0
                        d_rib = abs(rib_phase - round(rib_phase))
                        rib_trap = min(1.0, max(0.0, (0.25 - d_rib) / 0.07))
                        n_top += 0.0021 * rib_trap * spine_env
                    if a_eta > 0.86:
                        n_top += 0.0016 * min(1.0, (a_eta - 0.86) / 0.035) * spine_env
            else:

                spine_env_s = math.sin(math.pi * min(1.0, max(0.0, (s - 0.03) / 0.93))) ** 0.50
                ax_s = abs(x)
                a_eta_s = abs(eta)
                wear_groove = -0.0004 * (1.0 if (0.68 < a_eta_s < 0.72 or 0.84 < a_eta_s < 0.88) else 0.0)
                dorsal_bump = 0.0
                if ax_s < 0.0086:
                    plat_s = 1.0 if ax_s < 0.0038 else max(0.0, 1.0 - (ax_s - 0.0038) / 0.0048)
                    dip_s = -0.0005 * max(0.0, 1.0 - (ax_s / 0.0020) ** 2)
                    dorsal_bump = (0.0022 * plat_s + dip_s) * spine_env_s
                elif a_eta_s < 0.66 and 0.05 < s < 0.94:
                    n_ribs_s = 26.0 if flap_role == 'DIV_SEAL' else 14.0
                    rib_ph_s = (s - 0.05) / 0.89 * n_ribs_s
                    d_rb_s = abs(rib_ph_s - round(rib_ph_s))
                    rib_tr_s = min(1.0, max(0.0, (0.26 - d_rb_s) / 0.07))
                    lat_fd_s = min(1.0, max(0.0, min((a_eta_s - 0.10) / 0.05, (0.66 - a_eta_s) / 0.06)))
                    dorsal_bump = 0.00165 * rib_tr_s * lat_fd_s * spine_env_s
                elif 0.66 <= a_eta_s < 0.71:
                    dorsal_bump = 0.0008 * spine_env_s

                rolled_lip_s = 0.0
                if flap_role == 'DIV_SEAL' and s > 0.968:
                    rolled_lip_s = 0.0014 * (((s - 0.968) / 0.032) ** 1.5)

                n_top = arch_drop + wear_groove + dorsal_bump + rolled_lip_s
                n_bot = arch_drop - th_local - inner_groove + 0.50 * rolled_lip_s

            row_top.append(bm.verts.new((x, cy + ny * n_top, cz + nz * n_top)))
            row_bot.append(bm.verts.new((x, cy + ny * n_bot, cz + nz * n_bot)))

        top_grid.append(row_top)
        bot_grid.append(row_bot)

    bm_build_beveled_quad_shell(bm, bot_grid, top_grid, bulge_scale=0.18)


def build_shared_component_meshes(spec: NozzleSpec, mats: dict) -> dict:

    clay = mats["clay"]
    st0 = spec.state_closed
    st1 = spec.state_open
    fh = spec.fork_half_span
    sin_hp = math.sin(math.pi / spec.n_sectors)

    p0 = st0["P0"]
    p1_0 = st0["P1"]
    p2_0 = st0["P2"]
    pc_0 = st0["PC"]
    pd_0 = st0["PD"]
    ps = st0["PS"]
    pb = st0["PB"]
    ain_0 = st0["A_in"]
    aout_0 = st0["A_out"]
    pring_0 = st0["P_ring"]

    r_p0 = p0[1]
    r_p1_0, r_p1_1 = p1_0[1], st1["P1"][1]
    r_p2_0, r_p2_1 = p2_0[1], st1["P2"][1]

    meshes = {}

    def _add_comb_finger_seals(bm_dst: bmesh.types.BMesh, p_st: tuple, p_en: tuple,
                               rc0: float, rc1: float, ro0: float, ro1: float,
                               n_teeth: int, s0: float = 0.05, s1: float = 0.94):

        dy_f, dz_f = p_en[0] - p_st[0], p_en[1] - p_st[1]
        len_f = math.hypot(dy_f, dz_f)
        ny_f, nz_f = -dz_f / len_f, dy_f / len_f
        pitch = (s1 - s0) / n_teeth
        tooth_span = pitch * 0.74


        t_strip_s = subdivide_1d_with_support_loops(10, sup1=0.02, sup2=0.07)
        t_strip_w = subdivide_1d_with_support_loops(3,  sup1=0.05, sup2=0.16)
        for side_sgn in (-1.0, +1.0):
            gb_st, gt_st = [], []
            for fs_st in t_strip_s:
                s_cur = s0 + (s1 - s0) * fs_st
                cy = p_st[0] + dy_f * s_cur
                cz = p_st[1] + dz_f * s_cur
                rc_cur = (1.0 - s_cur) * rc0 + s_cur * rc1
                ro_cur = (1.0 - s_cur) * ro0 + s_cur * ro1
                rho_cur = 1.08 * ro_cur
                hw_m = 0.945 * rc_cur * sin_hp
                rb_st, rt_st = [], []
                for fw_st in t_strip_w:
                    xp = side_sgn * (0.875 + 0.075 * fw_st) * hw_m
                    ad_p = -(xp * xp) / (2.0 * rho_cur)
                    rb_st.append(bm_dst.verts.new((xp, cy + ny_f * (ad_p + 0.0056), cz + nz_f * (ad_p + 0.0056))))
                    rt_st.append(bm_dst.verts.new((xp, cy + ny_f * (ad_p + 0.0073), cz + nz_f * (ad_p + 0.0073))))
                gb_st.append(rb_st)
                gt_st.append(rt_st)
            bm_build_beveled_quad_shell(bm_dst, gb_st, gt_st, bulge_scale=0.16)

        t_tooth_s = subdivide_1d_with_support_loops(2, sup1=0.05, sup2=0.16)
        t_tooth_w = [0.00, 0.05, 0.18, 0.34, 0.52, 0.78, 0.92, 0.97, 1.00]

        for ti in range(n_teeth):
            sa = s0 + ti * pitch
            sb = sa + tooth_span
            for side_sgn in (-1.0, +1.0):
                g_cbot, g_ctop = [], []
                for fs in t_tooth_s:
                    s_cur = sa + (sb - sa) * fs
                    cy = p_st[0] + dy_f * s_cur
                    cz = p_st[1] + dz_f * s_cur
                    rc_cur = (1.0 - s_cur) * rc0 + s_cur * rc1
                    ro_cur = (1.0 - s_cur) * ro0 + s_cur * ro1
                    rho_cur = 1.08 * ro_cur
                    x_root = side_sgn * (0.915 * 0.945 * rc_cur * sin_hp)
                    x_tip  = side_sgn * (0.958 * rc_cur * sin_hp)
                    rb_row, rt_row = [], []
                    for fw in t_tooth_w:
                        xp = x_root + (x_tip - x_root) * fw
                        drop_p = -(xp * xp) / (2.0 * rho_cur)

                        step_down = -0.00105 * min(1.0, max(0.0, (fw - 0.16) / 0.28))
                        tip_curl = +0.0006 * max(0.0, (fw - 0.80) / 0.20) ** 1.5
                        nb = drop_p + 0.0038 + step_down + tip_curl
                        nt = nb + 0.00155
                        rb_row.append(bm_dst.verts.new((xp, cy + ny_f * nb, cz + nz_f * nb)))
                        rt_row.append(bm_dst.verts.new((xp, cy + ny_f * nt, cz + nz_f * nt)))
                    g_cbot.append(rb_row)
                    g_ctop.append(rt_row)
                bm_build_beveled_quad_shell(bm_dst, g_cbot, g_ctop, bulge_scale=0.18)


    bm_cm = bmesh.new()
    add_highres_flap_shell(
        bm_cm, p0, p1_0, r_p0, r_p1_0, r_p0, r_p1_1,
        spec.n_sectors, width_factor=0.945, thickness=0.0044,
        is_outer_master=True, flap_role='CONV_MASTER', n_long=104, n_lat=76,
    )
    t1_0 = st0["theta1"]
    n1_y, n1_z = -math.sin(t1_0), math.cos(t1_0)
    n1_vec = Vector((0.0, n1_y, n1_z))


    _add_comb_finger_seals(bm_cm, p0, p1_0, r_p0, r_p1_0, r_p0, r_p1_1, n_teeth=11, s0=0.06, s1=0.63)
    for rv_i in range(11):
        sf = 0.08 + 0.53 * (rv_i / 10.0)
        cy = p0[0] + (p1_0[0] - p0[0]) * sf
        cz = p0[1] + (p1_0[1] - p0[1]) * sf
        rc_s = (1.0 - sf) * r_p0 + sf * r_p1_0
        ro_s = (1.0 - sf) * r_p0 + sf * r_p1_1
        xw = 0.945 * rc_s * sin_hp * 0.912
        drop = -(xw * xw) / (2.0 * 1.08 * ro_s) + 0.0073
        for sx in (-xw, +xw):
            bm_add_dome_rivet(bm_cm, Vector((sx, cy + n1_y * drop, cz + n1_z * drop)), n1_vec, radius=0.0017)


    base_pc = (
        p0[0] + spec.conv_lug_a * math.cos(t1_0) + n1_y * 0.0108,
        p0[1] + spec.conv_lug_a * math.sin(t1_0) + n1_z * 0.0108,
    )
    pc_pad_f = (
        p0[0] + (spec.conv_lug_a - 0.025) * math.cos(t1_0) + n1_y * 0.0072,
        p0[1] + (spec.conv_lug_a - 0.025) * math.sin(t1_0) + n1_z * 0.0072,
    )
    pc_pad_b = (
        p0[0] + (spec.conv_lug_a + 0.025) * math.cos(t1_0) + n1_y * 0.0072,
        p0[1] + (spec.conv_lug_a + 0.025) * math.sin(t1_0) + n1_z * 0.0072,
    )
    for side_s in (-1.0, 1.0):
        xc = side_s * fh
        bm_add_clevis_plate(
            bm_cm, pc_pad_f, pc_pad_b, 0.0036, 0.0036,
            xc - 0.0112, xc + side_s * 0.0168 if side_s > 0 else xc - 0.0168,
            arc_segs=9,
        ) if False else None
        x_pad_lo = (xc - 0.0112) if side_s > 0 else (xc - 0.0168)
        x_pad_hi = (xc + 0.0168) if side_s > 0 else (xc + 0.0112)
        bm_add_clevis_plate(bm_cm, pc_pad_f, pc_pad_b, 0.0036, 0.0036, x_pad_lo, x_pad_hi, arc_segs=9)
        for sub_off in (-0.0082, +0.0082):
            bm_add_clevis_plate(
                bm_cm, base_pc, pc_0, 0.0085, 0.0108,
                xc + sub_off - 0.0020, xc + sub_off + 0.0020, arc_segs=21,
            )
            bm_add_chamfered_cyl_x(
                bm_cm, pc_0[0], pc_0[1],
                xc + sub_off - 0.0023, xc + sub_off + 0.0023,
                radius=0.0082, segs=22, hollow_r=0.0046,
            )
        pc_bolts = []
        x_bolt_out = xc + side_s * 0.0135
        for b_ds in (-0.016, +0.016):
            by = p0[0] + (spec.conv_lug_a + b_ds) * math.cos(t1_0) + n1_y * 0.0106
            bz = p0[1] + (spec.conv_lug_a + b_ds) * math.sin(t1_0) + n1_z * 0.0106
            p_cb = Vector((x_bolt_out, by, bz))
            pc_bolts.append(p_cb)
            bm_add_aerospace_bolt(bm_cm, p_cb, n1_vec, head_r=0.0024, height=0.0024)
        bm_add_bolt_pair_lockwire(bm_cm, pc_bolts[0], pc_bolts[1], n1_vec, head_r=0.0024, strand_r=0.00034)
        bm_add_clevis_pin_assembly_x(bm_cm, pc_0[0], pc_0[1], xc - 0.0112, xc + 0.0112, pin_r=0.0046, segs=24)


    def _add_hotgas_tile_studs(bm_dst: bmesh.types.BMesh, p_st: tuple, p_en: tuple,
                               rc0: float, rc1: float, ro0: float, ro1: float,
                               is_master: bool, n_tiles_ax: int, x_etas: tuple):
        dy_t, dz_t = p_en[0] - p_st[0], p_en[1] - p_st[1]
        len_t = math.hypot(dy_t, dz_t)
        ty_t, tz_t = dy_t / len_t, dz_t / len_t
        ny_t, nz_t = -tz_t, ty_t
        n_in_vec = Vector((0.0, -ny_t, -nz_t))
        wf = 0.945 if is_master else 0.948
        is_div = (n_tiles_ax >= 7)


        eta_lim_t = 0.38 if is_master else 0.88
        n_flutes = 6 if is_master else 12
        base_u_t = subdivide_1d_with_support_loops(32, sup1=0.015, sup2=0.045)
        flute_u = []
        for fk in range(n_flutes + 1):
            uc_f = 0.06 + (fk / float(n_flutes)) * 0.88
            pf_u = 0.88 / float(n_flutes)
            for df_u in (-0.26, -0.17, +0.17, +0.26):
                uv_f = uc_f + df_u * pf_u
                if 0.06 < uv_f < 0.94:
                    flute_u.append(uv_f)
        merged_ut = sorted(base_u_t + flute_u)
        u_tile_vals = [merged_ut[0]]
        for uv in merged_ut[1:]:
            if uv - u_tile_vals[-1] >= 0.0022 or uv > 0.9999:
                u_tile_vals.append(uv)
        u_tile_vals[-1] = 1.0

        bays_ax = (
            [(0.032, 0.340), (0.348, 0.664), (0.672, 0.992)]
            if is_div else
            [(0.042, 0.334), (0.342, 0.642), (0.650, 0.936)]
        )
        t_bay_s = subdivide_1d_with_support_loops(6, sup1=0.035, sup2=0.11)

        f0_hgt = len(bm_dst.faces)
        for bay_idx, (sb0, sb1) in enumerate(bays_ax):
            gb_tl, gt_tl = [], []
            for fs_b in t_bay_s:
                sc = sb0 + (sb1 - sb0) * fs_b
                cy = p_st[0] + dy_t * sc
                cz = p_st[1] + dz_t * sc
                rc_c = (1.0 - sc) * rc0 + sc * rc1
                ro_c = (1.0 - sc) * ro0 + sc * ro1
                rho_c = (1.08 * ro_c) if is_master else (0.93 * rc_c)
                hw_c = wf * rc_c * sin_hp
                lip_c = 1.0 - (0.52 * (((sc - 0.88) / 0.12) ** 1.35) if (is_div and sc > 0.88) else 0.0)
                shell_in = 0.0 if is_master else (-0.0036 * lip_c)

                louver_tilt = (-0.00048 if is_master else -0.00095) * fs_b
                standoff = (-0.00028 if is_master else -0.00070) + louver_tilt
                th_tile = 0.00095 if is_master else 0.00145
                rb_r, rt_r = [], []
                for fu_t in u_tile_vals:
                    et_v = (-1.0 + 2.0 * fu_t) * eta_lim_t
                    xv = et_v * hw_c
                    ad_c = -(xv * xv) / (2.0 * rho_c)
                    fl_groove = 0.0
                    if 0.06 < fu_t < 0.94:
                        fph = (fu_t - 0.06) / 0.88 * float(n_flutes)
                        d_fp = abs(fph - round(fph))
                        fl_groove = (0.00085 if is_master else 0.00115) * min(1.0, max(0.0, (0.26 - d_fp) / 0.09))
                    elif abs(-1.0 + 2.0 * fu_t) > 0.90:
                        fl_groove = +(0.00032 if is_master else 0.00052)
                    n_outer = ad_c + shell_in + standoff
                    n_inner = n_outer - th_tile - fl_groove
                    rb_r.append(bm_dst.verts.new((xv, cy + ny_t * n_inner, cz + nz_t * n_inner)))
                    rt_r.append(bm_dst.verts.new((xv, cy + ny_t * n_outer, cz + nz_t * n_outer)))
                gb_tl.append(rb_r)
                gt_tl.append(rt_r)
            bm_build_beveled_quad_shell(bm_dst, gb_tl, gt_tl, bulge_scale=0.15)


            sc_mid = 0.5 * (sb0 + sb1)
            cy_m = p_st[0] + dy_t * sc_mid
            cz_m = p_st[1] + dz_t * sc_mid
            rc_m = (1.0 - sc_mid) * rc0 + sc_mid * rc1
            ro_m = (1.0 - sc_mid) * ro0 + sc_mid * ro1
            rho_m = (1.08 * ro_m) if is_master else (0.93 * rc_m)
            hw_m = wf * rc_m * sin_hp
            lip_m = 1.0 - (0.52 * (((sc_mid - 0.88) / 0.12) ** 1.35) if (is_div and sc_mid > 0.88) else 0.0)
            sh_m = 0.0 if is_master else (-0.0036 * lip_m)
            for et_v in x_etas:
                xv = et_v * eta_lim_t * 1.10 * hw_m
                n_s = -(xv * xv) / (2.0 * rho_m) + sh_m - (0.00085 if is_master else 0.0021)
                yz_a = (cy_m - ty_t * 0.0044 + ny_t * n_s, cz_m - tz_t * 0.0044 + nz_t * n_s)
                yz_b = (cy_m + ty_t * 0.0044 + ny_t * n_s, cz_m + tz_t * 0.0044 + nz_t * n_s)
                bm_add_clevis_plate(bm_dst, yz_a, yz_b, 0.00045, 0.00045, xv - 0.0028, xv + 0.0028, arc_segs=8)
                bm_add_dome_rivet(bm_dst, Vector((xv, cy_m + ny_t * (n_s - 0.00038), cz_m + nz_t * (n_s - 0.00038))), n_in_vec, radius=0.0009)

            if bay_idx < 2:
                sc_lv = sb1
                cy_lv = p_st[0] + dy_t * sc_lv
                cz_lv = p_st[1] + dz_t * sc_lv
                rc_lv = (1.0 - sc_lv) * rc0 + sc_lv * rc1
                ro_lv = (1.0 - sc_lv) * ro0 + sc_lv * ro1
                rho_lv = (1.08 * ro_lv) if is_master else (0.93 * rc_lv)
                hw_lv = wf * rc_lv * sin_hp
                sh_lv = 0.0 if is_master else -0.0036
                for et_sp in (-0.65, -0.32, 0.0, +0.32, +0.65):
                    x_sp = et_sp * eta_lim_t * hw_lv
                    n_sp = -(x_sp * x_sp) / (2.0 * rho_lv) + sh_lv - (0.00038 if is_master else 0.00105)
                    yz_s0 = (cy_lv - ty_t * 0.0025 + ny_t * n_sp, cz_lv - tz_t * 0.0025 + nz_t * n_sp)
                    yz_s1 = (cy_lv + ty_t * 0.0045 + ny_t * n_sp, cz_lv + tz_t * 0.0045 + nz_t * n_sp)
                    bm_add_clevis_plate(bm_dst, yz_s0, yz_s1, 0.00038, 0.00038, x_sp - 0.0008, x_sp + 0.0008, arc_segs=6)

        bm_fill_unset_from(bm_dst, f0_hgt, MAT_HOTGAS_TILE)

        if is_div:
            hw_o_ld, hh_o_ld, hw_i_ld, hh_i_ld = (
                (0.0024, 0.00038, 0.0017, 0.00018)
                if is_master else
                (0.0028, 0.00055, 0.0020, 0.00028)
            )
            hw_end = wf * rc1 * sin_hp
            rho_end = (1.08 * ro1) if is_master else (0.93 * rc1)
            n2_v_loc = Vector((0.0, ny_t, nz_t))
            n_ducts = 5 if is_master else 9
            et_max_d = 0.30 if is_master else 0.76
            for k_ld in range(n_ducts):
                et_ld = -et_max_d + 2.0 * et_max_d * (k_ld / float(n_ducts - 1))
                x_ld = et_ld * hw_end
                n_ld = -(x_ld * x_ld) / (2.0 * rho_end) - (0.00042 if is_master else 0.0024)
                pa_ld = Vector((x_ld, p_en[0] - ty_t * 0.011 + ny_t * n_ld, p_en[1] - tz_t * 0.011 + nz_t * n_ld))
                pb_ld = Vector((x_ld, p_en[0] + ty_t * 0.0022 + ny_t * n_ld, p_en[1] + tz_t * 0.0022 + nz_t * n_ld))
                bm_add_hollow_rect_duct(
                    bm_dst, pa_ld, pb_ld, Vector((1.0, 0.0, 0.0)), n2_v_loc,
                    hw_o_ld, hh_o_ld, hw_i_ld, hh_i_ld, n_steps=3,
                )

    def _add_throat_shingle_lip(bm_dst: bmesh.types.BMesh, p_st: tuple, p_en: tuple,
                                rc0: float, rc1: float, ro0: float, ro1: float,
                                is_master: bool, eta_lim: float):

        dy_l, dz_l = p_en[0] - p_st[0], p_en[1] - p_st[1]
        len_l = math.hypot(dy_l, dz_l)
        ny_l, nz_l = -dz_l / len_l, dy_l / len_l
        wf_l = 0.945 if is_master else 0.948
        t_sh_s = subdivide_1d_with_support_loops(6, sup1=0.035, sup2=0.11)
        t_sh_e = subdivide_1d_with_support_loops(26, sup1=0.020, sup2=0.06)
        g_sh_b, g_sh_t = [], []
        for fs_l in t_sh_s:
            sc_l = (0.940 + 0.055 * fs_l) if is_master else (0.948 + 0.072 * fs_l)
            sc_cl = min(1.0, sc_l)
            cy_l = p_st[0] + dy_l * sc_l
            cz_l = p_st[1] + dz_l * sc_l
            rc_l = (1.0 - sc_cl) * rc0 + sc_cl * rc1
            ro_l = (1.0 - sc_cl) * ro0 + sc_cl * ro1
            rho_l = (1.08 * ro_l) if is_master else (0.93 * rc_l)
            hw_l = wf_l * rc_l * sin_hp
            curve_out = (0.00035 if is_master else 0.00075) * (fs_l ** 1.3)
            base_off = -0.0018 if is_master else -0.0064
            th_sh = 0.00075 if is_master else 0.0012
            flute_env = math.sin(math.pi * min(1.0, max(0.0, (fs_l - 0.08) / 0.84)))
            rb_l, rt_l = [], []
            for fe_l in t_sh_e:
                et_l = -eta_lim + 2.0 * eta_lim * fe_l
                xl = et_l * hw_l
                ad_l = -(xl * xl) / (2.0 * rho_l)
                fl_w = max(-1.0, min(1.0, 2.2 * math.cos(2.0 * math.pi * 8.0 * fe_l)))
                fl_sh = (-0.00020 if is_master else -0.00038) * (0.5 + 0.5 * fl_w) * flute_env * (1.0 if 0.08 < fe_l < 0.92 else 0.0)
                nb_l = ad_l + base_off + curve_out + fl_sh
                nt_l = ad_l + base_off + curve_out + th_sh
                rb_l.append(bm_dst.verts.new((xl, cy_l + ny_l * nb_l, cz_l + nz_l * nb_l)))
                rt_l.append(bm_dst.verts.new((xl, cy_l + ny_l * nt_l, cz_l + nz_l * nt_l)))
            g_sh_b.append(rb_l)
            g_sh_t.append(rt_l)
        bm_build_beveled_quad_shell(bm_dst, g_sh_b, g_sh_t, bulge_scale=0.16)

    def _add_master_edge_seal_keepers(bm_dst: bmesh.types.BMesh, p_st: tuple, p_en: tuple,
                                      rc0: float, rc1: float, ro0: float, ro1: float, stations_k: tuple):

        dy_k, dz_k = p_en[0] - p_st[0], p_en[1] - p_st[1]
        len_k = math.hypot(dy_k, dz_k)
        ty_k, tz_k = dy_k / len_k, dz_k / len_k
        ny_k, nz_k = -tz_k, ty_k
        n_vec_k = Vector((0.0, ny_k, nz_k))
        for sc_k in stations_k:
            cy_k = p_st[0] + dy_k * sc_k
            cz_k = p_st[1] + dz_k * sc_k
            rc_k = (1.0 - sc_k) * rc0 + sc_k * rc1
            ro_k = (1.0 - sc_k) * ro0 + sc_k * ro1
            hw_k = 0.945 * rc_k * sin_hp
            for sx_k in (-1.0, +1.0):
                xk = sx_k * 0.885 * hw_k
                ad_k = -(xk * xk) / (2.0 * 1.08 * ro_k) + 0.0066
                yz_ka = (cy_k - ty_k * 0.0075 + ny_k * ad_k, cz_k - tz_k * 0.0075 + nz_k * ad_k)
                yz_kb = (cy_k + ty_k * 0.0075 + ny_k * ad_k, cz_k + tz_k * 0.0075 + nz_k * ad_k)
                bm_add_clevis_plate(bm_dst, yz_ka, yz_kb, 0.0015, 0.0015, xk - 0.0036, xk + 0.0036, arc_segs=8)
                bm_add_aerospace_bolt(
                    bm_dst, Vector((xk, cy_k + ny_k * (ad_k + 0.0013), cz_k + nz_k * (ad_k + 0.0013))),
                    n_vec_k, head_r=0.0017, height=0.0016,
                )


    half_dth_mid = 0.25 * ((st0["theta2"] - st0["theta1"]) + (st1["theta2"] - st1["theta1"]))
    b1_ang = t1_0 + 0.5 * math.pi + half_dth_mid
    p0_kn = (p0[0] + n1_y * 0.0076, p0[1] + n1_z * 0.0076)
    p1_kn = (p1_0[0] + 0.0076 * math.cos(b1_ang), p1_0[1] + 0.0076 * math.sin(b1_ang))
    p0_root_cm = (p0[0] + 0.018 * math.cos(t1_0) + n1_y * 0.0042, p0[1] + 0.018 * math.sin(t1_0) + n1_z * 0.0042)
    p1_root_cm = (p1_0[0] - 0.025 * math.cos(t1_0) + n1_y * 0.0048, p1_0[1] - 0.025 * math.sin(t1_0) + n1_z * 0.0048)
    for x_kn in (-0.044, -0.022, 0.0, +0.022, +0.044):
        bm_add_clevis_plate(bm_cm, p0_root_cm, p0_kn, 0.0038, 0.0050, x_kn - 0.0038, x_kn + 0.0038, arc_segs=13)
        bm_add_chamfered_cyl_x(bm_cm, p0_kn[0], p0_kn[1], x_kn - 0.0046, x_kn + 0.0046, radius=0.0052, segs=24, hollow_r=0.0033)
    yz_rv_cm1 = (
        0.82 * p1_root_cm[0] + 0.18 * p1_kn[0] + n1_y * 0.0052,
        0.82 * p1_root_cm[1] + 0.18 * p1_kn[1] + n1_z * 0.0052,
    )
    yz_rv_cm2 = (
        0.52 * p1_root_cm[0] + 0.48 * p1_kn[0] + n1_y * 0.0046,
        0.52 * p1_root_cm[1] + 0.48 * p1_kn[1] + n1_z * 0.0046,
    )
    p1_pad_cm = (p1_0[0] - 0.032 * math.cos(t1_0) + n1_y * 0.0046, p1_0[1] - 0.032 * math.sin(t1_0) + n1_z * 0.0046)
    for x_kn in (-0.035, -0.012, +0.012, +0.035):

        bm_add_clevis_plate(bm_cm, p1_pad_cm, p1_root_cm, 0.0020, 0.0024, x_kn - 0.0054, x_kn + 0.0054, arc_segs=9)
        bm_add_clevis_plate(bm_cm, p1_root_cm, p1_kn, 0.0042, 0.0064, x_kn - 0.0038, x_kn + 0.0038, arc_segs=13)
        bm_add_clevis_plate(bm_cm, p1_pad_cm, p1_kn, 0.0028, 0.0052, x_kn - 0.0014, x_kn + 0.0014, arc_segs=11)
        bm_add_chamfered_cyl_x(bm_cm, p1_kn[0], p1_kn[1], x_kn - 0.0046, x_kn + 0.0046, radius=0.0068, segs=24, hollow_r=0.0042)
        bm_add_dome_rivet(bm_cm, Vector((x_kn, yz_rv_cm1[0], yz_rv_cm1[1])), n1_vec, radius=0.0014)
        bm_add_dome_rivet(bm_cm, Vector((x_kn, yz_rv_cm2[0], yz_rv_cm2[1])), n1_vec, radius=0.0013)

    for x_tw in (-0.0408, -0.0292, -0.0178, -0.0060, +0.0060, +0.0178, +0.0292, +0.0408):
        bm_add_chamfered_cyl_x(bm_cm, p1_kn[0], p1_kn[1], x_tw - 0.0007, x_tw + 0.0007, radius=0.0074, segs=24, chamfer=0.0002, hollow_r=0.0043)
    bm_add_clevis_pin_assembly_x(bm_cm, p1_kn[0], p1_kn[1], -0.052, +0.052, pin_r=0.0042, segs=24)
    _add_hotgas_tile_studs(bm_cm, p0, p1_0, r_p0, r_p1_0, r_p0, r_p1_1, True, 5, (-0.36, +0.36))
    _add_throat_shingle_lip(bm_cm, p0, p1_0, r_p0, r_p1_0, r_p0, r_p1_1, is_master=True, eta_lim=0.38)
    _add_master_edge_seal_keepers(bm_cm, p0, p1_0, r_p0, r_p1_0, r_p0, r_p1_1, (0.28, 0.56))


    t1_vec = Vector((0.0, math.cos(t1_0), math.sin(t1_0)))
    for sx_wp in (-1.0, +1.0):
        bm_add_grease_zerk(bm_cm, Vector((sx_wp * (fh + 0.0082), pc_0[0] + n1_y * 0.0080, pc_0[1] + n1_z * 0.0080)), n1_vec, scale=0.80)
        t_wp_s = subdivide_1d_with_support_loops(5, sup1=0.03, sup2=0.10)
        t_wp_e = subdivide_1d_with_support_loops(5, sup1=0.03, sup2=0.10)
        g_wp_b, g_wp_t = [], []
        for fs_w in t_wp_s:
            sc_w = 0.125 + 0.185 * fs_w
            cy_w = p0[0] + (p1_0[0] - p0[0]) * sc_w
            cz_w = p0[1] + (p1_0[1] - p0[1]) * sc_w
            rc_w = (1.0 - sc_w) * r_p0 + sc_w * r_p1_0
            ro_w = (1.0 - sc_w) * r_p0 + sc_w * r_p1_1
            hw_w = 0.945 * rc_w * sin_hp
            rb_w, rt_w = [], []
            for fe_w in t_wp_e:
                et_w = sx_wp * (0.58 + 0.23 * fe_w)
                xw_p = et_w * hw_w
                ad_w = -(xw_p * xw_p) / (2.0 * 1.08 * ro_w)
                rb_w.append(bm_cm.verts.new((xw_p, cy_w + n1_y * (ad_w + 0.0042), cz_w + n1_z * (ad_w + 0.0042))))
                rt_w.append(bm_cm.verts.new((xw_p, cy_w + n1_y * (ad_w + 0.0061), cz_w + n1_z * (ad_w + 0.0061))))
            g_wp_b.append(rb_w)
            g_wp_t.append(rt_w)
        bm_build_beveled_quad_shell(bm_cm, g_wp_b, g_wp_t, bulge_scale=0.16)
        for sc_wr in (0.155, 0.280):
            cy_wr = p0[0] + (p1_0[0] - p0[0]) * sc_wr
            cz_wr = p0[1] + (p1_0[1] - p0[1]) * sc_wr
            rc_wr = (1.0 - sc_wr) * r_p0 + sc_wr * r_p1_0
            ro_wr = (1.0 - sc_wr) * r_p0 + sc_wr * r_p1_1
            xr_w = sx_wp * 0.695 * (0.945 * rc_wr * sin_hp)
            ad_wr = -(xr_w * xr_w) / (2.0 * 1.08 * ro_wr) + 0.0061
            bm_add_dome_rivet(bm_cm, Vector((xr_w, cy_wr + n1_y * ad_wr, cz_wr + n1_z * ad_wr)), n1_vec, radius=0.0014)

        for sc_hc in (0.40, 0.54):
            cy_h = p0[0] + (p1_0[0] - p0[0]) * sc_hc
            cz_h = p0[1] + (p1_0[1] - p0[1]) * sc_hc
            rc_h = (1.0 - sc_hc) * r_p0 + sc_hc * r_p1_0
            ro_h = (1.0 - sc_hc) * r_p0 + sc_hc * r_p1_1
            hw_h = 0.945 * rc_h * sin_hp
            xh0, xh1 = sx_wp * 0.033, sx_wp * (0.81 * hw_h)
            ad_h0 = -(xh0 * xh0) / (2.0 * 1.08 * ro_h) + 0.0041
            ad_h1 = -(xh1 * xh1) / (2.0 * 1.08 * ro_h) + 0.0041
            bm_add_hat_stiffener_rib(
                bm_cm,
                Vector((xh0, cy_h + n1_y * ad_h0, cz_h + n1_z * ad_h0)),
                Vector((xh1, cy_h + n1_y * ad_h1, cz_h + n1_z * ad_h1)),
                t1_vec, n1_vec, w_foot=0.0058, w_top=0.0022, h_hat=0.0032, n_steps=5, rivet_r=0.0012,
            )
    for sc_mc in (0.24, 0.46):
        cy_m = p0[0] + (p1_0[0] - p0[0]) * sc_mc
        cz_m = p0[1] + (p1_0[1] - p0[1]) * sc_mc
        bm_add_hat_stiffener_rib(
            bm_cm,
            Vector((-0.0125, cy_m + n1_y * 0.0042, cz_m + n1_z * 0.0042)),
            Vector((+0.0125, cy_m + n1_y * 0.0042, cz_m + n1_z * 0.0042)),
            t1_vec, n1_vec, w_foot=0.0056, w_top=0.0022, h_hat=0.0030, n_steps=4, rivet_r=0.0011,
        )


    a_bm, x_bm, h_blk_cm = 0.324, +0.020, 0.0185
    a_blk_a, x_blk_a = 0.362, -0.006
    a_blk_b, x_blk_b = 0.324, +0.006
    dx_aft_bm, da_aft_bm = +0.008, +0.046
    dx_mid_bm, da_mid_bm = -0.015, -0.033
    dx_fwd_bm, da_fwd_bm = +0.054, -0.038
    c1_t, s1_t = math.cos(t1_0), math.sin(t1_0)
    ex_vec = Vector((1.0, 0.0, 0.0))

    def _cm_pt(xv: float, av: float, hv: float) -> Vector:
        return Vector((xv, p0[0] + av * c1_t + n1_y * hv, p0[1] + av * s1_t + n1_z * hv))


    bm_add_clevis_plate(
        bm_cm,
        (p0[0] + (a_bm - 0.015) * c1_t + n1_y * 0.0052, p0[1] + (a_bm - 0.015) * s1_t + n1_z * 0.0052),
        (p0[0] + (a_bm + 0.015) * c1_t + n1_y * 0.0052, p0[1] + (a_bm + 0.015) * s1_t + n1_z * 0.0052),
        0.0022, 0.0022, x_bm - 0.0165, x_bm + 0.0165, arc_segs=9,
    )
    for sx_bp in (-0.0125, +0.0125):
        pb_m0 = _cm_pt(x_bm + sx_bp, a_bm - 0.010, 0.0072)
        pb_m1 = _cm_pt(x_bm + sx_bp, a_bm + 0.010, 0.0072)
        bm_add_aerospace_bolt(bm_cm, pb_m0, n1_vec, head_r=0.0019, height=0.0019)
        bm_add_aerospace_bolt(bm_cm, pb_m1, n1_vec, head_r=0.0019, height=0.0019)
        bm_add_bolt_pair_lockwire(bm_cm, pb_m0, pb_m1, n1_vec, head_r=0.0019, strand_r=0.00028)
    bm_add_segmented_tube(
        bm_cm, _cm_pt(x_bm, a_bm, 0.0046), _cm_pt(x_bm, a_bm, h_blk_cm - 0.0024),
        [(0.0, 0.0138), (0.25, 0.0134), (0.72, 0.0110), (1.0, 0.0106)],
        segs=28, cap_ends=True,
    )
    bm_add_segmented_tube(
        bm_cm, _cm_pt(x_bm, a_bm, h_blk_cm - 0.0028), _cm_pt(x_bm, a_bm, h_blk_cm + 0.0086),
        [(0.0, 0.0058), (0.58, 0.0058), (0.62, 0.0094), (0.72, 0.0094), (0.75, 0.0032), (0.95, 0.0032), (1.0, 0.0024)],
        segs=24, cap_ends=True,
    )
    bm_add_bevel_locked_hex(
        bm_cm, _cm_pt(x_bm, a_bm, h_blk_cm + 0.0040), _cm_pt(x_bm, a_bm, h_blk_cm + 0.0068),
        [(0.0, 0.0054), (0.20, 0.0060), (0.80, 0.0060), (1.0, 0.0052)],
        corner_bevel=0.14, cap_ends=True,
    )

    for k_cr in range(6):
        ang_cr = k_cr * math.pi / 3.0
        d_cr = ex_vec * math.cos(ang_cr) + t1_vec * math.sin(ang_cr)
        p_cr_c = _cm_pt(x_bm, a_bm, h_blk_cm + 0.0072) + d_cr * 0.0044
        bm_add_segmented_tube(
            bm_cm, p_cr_c - n1_vec * 0.0005, p_cr_c + n1_vec * 0.0009,
            [(0.0, 0.0012), (0.5, 0.0012), (1.0, 0.0009)], segs=8, cap_ends=True,
        )
    p_cot_c = _cm_pt(x_bm, a_bm, h_blk_cm + 0.0075)
    for sgn_ct in (-1.0, +1.0):
        bm_add_spline_tube(bm_cm, [
            p_cot_c - t1_vec * 0.0052 + ex_vec * (sgn_ct * 0.00045),
            p_cot_c + ex_vec * (sgn_ct * 0.00045),
            p_cot_c + t1_vec * 0.0038 + ex_vec * (sgn_ct * 0.00065),
            p_cot_c + t1_vec * 0.0058 + ex_vec * (sgn_ct * 0.0028) - n1_vec * 0.0012,
        ], radius=0.00042, n_samples=16, segs=8, cap_ends=True)


    t_db_s = subdivide_1d_with_support_loops(6, sup1=0.035, sup2=0.11)
    t_db_u = subdivide_1d_with_support_loops(6, sup1=0.035, sup2=0.11)
    gb_db, gt_db = [], []
    for fs_db in t_db_s:
        sc_db = 0.718 + 0.146 * fs_db
        cy_db = p0[0] + (p1_0[0] - p0[0]) * sc_db
        cz_db = p0[1] + (p1_0[1] - p0[1]) * sc_db
        rc_db = (1.0 - sc_db) * r_p0 + sc_db * r_p1_0
        ro_db = (1.0 - sc_db) * r_p0 + sc_db * r_p1_1
        hw_db = 0.945 * rc_db * sin_hp
        x_db0, x_db1 = -0.80 * hw_db, -0.0325
        rb_db, rt_db = [], []
        for fu_db in t_db_u:
            xdb = x_db0 + (x_db1 - x_db0) * fu_db
            ad_db = -(xdb * xdb) / (2.0 * 1.08 * ro_db)
            bead_db = 0.0007 * math.sin(math.pi * min(1.0, max(0.0, (fs_db - 0.14) / 0.72))) * math.sin(math.pi * min(1.0, max(0.0, (fu_db - 0.18) / 0.64)))
            nb_db = ad_db + 0.0044 + 0.0023
            nt_db = nb_db + 0.0012 + bead_db
            rb_db.append(bm_cm.verts.new((xdb, cy_db + n1_y * nb_db, cz_db + n1_z * nb_db)))
            rt_db.append(bm_cm.verts.new((xdb, cy_db + n1_y * nt_db, cz_db + n1_z * nt_db)))
        gb_db.append(rb_db)
        gt_db.append(rt_db)
    bm_build_beveled_quad_shell(bm_cm, gb_db, gt_db, bulge_scale=0.16)
    for sc_dbr in (0.732, 0.791, 0.850):
        cy_dr = p0[0] + (p1_0[0] - p0[0]) * sc_dbr
        cz_dr = p0[1] + (p1_0[1] - p0[1]) * sc_dbr
        rc_dr = (1.0 - sc_dbr) * r_p0 + sc_dbr * r_p1_0
        ro_dr = (1.0 - sc_dbr) * r_p0 + sc_dbr * r_p1_1
        hw_dr = 0.945 * rc_dr * sin_hp
        for x_dr in (-0.75 * hw_dr, -0.0355):
            ad_dr = -(x_dr * x_dr) / (2.0 * 1.08 * ro_dr) + 0.0044 + 0.0035
            bm_add_dome_rivet(bm_cm, Vector((x_dr, cy_dr + n1_y * ad_dr, cz_dr + n1_z * ad_dr)), n1_vec, radius=0.00120)

    meshes["conv_master"] = finalize_shared_mesh("Mesh_ConvMaster", bm_cm, clay)

    def _add_seal_holddown_clips(bm_dst: bmesh.types.BMesh, p_st: tuple, p_en: tuple,
                                 rc0_s: float, rc1_s: float,
                                 ny_s: float, nz_s: float, stations_s: tuple):

        dy_s, dz_s = p_en[0] - p_st[0], p_en[1] - p_st[1]
        len_s = math.hypot(dy_s, dz_s)
        ty_s, tz_s = dy_s / len_s, dz_s / len_s
        u_vec = Vector((1.0, 0.0, 0.0))
        v_vec = Vector((0.0, ny_s, nz_s))
        n_vec_s = Vector((0.0, ny_s, nz_s))

        prof_clip = [
            (-0.0125, 0.0002), (-0.0125, 0.0014), (-0.0062, 0.0016), (-0.0048, 0.0029),
            (+0.0048, 0.0029), (+0.0062, 0.0016), (+0.0125, 0.0014), (+0.0125, 0.0002),
            (+0.0056, 0.0004), (+0.0042, 0.0019), (-0.0042, 0.0019), (-0.0056, 0.0004),
        ]
        clip_caps = [(0, 1, 2, 11), (11, 2, 3, 10), (10, 3, 4, 9), (9, 4, 5, 8), (8, 5, 6, 7)]
        t_bar_s = subdivide_1d_with_support_loops(4, sup1=0.04, sup2=0.12)
        t_bar_e = subdivide_1d_with_support_loops(10, sup1=0.025, sup2=0.08)
        for sc in stations_s:
            cy_c = p_st[0] + dy_s * sc
            cz_c = p_st[1] + dz_s * sc
            rc_c = (1.0 - sc) * rc0_s + sc * rc1_s
            rho_s = 0.93 * rc_c
            hw_s = 0.948 * rc_c * sin_hp
            pa_c = Vector((0.0, cy_c - ty_s * 0.0085, cz_c - tz_s * 0.0085))
            pb_c = Vector((0.0, cy_c + ty_s * 0.0085, cz_c + tz_s * 0.0085))
            bm_add_swept_quad_profile(bm_dst, pa_c, pb_c, u_vec, v_vec, prof_clip, n_steps=4, cap_quad_pairs=clip_caps)
            for sx_rv in (-0.0092, +0.0092):
                bm_add_dome_rivet(bm_dst, Vector((sx_rv, cy_c + ny_s * 0.0014, cz_c + nz_s * 0.0014)), n_vec_s, radius=0.0015)


            g_rb_b, g_rb_t = [], []
            for fs_b in t_bar_s:
                ax_off = (-0.0050 + 0.0100 * fs_b)
                cy_b = cy_c + ty_s * ax_off
                cz_b = cz_c + tz_s * ax_off
                rb_row, rt_row = [], []
                for fe_b in t_bar_e:
                    et_b = -0.78 + 1.56 * fe_b
                    xb = et_b * hw_s
                    ad_b = -(xb * xb) / (2.0 * rho_s)
                    rb_row.append(bm_dst.verts.new((xb, cy_b + ny_s * (ad_b + 0.0005), cz_b + nz_s * (ad_b + 0.0005))))
                    rt_row.append(bm_dst.verts.new((xb, cy_b + ny_s * (ad_b + 0.0021), cz_b + nz_s * (ad_b + 0.0021))))
                g_rb_b.append(rb_row)
                g_rb_t.append(rt_row)
            bm_build_beveled_quad_shell(bm_dst, g_rb_b, g_rb_t, bulge_scale=0.16)
            for et_puck in (-0.70, -0.42, +0.42, +0.70):
                xp = et_puck * hw_s
                ad_p = -(xp * xp) / (2.0 * rho_s)
                p_pk = Vector((xp, cy_c + ny_s * ad_p, cz_c + nz_s * ad_p))
                bm_add_segmented_tube(
                    bm_dst, p_pk + n_vec_s * 0.0018, p_pk + n_vec_s * 0.0028,
                    [(0.0, 0.0032), (0.35, 0.0032), (0.85, 0.0026), (1.0, 0.0018)],
                    segs=14, cap_ends=True,
                )


    dr_s = spec.seal_radial_offset
    p0_seal = (p0[0] - n1_y * dr_s, p0[1] - n1_z * dr_s)
    p1_seal_0 = (p1_0[0] - n1_y * dr_s, p1_0[1] - n1_z * dr_s)
    bm_cs = bmesh.new()
    add_highres_flap_shell(
        bm_cs, p0_seal, p1_seal_0, r_p0 - dr_s, r_p1_0 - dr_s, r_p0 - dr_s, r_p1_1 - dr_s,
        spec.n_sectors, width_factor=0.948, thickness=0.0036,
        is_outer_master=False, flap_role='CONV_SEAL', n_long=92, n_lat=68,
    )
    _add_seal_holddown_clips(bm_cs, p0_seal, p1_seal_0, r_p0 - dr_s, r_p1_0 - dr_s, n1_y, n1_z, (0.20, 0.39, 0.58))


    def _cs_pt(xv: float, av: float, h_above_master: float) -> Vector:
        return Vector((
            xv,
            p0_seal[0] + av * c1_t + n1_y * (h_above_master + dr_s),
            p0_seal[1] + av * s1_t + n1_z * (h_above_master + dr_s),
        ))

    t_cs_ped = subdivide_1d_with_support_loops(6, sup1=0.04, sup2=0.12)

    gb_csf, gt_csf = [], []
    for fx_f in t_cs_ped:
        xv_f = -0.0054 + 0.0108 * fx_f
        rb_f, rt_f = [], []
        for fa_f in t_cs_ped:
            av_f = 0.300 + 0.082 * fa_f
            rb_f.append(bm_cs.verts.new(_cs_pt(xv_f, av_f, -0.0056)))
            rt_f.append(bm_cs.verts.new(_cs_pt(xv_f, av_f, -0.0042)))
        gb_csf.append(rb_f)
        gt_csf.append(rt_f)
    bm_build_beveled_quad_shell(bm_cs, gb_csf, gt_csf, bulge_scale=0.15)
    for av_rv in (0.308, 0.341, 0.374):
        for sx_rv in (-0.0038, +0.0038):
            bm_add_dome_rivet(bm_cs, _cs_pt(sx_rv, av_rv, -0.0042), n1_vec, radius=0.00095)


    fa_web = subdivide_1d_with_support_loops(8, sup1=0.03, sup2=0.10)
    gb_csw, gt_csw = [], []
    for fx_p in t_cs_ped:
        xv_p = -0.0012 + 0.0024 * fx_p
        rb_p, rt_p = [], []
        for fa_p in fa_web:
            av_p = 0.302 + 0.078 * fa_p
            end_taper = min(1.0, min((av_p - 0.302) / 0.010, (0.380 - av_p) / 0.010))
            h_top_w = -0.0040 + 0.0174 * max(0.0, end_taper)
            rb_p.append(bm_cs.verts.new(_cs_pt(xv_p, av_p, -0.0046)))
            rt_p.append(bm_cs.verts.new(_cs_pt(xv_p, av_p, h_top_w)))
        gb_csw.append(rb_p)
        gt_csw.append(rt_p)
    bm_build_beveled_quad_shell(bm_cs, gb_csw, gt_csw, bulge_scale=0.02)
    for av_lh in (0.336, 0.350):
        p_lh_cs = _cs_pt(0.0, av_lh, 0.0056)
        bm_add_chamfered_cyl_x(bm_cs, p_lh_cs.y, p_lh_cs.z, -0.0017, +0.0017, radius=0.0025, segs=16, chamfer=0.00028, hollow_r=0.0014)


    gb_css, gt_css = [], []
    for fx_s in t_cs_ped:
        xv_s = -0.0022 + 0.0044 * fx_s
        rb_s, rt_s = [], []
        for fa_s in fa_web:
            av_s = 0.311 + 0.064 * fa_s
            x_scale = 1.0 - 0.22 * max(0.0, max((0.318 - av_s) / 0.007, (av_s - 0.368) / 0.007))
            rb_s.append(bm_cs.verts.new(_cs_pt(xv_s * x_scale, av_s, +0.0124)))
            rt_s.append(bm_cs.verts.new(_cs_pt(xv_s * x_scale, av_s, +0.0142)))
        gb_css.append(rb_s)
        gt_css.append(rt_s)
    bm_build_beveled_quad_shell(bm_cs, gb_css, gt_css, bulge_scale=0.04)
    pb_sp0 = _cs_pt(-0.0004, 0.337, +0.0142)
    pb_sp1 = _cs_pt(+0.0004, 0.349, +0.0142)
    bm_add_aerospace_bolt(bm_cs, pb_sp0, n1_vec, head_r=0.0021, height=0.0020)
    bm_add_aerospace_bolt(bm_cs, pb_sp1, n1_vec, head_r=0.0021, height=0.0020)
    bm_add_bolt_pair_lockwire(bm_cs, pb_sp0, pb_sp1, n1_vec, head_r=0.0021, strand_r=0.00032)

    rot_s0  = Matrix.Rotation(+math.pi / spec.n_sectors, 4, 'Y')
    rot_s15 = Matrix.Rotation(-math.pi / spec.n_sectors, 4, 'Y')
    p_aft_bm_ref = _cm_pt(x_bm + dx_aft_bm, a_bm + da_aft_bm, h_blk_cm)
    p_mid_bm_ref = _cm_pt(x_bm + dx_mid_bm, a_bm + da_mid_bm, h_blk_cm)
    p_sqA_seal_loc = _cs_pt(x_blk_a, a_blk_a, h_blk_cm)
    p_sqB_seal_loc = _cs_pt(x_blk_b, a_blk_b, h_blk_cm)
    open_dir_a = (rot_s15 @ p_aft_bm_ref) - p_sqA_seal_loc
    open_dir_b = (rot_s0  @ p_mid_bm_ref) - p_sqB_seal_loc

    bm_add_machined_clevis_trunnion(
        bm_cs, p_sqA_seal_loc, open_dir_a, n1_vec,
        slot_half_h=0.0023, base_drop=0.0062,
    )
    bm_add_machined_clevis_trunnion(
        bm_cs, p_sqB_seal_loc, open_dir_b, n1_vec,
        slot_half_h=0.0023, base_drop=0.0062,
    )

    p0_skn = (p0_seal[0] + n1_y * 0.0006, p0_seal[1] + n1_z * 0.0006)
    p1_skn = (p1_seal_0[0] + 0.0012 * math.cos(b1_ang), p1_seal_0[1] + 0.0012 * math.sin(b1_ang))
    p0_sroot = (p0_seal[0] + 0.018 * math.cos(t1_0) + n1_y * 0.0005, p0_seal[1] + 0.018 * math.sin(t1_0) + n1_z * 0.0005)
    p1_sroot = (p1_seal_0[0] - 0.018 * math.cos(t1_0) + n1_y * 0.0008, p1_seal_0[1] - 0.018 * math.sin(t1_0) + n1_z * 0.0008)
    for x_sk in (-0.011, +0.011):
        bm_add_clevis_plate(bm_cs, p0_sroot, p0_skn, 0.0016, 0.0024, x_sk - 0.0034, x_sk + 0.0034, arc_segs=11)
        bm_add_chamfered_cyl_x(bm_cs, p0_skn[0], p0_skn[1], x_sk - 0.0040, x_sk + 0.0040, radius=0.0026, segs=18, hollow_r=0.0015)
    bm_add_clevis_pin_assembly_x(bm_cs, p0_skn[0], p0_skn[1], -0.0175, +0.0175, pin_r=0.0016, segs=16, compact_ends=True)
    p1_spad = (p1_seal_0[0] - 0.025 * math.cos(t1_0) + n1_y * 0.0007, p1_seal_0[1] - 0.025 * math.sin(t1_0) + n1_z * 0.0007)
    yz_srv_c = (0.68 * p1_sroot[0] + 0.32 * p1_skn[0] + n1_y * 0.0024, 0.68 * p1_sroot[1] + 0.32 * p1_skn[1] + n1_z * 0.0024)

    for x_sk in (-0.021, 0.0, +0.021):
        bm_add_clevis_plate(bm_cs, p1_spad, p1_sroot, 0.0014, 0.0018, x_sk - 0.0046, x_sk + 0.0046, arc_segs=9)
        bm_add_clevis_plate(bm_cs, p1_sroot, p1_skn, 0.0020, 0.0030, x_sk - 0.0032, x_sk + 0.0032, arc_segs=11)
        bm_add_chamfered_cyl_x(bm_cs, p1_skn[0], p1_skn[1], x_sk - 0.0038, x_sk + 0.0038, radius=0.0034, segs=18, hollow_r=0.0019)
        bm_add_dome_rivet(bm_cs, Vector((x_sk, yz_srv_c[0], yz_srv_c[1])), n1_vec, radius=0.0011)
    for x_stw in (-0.0158, -0.0052, +0.0052, +0.0158):
        bm_add_chamfered_cyl_x(bm_cs, p1_skn[0], p1_skn[1], x_stw - 0.0006, x_stw + 0.0006, radius=0.0038, segs=18, chamfer=0.00015, hollow_r=0.0020)
    bm_add_clevis_pin_assembly_x(bm_cs, p1_skn[0], p1_skn[1], -0.0272, +0.0272, pin_r=0.0021, segs=18, compact_ends=True)
    _add_hotgas_tile_studs(bm_cs, p0_seal, p1_seal_0, r_p0 - dr_s, r_p1_0 - dr_s, r_p0 - dr_s, r_p1_1 - dr_s, False, 5, (-0.32, +0.32))
    _add_throat_shingle_lip(bm_cs, p0_seal, p1_seal_0, r_p0 - dr_s, r_p1_0 - dr_s, r_p0 - dr_s, r_p1_1 - dr_s, is_master=False, eta_lim=0.80)
    meshes["conv_seal"] = finalize_shared_mesh("Mesh_ConvSeal", bm_cs, clay)


    bm_dm = bmesh.new()
    add_highres_flap_shell(
        bm_dm, p1_0, p2_0, r_p1_0, r_p2_0, r_p1_1, r_p2_1,
        spec.n_sectors, width_factor=0.945, thickness=0.0044,
        is_outer_master=True, flap_role='DIV_MASTER', n_long=152, n_lat=84,
    )
    t2_0 = st0["theta2"]
    c2_t, s2_t = math.cos(t2_0), math.sin(t2_0)
    n2_y, n2_z = -s2_t, c2_t
    n2_vec = Vector((0.0, n2_y, n2_z))

    def _dm_yz(a_dist: float, h_norm: float) -> tuple:
        return (
            p1_0[0] + a_dist * c2_t + n2_y * h_norm,
            p1_0[1] + a_dist * s2_t + n2_z * h_norm,
        )


    _add_comb_finger_seals(bm_dm, p1_0, p2_0, r_p1_0, r_p2_0, r_p1_1, r_p2_1, n_teeth=22, s0=0.04, s1=0.89)
    for rv_i in range(16):
        sf = 0.05 + 0.82 * (rv_i / 15.0)
        cy = p1_0[0] + (p2_0[0] - p1_0[0]) * sf
        cz = p1_0[1] + (p2_0[1] - p1_0[1]) * sf
        rc_s = (1.0 - sf) * r_p1_0 + sf * r_p2_0
        ro_s = (1.0 - sf) * r_p1_1 + sf * r_p2_1
        xw = 0.945 * rc_s * sin_hp * 0.912
        drop = -(xw * xw) / (2.0 * 1.08 * ro_s) + 0.0073
        for sx in (-xw, +xw):
            bm_add_dome_rivet(bm_dm, Vector((sx, cy + n2_y * drop, cz + n2_z * drop)), n2_vec, radius=0.0017)


    tab_eta_ranges = [
        (-0.92, -0.69), (-0.62, -0.40), (-0.33, -0.14),
        (+0.14, +0.33), (+0.40, +0.62), (+0.69, +0.92),
    ]
    t_tab_s = subdivide_1d_with_support_loops(6, sup1=0.025, sup2=0.08)
    t_tab_e = subdivide_1d_with_support_loops(4, sup1=0.035, sup2=0.11)
    for (et0, et1) in tab_eta_ranges:
        g_bot_t, g_top_t = [], []
        for fs_t in t_tab_s:
            sc = 0.895 + 0.113 * fs_t
            sc_clamp = min(1.0, sc)
            cy_t = p1_0[0] + (p2_0[0] - p1_0[0]) * sc
            cz_t = p1_0[1] + (p2_0[1] - p1_0[1]) * sc
            rc_t = (1.0 - sc_clamp) * r_p1_0 + sc_clamp * r_p2_0
            ro_t = (1.0 - sc_clamp) * r_p1_1 + sc_clamp * r_p2_1
            hw_t = 0.945 * rc_t * sin_hp * (1.0 - 0.036 * max(0.0, (sc_clamp - 0.92) / 0.08))
            lip_th_t = 0.0044 * (1.0 - 0.52 * (((max(0.88, sc_clamp) - 0.88) / 0.12) ** 1.35))
            curl_h = 0.0016 * (fs_t ** 1.8)
            rb_row, rt_row = [], []
            for fe_t in t_tab_e:
                et_cur = et0 + (et1 - et0) * fe_t
                xt = et_cur * hw_t
                ad_t = -(xt * xt) / (2.0 * 1.08 * ro_t)
                mid_ridge = 0.0007 * (1.0 - abs(-1.0 + 2.0 * fe_t)) * math.sin(math.pi * fs_t)
                nb_t = ad_t + lip_th_t + 0.0002 + curl_h * 0.65
                nt_t = nb_t + 0.00145 + mid_ridge
                rb_row.append(bm_dm.verts.new((xt, cy_t + n2_y * nb_t, cz_t + n2_z * nb_t)))
                rt_row.append(bm_dm.verts.new((xt, cy_t + n2_y * nt_t, cz_t + n2_z * nt_t)))
            g_bot_t.append(rb_row)
            g_top_t.append(rt_row)
        bm_build_beveled_quad_shell(bm_dm, g_bot_t, g_top_t, bulge_scale=0.18)
        for et_rv in (et0 + 0.28 * (et1 - et0), et0 + 0.72 * (et1 - et0)):
            sc_r = 0.912
            cy_r = p1_0[0] + (p2_0[0] - p1_0[0]) * sc_r
            cz_r = p1_0[1] + (p2_0[1] - p1_0[1]) * sc_r
            rc_r = (1.0 - sc_r) * r_p1_0 + sc_r * r_p2_0
            ro_r = (1.0 - sc_r) * r_p1_1 + sc_r * r_p2_1
            hw_r = 0.945 * rc_r * sin_hp
            xr = et_rv * hw_r
            nh_r = -(xr * xr) / (2.0 * 1.08 * ro_r) + 0.0056
            bm_add_dome_rivet(bm_dm, Vector((xr, cy_r + n2_y * nh_r, cz_r + n2_z * nh_r)), n2_vec, radius=0.0014)


    a_trk0, a_trk1 = 0.052, 0.476
    t2_vec = Vector((0.0, c2_t, s2_t))
    for side_s in (-1.0, 1.0):

        bm_add_clevis_plate(
            bm_dm, _dm_yz(a_trk0, 0.0150), _dm_yz(a_trk1, 0.0150),
            0.0020, 0.0020, side_s * 0.0126 - 0.0048, side_s * 0.0126 + 0.0048, arc_segs=8,
        )
        bm_add_clevis_plate(
            bm_dm, _dm_yz(a_trk0, 0.0198), _dm_yz(a_trk1, 0.0198),
            0.0042, 0.0042, side_s * 0.0102 - 0.0014, side_s * 0.0102 + 0.0014, arc_segs=10,
        )
        bm_add_clevis_plate(
            bm_dm, _dm_yz(a_trk0, 0.0242), _dm_yz(a_trk1, 0.0242),
            0.0013, 0.0013, side_s * 0.0096 - 0.0028, side_s * 0.0096 + 0.0028, arc_segs=8,
        )

        for a_lh in (0.076, 0.108, 0.140, 0.344, 0.382, 0.418, 0.452):
            yz_lh = _dm_yz(a_lh, 0.0198)
            bm_add_chamfered_cyl_x(
                bm_dm, yz_lh[0], yz_lh[1],
                side_s * 0.0102 - 0.0019, side_s * 0.0102 + 0.0019,
                radius=0.0036, segs=18, chamfer=0.0004, hollow_r=0.0021,
            )

        for a_pair0, a_pair1 in [(0.068, 0.100), (0.126, 0.152), (0.338, 0.368), (0.398, 0.428), (0.444, 0.468)]:
            yz_b0 = _dm_yz(a_pair0, 0.0168)
            yz_b1 = _dm_yz(a_pair1, 0.0168)
            pb_t0 = Vector((side_s * 0.0145, yz_b0[0], yz_b0[1]))
            pb_t1 = Vector((side_s * 0.0145, yz_b1[0], yz_b1[1]))
            bm_add_aerospace_bolt(bm_dm, pb_t0, n2_vec, head_r=0.0022, height=0.0022)
            bm_add_aerospace_bolt(bm_dm, pb_t1, n2_vec, head_r=0.0022, height=0.0022)
            bm_add_bolt_pair_lockwire(bm_dm, pb_t0, pb_t1, n2_vec, head_r=0.0022, strand_r=0.00032)


    for a_stop, dir_buf in ((0.045, +1.0), (0.478, -1.0)):
        yz_st = _dm_yz(a_stop, 0.0205)
        bm_add_clevis_plate(bm_dm, yz_st, _dm_yz(a_stop + 0.006, 0.0205), 0.0052, 0.0052, -0.0156, +0.0156, arc_segs=8)
        yz_sb = _dm_yz(a_stop + 0.003, 0.0256)
        psb_l = Vector((-0.0085, yz_sb[0], yz_sb[1]))
        psb_r = Vector((+0.0085, yz_sb[0], yz_sb[1]))
        bm_add_aerospace_bolt(bm_dm, psb_l, n2_vec, head_r=0.0023, height=0.0023)
        bm_add_aerospace_bolt(bm_dm, psb_r, n2_vec, head_r=0.0023, height=0.0023)
        bm_add_bolt_pair_lockwire(bm_dm, psb_l, psb_r, n2_vec, head_r=0.0023, strand_r=0.00032)
        a_face = (a_stop + 0.006) if dir_buf > 0 else a_stop
        yz_bf0 = _dm_yz(a_face, 0.0222)
        yz_bf1 = _dm_yz(a_face + dir_buf * 0.0022, 0.0222)
        for sx_bf in (-0.0108, +0.0108):
            bm_add_segmented_tube(
                bm_dm, Vector((sx_bf, yz_bf0[0], yz_bf0[1])), Vector((sx_bf, yz_bf1[0], yz_bf1[1])),
                [(0.0, 0.0022), (0.45, 0.0022), (0.85, 0.0018), (1.0, 0.0013)],
                segs=16, cap_ends=True,
            )


    for a_tie in (0.095, 0.146, 0.356, 0.412, 0.460):
        bm_add_clevis_plate(bm_dm, _dm_yz(a_tie - 0.004, 0.0162), _dm_yz(a_tie + 0.004, 0.0162), 0.0018, 0.0018, -0.0084, +0.0084, arc_segs=9)
    bm_add_clevis_plate(bm_dm, _dm_yz(0.056, 0.0180), _dm_yz(0.474, 0.0180), 0.0014, 0.0014, -0.0044, +0.0044, arc_segs=9)
    for a_wb in (0.072, 0.125, 0.342, 0.394, 0.442):
        yz_wb = _dm_yz(a_wb, 0.0192)
        p_wb = Vector((0.0, yz_wb[0], yz_wb[1]))
        bm_add_segmented_tube(
            bm_dm, p_wb - n2_vec * 0.0003, p_wb + n2_vec * 0.0010,
            [(0.0, 0.0022), (0.35, 0.0022), (0.85, 0.0018), (1.0, 0.0011)], segs=14, cap_ends=True,
        )
        bm_add_bevel_locked_hex(
            bm_dm, p_wb + n2_vec * 0.0006, p_wb + n2_vec * 0.0013,
            [(0.0, 0.0011), (0.25, 0.0013), (0.85, 0.0013), (1.0, 0.0009)], corner_bevel=0.14, cap_ends=True,
        )

    for side_g in (-1.0, +1.0):
        bm_add_clevis_plate(
            bm_dm, _dm_yz(0.016, 0.0068), _dm_yz(0.046, 0.0185),
            0.0028, 0.0048, side_g * 0.0112 - 0.0018, side_g * 0.0112 + 0.0018, arc_segs=11,
        )
        bm_add_clevis_plate(
            bm_dm, _dm_yz(0.484, 0.0162), _dm_yz(0.506, 0.0064),
            0.0036, 0.0018, side_g * 0.0042 - 0.0015, side_g * 0.0042 + 0.0015, arc_segs=11,
        )
        yz_grv = _dm_yz(0.503, 0.0076)
        bm_add_dome_rivet(bm_dm, Vector((side_g * 0.0042, yz_grv[0], yz_grv[1])), n2_vec, radius=0.0014)
    _add_master_edge_seal_keepers(bm_dm, p1_0, p2_0, r_p1_0, r_p2_0, r_p1_1, r_p2_1, (0.13, 0.28, 0.68, 0.852))


    a_ring_div = spec.div_lug_a
    h_ring_m = 0.0268
    bm_sc = bmesh.new()
    a_car0, a_car1 = spec.div_lug_a - 0.036, spec.div_lug_a + 0.036
    base_pd_f = _dm_yz(spec.div_lug_a - 0.028, 0.0252)
    base_pd_b = _dm_yz(spec.div_lug_a + 0.028, 0.0252)
    t_s5_sc = subdivide_1d_with_support_loops(5, sup1=0.04, sup2=0.14)
    for side_s in (-1.0, 1.0):

        bm_add_clevis_plate(
            bm_sc, _dm_yz(a_car0, 0.0262), _dm_yz(a_car1, 0.0262),
            0.0018, 0.0018, side_s * 0.0112 - 0.0042, side_s * 0.0112 + 0.0042, arc_segs=9,
        )
        bm_add_clevis_plate(
            bm_sc, _dm_yz(spec.div_lug_a - 0.018, 0.0226), _dm_yz(spec.div_lug_a + 0.018, 0.0226),
            0.0032, 0.0032, side_s * 0.0150 - 0.0022, side_s * 0.0150 + 0.0022, arc_segs=11,
        )
        bm_add_clevis_plate(
            bm_sc, _dm_yz(a_car0 + 0.002, 0.0220), _dm_yz(a_car1 - 0.002, 0.0220),
            0.0032, 0.0032, side_s * 0.0188 - 0.0013, side_s * 0.0188 + 0.0013, arc_segs=11,
        )

        fa_wing = sorted(set(
            subdivide_1d_with_support_loops(12, sup1=0.02, sup2=0.07)
            + [0.11, 0.13, 0.31, 0.33, 0.67, 0.69, 0.87, 0.89]
        ))
        fx_wing = [0.00, 0.05, 0.18, 0.23, 0.68, 0.73, 0.94, 1.00]
        gb_bw, gt_bw = [], []
        for fa_w in fa_wing:
            da_w = -0.035 + 0.070 * fa_w
            yz_w = _dm_yz(spec.div_lug_a + da_w, 0.0)
            u_abs = abs(da_w) / 0.035
            waist = 1.0 - 0.16 * math.exp(-((abs(da_w) - 0.014) / 0.0065) ** 2) - 0.24 * max(0.0, (u_abs - 0.76) / 0.24) ** 2
            x_out_w = 0.0145 + 0.0195 * waist
            in_wpkt_a = 0.0
            for wa0, wa1 in ((0.12, 0.32), (0.68, 0.88)):
                if wa0 < fa_w < wa1:
                    in_wpkt_a = max(in_wpkt_a, min(1.0, min(fa_w - wa0, wa1 - fa_w) / 0.015))
            rb_w, rt_w = [], []
            for fx_w in fx_wing:
                xw = side_s * (0.0140 + (x_out_w - 0.0140) * fx_w)
                drop_bw = -0.0016 * (fx_w ** 1.4)
                in_wpkt_x = min(1.0, max(0.0, min((fx_w - 0.18) / 0.045, (0.73 - fx_w) / 0.045)))
                wpkt = 0.00085 * in_wpkt_a * in_wpkt_x
                rb_w.append(bm_sc.verts.new((xw, yz_w[0] + n2_y * (0.0246 + drop_bw), yz_w[1] + n2_z * (0.0246 + drop_bw))))
                rt_w.append(bm_sc.verts.new((xw, yz_w[0] + n2_y * (0.0264 + drop_bw - wpkt), yz_w[1] + n2_z * (0.0264 + drop_bw - wpkt))))
            gb_bw.append(rb_w)
            gt_bw.append(rt_w)
        bm_build_beveled_quad_shell(bm_sc, gb_bw, gt_bw, bulge_scale=0.18)
        yz_wbf = _dm_yz(spec.div_lug_a - 0.024, 0.0254)
        yz_wbb = _dm_yz(spec.div_lug_a + 0.024, 0.0254)
        pw_f = Vector((side_s * 0.0282, yz_wbf[0], yz_wbf[1]))
        pw_b = Vector((side_s * 0.0282, yz_wbb[0], yz_wbb[1]))
        bm_add_aerospace_bolt(bm_sc, pw_f, n2_vec, head_r=0.0022, height=0.0021)
        bm_add_aerospace_bolt(bm_sc, pw_b, n2_vec, head_r=0.0022, height=0.0021)
        bm_add_bolt_pair_lockwire(bm_sc, pw_f, pw_b, n2_vec, head_r=0.0022, strand_r=0.00030)


        gb_pr, gt_pr = [], []
        for fx_p in subdivide_1d_with_support_loops(5, sup1=0.04, sup2=0.14):
            xp = side_s * (0.0114 + 0.0078 * fx_p)
            ch_taper = 1.0 - 0.16 * fx_p
            rb_p, rt_p = [], []
            for fa_p in [-1.00, -0.92, -0.74, -0.52, -0.42, 0.00, +0.42, +0.52, +0.74, +0.92, +1.00]:
                ap = fa_p * 0.0128 * ch_taper
                rib_p = 0.0020 * min(1.0, max(0.0, (abs(fa_p) - 0.44) / 0.08))
                pkt_p = 0.0008 * max(0.0, 1.0 - abs(fa_p) / 0.36) * min(1.0, max(0.0, min((fx_p - 0.18) / 0.12, (0.82 - fx_p) / 0.12)))
                yz_p = _dm_yz(a_ring_div + ap, 0.0)
                drop_p = -0.0008 * fx_p
                rb_p.append(bm_sc.verts.new((xp, yz_p[0] + n2_y * (h_ring_m + 0.0028 + drop_p), yz_p[1] + n2_z * (h_ring_m + 0.0028 + drop_p))))
                rt_p.append(bm_sc.verts.new((xp, yz_p[0] + n2_y * (h_ring_m + 0.0054 + rib_p - pkt_p + drop_p), yz_p[1] + n2_z * (h_ring_m + 0.0054 + rib_p - pkt_p + drop_p))))
            gb_pr.append(rb_p)
            gt_pr.append(rt_p)
        bm_build_beveled_quad_shell(bm_sc, gb_pr, gt_pr, bulge_scale=0.14)


        for sa_pw in (-1.0, +1.0):
            gb_pw, gt_pw = [], []
            for fx_p in t_s5_sc:
                xp = side_s * (0.0118 + 0.0072 * fx_p)
                rb_w2, rt_w2 = [], []
                for fa_p in subdivide_1d_with_support_loops(4, sup1=0.05, sup2=0.16):
                    ap = sa_pw * (0.0102 + 0.0024 * fa_p)
                    yz_p = _dm_yz(a_ring_div + ap, 0.0)
                    drop_p = -0.0008 * fx_p
                    rb_w2.append(bm_sc.verts.new((xp, yz_p[0] + n2_y * (h_ring_m - 0.0012 + drop_p), yz_p[1] + n2_z * (h_ring_m - 0.0012 + drop_p))))
                    rt_w2.append(bm_sc.verts.new((xp, yz_p[0] + n2_y * (h_ring_m + 0.0028 + drop_p), yz_p[1] + n2_z * (h_ring_m + 0.0028 + drop_p))))
                gb_pw.append(rb_w2)
                gt_pw.append(rt_w2)
            bm_build_beveled_quad_shell(bm_sc, gb_pw, gt_pw, bulge_scale=0.15)

        yz_pbolt = _dm_yz(a_ring_div, h_ring_m + 0.0050)
        bm_add_aerospace_bolt(bm_sc, Vector((side_s * 0.0155, yz_pbolt[0], yz_pbolt[1])), n2_vec, head_r=0.0019, height=0.0019)


        gb_bt, gt_bt = [], []
        for fx_b in t_s5_sc:
            xb_t = side_s * (0.0112 + 0.0068 * fx_b)
            z_top_bt = 0.0355 - 0.0038 * fx_b
            rb_bt, rt_bt = [], []
            for fa_b in subdivide_1d_with_support_loops(4, sup1=0.06, sup2=0.18):
                yz_bt = _dm_yz(a_ring_div + (-0.0026 + 0.0052 * fa_b), 0.0)
                rb_bt.append(bm_sc.verts.new((xb_t, yz_bt[0] + n2_y * (h_ring_m + 0.0048), yz_bt[1] + n2_z * (h_ring_m + 0.0048))))
                rt_bt.append(bm_sc.verts.new((xb_t, yz_bt[0] + n2_y * z_top_bt, yz_bt[1] + n2_z * z_top_bt)))
            gb_bt.append(rb_bt)
            gt_bt.append(rt_bt)
        bm_build_beveled_quad_shell(bm_sc, gb_bt, gt_bt, bulge_scale=0.16)


        for a_rl in (spec.div_lug_a - 0.025, spec.div_lug_a + 0.025):
            yz_rl = _dm_yz(a_rl, 0.0216)
            bm_add_chamfered_cyl_x(
                bm_sc, yz_rl[0], yz_rl[1],
                side_s * 0.0192, side_s * 0.0205, radius=0.0036, segs=20, chamfer=0.00025,
            )
            bm_add_chamfered_cyl_x(
                bm_sc, yz_rl[0], yz_rl[1],
                side_s * 0.0175, side_s * 0.0216, radius=0.0020, segs=16,
            )
        for sa_gr in (-1.0, +1.0):
            yz_gr = _dm_yz(a_ring_div + sa_gr * 0.0094, 0.0)
            p_gr_b = Vector((side_s * 0.0182, yz_gr[0] + n2_y * (h_ring_m - 0.0014), yz_gr[1] + n2_z * (h_ring_m - 0.0014)))
            p_gr_t = Vector((side_s * 0.0182, yz_gr[0] + n2_y * (h_ring_m + 0.0034), yz_gr[1] + n2_z * (h_ring_m + 0.0034)))
            bm_add_segmented_tube(bm_sc, p_gr_b, p_gr_t, [(0.0, 0.0015), (0.2, 0.0019), (0.8, 0.0019), (1.0, 0.0015)], segs=14, cap_ends=True)

        xc = side_s * 0.0084
        bm_add_clevis_plate(bm_sc, _dm_yz(spec.div_lug_a, 0.0252), pd_0, 0.0185, 0.0102, xc - 0.0024, xc + 0.0024, arc_segs=23)
        bm_add_chamfered_cyl_x(bm_sc, pd_0[0], pd_0[1], xc - 0.0028, xc + 0.0028, radius=0.0084, segs=24, hollow_r=0.0048)

    bm_add_clevis_pin_assembly_x(bm_sc, pd_0[0], pd_0[1], -0.0138, +0.0138, pin_r=0.0048, segs=24)
    for a_wp_end in (a_car0 - 0.0025, a_car1 + 0.0025):
        bm_add_clevis_plate(bm_sc, _dm_yz(a_wp_end - 0.0018, 0.0252), _dm_yz(a_wp_end + 0.0018, 0.0252), 0.0024, 0.0024, -0.0156, +0.0156, arc_segs=9)
    for sx_zk in (-0.0084, +0.0084):
        bm_add_grease_zerk(bm_sc, Vector((sx_zk, pd_0[0] + n2_y * 0.0082, pd_0[1] + n2_z * 0.0082)), n2_vec, scale=0.82)
    meshes["slider_carriage"] = finalize_shared_mesh("Mesh_SliderCarriage", bm_sc, clay)

    bm_srp = bmesh.new()
    for side_s in (-1.0, +1.0):
        bm_add_spoked_roller_x(bm_srp, 0.0, 0.0, side_s * 0.0129, side_s * 0.0172, radius=0.0046, axle_r=0.0016, n_spokes=5)
    meshes["slider_roller_pair"] = finalize_shared_mesh("Mesh_SliderRollerPair", bm_srp, clay)
    meshes["_meta_slider_rl_f_yz"] = _dm_yz(spec.div_lug_a - 0.025, 0.0216)
    meshes["_meta_slider_rl_b_yz"] = _dm_yz(spec.div_lug_a + 0.025, 0.0216)


    for sc_hd in (0.13, 0.26):
        cy_hd = p1_0[0] + (p2_0[0] - p1_0[0]) * sc_hd
        cz_hd = p1_0[1] + (p2_0[1] - p1_0[1]) * sc_hd
        rc_hd = (1.0 - sc_hd) * r_p1_0 + sc_hd * r_p2_0
        ro_hd = (1.0 - sc_hd) * r_p1_1 + sc_hd * r_p2_1
        hw_hd = 0.945 * rc_hd * sin_hp
        x_in_hd = 0.0185
        for sx_hd in (-1.0, +1.0):
            xd0 = sx_hd * x_in_hd
            xd1 = sx_hd * max(0.028, 0.80 * hw_hd)
            ad_d0 = -(xd0 * xd0) / (2.0 * 1.08 * ro_hd) + 0.0040
            ad_d1 = -(xd1 * xd1) / (2.0 * 1.08 * ro_hd) + 0.0040
            bm_add_hat_stiffener_rib(
                bm_dm,
                Vector((xd0, cy_hd + n2_y * ad_d0, cz_hd + n2_z * ad_d0)),
                Vector((xd1, cy_hd + n2_y * ad_d1, cz_hd + n2_z * ad_d1)),
                t2_vec, n2_vec, w_foot=0.0062, w_top=0.0024, h_hat=0.0034, n_steps=5, rivet_r=0.0013,
            )


    f0_dm_skin = len(bm_dm.faces)
    t_fp_s = subdivide_1d_with_support_loops(12, sup1=0.02, sup2=0.065)
    t_fp_u = subdivide_1d_with_support_loops(8,  sup1=0.03, sup2=0.095)
    for sx_fp in (-1.0, +1.0):
        gb_fp, gt_fp = [], []
        for fs_f in t_fp_s:
            sc_f = 0.585 + 0.303 * fs_f
            cy_f = p1_0[0] + (p2_0[0] - p1_0[0]) * sc_f
            cz_f = p1_0[1] + (p2_0[1] - p1_0[1]) * sc_f
            rc_f = (1.0 - sc_f) * r_p1_0 + sc_f * r_p2_0
            ro_f = (1.0 - sc_f) * r_p1_1 + sc_f * r_p2_1
            hw_f = 0.945 * rc_f * sin_hp
            x_in_f = 0.0182
            x_out_f = 0.835 * hw_f
            rb_f, rt_f = [], []
            for fu_f in t_fp_u:
                xf_p = sx_fp * (x_in_f + (x_out_f - x_in_f) * fu_f)
                ad_f = -(xf_p * xf_p) / (2.0 * 1.08 * ro_f)
                lip_f = 1.0 - (0.52 * (((sc_f - 0.88) / 0.12) ** 1.35) if sc_f > 0.88 else 0.0)
                th_f = 0.0044 * lip_f

                seam_groove = -0.00045 if abs(fs_f - 0.50) < 0.028 else 0.0
                nb_f = ad_f + th_f + 0.00265
                nt_f = nb_f + 0.00145 + seam_groove
                rb_f.append(bm_dm.verts.new((xf_p, cy_f + n2_y * nb_f, cz_f + n2_z * nb_f)))
                rt_f.append(bm_dm.verts.new((xf_p, cy_f + n2_y * nt_f, cz_f + n2_z * nt_f)))
            gb_fp.append(rb_f)
            gt_fp.append(rt_f)
        bm_build_beveled_quad_shell(bm_dm, gb_fp, gt_fp, bulge_scale=0.16)

        for sc_rv in (0.600, 0.668, 0.736, 0.804, 0.872):
            cy_r = p1_0[0] + (p2_0[0] - p1_0[0]) * sc_rv
            cz_r = p1_0[1] + (p2_0[1] - p1_0[1]) * sc_rv
            rc_r = (1.0 - sc_rv) * r_p1_0 + sc_rv * r_p2_0
            ro_r = (1.0 - sc_rv) * r_p1_1 + sc_rv * r_p2_1
            hw_r = 0.945 * rc_r * sin_hp
            for fu_r in (0.09, 0.91):
                xr_p = sx_fp * (0.0182 + (0.835 * hw_r - 0.0182) * fu_r)
                ad_r = -(xr_p * xr_p) / (2.0 * 1.08 * ro_r) + 0.0044 + 0.00410
                bm_add_dome_rivet(bm_dm, Vector((xr_p, cy_r + n2_y * ad_r, cz_r + n2_z * ad_r)), n2_vec, radius=0.00125)
        for fu_m in (0.28, 0.52, 0.74):
            sc_m = 0.736
            cy_m = p1_0[0] + (p2_0[0] - p1_0[0]) * sc_m
            cz_m = p1_0[1] + (p2_0[1] - p1_0[1]) * sc_m
            rc_m = (1.0 - sc_m) * r_p1_0 + sc_m * r_p2_0
            ro_m = (1.0 - sc_m) * r_p1_1 + sc_m * r_p2_1
            hw_m = 0.945 * rc_m * sin_hp
            xr_m = sx_fp * (0.0182 + (0.835 * hw_m - 0.0182) * fu_m)
            ad_m = -(xr_m * xr_m) / (2.0 * 1.08 * ro_m) + 0.0044 + 0.0038
            bm_add_dome_rivet(bm_dm, Vector((xr_m, cy_m + n2_y * ad_m, cz_m + n2_z * ad_m)), n2_vec, radius=0.0012)
        if sx_fp > 0.0:

            sc_pn = 0.805
            cy_pn = p1_0[0] + (p2_0[0] - p1_0[0]) * sc_pn
            cz_pn = p1_0[1] + (p2_0[1] - p1_0[1]) * sc_pn
            rc_pn = (1.0 - sc_pn) * r_p1_0 + sc_pn * r_p2_0
            ro_pn = (1.0 - sc_pn) * r_p1_1 + sc_pn * r_p2_1
            hw_pn = 0.945 * rc_pn * sin_hp
            xr_pn = +0.46 * hw_pn
            ad_pn = -(xr_pn * xr_pn) / (2.0 * 1.08 * ro_pn) + 0.0044 + 0.00412
            p_pn_c = Vector((xr_pn, cy_pn + n2_y * ad_pn, cz_pn + n2_z * ad_pn))
            bm_add_3d_plate_text(
                bm_dm, ["P/N 407821", "TI-6AL-4V"], p_pn_c,
                t2_vec, Vector((-1.0, 0.0, 0.0)), n2_vec,
                width=0.036, height=0.013, relief=0.00038, with_border=True, with_dividers=True,
            )


        t_cb_s = subdivide_1d_with_support_loops(4, sup1=0.05, sup2=0.16)
        t_cb_u = subdivide_1d_with_support_loops(8, sup1=0.03, sup2=0.10)
        gb_cb, gt_cb = [], []
        for fs_cb in t_cb_s:
            sc_cb = 0.892 + 0.018 * fs_cb
            cy_cb = p1_0[0] + (p2_0[0] - p1_0[0]) * sc_cb
            cz_cb = p1_0[1] + (p2_0[1] - p1_0[1]) * sc_cb
            rc_cb = (1.0 - sc_cb) * r_p1_0 + sc_cb * r_p2_0
            ro_cb = (1.0 - sc_cb) * r_p1_1 + sc_cb * r_p2_1
            hw_cb = 0.945 * rc_cb * sin_hp
            x_cb0, x_cb1 = 0.135 * hw_cb, 0.878 * hw_cb
            rb_cb, rt_cb = [], []
            for fu_cb in t_cb_u:
                xc_p = sx_fp * (x_cb0 + (x_cb1 - x_cb0) * fu_cb)
                ad_cb = -(xc_p * xc_p) / (2.0 * 1.08 * ro_cb)
                lip_cb = 1.0 - 0.52 * (((sc_cb - 0.88) / 0.12) ** 1.35)
                th_cb = 0.0044 * lip_cb
                crown_cb = 0.0006 * math.sin(math.pi * fs_cb)
                nb_cb = ad_cb + th_cb + 0.0018
                nt_cb = nb_cb + 0.0014 + crown_cb
                rb_cb.append(bm_dm.verts.new((xc_p, cy_cb + n2_y * nb_cb, cz_cb + n2_z * nb_cb)))
                rt_cb.append(bm_dm.verts.new((xc_p, cy_cb + n2_y * nt_cb, cz_cb + n2_z * nt_cb)))
            gb_cb.append(rb_cb)
            gt_cb.append(rt_cb)
        bm_build_beveled_quad_shell(bm_dm, gb_cb, gt_cb, bulge_scale=0.16)
        cb_bolts = []
        for et_cbb in (0.235, 0.510, 0.805):
            sc_b2 = 0.901
            cy_b2 = p1_0[0] + (p2_0[0] - p1_0[0]) * sc_b2
            cz_b2 = p1_0[1] + (p2_0[1] - p1_0[1]) * sc_b2
            rc_b2 = (1.0 - sc_b2) * r_p1_0 + sc_b2 * r_p2_0
            ro_b2 = (1.0 - sc_b2) * r_p1_1 + sc_b2 * r_p2_1
            xb_2 = sx_fp * et_cbb * (0.945 * rc_b2 * sin_hp)
            nh_b2 = -(xb_2 * xb_2) / (2.0 * 1.08 * ro_b2) + 0.0044 * (1.0 - 0.52 * (((sc_b2 - 0.88) / 0.12) ** 1.35)) + 0.0036
            p_cbb = Vector((xb_2, cy_b2 + n2_y * nh_b2, cz_b2 + n2_z * nh_b2))
            cb_bolts.append(p_cbb)
            bm_add_aerospace_bolt(bm_dm, p_cbb, n2_vec, head_r=0.0015, height=0.0015)
        bm_add_bolt_pair_lockwire(bm_dm, cb_bolts[0], cb_bolts[1], n2_vec, head_r=0.0015, strand_r=0.00024)
        bm_add_bolt_pair_lockwire(bm_dm, cb_bolts[1], cb_bolts[2], n2_vec, head_r=0.0015, strand_r=0.00024)
    bm_fill_unset_from(bm_dm, f0_dm_skin, MAT_AVIONICS_BOX)


    f0_dm_hw = len(bm_dm.faces)
    for sc_br in (0.650, 0.785):
        a_br = sc_br * spec.L_div
        bm_add_clevis_plate(bm_dm, _dm_yz(a_br - 0.0045, 0.0256), _dm_yz(a_br + 0.0045, 0.0256), 0.0012, 0.0012, -0.0132, +0.0132, arc_segs=8)
        yz_bb = _dm_yz(a_br, 0.0266)
        p_br0 = Vector((-0.0096, yz_bb[0], yz_bb[1]))
        p_br1 = Vector((+0.0096, yz_bb[0], yz_bb[1]))
        bm_add_aerospace_bolt(bm_dm, p_br0, n2_vec, head_r=0.0015, height=0.0015)
        bm_add_aerospace_bolt(bm_dm, p_br1, n2_vec, head_r=0.0015, height=0.0015)
        bm_add_bolt_pair_lockwire(bm_dm, p_br0, p_br1, n2_vec, head_r=0.0015, strand_r=0.00022)
    for sx_ls in (-1.0, +1.0):
        yz_ls0 = _dm_yz(0.592 * spec.L_div, 0.0086)
        yz_ls1 = _dm_yz(0.878 * spec.L_div, 0.0082)
        bm_add_clevis_plate(bm_dm, yz_ls0, yz_ls1, 0.00055, 0.00055,
                            min(sx_ls * 0.0185, sx_ls * 0.0235), max(sx_ls * 0.0185, sx_ls * 0.0235), arc_segs=8)


    b2_ang = t2_0 + 0.5 * math.pi - half_dth_mid
    p1_dkn = (p1_0[0] + 0.0076 * math.cos(b2_ang), p1_0[1] + 0.0076 * math.sin(b2_ang))
    p1_root_dm = _dm_yz(0.023, 0.0048)
    p1_pad_dm  = _dm_yz(0.031, 0.0046)
    yz_rv_dm1 = (
        0.82 * p1_root_dm[0] + 0.18 * p1_dkn[0] + n2_y * 0.0052,
        0.82 * p1_root_dm[1] + 0.18 * p1_dkn[1] + n2_z * 0.0052,
    )
    yz_rv_dm2 = (
        0.52 * p1_root_dm[0] + 0.48 * p1_dkn[0] + n2_y * 0.0046,
        0.52 * p1_root_dm[1] + 0.48 * p1_dkn[1] + n2_z * 0.0046,
    )
    for x_kn in (-0.0465, -0.0235, 0.0, +0.0235, +0.0465):
        bm_add_clevis_plate(bm_dm, p1_root_dm, p1_pad_dm, 0.0024, 0.0020, x_kn - 0.0054, x_kn + 0.0054, arc_segs=9)
        bm_add_clevis_plate(bm_dm, p1_root_dm, p1_dkn, 0.0042, 0.0064, x_kn - 0.0038, x_kn + 0.0038, arc_segs=13)
        bm_add_clevis_plate(bm_dm, p1_pad_dm,  p1_dkn, 0.0028, 0.0052, x_kn - 0.0014, x_kn + 0.0014, arc_segs=11)
        bm_add_chamfered_cyl_x(bm_dm, p1_dkn[0], p1_dkn[1], x_kn - 0.0045, x_kn + 0.0045, radius=0.0068, segs=24, hollow_r=0.0042)
        bm_add_dome_rivet(bm_dm, Vector((x_kn, yz_rv_dm1[0], yz_rv_dm1[1])), n2_vec, radius=0.0014)
        bm_add_dome_rivet(bm_dm, Vector((x_kn, yz_rv_dm2[0], yz_rv_dm2[1])), n2_vec, radius=0.0013)
    bm_fill_unset_from(bm_dm, f0_dm_hw, MAT_MACHINED_ALU)


    _add_hotgas_tile_studs(bm_dm, p1_0, p2_0, r_p1_0, r_p2_0, r_p1_1, r_p2_1, True, 7, (-0.36, +0.36))
    hw_p2 = 0.945 * r_p2_0 * sin_hp * 0.964
    for et_vt in (-0.655, -0.365, 0.0, +0.365, +0.655):
        xv_t = et_vt * hw_p2
        ad_vt = -(xv_t * xv_t) / (2.0 * 1.08 * r_p2_1) + 0.00105
        yz_va = _dm_yz(spec.L_div - 0.014, ad_vt)
        yz_vb = _dm_yz(spec.L_div + 0.0018, ad_vt)
        bm_add_hollow_rect_duct(
            bm_dm, Vector((xv_t, yz_va[0], yz_va[1])), Vector((xv_t, yz_vb[0], yz_vb[1])),
            Vector((1.0, 0.0, 0.0)), n2_vec, 0.0036, 0.00075, 0.0026, 0.00038, n_steps=3,
        )
    meshes["div_master"] = finalize_shared_mesh("Mesh_DivMaster", bm_dm, clay)


    p2_seal_0 = (p2_0[0] - n2_y * dr_s, p2_0[1] - n2_z * dr_s)
    bm_ds = bmesh.new()
    add_highres_flap_shell(
        bm_ds, p1_seal_0, p2_seal_0, r_p1_0 - dr_s, r_p2_0 - dr_s, r_p1_1 - dr_s, r_p2_1 - dr_s,
        spec.n_sectors, width_factor=0.948, thickness=0.0036,
        is_outer_master=False, flap_role='DIV_SEAL', n_long=140, n_lat=72,
    )
    _add_seal_holddown_clips(bm_ds, p1_seal_0, p2_seal_0, r_p1_0 - dr_s, r_p2_0 - dr_s, n2_y, n2_z, (0.12, 0.25, 0.65, 0.84))
    _add_hotgas_tile_studs(bm_ds, p1_seal_0, p2_seal_0, r_p1_0 - dr_s, r_p2_0 - dr_s, r_p1_1 - dr_s, r_p2_1 - dr_s, False, 7, (-0.32, +0.32))
    p1_dskn = (p1_seal_0[0] + 0.0012 * math.cos(b2_ang), p1_seal_0[1] + 0.0012 * math.sin(b2_ang))
    p1_dsroot = (p1_seal_0[0] + 0.018 * c2_t + n2_y * 0.0008, p1_seal_0[1] + 0.018 * s2_t + n2_z * 0.0008)
    p1_dspad  = (p1_seal_0[0] + 0.025 * c2_t + n2_y * 0.0007, p1_seal_0[1] + 0.025 * s2_t + n2_z * 0.0007)
    yz_srv_d = (0.68 * p1_dsroot[0] + 0.32 * p1_dskn[0] + n2_y * 0.0024, 0.68 * p1_dsroot[1] + 0.32 * p1_dskn[1] + n2_z * 0.0024)

    for x_sk in (-0.0105, +0.0105):
        bm_add_clevis_plate(bm_ds, p1_dsroot, p1_dspad, 0.0018, 0.0014, x_sk - 0.0046, x_sk + 0.0046, arc_segs=9)
        bm_add_clevis_plate(bm_ds, p1_dsroot, p1_dskn,  0.0020, 0.0030, x_sk - 0.0032, x_sk + 0.0032, arc_segs=11)
        bm_add_chamfered_cyl_x(bm_ds, p1_dskn[0], p1_dskn[1], x_sk - 0.0038, x_sk + 0.0038, radius=0.0034, segs=18, hollow_r=0.0019)
        bm_add_dome_rivet(bm_ds, Vector((x_sk, yz_srv_d[0], yz_srv_d[1])), n2_vec, radius=0.0011)


    def _ds_yz(a_dist: float, h_norm: float) -> tuple:
        return (
            p1_seal_0[0] + a_dist * c2_t + n2_y * h_norm,
            p1_seal_0[1] + a_dist * s2_t + n2_z * h_norm,
        )


    bm_add_clevis_plate(bm_ds, _ds_yz(0.156, 0.0041), _ds_yz(0.348, 0.0041), 0.00065, 0.00065, -0.0021, +0.0021, arc_segs=8)


    a_fin_grid = [
        0.158, 0.161, 0.170, 0.174, 0.176, 0.178, 0.181, 0.183,
        0.210, 0.252, 0.294,
        0.321, 0.323, 0.326, 0.328, 0.330, 0.334, 0.343, 0.346,
    ]
    x_fin_eta = [-1.00, -0.90, -0.65, 0.00, +0.65, +0.90, +1.00]
    gb_fin, gt_fin = [], []
    for af in a_fin_grid:
        if af < 0.172:
            ramp = max(0.0, (af - 0.158) / 0.014)
        elif af > 0.332:
            ramp = max(0.0, (0.346 - af) / 0.014)
        else:
            ramp = 1.0
        stop_bump = 0.0
        if 0.175 < af < 0.183:
            stop_bump = 0.0014 * min(1.0, min(af - 0.175, 0.183 - af) / 0.002)
        elif 0.321 < af < 0.329:
            stop_bump = 0.0014 * min(1.0, min(af - 0.321, 0.329 - af) / 0.002)
        h_top_f = 0.0046 + 0.0122 * ramp + stop_bump
        yz_fb = _ds_yz(af, 0.0043)
        rb_f, rt_f = [], []
        for ef in x_fin_eta:
            cham_f = 0.00022 * max(0.0, (abs(ef) - 0.65) / 0.35)
            xf = ef * 0.00068
            yz_ft = _ds_yz(af, h_top_f - cham_f)
            rb_f.append(bm_ds.verts.new((xf, yz_fb[0], yz_fb[1])))
            rt_f.append(bm_ds.verts.new((xf, yz_ft[0], yz_ft[1])))
        gb_fin.append(rb_f)
        gt_fin.append(rt_f)
    bm_build_beveled_quad_shell(bm_ds, gb_fin, gt_fin, bulge_scale=0.14)
    meshes["div_seal"] = finalize_shared_mesh("Mesh_DivSeal", bm_ds, clay)


    h_ring_s = 0.0235
    yz_rear_sad = _ds_yz(a_ring_div, h_ring_s)
    bm_rks = bmesh.new()
    bm_add_ring_channel_saddle(
        bm_rks, yz_rear_sad, t2_vec, n2_vec,
        half_span_x=0.0215, half_chord_a=0.0128,
        h_foot=-0.0078, h_tunnel_bot=-0.0032, h_tunnel_top=+0.0032, h_roof=+0.0064,
        rho_curve=0.50, is_through_tunnel=True,
    )
    meshes["ring_keeper_saddle"] = finalize_shared_mesh("Mesh_RingKeeperSaddle", bm_rks, clay)


    bm_bell = bmesh.new()
    centroid_b = (
        (pb[0] + ain_0[0] + aout_0[0]) / 3.0,
        (pb[1] + ain_0[1] + aout_0[1]) / 3.0,
    )
    for side_s in (-1.0, 1.0):
        xc_base = side_s * fh
        for sub_c in (-0.0074, +0.0074):
            xc = xc_base + sub_c

            bm_add_clevis_plate(bm_bell, pb, ain_0,     0.01380, 0.00760, xc - 0.00195, xc + 0.00195, arc_segs=25)
            bm_add_clevis_plate(bm_bell, pb, aout_0,    0.01350, 0.01040, xc - 0.00180, xc + 0.00180, arc_segs=25)
            bm_add_clevis_plate(bm_bell, ain_0, aout_0, 0.00735, 0.01010, xc - 0.00165, xc + 0.00165, arc_segs=23)

            in_pb   = (0.68 * pb[0]    + 0.32 * centroid_b[0], 0.68 * pb[1]    + 0.32 * centroid_b[1])
            in_ain  = (0.68 * ain_0[0] + 0.32 * centroid_b[0], 0.68 * ain_0[1] + 0.32 * centroid_b[1])
            in_aout = (0.68 * aout_0[0]+ 0.32 * centroid_b[0], 0.68 * aout_0[1]+ 0.32 * centroid_b[1])
            bm_add_clevis_plate(bm_bell, in_pb,  in_ain,  0.0160, 0.0140, xc - 0.00090, xc + 0.00090, arc_segs=13)
            bm_add_clevis_plate(bm_bell, in_pb,  in_aout, 0.0157, 0.0148, xc - 0.00076, xc + 0.00076, arc_segs=13)
            for idx_sp, (v_pt, r_sp) in enumerate(((pb, 0.0046), (ain_0, 0.0042), (aout_0, 0.0046))):
                sp_half = 0.00150 - idx_sp * 0.00014
                bm_add_clevis_plate(bm_bell, v_pt, centroid_b, r_sp, 0.0046 - idx_sp * 0.00022, xc - sp_half, xc + sp_half, arc_segs=13)

            bm_add_chamfered_cyl_x(bm_bell, centroid_b[0], centroid_b[1], xc - 0.0021, xc + 0.0021, radius=0.0108, segs=28, chamfer=0.00045, hollow_r=0.0064)

            for v_pt, r_bsh in [(pb, 0.0114), (ain_0, 0.0078), (aout_0, 0.0094)]:
                bm_add_chamfered_cyl_x(bm_bell, v_pt[0], v_pt[1], xc - 0.0022, xc + 0.0022, radius=r_bsh, segs=26, hollow_r=r_bsh * 0.52)


        yz_tie_b = (0.56 * pb[0] + 0.44 * centroid_b[0], 0.56 * pb[1] + 0.44 * centroid_b[1])
        bm_add_chamfered_cyl_x(bm_bell, yz_tie_b[0], yz_tie_b[1], xc_base - 0.0056, xc_base + 0.0056, radius=0.0038, segs=18)

    bm_add_segmented_tube(
        bm_bell, Vector((-(fh - 0.0074), pb[0], pb[1])), Vector((+(fh - 0.0074), pb[0], pb[1])),
        [(0.0, 0.0096), (0.16, 0.0096), (0.20, 0.0082), (0.44, 0.0102), (0.56, 0.0102), (0.80, 0.0082), (0.84, 0.0096), (1.0, 0.0096)],
        segs=28, cap_ends=True,
    )
    for side_s in (-1.0, 1.0):
        xc = side_s * fh
        bm_add_clevis_pin_assembly_x(
            bm_bell, aout_0[0], aout_0[1], xc - 0.0096, xc + 0.0096,
            pin_r=0.0050, segs=24, compact_ends=True, flip_x=(side_s < 0.0),
        )
        bm_add_clevis_pin_assembly_x(
            bm_bell, ain_0[0], ain_0[1], xc - 0.0094, xc + 0.0094,
            pin_r=0.0040, segs=24, compact_ends=True, flip_x=(side_s > 0.0),
        )
        bm_add_grease_zerk(bm_bell, Vector((xc + side_s * 0.0076, aout_0[0], aout_0[1] + 0.0092)), Vector((0.0, 0.0, 1.0)), scale=0.80)
        bm_add_grease_zerk(bm_bell, Vector((xc + side_s * 0.0076, pb[0] - 0.0085, pb[1] + 0.0075)), Vector((0.0, -0.65, 0.76)).normalized(), scale=0.82)
    meshes["bellcrank"] = finalize_shared_mesh("Mesh_Bellcrank", bm_bell, clay)


    bm_sl = bmesh.new()
    sl_dir = Vector((0.0, pring_0[0] - ain_0[0], pring_0[1] - ain_0[1])).normalized()
    sl_up  = Vector((0.0, sl_dir.z, -sl_dir.y)).normalized()
    if sl_up.z < 0.0:
        sl_up = -sl_up
    for side_s in (-1.0, 1.0):
        xc = side_s * fh
        p_sl_a = Vector((xc, ain_0[0], ain_0[1]))
        p_sl_b = Vector((xc, pring_0[0], pring_0[1]))
        bm_add_flattened_aero_strut(
            bm_sl, p_sl_a, p_sl_b,
            half_w_mid=0.0068, half_h_mid=0.0042,
            end_half_w=0.0038, end_half_h=0.0040,
            eye_r=0.0076, eye_half_w=0.0040, taper_frac=0.24, pin_along_x=True, ribbed_pockets=2,
        )
        bm_add_chamfered_cyl_x(bm_sl, ain_0[0],   ain_0[1],   xc - 0.0045, xc + 0.0045, radius=0.0054, segs=20, hollow_r=0.0040)
        bm_add_chamfered_cyl_x(bm_sl, pring_0[0], pring_0[1], xc - 0.0045, xc + 0.0045, radius=0.0056, segs=20, hollow_r=0.0046)
        for f_s0, f_s1, f_h0, f_h1 in ((0.065, 0.205, 0.115, 0.158), (0.935, 0.795, 0.885, 0.842)):
            bm_add_helical_thread_cylinder(
                bm_sl, p_sl_a.lerp(p_sl_b, f_s0), p_sl_a.lerp(p_sl_b, f_h0),
                r_core=0.0032, thread_h=0.00058, n_turns=4, segs_turn=14, cap_core=False,
            )
            bm_add_bevel_locked_hex(
                bm_sl, p_sl_a.lerp(p_sl_b, min(f_h0, f_h1)), p_sl_a.lerp(p_sl_b, max(f_h0, f_h1)),
                [(0.0, 0.0036), (0.20, 0.0041), (0.80, 0.0041), (1.0, 0.0036)],
                corner_bevel=0.14, cap_ends=True,
            )
            bm_add_twisted_lockwire(
                bm_sl,
                [
                    p_sl_a.lerp(p_sl_b, 0.5 * (f_h0 + f_h1)) + sl_up * 0.0042,
                    p_sl_a.lerp(p_sl_b, 0.5 * (f_h1 + f_s1)) + sl_up * 0.0048,
                    p_sl_a.lerp(p_sl_b, f_s1) + sl_up * 0.0044,
                ],
                strand_r=0.00022, twists_per_m=260.0, n_samples=14, segs=10,
            )
    meshes["short_link"] = finalize_shared_mesh("Mesh_ShortLink", bm_sl, clay)


    bm_pr = bmesh.new()
    p_pr_a_l = Vector((-fh, pc_0[0], pc_0[1]))
    p_pr_b_l = Vector((-fh, aout_0[0], aout_0[1]))
    p_pr_a_r = Vector((+fh, pc_0[0], pc_0[1]))
    p_pr_b_r = Vector((+fh, aout_0[0], aout_0[1]))
    pr_dir_yz = Vector((0.0, aout_0[0] - pc_0[0], aout_0[1] - pc_0[1])).normalized()
    pr_nrm_yz = Vector((0.0, -pr_dir_yz.z, pr_dir_yz.y)).normalized()
    pr_out_up = -pr_nrm_yz if pr_nrm_yz.z < 0.0 else pr_nrm_yz
    for side_s in (-1.0, 1.0):
        xc = side_s * fh
        p_pc_v = Vector((xc, pc_0[0], pc_0[1]))
        p_ao_v = Vector((xc, aout_0[0], aout_0[1]))
        bm_add_flattened_aero_strut(
            bm_pr,
            p_pc_v,
            p_ao_v,
            half_w_mid=0.0078, half_h_mid=0.0045,
            end_half_w=0.0034, end_half_h=0.0035,
            eye_r=0.0096, eye_half_w=0.0040, taper_frac=0.25, pin_along_x=True, ribbed_pockets=2,
        )

        bm_add_chamfered_cyl_x(bm_pr, pc_0[0],   pc_0[1],   xc - 0.0058, xc + 0.0058, radius=0.0066, segs=22, hollow_r=0.0047)
        bm_add_chamfered_cyl_x(bm_pr, aout_0[0], aout_0[1], xc - 0.0050, xc + 0.0050, radius=0.0066, segs=22, hollow_r=0.0050)
        for f_sh0, f_sh1, f_hx0, f_hx1 in ((0.036, 0.124, 0.064, 0.092), (0.964, 0.876, 0.936, 0.908)):
            bm_add_segmented_tube(
                bm_pr, p_pc_v.lerp(p_ao_v, f_sh0), p_pc_v.lerp(p_ao_v, f_sh1),
                [(0.0, 0.0046), (0.08, 0.0042), (0.44, 0.0042), (0.50, 0.0060), (0.92, 0.0060), (1.0, 0.0052)],
                segs=22, cap_ends=True,
            )
            bm_add_helical_thread_cylinder(
                bm_pr, p_pc_v.lerp(p_ao_v, f_sh0 + (f_sh1 - f_sh0) * 0.05), p_pc_v.lerp(p_ao_v, f_hx0),
                r_core=0.0042, thread_h=0.00085, n_turns=4, segs_turn=16, cap_core=False,
            )
            bm_add_bevel_locked_hex(
                bm_pr, p_pc_v.lerp(p_ao_v, min(f_hx0, f_hx1)), p_pc_v.lerp(p_ao_v, max(f_hx0, f_hx1)),
                [(0.0, 0.0058), (0.18, 0.0066), (0.82, 0.0066), (1.0, 0.0058)],
                corner_bevel=0.14, cap_ends=True,
            )
            f_lug_pr = f_sh0 + (f_sh1 - f_sh0) * 0.86
            f_jm_pr  = 0.5 * (f_hx0 + f_hx1)
            p_pr_lug = p_pc_v.lerp(p_ao_v, f_lug_pr) + pr_out_up * 0.0054 + Vector((side_s * 0.0018, 0.0, 0.0))
            bm_add_segmented_tube(
                bm_pr, p_pr_lug - pr_out_up * 0.0008, p_pr_lug + pr_out_up * 0.0012,
                [(0.0, 0.0011), (0.5, 0.0011), (1.0, 0.0008)], segs=12, cap_ends=True,
            )
            bm_add_twisted_lockwire(bm_pr, [
                p_pc_v.lerp(p_ao_v, f_jm_pr) + pr_out_up * 0.0046 - Vector((side_s * 0.0042, 0.0, 0.0)),
                p_pc_v.lerp(p_ao_v, 0.5 * (f_jm_pr + f_lug_pr)) + pr_out_up * 0.0062,
                p_pr_lug + pr_out_up * 0.0008,
                p_pc_v.lerp(p_ao_v, f_sh1) + pr_out_up * 0.0056 + Vector((side_s * 0.0024, 0.0, 0.0)),
            ], strand_r=0.00028, twists_per_m=240.0, n_samples=18, segs=10)
    for p_ra, p_rb in ((p_pr_a_l, p_pr_b_l), (p_pr_a_r, p_pr_b_r)):
        bm_add_segmented_tube(
            bm_pr, p_ra.lerp(p_rb, 0.785), p_ra.lerp(p_rb, 0.835),
            [
                (0.00, 0.0072), (0.12, 0.0084), (0.28, 0.0084), (0.50, 0.0078),
                (0.72, 0.0084), (0.88, 0.0084), (1.00, 0.0072),
            ],
            segs=26, cap_ends=True,
        )
    c_br_0 = Vector((0.0, pc_0[0], pc_0[1])).lerp(Vector((0.0, aout_0[0], aout_0[1])), 0.796)
    c_br_1 = Vector((0.0, pc_0[0], pc_0[1])).lerp(Vector((0.0, aout_0[0], aout_0[1])), 0.824)
    bm_add_clevis_plate(bm_pr, (c_br_0.y, c_br_0.z), (c_br_1.y, c_br_1.z), 0.0040, 0.0040, -(fh - 0.0045), +(fh - 0.0045), arc_segs=13)
    for c_br in (c_br_0, c_br_1):
        bm_add_segmented_tube(
            bm_pr, Vector((-(fh + 0.001), c_br.y, c_br.z)), Vector((+(fh + 0.001), c_br.y, c_br.z)),
            [(0.0, 0.0042), (0.12, 0.0048), (0.88, 0.0048), (1.0, 0.0042)],
            segs=20, cap_ends=True,
        )
    for sx_cl in (-1.0, +1.0):
        xb_cl = sx_cl * (fh - 0.0078)
        pb_c0 = Vector((xb_cl, c_br_0.y, c_br_0.z)) + pr_nrm_yz * 0.0044
        pb_c1 = Vector((xb_cl, c_br_1.y, c_br_1.z)) + pr_nrm_yz * 0.0044
        bm_add_aerospace_bolt(bm_pr, pb_c0, pr_nrm_yz, head_r=0.0021, height=0.0022)
        bm_add_aerospace_bolt(bm_pr, pb_c1, pr_nrm_yz, head_r=0.0021, height=0.0022)
        bm_add_bolt_pair_lockwire(bm_pr, pb_c0, pb_c1, pr_nrm_yz, head_r=0.0021, strand_r=0.00030)
    meshes["pull_rod"] = finalize_shared_mesh("Mesh_TwinFork_PullRod", bm_pr, clay)


    bm_ds_rod = bmesh.new()
    p_pd_v = Vector((0.0, pd_0[0], pd_0[1]))
    p_ps_v = Vector((0.0, ps[0], ps[1]))
    bm_add_flattened_aero_strut(
        bm_ds_rod,
        p_pd_v,
        p_ps_v,
        half_w_mid=0.0094, half_h_mid=0.0048,
        end_half_w=0.0036, end_half_h=0.0038,
        eye_r=0.0104, eye_half_w=0.0042, taper_frac=0.20, pin_along_x=True, with_flutes=True, ribbed_pockets=4,
    )
    bm_add_chamfered_cyl_x(bm_ds_rod, pd_0[0], pd_0[1], -0.0056, +0.0056, radius=0.0070, segs=24, hollow_r=0.0049)
    bm_add_chamfered_cyl_x(bm_ds_rod, ps[0],   ps[1],   -0.0052, +0.0052, radius=0.0068, segs=24, hollow_r=0.0050)
    ds_dir = (p_ps_v - p_pd_v).normalized()
    ds_up  = Vector((0.0, ds_dir.z, -ds_dir.y)).normalized()
    for p_end, u_in_d in ((p_pd_v, ds_dir), (p_ps_v, -ds_dir)):
        bm_add_segmented_tube(
            bm_ds_rod, p_end + u_in_d * 0.0115, p_end + u_in_d * 0.0440,
            [
                (0.00, 0.0049), (0.08, 0.0044), (0.46, 0.0044),
                (0.56, 0.0064), (0.90, 0.0064), (1.00, 0.0055),
            ],
            segs=24, cap_ends=True,
        )
        bm_add_helical_thread_cylinder(
            bm_ds_rod, p_end + u_in_d * 0.0122, p_end + u_in_d * 0.0215,
            r_core=0.0044, thread_h=0.00090, n_turns=5, segs_turn=18, cap_core=False,
        )
        bm_add_bevel_locked_hex(
            bm_ds_rod, p_end + u_in_d * 0.0205, p_end + u_in_d * 0.0290,
            [(0.0, 0.0064), (0.18, 0.0074), (0.82, 0.0074), (1.0, 0.0064)],
            corner_bevel=0.14, cap_ends=True,
        )
        p_ds_lug = p_end + u_in_d * 0.0365 + ds_up * 0.0060 + Vector((0.0028, 0.0, 0.0))
        bm_add_segmented_tube(
            bm_ds_rod, p_ds_lug - ds_up * 0.0010, p_ds_lug + ds_up * 0.0016,
            [(0.0, 0.0014), (0.5, 0.0014), (1.0, 0.0010)], segs=12, cap_ends=True,
        )
        bm_add_twisted_lockwire(bm_ds_rod, [
            p_end + u_in_d * 0.0235 + ds_up * 0.0050 - Vector((0.0052, 0.0, 0.0)),
            p_end + u_in_d * 0.0260 + ds_up * 0.0070 - Vector((0.0022, 0.0, 0.0)),
            p_end + u_in_d * 0.0315 + ds_up * 0.0068 + Vector((0.0008, 0.0, 0.0)),
            p_end + u_in_d * 0.0365 + ds_up * 0.0068 + Vector((0.0028, 0.0, 0.0)),
            p_end + u_in_d * 0.0402 + ds_up * 0.0064 + Vector((0.0036, 0.0, 0.0)),
        ], strand_r=0.00032, twists_per_m=240.0, n_samples=22, segs=12)

    ds_mid_p = p_pd_v.lerp(p_ps_v, 0.50)
    ds_rot_m = Matrix.Identity(4)
    ds_rot_m[0][0], ds_rot_m[1][0], ds_rot_m[2][0] = 1.0, 0.0, 0.0
    ds_rot_m[0][1], ds_rot_m[1][1], ds_rot_m[2][1] = ds_dir.x, ds_dir.y, ds_dir.z
    ds_rot_m[0][2], ds_rot_m[1][2], ds_rot_m[2][2] = ds_up.x,  ds_up.y,  ds_up.z
    ds_rot_m[0][3], ds_rot_m[1][3], ds_rot_m[2][3] = ds_mid_p.x, ds_mid_p.y, ds_mid_p.z
    b_ns_plq = bmesh.new()
    f0_nsp = len(b_ns_plq.faces)
    _bm_add_box_x_yz(b_ns_plq, -0.0056, +0.0056, -0.046, +0.046, 0.0044, 0.0053, nx=2, ny=4, nz=1, chamfer=0.00025)
    bm_fill_unset_from(b_ns_plq, f0_nsp, MAT_PLATE_BASE)
    for sy_nsr in (-0.041, +0.041):
        bm_add_dome_rivet(b_ns_plq, Vector((0.0, sy_nsr, 0.0053)), Vector((0, 0, 1)), radius=0.00085)
    bm_merge(bm_ds_rod, b_ns_plq, ds_rot_m)
    meshes["diag_strut"] = finalize_shared_mesh("Mesh_Diagonal_Strut", bm_ds_rod, clay)


    p_piv_bm0 = _cm_pt(x_bm, a_bm, h_blk_cm)
    p_aft_bm0 = _cm_pt(x_bm + dx_aft_bm, a_bm + da_aft_bm, h_blk_cm)
    p_mid_bm0 = _cm_pt(x_bm + dx_mid_bm, a_bm + da_mid_bm, h_blk_cm)
    p_fwd_bm0 = _cm_pt(x_bm + dx_fwd_bm, a_bm + da_fwd_bm, h_blk_cm + 0.0035)
    p_guide_0 = Vector((+0.0510, -0.014, 0.588))

    p_sqA_s0  = rot_s0 @ p_sqA_seal_loc
    p_sqB_s15 = rot_s15 @ p_sqB_seal_loc


    bm_boom = bmesh.new()
    s_sp = sorted(set(
        subdivide_1d_with_support_loops(18, sup1=0.02, sup2=0.06)
        + [0.115, 0.125, 0.135, 0.415, 0.425, 0.435, 0.565, 0.575, 0.585, 0.865, 0.875, 0.885]
    ))
    u_sp = [-1.00, -0.94, -0.80, -0.68, -0.63, -0.45, 0.00, +0.45, +0.63, +0.68, +0.80, +0.94, +1.00]
    gb_bm, gt_bm = [], []
    p_elbow_bm = _cm_pt(x_bm + 0.005, a_bm + 0.004, h_blk_cm + 0.0008)
    lh_centers_bm = []
    for fs_b in s_sp:
        om = 1.0 - fs_b
        ctr_b = (p_aft_bm0 * (om * om) + p_elbow_bm * (2.0 * om * fs_b) + p_fwd_bm0 * (fs_b * fs_b))
        if abs(fs_b - 0.275) < 1e-4 or abs(fs_b - 0.725) < 1e-4:
            lh_centers_bm.append(ctr_b)
        d_next = min(1.0, fs_b + 0.01)
        d_prev = max(0.0, fs_b - 0.01)
        tang_b = (
            (p_aft_bm0 * ((1 - d_next) ** 2) + p_elbow_bm * (2 * (1 - d_next) * d_next) + p_fwd_bm0 * (d_next ** 2))
            - (p_aft_bm0 * ((1 - d_prev) ** 2) + p_elbow_bm * (2 * (1 - d_prev) * d_prev) + p_fwd_bm0 * (d_prev ** 2))
        ).normalized()
        lat_b = n1_vec.cross(tang_b).normalized()
        hw_b = 0.0052 + 0.0066 * math.sin(math.pi * fs_b) ** 0.85

        in_pkt_b = 0.0
        for (pa_b, pb_b) in ((0.125, 0.425), (0.575, 0.875)):
            if pa_b < fs_b < pb_b:
                in_pkt_b = max(in_pkt_b, min(1.0, min(fs_b - pa_b, pb_b - fs_b) / 0.010))
        rim_env = math.sin(math.pi * min(1.0, max(0.0, (fs_b - 0.05) / 0.90))) ** 0.45

        rb_b, rt_b = [], []
        for eu_b in u_sp:
            ae_b = abs(eu_b)
            rim_wall = min(1.0, max(0.0, (ae_b - 0.63) / 0.05))
            pkt_lat  = min(1.0, max(0.0, (0.63 - ae_b) / 0.05))
            rim = 0.0016 * rim_wall * rim_env
            pkt = 0.0014 * pkt_lat * in_pkt_b
            pt_base = ctr_b + lat_b * (eu_b * hw_b)
            rb_b.append(bm_boom.verts.new(pt_base - n1_vec * (0.0021 + 0.55 * rim - 0.45 * pkt)))
            rt_b.append(bm_boom.verts.new(pt_base + n1_vec * (0.0021 + rim - pkt)))
        gb_bm.append(rb_b)
        gt_bm.append(rt_b)
    bm_build_beveled_quad_shell(bm_boom, gb_bm, gt_bm, bulge_scale=0.18)


    for fs_lh in (0.275, 0.725):
        om_l = 1.0 - fs_lh
        c_lh = p_aft_bm0 * (om_l * om_l) + p_elbow_bm * (2.0 * om_l * fs_lh) + p_fwd_bm0 * (fs_lh * fs_lh)
        bm_add_segmented_tube(
            bm_boom, c_lh + n1_vec * 0.0005, c_lh + n1_vec * 0.0026,
            [(0.0, 0.0036), (0.35, 0.0036), (0.55, 0.0023), (0.85, 0.0023), (1.0, 0.0018)],
            segs=20, cap_ends=True,
        )


    s_md = sorted(set(subdivide_1d_with_support_loops(8, sup1=0.04, sup2=0.12) + [0.22, 0.25, 0.74, 0.77]))
    gb_md, gt_md = [], []
    tang_md = (p_mid_bm0 - p_piv_bm0).normalized()
    lat_md = n1_vec.cross(tang_md).normalized()
    for fs_m in s_md:
        ctr_m = p_piv_bm0.lerp(p_mid_bm0, fs_m)
        hw_m = 0.0088 * (1.0 - 0.38 * fs_m)
        in_pkt_m = min(1.0, min(fs_m - 0.22, 0.77 - fs_m) / 0.03) if 0.22 < fs_m < 0.77 else 0.0
        rb_m, rt_m = [], []
        for eu_b in u_sp:
            ae_m = abs(eu_b)
            rim_m = 0.0013 * min(1.0, max(0.0, (ae_m - 0.63) / 0.05))
            pkt_m = 0.0011 * min(1.0, max(0.0, (0.63 - ae_m) / 0.05)) * in_pkt_m
            pt_m = ctr_m + lat_md * (eu_b * hw_m)
            rb_m.append(bm_boom.verts.new(pt_m - n1_vec * (0.0020 + 0.4 * rim_m - 0.4 * pkt_m)))
            rt_m.append(bm_boom.verts.new(pt_m + n1_vec * (0.0020 + rim_m - pkt_m)))
        gb_md.append(rb_m)
        gt_md.append(rt_m)
    bm_build_beveled_quad_shell(bm_boom, gb_md, gt_md, bulge_scale=0.18)


    for p_eye, r_eye, h_eye in (
        (p_piv_bm0, 0.0124, 0.0030),
        (p_aft_bm0, 0.0064, 0.0034),
        (p_mid_bm0, 0.0062, 0.0034),
        (p_fwd_bm0, 0.0066, 0.0034),
    ):
        bm_add_segmented_tube(
            bm_boom, p_eye - n1_vec * h_eye, p_eye + n1_vec * h_eye,
            [(0.0, r_eye * 0.86), (0.14, r_eye), (0.86, r_eye), (1.0, r_eye * 0.86)],
            segs=24, cap_ends=True,
        )
        if r_eye < 0.010:
            bm_add_segmented_tube(
                bm_boom, p_eye - n1_vec * (h_eye + 0.0016), p_eye + n1_vec * (h_eye + 0.0036),
                [
                    (0.00, r_eye * 0.56), (0.14, r_eye * 0.62), (0.20, r_eye * 0.42),
                    (0.66, r_eye * 0.42), (0.70, r_eye * 0.72), (0.78, r_eye * 0.72),
                    (0.82, r_eye * 0.34), (0.96, r_eye * 0.34), (1.00, r_eye * 0.25),
                ],
                segs=20, cap_ends=True,
            )
            bm_add_bevel_locked_hex(
                bm_boom, p_eye + n1_vec * (h_eye + 0.0006), p_eye + n1_vec * (h_eye + 0.0024),
                [(0.0, r_eye * 0.50), (0.20, r_eye * 0.58), (0.80, r_eye * 0.58), (1.0, r_eye * 0.48)],
                corner_bevel=0.14, cap_ends=True,
            )
            p_ct_o = p_eye + n1_vec * (h_eye + 0.0028)
            for sgn_co in (-1.0, +1.0):
                bm_add_spline_tube(bm_boom, [
                    p_ct_o - t1_vec * 0.0028 + ex_vec * (sgn_co * 0.00028),
                    p_ct_o + ex_vec * (sgn_co * 0.00028),
                    p_ct_o + t1_vec * 0.0024 + ex_vec * (sgn_co * 0.0016) - n1_vec * 0.0006,
                ], radius=0.00026, n_samples=12, segs=8, cap_ends=True)
    meshes["boomerang"] = finalize_shared_mesh("Mesh_Boomerang", bm_boom, clay)


    def _add_rod_end_jam_collar(bm_rod: bmesh.types.BMesh, p_clevis: Vector, p_other: Vector):
        u_in = (p_other - p_clevis).normalized()
        v_up = (n1_vec - u_in * n1_vec.dot(u_in)).normalized()
        w_lat = v_up.cross(u_in).normalized()
        bm_add_segmented_tube(
            bm_rod, p_clevis + u_in * 0.0056, p_clevis + u_in * 0.0182,
            [(0.00, 0.0028), (0.08, 0.0022), (0.64, 0.0022), (0.72, 0.0031), (0.90, 0.0032), (1.00, 0.0026)],
            segs=20, cap_ends=True,
        )
        bm_add_helical_thread_cylinder(
            bm_rod, p_clevis + u_in * 0.0062, p_clevis + u_in * 0.0142,
            r_core=0.0022, thread_h=0.00055, n_turns=5, segs_turn=16, cap_core=False,
        )
        bm_add_bevel_locked_hex(
            bm_rod, p_clevis + u_in * 0.0138, p_clevis + u_in * 0.0166,
            [(0.0, 0.0036), (0.18, 0.0039), (0.82, 0.0039), (1.0, 0.0035)],
            corner_bevel=0.14, cap_ends=True,
        )

        p_lug = p_clevis + u_in * 0.0232 + v_up * 0.0019 + w_lat * 0.0024
        bm_add_segmented_tube(
            bm_rod, p_lug - v_up * 0.0006, p_lug + v_up * 0.0011,
            [(0.0, 0.0011), (0.30, 0.0012), (0.80, 0.0011), (1.0, 0.00075)],
            segs=14, cap_ends=True,
        )
        bm_add_twisted_lockwire(
            bm_rod,
            [
                p_clevis + u_in * 0.0144 + v_up * 0.0024 - w_lat * 0.0031,
                p_clevis + u_in * 0.0154 + v_up * 0.0036 - w_lat * 0.0018,
                p_clevis + u_in * 0.0168 + v_up * 0.0033 - w_lat * 0.0004,
                p_clevis + u_in * 0.0196 + v_up * 0.0029 + w_lat * 0.0010,
                p_clevis + u_in * 0.0223 + v_up * 0.0026 + w_lat * 0.0021,
                p_clevis + u_in * 0.0235 + v_up * 0.0025 + w_lat * 0.0031,
                p_clevis + u_in * 0.0262 + v_up * 0.0022 + w_lat * 0.0035,
            ],
            strand_r=0.00027, twists_per_m=250.0, n_samples=26, segs=12,
        )

    bm_tra = bmesh.new()
    bm_add_flattened_aero_strut(
        bm_tra, p_aft_bm0, p_sqA_s0,
        half_w_mid=0.0042, half_h_mid=0.0022,
        end_half_w=0.0026, end_half_h=0.0019,
        eye_r=0.0049, eye_half_w=0.0019, taper_frac=0.28,
        pin_along_x=False, up_hint=n1_vec, with_flutes=True,
    )
    _add_rod_end_jam_collar(bm_tra, p_sqA_s0, p_aft_bm0)
    meshes["trans_rod_a"] = finalize_shared_mesh("Mesh_TransRodA", bm_tra, clay)

    bm_trb = bmesh.new()
    bm_add_flattened_aero_strut(
        bm_trb, p_mid_bm0, p_sqB_s15,
        half_w_mid=0.0042, half_h_mid=0.0022,
        end_half_w=0.0026, end_half_h=0.0019,
        eye_r=0.0049, eye_half_w=0.0019, taper_frac=0.22,
        pin_along_x=False, up_hint=n1_vec, with_flutes=True,
    )
    _add_rod_end_jam_collar(bm_trb, p_sqB_s15, p_mid_bm0)
    meshes["trans_rod_b"] = finalize_shared_mesh("Mesh_TransRodB", bm_trb, clay)


    bm_bfr = bmesh.new()
    dir_fg0 = (p_guide_0 - p_fwd_bm0).normalized()
    bfr_up = (n1_vec - dir_fg0 * n1_vec.dot(dir_fg0)).normalized()
    p_rod_end = p_fwd_bm0 + dir_fg0 * 0.228
    bm_add_segmented_tube(
        bm_bfr, p_fwd_bm0, p_rod_end,
        [
            (0.00, 0.0042), (0.040, 0.0048), (0.048, 0.0029), (0.100, 0.0029),
            (0.108, 0.0038), (0.155, 0.0038), (0.175, 0.0033),
            (0.96, 0.0033), (1.00, 0.0024),
        ],
        segs=22, cap_ends=True,
    )
    bm_add_helical_thread_cylinder(
        bm_bfr, p_fwd_bm0 + dir_fg0 * 0.010, p_fwd_bm0 + dir_fg0 * 0.023,
        r_core=0.0029, thread_h=0.00072, n_turns=5, segs_turn=16, cap_core=False,
    )
    bm_add_bevel_locked_hex(
        bm_bfr, p_fwd_bm0 + dir_fg0 * 0.022, p_fwd_bm0 + dir_fg0 * 0.029,
        [(0.0, 0.0040), (0.20, 0.0046), (0.80, 0.0046), (1.0, 0.0040)],
        corner_bevel=0.14, cap_ends=True,
    )
    bm_add_twisted_lockwire(bm_bfr, [
        p_fwd_bm0 + dir_fg0 * 0.025 + bfr_up * 0.0045,
        p_fwd_bm0 + dir_fg0 * 0.031 + bfr_up * 0.0048,
        p_fwd_bm0 + dir_fg0 * 0.036 + bfr_up * 0.0040,
    ], strand_r=0.00024, twists_per_m=250.0, n_samples=14, segs=10)
    bm_add_segmented_tube(
        bm_bfr, p_fwd_bm0 - n1_vec * 0.0026, p_fwd_bm0 + n1_vec * 0.0026,
        [(0.0, 0.0056), (0.15, 0.0064), (0.85, 0.0064), (1.0, 0.0056)],
        segs=22, cap_ends=True,
    )
    meshes["boom_fwd_rod"] = finalize_shared_mesh("Mesh_BoomFwdRod", bm_bfr, clay)

    bm_bgc = bmesh.new()
    p_cyl_end = p_guide_0 - dir_fg0 * 0.132
    bgc_up = (n1_vec - dir_fg0 * n1_vec.dot(dir_fg0)).normalized()
    bgc_lat = bgc_up.cross(dir_fg0).normalized()

    bm_add_chamfered_cyl_x(bm_bgc, p_guide_0.y, p_guide_0.z, p_guide_0.x - 0.0023, p_guide_0.x + 0.0023, radius=0.0056, segs=22, hollow_r=0.0035)
    bm_add_segmented_tube(
        bm_bgc, p_guide_0 - dir_fg0 * 0.0045, p_cyl_end,
        [
            (0.00, 0.0022), (0.075, 0.0023), (0.115, 0.0068), (0.22, 0.0068),
            (0.25, 0.0060), (0.46, 0.0060), (0.49, 0.0068), (0.56, 0.0068),
            (0.59, 0.0060), (0.78, 0.0060), (0.81, 0.0072), (0.88, 0.0072),
            (0.92, 0.0060), (0.97, 0.0056), (1.00, 0.0044),
        ],
        segs=26, cap_ends=True,
    )
    bm_add_bevel_locked_hex(
        bm_bgc, p_guide_0 - dir_fg0 * 0.112, p_guide_0 - dir_fg0 * 0.123,
        [(0.0, 0.0074), (0.18, 0.0084), (0.82, 0.0084), (1.0, 0.0072)],
        corner_bevel=0.14, cap_ends=True,
    )
    p_bgc_lug = p_guide_0 - dir_fg0 * 0.096 + bgc_up * 0.0070 + bgc_lat * 0.0024
    bm_add_segmented_tube(
        bm_bgc, p_bgc_lug - bgc_up * 0.0008, p_bgc_lug + bgc_up * 0.0014,
        [(0.0, 0.0012), (0.5, 0.0012), (1.0, 0.0008)], segs=12, cap_ends=True,
    )
    bm_add_twisted_lockwire(bm_bgc, [
        p_guide_0 - dir_fg0 * 0.117 + bgc_up * 0.0066 - bgc_lat * 0.0042,
        p_guide_0 - dir_fg0 * 0.112 + bgc_up * 0.0086 - bgc_lat * 0.0014,
        p_guide_0 - dir_fg0 * 0.103 + bgc_up * 0.0080 + bgc_lat * 0.0012,
        p_guide_0 - dir_fg0 * 0.096 + bgc_up * 0.0078 + bgc_lat * 0.0024,
        p_guide_0 - dir_fg0 * 0.091 + bgc_up * 0.0072 + bgc_lat * 0.0030,
    ], strand_r=0.00028, twists_per_m=250.0, n_samples=20, segs=12)
    meshes["boom_guide_cyl"] = finalize_shared_mesh("Mesh_BoomGuideCyl", bm_bgc, clay)


    yz_hub_m = _dm_yz(a_ring_div, h_ring_m)
    hub_m0 = Vector((0.0, yz_hub_m[0], yz_hub_m[1]))
    rot_hs_r = Matrix.Rotation(+math.pi / spec.n_sectors, 4, 'Y')
    rot_hs_l = Matrix.Rotation(-math.pi / spec.n_sectors, 4, 'Y')
    hub_sr0 = rot_hs_r @ Vector((0.0, yz_rear_sad[0], yz_rear_sad[1]))
    hub_sl0 = rot_hs_l @ Vector((0.0, yz_rear_sad[0], yz_rear_sad[1]))

    def _strap_basis_mat(h_start: Vector, h_target: Vector, up_vec: Vector) -> Matrix:
        ey = (h_target - h_start).normalized()
        ez = (up_vec - ey * up_vec.dot(ey)).normalized()
        ex_s = ey.cross(ez).normalized()
        m4 = Matrix.Identity(4)
        m4[0][0], m4[1][0], m4[2][0] = ex_s.x, ex_s.y, ex_s.z
        m4[0][1], m4[1][1], m4[2][1] = ey.x,   ey.y,   ey.z
        m4[0][2], m4[1][2], m4[2][2] = ez.x,   ez.y,   ez.z
        m4[0][3], m4[1][3], m4[2][3] = h_start.x, h_start.y, h_start.z
        return m4

    bm_str_r = bmesh.new()
    bm_add_slotted_ring_strap(bm_str_r, y_start=0.0135, y_end=0.1055, half_w=0.0086, z_center=+0.0014, half_th=0.00085)
    bmesh.ops.transform(bm_str_r, matrix=_strap_basis_mat(hub_m0, hub_sr0, n2_vec), verts=bm_str_r.verts)
    meshes["ring_strap_r"] = finalize_shared_mesh("Mesh_RingStrap_R", bm_str_r, clay)

    bm_str_l = bmesh.new()
    bm_add_slotted_ring_strap(bm_str_l, y_start=0.0135, y_end=0.1055, half_w=0.0086, z_center=-0.0014, half_th=0.00085)
    bmesh.ops.transform(bm_str_l, matrix=_strap_basis_mat(hub_m0, hub_sl0, n2_vec), verts=bm_str_l.verts)
    meshes["ring_strap_l"] = finalize_shared_mesh("Mesh_RingStrap_L", bm_str_l, clay)

    meshes["_meta_boom_cfg"] = {
        "p_piv": p_piv_bm0, "t1_vec": t1_vec, "n1_vec": n1_vec,
        "p_aft": p_aft_bm0, "p_mid": p_mid_bm0, "p_fwd": p_fwd_bm0,
        "p_sqA_seal_local": p_sqA_seal_loc,
        "p_sqB_seal_local": p_sqB_seal_loc,
        "p_guide": p_guide_0,
        "hub_m0": hub_m0, "yz_rear_sad": yz_rear_sad, "n2_vec": n2_vec,
    }


    bm_bal = bmesh.new()
    p_bal0 = (0.008, 0.562)
    z_ax_bal0 = (0.545 + 0.0840 * math.tan(t1_0) + 0.00616 / math.cos(t1_0)) / 0.98078528 - 0.0063652
    ang_bal0 = math.asin(max(-0.98, min(0.98, (z_ax_bal0 - 0.562) / 0.0896505))) + 0.0685360
    L_bal = 0.104
    p_bal1 = (p_bal0[0] + L_bal * math.cos(ang_bal0), p_bal0[1] + L_bal * math.sin(ang_bal0))
    cb_t, sb_t = math.cos(ang_bal0), math.sin(ang_bal0)
    nb_y, nb_z = -sb_t, cb_t
    nb_vec = Vector((0.0, nb_y, nb_z))

    nb_long, nb_lat = 56, 48
    hw_bal = 0.0455
    th_bal = 0.0034
    base_bs = subdivide_1d_with_support_loops(nb_long, sup1=0.006, sup2=0.018)
    bal_extra_s = []
    for bk in range(6):
        sc_bk = 0.12 + (bk / 5.0) * 0.74
        for df_b in (-0.26, -0.18, +0.18, +0.26):
            sv_b = sc_bk + df_b * (0.74 / 5.0)
            if 0.12 < sv_b < 0.86:
                bal_extra_s.append(sv_b)
    merged_bs = sorted(base_bs + bal_extra_s)
    t_bal_s = [merged_bs[0]]
    for sv_b in merged_bs[1:]:
        if sv_b - t_bal_s[-1] >= 0.0012 or sv_b > 0.9999:
            t_bal_s.append(sv_b)
    t_bal_s[-1] = 1.0
    t_bal_u = subdivide_1d_with_support_loops(nb_lat,  sup1=0.006, sup2=0.018)
    g_btop, g_bbot = [], []
    for sb in t_bal_s:
        cy_b = p_bal0[0] + (p_bal1[0] - p_bal0[0]) * sb
        cz_b = p_bal0[1] + (p_bal1[1] - p_bal0[1]) * sb
        env_b = math.sin(math.pi * min(1.0, max(0.0, (sb - 0.05) / 0.90))) ** 0.55
        curl_b = 0.0024 * (((sb - 0.90) / 0.10) ** 1.6) if sb > 0.90 else 0.0
        rt_b, rb_b = [], []
        for ub in t_bal_u:
            et_b = -1.0 + 2.0 * ub
            xb = et_b * hw_bal
            ad_b = -(xb * xb) / (2.0 * 0.545)
            d3_b = min(abs(xb), abs(abs(xb) - 0.025))
            bump_b = 0.0
            if d3_b < 0.0050:
                bump_b = 0.0032 * max(0.0, 1.0 - (d3_b / 0.0050) ** 1.6) * env_b
            elif 0.12 < sb < 0.86 and abs(et_b) < 0.86:
                rph_b = (sb - 0.12) / 0.74 * 5.0
                d_rb_b = abs(rph_b - round(rph_b))
                bump_b = 0.0020 * min(1.0, max(0.0, (0.26 - d_rb_b) / 0.08)) * env_b
            elif abs(et_b) >= 0.88:
                bump_b = 0.0012 * min(1.0, (abs(et_b) - 0.88) / 0.04) * env_b
            nt_b = ad_b + th_bal + bump_b + curl_b
            nb_b = ad_b + 0.5 * curl_b
            rt_b.append(bm_bal.verts.new((xb, cy_b + nb_y * nt_b, cz_b + nb_z * nt_b)))
            rb_b.append(bm_bal.verts.new((xb, cy_b + nb_y * nb_b, cz_b + nb_z * nb_b)))
        g_btop.append(rt_b)
        g_bbot.append(rb_b)

    bm_build_beveled_quad_shell(bm_bal, g_bbot, g_btop, bulge_scale=0.18)


    for x_bkn in (-0.0405, -0.0145, +0.0145, +0.0405):
        bm_add_chamfered_cyl_x(bm_bal, p_bal0[0], p_bal0[1], x_bkn - 0.0035, x_bkn + 0.0035, radius=0.0056, segs=20, hollow_r=0.0034)
    bm_add_clevis_pin_assembly_x(bm_bal, p_bal0[0], p_bal0[1], -0.0445, +0.0445, pin_r=0.0031, segs=18, compact_ends=True)
    for sb_brv in (0.18, 0.36, 0.54, 0.72, 0.88):
        cy_brv = p_bal0[0] + (p_bal1[0] - p_bal0[0]) * sb_brv
        cz_brv = p_bal0[1] + (p_bal1[1] - p_bal0[1]) * sb_brv
        for sx_brv in (-0.0412, +0.0412):
            ad_brv = -(sx_brv * sx_brv) / (2.0 * 0.545) + th_bal + 0.0012
            bm_add_dome_rivet(bm_bal, Vector((sx_brv, cy_brv + nb_y * ad_brv, cz_brv + nb_z * ad_brv)), nb_vec, radius=0.00105)

    cy_bpn = p_bal0[0] + (p_bal1[0] - p_bal0[0]) * 0.56
    cz_bpn = p_bal0[1] + (p_bal1[1] - p_bal0[1]) * 0.56
    tb_vec = Vector((0.0, cb_t, sb_t))
    bal_plq_m = Matrix.Identity(4)
    bal_plq_m[0][0], bal_plq_m[1][0], bal_plq_m[2][0] = 1.0, 0.0, 0.0
    bal_plq_m[0][1], bal_plq_m[1][1], bal_plq_m[2][1] = tb_vec.x, tb_vec.y, tb_vec.z
    bal_plq_m[0][2], bal_plq_m[1][2], bal_plq_m[2][2] = nb_vec.x, nb_vec.y, nb_vec.z
    bal_plq_m[0][3] = 0.0
    bal_plq_m[1][3] = cy_bpn + nb_y * (th_bal + 0.0030)
    bal_plq_m[2][3] = cz_bpn + nb_z * (th_bal + 0.0030)
    b_bpn = bmesh.new()
    f0_bpn = len(b_bpn.faces)
    _bm_add_box_x_yz(b_bpn, -0.0046, +0.0046, -0.0195, +0.0195, 0.0, 0.00075, nx=2, ny=3, nz=1, chamfer=0.00018)
    bm_fill_unset_from(b_bpn, f0_bpn, MAT_PLATE_BASE)
    for sy_bpr in (-0.0168, +0.0168):
        bm_add_dome_rivet(b_bpn, Vector((0.0, sy_bpr, 0.00075)), Vector((0, 0, 1)), radius=0.00065)
    bm_merge(bm_bal, b_bpn, bal_plq_m)
    for sx_sp in (-1.0, +1.0):
        x_sp0 = sx_sp * 0.0200
        x_sp1 = sx_sp * 0.0248
        n_coil_pts = 42
        pts_coil = []
        r_coil = 0.0046
        ang_top = math.radians(295.0)
        for ic in range(n_coil_pts + 1):
            fc = ic / n_coil_pts
            x_c = x_sp0 + (x_sp1 - x_sp0) * fc
            th_c = ang_top - (4.65 * 2.0 * math.pi) * fc
            pts_coil.append(Vector((x_c, p_bal0[0] + r_coil * math.cos(th_c), p_bal0[1] + r_coil * math.sin(th_c))))
        for sb_leg in (0.07, 0.16, 0.25, 0.32):
            ad_sp = -(x_sp1 * x_sp1) / (2.0 * 0.545)
            cy_s = p_bal0[0] + (p_bal1[0] - p_bal0[0]) * sb_leg
            cz_s = p_bal0[1] + (p_bal1[1] - p_bal0[1]) * sb_leg
            h_sp = ad_sp + th_bal + 0.0022
            pts_coil.append(Vector((x_sp1, cy_s + nb_y * h_sp, cz_s + nb_z * h_sp)))
        bm_add_spline_tube(bm_bal, pts_coil, radius=0.00046, n_samples=76, segs=12, cap_ends=True)
        cy_cl = p_bal0[0] + (p_bal1[0] - p_bal0[0]) * 0.27
        cz_cl = p_bal0[1] + (p_bal1[1] - p_bal0[1]) * 0.27
        ad_cl = -(x_sp1 * x_sp1) / (2.0 * 0.545)
        p_cl0 = (cy_cl - cb_t * 0.0045 + nb_y * (ad_cl + th_bal + 0.0018), cz_cl - sb_t * 0.0045 + nb_z * (ad_cl + th_bal + 0.0018))
        p_cl1 = (cy_cl + cb_t * 0.0045 + nb_y * (ad_cl + th_bal + 0.0018), cz_cl + sb_t * 0.0045 + nb_z * (ad_cl + th_bal + 0.0018))
        bm_add_clevis_plate(bm_bal, p_cl0, p_cl1, 0.0016, 0.0016, x_sp1 - 0.0026, x_sp1 + 0.0026, arc_segs=9)
        bm_add_dome_rivet(
            bm_bal,
            Vector((x_sp1 - sx_sp * 0.0038, cy_cl + nb_y * (ad_cl + th_bal + 0.0018), cz_cl + nb_z * (ad_cl + th_bal + 0.0018))),
            nb_vec, radius=0.0014,
        )
    cy_rl = p_bal0[0] + (p_bal1[0] - p_bal0[0]) * 0.86
    cz_rl = p_bal0[1] + (p_bal1[1] - p_bal0[1]) * 0.86
    p_rl_ctr = (0.0, 0.0)
    for sx_rl in (-0.032, +0.032):
        ad_rl = -(sx_rl * sx_rl) / (2.0 * 0.545)
        p_rl_base = (cy_rl + nb_y * ad_rl, cz_rl + nb_z * ad_rl)
        p_rl_ctr  = (cy_rl + nb_y * (ad_rl - 0.0052), cz_rl + nb_z * (ad_rl - 0.0052))
        for ch_off in (-0.0046, +0.0046):
            bm_add_clevis_plate(bm_bal, p_rl_base, p_rl_ctr, 0.0042, 0.0045, sx_rl + ch_off - 0.0014, sx_rl + ch_off + 0.0014, arc_segs=12)
        for ax_s0, ax_s1 in ((sx_rl - 0.0066, sx_rl - 0.0033), (sx_rl + 0.0033, sx_rl + 0.0066)):
            bm_add_chamfered_cyl_x(bm_bal, p_rl_ctr[0], p_rl_ctr[1], ax_s0, ax_s1, radius=0.0018, segs=14)
        bm_add_dome_rivet(bm_bal, Vector((sx_rl, cy_rl + nb_y * (ad_rl + th_bal + 0.002), cz_rl + nb_z * (ad_rl + th_bal + 0.002))), nb_vec, radius=0.0018)

    meshes["balance_flap"] = finalize_shared_mesh("Mesh_BalanceFlap", bm_bal, clay)

    bm_brp = bmesh.new()
    for sx_rl in (-0.032, +0.032):
        bm_add_spoked_roller_x(bm_brp, 0.0, 0.0, sx_rl - 0.0026, sx_rl + 0.0026, radius=0.0042, axle_r=0.0015, n_spokes=5)
    meshes["balance_roller_pair"] = finalize_shared_mesh("Mesh_BalanceRollerPair", bm_brp, clay)
    meshes["_meta_bal_rl_yz"] = p_rl_ctr
    meshes["_meta_ang_bal0"] = ang_bal0

    bm_syr = bmesh.new()
    bm_add_spoked_roller_x(bm_syr, 0.0, 0.0, -0.0048, +0.0048, radius=0.0060, axle_r=0.0020, n_spokes=5)
    meshes["sync_roller"] = finalize_shared_mesh("Mesh_SyncRoller", bm_syr, clay)

    return meshes


def build_sync_ring_and_pistons_mesh(spec: NozzleSpec, mats: dict):

    clay = mats["clay"]
    y0 = spec.y_ring_closed
    rr = spec.r_sync_ring
    fh = spec.fork_half_span

    bm = bmesh.new()

    for y_rail in (y0 - 0.015, y0 + 0.015):
        prof_r = [
            (y_rail - 0.0065, rr - 0.0095),
            (y_rail + 0.0065, rr - 0.0095),
            (y_rail + 0.0065, rr - 0.0060),
            (y_rail + 0.0032, rr - 0.0060),
            (y_rail + 0.0032, rr + 0.0065),
            (y_rail + 0.0065, rr + 0.0065),
            (y_rail + 0.0065, rr + 0.0105),
            (y_rail - 0.0065, rr + 0.0105),
            (y_rail - 0.0065, rr + 0.0065),
            (y_rail - 0.0032, rr + 0.0065),
            (y_rail - 0.0032, rr - 0.0060),
            (y_rail - 0.0065, rr - 0.0060),
        ]
        bm_add_revolved_profile(bm, prof_r, segs=256, close_loop=True, bevel_r=0.00042)


    b_rib = bmesh.new()
    bm_add_clevis_plate(b_rib, (y0 - 0.014, rr + 0.0058), (y0 + 0.014, rr + 0.0058), 0.0028, 0.0028, -0.0022, +0.0022, arc_segs=9)
    bm_add_clevis_plate(b_rib, (y0 - 0.014, rr - 0.0052), (y0 + 0.014, rr - 0.0052), 0.0028, 0.0028, -0.0022, +0.0022, arc_segs=9)
    bm_add_clevis_plate(b_rib, (y0 - 0.013, rr), (y0 + 0.013, rr), 0.0068, 0.0068, -0.0010, +0.0010, arc_segs=9)
    bm_add_chamfered_cyl_x(b_rib, y0, rr + 0.0003, -0.0024, +0.0024, radius=0.0060, segs=22, chamfer=0.00035, hollow_r=0.0036)
    for k in range(64):
        bm_merge(bm, b_rib.copy(), Matrix.Rotation(2.0 * math.pi * (k + 0.5) / 64.0, 4, 'Y'))
    b_rib.free()


    for i in range(spec.n_sectors):
        phi = 2.0 * math.pi * i / spec.n_sectors
        bt = bmesh.new()
        for side_s in (-1.0, 1.0):
            xc = side_s * fh
            for sub in (-0.0082, +0.0082):
                bm_add_clevis_plate(
                    bt, (y0 - 0.013, rr), (y0 + 0.002, rr),
                    0.011, 0.011, xc + sub - 0.0022, xc + sub + 0.0022, arc_segs=18,
                )
            bm_add_clevis_pin_assembly_x(bt, y0, rr, xc - 0.011, xc + 0.011, pin_r=0.0046, segs=22)


        r_bogie_rl = 0.5572 + 0.0060
        for sx_bg in (-0.0070, +0.0070):
            bm_add_clevis_plate(bt, (y0 - 0.013, rr - 0.0075), (y0 + 0.013, rr - 0.0075), 0.0045, 0.0045, sx_bg - 0.0014, sx_bg + 0.0014, arc_segs=11)
            for y_bg_r in (y0 - 0.011, y0 + 0.011):
                bm_add_clevis_plate(bt, (y_bg_r, rr - 0.0075), (y_bg_r, r_bogie_rl), 0.0046, 0.0048, sx_bg - 0.0014, sx_bg + 0.0014, arc_segs=13)
        for y_bg_r in (y0 - 0.011, y0 + 0.011):
            for ax_b0, ax_b1 in ((-0.0094, -0.0055), (+0.0055, +0.0094)):
                bm_add_chamfered_cyl_x(bt, y_bg_r, r_bogie_rl, ax_b0, ax_b1, radius=0.0024, segs=16)
            bm_add_clevis_pin_assembly_x(bt, y_bg_r, r_bogie_rl, -0.0096, +0.0096, pin_r=0.0020, segs=16, compact_ends=True)
        bm_add_grease_zerk(bt, Vector((+0.0086, y0, rr - 0.0068)), Vector((+1.0, 0.0, 0.0)), scale=0.72)

        if i % 2 == 0:

            p_rod_a = Vector((0.0, -0.815, rr))
            p_rod_b = Vector((0.0, y0 - 0.017, rr))
            bm_add_segmented_tube(
                bt, p_rod_a, p_rod_b,
                [
                    (0.000, 0.0122), (0.964, 0.0122), (0.966, 0.0114), (0.982, 0.0114),
                    (0.983, 0.0142), (0.992, 0.0142),
                    (0.995, 0.0178), (0.999, 0.0178), (1.000, 0.0168),
                ],
                segs=36, cap_ends=True,
            )
            bm_add_helical_thread_cylinder(
                bt, p_rod_a.lerp(p_rod_b, 0.965), p_rod_a.lerp(p_rod_b, 0.983),
                r_core=0.0114, thread_h=0.0022, n_turns=4, segs_turn=20, cap_core=False,
            )
            bm_add_bevel_locked_hex(
                bt, p_rod_a.lerp(p_rod_b, 0.983), p_rod_a.lerp(p_rod_b, 0.991),
                [(0.0, 0.0156), (0.14, 0.0174), (0.86, 0.0174), (1.0, 0.0156)],
                corner_bevel=0.14, cap_ends=True,
            )
            y_jam_p = -0.815 + (y0 - 0.017 + 0.815) * 0.987
            bm_add_segmented_tube(
                bt, Vector((0.0142, y0 - 0.016, rr + 0.0095)), Vector((0.0176, y0 - 0.016, rr + 0.0132)),
                [(0.0, 0.0016), (0.5, 0.0016), (1.0, 0.0011)], segs=14, cap_ends=True,
            )
            bm_add_twisted_lockwire(bt, [
                Vector((0.0142, y_jam_p,         rr + 0.0076)),
                Vector((0.0176, y_jam_p + 0.003, rr + 0.0108)),
                Vector((0.0182, y0 - 0.019,      rr + 0.0128)),
                Vector((0.0172, y0 - 0.016,      rr + 0.0132)),
                Vector((0.0182, y0 - 0.013,      rr + 0.0140)),
            ], strand_r=0.00040, twists_per_m=240.0, n_samples=22, segs=12)

            for side_yoke in (-0.011, +0.011):
                bm_add_clevis_plate(
                    bt, (y0 - 0.019, rr), (y0 - 0.008, rr),
                    0.0112, 0.0122, side_yoke - 0.0030, side_yoke + 0.0030, arc_segs=18,
                )
            for sx_gus in (-0.0165, +0.0165):
                bm_add_clevis_plate(bt, (y0 - 0.015, rr + 0.002), (y0 + 0.015, rr + 0.002), 0.0088, 0.0072, sx_gus - 0.0022, sx_gus + 0.0022, arc_segs=13)
            bm_add_clevis_pin_assembly_x(bt, y0 - 0.008, rr, -0.015, +0.015, pin_r=0.0055, segs=24)

        bm_merge(bm, bt, Matrix.Rotation(phi, 4, 'Y'))


    bt_rvdt = bmesh.new()
    p_rv_a, p_rv_b = Vector((0.0, -0.805, rr)), Vector((0.0, y0 - 0.010, rr))
    bm_add_segmented_tube(
        bt_rvdt, p_rv_a, p_rv_b,
        [
            (0.00, 0.0052), (0.952, 0.0052), (0.955, 0.0046), (0.980, 0.0046),
            (0.982, 0.0092), (0.992, 0.0092), (1.00, 0.0105),
        ],
        segs=24, cap_ends=True,
    )
    bm_add_helical_thread_cylinder(
        bt_rvdt, p_rv_a.lerp(p_rv_b, 0.954), p_rv_a.lerp(p_rv_b, 0.981),
        r_core=0.0046, thread_h=0.0012, n_turns=5, segs_turn=16, cap_core=False,
    )
    for sx_rv in (-0.0068, +0.0068):
        bm_add_clevis_plate(bt_rvdt, (y0 - 0.018, rr), (y0 + 0.004, rr), 0.010, 0.010, sx_rv - 0.0022, sx_rv + 0.0022, arc_segs=16)

    bm_add_clevis_plate(bt_rvdt, (y0 - 0.014, rr + 0.010), (y0 + 0.014, rr + 0.014), 0.0038, 0.0028, -0.0025, +0.0025, arc_segs=11)
    bm_add_clevis_pin_assembly_x(bt_rvdt, y0 - 0.008, rr, -0.0105, +0.0105, pin_r=0.0040, segs=20)
    bm_merge(bm, bt_rvdt, Matrix.Rotation(math.radians(56.25), 4, 'Y'))

    return finalize_shared_mesh("Mesh_SyncRing_Pistons", bm, clay)


def build_armature_and_bind_parts(spec: NozzleSpec, mats: dict, coll_moving, coll_rig):
    st0 = spec.state_closed
    shared_meshes = build_shared_component_meshes(spec, mats)
    sync_mesh = build_sync_ring_and_pistons_mesh(spec, mats)
    lpt_rotor_mesh = build_lpt_rotor_mesh(mats)
    fan_rotor_mesh = build_fan_rotor_3stage_mesh(mats)

    arm_data = bpy.data.armatures.new("F15_Nozzle_Armature")
    arm_data.display_type = 'STICK'
    arm_obj = bpy.data.objects.new("F15_Nozzle_Rig", arm_data)
    coll_rig.objects.link(arm_obj)
    arm_obj.show_in_front = False

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_data.edit_bones

    def _v_sec(yr: tuple, rot_m: Matrix) -> Vector:
        return rot_m @ Vector((0.0, yr[0], yr[1]))

    def _make_bone(name: str, head: Vector, tail: Vector, roll_z: Vector, parent=None, connect=False):
        b = eb.new(name)
        b.head = head
        b.tail = tail
        b.align_roll(roll_z)
        if parent is not None:
            b.parent = parent
            b.use_connect = connect
        return b

    b_root = _make_bone("B_Root", Vector((0, -0.50, 0)), Vector((0, 0.0, 0)), Vector((0, 0, 1)))
    _make_bone("B_LPT_Rotor", Vector((0.0, -1.40, 0.0)), Vector((0.0, -0.95, 0.0)), Vector((0, 0, 1)), parent=b_root)
    _make_bone("B_SliderParam", Vector((0.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0, 0, 1)), parent=b_root)
    b_sync = _make_bone(
        "B_SyncRing",
        Vector((0.0, spec.y_ring_closed, 0.0)),
        Vector((0.0, spec.y_ring_closed + 0.10, 0.0)),
        Vector((0, 0, 1)),
        parent=b_root,
    )

    t1_0 = st0["theta1"]
    t1_n = (-math.sin(t1_0), math.cos(t1_0))
    dr_s = spec.seal_radial_offset
    p0_seal = (st0["P0"][0] - t1_n[0] * dr_s, st0["P0"][1] - t1_n[1] * dr_s)
    p1_seal = (st0["P1"][0] - t1_n[0] * dr_s, st0["P1"][1] - t1_n[1] * dr_s)
    t2_0 = st0["theta2"]
    t2_n = (-math.sin(t2_0), math.cos(t2_0))
    p2_seal = (st0["P2"][0] - t2_n[0] * dr_s, st0["P2"][1] - t2_n[1] * dr_s)

    ang_bal0 = shared_meshes["_meta_ang_bal0"]
    bal_rl_yz = shared_meshes["_meta_bal_rl_yz"]
    sl_rl_f_yz = shared_meshes["_meta_slider_rl_f_yz"]
    sl_rl_b_yz = shared_meshes["_meta_slider_rl_b_yz"]
    boom_cfg = shared_meshes["_meta_boom_cfg"]
    r_bogie_rl = 0.5572 + 0.0060
    y_sync_f = spec.y_ring_closed - 0.011
    y_sync_b = spec.y_ring_closed + 0.011

    for i in range(spec.n_sectors):
        phi_m = 2.0 * math.pi * i / spec.n_sectors
        rot_m = Matrix.Rotation(phi_m, 4, 'Y')
        tang_m = rot_m @ Vector((1.0, 0.0, 0.0))

        Pring = _v_sec(st0["P_ring"], rot_m)
        PB    = _v_sec(st0["PB"], rot_m)
        Ain   = _v_sec(st0["A_in"], rot_m)
        Aout  = _v_sec(st0["A_out"], rot_m)
        P0    = _v_sec(st0["P0"], rot_m)
        PC    = _v_sec(st0["PC"], rot_m)
        P1    = _v_sec(st0["P1"], rot_m)
        PD    = _v_sec(st0["PD"], rot_m)
        P2    = _v_sec(st0["P2"], rot_m)
        PS    = _v_sec(st0["PS"], rot_m)

        _make_bone(f"B_SyncLug_{i:02d}", Pring, Pring + Vector((0, 0.04, 0)), tang_m, b_sync, False)
        S_RLF = _v_sec((y_sync_f, r_bogie_rl), rot_m)
        S_RLB = _v_sec((y_sync_b, r_bogie_rl), rot_m)
        _make_bone(f"B_SyncRollerF_{i:02d}", S_RLF, S_RLF + Vector((0, 0.025, 0)), tang_m, b_sync, False)
        _make_bone(f"B_SyncRollerB_{i:02d}", S_RLB, S_RLB + Vector((0, 0.025, 0)), tang_m, b_sync, False)
        _make_bone(f"B_BracketPS_{i:02d}", PS, PS + Vector((0, 0.04, 0)), tang_m, b_root, False)

        b_bin = _make_bone(f"B_BellIn_{i:02d}", PB, Ain, tang_m, b_root, False)
        _make_bone(f"B_ShortLink_{i:02d}", Ain, Pring, tang_m, b_bin, True)
        _make_bone(f"B_BellOutTgt_{i:02d}", Aout, Aout + Vector((0, 0.04, 0)), tang_m, b_bin, False)

        b_clug = _make_bone(f"B_ConvLug_{i:02d}", P0, PC, tang_m, b_root, False)
        _make_bone(f"B_PullRod_{i:02d}", PC, Aout, tang_m, b_clug, True)
        b_cflap = _make_bone(f"B_ConvFlap_{i:02d}", P0, P1, tang_m, b_clug, False)
        if i == 0:
            _make_bone("B_P1_Probe", P1, P1 + Vector((0, 0.025, 0)), tang_m, b_cflap, True)


        b_dflap = _make_bone(f"B_DivFlap_{i:02d}", P1, P2, tang_m, b_root, False)
        dir_trk = (P2 - P1).normalized()
        b_scar = _make_bone(f"B_SliderCarriage_{i:02d}", PD - dir_trk * 0.050, PD, tang_m, b_dflap, False)
        SL_F = _v_sec(sl_rl_f_yz, rot_m)
        SL_B = _v_sec(sl_rl_b_yz, rot_m)
        _make_bone(f"B_SliderRollerF_{i:02d}", SL_F, SL_F + dir_trk * 0.025, tang_m, b_scar, False)
        _make_bone(f"B_SliderRollerB_{i:02d}", SL_B, SL_B + dir_trk * 0.025, tang_m, b_scar, False)
        _make_bone(f"B_DiagStrut_{i:02d}", PD, PS, tang_m, b_scar, True)

        phi_s = phi_m + (math.pi / spec.n_sectors)
        rot_s = Matrix.Rotation(phi_s, 4, 'Y')
        rot_s_prev = Matrix.Rotation(phi_m - (math.pi / spec.n_sectors), 4, 'Y')
        tang_s = rot_s @ Vector((1.0, 0.0, 0.0))
        P0_s = _v_sec(p0_seal, rot_s)
        P1_s = _v_sec(p1_seal, rot_s)
        P2_s = _v_sec(p2_seal, rot_s)
        b_cseal = _make_bone(f"B_ConvSeal_{i:02d}", P0_s, P1_s, tang_s, b_root, False)
        b_dseal = _make_bone(f"B_DivSeal_{i:02d}", P1_s, P2_s, tang_s, b_root, False)


        n1_m = rot_m @ boom_cfg["n1_vec"]
        t1_m = rot_m @ boom_cfg["t1_vec"]
        n1_s = rot_s @ boom_cfg["n1_vec"]
        t1_s = rot_s @ boom_cfg["t1_vec"]
        P_bm    = rot_m @ boom_cfg["p_piv"]
        P_aft   = rot_m @ boom_cfg["p_aft"]
        P_mid   = rot_m @ boom_cfg["p_mid"]
        P_fwd   = rot_m @ boom_cfg["p_fwd"]
        P_sqA_s = rot_s @ boom_cfg["p_sqA_seal_local"]
        P_sqB_s = rot_s @ boom_cfg["p_sqB_seal_local"]
        P_sqB_p = rot_s_prev @ boom_cfg["p_sqB_seal_local"]
        P_gde   = rot_m @ boom_cfg["p_guide"]

        _make_bone(f"B_SqBlkA_{i:02d}",    P_sqA_s, P_sqA_s + t1_s * 0.020, n1_s, b_cseal, False)
        _make_bone(f"B_SqBlkB_{i:02d}",    P_sqB_s, P_sqB_s + t1_s * 0.020, n1_s, b_cseal, False)
        _make_bone(f"B_ConvUpRef_{i:02d}", P_bm + n1_m * 0.080, P_bm + n1_m * 0.100, tang_m, b_cflap, False)
        b_boom = _make_bone(f"B_Boomerang_{i:02d}", P_bm, P_bm + t1_m * 0.040, n1_m, b_cflap, False)
        _make_bone(f"B_TransRodA_{i:02d}", P_aft, P_sqA_s, n1_m, b_boom, False)
        _make_bone(f"B_TransRodB_{i:02d}", P_mid, P_sqB_p, n1_m, b_boom, False)
        _make_bone(f"B_BoomFwdPin_{i:02d}",   P_fwd, P_fwd + t1_m * 0.015, n1_m, b_boom, False)
        _make_bone(f"B_BoomGuidePin_{i:02d}", P_gde, P_gde + Vector((0, 0.015, 0)), tang_m, b_root, False)
        _make_bone(f"B_BoomFwdRod_{i:02d}",   P_fwd, P_gde, n1_m, b_boom, False)
        _make_bone(f"B_BoomGuideCyl_{i:02d}", P_gde, P_fwd, n1_m, b_root, False)


        n2_m = rot_m @ boom_cfg["n2_vec"]
        dir_trk_s = (P2_s - P1_s).normalized()
        HubM_i  = rot_m @ boom_cfg["hub_m0"]
        HubSR_i = rot_s @ Vector((0.0, boom_cfg["yz_rear_sad"][0], boom_cfg["yz_rear_sad"][1]))
        HubSL_i = rot_s_prev @ Vector((0.0, boom_cfg["yz_rear_sad"][0], boom_cfg["yz_rear_sad"][1]))
        _make_bone(f"B_RingHubS_{i:02d}",   HubSR_i, HubSR_i + dir_trk_s * 0.030, tang_s, b_dseal, False)
        _make_bone(f"B_RingUpRef_{i:02d}",  HubM_i + n2_m * 0.080, HubM_i + n2_m * 0.100, tang_m, b_scar, False)
        _make_bone(f"B_RingStrapR_{i:02d}", HubM_i, HubSR_i, n2_m, b_scar, False)
        _make_bone(f"B_RingStrapL_{i:02d}", HubM_i, HubSL_i, n2_m, b_scar, False)


        p_bal0_2d = (0.008, 0.562)
        p_bal1_2d = (0.008 + 0.104 * math.cos(ang_bal0), 0.562 + 0.104 * math.sin(ang_bal0))
        P_bal0_s = _v_sec(p_bal0_2d, rot_s)
        P_bal1_s = _v_sec(p_bal1_2d, rot_s)
        b_bflap = _make_bone(f"B_BalanceFlap_{i:02d}", P_bal0_s, P_bal1_s, tang_s, b_root, False)
        B_RL = _v_sec(bal_rl_yz, rot_s)
        _make_bone(f"B_BalanceRoller_{i:02d}", B_RL, B_RL + Vector((0, 0.025, 0)), tang_s, b_bflap, False)

    bpy.ops.object.mode_set(mode='POSE')
    pb = arm_obj.pose.bones

    for i in range(spec.n_sectors):
        i_next = (i + 1) % spec.n_sectors
        i_prev = (i - 1) % spec.n_sectors
        for bn in (
            f"B_BellIn_{i:02d}", f"B_ShortLink_{i:02d}",
            f"B_ConvLug_{i:02d}", f"B_PullRod_{i:02d}",
            f"B_DiagStrut_{i:02d}",
        ):
            pb[bn].lock_ik_x = True
            pb[bn].lock_ik_y = True

        for solver_bn, target_bn, ch_cnt in [
            (f"B_ShortLink_{i:02d}", f"B_SyncLug_{i:02d}", 2),
            (f"B_PullRod_{i:02d}",   f"B_BellOutTgt_{i:02d}", 2),
            (f"B_DiagStrut_{i:02d}", f"B_BracketPS_{i:02d}", 1),
        ]:
            c = pb[solver_bn].constraints.new('IK')
            c.target = arm_obj
            c.subtarget = target_bn
            c.chain_count = ch_cnt
            c.use_stretch = False
            c.iterations = 500

        if i > 0:
            c_dfr = pb[f"B_DivFlap_{i:02d}"].constraints.new('COPY_ROTATION')
            c_dfr.target = arm_obj
            c_dfr.subtarget = "B_DivFlap_00"
            c_dfr.target_space = 'LOCAL'
            c_dfr.owner_space = 'LOCAL'
        c_dfl = pb[f"B_DivFlap_{i:02d}"].constraints.new('COPY_LOCATION')
        c_dfl.target = arm_obj
        c_dfl.subtarget = f"B_ConvFlap_{i:02d}"
        c_dfl.head_tail = 1.0

        c_cs = pb[f"B_ConvSeal_{i:02d}"].constraints.new('COPY_ROTATION')
        c_cs.target = arm_obj
        c_cs.subtarget = f"B_ConvLug_{i:02d}"
        c_cs.target_space = 'LOCAL'
        c_cs.owner_space = 'LOCAL'

        c_dsr = pb[f"B_DivSeal_{i:02d}"].constraints.new('COPY_ROTATION')
        c_dsr.target = arm_obj
        c_dsr.subtarget = "B_DivFlap_00"
        c_dsr.target_space = 'LOCAL'
        c_dsr.owner_space = 'LOCAL'
        c_dsl = pb[f"B_DivSeal_{i:02d}"].constraints.new('COPY_LOCATION')
        c_dsl.target = arm_obj
        c_dsl.subtarget = f"B_ConvSeal_{i:02d}"
        c_dsl.head_tail = 1.0

        if i > 0:
            c_bf = pb[f"B_BalanceFlap_{i:02d}"].constraints.new('COPY_ROTATION')
            c_bf.target = arm_obj
            c_bf.subtarget = "B_BalanceFlap_00"
            c_bf.target_space = 'LOCAL'
            c_bf.owner_space = 'LOCAL'


        for trk_bn, tgt_bn, up_bn in [
            (f"B_TransRodA_{i:02d}",    f"B_SqBlkA_{i:02d}",            f"B_ConvUpRef_{i:02d}"),
            (f"B_TransRodB_{i:02d}",    f"B_SqBlkB_{i_prev:02d}",       f"B_ConvUpRef_{i:02d}"),
            (f"B_BoomFwdRod_{i:02d}",   f"B_BoomGuidePin_{i:02d}",      None),
            (f"B_BoomGuideCyl_{i:02d}", f"B_BoomFwdPin_{i:02d}",        None),
            (f"B_RingStrapR_{i:02d}",   f"B_RingHubS_{i:02d}",          f"B_RingUpRef_{i:02d}"),
            (f"B_RingStrapL_{i:02d}",   f"B_RingHubS_{i_prev:02d}",     f"B_RingUpRef_{i:02d}"),
        ]:
            cdt = pb[trk_bn].constraints.new('DAMPED_TRACK')
            cdt.target = arm_obj
            cdt.subtarget = tgt_bn
            cdt.track_axis = 'TRACK_Y'
            if up_bn is not None:
                clt = pb[trk_bn].constraints.new('LOCKED_TRACK')
                clt.target = arm_obj
                clt.subtarget = up_bn
                clt.track_axis = 'TRACK_Z'
                clt.lock_axis = 'LOCK_Y'

    bpy.ops.object.mode_set(mode='OBJECT')
    arm_obj.hide_viewport = False
    bpy.context.view_layer.update()

    def _bind_rigid_part(obj_name: str, mesh_data, bone_name: str, world_mat_rest: Matrix):
        obj = bpy.data.objects.new(obj_name, mesh_data)
        coll_moving.objects.link(obj)
        attach_weighted_normal_modifier(obj)
        bone = arm_data.bones[bone_name]
        pbone = arm_obj.pose.bones[bone_name]
        tail_world = arm_obj.matrix_world @ pbone.matrix @ Matrix.Translation((0.0, bone.length, 0.0))
        obj.parent = arm_obj
        obj.parent_type = 'BONE'
        obj.parent_bone = bone_name
        obj.matrix_parent_inverse = tail_world.inverted()
        obj.matrix_basis = world_mat_rest
        return obj

    _bind_rigid_part("Moving_SyncRing_And_Pistons", sync_mesh, "B_SyncRing", Matrix.Identity(4))
    _bind_rigid_part("Moving_LPT_Rotor_And_SplineShaft", lpt_rotor_mesh, "B_LPT_Rotor", Matrix.Identity(4))
    _bind_rigid_part("Moving_Fan_Rotor_3Stage", fan_rotor_mesh, "B_LPT_Rotor", Matrix.Identity(4))

    for i in range(spec.n_sectors):
        phi_m = 2.0 * math.pi * i / spec.n_sectors
        rot_m = Matrix.Rotation(phi_m, 4, 'Y')
        phi_s = phi_m + (math.pi / spec.n_sectors)
        rot_s = Matrix.Rotation(phi_s, 4, 'Y')

        _bind_rigid_part(f"Flap_ConvMaster_{i:02d}",  shared_meshes["conv_master"],     f"B_ConvFlap_{i:02d}",       rot_m)
        _bind_rigid_part(f"Flap_DivMaster_{i:02d}",   shared_meshes["div_master"],      f"B_DivFlap_{i:02d}",        rot_m)
        _bind_rigid_part(f"Slider_Carriage_{i:02d}",  shared_meshes["slider_carriage"], f"B_SliderCarriage_{i:02d}", rot_m)
        _bind_rigid_part(
            f"Slider_RollerF_{i:02d}", shared_meshes["slider_roller_pair"], f"B_SliderRollerF_{i:02d}",
            rot_m @ Matrix.Translation((0.0, sl_rl_f_yz[0], sl_rl_f_yz[1])),
        )
        _bind_rigid_part(
            f"Slider_RollerB_{i:02d}", shared_meshes["slider_roller_pair"], f"B_SliderRollerB_{i:02d}",
            rot_m @ Matrix.Translation((0.0, sl_rl_b_yz[0], sl_rl_b_yz[1])),
        )
        _bind_rigid_part(
            f"Sync_RollerF_{i:02d}", shared_meshes["sync_roller"], f"B_SyncRollerF_{i:02d}",
            rot_m @ Matrix.Translation((0.0, y_sync_f, r_bogie_rl)),
        )
        _bind_rigid_part(
            f"Sync_RollerB_{i:02d}", shared_meshes["sync_roller"], f"B_SyncRollerB_{i:02d}",
            rot_m @ Matrix.Translation((0.0, y_sync_b, r_bogie_rl)),
        )

        _bind_rigid_part(f"Flap_ConvSeal_{i:02d}",    shared_meshes["conv_seal"],    f"B_ConvSeal_{i:02d}",    rot_s)
        _bind_rigid_part(f"Flap_DivSeal_{i:02d}",     shared_meshes["div_seal"],     f"B_DivSeal_{i:02d}",     rot_s)
        _bind_rigid_part(f"Flap_Balance_{i:02d}",     shared_meshes["balance_flap"], f"B_BalanceFlap_{i:02d}", rot_s)
        _bind_rigid_part(
            f"Balance_Roller_{i:02d}", shared_meshes["balance_roller_pair"], f"B_BalanceRoller_{i:02d}",
            rot_s @ Matrix.Translation((0.0, bal_rl_yz[0], bal_rl_yz[1])),
        )

        _bind_rigid_part(f"Link_Bellcrank_{i:02d}",    shared_meshes["bellcrank"],      f"B_BellIn_{i:02d}",       rot_m)
        _bind_rigid_part(f"Link_ShortLink_{i:02d}",    shared_meshes["short_link"],     f"B_ShortLink_{i:02d}",    rot_m)
        _bind_rigid_part(f"Link_PullRod_{i:02d}",      shared_meshes["pull_rod"],       f"B_PullRod_{i:02d}",      rot_m)
        _bind_rigid_part(f"Link_DiagStrut_{i:02d}",    shared_meshes["diag_strut"],     f"B_DiagStrut_{i:02d}",    rot_m)
        _bind_rigid_part(f"Link_Boomerang_{i:02d}",    shared_meshes["boomerang"],      f"B_Boomerang_{i:02d}",    rot_m)
        _bind_rigid_part(f"Link_TransRodA_{i:02d}",    shared_meshes["trans_rod_a"],    f"B_TransRodA_{i:02d}",    rot_m)
        _bind_rigid_part(f"Link_TransRodB_{i:02d}",    shared_meshes["trans_rod_b"],    f"B_TransRodB_{i:02d}",    rot_m)
        _bind_rigid_part(f"Link_BoomFwdRod_{i:02d}",   shared_meshes["boom_fwd_rod"],   f"B_BoomFwdRod_{i:02d}",   rot_m)
        _bind_rigid_part(f"Link_BoomGuideCyl_{i:02d}", shared_meshes["boom_guide_cyl"], f"B_BoomGuideCyl_{i:02d}", rot_m)
        _bind_rigid_part(f"Ring_StrapR_{i:02d}",       shared_meshes["ring_strap_r"],       f"B_RingStrapR_{i:02d}",   rot_m)
        _bind_rigid_part(f"Ring_StrapL_{i:02d}",       shared_meshes["ring_strap_l"],       f"B_RingStrapL_{i:02d}",   rot_m)
        _bind_rigid_part(f"Ring_KeeperSaddle_{i:02d}", shared_meshes["ring_keeper_saddle"], f"B_RingHubS_{i:02d}",     rot_s)

    scn = bpy.context.scene

    def _add_scene_var(drv, var_name: str, prop_path: str):
        v = drv.variables.new()
        v.name = var_name
        v.type = 'SINGLE_PROP'
        v.targets[0].id_type = 'SCENE'
        v.targets[0].id = scn
        v.targets[0].data_path = prop_path
        return v

    def _add_bone_var(drv, var_name: str, bone_name: str, tf_type: str, space: str = 'WORLD_SPACE'):
        v = drv.variables.new()
        v.name = var_name
        v.type = 'TRANSFORMS'
        v.targets[0].id = arm_obj
        v.targets[0].bone_target = bone_name
        v.targets[0].transform_type = tf_type
        v.targets[0].transform_space = space
        return v


    pb_sync = pb["B_SyncRing"]
    pb_sync.rotation_mode = 'XYZ'
    drv_sync = pb_sync.driver_add("location", 1).driver
    drv_sync.type = 'SCRIPTED'
    _add_scene_var(drv_sync, "open_val", "f15_nozzle_open")
    drv_sync.expression = f"clamp(open_val, 0.0, 1.0) * {spec.ring_stroke:.6f}"

    for i in range(spec.n_sectors):
        for r_bn in (f"B_SyncRollerF_{i:02d}", f"B_SyncRollerB_{i:02d}"):
            pb[r_bn].rotation_mode = 'XYZ'
            d_sr = pb[r_bn].driver_add("rotation_euler", 2).driver
            d_sr.type = 'SCRIPTED'
            _add_scene_var(d_sr, "open_val", "f15_nozzle_open")
            d_sr.expression = f"-clamp(open_val, 0.0, 1.0) * {spec.ring_stroke / 0.0060:.6f}"


    da_sub = (
        f"(clamp(u_val,0.0,1.0)*{spec.slider_auto_stroke:.6f}"
        f"+(clamp(m_val,0.0,1.0)+clamp(m_val,-1.0,0.0)*clamp(u_val/0.28,0.0,1.0))*{spec.slider_mode_stroke:.6f})"
    )
    pb_sp = pb["B_SliderParam"]
    pb_sp.rotation_mode = 'XYZ'
    drv_sp = pb_sp.driver_add("location", 1).driver
    drv_sp.type = 'SCRIPTED'
    _add_scene_var(drv_sp, "u_val", "f15_nozzle_open")
    _add_scene_var(drv_sp, "m_val", "f15_slider_mode")
    drv_sp.expression = f"{spec.div_lug_a:.6f}+{da_sub}"
    assert len(drv_sp.expression) < 240

    h_val = spec.div_lug_h
    h2_val = h_val * h_val
    ls2_val = spec.L_strut * spec.L_strut

    pb_df0 = pb["B_DivFlap_00"]
    pb_df0.rotation_mode = 'XYZ'
    drv_df0 = pb_df0.driver_add("rotation_euler", 2).driver
    drv_df0.type = 'SCRIPTED'
    _add_bone_var(drv_df0, "py", "B_P1_Probe", 'LOC_Y', 'WORLD_SPACE')
    _add_bone_var(drv_df0, "pz", "B_P1_Probe", 'LOC_Z', 'WORLD_SPACE')
    _add_bone_var(drv_df0, "ac", "B_SliderParam", 'LOC_Y', 'LOCAL_SPACE')
    v_dist = drv_df0.variables.new()
    v_dist.name = "d"
    v_dist.type = 'LOC_DIFF'
    v_dist.targets[0].id = arm_obj
    v_dist.targets[0].bone_target = "B_P1_Probe"
    v_dist.targets[0].transform_space = 'WORLD_SPACE'
    v_dist.targets[1].id = arm_obj
    v_dist.targets[1].bone_target = "B_BracketPS_00"
    v_dist.targets[1].transform_space = 'WORLD_SPACE'
    drv_df0.expression = (
        f"atan2({spec.ps_r:.6f}-pz,{spec.ps_y:.6f}-py)"
        f"-acos(clamp((d*d+ac*ac+{h2_val:.8f}-{ls2_val:.8f})/(2.0*d*sqrt(ac*ac+{h2_val:.8f})),-1.0,1.0))"
        f"-atan2({h_val:.6f},ac)-({t2_0:.8f})"
    )
    assert len(drv_df0.expression) < 240, f"drv_df0 expression exceeds limit: {len(drv_df0.expression)}"


    for i in range(spec.n_sectors):
        for sl_bn in (f"B_SliderCarriage_{i:02d}", f"B_RingHubS_{i:02d}"):
            pb_sc = pb[sl_bn]
            pb_sc.rotation_mode = 'XYZ'
            d_sc = pb_sc.driver_add("location", 1).driver
            d_sc.type = 'SCRIPTED'
            _add_scene_var(d_sc, "u_val", "f15_nozzle_open")
            _add_scene_var(d_sc, "m_val", "f15_slider_mode")
            d_sc.expression = da_sub

        for sr_bn in (f"B_SliderRollerF_{i:02d}", f"B_SliderRollerB_{i:02d}"):
            pb[sr_bn].rotation_mode = 'XYZ'
            d_slr = pb[sr_bn].driver_add("rotation_euler", 2).driver
            d_slr.type = 'SCRIPTED'
            _add_scene_var(d_slr, "u_val", "f15_nozzle_open")
            _add_scene_var(d_slr, "m_val", "f15_slider_mode")
            d_slr.expression = f"-{da_sub}/0.0046"


    pb_bf0 = pb["B_BalanceFlap_00"]
    pb_bf0.rotation_mode = 'XYZ'
    drv_bf0 = pb_bf0.driver_add("rotation_euler", 2).driver
    drv_bf0.type = 'SCRIPTED'
    _add_bone_var(drv_bf0, "dth1", "B_ConvLug_00", 'ROT_Z', 'LOCAL_SPACE')
    th1_sub = f"({t1_0:.6f}+dth1)"
    zax_sub = f"((0.545+(0.084+0.035*dth1)*tan({th1_sub})+0.00616/cos({th1_sub}))/0.9807853-0.0063652)"
    drv_bf0.expression = f"asin(clamp(({zax_sub}-0.562)/0.0896505,-0.98,0.98))+({0.0685360 - ang_bal0:.7f})"
    assert len(drv_bf0.expression) < 240, f"drv_bf0 expression exceeds limit: {len(drv_bf0.expression)}"

    for i in range(spec.n_sectors):
        br_bn = f"B_BalanceRoller_{i:02d}"
        pb[br_bn].rotation_mode = 'XYZ'
        d_br = pb[br_bn].driver_add("rotation_euler", 2).driver
        d_br.type = 'SCRIPTED'
        _add_bone_var(d_br, "dth1", "B_ConvLug_00", 'ROT_Z', 'LOCAL_SPACE')
        d_br.expression = "-dth1 * 4.25"

        bm_bn = f"B_Boomerang_{i:02d}"
        pb[bm_bn].rotation_mode = 'XYZ'
        d_bm = pb[bm_bn].driver_add("rotation_euler", 2).driver
        d_bm.type = 'SCRIPTED'
        _add_bone_var(d_bm, "dth1", "B_ConvLug_00", 'ROT_Z', 'LOCAL_SPACE')
        d_bm.expression = "dth1*(-1.492679+dth1*(0.288075-2.506695*dth1))"
        assert len(d_bm.expression) < 240


    pb_rot = pb["B_LPT_Rotor"]
    pb_rot.rotation_mode = 'XYZ'
    drv_rot = pb_rot.driver_add("rotation_euler", 1).driver
    drv_rot.type = 'SCRIPTED'
    _add_scene_var(drv_rot, "spin_on", "f15_turbine_spin")
    _add_scene_var(drv_rot, "man_deg", "f15_rotor_manual")
    drv_rot.expression = "((frame - 1.0) * 0.31415927) * spin_on + man_deg * 0.01745329"


    arm_obj.hide_set(True)

    return arm_obj


def refresh_nozzle_tag_meshes_in_current_scene():

    existing_mesh_names = set(m.name for m in bpy.data.meshes)
    clay_mat = bpy.data.materials.get("Mat_Clay_WhiteModel")
    mats = {"clay": clay_mat} if clay_mat is not None else build_clay_materials()
    spec = NozzleSpec()
    fresh = build_shared_component_meshes(spec, mats)

    me_bal_new = fresh["balance_flap"]
    me_ds_new = fresh["diag_strut"]
    keep_set = {me_bal_new, me_ds_new}


    for v in fresh.values():
        if isinstance(v, bpy.types.Mesh) and v not in keep_set and v.users == 0:
            bpy.data.meshes.remove(v)


    old_to_drop = set()
    for i in range(spec.n_sectors):
        ob_b = bpy.data.objects.get(f"Flap_Balance_{i:02d}")
        if ob_b is not None:
            if ob_b.data is not None:
                old_to_drop.add(ob_b.data)
            ob_b.data = me_bal_new
        ob_d = bpy.data.objects.get(f"Link_DiagStrut_{i:02d}")
        if ob_d is not None:
            if ob_d.data is not None:
                old_to_drop.add(ob_d.data)
            ob_d.data = me_ds_new

    for om in list(old_to_drop):
        if om not in keep_set and om.users == 0:
            bpy.data.meshes.remove(om)


    me_bal_new.name = "Mesh_BalanceFlap"
    me_ds_new.name = "Mesh_Diagonal_Strut"
    bpy.context.view_layer.update()
    print(f"[Nozzle Placard Refresh] Mesh_BalanceFlap={len(me_bal_new.polygons):,} faces | Mesh_Diagonal_Strut={len(me_ds_new.polygons):,} faces")


def refresh_static_engine_in_current_scene(save_blend_path: str = None):

    clay_mat = bpy.data.materials.get("Mat_Clay_WhiteModel")
    ab_mat = bpy.data.materials.get("Mat_Clay_AB_Accent")
    if clay_mat is not None and ab_mat is not None:
        mats = {"clay": clay_mat, "ab_glow_ring": ab_mat}
    else:
        mats = build_clay_materials()
        if clay_mat is not None:
            mats["clay"] = clay_mat
        if ab_mat is not None:
            mats["ab_glow_ring"] = ab_mat

    def _clear_or_get_coll(cname: str):
        coll = bpy.data.collections.get(cname)
        if coll is None:
            coll = bpy.data.collections.new(cname)
            bpy.context.scene.collection.children.link(coll)
        else:
            old_objs = list(coll.all_objects)
            old_meshes = [o.data for o in old_objs if o.type == "MESH" and o.data is not None]
            for o in old_objs:
                bpy.data.objects.remove(o, do_unlink=True)
            for me in old_meshes:
                if me.users == 0:
                    bpy.data.meshes.remove(me)
        return coll

    coll_fwd = _clear_or_get_coll("00_Forward_Turbofan_Core_And_Fan")
    coll_static = _clear_or_get_coll("01_Static_Base_Casing")
    coll_interior = _clear_or_get_coll("02_Afterburner_Interior")

    spec = NozzleSpec()
    build_forward_turbofan_core_and_fan(spec, mats, coll_fwd)
    build_static_base_and_interior(spec, mats, coll_static, coll_interior)
    bpy.context.view_layer.update()

    refreshed_objs = [
        o for c in (coll_fwd, coll_static, coll_interior)
        for o in c.all_objects if o.type == "MESH"
    ]
    tot_q = sum(sum(1 for p in o.data.polygons if len(p.vertices) == 4) for o in refreshed_objs)
    tot_f = sum(len(o.data.polygons) for o in refreshed_objs)
    assert tot_q == tot_f, f"Static engine non-quads detected: {tot_f - tot_q}"
    print(f"[Engine Static Refresh] Objects={len(refreshed_objs)} | Quads={tot_q:,}")

    if save_blend_path:
        bpy.ops.wm.save_as_mainfile(filepath=save_blend_path)
        print(f"[Saved] {save_blend_path}")
