import math
import bmesh
from mathutils import Vector, Matrix

from f100_geom_primitives import *
import f100_geom_primitives as _gp

_bm_add_box_x_yz = _gp._bm_add_box_x_yz
_bm_add_annular_quad_extrusion = _gp._bm_add_annular_quad_extrusion


def add_shroud_detailing(bm_shroud, _r_sh_out, _cyl_p, _cyl_n):

    for ph_spl in (0.0, 45.0, 90.0, 135.0, 225.0, 270.0, 315.0):
        is_main = (ph_spl in (90.0, 270.0))
        half_ang = 3.40
        ya_sp, yb_sp = (-0.502, -0.338) if is_main else (-0.476, -0.364)

        if is_main:

            f0_gsk = len(bm_shroud.faces)
            bm_add_curved_cyl_panel(bm_shroud, ph_spl - 3.85, ph_spl + 3.85, ya_sp - 0.0055, ya_sp - 0.0004, +0.0024, +0.0046, n_phi=6, n_y=2, r_surf_fn=_r_sh_out)
            bm_add_curved_cyl_panel(bm_shroud, ph_spl - 3.85, ph_spl + 3.85, yb_sp + 0.0004, yb_sp + 0.0055, +0.0024, +0.0046, n_phi=6, n_y=2, r_surf_fn=_r_sh_out)
            bm_add_curved_cyl_panel(bm_shroud, ph_spl - 3.85, ph_spl - 3.43, ya_sp - 0.0004, yb_sp + 0.0004, +0.0024, +0.0046, n_phi=2, n_y=8, r_surf_fn=_r_sh_out)
            bm_add_curved_cyl_panel(bm_shroud, ph_spl + 3.43, ph_spl + 3.85, ya_sp - 0.0004, yb_sp + 0.0004, +0.0024, +0.0046, n_phi=2, n_y=8, r_surf_fn=_r_sh_out)
            bm_fill_unset_from(bm_shroud, f0_gsk, MAT_AVIONICS_BOX)


            ys_m = (-0.464, -0.436, -0.404, -0.376)
            for d_col in (-2.244, +2.244):
                ph_b = ph_spl + d_col
                n_b = _cyl_n(ph_b)
                pts_col = []
                for y_b in ys_m:
                    r_b = _r_sh_out(y_b) + 0.0050
                    p_b = _cyl_p(ph_b, y_b, r_b)
                    pts_col.append(p_b)
                    f0_fw = len(bm_shroud.faces)
                    bm_add_segmented_tube(bm_shroud, p_b - n_b * 0.0003, p_b + n_b * 0.0011,
                                          [(0.0, 0.0052), (1.0, 0.0048)], segs=16, cap_ends=True)
                    bm_fill_unset_from(bm_shroud, f0_fw, MAT_FASTENER)
                    bm_add_aerospace_bolt(bm_shroud, p_b + n_b * 0.0010, n_b, head_r=0.0032, height=0.0028)
                bm_add_bolt_pair_lockwire(bm_shroud, pts_col[0] + n_b * 0.0011, pts_col[1] + n_b * 0.0011, n_b, head_r=0.0032, strand_r=0.00036)
                bm_add_bolt_pair_lockwire(bm_shroud, pts_col[2] + n_b * 0.0011, pts_col[3] + n_b * 0.0011, n_b, head_r=0.0032, strand_r=0.00036)

            f0_rib = len(bm_shroud.faces)
            bm_add_curved_cyl_panel(bm_shroud, ph_spl - 1.35, ph_spl + 1.35, -0.452, -0.388, +0.0050, +0.0068, n_phi=4, n_y=4, r_surf_fn=_r_sh_out)
            bm_fill_unset_from(bm_shroud, f0_rib, MAT_MACHINED_ALU)
            for y_mid_b in (-0.440, -0.400):
                p_mb = _cyl_p(ph_spl, y_mid_b, _r_sh_out(y_mid_b) + 0.0068)
                bm_add_aerospace_bolt(bm_shroud, p_mb, _cyl_n(ph_spl), head_r=0.0032, height=0.0026)
        else:

            for d_col in (-2.244, +2.244):
                ph_b = ph_spl + d_col
                n_b = _cyl_n(ph_b)
                for y_b in (-0.438, -0.402):
                    p_b = _cyl_p(ph_b, y_b, _r_sh_out(y_b) + 0.0050)
                    bm_add_aerospace_bolt(bm_shroud, p_b, n_b, head_r=0.0028, height=0.0024)


    splice_angs = (0.0, 45.0, 90.0, 135.0, 225.0, 270.0, 315.0)
    for y_rv_b in (-0.470, -0.370):
        r_rv_b = _r_sh_out(y_rv_b) + 0.0026
        for k_rv in range(96):
            ph_r = k_rv * (360.0 / 96.0) + 1.875
            if 173.0 <= ph_r <= 187.0:
                continue
            if any(min(abs(ph_r - sa), 360.0 - abs(ph_r - sa)) < 4.0 for sa in splice_angs):
                continue
            bm_add_dome_rivet(bm_shroud, _cyl_p(ph_r, y_rv_b, r_rv_b), _cyl_n(ph_r), radius=0.0018)


    f0_skp = len(bm_shroud.faces)
    for k_sec_sh in range(8):
        for off_sk in (11.2, 18.7, 26.3, 33.8):
            ph_sk = k_sec_sh * 45.0 + off_sk
            if 173.0 <= ph_sk <= 187.0:
                continue
            n_sk = _cyl_n(ph_sk)
            p_sk0 = _cyl_p(ph_sk, -0.420, _r_sh_out(-0.420) + 0.0018)
            p_sk1 = _cyl_p(ph_sk, -0.420, _r_sh_out(-0.420) + 0.0044)
            bm_add_bevel_locked_hex(bm_shroud, p_sk0, p_sk1,
                                    [(0.0, 0.0028), (0.25, 0.0034), (0.80, 0.0034), (1.0, 0.0024)],
                                    corner_bevel=0.14, cap_ends=True)
    bm_fill_unset_from(bm_shroud, f0_skp, MAT_BRASS_FITTING)


    for k_ap in range(8):
        ph_ap = k_ap * 45.0 + 22.5
        n_ap = _cyl_n(ph_ap)
        r_ap = _r_sh_out(-0.268)
        p_pl0 = _cyl_p(ph_ap, -0.268, r_ap - 0.0006)
        p_pl1 = _cyl_p(ph_ap, -0.268, r_ap + 0.0032)
        f0_bp = len(bm_shroud.faces)
        bm_add_bevel_locked_hex(bm_shroud, p_pl0, p_pl1,
                                [(0.0, 0.0046), (0.30, 0.0050), (0.82, 0.0050), (1.0, 0.0038)],
                                corner_bevel=0.14, cap_ends=True)
        bm_fill_unset_from(bm_shroud, f0_bp, MAT_FASTENER)

        ph_anc = ph_ap + 1.65
        n_anc = _cyl_n(ph_anc)
        p_anc = _cyl_p(ph_anc, -0.268, r_ap + 0.0005)
        bm_add_aerospace_bolt(bm_shroud, p_anc, n_anc, head_r=0.0018, height=0.0018)
        bm_add_bolt_pair_lockwire(bm_shroud, p_pl1 - n_ap * 0.0012, p_anc + n_anc * 0.0006, n_ap, head_r=0.0026, strand_r=0.00028)


    for ph_h in (82.0, 98.0, 262.0, 278.0):
        n_h = _cyl_n(ph_h)
        h_pts = [
            _cyl_p(ph_h, -0.706, _r_sh_out(-0.706) + 0.0075),
            _cyl_p(ph_h, -0.600, _r_sh_out(-0.600) + 0.0075),
            _cyl_p(ph_h, -0.502, _r_sh_out(-0.502) + 0.0080),
            _cyl_p(ph_h, -0.480, _r_sh_out(-0.480) + 0.0115),
            _cyl_p(ph_h, -0.420, _r_sh_out(-0.420) + 0.0115),
            _cyl_p(ph_h, -0.360, _r_sh_out(-0.360) + 0.0115),
            _cyl_p(ph_h, -0.338, _r_sh_out(-0.338) + 0.0080),
            _cyl_p(ph_h, -0.220, _r_sh_out(-0.220) + 0.0075),
            _cyl_p(ph_h, -0.092, _r_sh_out(-0.092) + 0.0075),
        ]
        f0_ht = len(bm_shroud.faces)
        bm_add_spline_tube(bm_shroud, h_pts, radius=0.0038, n_samples=54, segs=14, cap_ends=True)
        bm_fill_unset_from(bm_shroud, f0_ht, MAT_RIGID_TUBE)

        for y_end_h, dir_y_h in ((-0.704, 1.0), (-0.094, -1.0)):
            r_end_s = _r_sh_out(y_end_h)
            p_bn = _cyl_p(ph_h, y_end_h + dir_y_h * 0.011, r_end_s + 0.0075)
            bm_add_bnut_flare_fitting(bm_shroud, p_bn, Vector((0.0, dir_y_h, 0.0)), tube_r=0.0038)
            b_bk = bmesh.new()
            f0_bk = len(b_bk.faces)
            _bm_add_box_x_yz(b_bk, -0.015, +0.015, y_end_h - 0.009, y_end_h + 0.009, r_end_s - 0.0012, r_end_s + 0.0125, nx=2, ny=2, nz=2, chamfer=0.0008)
            bm_fill_unset_from(b_bk, f0_bk, MAT_MACHINED_ALU)
            for dx_bkb in (-0.010, +0.010):
                bm_add_aerospace_bolt(b_bk, Vector((dx_bkb, y_end_h, r_end_s + 0.0125)), Vector((0.0, 0.0, 1.0)), head_r=0.0022, height=0.0020)
            bm_merge(bm_shroud, b_bk, Matrix.Rotation(math.radians(ph_h), 4, 'Y'))

        for y_cl_h, dr_cl in ((-0.625, 0.0075), (-0.516, 0.0078), (-0.420, 0.0115), (-0.312, 0.0078), (-0.188, 0.0075)):
            r_surf_c = _r_sh_out(y_cl_h) + (0.0024 if -0.478 <= y_cl_h <= -0.362 else 0.0)
            p_t_c = _cyl_p(ph_h, y_cl_h, _r_sh_out(y_cl_h) + dr_cl)
            p_b_c = _cyl_p(ph_h, y_cl_h, r_surf_c)
            bm_add_adel_loop_clamp(bm_shroud, p_t_c, Vector((0.0, 1.0, 0.0)), p_b_c, n_h, tube_r=0.0038)


def add_cradle_02_pedestal_and_bogie_details(bm_peds, X_RAIL, Y_FWD_ARCH, Y_AFT_ARCH, _bm_add_steel_plate):
    for y_ped, is_front in ((Y_FWD_ARCH, True), (Y_AFT_ARCH, False)):
        for sx in (-1.0, +1.0):
            x_in = sx * 0.588
            x_out = sx * 0.744
            x_lo, x_hi = min(x_in, x_out), max(x_in, x_out)
            xc = 0.5 * (x_in + x_out)
            x_rib_in = x_in + sx * 0.022
            xr_lo, xr_hi = min(x_rib_in, x_out), max(x_rib_in, x_out)
            xm_rib = 0.5 * (xr_lo + xr_hi)


            f0_ovc = len(bm_peds.faces)
            def _rib_oval_collar_pt(k: int, vr: float, vt: float, _yp=y_ped, _xm=xm_rib):
                th = 2.0 * math.pi * k / 40.0
                ct, st = math.cos(th), math.sin(th)
                xv = _xm + (0.0255 + 0.0085 * vr) * ct
                zv = -0.385 + (0.0875 + 0.0085 * vr) * st
                yv = (_yp - 0.0162) + 0.0324 * vt
                return Vector((xv, yv, zv))
            _bm_add_annular_quad_extrusion(bm_peds, _rib_oval_collar_pt, 40, [0.00, 0.16, 0.84, 1.00], [0.00, 0.16, 0.84, 1.00])
            bm_fill_unset_from(bm_peds, f0_ovc, MAT_STAND_STEEL)
            for sy_rv in (-1.0, +1.0):
                for kr_o in range(10):
                    th_r = 2.0 * math.pi * kr_o / 10.0
                    bm_add_dome_rivet(
                        bm_peds,
                        Vector((xm_rib + 0.0298 * math.cos(th_r), y_ped + sy_rv * 0.0162, -0.385 + 0.0918 * math.sin(th_r))),
                        Vector((0.0, sy_rv, 0.0)), radius=0.0018,
                    )


            for s_side in (-1.0, +1.0):
                y_fl_bot = y_ped + s_side * 0.112
                y_fl_top = y_ped + s_side * 0.098
                xc_m = 0.5 * (x_lo + x_hi)
                for kr_f in range(12):
                    th_f = 2.0 * math.pi * kr_f / 12.0
                    xv_f = xc_m + 0.0385 * math.cos(th_f)
                    zv_f = -0.385 + 0.0965 * math.sin(th_f)
                    fz_f = (zv_f - (-0.592)) / 0.424
                    yc_f = y_fl_bot + (y_fl_top - y_fl_bot) * fz_f
                    bm_add_dome_rivet(bm_peds, Vector((xv_f, yc_f + s_side * 0.0156, zv_f)), Vector((0.0, s_side, 0.0)), radius=0.0019)


            for z_gp in (-0.525, -0.245):
                f0_gp = len(bm_peds.faces)
                _bm_add_steel_plate(
                    bm_peds, min(x_out - sx * 0.002, x_out + sx * 0.008), max(x_out - sx * 0.002, x_out + sx * 0.008),
                    y_ped - 0.018, y_ped + 0.018, z_gp - 0.024, z_gp + 0.024, nu=1, nv=2, chamfer=0.0008,
                )
                bm_fill_unset_from(bm_peds, f0_gp, MAT_PUMPS_OILTANK)
                for dz_gb in (-0.013, +0.013):
                    bm_add_aerospace_bolt(bm_peds, Vector((x_out + sx * 0.008, y_ped, z_gp + dz_gb)), Vector((sx, 0, 0)), head_r=0.0032, height=0.0030)


            f0_brb = len(bm_peds.faces)
            _bm_add_steel_plate(
                bm_peds, min(x_out + sx * 0.008, x_out + sx * 0.019), max(x_out + sx * 0.008, x_out + sx * 0.019),
                y_ped - 0.158, y_ped + 0.158, -0.648, -0.630, nu=1, nv=4, chamfer=0.0010,
            )
            bm_fill_unset_from(bm_peds, f0_brb, MAT_PUMPS_OILTANK)
            for dy_bb0, dy_bb1 in ((-0.142, -0.104), (+0.104, +0.142)):
                p0 = Vector((x_out + sx * 0.019, y_ped + dy_bb0, -0.639))
                p1 = Vector((x_out + sx * 0.019, y_ped + dy_bb1, -0.639))
                bm_add_aerospace_bolt(bm_peds, p0, Vector((sx, 0, 0)), head_r=0.0034, height=0.0030)
                bm_add_aerospace_bolt(bm_peds, p1, Vector((sx, 0, 0)), head_r=0.0034, height=0.0030)
                bm_add_bolt_pair_lockwire(bm_peds, p0, p1, Vector((sx, 0, 0)), head_r=0.0034, strand_r=0.00040)


            for dy_rol in (-0.124, +0.124):
                bm_add_grease_zerk(bm_peds, Vector((x_out + sx * 0.025, y_ped + dy_rol, -0.692)), Vector((sx, 0, 0)), scale=1.15)
            bm_add_grease_zerk(bm_peds, Vector((x_out + sx * 0.019, y_ped, -0.639)), Vector((sx, 0, 0)), scale=1.0)


            for sy_bm in (-1.0, +1.0):
                y_bm0 = y_ped + sy_bm * 0.200
                y_bm1 = y_ped + sy_bm * 0.214
                f0_rbm = len(bm_peds.faces)
                _bm_add_steel_plate(bm_peds, xc - 0.044, xc + 0.044, min(y_bm0, y_bm1), max(y_bm0, y_bm1), -0.668, -0.632, nu=3, nv=2, chamfer=0.0014)
                bm_fill_unset_from(bm_peds, f0_rbm, MAT_RUBBER_DARK)
                for dx_rb in (-0.026, +0.026):
                    bm_add_aerospace_bolt(bm_peds, Vector((xc + dx_rb, y_bm1, -0.650)), Vector((0.0, sy_bm, 0.0)), head_r=0.0028, height=0.0020)


def add_cradle_03_wormgear_and_pippins(bm_arches, Y_FWD_ARCH, Y_AFT_ARCH, _bm_add_steel_plate):


    f0_wg = len(bm_arches.faces)
    bm_add_chamfered_cyl_x(bm_arches, Y_FWD_ARCH, -0.054, +0.770, +0.793, radius=0.053, segs=32, hollow_r=0.027)
    for k_rib in range(8):
        ar_w = 2.0 * math.pi * k_rib / 8.0
        p_w0 = Vector((+0.772, Y_FWD_ARCH + 0.052 * math.cos(ar_w), -0.054 + 0.052 * math.sin(ar_w)))
        p_w1 = Vector((+0.791, Y_FWD_ARCH + 0.052 * math.cos(ar_w), -0.054 + 0.052 * math.sin(ar_w)))
        bm_add_segmented_tube(bm_arches, p_w0, p_w1, [(0.0, 0.0032), (1.0, 0.0032)], segs=8, cap_ends=True)
    bm_fill_unset_from(bm_arches, f0_wg, MAT_AVIONICS_BOX)

    f0_wgb = len(bm_arches.faces)
    for sz_op in (-1.0, +1.0):
        bm_add_bevel_locked_hex(
            bm_arches, Vector((+0.782, Y_FWD_ARCH, -0.054 + sz_op * 0.052)),
            Vector((+0.782, Y_FWD_ARCH, -0.054 + sz_op * 0.062)),
            [(0.0, 0.0068), (0.30, 0.0082), (0.80, 0.0082), (1.0, 0.0062)], corner_bevel=0.14, cap_ends=True,
        )
    bm_fill_unset_from(bm_arches, f0_wgb, MAT_BRASS_FITTING)

    f0_rat = len(bm_arches.faces)
    for kt in range(16):
        at = 2.0 * math.pi * kt / 16.0
        p_t0 = Vector((+0.793, Y_FWD_ARCH + 0.028 * math.cos(at), -0.054 + 0.028 * math.sin(at)))
        p_t1 = Vector((+0.793, Y_FWD_ARCH + 0.038 * math.cos(at), -0.054 + 0.038 * math.sin(at)))
        bm_add_segmented_tube(bm_arches, p_t0, p_t1, [(0.0, 0.0034), (1.0, 0.0020)], segs=8, cap_ends=True)
    bm_add_clevis_plate(bm_arches, (Y_FWD_ARCH - 0.036, -0.012), (Y_FWD_ARCH - 0.006, -0.022), 0.0065, 0.0045, +0.789, +0.797, arc_segs=9)
    bm_fill_unset_from(bm_arches, f0_rat, MAT_FASTENER)
    bm_add_aerospace_bolt(bm_arches, Vector((+0.797, Y_FWD_ARCH - 0.036, -0.012)), Vector((1, 0, 0)), head_r=0.0036, height=0.0034)

    for ib_wg in (1, 4, 6, 9):
        ang_wg = 2.0 * math.pi * ib_wg / 10.0
        bm_add_aerospace_bolt(
            bm_arches, Vector((+0.793, Y_FWD_ARCH + 0.044 * math.cos(ang_wg), -0.054 + 0.044 * math.sin(ang_wg))),
            Vector((1.0, 0.0, 0.0)), head_r=0.0038, height=0.0034,
        )


    for y_ar, is_front_ar in ((Y_FWD_ARCH, True), (Y_AFT_ARCH, False)):
        for s_ang in (-1.0, +1.0):
            for d_pin_s in (46.0, 68.0):
                ar_pin = math.radians(s_ang * d_pin_s)
                xp = 0.643 * math.sin(ar_pin)
                zp = 0.643 * math.cos(ar_pin)
                sy_out = -1.0 if is_front_ar else +1.0

                f0_pp = len(bm_arches.faces)
                bm_add_segmented_tube(
                    bm_arches, Vector((xp, y_ar + sy_out * 0.064, zp)), Vector((xp, y_ar + sy_out * 0.076, zp)),
                    [(0.0, 0.011), (0.40, 0.013), (0.85, 0.013), (1.0, 0.009)], segs=16, cap_ends=True,
                )
                ring_p = [
                    Vector((xp + 0.012 * math.cos(2.0 * math.pi * i / 14.0), y_ar + sy_out * 0.078, zp + 0.012 * math.sin(2.0 * math.pi * i / 14.0)))
                    for i in range(15)
                ]
                bm_add_spline_tube(bm_arches, ring_p, radius=0.0016, n_samples=16, segs=8, cap_ends=False, closed=True)
                bm_fill_unset_from(bm_arches, f0_pp, MAT_FASTENER)

                ar_anc = math.radians(s_ang * (d_pin_s + 5.5))
                bm_add_twisted_lockwire(
                    bm_arches,
                    [
                        Vector((xp, y_ar + sy_out * 0.078, zp - 0.011)),
                        Vector((0.5 * (xp + 0.625 * math.sin(ar_anc)), y_ar + sy_out * 0.082, zp - 0.042)),
                        Vector((0.625 * math.sin(ar_anc), y_ar + sy_out * 0.054, 0.625 * math.cos(ar_anc))),
                    ],
                    strand_r=0.00068, twists_per_m=170.0, n_samples=18, segs=10,
                )

        z_crown = 0.736 if is_front_ar else 0.748
        y_lug_c = y_ar + (0.072 if is_front_ar else -0.072)
        f0_cp = len(bm_arches.faces)
        ring_c = [
            Vector((-0.054, y_lug_c + 0.013 * math.cos(2.0 * math.pi * i / 14.0), z_crown + 0.013 * math.sin(2.0 * math.pi * i / 14.0)))
            for i in range(15)
        ]
        bm_add_spline_tube(bm_arches, ring_c, radius=0.0018, n_samples=16, segs=8, cap_ends=False, closed=True)

        _bm_add_steel_plate(bm_arches, -0.067, -0.060, y_ar - 0.008, y_ar + 0.008, z_crown - 0.032, z_crown - 0.020, nu=1, nv=1, chamfer=0.0005)
        bm_add_dome_rivet(bm_arches, Vector((-0.067, y_ar, z_crown - 0.026)), Vector((-1.0, 0.0, 0.0)), radius=0.0022)
        bm_fill_unset_from(bm_arches, f0_cp, MAT_FASTENER)
        bm_add_twisted_lockwire(
            bm_arches,
            [
                Vector((-0.054, y_lug_c, z_crown - 0.013)),
                Vector((-0.066, y_lug_c, z_crown - 0.014)),
                Vector((-0.071, 0.5 * (y_ar + y_lug_c), z_crown - 0.025)),
                Vector((-0.066, y_ar, z_crown - 0.026)),
            ],
            strand_r=0.00072, twists_per_m=165.0, n_samples=22, segs=10,
        )


def add_fwd02_inlet_and_deec_detailing(bm_fcase, _cp, _cn):

    front_bolts = []
    for k_fb in range(40):
        ang_d = k_fb * 9.0 + 4.5
        if (168.0 <= ang_d <= 192.0) or any(min(abs(ang_d - dp), 360.0 - abs(ang_d - dp)) < 4.8 for dp in (45.0, 135.0, 225.0, 315.0)):
            front_bolts.append(None)
            continue
        p_fb = _cp(ang_d, -4.273, 0.514)
        front_bolts.append(p_fb)
        bm_add_aerospace_bolt(bm_fcase, p_fb, Vector((0.0, -1.0, 0.0)), head_r=0.0028, height=0.0026)
    for k_fb in range(0, 40, 2):
        if front_bolts[k_fb] is not None and front_bolts[k_fb + 1] is not None:
            bm_add_bolt_pair_lockwire(bm_fcase, front_bolts[k_fb], front_bolts[k_fb + 1], Vector((0.0, -1.0, 0.0)), head_r=0.0028, strand_r=0.00032)


    for ph_pr in (30.0, 148.0, 212.0, 330.0):
        rot_pr = Matrix.Rotation(math.radians(ph_pr), 4, 'Y')
        b_pr = bmesh.new()

        f0_pb = len(b_pr.faces)
        _bm_add_box_x_yz(b_pr, -0.015, +0.015, -4.268, -4.242, 0.522, 0.533, nx=2, ny=2, nz=1, chamfer=0.0010)
        bm_fill_unset_from(b_pr, f0_pb, MAT_MACHINED_ALU)
        for dx_pb in (-0.010, +0.010):
            bm_add_aerospace_bolt(b_pr, Vector((dx_pb, -4.255, 0.533)), Vector((0, 0, 1)), head_r=0.0020, height=0.0018)

        f0_pt = len(b_pr.faces)
        bm_add_segmented_tube(
            b_pr, Vector((0.0, -4.256, 0.531)), Vector((0.0, -4.294, 0.531)),
            [(0.0, 0.0062), (0.35, 0.0062), (0.48, 0.0042), (0.92, 0.0042), (1.0, 0.0032)], segs=16, cap_ends=True,
        )

        bm_add_spline_tube(
            b_pr,
            [Vector((0.0, -4.244, 0.531)), Vector((0.0, -4.228, 0.531)), Vector((0.0, -4.214, 0.522))],
            radius=0.0030, n_samples=16, segs=12, cap_ends=True,
        )
        bm_fill_unset_from(b_pr, f0_pt, MAT_RIGID_TUBE)
        bm_add_bnut_flare_fitting(b_pr, Vector((0.0, -4.242, 0.531)), Vector((0.0, 1.0, 0.0)), tube_r=0.0030)
        bm_merge(bm_fcase, b_pr, rot_pr)


    for k_vb in range(8):
        rot_vb = Matrix.Rotation(math.radians(k_vb * 45.0), 4, 'Y')
        b_vb = bmesh.new()
        f0_vb = len(b_vb.faces)
        for dx_ve in (-0.011, +0.011):
            _bm_add_box_x_yz(b_vb, dx_ve - 0.0032, dx_ve + 0.0032, -4.266, -4.254, 0.526, 0.542, nx=1, ny=1, nz=2, chamfer=0.0005)
        bm_add_segmented_tube(b_vb, Vector((-0.018, -4.260, 0.536)), Vector((+0.018, -4.260, 0.536)), [(0.0, 0.0024), (1.0, 0.0024)], segs=10, cap_ends=True)
        bm_add_bevel_locked_hex(b_vb, Vector((+0.014, -4.260, 0.536)), Vector((+0.020, -4.260, 0.536)),
                                [(0.0, 0.0040), (0.30, 0.0048), (0.80, 0.0048), (1.0, 0.0036)], corner_bevel=0.14, cap_ends=True)
        bm_fill_unset_from(b_vb, f0_vb, MAT_FASTENER)
        bm_merge(bm_fcase, b_vb, rot_vb)


    b_deec18 = bmesh.new()
    for kx_cp, xc_p in enumerate((-0.058, -0.019, +0.019, +0.058)):
        zc_p = 0.556 + (0.008 if kx_cp % 2 == 0 else -0.006)

        f0_kn = len(b_deec18.faces)
        bm_add_segmented_tube(
            b_deec18, Vector((xc_p, -3.994, zc_p)), Vector((xc_p, -3.980, zc_p)),
            [(0.0, 0.0135), (0.20, 0.0145), (0.50, 0.0138), (0.80, 0.0145), (1.0, 0.0128)], segs=22, cap_ends=True,
        )
        bm_fill_unset_from(b_deec18, f0_kn, MAT_AVIONICS_BOX)

        f0_el = len(b_deec18.faces)
        bm_add_spline_tube(
            b_deec18,
            [Vector((xc_p, -3.981, zc_p)), Vector((xc_p, -3.969, zc_p - 0.003)), Vector((xc_p, -3.964, zc_p - 0.012))],
            radius=0.0092, n_samples=12, segs=16, cap_ends=True,
        )
        bm_fill_unset_from(b_deec18, f0_el, MAT_MACHINED_ALU)

        f0_hs = len(b_deec18.faces)
        bm_add_segmented_tube(
            b_deec18, Vector((xc_p, -3.964, zc_p - 0.010)), Vector((xc_p, -3.956, zc_p - 0.024)),
            [(0.0, 0.0096), (0.35, 0.0086), (1.0, 0.0072)], segs=16, cap_ends=True,
        )
        bm_fill_unset_from(b_deec18, f0_hs, (MAT_ADEL_CUSHION if kx_cp % 2 == 0 else MAT_YELLOW_HARNESS))

        f0_bc = len(b_deec18.faces)
        bm_add_spline_tube(
            b_deec18,
            [Vector((xc_p, -3.958, max(0.548, zc_p - 0.016))), Vector((xc_p * 0.92, -3.944, 0.552)), Vector((xc_p * 0.85, -3.918, 0.550))],
            radius=0.0062, n_samples=16, segs=14, cap_ends=True, corrugations=10,
        )
        bm_fill_unset_from(b_deec18, f0_bc, MAT_BRAID_HOSE)


    f0_df = len(b_deec18.faces)
    _bm_add_box_x_yz(b_deec18, +0.022, +0.052, -4.233, -4.227, 0.542, 0.572, nx=2, ny=1, nz=2, chamfer=0.0008)
    bm_fill_unset_from(b_deec18, f0_df, MAT_MACHINED_ALU)
    for dx_dt in (+0.026, +0.048):
        for dz_dt in (0.546, 0.568):
            bm_add_aerospace_bolt(b_deec18, Vector((dx_dt, -4.233, dz_dt)), Vector((0, -1, 0)), head_r=0.0018, height=0.0016)
    f0_dc = len(b_deec18.faces)
    bm_add_segmented_tube(
        b_deec18, Vector((+0.037, -4.232, 0.557)), Vector((+0.037, -4.248, 0.557)),
        [(0.0, 0.0095), (0.35, 0.0108), (0.75, 0.0108), (1.0, 0.0086)], segs=20, cap_ends=True,
    )
    bm_fill_unset_from(b_deec18, f0_dc, MAT_BRASS_FITTING)
    bm_add_twisted_lockwire(
        b_deec18,
        [Vector((+0.037, -4.248, 0.557)), Vector((+0.054, -4.250, 0.548)), Vector((+0.064, -4.232, 0.548))],
        strand_r=0.00055, twists_per_m=180.0, n_samples=16, segs=10,
    )

    f0_emi = len(b_deec18.faces)
    bm_add_spline_tube(
        b_deec18,
        [Vector((+0.068, -4.230, 0.546)), Vector((+0.072, -4.238, 0.526)), Vector((+0.068, -4.236, 0.512))],
        radius=0.0036, n_samples=14, segs=12, cap_ends=True, corrugations=10,
    )
    bm_fill_unset_from(b_deec18, f0_emi, MAT_BRASS_FITTING)
    bm_add_aerospace_bolt(b_deec18, Vector((+0.068, -4.230, 0.546)), Vector((0, -1, 0)), head_r=0.0024, height=0.0022)
    bm_merge(bm_fcase, b_deec18, Matrix.Rotation(math.radians(132.0), 4, 'Y'))


def add_trailer_05_4w_steering_and_xbracing(bm_tr, X_RAIL, Z_AXLE, _bm_add_steel_plate):

    p_fpiv = Vector((0.0, -4.002, -1.410))
    p_rpiv = Vector((0.0, -1.373, -1.410))
    p_fend = Vector((-0.115, -3.760, -1.410))
    p_rend = Vector((+0.095, -1.615, -1.410))
    dir_dl = (p_rend - p_fend).normalized()


    f0_arms = len(bm_tr.faces)
    bm_add_segmented_tube(bm_tr, p_fpiv, p_fend, [(0.0, 0.022), (0.18, 0.020), (0.85, 0.017), (1.0, 0.019)], segs=18, cap_ends=True)
    bm_add_segmented_tube(bm_tr, p_rpiv, p_rend, [(0.0, 0.022), (0.18, 0.020), (0.85, 0.017), (1.0, 0.019)], segs=18, cap_ends=True)
    for p_piv in (p_fpiv, p_rpiv):
        bm_add_segmented_tube(bm_tr, p_piv + Vector((0, 0, -0.018)), p_piv + Vector((0, 0, +0.020)),
                              [(0.0, 0.035), (0.15, 0.037), (0.85, 0.037), (1.0, 0.034)], segs=24, cap_ends=True)
    bm_fill_unset_from(bm_tr, f0_arms, MAT_PUMPS_OILTANK)


    f0_sh = len(bm_tr.faces)
    bm_add_segmented_tube(
        bm_tr, p_fend + dir_dl * 0.014, p_rend - dir_dl * 0.014,
        [
            (0.00, 0.013), (0.03, 0.0165), (0.97, 0.0165), (1.00, 0.013),
        ],
        segs=22, cap_ends=True,
    )
    bm_fill_unset_from(bm_tr, f0_sh, MAT_STAND_STEEL)


    for p_je in (p_fend, p_rend):
        f0_jb = len(bm_tr.faces)
        bm_add_segmented_tube(bm_tr, p_je + Vector((0, 0, -0.018)), p_je + Vector((0, 0, +0.020)),
                              [(0.0, 0.014), (0.35, 0.022), (0.65, 0.022), (1.0, 0.014)], segs=18, cap_ends=True)
        bm_fill_unset_from(bm_tr, f0_jb, MAT_RUBBER_DARK)
        f0_jn = len(bm_tr.faces)
        bm_add_bevel_locked_hex(bm_tr, p_je + Vector((0, 0, -0.028)), p_je + Vector((0, 0, -0.016)),
                                [(0.0, 0.012), (0.25, 0.015), (0.80, 0.015), (1.0, 0.011)], corner_bevel=0.14, cap_ends=True)
        bm_fill_unset_from(bm_tr, f0_jn, MAT_FASTENER)
        bm_add_grease_zerk(bm_tr, p_je + Vector((0, 0, -0.028)), Vector((0, 0, -1)), scale=0.95)


    for t_tb in (0.16, 0.84):
        p_tc = p_fend.lerp(p_rend, t_tb)
        f0_tb = len(bm_tr.faces)
        bm_add_bevel_locked_hex(bm_tr, p_tc - dir_dl * 0.055, p_tc + dir_dl * 0.055,
                                [(0.0, 0.019), (0.15, 0.023), (0.85, 0.023), (1.0, 0.019)], corner_bevel=0.14, cap_ends=True)
        bm_fill_unset_from(bm_tr, f0_tb, MAT_STAND_STEEL)
        f0_th = len(bm_tr.faces)
        for s_e in (-1.0, +1.0):
            bm_add_helical_thread_cylinder(bm_tr, p_tc + dir_dl * (s_e * 0.054), p_tc + dir_dl * (s_e * 0.092),
                                           r_core=0.0145, thread_h=0.0020, n_turns=6, segs_turn=16, cap_core=False)
            bm_add_bevel_locked_hex(bm_tr, p_tc + dir_dl * (s_e * 0.058), p_tc + dir_dl * (s_e * 0.074),
                                    [(0.0, 0.018), (0.25, 0.022), (0.80, 0.022), (1.0, 0.017)], corner_bevel=0.14, cap_ends=True)
        bm_fill_unset_from(bm_tr, f0_th, MAT_FASTENER)

        bm_add_twisted_lockwire(
            bm_tr,
            [p_tc - dir_dl * 0.066 + Vector((0, 0, -0.0190)), p_tc + Vector((0, 0, -0.0215)), p_tc + dir_dl * 0.066 + Vector((0, 0, -0.0190))],
            strand_r=0.00060, twists_per_m=150.0, n_samples=20, segs=10,
        )


    p_mid = p_fend.lerp(p_rend, 0.50)

    f0_mb = len(bm_tr.faces)
    _bm_add_steel_plate(bm_tr, -X_RAIL + 0.074, +X_RAIL - 0.074, p_mid.y - 0.036, p_mid.y + 0.036, -1.012, -0.968, nu=12, nv=2, chamfer=0.0014)
    bm_fill_unset_from(bm_tr, f0_mb, MAT_TRAILER_YELLOW)


    y_loc = dir_dl.normalized()
    z_loc = Vector((0.0, 0.0, 1.0))
    x_loc = y_loc.cross(z_loc).normalized()
    m_mid = Matrix.Identity(4)
    m_mid[0][0], m_mid[1][0], m_mid[2][0] = x_loc.x, x_loc.y, x_loc.z
    m_mid[0][1], m_mid[1][1], m_mid[2][1] = y_loc.x, y_loc.y, y_loc.z
    m_mid[0][2], m_mid[1][2], m_mid[2][2] = 0.0, 0.0, 1.0
    m_mid[0][3], m_mid[1][3], m_mid[2][3] = p_mid.x, p_mid.y, p_mid.z
    dz_top = -1.010 - p_mid.z

    b_mb = bmesh.new()
    f0_cb = len(b_mb.faces)

    _bm_add_steel_plate(b_mb, -0.056, +0.056, -0.032, +0.032, dz_top - 0.014, dz_top + 0.002, nu=3, nv=2, chamfer=0.0012)

    for sx_p in (-1.0, +1.0):
        _bm_add_steel_plate(b_mb, sx_p * 0.025, sx_p * 0.037, -0.020, +0.020, 0.004, dz_top - 0.012, nu=1, nv=3, chamfer=0.0010)

    _bm_add_steel_plate(b_mb, -0.026, +0.026, -0.007, +0.007, 0.030, dz_top - 0.012, nu=2, nv=4, chamfer=0.0010)

    b_col = bmesh.new()
    bm_add_chamfered_cyl_x(b_col, 0.0, 0.0, -0.024, +0.024, radius=0.034, segs=32, chamfer=0.0012, hollow_r=0.0235)
    bm_merge(b_mb, b_col, Matrix.Rotation(0.5 * math.pi, 4, 'Z'))
    bm_fill_unset_from(b_mb, f0_cb, MAT_AVIONICS_BOX)
    for dx_fb in (-0.042, +0.042):
        for dy_fb in (-0.020, +0.020):
            bm_add_aerospace_bolt(b_mb, Vector((dx_fb, dy_fb, dz_top - 0.014)), Vector((0, 0, -1)), head_r=0.0036, height=0.0034)

    f0_bb = len(b_mb.faces)
    b_brz = bmesh.new()
    bm_add_chamfered_cyl_x(b_brz, 0.0, 0.0, -0.0285, +0.0285, radius=0.0240, segs=28, chamfer=0.0008, hollow_r=0.0168)
    for sy_fl in (-0.0262, +0.0262):
        bm_add_chamfered_cyl_x(b_brz, 0.0, 0.0, sy_fl - 0.0024, sy_fl + 0.0024, radius=0.0285, segs=28, chamfer=0.0006, hollow_r=0.0168)
    bm_merge(b_mb, b_brz, Matrix.Rotation(0.5 * math.pi, 4, 'Z'))
    bm_fill_unset_from(b_mb, f0_bb, MAT_BRASS_FITTING)
    bm_merge(bm_tr, b_mb, m_mid)


    for ya_x, yb_x in ((-3.660, -3.080), (-2.020, -1.440)):
        ym_x = 0.5 * (ya_x + yb_x)
        f0_xb = len(bm_tr.faces)
        for p0_x, p1_x in (
            (Vector((-0.525, ya_x, -0.978)), Vector((+0.525, yb_x, -0.978))),
            (Vector((+0.525, ya_x, -0.978)), Vector((-0.525, yb_x, -0.978))),
        ):
            bm_add_segmented_tube(bm_tr, p0_x, p1_x, [(0.0, 0.011), (0.06, 0.010), (0.94, 0.010), (1.0, 0.011)], segs=14, cap_ends=True)

        _bm_add_steel_plate(bm_tr, -0.056, +0.056, ym_x - 0.048, ym_x + 0.048, -0.992, -0.964, nu=2, nv=2, chamfer=0.0012)
        bm_fill_unset_from(bm_tr, f0_xb, MAT_PUMPS_OILTANK)
        for dx_xb in (-0.036, +0.036):
            for dy_xb in (-0.028, +0.028):
                bm_add_aerospace_bolt(bm_tr, Vector((dx_xb, ym_x + dy_xb, -0.992)), Vector((0, 0, -1)), head_r=0.0038, height=0.0035)


    f0_rg = len(bm_tr.faces)
    for y_gm in (-3.500, -2.500, -1.800):
        for sx_g in (-1.0, +1.0):
            bm_add_segmented_tube(bm_tr, Vector((sx_g * 0.616, y_gm - 0.009, -0.957)), Vector((sx_g * 0.616, y_gm + 0.009, -0.957)),
                                  [(0.0, 0.010), (0.25, 0.012), (0.75, 0.012), (1.0, 0.010)], segs=16, cap_ends=True)
    bm_fill_unset_from(bm_tr, f0_rg, MAT_RUBBER_DARK)


def add_cradle_01_fodbox_and_canister_hw(bm_rails, _bm_add_steel_plate):
    bx0, bx1 = +0.105, +0.525
    by0, by1 = -1.940, -1.440
    bz1 = -0.712


    f0_pl = len(bm_rails.faces)
    _bm_add_steel_plate(bm_rails, +0.328, +0.414, -1.858, -1.522, -0.6905, -0.6890, nu=2, nv=4, chamfer=0.0004)
    bm_fill_unset_from(bm_rails, f0_pl, MAT_PLATE_BASE)


    f0_hp = len(bm_rails.faces)
    bm_add_segmented_tube(bm_rails, Vector((bx0 + 0.022, by0 - 0.011, bz1 + 0.001)), Vector((bx1 - 0.022, by0 - 0.011, bz1 + 0.001)),
                          [(0.0, 0.0028), (1.0, 0.0028)], segs=12, cap_ends=True)
    bm_fill_unset_from(bm_rails, f0_hp, MAT_FASTENER)


    xc_lk = +0.315
    f0_hsp = len(bm_rails.faces)
    _bm_add_steel_plate(bm_rails, xc_lk - 0.017, xc_lk + 0.017, by1 + 0.004, by1 + 0.014, bz1 - 0.032, bz1 + 0.016, nu=2, nv=2, chamfer=0.0008)
    st_pts = [
        Vector((xc_lk, by1 + 0.013, bz1 - 0.022)),
        Vector((xc_lk, by1 + 0.028, bz1 - 0.022)),
        Vector((xc_lk, by1 + 0.028, bz1 - 0.010)),
        Vector((xc_lk, by1 + 0.013, bz1 - 0.010)),
    ]
    bm_add_spline_tube(bm_rails, st_pts, radius=0.0026, n_samples=16, segs=10, cap_ends=True)
    bm_fill_unset_from(bm_rails, f0_hsp, MAT_FASTENER)
    for dx_hr in (-0.010, +0.010):
        bm_add_dome_rivet(bm_rails, Vector((xc_lk + dx_hr, by1 + 0.014, bz1 + 0.008)), Vector((0, 1, 0)), radius=0.0020)
    f0_plk = len(bm_rails.faces)
    _bm_add_steel_plate(bm_rails, xc_lk - 0.015, xc_lk + 0.015, by1 + 0.019, by1 + 0.035, bz1 - 0.056, bz1 - 0.026, nu=2, nv=2, chamfer=0.0014)
    bm_fill_unset_from(bm_rails, f0_plk, MAT_BRASS_FITTING)
    f0_shk = len(bm_rails.faces)
    shk_pts = [
        Vector((xc_lk - 0.008, by1 + 0.027, bz1 - 0.027)),
        Vector((xc_lk - 0.008, by1 + 0.027, bz1 - 0.016)),
        Vector((xc_lk,         by1 + 0.027, bz1 - 0.012)),
        Vector((xc_lk + 0.008, by1 + 0.027, bz1 - 0.016)),
        Vector((xc_lk + 0.008, by1 + 0.027, bz1 - 0.027)),
    ]
    bm_add_spline_tube(bm_rails, shk_pts, radius=0.0024, n_samples=18, segs=10, cap_ends=True)
    bm_fill_unset_from(bm_rails, f0_shk, MAT_FASTENER)


def add_trailer_07_drippan_mesh_and_drain(bm_gse, Z_FL, _bm_add_steel_plate):
    px0, px1 = -0.354, +0.354
    py0, py1 = -2.914, -2.386
    mx0, mx1 = px0 + 0.018, px1 - 0.018
    my0, my1 = py0 + 0.018, py1 - 0.018
    zm = Z_FL + 0.0165


    f0_ms = len(bm_gse.faces)

    for xs_l in (mx0 + 0.016, mx1 - 0.016):
        for ys_l in (my0 + 0.016, my1 - 0.016):
            _bm_add_steel_plate(bm_gse, xs_l - 0.005, xs_l + 0.005, ys_l - 0.005, ys_l + 0.005, Z_FL + 0.0025, zm, nu=1, nv=1, chamfer=0.0005)

    _bm_add_steel_plate(bm_gse, mx0, mx1, my0, my0 + 0.010, zm - 0.0018, zm + 0.0018, nu=6, nv=1, chamfer=0.0004)
    _bm_add_steel_plate(bm_gse, mx0, mx1, my1 - 0.010, my1, zm - 0.0018, zm + 0.0018, nu=6, nv=1, chamfer=0.0004)
    _bm_add_steel_plate(bm_gse, mx0, mx0 + 0.010, my0 + 0.009, my1 - 0.009, zm - 0.0018, zm + 0.0018, nu=1, nv=6, chamfer=0.0004)
    _bm_add_steel_plate(bm_gse, mx1 - 0.010, mx1, my0 + 0.009, my1 - 0.009, zm - 0.0018, zm + 0.0018, nu=1, nv=6, chamfer=0.0004)

    for ix in range(13):
        xg = (mx0 + 0.032) + (mx1 - mx0 - 0.064) * (ix / 12.0)
        _bm_add_steel_plate(bm_gse, xg - 0.0014, xg + 0.0014, my0 + 0.009, my1 - 0.009, zm - 0.0016, zm + 0.0016, nu=1, nv=2, chamfer=0.00025)
    for iy in range(9):
        yg = (my0 + 0.030) + (my1 - my0 - 0.060) * (iy / 8.0)
        _bm_add_steel_plate(bm_gse, mx0 + 0.009, mx1 - 0.009, yg - 0.0014, yg + 0.0014, zm - 0.0012, zm + 0.0012, nu=2, nv=1, chamfer=0.00025)

    for sx_t in (-1.0, +1.0):
        xt = sx_t * (mx1 - 0.005)
        bm_add_spline_tube(
            bm_gse,
            [Vector((xt, -2.675, zm)), Vector((xt, -2.675, zm + 0.012)), Vector((xt, -2.625, zm + 0.012)), Vector((xt, -2.625, zm))],
            radius=0.0015, n_samples=16, segs=8, cap_ends=True,
        )
    bm_fill_unset_from(bm_gse, f0_ms, MAT_RIGID_TUBE)


    y_dr, z_dr = -2.455, Z_FL + 0.011
    f0_dr = len(bm_gse.faces)
    bm_add_bevel_locked_hex(bm_gse, Vector((+0.354, y_dr, z_dr)), Vector((+0.365, y_dr, z_dr)),
                            [(0.0, 0.0075), (0.30, 0.0092), (0.80, 0.0092), (1.0, 0.0072)], corner_bevel=0.14, cap_ends=True)
    bm_add_segmented_tube(bm_gse, Vector((+0.364, y_dr, z_dr)), Vector((+0.394, y_dr, z_dr)),
                          [(0.0, 0.0054), (0.55, 0.0058), (0.70, 0.0042), (1.0, 0.0038)], segs=16, cap_ends=True)

    bm_add_segmented_tube(bm_gse, Vector((+0.376, y_dr, z_dr - 0.004)), Vector((+0.376, y_dr, z_dr + 0.014)),
                          [(0.0, 0.0046), (0.65, 0.0046), (1.0, 0.0032)], segs=14, cap_ends=True)
    bm_add_segmented_tube(bm_gse, Vector((+0.376, y_dr - 0.014, z_dr + 0.013)), Vector((+0.376, y_dr + 0.014, z_dr + 0.013)),
                          [(0.0, 0.0024), (0.50, 0.0032), (1.0, 0.0024)], segs=12, cap_ends=True)
    bm_fill_unset_from(bm_gse, f0_dr, MAT_BRASS_FITTING)


def add_cradle_05_inlet_probe_rbf(bm_rbf, uv_rbf):
    ph_p = math.radians(148.0)
    xp = 0.531 * math.sin(ph_p)
    zp = 0.531 * math.cos(ph_p)
    yp = -4.286


    f0_cap = len(bm_rbf.faces)
    bm_add_segmented_tube(
        bm_rbf, Vector((xp, yp + 0.010, zp)), Vector((xp, yp - 0.024, zp)),
        [(0.0, 0.0072), (0.12, 0.0090), (0.82, 0.0090), (0.94, 0.0070), (1.0, 0.0046)], segs=18, cap_ends=True,
    )
    bm_fill_unset_from(bm_rbf, f0_cap, MAT_RUBBER_DARK)


    f0_eye = len(bm_rbf.faces)
    bm_add_segmented_tube(bm_rbf, Vector((xp, yp - 0.018, zp - 0.006)), Vector((xp, yp - 0.018, zp - 0.016)),
                          [(0.0, 0.0042), (1.0, 0.0036)], segs=12, cap_ends=True)
    bm_fill_unset_from(bm_rbf, f0_eye, MAT_BRASS_FITTING)

    y_anc = yp - 0.018
    z_anc = zp - 0.015
    f0_kr = len(bm_rbf.faces)
    kr_pts = [
        Vector((xp + 0.0085 * math.cos(2.0 * math.pi * ik / 18.0), y_anc, z_anc - 0.009 + 0.0095 * math.sin(2.0 * math.pi * ik / 18.0)))
        for ik in range(19)
    ]
    bm_add_spline_tube(bm_rbf, kr_pts, radius=0.0012, n_samples=20, segs=10, cap_ends=False, closed=True)
    bm_fill_unset_from(bm_rbf, f0_kr, MAT_FASTENER)


    yaw_rad = math.radians(-38.0)
    n_vec0 = Vector((-math.sin(yaw_rad), -math.cos(yaw_rad), 0.0)).normalized()
    w_vec0 = Vector((math.cos(yaw_rad),  -math.sin(yaw_rad), 0.0)).normalized()
    z_vec0 = Vector((0.0, 0.0, 1.0))

    p_grom = Vector((xp, y_anc, z_anc - 0.018))
    f0_gm = len(bm_rbf.faces)
    for sn_g in (-0.0006, +0.0006):
        gm_pts = [
            p_grom + n_vec0 * sn_g + w_vec0 * (0.0036 * math.cos(2.0 * math.pi * ig / 16.0)) + z_vec0 * (0.0036 * math.sin(2.0 * math.pi * ig / 16.0))
            for ig in range(17)
        ]
        bm_add_spline_tube(bm_rbf, gm_pts, radius=0.00105, n_samples=18, segs=10, cap_ends=False, closed=True)
    bm_fill_unset_from(bm_rbf, f0_gm, MAT_BRASS_FITTING)


    L_rib = 0.162
    W_rib = 0.035
    th_half = 0.00028
    nu_r, nv_r = 24, 6
    ph_s = 2.40
    grid_f, grid_b = [], []
    for iu in range(nu_r + 1):
        u = iu / float(nu_r)
        tw_ang = math.radians(15.0 * (u ** 1.35) * math.sin(1.6 * u + ph_s))
        ca_t, sa_t = math.cos(tw_ang), math.sin(tw_ang)
        w_cur = (w_vec0 * ca_t + n_vec0 * sa_t).normalized()
        n_cur = (n_vec0 * ca_t - w_vec0 * sa_t).normalized()
        sway_n = 0.0085 * (u ** 1.30) * math.sin(3.6 * u + ph_s)
        sway_w = 0.0042 * (u ** 1.45) * math.cos(2.4 * u + ph_s)
        p_ctr = Vector((xp, y_anc, z_anc - 0.011 - L_rib * u)) + n_cur * sway_n + w_cur * sway_w
        row_f, row_b = [], []
        for iv in range(nv_r + 1):
            v = iv / float(nv_r)
            cup = 0.00075 * ((2.0 * v - 1.0) ** 2 - 0.35) * (0.25 + 0.75 * u)
            p_pt = p_ctr + w_cur * ((v - 0.5) * W_rib) + n_cur * cup
            row_f.append(bm_rbf.verts.new(p_pt + n_cur * th_half))
            row_b.append(bm_rbf.verts.new(p_pt - n_cur * th_half))
        grid_f.append(row_f)
        grid_b.append(row_b)

    def _add_uv_quad(v0, v1, v2, v3, uv0, uv1, uv2, uv3, ref_n):
        f = bm_rbf.faces.new((v0, v1, v2, v3))
        f.material_index = MAT_STAND_ORANGE
        f.smooth = True
        f.normal_update()
        if f.normal.dot(ref_n) < 0.0:
            f.normal_flip()
        uv_map = {v0: uv0, v1: uv1, v2: uv2, v3: uv3}
        for lp in f.loops:
            lp[uv_rbf].uv = uv_map[lp.vert]

    for iu in range(nu_r):
        u0, u1 = iu / float(nu_r), (iu + 1) / float(nu_r)
        for iv in range(nv_r):
            v0, v1 = iv / float(nv_r), (iv + 1) / float(nv_r)
            _add_uv_quad(
                grid_f[iu][iv], grid_f[iu + 1][iv], grid_f[iu + 1][iv + 1], grid_f[iu][iv + 1],
                (u0, v0), (u1, v0), (u1, v1), (u0, v1), n_vec0,
            )
            _add_uv_quad(
                grid_b[iu][iv + 1], grid_b[iu + 1][iv + 1], grid_b[iu + 1][iv], grid_b[iu][iv],
                (u0, 1.0 - v1), (u1, 1.0 - v1), (u1, 1.0 - v0), (u0, 1.0 - v0), -n_vec0,
            )
    for iu in range(nu_r):
        u0, u1 = iu / float(nu_r), (iu + 1) / float(nu_r)
        _add_uv_quad(grid_f[iu][0], grid_b[iu][0], grid_b[iu + 1][0], grid_f[iu + 1][0],
                     (u0, 0.01), (u0, 0.00), (u1, 0.00), (u1, 0.01), -w_vec0)
        _add_uv_quad(grid_f[iu][nv_r], grid_f[iu + 1][nv_r], grid_b[iu + 1][nv_r], grid_b[iu][nv_r],
                     (u0, 0.99), (u1, 0.99), (u1, 1.00), (u0, 1.00), w_vec0)
    for iv in range(nv_r):
        v0, v1 = iv / float(nv_r), (iv + 1) / float(nv_r)
        _add_uv_quad(grid_f[0][iv], grid_f[0][iv + 1], grid_b[0][iv + 1], grid_b[0][iv],
                     (0.01, v0), (0.01, v1), (0.00, v1), (0.00, v0), Vector((0, 0, 1)))
        _add_uv_quad(grid_f[nu_r][iv], grid_b[nu_r][iv], grid_b[nu_r][iv + 1], grid_f[nu_r][iv + 1],
                     (0.99, v0), (1.00, v0), (1.00, v1), (0.99, v1), Vector((0, 0, -1)))


def _patch_mesh_with_hardware_details(obj_name: str, builder_fn):
    import bpy
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        raise RuntimeError(f"Missing target mesh object: {obj_name}")
    me = obj.data


    bm_base = bmesh.new()
    bm_base.from_mesh(me)
    f18_l = bm_base.faces.layers.int.get("hw_detail")
    if f18_l is None:
        f18_l = bm_base.faces.layers.int.new("hw_detail")
    else:
        old_faces = [f for f in bm_base.faces if f[f18_l] == 1]
        if old_faces:
            bmesh.ops.delete(bm_base, geom=old_faces, context='FACES')
            orphan_edges = [e for e in bm_base.edges if len(e.link_faces) == 0]
            if orphan_edges:
                bmesh.ops.delete(bm_base, geom=orphan_edges, context='EDGES')
            orphan_verts = [v for v in bm_base.verts if len(v.link_faces) == 0]
            if orphan_verts:
                bmesh.ops.delete(bm_base, geom=orphan_verts, context='VERTS')


    bm_add = bmesh.new()
    builder_fn(bm_add)
    bm_apply_hard_surface_smoothing(bm_add, angle_deg=38.0)

    nq = sum(1 for f in bm_add.faces if len(f.verts) != 4)
    za = sum(1 for f in bm_add.faces if f.calc_area() < 1e-10)
    if nq != 0 or za != 0:
        bm_add.free()
        bm_base.free()
        raise RuntimeError(f"Detail patch topology error on {obj_name}: NonQuads={nq}, ZeroArea={za}")


    slot_by_name = {m.name: idx for idx, m in enumerate(me.materials) if m is not None}
    mat_id_to_slot = {}
    for f in bm_add.faces:
        mid = f.material_index
        if mid not in mat_id_to_slot:
            mname = COLOR_ID_SPECS[mid][0] if mid in COLOR_ID_SPECS else COLOR_ID_SPECS[MAT_FASTENER][0]
            if mname not in slot_by_name:
                m_obj = bpy.data.materials.get(mname) or bpy.data.materials.new(mname)
                me.materials.append(m_obj)
                slot_by_name[mname] = len(me.materials) - 1
            mat_id_to_slot[mid] = slot_by_name[mname]


    v_map = [bm_base.verts.new(v.co) for v in bm_add.verts]
    bm_base.verts.ensure_lookup_table()
    bm_add.verts.index_update()
    sharp_edge_pairs = {(min(e.verts[0].index, e.verts[1].index), max(e.verts[0].index, e.verts[1].index))
                        for e in bm_add.edges if not e.smooth}
    n_added = 0
    for f in bm_add.faces:
        try:
            nf = bm_base.faces.new(tuple(v_map[v.index] for v in f.verts))
            nf.smooth = f.smooth
            nf.material_index = mat_id_to_slot[f.material_index]
            nf[f18_l] = 1
            n_added += 1
        except ValueError:
            pass
    if sharp_edge_pairs:
        idx_of_v = {v_map[i]: i for i in range(len(v_map))}
        for e in bm_base.edges:
            i0 = idx_of_v.get(e.verts[0], -1)
            i1 = idx_of_v.get(e.verts[1], -1)
            if i0 >= 0 and i1 >= 0 and (min(i0, i1), max(i0, i1)) in sharp_edge_pairs:
                e.smooth = False

    bm_add.free()
    bm_base.normal_update()
    bm_base.to_mesh(me)
    bm_base.free()
    me.update()
    print(f"[Detail Patch] {obj_name}: +{n_added:,} quads (total={len(me.polygons):,})", flush=True)


def apply_engine_static_detail_patch():

    def _r_sh_out(yv: float) -> float:
        if yv <= -0.415:
            t_f = max(0.0, min(1.0, (-0.415 - yv) / 0.305))
            return 0.614 + 0.032 * (1.0 - (t_f ** 1.80))
        t_a = max(0.0, min(1.28, (yv - (-0.415)) / 0.360))
        return 0.646 - 0.0225 * (t_a ** 1.55)

    def _cyl_p(phi_deg: float, yv: float, rv: float) -> Vector:
        pr = math.radians(phi_deg)
        return Vector((rv * math.sin(pr), yv, rv * math.cos(pr)))

    def _cyl_n(phi_deg: float) -> Vector:
        pr = math.radians(phi_deg)
        return Vector((math.sin(pr), 0.0, math.cos(pr)))

    _patch_mesh_with_hardware_details(
        "Static_Nozzle_Outer_Shroud",
        lambda bm: add_shroud_detailing(bm, _r_sh_out, _cyl_p, _cyl_n),
    )
    _patch_mesh_with_hardware_details(
        "Fwd_02_Fan_Case_VaneLinkage_And_DEEC",
        lambda bm: add_fwd02_inlet_and_deec_detailing(bm, _cyl_p, _cyl_n),
    )
