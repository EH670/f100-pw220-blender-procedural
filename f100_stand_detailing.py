import math
import bmesh
from mathutils import Vector, Matrix

from f100_geom_primitives import *
import f100_geom_primitives as _gp

_bm_add_box_x_yz = _gp._bm_add_box_x_yz
_bm_add_annular_quad_extrusion = _gp._bm_add_annular_quad_extrusion


def _add_mig_weld_line(bm, p0: Vector, p1: Vector, radius: float = 0.0024, step_m: float = 0.011):

    dist = (p1 - p0).length
    if dist < 0.006:
        return
    n_corr = max(4, int(round(dist / step_m)))
    n_samp = max(8, min(64, n_corr * 2))
    bm_add_spline_tube(bm, [p0, p1], radius=radius, n_samples=n_samp, segs=8, cap_ends=True, corrugations=n_corr)


def _add_coil_spring_x(bm, x0: float, x1: float, y_c: float, z_c: float,
                       r_helix: float, r_wire: float, n_turns: int = 6, pts_per_turn: int = 12):

    n_tot = max(12, n_turns * pts_per_turn)
    pts = []
    for i in range(n_tot + 1):
        t = i / float(n_tot)
        th = 2.0 * math.pi * n_turns * t
        xv = x0 + (x1 - x0) * t
        pts.append(Vector((xv, y_c + r_helix * math.cos(th), z_c + r_helix * math.sin(th))))
    bm_add_spline_tube(bm, pts, radius=r_wire, n_samples=n_tot + 1, segs=8, cap_ends=True)


def add_cradle_01_rail_and_catwalk_details(bm_rails, X_RAIL, Y_FWD_ARCH, Y_AFT_ARCH, _bm_add_steel_plate):
    y_r0, y_r1 = -4.385, -0.505


    track_spans = ((-4.315, -4.065), (-3.645, -2.545), (-2.125, -0.575))
    for sx in (-1.0, +1.0):
        xc = sx * X_RAIL
        for ya_t, yb_t in track_spans:
            f0_tr = len(bm_rails.faces)

            for dx_tr in (-0.044, +0.044):
                _bm_add_steel_plate(bm_rails, xc + dx_tr - 0.022, xc + dx_tr + 0.022, ya_t, yb_t, -0.684, -0.6785, nu=2, nv=max(4, int((yb_t - ya_t) * 14)), chamfer=0.0010)
            bm_fill_unset_from(bm_rails, f0_tr, MAT_MACHINED_ALU)

            n_tb = max(2, int(round((yb_t - ya_t) / 0.135)))
            for kb in range(n_tb):
                yb_pos = ya_t + (kb + 0.5) * (yb_t - ya_t) / float(n_tb)
                for dx_tr in (-0.044, +0.044):
                    bm_add_aerospace_bolt(bm_rails, Vector((xc + dx_tr, yb_pos, -0.6785)), Vector((0, 0, 1)), head_r=0.0038, height=0.0032)


        for kr_st in range(10):
            yr_st = -4.180 + kr_st * 0.390
            for side_w in (-1.0, +1.0):
                xw_in = xc + side_w * 0.020
                xw_out = xc + side_w * 0.072
                xa_g, xb_g = min(xw_in, xw_out), max(xw_in, xw_out)

                for sy_w in (-1.0, +1.0):
                    _add_mig_weld_line(bm_rails, Vector((xa_g + 0.003, yr_st + sy_w * 0.0075, -0.883)), Vector((xa_g + 0.003, yr_st + sy_w * 0.0075, -0.775)), radius=0.0020)

                xg_a, xg_b = sorted((xw_in, xw_in + side_w * 0.028))
                for za_g, zb_g in ((-0.884, -0.873), (-0.785, -0.774)):
                    _bm_add_steel_plate(bm_rails, xg_a, xg_b, yr_st - 0.015, yr_st + 0.015, za_g, zb_g, nu=1, nv=1, chamfer=0.0008)


        for kp_h in range(20):
            yh_p = -4.210 + kp_h * 0.192
            f0_bh = len(bm_rails.faces)
            bm_add_chamfered_cyl_x(bm_rails, yh_p, -0.718, min(xc + sx * 0.024, xc + sx * 0.030), max(xc + sx * 0.024, xc + sx * 0.030),
                                   radius=0.0145, segs=18, hollow_r=0.0090)
            bm_fill_unset_from(bm_rails, f0_bh, MAT_MACHINED_ALU)
            for dy_br in (-0.0115, +0.0115):
                bm_add_dome_rivet(bm_rails, Vector((xc + sx * 0.030, yh_p + dy_br, -0.718)), Vector((sx, 0, 0)), radius=0.0014)


        for y_fk in (-3.220, -2.520):

            for sy_fg in (-1.0, +1.0):
                _bm_add_steel_plate(bm_rails, xc + sx * 0.076, xc + sx * 0.095,
                                    y_fk + sy_fg * 0.135, y_fk + sy_fg * 0.146, -0.994, -0.902, nu=1, nv=2, chamfer=0.0012)
                _add_mig_weld_line(bm_rails, Vector((xc + sx * 0.077, y_fk + sy_fg * 0.137, -0.992)),
                                   Vector((xc + sx * 0.077, y_fk + sy_fg * 0.137, -0.904)), radius=0.0024)

            f0_fkw = len(bm_rails.faces)
            _bm_add_steel_plate(bm_rails, xc - 0.076, xc + 0.076, y_fk - 0.104, y_fk + 0.104, -0.976, -0.971, nu=2, nv=2, chamfer=0.0008)
            bm_fill_unset_from(bm_rails, f0_fkw, MAT_PUMPS_OILTANK)
            for dy_kw in (-0.082, +0.082):
                bm_add_aerospace_bolt(bm_rails, Vector((xc + sx * 0.056, y_fk + dy_kw, -0.971)), Vector((0, 0, 1)), head_r=0.0034, height=0.0022)


        for y_end, dir_y in ((y_r0, -1.0), (y_r1, +1.0)):

            for dx_eb in (-0.058, +0.058):
                bm_add_bolt_pair_lockwire(
                    bm_rails,
                    Vector((xc + dx_eb, y_end + dir_y * 0.018, -0.870)),
                    Vector((xc + dx_eb, y_end + dir_y * 0.018, -0.818)),
                    Vector((0, dir_y, 0)), head_r=0.0050, strand_r=0.00050,
                )

            y_stop = y_end - dir_y * 0.038
            for dx_sg in (-0.068, +0.068):
                bm_add_clevis_plate(bm_rails, (y_stop - dir_y * 0.020, -0.684), (y_stop + dir_y * 0.014, -0.642),
                                    0.010, 0.008, xc + dx_sg - 0.005, xc + dx_sg + 0.005, arc_segs=9)

            f0_dr = len(bm_rails.faces)
            _bm_add_steel_plate(bm_rails, xc - 0.026, xc + 0.026, y_end + dir_y * 0.018, y_end + dir_y * 0.030, -0.858, -0.800, nu=2, nv=2, chamfer=0.0012)
            bm_add_chamfered_cyl_x(bm_rails, y_end + dir_y * 0.032, -0.829, xc - 0.022, xc + 0.022, radius=0.011, segs=18, hollow_r=0.0055)

            dr_pts = [
                Vector((xc - 0.018, y_end + dir_y * 0.033, -0.829)),
                Vector((xc - 0.030, y_end + dir_y * 0.044, -0.858)),
                Vector((xc - 0.022, y_end + dir_y * 0.050, -0.884)),
                Vector((xc,         y_end + dir_y * 0.052, -0.892)),
                Vector((xc + 0.022, y_end + dir_y * 0.050, -0.884)),
                Vector((xc + 0.030, y_end + dir_y * 0.044, -0.858)),
                Vector((xc + 0.018, y_end + dir_y * 0.033, -0.829)),
            ]
            bm_add_spline_tube(bm_rails, dr_pts, radius=0.0068, n_samples=24, segs=12, cap_ends=True)
            bm_fill_unset_from(bm_rails, f0_dr, MAT_PUMPS_OILTANK)
            for dz_drb in (-0.850, -0.808):
                bm_add_aerospace_bolt(bm_rails, Vector((xc - 0.018, y_end + dir_y * 0.030, dz_drb)), Vector((0, dir_y, 0)), head_r=0.0032, height=0.0030)
                bm_add_aerospace_bolt(bm_rails, Vector((xc + 0.018, y_end + dir_y * 0.030, dz_drb)), Vector((0, dir_y, 0)), head_r=0.0032, height=0.0030)


        for y_cl_s in (-3.950, -3.020, -2.180, -1.250):
            f0_clb = len(bm_rails.faces)
            _bm_add_steel_plate(bm_rails, xc + sx * 0.068, xc + sx * 0.088, y_cl_s - 0.046, y_cl_s + 0.046, -0.906, -0.882, nu=2, nv=1, chamfer=0.0012)
            bm_fill_unset_from(bm_rails, f0_clb, MAT_PUMPS_OILTANK)
            for dy_clb in (-0.028, +0.028):
                bm_add_aerospace_bolt(bm_rails, Vector((xc + sx * 0.088, y_cl_s + dy_clb, -0.894)), Vector((sx, 0, 0)), head_r=0.0044, height=0.0040)


    grate_bays = [
        (-4.202, -3.628, True,  True),
        (-3.492, -2.928, True,  True),
        (-1.992, -1.388, False, True),
        (-1.252, -0.688, True,  True),
    ]
    z_g_bot, z_g_top = -0.788, -0.768
    for ya_g, yb_g, do_pos_x, do_neg_x in grate_bays:
        for sx_g, enabled in ((+1.0, do_pos_x), (-1.0, do_neg_x)):
            if not enabled:
                continue
            xa_p = min(sx_g * 0.072, sx_g * 0.562)
            xb_p = max(sx_g * 0.072, sx_g * 0.562)

            f0_gr = len(bm_rails.faces)
            _bm_add_steel_plate(bm_rails, xa_p, xb_p, ya_g, ya_g + 0.014, z_g_bot, z_g_top, nu=4, nv=1, chamfer=0.0010)
            _bm_add_steel_plate(bm_rails, xa_p, xb_p, yb_g - 0.014, yb_g, z_g_bot, z_g_top, nu=4, nv=1, chamfer=0.0010)
            _bm_add_steel_plate(bm_rails, xa_p, xa_p + 0.014, ya_g + 0.013, yb_g - 0.013, z_g_bot, z_g_top, nu=1, nv=4, chamfer=0.0010)
            _bm_add_steel_plate(bm_rails, xb_p - 0.014, xb_p, ya_g + 0.013, yb_g - 0.013, z_g_bot, z_g_top, nu=1, nv=4, chamfer=0.0010)

            n_bars_x = 11
            for ibx in range(n_bars_x):
                x_bar = (xa_p + 0.036) + (xb_p - xa_p - 0.072) * (ibx / float(n_bars_x - 1))
                _bm_add_steel_plate(bm_rails, x_bar - 0.0024, x_bar + 0.0024, ya_g + 0.013, yb_g - 0.013, z_g_bot + 0.002, z_g_top, nu=1, nv=2, chamfer=0.0004)
            n_bars_y = 6
            for iby in range(n_bars_y):
                y_bar = (ya_g + 0.068) + (yb_g - ya_g - 0.136) * (iby / float(n_bars_y - 1))
                _bm_add_steel_plate(bm_rails, xa_p + 0.013, xb_p - 0.013, y_bar - 0.0028, y_bar + 0.0028, z_g_top - 0.006, z_g_top - 0.0008, nu=2, nv=1, chamfer=0.0004)
            bm_fill_unset_from(bm_rails, f0_gr, MAT_STAND_STEEL)

            for x_cl in (xa_p + 0.042, xb_p - 0.042):
                for y_cl in (ya_g + 0.018, yb_g - 0.018):
                    f0_gc = len(bm_rails.faces)
                    _bm_add_steel_plate(bm_rails, x_cl - 0.016, x_cl + 0.016, y_cl - 0.012, y_cl + 0.012, z_g_top - 0.001, z_g_top + 0.0045, nu=1, nv=1, chamfer=0.0006)
                    bm_fill_unset_from(bm_rails, f0_gc, MAT_MACHINED_ALU)
                    bm_add_aerospace_bolt(bm_rails, Vector((x_cl, y_cl, z_g_top + 0.0045)), Vector((0, 0, 1)), head_r=0.0038, height=0.0035)


    for sx_o in (-1.0, +1.0):
        xo_c = sx_o * 0.562
        _bm_add_steel_plate(bm_rails, min(xo_c, xo_c - sx_o * 0.024), max(xo_c, xo_c - sx_o * 0.024), -2.796, -2.124, -0.788, -0.766, nu=1, nv=6, chamfer=0.0010)
        for y_ob in (-2.720, -2.460, -2.200):
            bm_add_aerospace_bolt(bm_rails, Vector((xo_c - sx_o * 0.012, y_ob, -0.766)), Vector((0, 0, 1)), head_r=0.0036, height=0.0032)


    bx0, bx1 = +0.105, +0.525
    by0, by1 = -1.940, -1.440
    bz0, bz1 = -0.855, -0.712
    f0_fbx = len(bm_rails.faces)

    _bm_add_steel_plate(bm_rails, bx0, bx1, by0, by1, bz0, bz1, nu=5, nv=5, chamfer=0.0022)
    _bm_add_steel_plate(bm_rails, bx0 - 0.008, bx1 + 0.008, by0 - 0.008, by1 + 0.008, bz1 - 0.002, bz1 + 0.022, nu=5, nv=5, chamfer=0.0018)
    for x_rib_l in (bx0 + 0.095, bx0 + 0.210, bx0 + 0.325):
        _bm_add_steel_plate(bm_rails, x_rib_l - 0.010, x_rib_l + 0.010, by0 + 0.036, by1 - 0.036, bz1 + 0.021, bz1 + 0.028, nu=1, nv=3, chamfer=0.0010)

    for x_ft in (bx0 + 0.045, bx1 - 0.045):
        _bm_add_steel_plate(bm_rails, x_ft - 0.024, x_ft + 0.024, -2.035, by0 + 0.004, -0.770, -0.758, nu=1, nv=2, chamfer=0.0010)
        _bm_add_steel_plate(bm_rails, x_ft - 0.024, x_ft + 0.024, by1 - 0.004, -1.345, -0.770, -0.758, nu=1, nv=2, chamfer=0.0010)
    bm_fill_unset_from(bm_rails, f0_fbx, MAT_AVIONICS_BOX)

    for x_ft in (bx0 + 0.045, bx1 - 0.045):
        for y_ft in (-2.018, -1.362):
            bm_add_aerospace_bolt(bm_rails, Vector((x_ft, y_ft, -0.758)), Vector((0, 0, 1)), head_r=0.0044, height=0.0040)

    f0_fbh = len(bm_rails.faces)
    for k_hg in range(7):
        x_hg = bx0 + 0.042 + k_hg * (bx1 - bx0 - 0.084) / 6.0
        bm_add_segmented_tube(bm_rails, Vector((x_hg - 0.014, by0 - 0.011, bz1 + 0.001)), Vector((x_hg + 0.014, by0 - 0.011, bz1 + 0.001)),
                              [(0.0, 0.0055), (1.0, 0.0055)], segs=12, cap_ends=True)
    for x_lt in (bx0 + 0.085, bx1 - 0.085):
        _bm_add_steel_plate(bm_rails, x_lt - 0.016, x_lt + 0.016, by1 + 0.004, by1 + 0.015, bz1 - 0.040, bz1 + 0.014, nu=2, nv=2, chamfer=0.0008)
        bm_add_segmented_tube(bm_rails, Vector((x_lt, by1 + 0.013, bz1 - 0.012)), Vector((x_lt, by1 + 0.022, bz1 - 0.012)),
                              [(0.0, 0.0052), (1.0, 0.0048)], segs=14, cap_ends=True)

    bm_add_spline_tube(
        bm_rails,
        [
            Vector((bx0 + 0.145, by1 - 0.046, bz1 + 0.024)),
            Vector((bx0 + 0.145, by1 - 0.032, bz1 + 0.042)),
            Vector((bx1 - 0.145, by1 - 0.032, bz1 + 0.042)),
            Vector((bx1 - 0.145, by1 - 0.046, bz1 + 0.024)),
        ],
        radius=0.0038, n_samples=20, segs=12, cap_ends=True,
    )
    bm_fill_unset_from(bm_rails, f0_fbh, MAT_RIGID_TUBE)
    for x_lt in (bx0 + 0.085, bx1 - 0.085):
        bm_add_dome_rivet(bm_rails, Vector((x_lt, by1 + 0.015, bz1 - 0.032)), Vector((0, 1, 0)), radius=0.0020)
        bm_add_dome_rivet(bm_rails, Vector((x_lt, by1 + 0.015, bz1 + 0.008)), Vector((0, 1, 0)), radius=0.0020)


    xc_doc, zc_doc = -0.455, -0.730
    f0_doc = len(bm_rails.faces)
    bm_add_segmented_tube(
        bm_rails, Vector((xc_doc, -1.915, zc_doc)), Vector((xc_doc, -1.525, zc_doc)),
        [(0.0, 0.036), (0.04, 0.042), (0.94, 0.042), (1.0, 0.038)], segs=28, cap_ends=True,
    )
    bm_fill_unset_from(bm_rails, f0_doc, MAT_MACHINED_ALU)

    f0_dcap = len(bm_rails.faces)
    bm_add_segmented_tube(
        bm_rails, Vector((xc_doc, -1.528, zc_doc)), Vector((xc_doc, -1.480, zc_doc)),
        [(0.0, 0.046), (0.20, 0.048), (0.82, 0.048), (1.0, 0.042)], segs=28, cap_ends=True,
    )
    bm_fill_unset_from(bm_rails, f0_dcap, MAT_AVIONICS_BOX)
    for k_dc in range(12):
        adc = 2.0 * math.pi * k_dc / 12.0
        f0_dcr = len(bm_rails.faces)
        bm_add_segmented_tube(
            bm_rails,
            Vector((xc_doc + 0.047 * math.cos(adc), -1.520, zc_doc + 0.047 * math.sin(adc))),
            Vector((xc_doc + 0.047 * math.cos(adc), -1.488, zc_doc + 0.047 * math.sin(adc))),
            [(0.0, 0.0020), (1.0, 0.0020)], segs=8, cap_ends=True,
        )
        bm_fill_unset_from(bm_rails, f0_dcr, MAT_AVIONICS_BOX)
    for y_ds in (-1.825, -1.615):
        f0_dsb = len(bm_rails.faces)
        bm_add_segmented_tube(
            bm_rails, Vector((xc_doc, y_ds - 0.012, zc_doc)), Vector((xc_doc, y_ds + 0.012, zc_doc)),
            [(0.0, 0.0455), (1.0, 0.0455)], segs=28, cap_ends=True,
        )
        _bm_add_steel_plate(bm_rails, xc_doc - 0.048, xc_doc + 0.048, y_ds - 0.012, y_ds + 0.012, -0.772, zc_doc - 0.036, nu=2, nv=1, chamfer=0.0008)
        bm_fill_unset_from(bm_rails, f0_dsb, MAT_RIGID_TUBE)
        for dx_ds in (-0.036, +0.036):
            bm_add_aerospace_bolt(bm_rails, Vector((xc_doc + dx_ds, y_ds, zc_doc - 0.036)), Vector((0, 0, 1)), head_r=0.0032, height=0.0030)
    bm_add_twisted_lockwire(
        bm_rails,
        [Vector((xc_doc, -1.480, zc_doc)), Vector((xc_doc - 0.048, -1.505, zc_doc - 0.025)), Vector((xc_doc - 0.045, -1.615, zc_doc - 0.020))],
        strand_r=0.00075, twists_per_m=150.0, n_samples=20, segs=12,
    )


def add_cradle_02_bogie_and_pedestal_details(bm_peds, X_RAIL, Y_FWD_ARCH, Y_AFT_ARCH, _bm_add_steel_plate):
    for y_ped, is_front in ((Y_FWD_ARCH, True), (Y_AFT_ARCH, False)):
        for sx in (-1.0, +1.0):
            x_in = sx * 0.588
            x_out = sx * 0.744
            xc = 0.5 * (x_in + x_out)


            f0_ad = len(bm_peds.faces)
            for dy_hk in (-0.172, +0.172):

                _bm_add_steel_plate(bm_peds, min(x_out + sx * 0.004, x_out + sx * 0.018), max(x_out + sx * 0.004, x_out + sx * 0.018),
                                    y_ped + dy_hk - 0.016, y_ped + dy_hk + 0.016, -0.773, -0.735, nu=1, nv=2, chamfer=0.0010)
                _bm_add_steel_plate(bm_peds, min(xc + sx * 0.028, x_out + sx * 0.018), max(xc + sx * 0.028, x_out + sx * 0.018),
                                    y_ped + dy_hk - 0.016, y_ped + dy_hk + 0.016, -0.773, -0.758, nu=2, nv=1, chamfer=0.0010)
            bm_fill_unset_from(bm_peds, f0_ad, MAT_PUMPS_OILTANK)
            for dy_hk in (-0.172, +0.172):
                bm_add_aerospace_bolt(bm_peds, Vector((x_out + sx * 0.018, y_ped + dy_hk, -0.744)), Vector((sx, 0, 0)), head_r=0.0038, height=0.0035)


            for sy_wp in (-1.0, +1.0):
                y_wp = y_ped + sy_wp * 0.190
                f0_wp = len(bm_peds.faces)
                _bm_add_steel_plate(bm_peds, xc - 0.056, xc + 0.056, y_wp, y_wp + sy_wp * 0.014, -0.6815, -0.628, nu=3, nv=2, chamfer=0.0008)
                bm_fill_unset_from(bm_peds, f0_wp, MAT_STAND_STEEL)
                f0_wpb = len(bm_peds.faces)
                _bm_add_steel_plate(bm_peds, xc - 0.052, xc + 0.052, y_wp + sy_wp * 0.003, y_wp + sy_wp * 0.012, -0.6845, -0.674, nu=3, nv=1, chamfer=0.0005)
                bm_fill_unset_from(bm_peds, f0_wpb, MAT_BRASS_FITTING)
                for dx_wp in (-0.036, +0.036):
                    bm_add_aerospace_bolt(bm_peds, Vector((xc + dx_wp, y_wp + sy_wp * 0.014, -0.652)), Vector((0, sy_wp, 0)), head_r=0.0034, height=0.0032)


            for dy_rol in (-0.124, +0.124):
                for kb_pair in (0, 2, 4, 6):
                    a0 = 2.0 * math.pi * kb_pair / 8.0
                    a1 = 2.0 * math.pi * (kb_pair + 1) / 8.0
                    p0 = Vector((x_out + sx * 0.024, y_ped + dy_rol + 0.032 * math.cos(a0), -0.692 + 0.032 * math.sin(a0)))
                    p1 = Vector((x_out + sx * 0.024, y_ped + dy_rol + 0.032 * math.cos(a1), -0.692 + 0.032 * math.sin(a1)))
                    bm_add_bolt_pair_lockwire(bm_peds, p0, p1, Vector((sx, 0, 0)), head_r=0.0034, strand_r=0.00044)


            y_plg_r = y_ped + (-0.176 if is_front else +0.176)
            f0_spg = len(bm_peds.faces)
            _add_coil_spring_x(bm_peds, x_out + sx * 0.028, x_out + sx * 0.047, y_plg_r, -0.718, r_helix=0.0102, r_wire=0.0014, n_turns=5, pts_per_turn=12)
            bm_fill_unset_from(bm_peds, f0_spg, MAT_FASTENER)
            f0_spb = len(bm_peds.faces)
            bm_add_bevel_locked_hex(bm_peds, Vector((x_out + sx * 0.024, y_plg_r, -0.718)), Vector((x_out + sx * 0.031, y_plg_r, -0.718)),
                                    [(0.0, 0.015), (0.30, 0.0175), (0.80, 0.0175), (1.0, 0.014)], corner_bevel=0.14, cap_ends=True)
            bm_fill_unset_from(bm_peds, f0_spb, MAT_BRASS_FITTING)


            for sy_w in (-1.0, +1.0):
                _add_mig_weld_line(bm_peds, Vector((min(x_in, x_out) + 0.006, y_ped + sy_w * 0.125, -0.591)),
                                   Vector((max(x_in, x_out) - 0.006, y_ped + sy_w * 0.125, -0.591)), radius=0.0026)
                _add_mig_weld_line(bm_peds, Vector((min(x_in, x_out) + 0.006, y_ped + sy_w * 0.100, -0.168)),
                                   Vector((max(x_in, x_out) - 0.006, y_ped + sy_w * 0.100, -0.168)), radius=0.0026)
            for dy_pfb in (-0.168, +0.168):
                bm_add_bolt_pair_lockwire(
                    bm_peds,
                    Vector((xc - 0.054, y_ped + dy_pfb, -0.592)),
                    Vector((xc + 0.054, y_ped + dy_pfb, -0.592)),
                    Vector((0, 0, 1)), head_r=0.0048, strand_r=0.00050,
                )

            for s_bay in (-1.0, +1.0):
                bm_add_aerospace_bolt(bm_peds, Vector((xc + sx * 0.025, y_ped + s_bay * 0.052, -0.372)), Vector((0, 0, 1)), head_r=0.0036, height=0.0034)


def add_cradle_03_arch_and_trunnion_details(bm_arches, Y_FWD_ARCH, Y_AFT_ARCH, R_A_IN, R_A_OUT, R_W_IN, R_W_OUT, _bm_add_steel_plate):
    for y_ar, is_front_ar in ((Y_FWD_ARCH, True), (Y_AFT_ARCH, False)):

        for deg_r in (-88, -80, -72, -64, -56, -48, -40, -28, -16, +16, +28, +40, +48, +56, +64, +72, +80, +88):
            ar = math.radians(deg_r)
            sa, ca = math.sin(ar), math.cos(ar)
            for r_rv in (0.5 * (R_A_IN + R_W_IN), 0.5 * (R_W_OUT + R_A_OUT)):
                for sy_rv in (-1.0, +1.0):
                    bm_add_dome_rivet(
                        bm_arches,
                        Vector((r_rv * sa, y_ar + sy_rv * 0.069, r_rv * ca)),
                        Vector((0.0, sy_rv, 0.0)), radius=0.0034,
                    )

        for deg_b in (-42, -34, -26, -18, +18, +26, +34, +42):
            ar = math.radians(deg_b)
            sa, ca = math.sin(ar), math.cos(ar)
            n_out = Vector((sa, 0.0, ca))
            for dy_ab in (-0.042, +0.042):
                bm_add_aerospace_bolt(bm_arches, Vector((R_A_OUT * sa, y_ar + dy_ab, R_A_OUT * ca)), n_out, head_r=0.0038, height=0.0034)


        for s_ang in (-1.0, +1.0):
            for d_pin_s in (46.0, 68.0):
                ar_pin = math.radians(s_ang * d_pin_s)
                xp, zp = 0.643 * math.sin(ar_pin), 0.643 * math.cos(ar_pin)
                f0_pb = len(bm_arches.faces)
                for sy_p in (-1.0, +1.0):
                    bm_add_segmented_tube(bm_arches, Vector((xp, y_ar + sy_p * 0.054, zp)), Vector((xp, y_ar + sy_p * 0.066, zp)),
                                          [(0.0, 0.022), (0.80, 0.022), (1.0, 0.019)], segs=22, cap_ends=True)
                bm_fill_unset_from(bm_arches, f0_pb, MAT_BRASS_FITTING)
                f0_pn = len(bm_arches.faces)
                for sy_p in (-1.0, +1.0):
                    bm_add_bevel_locked_hex(bm_arches, Vector((xp, y_ar + sy_p * 0.064, zp)), Vector((xp, y_ar + sy_p * 0.074, zp)),
                                            [(0.0, 0.013), (0.25, 0.0165), (0.80, 0.0165), (1.0, 0.012)], corner_bevel=0.14, cap_ends=True)
                bm_fill_unset_from(bm_arches, f0_pn, MAT_FASTENER)


        for sx in (-1.0, +1.0):
            for ib_pair in (0, 2, 4, 6, 8):
                a0 = 2.0 * math.pi * ib_pair / 10.0
                a1 = 2.0 * math.pi * (ib_pair + 1) / 10.0
                p0 = Vector((sx * 0.772, y_ar + 0.044 * math.cos(a0), -0.054 + 0.044 * math.sin(a0)))
                p1 = Vector((sx * 0.772, y_ar + 0.044 * math.cos(a1), -0.054 + 0.044 * math.sin(a1)))
                bm_add_bolt_pair_lockwire(bm_arches, p0, p1, Vector((sx, 0, 0)), head_r=0.0040, strand_r=0.00046)


        z_crown = 0.736 if is_front_ar else 0.748
        y_lug_c = y_ar + (0.072 if is_front_ar else -0.072)
        for dx_ct in (-0.048, +0.048):
            for dy_ct in (-0.050, +0.050):
                bm_add_aerospace_bolt(bm_arches, Vector((dx_ct, y_ar + dy_ct, z_crown - 0.020)), Vector((0, 0, 1)), head_r=0.0046, height=0.0042)

        f0_cw = len(bm_arches.faces)
        for sx_w in (-1.0, +1.0):
            bm_add_chamfered_cyl_x(bm_arches, y_lug_c, z_crown, min(sx_w * 0.0405, sx_w * 0.0445), max(sx_w * 0.0405, sx_w * 0.0445),
                                   radius=0.023, segs=22, hollow_r=0.0142)
        bm_fill_unset_from(bm_arches, f0_cw, MAT_BRASS_FITTING)


def add_cradle_04_turnbuckle_and_hoist_details(bm_bars, Y_FWD_ARCH, Y_AFT_ARCH, _bm_add_steel_plate):

    p_top_f = Vector((0.0, Y_FWD_ARCH + 0.072, 0.736))
    p_top_a = Vector((0.0, Y_AFT_ARCH - 0.072, 0.748))
    p_mid = p_top_f.lerp(p_top_a, 0.50)
    dir_top = (p_top_a - p_top_f).normalized()

    f0_hc = len(bm_bars.faces)
    bm_add_segmented_tube(bm_bars, p_mid - dir_top * 0.056, p_mid + dir_top * 0.056,
                          [(0.0, 0.048), (0.14, 0.054), (0.86, 0.054), (1.0, 0.048)], segs=32, cap_ends=True)

    for sx_h in (-0.016, +0.016):
        bm_add_clevis_plate(bm_bars, (p_mid.y, p_mid.z + 0.046), (p_mid.y, p_mid.z + 0.118), 0.036, 0.030, sx_h - 0.008, sx_h + 0.008, arc_segs=15)
    bm_add_clevis_pin_assembly_x(bm_bars, p_mid.y, p_mid.z + 0.118, -0.036, +0.036, pin_r=0.013, segs=22, compact_ends=True)
    bm_fill_unset_from(bm_bars, f0_hc, MAT_PUMPS_OILTANK)

    for sx_e in (-1.0, +1.0):
        _bm_add_steel_plate(bm_bars, min(sx_e * 0.048, sx_e * 0.074), max(sx_e * 0.048, sx_e * 0.074),
                            p_mid.y - 0.046, p_mid.y + 0.046, p_mid.z - 0.016, p_mid.z + 0.016, nu=2, nv=1, chamfer=0.0012)
        bpts_h = []
        for dy_hb in (-0.028, +0.028):
            pb = Vector((sx_e * 0.062, p_mid.y + dy_hb, p_mid.z + 0.016))
            bpts_h.append(pb)
            bm_add_aerospace_bolt(bm_bars, pb, Vector((0, 0, 1)), head_r=0.0044, height=0.0042)
        bm_add_bolt_pair_lockwire(bm_bars, bpts_h[0], bpts_h[1], Vector((0, 0, 1)), head_r=0.0044, strand_r=0.00048)


    for sx in (-1.0, +1.0):
        xb = sx * 0.700
        p_df = Vector((xb, Y_FWD_ARCH + 0.176, -0.538))
        p_da = Vector((xb, Y_AFT_ARCH - 0.166, -0.068))
        dir_d = (p_da - p_df).normalized()
        p_dm = p_df.lerp(p_da, 0.50)


        f0_tb = len(bm_bars.faces)
        bm_add_segmented_tube(
            bm_bars, p_dm - dir_d * 0.118, p_dm + dir_d * 0.118,
            [
                (0.00, 0.036), (0.025, 0.044), (0.19, 0.044), (0.21, 0.040),
                (0.79, 0.040), (0.81, 0.044), (0.975, 0.044), (1.00, 0.036),
            ],
            segs=32, cap_ends=True,
        )
        bm_add_bevel_locked_hex(
            bm_bars, p_dm - dir_d * 0.042, p_dm + dir_d * 0.042,
            [(0.0, 0.042), (0.14, 0.049), (0.86, 0.049), (1.0, 0.042)],
            corner_bevel=0.12, cap_ends=True,
        )
        bm_fill_unset_from(bm_bars, f0_tb, MAT_STAND_STEEL)

        f0_tbh = len(bm_bars.faces)
        for s_end in (-1.0, +1.0):
            p_t0 = p_dm + dir_d * (s_end * 0.114)
            p_t1 = p_dm + dir_d * (s_end * 0.214)
            bm_add_helical_thread_cylinder(bm_bars, p_t0, p_t1, r_core=0.0255, thread_h=0.0032, n_turns=8, segs_turn=22, cap_core=True)
            p_n0 = p_dm + dir_d * (s_end * 0.120)
            p_n1 = p_dm + dir_d * (s_end * 0.148)
            bm_add_bevel_locked_hex(bm_bars, p_n0, p_n1, [(0.0, 0.038), (0.20, 0.045), (0.80, 0.045), (1.0, 0.038)], corner_bevel=0.14, cap_ends=True)
        bm_fill_unset_from(bm_bars, f0_tbh, MAT_FASTENER)

        f0_whr = len(bm_bars.faces)
        bm_add_chamfered_cyl_x(bm_bars, p_dm.y, p_dm.z, xb - 0.047, xb + 0.047, radius=0.0115, segs=20, hollow_r=0.0068)
        bm_fill_unset_from(bm_bars, f0_whr, MAT_FASTENER)
        f0_whd = len(bm_bars.faces)
        bm_add_chamfered_cyl_x(bm_bars, p_dm.y, p_dm.z, xb - 0.044, xb + 0.044, radius=0.0068, segs=16)
        bm_fill_unset_from(bm_bars, f0_whd, MAT_AVIONICS_BOX)

        bm_add_twisted_lockwire(
            bm_bars,
            [
                p_dm - dir_d * 0.134 + Vector((sx * 0.044, 0, 0)),
                p_dm - dir_d * 0.055 + Vector((sx * 0.047, 0, 0)),
                p_dm + Vector((sx * 0.049, 0, 0)),
                p_dm + dir_d * 0.055 + Vector((sx * 0.047, 0, 0)),
                p_dm + dir_d * 0.134 + Vector((sx * 0.044, 0, 0)),
            ],
            strand_r=0.00080, twists_per_m=160.0, n_samples=32, segs=12,
        )


def add_trailer_05_surge_brake_assembly(bm_tr, _bm_add_steel_plate):

    f0_sbb = len(bm_tr.faces)
    _bm_add_steel_plate(bm_tr, -0.048, +0.048, -4.875, -4.620, -1.324, -1.308, nu=3, nv=4, chamfer=0.0014)

    bm_add_segmented_tube(
        bm_tr, Vector((0.0, -4.865, -1.288)), Vector((0.0, -4.640, -1.288)),
        [
            (0.00, 0.020), (0.06, 0.026), (0.18, 0.026), (0.21, 0.030), (0.27, 0.030), (0.30, 0.026),
            (0.42, 0.026), (0.45, 0.030), (0.51, 0.030), (0.54, 0.026),
            (0.66, 0.026), (0.69, 0.030), (0.75, 0.030), (0.78, 0.026),
            (0.92, 0.028), (1.00, 0.022),
        ],
        segs=28, cap_ends=True,
    )

    _bm_add_steel_plate(bm_tr, -0.034, +0.034, -4.822, -4.692, -1.278, -1.248, nu=3, nv=3, chamfer=0.0018)
    _bm_add_steel_plate(bm_tr, -0.038, +0.038, -4.826, -4.688, -1.249, -1.241, nu=3, nv=3, chamfer=0.0012)
    bm_fill_unset_from(bm_tr, f0_sbb, MAT_AVIONICS_BOX)


    for sx_mb in (-0.038, +0.038):
        p_a = Vector((sx_mb, -4.858, -1.308))
        p_b = Vector((sx_mb, -4.636, -1.308))
        bm_add_aerospace_bolt(bm_tr, p_a, Vector((0, 0, 1)), head_r=0.0040, height=0.0038)
        bm_add_aerospace_bolt(bm_tr, p_b, Vector((0, 0, 1)), head_r=0.0040, height=0.0038)
    for sx_tc in (-0.028, +0.028):
        p_c0 = Vector((sx_tc, -4.812, -1.241))
        p_c1 = Vector((sx_tc, -4.702, -1.241))
        bm_add_aerospace_bolt(bm_tr, p_c0, Vector((0, 0, 1)), head_r=0.0030, height=0.0028)
        bm_add_aerospace_bolt(bm_tr, p_c1, Vector((0, 0, 1)), head_r=0.0030, height=0.0028)
        bm_add_bolt_pair_lockwire(bm_tr, p_c0, p_c1, Vector((0, 0, 1)), head_r=0.0030, strand_r=0.00042)


    f0_sbbr = len(bm_tr.faces)
    bm_add_bevel_locked_hex(bm_tr, Vector((0.0, -4.756, -1.241)), Vector((0.0, -4.756, -1.226)),
                            [(0.0, 0.015), (0.30, 0.0185), (0.80, 0.0185), (1.0, 0.014)], corner_bevel=0.14, cap_ends=True)
    bm_add_grease_zerk(bm_tr, Vector((0.0, -4.756, -1.226)), Vector((0, 0, 1)), scale=1.0)

    bm_add_chamfered_cyl_x(bm_tr, -4.756, -1.263, +0.033, +0.039, radius=0.0095, segs=20, hollow_r=0.0062)
    bm_add_chamfered_cyl_x(bm_tr, -4.756, -1.263, -0.039, -0.033, radius=0.0095, segs=20, hollow_r=0.0062)

    bm_add_bevel_locked_hex(bm_tr, Vector((0.0, -4.640, -1.288)), Vector((0.0, -4.622, -1.288)),
                            [(0.0, 0.010), (0.30, 0.012), (0.80, 0.012), (1.0, 0.009)], corner_bevel=0.14, cap_ends=True)
    bm_fill_unset_from(bm_tr, f0_sbbr, MAT_BRASS_FITTING)


    f0_sbbt = len(bm_tr.faces)
    bm_add_spline_tube(bm_tr, [Vector((0.0, -4.936, -1.288)), Vector((0.0, -4.864, -1.288))],
                       radius=0.019, n_samples=18, segs=18, cap_ends=True, corrugations=6)
    bm_fill_unset_from(bm_tr, f0_sbbt, MAT_RUBBER_DARK)
    f0_sbp = len(bm_tr.faces)
    bm_add_segmented_tube(bm_tr, Vector((0.0, -4.955, -1.320)), Vector((0.0, -4.934, -1.288)),
                          [(0.0, 0.009), (1.0, 0.009)], segs=14, cap_ends=True)
    bm_fill_unset_from(bm_tr, f0_sbp, MAT_STAND_STEEL)


    f0_sbl = len(bm_tr.faces)
    bm_add_chamfered_cyl_x(bm_tr, -4.668, -1.280, -0.018, -0.011, radius=0.022, segs=22)
    bm_add_clevis_plate(bm_tr, (-4.668, -1.280), (-4.636, -1.186), 0.011, 0.0085, -0.006, +0.006, arc_segs=11)
    bm_add_segmented_tube(bm_tr, Vector((+0.009, -4.660, -1.252)), Vector((+0.009, -4.640, -1.196)),
                          [(0.0, 0.0028), (1.0, 0.0028)], segs=10, cap_ends=True)
    bm_fill_unset_from(bm_tr, f0_sbl, MAT_STAND_STEEL)
    f0_sbh = len(bm_tr.faces)
    bm_add_chamfered_cyl_x(bm_tr, -4.636, -1.186, -0.028, +0.028, radius=0.0090, segs=18)
    bm_fill_unset_from(bm_tr, f0_sbh, MAT_RUBBER_DARK)

    bm_add_twisted_lockwire(
        bm_tr,
        [
            Vector((-0.008, -4.652, -1.230)),
            Vector((-0.045, -4.745, -1.264)),
            Vector((-0.062, -4.855, -1.302)),
            Vector((-0.068, -4.935, -1.342)),
            Vector((-0.068, -4.956, -1.366)),
        ],
        strand_r=0.0014, twists_per_m=95.0, n_samples=28, segs=12,
    )


    for sx_hk in (-1.0, +1.0):
        f0_shk = len(bm_tr.faces)
        hk_pts = [
            Vector((sx_hk * 0.068, -4.962, -1.368)),
            Vector((sx_hk * 0.082, -4.972, -1.384)),
            Vector((sx_hk * 0.094, -4.988, -1.404)),
            Vector((sx_hk * 0.090, -4.994, -1.422)),
            Vector((sx_hk * 0.074, -4.984, -1.420)),
            Vector((sx_hk * 0.072, -4.976, -1.406)),
        ]
        bm_add_spline_tube(bm_tr, hk_pts, radius=0.0052, n_samples=20, segs=12, cap_ends=True)
        bm_fill_unset_from(bm_tr, f0_shk, MAT_PUMPS_OILTANK)

        f0_kpr = len(bm_tr.faces)
        bm_add_spline_tube(
            bm_tr,
            [Vector((sx_hk * 0.082, -4.974, -1.386)), Vector((sx_hk * 0.073, -4.977, -1.406))],
            radius=0.0018, n_samples=8, segs=8, cap_ends=True,
        )
        bm_fill_unset_from(bm_tr, f0_kpr, MAT_FASTENER)

    for p_t, t_dir, p_b in (
        (Vector((+0.110, -4.545, -1.325)), Vector((+0.250, +0.968, 0.0)), Vector((+0.110, -4.545, -1.332))),
        (Vector((+0.159, -4.350, -1.322)), Vector((+0.207, +0.978, 0.0)), Vector((+0.159, -4.350, -1.329))),
    ):
        bm_add_adel_loop_clamp(bm_tr, p_t, t_dir.normalized(), p_b, Vector((0.0, 0.0, 1.0)), tube_r=0.0046)


def add_trailer_05_chassis_details(bm_tr, X_RAIL, Z_AXLE, _bm_add_steel_plate):

    for sx in (-1.0, +1.0):
        xc = sx * X_RAIL
        for y_twl in (-4.140, -3.520, -1.880, -0.940):

            f0_twc = len(bm_tr.faces)
            _bm_add_steel_plate(bm_tr, min(xc + sx * 0.078, xc + sx * 0.094), max(xc + sx * 0.078, xc + sx * 0.094),
                                y_twl - 0.042, y_twl + 0.042, -0.888, -0.876, nu=2, nv=1, chamfer=0.0010)
            bm_add_chamfered_cyl_x(bm_tr, y_twl, -0.914, min(xc + sx * 0.088, xc + sx * 0.100), max(xc + sx * 0.088, xc + sx * 0.100),
                                   radius=0.011, segs=18)
            bm_add_clevis_plate(bm_tr, (y_twl, -0.914), (y_twl + 0.034, -0.934), 0.007, 0.0055,
                                min(xc + sx * 0.094, xc + sx * 0.101), max(xc + sx * 0.094, xc + sx * 0.101), arc_segs=9)
            bm_fill_unset_from(bm_tr, f0_twc, MAT_STAND_STEEL)

            bm_add_bolt_pair_lockwire(
                bm_tr,
                Vector((xc + sx * 0.0896, y_twl - 0.034, -0.914)),
                Vector((xc + sx * 0.0896, y_twl + 0.034, -0.914)),
                Vector((sx, 0, 0)), head_r=0.0052, strand_r=0.00050,
            )


        x_t = sx * 0.616
        x_w = sx * 0.638
        for z_ln, r_ln, mat_ln in ((-0.946, 0.0036, MAT_RIGID_TUBE), (-0.968, 0.0044, MAT_BRAID_HOSE)):
            f0_hl = len(bm_tr.faces)
            bm_add_spline_tube(
                bm_tr,
                [Vector((x_t, -4.160, z_ln)), Vector((x_t, -3.350, z_ln)), Vector((x_t, -2.500, z_ln)), Vector((x_t, -1.650, z_ln)), Vector((x_t, -0.900, z_ln))],
                radius=r_ln, n_samples=44, segs=14, cap_ends=True, corrugations=(42 if mat_ln == MAT_BRAID_HOSE else 0),
            )
            bm_fill_unset_from(bm_tr, f0_hl, mat_ln)

            for y_ad in (-4.020, -3.620, -3.220, -2.820, -2.420, -2.020, -1.620, -1.220):
                bm_add_adel_loop_clamp(bm_tr, Vector((x_t, y_ad, z_ln)), Vector((0.0, 1.0, 0.0)),
                                       Vector((x_w, y_ad, z_ln)), Vector((-sx, 0.0, 0.0)), tube_r=r_ln)

            for y_bn in (-3.020, -1.820):
                bm_add_bnut_flare_fitting(bm_tr, Vector((x_t, y_bn, z_ln)), Vector((0.0, 1.0, 0.0)), tube_r=r_ln)


    x_pv, y_pv, z_pv = +0.490, -3.230, -0.956
    f0_pv = len(bm_tr.faces)
    _bm_add_steel_plate(bm_tr, x_pv - 0.024, x_pv + 0.024, y_pv - 0.028, y_pv + 0.028, z_pv - 0.024, z_pv + 0.024, nu=2, nv=2, chamfer=0.0012)
    bm_fill_unset_from(bm_tr, f0_pv, MAT_BRASS_FITTING)
    for dy_pvb in (-0.018, +0.018):
        bm_add_aerospace_bolt(bm_tr, Vector((x_pv, y_pv + dy_pvb, z_pv + 0.024)), Vector((0, 0, 1)), head_r=0.0032, height=0.0030)

    f0_pg = len(bm_tr.faces)
    bm_add_segmented_tube(bm_tr, Vector((x_pv, y_pv + 0.028, z_pv)), Vector((x_pv, y_pv + 0.052, z_pv)),
                          [(0.0, 0.008), (0.35, 0.008), (0.42, 0.024), (0.94, 0.024), (1.0, 0.021)], segs=24, cap_ends=True)
    bm_fill_unset_from(bm_tr, f0_pg, MAT_RIGID_TUBE)
    bm_add_bnut_flare_fitting(bm_tr, Vector((x_pv + 0.024, y_pv, z_pv + 0.010)), Vector((1.0, 0.0, 0.0)), tube_r=0.0036)
    bm_add_bnut_flare_fitting(bm_tr, Vector((x_pv + 0.024, y_pv, z_pv - 0.010)), Vector((1.0, 0.0, 0.0)), tube_r=0.0036)
    f0_pvt = len(bm_tr.faces)
    bm_add_spline_tube(bm_tr, [Vector((x_pv + 0.024, y_pv, z_pv + 0.010)), Vector((+0.570, y_pv, z_pv + 0.010)), Vector((+0.616, y_pv + 0.030, -0.946))],
                       radius=0.0034, n_samples=16, segs=12, cap_ends=True)
    bm_fill_unset_from(bm_tr, f0_pvt, MAT_RIGID_TUBE)


    bm_add_turnbuckle_rod(bm_tr, Vector((+0.245, -2.030, -1.030)), Vector((+0.120, -1.835, -1.132)),
                          rod_r=0.0046, eye_r=0.0085, eye_half_w=0.0040, n_threads=7, segs=20, with_lockwire=True)

    f0_pl = len(bm_tr.faces)
    bm_add_chamfered_cyl_x(bm_tr, -1.750, -1.235, -0.022, +0.022, radius=0.014, segs=20, hollow_r=0.0055)
    bm_fill_unset_from(bm_tr, f0_pl, MAT_BRASS_FITTING)
    for sx_bt in (-1.0, +1.0):
        f0_rbt = len(bm_tr.faces)
        bm_add_spline_tube(bm_tr, [Vector((sx_bt * 0.090, -1.716, -1.235)), Vector((sx_bt * 0.145, -1.684, -1.236))],
                           radius=0.0068, n_samples=14, segs=14, cap_ends=True, corrugations=5)
        bm_fill_unset_from(bm_tr, f0_rbt, MAT_RUBBER_DARK)


    for sx in (-1.0, +1.0):
        xc = sx * X_RAIL
        for y_jk in (-4.210, -0.840):
            xj_c = xc + sx * 0.114

            f0_jgb = len(bm_tr.faces)
            _bm_add_steel_plate(bm_tr, xj_c - 0.042, xj_c + 0.042, y_jk - 0.042, y_jk + 0.042, -0.896, -0.850, nu=3, nv=3, chamfer=0.0020)
            bm_add_chamfered_cyl_x(bm_tr, y_jk, -0.872, min(xj_c + sx * 0.036, xj_c + sx * 0.048), max(xj_c + sx * 0.036, xj_c + sx * 0.048),
                                   radius=0.022, segs=22)
            bm_fill_unset_from(bm_tr, f0_jgb, MAT_AVIONICS_BOX)
            for k_sb in range(4):
                asb = 0.25 * math.pi + 0.5 * math.pi * k_sb
                bm_add_aerospace_bolt(bm_tr, Vector((xj_c + sx * 0.048, y_jk + 0.0155 * math.cos(asb), -0.872 + 0.0155 * math.sin(asb))),
                                      Vector((sx, 0, 0)), head_r=0.0026, height=0.0024)

            for z_cl in (-0.996, -0.924):
                bm_add_segmented_tube(bm_tr, Vector((xj_c, y_jk, z_cl - 0.012)), Vector((xj_c, y_jk, z_cl + 0.012)),
                                      [(0.0, 0.043), (1.0, 0.043)], segs=26, cap_ends=True)
            for sy_jg in (-1.0, +1.0):
                _add_mig_weld_line(bm_tr, Vector((xc + sx * 0.082, y_jk + sy_jg * 0.048, -1.006)),
                                   Vector((xc + sx * 0.082, y_jk + sy_jg * 0.048, -0.914)), radius=0.0024)

            f0_jwg = len(bm_tr.faces)
            bm_add_segmented_tube(bm_tr, Vector((xj_c, y_jk, -1.124)), Vector((xj_c, y_jk, -1.110)),
                                  [(0.0, 0.031), (0.25, 0.039), (0.85, 0.039), (1.0, 0.036)], segs=24, cap_ends=True)
            bm_fill_unset_from(bm_tr, f0_jwg, MAT_STAND_STEEL)

            f0_jrp = len(bm_tr.faces)
            bm_add_segmented_tube(bm_tr, Vector((xj_c, y_jk, -1.193)), Vector((xj_c, y_jk, -1.187)),
                                  [(0.0, 0.059), (0.80, 0.061), (1.0, 0.059)], segs=28, cap_ends=True)
            bm_fill_unset_from(bm_tr, f0_jrp, MAT_RUBBER_DARK)


    xc_neg = -X_RAIL
    xw_in, xw_out = xc_neg - 0.172, xc_neg - 0.082
    f0_csr = len(bm_tr.faces)

    bm_add_chamfered_cyl_x(bm_tr, -2.748, -0.970, xw_in + 0.008, xw_out - 0.008, radius=0.0115, segs=18, hollow_r=0.0072)

    strap_pts = [
        Vector((xw_in + 0.045, -2.838, -0.954)),
        Vector((xw_in + 0.045, -2.815, -0.960)),
        Vector((xw_in + 0.045, -2.748, -0.920)),
        Vector((xw_in + 0.045, -2.681, -0.960)),
        Vector((xw_in + 0.045, -2.658, -0.954)),
    ]
    bm_add_spline_tube(bm_tr, strap_pts, radius=0.0038, n_samples=22, segs=12, cap_ends=True)
    for y_cl_c in (-2.836, -2.660):
        _bm_add_steel_plate(bm_tr, xw_in + 0.032, xw_in + 0.058, y_cl_c - 0.009, y_cl_c + 0.009, -0.976, -0.950, nu=1, nv=1, chamfer=0.0006)
    bm_fill_unset_from(bm_tr, f0_csr, MAT_RIGID_TUBE)
    for y_cl_c in (-2.836, -2.660):
        bm_add_aerospace_bolt(bm_tr, Vector((xw_in + 0.045, y_cl_c, -0.950)), Vector((0, 0, 1)), head_r=0.0032, height=0.0030)


    for y_ax, is_front_ax in ((-3.880, True), (-1.495, False)):
        sy_ax = -1.0 if is_front_ax else +1.0

        for sx_k in (-1.0, +1.0):
            f0_kb = len(bm_tr.faces)
            bm_add_segmented_tube(bm_tr, Vector((sx_k * 0.462, y_ax, -1.264)), Vector((sx_k * 0.462, y_ax, -1.254)),
                                  [(0.0, 0.026), (1.0, 0.026)], segs=22, cap_ends=True)
            bm_add_segmented_tube(bm_tr, Vector((sx_k * 0.462, y_ax, -1.404)), Vector((sx_k * 0.462, y_ax, -1.394)),
                                  [(0.0, 0.026), (1.0, 0.026)], segs=22, cap_ends=True)
            bm_fill_unset_from(bm_tr, f0_kb, MAT_BRASS_FITTING)

            f0_tb_bt = len(bm_tr.faces)
            z_tr_o = -1.328 if sx_k < 0 else -1.364
            bm_add_segmented_tube(bm_tr, Vector((sx_k * 0.474, y_ax + sy_ax * 0.088, z_tr_o - 0.018)),
                                  Vector((sx_k * 0.474, y_ax + sy_ax * 0.088, z_tr_o + 0.018)),
                                  [(0.0, 0.014), (0.35, 0.022), (0.65, 0.022), (1.0, 0.014)], segs=18, cap_ends=True)
            bm_fill_unset_from(bm_tr, f0_tb_bt, MAT_RUBBER_DARK)

            bm_add_aerospace_bolt(bm_tr, Vector((sx_k * 0.440, y_ax - sy_ax * 0.048, -1.320)), Vector((0, -sy_ax, 0)), head_r=0.0052, height=0.0065)

        y_box_c = y_ax + sy_ax * 0.118
        z_box_t = -1.248 if is_front_ax else -1.252
        for sx_sb in (-0.068, +0.068):
            p0 = Vector((sx_sb, y_box_c - 0.042, z_box_t))
            p1 = Vector((sx_sb, y_box_c + 0.042, z_box_t))
            bm_add_aerospace_bolt(bm_tr, p0, Vector((0, 0, 1)), head_r=0.0038, height=0.0035)
            bm_add_aerospace_bolt(bm_tr, p1, Vector((0, 0, 1)), head_r=0.0038, height=0.0035)
            bm_add_bolt_pair_lockwire(bm_tr, p0, p1, Vector((0, 0, 1)), head_r=0.0038, strand_r=0.00046)


def add_trailer_06_wheel_hub_and_rim_details(b_wunit, _bm_add_steel_plate):

    for ib_w in range(8):
        aw = 2.0 * math.pi * ib_w / 8.0
        ca, sa = math.cos(aw), math.sin(aw)
        yw, zw = 0.082 * ca, 0.082 * sa

        f0_st = len(b_wunit.faces)
        bm_add_helical_thread_cylinder(
            b_wunit, Vector((+0.032, yw, zw)), Vector((+0.042, yw, zw)),
            r_core=0.0052, thread_h=0.0010, n_turns=4, segs_turn=14, cap_core=True,
        )
        bm_fill_unset_from(b_wunit, f0_st, MAT_THREAD)

        dir_tan = 1.0 if (ib_w % 2 == 0) else -1.0
        ty_d, tz_d = -sa * dir_tan, ca * dir_tan
        f0_ind = len(b_wunit.faces)
        bm_add_clevis_plate(
            b_wunit,
            (yw, zw),
            (yw + ty_d * 0.022, zw + tz_d * 0.022),
            0.0096, 0.0022, +0.0272, +0.0315, arc_segs=9,
        )
        bm_fill_unset_from(b_wunit, f0_ind, MAT_STAND_ORANGE)


    f0_rw = len(b_wunit.faces)
    for ib_r in range(10):
        ar_b = 2.0 * math.pi * ib_r / 10.0
        bm_add_chamfered_cyl_x(b_wunit, 0.128 * math.cos(ar_b), 0.128 * math.sin(ar_b), +0.0268, +0.0286,
                               radius=0.0094, segs=18, hollow_r=0.0055)
    bm_fill_unset_from(b_wunit, f0_rw, MAT_FASTENER)
    for ib_pair in (0, 2, 4, 6, 8):
        a0 = 2.0 * math.pi * ib_pair / 10.0
        a1 = 2.0 * math.pi * (ib_pair + 1) / 10.0
        p0 = Vector((+0.027, 0.128 * math.cos(a0), 0.128 * math.sin(a0)))
        p1 = Vector((+0.027, 0.128 * math.cos(a1), 0.128 * math.sin(a1)))
        bm_add_bolt_pair_lockwire(b_wunit, p0, p1, Vector((1, 0, 0)), head_r=0.0060, strand_r=0.00050)


    for ib_hp in (0, 2, 4):
        a0 = 2.0 * math.pi * ib_hp / 6.0
        a1 = 2.0 * math.pi * (ib_hp + 1) / 6.0
        p0 = Vector((+0.118, 0.045 * math.cos(a0), 0.045 * math.sin(a0)))
        p1 = Vector((+0.118, 0.045 * math.cos(a1), 0.045 * math.sin(a1)))
        bm_add_bolt_pair_lockwire(b_wunit, p0, p1, Vector((1, 0, 0)), head_r=0.0048, strand_r=0.00046)
    f0_hhex = len(b_wunit.faces)
    bm_add_bevel_locked_hex(b_wunit, Vector((+0.116, 0.0, 0.0)), Vector((+0.126, 0.0, 0.0)),
                            [(0.0, 0.020), (0.25, 0.024), (0.80, 0.024), (1.0, 0.019)], corner_bevel=0.14, cap_ends=True)
    bm_fill_unset_from(b_wunit, f0_hhex, MAT_STAND_STEEL)


    f0_vg = len(b_wunit.faces)
    for dy_vg in (-0.014, +0.014):
        _bm_add_steel_plate(b_wunit, +0.027, +0.062, dy_vg - 0.0025, dy_vg + 0.0025, +0.126, +0.150, nu=1, nv=1, chamfer=0.0006)
    bm_fill_unset_from(b_wunit, f0_vg, MAT_TRAILER_YELLOW)


    for ib_d in range(8):
        ad = 2.0 * math.pi * ib_d / 8.0
        bm_add_aerospace_bolt(b_wunit, Vector((-0.148, 0.118 * math.cos(ad), 0.118 * math.sin(ad))),
                              Vector((-1, 0, 0)), head_r=0.0042, height=0.0038)
    f0_bp = len(b_wunit.faces)
    _bm_add_steel_plate(b_wunit, -0.153, -0.147, -0.016, +0.016, -0.106, -0.094, nu=1, nv=1, chamfer=0.0008)
    bm_fill_unset_from(b_wunit, f0_bp, MAT_RUBBER_DARK)
