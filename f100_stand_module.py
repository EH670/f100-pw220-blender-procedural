import math
import bpy
import bmesh
from mathutils import Vector, Matrix

from f100_geom_primitives import *
import f100_geom_primitives as _gp
import f100_stand_detailing as _sd_stand
import f100_hardware_detailing as _hd_detail

_bm_emit_bevel_stroke_quad_prism = _gp._bm_emit_bevel_stroke_quad_prism
_bm_add_annular_quad_extrusion = _gp._bm_add_annular_quad_extrusion
_bm_add_box_x_yz = _gp._bm_add_box_x_yz

def build_maintenance_stand_and_trailer(mats: dict, coll_stand):

    clay = mats["clay"]
    X_RAIL = 0.664
    Y_FWD_ARCH = -3.856
    Y_AFT_ARCH = -2.333

    t_rad_std = [0.00, 0.04, 0.14, 0.50, 0.86, 0.96, 1.00]
    t_thk_std = [0.00, 0.06, 0.18, 0.82, 0.94, 1.00]

    def _bm_add_steel_plate(bm_dst: bmesh.types.BMesh, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float,
                            nu: int = 3, nv: int = 3, chamfer: float = 0.0018):

        dx = max(abs(x1 - x0), 1e-5)
        dy = max(abs(y1 - y0), 1e-5)
        dz = max(abs(z1 - z0), 1e-5)
        if dx <= dy and dx <= dz:
            nx, ny, nz = 1, max(2, nu), max(2, nv)
        elif dy <= dx and dy <= dz:
            nx, ny, nz = max(2, nu), 1, max(2, nv)
        else:
            nx, ny, nz = max(2, nu), max(2, nv), 1
        bm_add_chamfered_box(bm_dst, x0, x1, y0, y1, z0, z1, nx=nx, ny=ny, nz=nz, chamfer=chamfer)


    bm_rails = bmesh.new()
    y_r0, y_r1 = -4.385, -0.505

    for sx in (-1.0, +1.0):
        xc = sx * X_RAIL

        _bm_add_steel_plate(bm_rails, xc - 0.078, xc + 0.078, y_r0, y_r1, -0.906, -0.884, nu=6, nv=36)
        _bm_add_steel_plate(bm_rails, xc - 0.078, xc + 0.078, y_r0, y_r1, -0.774, -0.752, nu=6, nv=36)
        _bm_add_steel_plate(bm_rails, xc - 0.020, xc + 0.020, y_r0 + 0.014, y_r1 - 0.014, -0.885, -0.773, nu=36, nv=4)


        for kr_st in range(10):
            yr_st = -4.180 + kr_st * 0.390
            for side_w in (-1.0, +1.0):
                xw_a = xc + side_w * 0.019
                xw_b = xc + side_w * 0.074
                _bm_add_steel_plate(bm_rails, xw_a, xw_b, yr_st - 0.007, yr_st + 0.007, -0.885, -0.773, nu=2, nv=2, chamfer=0.0014)


        _bm_add_steel_plate(bm_rails, xc - 0.074, xc + 0.074, y_r0 + 0.014, y_r1 - 0.014, -0.752, -0.733, nu=6, nv=36)
        _bm_add_steel_plate(bm_rails, xc - 0.074, xc + 0.074, y_r0 + 0.014, y_r1 - 0.014, -0.703, -0.684, nu=6, nv=36)
        _bm_add_steel_plate(bm_rails, xc - 0.024, xc + 0.024, y_r0 + 0.016, y_r1 - 0.016, -0.734, -0.702, nu=36, nv=3)


        for kp_h in range(20):
            yh_p = -4.210 + kp_h * 0.192
            bm_add_chamfered_cyl_x(bm_rails, yh_p, -0.718, min(xc + sx * 0.023, xc + sx * 0.036), max(xc + sx * 0.023, xc + sx * 0.036),
                                   radius=0.0092, segs=18, hollow_r=0.0050)

        for (ya_pl, yb_pl) in ((-4.060, -3.780), (-1.940, -1.680), (-1.320, -1.110)):
            _bm_add_steel_plate(bm_rails, xc + sx * 0.019, xc + sx * 0.0745, ya_pl + 0.003, yb_pl - 0.003, -0.864, -0.794, nu=3, nv=2, chamfer=0.0010)
            f0_rpl = len(bm_rails.faces)
            _bm_add_steel_plate(bm_rails, xc + sx * 0.074, xc + sx * 0.078, ya_pl, yb_pl, -0.862, -0.796, nu=3, nv=2, chamfer=0.0006)
            bm_fill_unset_from(bm_rails, f0_rpl, MAT_PLATE_BASE)
            for sy_rv in (ya_pl + 0.010, yb_pl - 0.010):
                for sz_rv in (-0.854, -0.804):
                    bm_add_dome_rivet(bm_rails, Vector((xc + sx * 0.078, sy_rv, sz_rv)), Vector((sx, 0, 0)), radius=0.0018)


        for y_fod in (-2.520, -1.460):
            _bm_add_steel_plate(bm_rails, xc + sx * 0.019, xc + sx * 0.0745, y_fod - 0.064, y_fod + 0.064, -0.856, -0.798, nu=3, nv=2, chamfer=0.0010)
            xf_in, xf_out = min(xc + sx * 0.074, xc + sx * 0.083), max(xc + sx * 0.074, xc + sx * 0.083)
            _bm_add_steel_plate(bm_rails, xf_in, xf_out, y_fod - 0.068, y_fod + 0.068, -0.858, -0.848, nu=3, nv=1, chamfer=0.0006)
            _bm_add_steel_plate(bm_rails, xf_in, xf_out, y_fod - 0.068, y_fod + 0.068, -0.806, -0.796, nu=3, nv=1, chamfer=0.0006)
            _bm_add_steel_plate(bm_rails, xf_in, xf_out, y_fod - 0.068, y_fod - 0.058, -0.849, -0.805, nu=1, nv=2, chamfer=0.0006)
            _bm_add_steel_plate(bm_rails, xf_in, xf_out, y_fod + 0.058, y_fod + 0.068, -0.849, -0.805, nu=1, nv=2, chamfer=0.0006)
            f0_fpl = len(bm_rails.faces)
            _bm_add_steel_plate(bm_rails, xc + sx * 0.074, xc + sx * 0.0775, y_fod - 0.058, y_fod + 0.058, -0.849, -0.805, nu=2, nv=2, chamfer=0.0005)
            bm_fill_unset_from(bm_rails, f0_fpl, MAT_PLATE_BASE)
            for sy_rv in (y_fod - 0.051, y_fod + 0.051):
                for sz_rv in (-0.843, -0.811):
                    bm_add_dome_rivet(bm_rails, Vector((xc + sx * 0.0775, sy_rv, sz_rv)), Vector((sx, 0, 0)), radius=0.0014)


        for ya_warn, yb_warn in ((-3.566, -3.414), (-2.428, -2.248)):
            _bm_add_steel_plate(bm_rails, xc + sx * 0.019, xc + sx * 0.0745, ya_warn + 0.003, yb_warn - 0.003, -0.860, -0.796, nu=3, nv=2, chamfer=0.0010)
            f0_wpl = len(bm_rails.faces)
            _bm_add_steel_plate(bm_rails, xc + sx * 0.074, xc + sx * 0.0795, ya_warn, yb_warn, -0.858, -0.798, nu=3, nv=2, chamfer=0.0006)
            bm_fill_unset_from(bm_rails, f0_wpl, MAT_PLATE_BASE)
            for sy_rv in (ya_warn + 0.009, yb_warn - 0.009):
                for sz_rv in (-0.851, -0.805):
                    bm_add_dome_rivet(bm_rails, Vector((xc + sx * 0.0795, sy_rv, sz_rv)), Vector((sx, 0, 0)), radius=0.0016)


        for y_fk in (-3.220, -2.520):
            _bm_add_steel_plate(bm_rails, xc - 0.082, xc + 0.082, y_fk - 0.125, y_fk + 0.125, -0.992, -0.976, nu=3, nv=3)
            _bm_add_steel_plate(bm_rails, xc - 0.082, xc + 0.082, y_fk - 0.125, y_fk + 0.125, -0.920, -0.904, nu=3, nv=3)
            for dy_fk in (-0.115, +0.115):
                _bm_add_steel_plate(bm_rails, xc - 0.082, xc + 0.082, y_fk + dy_fk - 0.010, y_fk + dy_fk + 0.010, -0.977, -0.919, nu=3, nv=2)
            _bm_add_steel_plate(bm_rails, xc + sx * 0.076, xc + sx * 0.088, y_fk - 0.136, y_fk + 0.136, -0.998, -0.980, nu=3, nv=2, chamfer=0.0012)
            _bm_add_steel_plate(bm_rails, xc + sx * 0.076, xc + sx * 0.088, y_fk - 0.136, y_fk + 0.136, -0.916, -0.898, nu=3, nv=2, chamfer=0.0012)
            for dy_fkl in (-0.126, +0.126):
                _bm_add_steel_plate(bm_rails, xc + sx * 0.076, xc + sx * 0.088, y_fk + dy_fkl - 0.010, y_fk + dy_fkl + 0.010, -0.981, -0.915, nu=2, nv=2, chamfer=0.0012)

            bm_add_spline_tube(
                bm_rails,
                [Vector((xc + sx * 0.080, y_fk - 0.130, -0.900)), Vector((xc + sx * 0.080, y_fk + 0.130, -0.900))],
                radius=0.0028, n_samples=26, segs=10, cap_ends=True, corrugations=22,
            )


        for y_end, dir_y in ((y_r0, -1.0), (y_r1, +1.0)):
            _bm_add_steel_plate(bm_rails, xc - 0.082, xc + 0.082, y_end, y_end + dir_y * 0.018, -0.908, -0.682, nu=4, nv=4, chamfer=0.0018)

            _bm_add_steel_plate(bm_rails, xc - 0.078, xc + 0.078, y_end + dir_y * 0.017, y_end + dir_y * 0.023, -0.768, -0.748, nu=3, nv=1, chamfer=0.0010)
            _bm_add_steel_plate(bm_rails, xc - 0.078, xc + 0.078, y_end + dir_y * 0.017, y_end + dir_y * 0.023, -0.904, -0.884, nu=3, nv=1, chamfer=0.0010)
            for dx_eb in (-0.058, +0.058):
                for dz_eb in (-0.870, -0.818, -0.774, -0.706):
                    bm_add_aerospace_bolt(bm_rails, Vector((xc + dx_eb, y_end + dir_y * 0.018, dz_eb)), Vector((0, dir_y, 0)), head_r=0.0050, height=0.0046)
            for dz_em in (-0.894, -0.758):
                bm_add_aerospace_bolt(bm_rails, Vector((xc, y_end + dir_y * 0.023, dz_em)), Vector((0, dir_y, 0)), head_r=0.0050, height=0.0046)
            y_stop = y_end - dir_y * 0.038
            _bm_add_steel_plate(bm_rails, xc - 0.072, xc + 0.072, y_stop - 0.024, y_stop + 0.024, -0.684, -0.636, nu=3, nv=2, chamfer=0.0018)
            for dx_bf in (-0.036, +0.036):
                f0_rbmp = len(bm_rails.faces)
                bm_add_segmented_tube(
                    bm_rails, Vector((xc + dx_bf, y_stop - dir_y * 0.024, -0.660)), Vector((xc + dx_bf, y_stop - dir_y * 0.048, -0.660)),
                    [(0.0, 0.018), (0.72, 0.018), (1.0, 0.014)], segs=20, cap_ends=True,
                )
                bm_fill_unset_from(bm_rails, f0_rbmp, MAT_RUBBER_DARK)
                bm_add_aerospace_bolt(bm_rails, Vector((xc + dx_bf, y_stop, -0.636)), Vector((0, 0, 1)), head_r=0.0052, height=0.0048)


        f0_shk = len(bm_rails.faces)
        for y_td in (-4.240, -0.650):
            sh_lo, sh_hi = min(xc + sx * 0.076, xc + sx * 0.116), max(xc + sx * 0.076, xc + sx * 0.116)
            for dy_sh in (-0.022, +0.022):
                bm_add_clevis_plate(bm_rails, (y_td + dy_sh - 0.006, -0.865), (y_td + dy_sh + 0.006, -0.902), 0.022, 0.024, sh_lo, sh_hi, arc_segs=13)
            rg_lo, rg_hi = min(xc + sx * 0.095, xc + sx * 0.126), max(xc + sx * 0.095, xc + sx * 0.126)
            bm_add_chamfered_cyl_x(bm_rails, y_td, -0.886, rg_lo, rg_hi, radius=0.024, segs=22, hollow_r=0.013)
        bm_fill_unset_from(bm_rails, f0_shk, MAT_STAND_ORANGE)


    cross_ys = (-4.270, -3.560, -2.860, -2.060, -1.320, -0.620)
    for y_cb in cross_ys:
        _bm_add_steel_plate(bm_rails, -X_RAIL + 0.072, +X_RAIL - 0.072, y_cb - 0.062, y_cb + 0.062, -0.904, -0.884, nu=18, nv=3)
        _bm_add_steel_plate(bm_rails, -X_RAIL + 0.072, +X_RAIL - 0.072, y_cb - 0.062, y_cb + 0.062, -0.790, -0.770, nu=18, nv=3)
        _bm_add_steel_plate(bm_rails, -X_RAIL + 0.074, +X_RAIL - 0.074, y_cb - 0.015, y_cb + 0.015, -0.885, -0.789, nu=18, nv=3)
        for sy_stf in (-1.0, +1.0):

            is_exposed_bumper = (y_cb == -4.270 and sy_stf < 0.0) or (y_cb == -0.620 and sy_stf > 0.0)
            xs_stf = (-0.550, -0.240, +0.240, +0.550) if is_exposed_bumper else (-0.380, -0.190, +0.190, +0.380)
            for x_stf in xs_stf:
                _bm_add_steel_plate(bm_rails, x_stf - 0.006, x_stf + 0.006, y_cb + sy_stf * 0.014, y_cb + sy_stf * 0.056, -0.885, -0.789, nu=2, nv=2, chamfer=0.0012)
        for sx in (-1.0, +1.0):
            xg = sx * (X_RAIL - 0.076)
            _bm_add_steel_plate(bm_rails, xg - 0.010, xg + 0.010, y_cb - 0.096, y_cb + 0.096, -0.902, -0.768, nu=3, nv=3)
            _bm_add_steel_plate(bm_rails, xg, xg - sx * 0.085, y_cb - 0.088, y_cb + 0.088, -0.790, -0.772, nu=2, nv=2)
            for dy_b in (-0.068, -0.024, +0.024, +0.068):
                for dz_b in (-0.868, -0.804):
                    bm_add_aerospace_bolt(bm_rails, Vector((xg - sx * 0.011, y_cb + dy_b, dz_b)), Vector((-sx, 0, 0)), head_r=0.0046, height=0.0044)

    for k_sp in range(len(cross_ys) - 1):
        ya_s = cross_ys[k_sp] + 0.060
        yb_s = cross_ys[k_sp + 1] - 0.060
        _bm_add_steel_plate(bm_rails, -0.058, +0.058, ya_s, yb_s, -0.900, -0.882, nu=2, nv=6)
        _bm_add_steel_plate(bm_rails, -0.058, +0.058, ya_s, yb_s, -0.810, -0.792, nu=2, nv=6)
        _bm_add_steel_plate(bm_rails, -0.014, +0.014, ya_s, yb_s, -0.883, -0.809, nu=6, nv=2)

    _sd_stand.add_cradle_01_rail_and_catwalk_details(bm_rails, X_RAIL, Y_FWD_ARCH, Y_AFT_ARCH, _bm_add_steel_plate)
    _hd_detail.add_cradle_01_fodbox_and_canister_hw(bm_rails, _bm_add_steel_plate)
    create_mesh_obj("Stand_Cradle_01_Longitudinal_Rails_And_Crossbeams", bm_rails, coll_stand, clay)


    bm_peds = bmesh.new()

    for y_ped, is_front in ((Y_FWD_ARCH, True), (Y_AFT_ARCH, False)):
        for sx in (-1.0, +1.0):
            x_in  = sx * 0.588
            x_out = sx * 0.744
            x_lo, x_hi = min(x_in, x_out), max(x_in, x_out)
            xc = 0.5 * (x_in + x_out)


            bg_lo, bg_hi = min(x_in - sx * 0.004, x_out + sx * 0.014), max(x_in - sx * 0.004, x_out + sx * 0.014)
            _bm_add_steel_plate(bm_peds, bg_lo, bg_hi, y_ped - 0.198, y_ped + 0.198, -0.626, -0.592, nu=4, nv=6, chamfer=0.0022)
            for xa_s, xb_s in ((x_in - sx * 0.004, x_in + sx * 0.022), (x_out - sx * 0.014, x_out + sx * 0.014)):
                x_sk_a, x_sk_b = min(xa_s, xb_s), max(xa_s, xb_s)
                bm_add_clevis_plate(bm_peds, (y_ped - 0.136, -0.687), (y_ped + 0.136, -0.687), 0.062, 0.062, x_sk_a, x_sk_b, arc_segs=15)
                _bm_add_steel_plate(bm_peds, x_sk_a + 0.0030, x_sk_b - 0.0030, y_ped - 0.136, y_ped + 0.136, -0.754, -0.625, nu=5, nv=3, chamfer=0.0018)


            for dy_pfb in (-0.168, 0.0, +0.168):
                for dx_pfb in (-0.054, +0.054):
                    bm_add_aerospace_bolt(bm_peds, Vector((xc + dx_pfb, y_ped + dy_pfb, -0.592)), Vector((0, 0, 1)), head_r=0.0048, height=0.0045)


            f0_ped_rl = len(bm_peds.faces)
            for dy_rol in (-0.124, +0.124):
                rl1_a, rl1_b = min(x_in - sx * 0.006, x_out + sx * 0.024), max(x_in - sx * 0.006, x_out + sx * 0.024)
                rl2_a, rl2_b = min(x_in - sx * 0.008, x_out + sx * 0.030), max(x_in - sx * 0.008, x_out + sx * 0.030)
                bm_add_chamfered_cyl_x(bm_peds, y_ped + dy_rol, -0.692, rl1_a, rl1_b, radius=0.043, segs=32, hollow_r=0.022)
                bm_add_chamfered_cyl_x(bm_peds, y_ped + dy_rol, -0.692, rl2_a, rl2_b, radius=0.019, segs=24)
                bm_add_grease_zerk(bm_peds, Vector((x_out + sx * 0.030, y_ped + dy_rol, -0.692)), Vector((sx, 0, 0)), scale=1.15)
                for kb_rl in range(8):
                    arl = 2.0 * math.pi * kb_rl / 8.0
                    bm_add_aerospace_bolt(
                        bm_peds, Vector((x_out + sx * 0.024, y_ped + dy_rol + 0.032 * math.cos(arl), -0.692 + 0.032 * math.sin(arl))),
                        Vector((sx, 0, 0)), head_r=0.0034, height=0.0032,
                    )
                _bm_add_steel_plate(bm_peds, xc + sx * 0.045, x_out + sx * 0.026,
                                    y_ped + dy_rol - 0.030, y_ped + dy_rol + 0.030, -0.766, -0.740, nu=2, nv=2, chamfer=0.0016)
                bm_add_aerospace_bolt(bm_peds, Vector((x_out + sx * 0.016, y_ped + dy_rol, -0.766)), Vector((0, 0, -1)), head_r=0.0040, height=0.0038)


            y_plg_r = y_ped + (-0.176 if is_front else +0.176)
            bm_add_chamfered_cyl_x(bm_peds, y_plg_r, -0.718, min(x_out, x_out + sx * 0.028), max(x_out, x_out + sx * 0.028), radius=0.014, segs=20)
            bm_add_chamfered_cyl_x(bm_peds, y_plg_r, -0.718, min(x_out + sx * 0.026, x_out + sx * 0.052), max(x_out + sx * 0.026, x_out + sx * 0.052), radius=0.0065, segs=16)
            bm_add_segmented_tube(
                bm_peds, Vector((x_out + sx * 0.050, y_plg_r - 0.024, -0.718)), Vector((x_out + sx * 0.050, y_plg_r + 0.024, -0.718)),
                [(0.0, 0.0055), (0.20, 0.0068), (0.80, 0.0068), (1.0, 0.0055)], segs=14, cap_ends=True,
            )
            bm_add_spline_tube(
                bm_peds,
                [
                    Vector((x_out + sx * 0.036, y_plg_r, -0.706)),
                    Vector((x_out + sx * 0.032, y_plg_r, -0.668)),
                    Vector((x_out + sx * 0.014, y_plg_r, -0.642)),
                ],
                radius=0.0014, n_samples=16, segs=10, cap_ends=True,
            )


            x_hw0 = x_out + sx * 0.020
            x_hw1 = x_out + sx * 0.046
            bm_add_chamfered_cyl_x(bm_peds, y_ped, -0.692, min(x_hw0 - sx * 0.012, x_hw1 - sx * 0.008), max(x_hw0 - sx * 0.012, x_hw1 - sx * 0.008), radius=0.024, segs=28)
            n_hw = 80
            def _hw_pt(k: int, vr: float, vt: float, _yp=y_ped, _xa=min(x_hw0, x_hw1), _xb=max(x_hw0, x_hw1)):
                th = 2.0 * math.pi * k / float(n_hw)
                r_in = 0.0205
                r_out = 0.0465 + 0.0125 * math.cos(4.0 * th)
                rr = r_in + (r_out - r_in) * vr
                crown = 0.0032 * math.sin(math.pi * vr) * (vt - 0.5) * 2.0
                xv = _xa + (_xb - _xa) * vt + crown
                return Vector((xv, _yp + rr * math.cos(th), -0.692 + rr * math.sin(th)))
            _bm_add_annular_quad_extrusion(bm_peds, _hw_pt, n_hw, t_rad_std, t_thk_std)

            x_sq_in  = x_hw1 - sx * 0.008
            x_sq_out = x_hw1 + sx * 0.007
            x_sq_a, x_sq_b = min(x_sq_in, x_sq_out), max(x_sq_in, x_sq_out)
            def _hw_sq_boss_pt(k: int, vr: float, vt: float, _yp=y_ped, _xa=x_sq_a, _xb=x_sq_b):
                th = 2.0 * math.pi * k / 32.0
                ct, st = math.cos(th), math.sin(th)
                m_inf = max(abs(ct), abs(st), 1e-6)
                yi, zi = 0.0095 * (ct / m_inf), 0.0095 * (st / m_inf)
                yo, zo = 0.0225 * ct, 0.0225 * st
                xv = _xa + (_xb - _xa) * vt
                return Vector((xv, _yp + yi + (yo - yi) * vr, -0.692 + zi + (zo - zi) * vr))
            _bm_add_annular_quad_extrusion(bm_peds, _hw_sq_boss_pt, 32, [0.00, 0.08, 0.26, 0.74, 0.92, 1.00], [0.00, 0.14, 0.86, 1.00])
            _bm_add_steel_plate(bm_peds, x_sq_in - sx * 0.005, x_sq_in + sx * 0.001, y_ped - 0.015, y_ped + 0.015, -0.707, -0.677, nu=2, nv=2, chamfer=0.0006)
            bm_fill_unset_from(bm_peds, f0_ped_rl, MAT_STAND_STEEL)


            _bm_add_steel_plate(bm_peds, x_in, x_in + sx * 0.024, y_ped - 0.100, y_ped + 0.100, -0.592, -0.168, nu=3, nv=4, chamfer=0.0018)


            for s_side in (-1.0, +1.0):
                y_fl_bot = y_ped + s_side * 0.112
                y_fl_top = y_ped + s_side * 0.098
                n_pf = 64
                def _ped_flange_pt(k: int, vr: float, vt: float, _yb=y_fl_bot, _yt=y_fl_top, _xl=x_lo, _xh=x_hi):
                    th = 2.0 * math.pi * k / float(n_pf)
                    ct, st = math.cos(th), math.sin(th)
                    xc_m = 0.5 * (_xl + _xh)
                    hw_x = 0.5 * (_xh - _xl)
                    xi = xc_m + 0.034 * ct
                    zi = -0.385 + 0.092 * st
                    ab_c, ab_s = abs(ct), abs(st)
                    m_inf = max(ab_c, ab_s, 1e-6)
                    ux, uz = ct / m_inf, st / m_inf
                    z_o = -0.380 + 0.212 * uz
                    xo = xc_m + hw_x * ux
                    xv = xi + (xo - xi) * vr
                    zv = zi + (z_o - zi) * vr
                    fz = (zv - (-0.592)) / 0.424
                    yc_cur = _yb + (_yt - _yb) * fz
                    yv = (yc_cur - 0.012) + 0.024 * vt
                    return Vector((xv, yv, zv))
                _bm_add_annular_quad_extrusion(bm_peds, _ped_flange_pt, n_pf, t_rad_std, t_thk_std)

                def _ped_oval_lip_pt(k: int, vr: float, vt: float, _yb=y_fl_bot, _yt=y_fl_top, _xl=x_lo, _xh=x_hi):
                    th = 2.0 * math.pi * k / 48.0
                    ct, st = math.cos(th), math.sin(th)
                    xc_m = 0.5 * (_xl + _xh)
                    xv = xc_m + (0.032 + 0.011 * vr) * ct
                    zv = -0.385 + (0.090 + 0.011 * vr) * st
                    fz = (zv - (-0.592)) / 0.424
                    yc_cur = _yb + (_yt - _yb) * fz
                    yv = (yc_cur - 0.0155) + 0.031 * vt
                    return Vector((xv, yv, zv))
                _bm_add_annular_quad_extrusion(bm_peds, _ped_oval_lip_pt, 48, [0.00, 0.14, 0.86, 1.00], [0.00, 0.14, 0.86, 1.00])


            n_pw = 64
            x_rib_in  = x_in + sx * 0.022
            x_rib_out = x_out
            xr_lo, xr_hi = min(x_rib_in, x_rib_out), max(x_rib_in, x_rib_out)
            def _ped_center_rib_pt(k: int, vr: float, vt: float, _yp=y_ped, _x0=xr_lo, _x1=xr_hi):
                th = 2.0 * math.pi * k / float(n_pw)
                ct, st = math.cos(th), math.sin(th)
                xm = 0.5 * (_x0 + _x1)
                hx = 0.5 * (_x1 - _x0)
                xi = xm + 0.026 * ct
                zi = -0.385 + 0.088 * st
                ab_c, ab_s = abs(ct), abs(st)
                m_inf = max(ab_c, ab_s, 1e-6)
                ux, uz = ct / m_inf, st / m_inf
                xo = xm + hx * ux
                zo = -0.380 + 0.212 * uz
                xv = xi + (xo - xi) * vr
                zv = zi + (zo - zi) * vr
                yv = (_yp - 0.012) + 0.024 * vt
                return Vector((xv, yv, zv))
            _bm_add_annular_quad_extrusion(bm_peds, _ped_center_rib_pt, n_pw, t_rad_std, t_thk_std)


            bay_lo, bay_hi = min(x_rib_in, x_out - sx * 0.008), max(x_rib_in, x_out - sx * 0.008)
            for s_bay in (-1.0, +1.0):
                ya_b = min(y_ped + s_bay * 0.012, y_ped + s_bay * 0.090)
                yb_b = max(y_ped + s_bay * 0.012, y_ped + s_bay * 0.090)
                _bm_add_steel_plate(bm_peds, bay_lo, bay_hi, ya_b, yb_b, -0.388, -0.372, nu=2, nv=2, chamfer=0.0014)
                bm_add_clevis_plate(bm_peds, (y_ped + s_bay * 0.132, -0.592), (y_ped + s_bay * 0.106, -0.495),
                                    0.014, 0.010, x_lo + 0.018, x_hi - 0.018, arc_segs=9)

            cap_lo, cap_hi = min(x_in, x_out + sx * 0.014), max(x_in, x_out + sx * 0.014)
            _bm_add_steel_plate(bm_peds, cap_lo, cap_hi, y_ped - 0.122, y_ped + 0.122, -0.168, -0.142, nu=4, nv=4, chamfer=0.0020)
            for dx_fb in (-0.048, +0.048):
                for dy_fb in (-0.092, -0.034, +0.034, +0.092):
                    bm_add_aerospace_bolt(bm_peds, Vector((xc + dx_fb, y_ped + dy_fb, -0.142)), Vector((0, 0, 1)), head_r=0.0052, height=0.0048)


            x_br_c = sx * 0.700
            if is_front:
                _bm_add_steel_plate(bm_peds, x_br_c - 0.036, x_br_c + 0.036, y_ped + 0.108, y_ped + 0.168, -0.592, -0.498, nu=2, nv=2, chamfer=0.0018)
                for sub_e in (-0.025, +0.025):
                    bm_add_clevis_plate(bm_peds, (y_ped + 0.118, -0.560), (y_ped + 0.176, -0.538), 0.032, 0.036, x_br_c + sub_e - 0.008, x_br_c + sub_e + 0.008, arc_segs=17)
                bm_add_clevis_pin_assembly_x(bm_peds, y_ped + 0.176, -0.538, x_br_c - 0.042, x_br_c + 0.042, pin_r=0.0135, segs=24, compact_ends=True, flip_x=(sx < 0.0))
            else:
                _bm_add_steel_plate(bm_peds, x_br_c - 0.036, x_br_c + 0.036, y_ped - 0.158, y_ped - 0.098, -0.142, -0.022, nu=2, nv=2, chamfer=0.0018)
                for sub_e in (-0.025, +0.025):
                    bm_add_clevis_plate(bm_peds, (y_ped - 0.106, -0.092), (y_ped - 0.166, -0.068), 0.032, 0.036, x_br_c + sub_e - 0.008, x_br_c + sub_e + 0.008, arc_segs=17)
                bm_add_clevis_pin_assembly_x(bm_peds, y_ped - 0.166, -0.068, x_br_c - 0.042, x_br_c + 0.042, pin_r=0.0135, segs=24, compact_ends=True, flip_x=(sx < 0.0))

    _sd_stand.add_cradle_02_bogie_and_pedestal_details(bm_peds, X_RAIL, Y_FWD_ARCH, Y_AFT_ARCH, _bm_add_steel_plate)
    _hd_detail.add_cradle_02_pedestal_and_bogie_details(bm_peds, X_RAIL, Y_FWD_ARCH, Y_AFT_ARCH, _bm_add_steel_plate)
    create_mesh_obj("Stand_Cradle_02_Pedestals_Rollers_And_Handwheels", bm_peds, coll_stand, clay)


    bm_arches = bmesh.new()
    R_A_IN  = 0.590
    R_A_OUT = 0.696
    R_W_IN  = 0.605
    R_W_OUT = 0.681

    def _add_curved_arch_band(bm_t, y_c: float, half_y: float, r0: float, r1: float, deg0: float, deg1: float, n_p: int = 28, n_y: int = 3):
        tp = subdivide_1d_with_support_loops(n_p, sup1=0.015, sup2=0.050)
        ty = subdivide_1d_with_support_loops(n_y, sup1=0.05, sup2=0.16)
        g_in, g_out = [], []
        for fp in tp:
            ang = math.radians(deg0 + (deg1 - deg0) * fp)
            sa, ca = math.sin(ang), math.cos(ang)
            ci, co = [], []
            for fy in ty:
                yv = (y_c - half_y) + 2.0 * half_y * fy
                ci.append(bm_t.verts.new((r0 * sa, yv, r0 * ca)))
                co.append(bm_t.verts.new((r1 * sa, yv, r1 * ca)))
            g_in.append(ci)
            g_out.append(co)
        bm_build_beveled_quad_shell(bm_t, g_in, g_out, bulge_scale=0.05, max_bevel=0.0020, inset_bevel=True)

    for y_ar, is_front_ar in ((Y_FWD_ARCH, True), (Y_AFT_ARCH, False)):

        _add_curved_arch_band(bm_arches, y_ar, 0.069, R_A_IN, R_W_IN, -96.0, +96.0, n_p=76, n_y=4)
        _add_curved_arch_band(bm_arches, y_ar, 0.069, R_W_OUT, R_A_OUT, -96.0, +96.0, n_p=76, n_y=4)


        _add_curved_arch_band(bm_arches, y_ar, 0.054, R_W_IN, R_W_OUT, -35.0, +35.0, n_p=30, n_y=3)
        for deg_stf in (-85.0, -32.0, -21.0, -10.5, +10.5, +21.0, +32.0, +85.0):
            b_astf = bmesh.new()
            for sy_stf in (-1.0, +1.0):
                _bm_add_steel_plate(b_astf, -0.006, +0.006, y_ar + sy_stf * 0.053, y_ar + sy_stf * 0.068, R_W_IN, R_W_OUT, nu=2, nv=2, chamfer=0.0010)
            bm_merge(bm_arches, b_astf, Matrix.Rotation(math.radians(deg_stf), 4, 'Y'))

        for s_ang in (-1.0, +1.0):
            d_a, d_b = sorted((s_ang * 81.0, s_ang * 96.0))
            _add_curved_arch_band(bm_arches, y_ar, 0.054, R_W_IN, R_W_OUT, d_a, d_b, n_p=8, n_y=3)

            n_sl = 72
            def _arch_slot_pt(k: int, vr: float, vt: float, _yc=y_ar, _sa=s_ang):
                th = 2.0 * math.pi * k / float(n_sl)
                ct, st = math.cos(th), math.sin(th)
                deg_mid = _sa * 58.0
                deg_in  = deg_mid + 20.0 * ct
                r_in_s  = 0.643 + 0.010 * st
                ab_c, ab_s = abs(ct), abs(st)
                m_inf = max(ab_c, ab_s, 1e-6)
                deg_out = deg_mid + 23.0 * (ct / m_inf)
                r_out_s = 0.643 + 0.038 * (st / m_inf)
                deg_cur = deg_in + (deg_out - deg_in) * vr
                r_cur   = r_in_s + (r_out_s - r_in_s) * vr
                ar = math.radians(deg_cur)
                yv = (_yc - 0.054) + 0.108 * vt
                return Vector((r_cur * math.sin(ar), yv, r_cur * math.cos(ar)))
            _bm_add_annular_quad_extrusion(bm_arches, _arch_slot_pt, n_sl, t_rad_std, t_thk_std)

            for d_pin_s in (46.0, 68.0):
                ar_pin = math.radians(s_ang * d_pin_s)
                p_sl_f = Vector((0.643 * math.sin(ar_pin), y_ar - 0.064, 0.643 * math.cos(ar_pin)))
                p_sl_b = Vector((0.643 * math.sin(ar_pin), y_ar + 0.064, 0.643 * math.cos(ar_pin)))
                bm_add_segmented_tube(bm_arches, p_sl_f, p_sl_b,
                                      [(0.0, 0.017), (0.14, 0.017), (0.18, 0.010), (0.82, 0.010), (0.86, 0.017), (1.0, 0.017)], segs=22, cap_ends=True)


        for sx in (-1.0, +1.0):
            xb_lo, xb_hi = min(sx * 0.590, sx * 0.754), max(sx * 0.590, sx * 0.754)
            _bm_add_steel_plate(bm_arches, xb_lo, xb_hi, y_ar - 0.106, y_ar + 0.106, -0.142, -0.058, nu=4, nv=4, chamfer=0.0022)
            _bm_add_steel_plate(bm_arches, xb_lo, xb_hi, y_ar - 0.098, y_ar + 0.098, -0.050, +0.038, nu=4, nv=4, chamfer=0.0022)
            for dy_cl in (-0.114, +0.114):
                _bm_add_steel_plate(bm_arches, sx * 0.612, sx * 0.744, y_ar + dy_cl - 0.018, y_ar + dy_cl + 0.018, -0.094, -0.014, nu=3, nv=2, chamfer=0.0018)
                for rx_cb in (0.636, 0.720):
                    bm_add_aerospace_bolt(bm_arches, Vector((sx * rx_cb, y_ar + dy_cl, -0.014)), Vector((0, 0, 1)), head_r=0.0056, height=0.0052)


            f0_trn_s = len(bm_arches.faces)
            bm_add_chamfered_cyl_x(bm_arches, y_ar, -0.054, min(sx * 0.590, sx * 0.772), max(sx * 0.590, sx * 0.772), radius=0.058, segs=36)
            bm_add_chamfered_cyl_x(bm_arches, y_ar, -0.054, min(sx * 0.588, sx * 0.786), max(sx * 0.588, sx * 0.786), radius=0.027, segs=28, hollow_r=0.012)
            bm_add_bevel_locked_hex(bm_arches, Vector((sx * 0.784, y_ar, -0.054)), Vector((sx * 0.798, y_ar, -0.054)),
                                    [(0.0, 0.015), (0.25, 0.019), (0.85, 0.019), (1.0, 0.014)], corner_bevel=0.14, cap_ends=True)
            bm_add_grease_zerk(bm_arches, Vector((sx * 0.764, y_ar - 0.058, -0.054)), Vector((0, -1, 0)), scale=1.2)
            if is_front_ar and sx > 0.0:

                bm_add_chamfered_cyl_x(bm_arches, y_ar, -0.054, sx * 0.786, sx * 0.812, radius=0.023, segs=24)
                bm_add_clevis_plate(bm_arches, (y_ar, -0.054), (y_ar - 0.132, -0.010), 0.018, 0.014, sx * 0.796, sx * 0.810, arc_segs=13)
                bm_add_segmented_tube(
                    bm_arches, Vector((sx * 0.808, y_ar - 0.132, -0.010)), Vector((sx * 0.888, y_ar - 0.132, -0.010)),
                    [(0.0, 0.010), (0.14, 0.0145), (0.86, 0.0145), (1.0, 0.011)], segs=20, cap_ends=True,
                )
            bm_fill_unset_from(bm_arches, f0_trn_s, MAT_STAND_STEEL)
            for ib_t in range(10):
                at = 2.0 * math.pi * ib_t / 10.0
                bm_add_aerospace_bolt(
                    bm_arches, Vector((sx * 0.772, y_ar + 0.044 * math.cos(at), -0.054 + 0.044 * math.sin(at))),
                    Vector((sx, 0, 0)), head_r=0.0040, height=0.0038,
                )


            bm_add_clevis_plate(bm_arches, (y_ar - 0.044, +0.012), (y_ar + 0.044, +0.012), 0.015, 0.015,
                                min(sx * 0.753, sx * 0.762), max(sx * 0.753, sx * 0.762), arc_segs=13)
            db_bpts = []
            for dy_db in (-0.044, +0.044):
                p_dbb = Vector((sx * 0.762, y_ar + dy_db, +0.012))
                db_bpts.append(p_dbb)
                bm_add_aerospace_bolt(bm_arches, p_dbb, Vector((sx, 0, 0)), head_r=0.0044, height=0.0042)

            bm_add_bolt_pair_lockwire(bm_arches, db_bpts[0], db_bpts[1], Vector((sx, 0, 0)), head_r=0.0044, strand_r=0.00052)
            _bm_add_steel_plate(bm_arches, sx * 0.753, sx * 0.758, y_ar - 0.016, y_ar + 0.016, +0.032, +0.068, nu=2, nv=2, chamfer=0.0006)
            bm_add_aerospace_bolt(bm_arches, Vector((sx * 0.758, y_ar, +0.046)), Vector((sx, 0, 0)), head_r=0.0024, height=0.0022)

            b_deg_sc = bmesh.new()
            for s_side_dg in (-1.0, +1.0):
                for idx_dg, deg_abs in enumerate((52.0, 58.0, 64.0, 70.0, 76.0, 82.0)):
                    ang_dg = math.radians(s_side_dg * deg_abs)
                    r_dg0 = 0.0615
                    r_dg1 = 0.0705 if (idx_dg % 2 == 0) else 0.0668
                    tk_len = r_dg1 - r_dg0
                    tk_w = 0.0014 if (idx_dg % 2 == 0) else 0.0010
                    b_tk = bmesh.new()
                    bm_add_chamfered_box(
                        b_tk,
                        min(sx * 0.7536, sx * 0.7549), max(sx * 0.7536, sx * 0.7549),
                        -0.5 * tk_w, +0.5 * tk_w,
                        r_dg0, r_dg1,
                        nx=1, ny=1, nz=1, chamfer=0.00022,
                    )
                    tf_tk = Matrix.Translation((0.0, y_ar, -0.054)) @ Matrix.Rotation(-ang_dg, 4, 'X')
                    bm_merge(b_deg_sc, b_tk, tf_tk, default_mat_id=MAT_3D_TEXT)
            bm_merge(bm_arches, b_deg_sc)


            bm_add_segmented_tube(bm_arches, Vector((sx * 0.752, y_ar + 0.072, +0.008)), Vector((sx * 0.784, y_ar + 0.072, +0.008)),
                                  [(0.0, 0.0082), (0.65, 0.0082), (1.0, 0.0055)], segs=18, cap_ends=True)
            bm_add_spline_tube(bm_arches, [Vector((sx * 0.782, y_ar + 0.072, +0.008)), Vector((sx * 0.774, y_ar + 0.090, -0.036)), Vector((sx * 0.756, y_ar + 0.082, -0.088))],
                               radius=0.0015, n_samples=18, segs=12, cap_ends=True)


            f0_rl_pl = len(bm_arches.faces)
            _bm_add_steel_plate(bm_arches, sx * 0.7535, sx * 0.7578, y_ar - 0.056, y_ar + 0.056, -0.140, -0.115, nu=2, nv=2, chamfer=0.0005)
            bm_fill_unset_from(bm_arches, f0_rl_pl, MAT_PLATE_BASE)
            for sy_mr in (-0.049, +0.049):
                for sz_mr in (-0.136, -0.119):
                    bm_add_dome_rivet(bm_arches, Vector((sx * 0.7578, y_ar + sy_mr, sz_mr)), Vector((sx, 0, 0)), radius=0.0014)
            for (deg_pl_a, hy_mp, hw_ang) in ((81.5, 0.048, 0.024), (74.5, 0.044, 0.020)):
                b_apl = bmesh.new()
                f0_apl = len(b_apl.faces)
                _bm_add_steel_plate(b_apl, -hw_ang, +hw_ang, y_ar - hy_mp, y_ar + hy_mp, R_A_OUT - 0.001, R_A_OUT + 0.0022, nu=2, nv=2, chamfer=0.0005)
                bm_fill_unset_from(b_apl, f0_apl, MAT_PLATE_BASE)
                for sx_mr in (-hw_ang + 0.004, +hw_ang - 0.004):
                    for sy_mr in (-hy_mp + 0.006, +hy_mp - 0.006):
                        bm_add_dome_rivet(b_apl, Vector((sx_mr, y_ar + sy_mr, R_A_OUT + 0.0022)), Vector((0, 0, 1)), radius=0.0015)
                bm_merge(bm_arches, b_apl, Matrix.Rotation(math.radians(-sx * deg_pl_a), 4, 'Y'))


            if not is_front_ar:
                bm_add_clevis_plate(bm_arches, (y_ar + 0.068, 0.0), (-2.186, 0.0), 0.044, 0.038,
                                    min(sx * 0.594, sx * 0.656), max(sx * 0.594, sx * 0.656), arc_segs=17)
                bm_add_clevis_plate(bm_arches, (y_ar + 0.068, +0.052), (-2.202, +0.030), 0.012, 0.010,
                                    min(sx * 0.608, sx * 0.642), max(sx * 0.608, sx * 0.642), arc_segs=11)
                bm_add_clevis_plate(bm_arches, (y_ar + 0.068, -0.052), (-2.202, -0.030), 0.012, 0.010,
                                    min(sx * 0.608, sx * 0.642), max(sx * 0.608, sx * 0.642), arc_segs=11)
                bm_add_chamfered_cyl_x(bm_arches, -2.165, 0.0, sx * 0.575, sx * 0.658, radius=0.036, segs=32, hollow_r=0.0186)
                for dy_cp in (-0.011, +0.011):
                    bm_add_aerospace_bolt(bm_arches, Vector((sx * 0.626, -2.165 + dy_cp, +0.0342)), Vector((0, 0, 1)), head_r=0.0036, height=0.0034)
            else:
                ar_s = math.radians(sx * 40.0)
                n_s = Vector((math.sin(ar_s), 0.0, math.cos(ar_s)))
                bm_add_segmented_tube(
                    bm_arches, n_s * 0.592 + Vector((0, y_ar, 0)), n_s * 0.512 + Vector((0, y_ar, 0)),
                    [(0.0, 0.024), (0.65, 0.020), (0.85, 0.026), (1.0, 0.024)], segs=24, cap_ends=True,
                )
                bm_add_segmented_tube(
                    bm_arches, n_s * 0.694 + Vector((0, y_ar, 0)), n_s * 0.716 + Vector((0, y_ar, 0)),
                    [(0.0, 0.040), (0.65, 0.040), (1.0, 0.026)], segs=24, cap_ends=True,
                )
                bm_add_helical_thread_cylinder(
                    bm_arches, n_s * 0.714 + Vector((0, y_ar, 0)), n_s * 0.772 + Vector((0, y_ar, 0)),
                    r_core=0.0155, thread_h=0.0024, n_turns=8, segs_turn=18, cap_core=True,
                )
                bm_add_bevel_locked_hex(bm_arches, n_s * 0.736 + Vector((0, y_ar, 0)), n_s * 0.756 + Vector((0, y_ar, 0)),
                                        [(0.0, 0.022), (0.25, 0.027), (0.80, 0.027), (1.0, 0.021)], corner_bevel=0.14, cap_ends=True)


        z_crown = 0.736 if is_front_ar else 0.748
        y_lug_c = y_ar + (0.072 if is_front_ar else -0.072)
        _bm_add_steel_plate(bm_arches, -0.062, +0.062, min(y_ar - 0.068, y_lug_c - 0.036), max(y_ar + 0.068, y_lug_c + 0.036), 0.690, z_crown - 0.020, nu=3, nv=3, chamfer=0.0020)
        for sx_top in (-0.031, +0.031):

            bm_add_clevis_plate(bm_arches, (y_ar - 0.028, 0.692), (y_lug_c, z_crown), 0.0360, 0.0360, sx_top - 0.0095, sx_top + 0.0095, arc_segs=17)
            bm_add_clevis_plate(bm_arches, (y_ar + 0.028, 0.692), (y_lug_c, z_crown), 0.0342, 0.0342, sx_top - 0.0083, sx_top + 0.0083, arc_segs=17)
        bm_add_clevis_pin_assembly_x(bm_arches, y_lug_c, z_crown, -0.050, +0.050, pin_r=0.0140, segs=26, compact_ends=True)

        y_sub_pin = y_ar + (0.026 if is_front_ar else -0.026)
        bm_add_clevis_pin_assembly_x(bm_arches, y_sub_pin, z_crown - 0.008, -0.048, +0.048, pin_r=0.0085, segs=22, compact_ends=True)

    _sd_stand.add_cradle_03_arch_and_trunnion_details(bm_arches, Y_FWD_ARCH, Y_AFT_ARCH, R_A_IN, R_A_OUT, R_W_IN, R_W_OUT, _bm_add_steel_plate)
    _hd_detail.add_cradle_03_wormgear_and_pippins(bm_arches, Y_FWD_ARCH, Y_AFT_ARCH, _bm_add_steel_plate)
    create_mesh_obj("Stand_Cradle_03_RollOver_Arches_And_MountPickups", bm_arches, coll_stand, clay)


    bm_bars = bmesh.new()

    p_top_f = Vector((0.0, Y_FWD_ARCH + 0.072, 0.736))
    p_top_a = Vector((0.0, Y_AFT_ARCH - 0.072, 0.748))
    dir_top = (p_top_a - p_top_f).normalized()
    bm_add_segmented_tube(
        bm_bars, p_top_f + dir_top * 0.046, p_top_a - dir_top * 0.046,
        [
            (0.00, 0.028), (0.04, 0.047), (0.09, 0.047), (0.11, 0.041),
            (0.47, 0.041), (0.48, 0.046), (0.52, 0.046), (0.53, 0.041),
            (0.89, 0.041), (0.91, 0.047), (0.96, 0.047), (1.00, 0.028),
        ],
        segs=36, cap_ends=True,
    )
    for p_he, s_he in ((p_top_f + dir_top * 0.095, +1.0), (p_top_a - dir_top * 0.095, -1.0)):
        bm_add_bevel_locked_hex(bm_bars, p_he, p_he + dir_top * (s_he * 0.042),
                                [(0.0, 0.042), (0.20, 0.049), (0.80, 0.049), (1.0, 0.042)], corner_bevel=0.14, cap_ends=True)
        p_tc_c = p_he - dir_top * (s_he * 0.024)
        for sx_tear in (-0.012, +0.012):
            _bm_add_steel_plate(bm_bars, sx_tear - 0.006, sx_tear + 0.006, p_tc_c.y - 0.022, p_tc_c.y + 0.022, p_tc_c.z + 0.040, p_tc_c.z + 0.066, nu=1, nv=2, chamfer=0.0010)
        bm_add_aerospace_bolt(bm_bars, Vector((+0.018, p_tc_c.y, p_tc_c.z + 0.055)), Vector((1, 0, 0)), head_r=0.0052, height=0.0048, with_12pt_nut_depth=0.036)
        f0_tbsh = len(bm_bars.faces)
        bm_add_segmented_tube(bm_bars, Vector((-0.018, p_tc_c.y, p_tc_c.z + 0.055)), Vector((+0.018, p_tc_c.y, p_tc_c.z + 0.055)), [(0.0, 0.0036), (1.0, 0.0036)], segs=14, cap_ends=True)
        bm_add_helical_thread_cylinder(bm_bars, Vector((-0.014, p_tc_c.y, p_tc_c.z + 0.055)), Vector((+0.014, p_tc_c.y, p_tc_c.z + 0.055)), r_core=0.0034, thread_h=0.00065, n_turns=6, segs_turn=14, cap_core=False)
        bm_fill_unset_from(bm_bars, f0_tbsh, MAT_FASTENER)
    bm_add_clevis_plate(bm_bars, (p_top_f.y, p_top_f.z), (p_top_f.y + 0.062, p_top_f.z), 0.034, 0.030, -0.019, +0.019, arc_segs=17)
    bm_add_clevis_plate(bm_bars, (p_top_a.y, p_top_a.z), (p_top_a.y - 0.062, p_top_a.z), 0.034, 0.030, -0.019, +0.019, arc_segs=17)

    for sx in (-1.0, +1.0):
        xb = sx * 0.700
        p_df = Vector((xb, Y_FWD_ARCH + 0.176, -0.538))
        p_da = Vector((xb, Y_AFT_ARCH - 0.166, -0.068))
        dir_d = (p_da - p_df).normalized()
        up_d  = Vector((1.0, 0.0, 0.0)).cross(dir_d).normalized()
        p_dm = p_df.lerp(p_da, 0.50)

        bm_add_segmented_tube(
            bm_bars, p_df + dir_d * 0.046, p_dm - dir_d * 0.210,
            [
                (0.00, 0.026), (0.06, 0.046), (0.14, 0.046), (0.18, 0.040),
                (0.86, 0.040), (0.91, 0.044), (0.96, 0.044), (1.00, 0.028),
            ],
            segs=32, cap_ends=True,
        )
        bm_add_segmented_tube(
            bm_bars, p_dm + dir_d * 0.210, p_da - dir_d * 0.046,
            [
                (0.00, 0.028), (0.04, 0.044), (0.09, 0.044), (0.14, 0.040),
                (0.82, 0.040), (0.86, 0.046), (0.94, 0.046), (1.00, 0.026),
            ],
            segs=32, cap_ends=True,
        )
        for p_hd, s_hd in ((p_df + dir_d * 0.095, +1.0), (p_da - dir_d * 0.095, -1.0)):
            bm_add_bevel_locked_hex(bm_bars, p_hd, p_hd + dir_d * (s_hd * 0.038),
                                    [(0.0, 0.040), (0.20, 0.047), (0.80, 0.047), (1.0, 0.040)], corner_bevel=0.14, cap_ends=True)
            bm_add_grease_zerk(bm_bars, p_hd + Vector((sx * 0.046, 0, 0)), Vector((sx, 0, 0)), scale=1.0)

            p_cl_c = p_hd - dir_d * (s_hd * 0.024)
            m_cl = Matrix.Identity(4)
            m_cl[0][0], m_cl[1][0], m_cl[2][0] = 1.0, 0.0, 0.0
            m_cl[0][1], m_cl[1][1], m_cl[2][1] = dir_d.x, dir_d.y, dir_d.z
            m_cl[0][2], m_cl[1][2], m_cl[2][2] = up_d.x,  up_d.y,  up_d.z
            m_cl[0][3], m_cl[1][3], m_cl[2][3] = p_cl_c.x, p_cl_c.y, p_cl_c.z
            b_pcl = bmesh.new()
            for sx_ear in (-0.011, +0.011):
                _bm_add_box_x_yz(b_pcl, sx_ear - 0.006, sx_ear + 0.006, -0.022, +0.022, 0.038, 0.064, nx=1, ny=2, nz=2, chamfer=0.0010)
            bm_add_aerospace_bolt(b_pcl, Vector((+0.017, 0.0, 0.053)), Vector((1, 0, 0)), head_r=0.0052, height=0.0048, with_12pt_nut_depth=0.034)
            f0_dbsh = len(b_pcl.faces)
            bm_add_segmented_tube(b_pcl, Vector((-0.017, 0.0, 0.053)), Vector((+0.017, 0.0, 0.053)), [(0.0, 0.0035), (1.0, 0.0035)], segs=14, cap_ends=True)
            bm_add_helical_thread_cylinder(b_pcl, Vector((-0.013, 0.0, 0.053)), Vector((+0.013, 0.0, 0.053)), r_core=0.0033, thread_h=0.00062, n_turns=6, segs_turn=14, cap_core=False)
            bm_fill_unset_from(b_pcl, f0_dbsh, MAT_FASTENER)
            bm_merge(bm_bars, b_pcl, m_cl)
        bm_add_clevis_plate(bm_bars, (p_df.y, p_df.z), (p_df.y + dir_d.y * 0.058, p_df.z + dir_d.z * 0.058), 0.032, 0.028, xb - 0.015, xb + 0.015, arc_segs=17)
        bm_add_clevis_plate(bm_bars, (p_da.y, p_da.z), (p_da.y - dir_d.y * 0.058, p_da.z - dir_d.z * 0.058), 0.032, 0.028, xb - 0.015, xb + 0.015, arc_segs=17)

    _sd_stand.add_cradle_04_turnbuckle_and_hoist_details(bm_bars, Y_FWD_ARCH, Y_AFT_ARCH, _bm_add_steel_plate)
    create_mesh_obj("Stand_Cradle_04_TopTieBar_And_DiagonalBraces", bm_bars, coll_stand, clay)


    bm_tr = bmesh.new()
    Z_AXLE = -1.320


    for y_eb_tr, dir_eb in ((-4.268, -1.0), (-0.772, +1.0)):
        _bm_add_steel_plate(bm_tr, -X_RAIL + 0.078, +X_RAIL - 0.078, y_eb_tr - 0.014, y_eb_tr + 0.014, -1.012, -0.908, nu=18, nv=3, chamfer=0.0018)
        for x_ebs in (-0.380, 0.0, +0.380):
            _bm_add_steel_plate(bm_tr, x_ebs - 0.007, x_ebs + 0.007, y_eb_tr - dir_eb * 0.048, y_eb_tr, -1.008, -0.912, nu=2, nv=2, chamfer=0.0014)
        for sx_erb in (-1.0, +1.0):

            _bm_add_steel_plate(bm_tr, sx_erb * 0.525, sx_erb * 0.586, y_eb_tr + dir_eb * 0.012, y_eb_tr + dir_eb * 0.019, -1.004, -0.916, nu=2, nv=2, chamfer=0.0010)
            for dz_bb in (-0.986, -0.934):
                bm_add_aerospace_bolt(bm_tr, Vector((sx_erb * 0.556, y_eb_tr + dir_eb * 0.019, dz_bb)), Vector((0.0, dir_eb, 0.0)), head_r=0.0048, height=0.0044)
            f0_erb = len(bm_tr.faces)
            _bm_add_steel_plate(bm_tr, sx_erb * 0.365, sx_erb * 0.485, y_eb_tr + dir_eb * 0.013, y_eb_tr + dir_eb * 0.017, -0.986, -0.934, nu=3, nv=2, chamfer=0.0006)
            bm_fill_unset_from(bm_tr, f0_erb, MAT_THREAD)
            for dx_er in (0.375, 0.475):
                bm_add_dome_rivet(bm_tr, Vector((sx_erb * dx_er, y_eb_tr + dir_eb * 0.017, -0.960)), Vector((0.0, dir_eb, 0.0)), radius=0.0020)

    f0_fbp = len(bm_tr.faces)
    _bm_add_steel_plate(bm_tr, -0.135, +0.135, -4.286, -4.281, -0.988, -0.932, nu=4, nv=2, chamfer=0.0006)
    bm_fill_unset_from(bm_tr, f0_fbp, MAT_PLATE_BASE)
    for x_rv in (-0.125, +0.125):
        for z_rv in (-0.980, -0.940):
            bm_add_dome_rivet(bm_tr, Vector((x_rv, -4.286, z_rv)), Vector((0.0, -1.0, 0.0)), radius=0.0018)

    for sx in (-1.0, +1.0):
        xc = sx * X_RAIL
        _bm_add_steel_plate(bm_tr, xc - 0.082, xc + 0.082, -4.280, -0.760, -1.012, -0.992, nu=5, nv=34)
        _bm_add_steel_plate(bm_tr, xc - 0.082, xc + 0.082, -4.280, -0.760, -0.926, -0.906, nu=5, nv=34)
        _bm_add_steel_plate(bm_tr, xc - 0.026, xc + 0.026, -4.270, -0.770, -0.993, -0.925, nu=34, nv=3)

        for y_tsf in (-3.725, -3.110, -2.280, -1.805, -1.160, -0.990):
            for side_ts in (-1.0, +1.0):
                _bm_add_steel_plate(bm_tr, xc + side_ts * 0.025, xc + side_ts * 0.076, y_tsf - 0.006, y_tsf + 0.006, -0.993, -0.925, nu=2, nv=2, chamfer=0.0010)
        for y_twl in (-4.140, -3.520, -1.880, -0.940):
            _bm_add_steel_plate(bm_tr, xc - 0.088, xc + 0.088, y_twl - 0.055, y_twl + 0.055, -0.945, -0.882, nu=2, nv=2, chamfer=0.0018)
            bm_add_spline_tube(
                bm_tr,
                [Vector((xc + sx * 0.084, y_twl - 0.052, -0.944)), Vector((xc + sx * 0.084, y_twl + 0.052, -0.944))],
                radius=0.0024, n_samples=16, segs=8, cap_ends=True, corrugations=14,
            )

            for sy_tw in (-1.0, +1.0):
                bm_add_spline_tube(
                    bm_tr,
                    [Vector((xc + sx * 0.083, y_twl + sy_tw * 0.055, -0.943)), Vector((xc + sx * 0.083, y_twl + sy_tw * 0.055, -0.907))],
                    radius=0.0022, n_samples=12, segs=8, cap_ends=True, corrugations=10,
                )
            for dy_tb in (-0.034, +0.034):
                f0_tw = len(bm_tr.faces)
                bm_add_chamfered_cyl_x(bm_tr, y_twl + dy_tb, -0.914, min(xc + sx * 0.0875, xc + sx * 0.0898), max(xc + sx * 0.0875, xc + sx * 0.0898),
                                       radius=0.0094, segs=20, hollow_r=0.0048)
                bm_fill_unset_from(bm_tr, f0_tw, MAT_FASTENER)
                bm_add_aerospace_bolt(bm_tr, Vector((xc + sx * 0.0896, y_twl + dy_tb, -0.914)), Vector((sx, 0, 0)), head_r=0.0052, height=0.0048)


        for y_jk in (-4.210, -0.840):
            xj_c = xc + sx * 0.114
            _bm_add_steel_plate(bm_tr, xc + sx * 0.080, xj_c + sx * 0.008, y_jk - 0.048, y_jk + 0.048, -1.008, -0.912, nu=2, nv=2, chamfer=0.0016)
            for dy_jb in (-0.034, +0.034):
                for dz_jb in (-0.992, -0.928):
                    bm_add_aerospace_bolt(bm_tr, Vector((xc + sx * 0.082, y_jk + dy_jb, dz_jb)), Vector((sx, 0, 0)), head_r=0.0038, height=0.0036)

            f0_jk_b = len(bm_tr.faces)
            bm_add_segmented_tube(bm_tr, Vector((xj_c, y_jk, -1.116)), Vector((xj_c, y_jk, -0.848)),
                                  [(0.0, 0.035), (0.86, 0.035), (0.90, 0.043), (0.98, 0.043), (1.0, 0.036)], segs=26, cap_ends=True)
            bm_fill_unset_from(bm_tr, f0_jk_b, MAT_AVIONICS_BOX)

            f0_jk_sc = len(bm_tr.faces)
            bm_add_helical_thread_cylinder(
                bm_tr, Vector((xj_c, y_jk, -1.170)), Vector((xj_c, y_jk, -1.112)),
                r_core=0.0185, thread_h=0.0028, n_turns=7, segs_turn=18, cap_core=True,
            )
            bm_fill_unset_from(bm_tr, f0_jk_sc, MAT_FASTENER)

            f0_jk_ft = len(bm_tr.faces)
            bm_add_segmented_tube(bm_tr, Vector((xj_c, y_jk, -1.188)), Vector((xj_c, y_jk, -1.168)),
                                  [(0.0, 0.062), (0.55, 0.062), (1.0, 0.028)], segs=28, cap_ends=True)
            for k_jpad in range(4):
                ang_jp = 0.25 * math.pi + 0.5 * math.pi * k_jpad
                b_jpg = bmesh.new()
                _bm_add_box_x_yz(b_jpg, -0.004, +0.004, 0.020, 0.056, -1.180, -1.164, nx=1, ny=2, nz=1, chamfer=0.0008)
                bm_merge(bm_tr, b_jpg, Matrix.Translation((xj_c, y_jk, 0.0)) @ Matrix.Rotation(ang_jp, 4, 'Z'))
            bm_fill_unset_from(bm_tr, f0_jk_ft, MAT_PUMPS_OILTANK)

            bm_add_grease_zerk(bm_tr, Vector((xj_c, y_jk, -0.848)), Vector((0, 0, 1)), scale=1.1)
            for kb_jk in range(4):
                ajk = 0.25 * math.pi + 0.5 * math.pi * kb_jk
                bm_add_aerospace_bolt(bm_tr, Vector((xj_c + 0.031 * math.cos(ajk), y_jk + 0.031 * math.sin(ajk), -0.849)),
                                      Vector((0, 0, 1)), head_r=0.0030, height=0.0028)
            f0_jk_cr = len(bm_tr.faces)
            bm_add_chamfered_cyl_x(bm_tr, y_jk, -0.872, min(xj_c, xj_c + sx * 0.052), max(xj_c, xj_c + sx * 0.052), radius=0.011, segs=18)
            bm_add_clevis_plate(bm_tr, (y_jk, -0.872), (y_jk + (0.028 if y_jk < -2.5 else -0.028), -1.005),
                                0.011, 0.009, min(xj_c + sx * 0.048, xj_c + sx * 0.058), max(xj_c + sx * 0.048, xj_c + sx * 0.058), arc_segs=11)
            bm_fill_unset_from(bm_tr, f0_jk_cr, MAT_STAND_STEEL)
            y_hnd_j = y_jk + (0.028 if y_jk < -2.5 else -0.028)
            f0_jk_gr = len(bm_tr.faces)
            bm_add_segmented_tube(
                bm_tr, Vector((xj_c + sx * 0.056, y_hnd_j, -1.005)), Vector((xj_c + sx * 0.112, y_hnd_j, -1.005)),
                [(0.0, 0.0075), (0.18, 0.0108), (0.86, 0.0108), (1.0, 0.0080)], segs=16, cap_ends=True,
            )
            bm_fill_unset_from(bm_tr, f0_jk_gr, MAT_RUBBER_DARK)
            bm_add_clevis_pin_assembly_x(bm_tr, y_jk, -0.960, min(xc + sx * 0.076, xj_c + sx * 0.042), max(xc + sx * 0.076, xj_c + sx * 0.042),
                                         pin_r=0.0080, segs=18, compact_ends=True, flip_x=(sx < 0.0))
            f0_jk_ln = len(bm_tr.faces)
            bm_add_spline_tube(
                bm_tr,
                [
                    Vector((xj_c + sx * 0.040, y_jk, -0.960)),
                    Vector((xj_c + sx * 0.036, y_jk + 0.034, -0.992)),
                    Vector((xc + sx * 0.088,   y_jk + 0.040, -0.985)),
                ],
                radius=0.0014, n_samples=16, segs=10, cap_ends=True,
            )
            bm_fill_unset_from(bm_tr, f0_jk_ln, MAT_LOCKWIRE)


        xw_in, xw_out = min(xc + sx * 0.082, xc + sx * 0.172), max(xc + sx * 0.082, xc + sx * 0.172)
        _bm_add_steel_plate(bm_tr, xw_in, xw_out, -2.836, -2.660, -1.012, -1.002, nu=3, nv=4, chamfer=0.0012)
        for dy_wc in (-0.082, +0.082):
            _bm_add_steel_plate(bm_tr, xw_in, xw_out, -2.748 + dy_wc - 0.006, -2.748 + dy_wc + 0.006, -1.003, -0.954, nu=2, nv=2, chamfer=0.0010)
        bm_add_spline_tube(
            bm_tr,
            [Vector((xc + sx * 0.084, -2.830, -1.002)), Vector((xc + sx * 0.084, -2.666, -1.002))],
            radius=0.0024, n_samples=20, segs=8, cap_ends=True, corrugations=16,
        )
        if sx < 0.0:
            b_chk = bmesh.new()
            t_ck_y = subdivide_1d_with_support_loops(6, sup1=0.04, sup2=0.14)
            t_ck_x = subdivide_1d_with_support_loops(4, sup1=0.05, sup2=0.16)
            gb_ck, gt_ck = [], []
            for fy_ck in t_ck_y:
                yv_ck = -2.820 + 0.144 * fy_ck
                h_tri = 0.076 * (1.0 - abs(-1.0 + 2.0 * fy_ck) * 0.82)
                rb_ck, rt_ck = [], []
                for fx_ck in t_ck_x:
                    xv_ck = xw_in + 0.010 + (xw_out - xw_in - 0.020) * fx_ck
                    tread_ck = 0.0020 * math.cos(2.0 * math.pi * 5.0 * fy_ck) if 0.10 < fy_ck < 0.90 else 0.0
                    rb_ck.append(b_chk.verts.new((xv_ck, yv_ck, -1.001)))
                    rt_ck.append(b_chk.verts.new((xv_ck, yv_ck, -1.001 + h_tri + tread_ck)))
                gb_ck.append(rb_ck)
                gt_ck.append(rt_ck)
            bm_build_beveled_quad_shell(b_chk, gb_ck, gt_ck, bulge_scale=0.05, max_bevel=0.0024, inset_bevel=True)
            bm_add_chamfered_cyl_x(b_chk, -2.748, -0.970, xw_in + 0.010, xw_out - 0.010, radius=0.013, segs=18, hollow_r=0.0075)
            bm_merge(bm_tr, b_chk, default_mat_id=MAT_RUBBER_DARK)
        else:
            f0_ech = len(bm_tr.faces)
            bm_add_spline_tube(
                bm_tr,
                [
                    Vector((xw_out - 0.006, -2.670, -0.962)),
                    Vector((xw_out + 0.010, -2.680, -0.996)),
                    Vector((xw_out + 0.008, -2.725, -1.028)),
                    Vector((xw_out - 0.008, -2.770, -1.008)),
                    Vector((xw_out - 0.016, -2.796, -0.996)),
                ],
                radius=0.0020, n_samples=22, segs=10, cap_ends=True,
            )
            bm_fill_unset_from(bm_tr, f0_ech, MAT_LOCKWIRE)

    for y_tcb in (-4.180, -3.280, -2.080, -0.880):
        _bm_add_steel_plate(bm_tr, -X_RAIL + 0.076, +X_RAIL - 0.076, y_tcb - 0.048, y_tcb + 0.048, -1.008, -0.912, nu=8, nv=2)


    for sx_pl in (-1.0, +1.0):
        xc_pl = sx_pl * X_RAIL
        for (ya_tp, yb_tp) in ((-2.235, -2.015), (-1.450, -1.210)):
            f0_tp = len(bm_tr.faces)
            _bm_add_steel_plate(bm_tr, xc_pl + sx_pl * 0.025, xc_pl + sx_pl * 0.030, ya_tp, yb_tp, -0.985, -0.932, nu=3, nv=2, chamfer=0.0006)
            bm_fill_unset_from(bm_tr, f0_tp, MAT_PLATE_BASE)
            for y_rv in (ya_tp + 0.008, yb_tp - 0.008):
                for z_rv in (-0.979, -0.938):
                    bm_add_dome_rivet(bm_tr, Vector((xc_pl + sx_pl * 0.030, y_rv, z_rv)), Vector((sx_pl, 0, 0)), radius=0.0016)


    y_hb = -3.635
    f0_hbh = len(bm_tr.faces)
    _bm_add_steel_plate(bm_tr, +X_RAIL + 0.080, +X_RAIL + 0.098, y_hb - 0.060, y_hb + 0.060, -1.005, -0.922, nu=3, nv=2, chamfer=0.0014)
    _bm_add_steel_plate(bm_tr, +X_RAIL + 0.096, +X_RAIL + 0.104, y_hb - 0.050, y_hb + 0.050, -1.002, -0.986, nu=2, nv=1, chamfer=0.0010)
    bm_fill_unset_from(bm_tr, f0_hbh, MAT_AVIONICS_BOX)
    for dy_hb in (-0.046, +0.046):
        for dz_hb in (-0.992, -0.934):
            f0_hbw = len(bm_tr.faces)
            bm_add_chamfered_cyl_x(bm_tr, y_hb + dy_hb, dz_hb, +X_RAIL + 0.0975, +X_RAIL + 0.0994, radius=0.0068, segs=18, hollow_r=0.0036)
            bm_fill_unset_from(bm_tr, f0_hbw, MAT_FASTENER)
            bm_add_aerospace_bolt(bm_tr, Vector((+X_RAIL + 0.0992, y_hb + dy_hb, dz_hb)), Vector((1, 0, 0)), head_r=0.0038, height=0.0036)

    f0_hbr = len(bm_tr.faces)
    bm_add_chamfered_cyl_x(bm_tr, y_hb, -0.956, +X_RAIL + 0.096, +X_RAIL + 0.1004, radius=0.044, segs=36)
    bm_add_chamfered_cyl_x(bm_tr, y_hb, -0.956, +X_RAIL + 0.0985, +X_RAIL + 0.1030, radius=0.044, segs=36, hollow_r=0.0345)
    bm_add_chamfered_cyl_x(bm_tr, y_hb, -0.956, +X_RAIL + 0.0985, +X_RAIL + 0.1028, radius=0.0205, segs=28, hollow_r=0.0065)
    for kt_r in range(12):
        ang_rt = math.radians(-52.0 + kt_r * 9.5)
        b_rt = bmesh.new()
        _bm_add_box_x_yz(b_rt, +X_RAIL + 0.0965, +X_RAIL + 0.1025, -0.0024, +0.0024, 0.041, 0.0488, nx=1, ny=1, nz=1, chamfer=0.0005)
        bm_merge(bm_tr, b_rt, Matrix.Translation((0.0, y_hb, -0.956)) @ Matrix.Rotation(-ang_rt, 4, 'X'))

    bm_add_clevis_plate(bm_tr, (y_hb - 0.014, -0.916), (y_hb + 0.002, -0.908), 0.0065, 0.0045, +X_RAIL + 0.097, +X_RAIL + 0.1032, arc_segs=9)
    bm_fill_unset_from(bm_tr, f0_hbr, MAT_STAND_STEEL)

    for ang_rpin in (-160.0, -90.0, -20.0):
        ar_p = math.radians(ang_rpin)
        bm_add_dome_rivet(bm_tr, Vector((+X_RAIL + 0.1004, y_hb + 0.0275 * math.cos(ar_p), -0.956 + 0.0275 * math.sin(ar_p))), Vector((1, 0, 0)), radius=0.0028)
    f0_hbpw = len(bm_tr.faces)
    bm_add_chamfered_cyl_x(bm_tr, y_hb, -0.956, +X_RAIL + 0.1135, +X_RAIL + 0.1160, radius=0.0112, segs=24, hollow_r=0.0058)
    bm_fill_unset_from(bm_tr, f0_hbpw, MAT_FASTENER)
    bm_add_aerospace_bolt(bm_tr, Vector((+X_RAIL + 0.1158, y_hb, -0.956)), Vector((1, 0, 0)), head_r=0.0062, height=0.0050)

    f0_hbl = len(bm_tr.faces)
    bm_add_clevis_plate(bm_tr, (y_hb, -0.956), (y_hb + 0.046, -0.785), 0.015, 0.011, +X_RAIL + 0.103, +X_RAIL + 0.114, arc_segs=13)
    bm_fill_unset_from(bm_tr, f0_hbl, MAT_AVIONICS_BOX)
    f0_hbp = len(bm_tr.faces)
    bm_add_segmented_tube(bm_tr, Vector((+X_RAIL + 0.108, y_hb + 0.012, -0.918)), Vector((+X_RAIL + 0.108, y_hb + 0.042, -0.808)),
                          [(0.0, 0.0032), (1.0, 0.0032)], segs=12, cap_ends=True)

    bm_add_segmented_tube(bm_tr, Vector((+X_RAIL + 0.108, y_hb + 0.034, -0.826)), Vector((+X_RAIL + 0.108, y_hb + 0.037, -0.814)),
                          [(0.0, 0.0132), (0.20, 0.0152), (0.80, 0.0152), (1.0, 0.0136)], segs=24, cap_ends=True)
    bm_fill_unset_from(bm_tr, f0_hbp, MAT_RIGID_TUBE)
    f0_hbg = len(bm_tr.faces)
    bm_add_segmented_tube(bm_tr, Vector((+X_RAIL + 0.108, y_hb + 0.036, -0.818)), Vector((+X_RAIL + 0.108, y_hb + 0.056, -0.738)),
                          [(0.0, 0.0125), (0.14, 0.0158), (0.86, 0.0158), (1.0, 0.0118)], segs=18, cap_ends=True)
    bm_fill_unset_from(bm_tr, f0_hbg, MAT_STAND_STEEL)

    f0_hbf = len(bm_tr.faces)
    bm_add_bevel_locked_hex(bm_tr, Vector((+X_RAIL + 0.104, y_hb - 0.006, -0.994)), Vector((+X_RAIL + 0.104, y_hb - 0.006, -1.010)),
                            [(0.0, 0.0065), (0.30, 0.0080), (0.80, 0.0080), (1.0, 0.0060)], corner_bevel=0.14, cap_ends=True)
    bm_fill_unset_from(bm_tr, f0_hbf, MAT_BRASS_FITTING)
    f0_hbc = len(bm_tr.faces)
    bm_add_spline_tube(
        bm_tr,
        [
            Vector((+X_RAIL + 0.104, y_hb - 0.006, -1.006)),
            Vector((+X_RAIL + 0.088, y_hb + 0.095, -1.022)),
            Vector((+0.420,          -3.420,       -1.022)),
            Vector((+0.340,          -2.680,       -1.022)),
            Vector((+0.250,          -2.040,       -1.026)),
            Vector((+0.110,          -1.825,       -1.140)),
            Vector((0.000,           -1.720,       -1.235)),
        ],
        radius=0.0036, n_samples=46, segs=12, cap_ends=True,
    )
    for sx_pbc in (-1.0, +1.0):
        bm_add_spline_tube(
            bm_tr,
            [
                Vector((sx_pbc * 0.082, -1.720, -1.235)),
                Vector((sx_pbc * 0.285, -1.605, -1.238)),
                Vector((sx_pbc * 0.476, -1.515, -1.232)),
            ],
            radius=0.0030, n_samples=22, segs=12, cap_ends=True,
        )
    bm_fill_unset_from(bm_tr, f0_hbc, MAT_RUBBER_DARK)

    for x_hc, y_hc in ((+0.395, -3.280), (+0.265, -2.080)):
        f0_hcl = len(bm_tr.faces)
        _bm_add_steel_plate(bm_tr, x_hc - 0.008, x_hc + 0.008, y_hc - 0.009, y_hc + 0.009, -1.030, -1.007, nu=1, nv=1, chamfer=0.0006)
        bm_fill_unset_from(bm_tr, f0_hcl, MAT_STAND_STEEL)
        bm_add_aerospace_bolt(bm_tr, Vector((x_hc, y_hc, -1.030)), Vector((0, 0, -1)), head_r=0.0030, height=0.0028)
    f0_eqb = len(bm_tr.faces)
    _bm_add_steel_plate(bm_tr, -0.095, +0.095, -1.732, -1.708, -1.244, -1.226, nu=3, nv=1, chamfer=0.0012)
    bm_fill_unset_from(bm_tr, f0_eqb, MAT_STAND_STEEL)
    for sx_eqp in (-0.082, 0.0, +0.082):
        bm_add_aerospace_bolt(bm_tr, Vector((sx_eqp, -1.720, -1.226)), Vector((0, 0, 1)), head_r=0.0042, height=0.0038)


    for y_ax, is_front_ax in ((-3.880, True), (-1.495, False)):
        f0_axb = len(bm_tr.faces)
        _bm_add_steel_plate(bm_tr, -0.448, +0.448, y_ax - 0.059, y_ax + 0.059, -1.388, -1.252, nu=10, nv=3, chamfer=0.0022)
        for x_axr in (-0.340, 0.0, +0.340):
            _bm_add_steel_plate(bm_tr, x_axr - 0.009, x_axr + 0.009, y_ax - 0.066, y_ax + 0.066, -1.392, -1.248, nu=2, nv=2, chamfer=0.0014)

            for sy_aw in (-1.0, +1.0):
                bm_add_spline_tube(
                    bm_tr,
                    [Vector((x_axr - 0.009, y_ax + sy_aw * 0.059, -1.386)), Vector((x_axr - 0.009, y_ax + sy_aw * 0.059, -1.254))],
                    radius=0.0026, n_samples=16, segs=8, cap_ends=True, corrugations=14,
                )
        bm_fill_unset_from(bm_tr, f0_axb, MAT_AVIONICS_BOX)


        for sx_sp in (-1.0, +1.0):
            x_sp = sx_sp * 0.395
            f0_lf = len(bm_tr.faces)
            _bm_add_steel_plate(bm_tr, x_sp - 0.038, x_sp + 0.038, y_ax - 0.062, y_ax + 0.062, -1.253, -1.236, nu=2, nv=2, chamfer=0.0014)
            _bm_add_steel_plate(bm_tr, x_sp - 0.042, x_sp + 0.042, y_ax - 0.074, y_ax + 0.074, -1.404, -1.388, nu=2, nv=2, chamfer=0.0014)
            for k_lf in range(7):
                hl_lf = 0.256 - k_lf * 0.028
                z_lf0 = -1.236 + (6 - k_lf) * 0.0092
                _bm_add_steel_plate(bm_tr, x_sp - 0.031, x_sp + 0.031, y_ax - hl_lf, y_ax + hl_lf, z_lf0, z_lf0 + 0.0086, nu=2, nv=4, chamfer=0.0010)
            _bm_add_steel_plate(bm_tr, x_sp - 0.040, x_sp + 0.040, y_ax - 0.295, y_ax + 0.295, -1.012, -0.988, nu=2, nv=5, chamfer=0.0014)
            for s_end_lf in (-1.0, +1.0):
                y_eye_lf = y_ax + s_end_lf * 0.256
                bm_add_chamfered_cyl_x(bm_tr, y_eye_lf, -1.174, x_sp - 0.034, x_sp + 0.034, radius=0.017, segs=20, hollow_r=0.008)
                y_hng_top = y_eye_lf + (0.018 if s_end_lf > 0 else 0.0)
                for dx_shk in (-0.036, +0.036):
                    _bm_add_steel_plate(bm_tr, x_sp + dx_shk - 0.006, x_sp + dx_shk + 0.006, y_hng_top - 0.026, y_hng_top + 0.026, -1.095, -0.990, nu=1, nv=2, chamfer=0.0010)
                    bm_add_clevis_plate(bm_tr, (y_hng_top, -1.078), (y_eye_lf, -1.174), 0.016, 0.016, x_sp + dx_shk - 0.005, x_sp + dx_shk + 0.005, arc_segs=11)
            for dy_clp in (-0.145, +0.145):
                _bm_add_steel_plate(bm_tr, x_sp - 0.035, x_sp + 0.035, y_ax + dy_clp - 0.010, y_ax + dy_clp + 0.010, -1.238, -1.168, nu=2, nv=2, chamfer=0.0008)
            _bm_add_steel_plate(bm_tr, x_sp - 0.038, x_sp + 0.038, y_ax - 0.052, y_ax + 0.052, -1.172, -1.156, nu=2, nv=2, chamfer=0.0012)
            bm_fill_unset_from(bm_tr, f0_lf, MAT_PUMPS_OILTANK)


            f0_ub = len(bm_tr.faces)
            for s_end_lf in (-1.0, +1.0):
                y_eye_lf = y_ax + s_end_lf * 0.256
                y_hng_top = y_eye_lf + (0.018 if s_end_lf > 0 else 0.0)
                bm_add_clevis_pin_assembly_x(bm_tr, y_eye_lf, -1.174, x_sp - 0.044, x_sp + 0.044, pin_r=0.0075, segs=16, compact_ends=True)
                bm_add_clevis_pin_assembly_x(bm_tr, y_hng_top, -1.078, x_sp - 0.044, x_sp + 0.044, pin_r=0.0075, segs=16, compact_ends=True)
            for dy_ub in (-0.064, +0.064):
                ub_pts = [
                    Vector((x_sp - 0.033, y_ax + dy_ub, -1.404)),
                    Vector((x_sp - 0.033, y_ax + dy_ub, -1.162)),
                    Vector((x_sp,         y_ax + dy_ub, -1.146)),
                    Vector((x_sp + 0.033, y_ax + dy_ub, -1.162)),
                    Vector((x_sp + 0.033, y_ax + dy_ub, -1.404)),
                ]
                bm_add_spline_tube(bm_tr, ub_pts, radius=0.0058, n_samples=22, segs=14, cap_ends=True)
                for dx_ub in (-0.033, +0.033):
                    bm_add_bevel_locked_hex(
                        bm_tr, Vector((x_sp + dx_ub, y_ax + dy_ub, -1.404)), Vector((x_sp + dx_ub, y_ax + dy_ub, -1.418)),
                        [(0.0, 0.0086), (0.25, 0.0105), (0.80, 0.0105), (1.0, 0.0082)], corner_bevel=0.14, cap_ends=True,
                    )
                    bm_add_helical_thread_cylinder(
                        bm_tr, Vector((x_sp + dx_ub, y_ax + dy_ub, -1.417)), Vector((x_sp + dx_ub, y_ax + dy_ub, -1.434)),
                        r_core=0.0052, thread_h=0.0011, n_turns=5, segs_turn=16, cap_core=True,
                    )
            bm_fill_unset_from(bm_tr, f0_ub, MAT_FASTENER)


            p_shk_b = Vector((sx_sp * 0.305, y_ax + 0.054, -1.264))
            p_shk_t = Vector((sx_sp * 0.345, y_ax + 0.168, -1.016))
            f0_shk_bk = len(bm_tr.faces)
            bm_add_segmented_tube(
                bm_tr, p_shk_b.lerp(p_shk_t, 0.06), p_shk_b.lerp(p_shk_t, 0.52),
                [(0.0, 0.016), (0.06, 0.022), (0.94, 0.022), (1.0, 0.019)], segs=24, cap_ends=True,
            )
            bm_add_segmented_tube(
                bm_tr, p_shk_b.lerp(p_shk_t, 0.61), p_shk_b.lerp(p_shk_t, 0.94),
                [(0.0, 0.0265), (0.04, 0.0265), (0.94, 0.0265), (1.0, 0.018)], segs=24, cap_ends=True,
            )
            bm_fill_unset_from(bm_tr, f0_shk_bk, MAT_AVIONICS_BOX)
            f0_shk_gl = len(bm_tr.faces)
            bm_add_segmented_tube(
                bm_tr, p_shk_b.lerp(p_shk_t, 0.51), p_shk_b.lerp(p_shk_t, 0.545),
                [(0.0, 0.018), (0.20, 0.0175), (0.85, 0.0175), (1.0, 0.013)], segs=20, cap_ends=True,
            )
            bm_fill_unset_from(bm_tr, f0_shk_gl, MAT_STAND_STEEL)
            f0_shk_rd = len(bm_tr.faces)
            bm_add_segmented_tube(
                bm_tr, p_shk_b.lerp(p_shk_t, 0.53), p_shk_b.lerp(p_shk_t, 0.63),
                [(0.0, 0.0092), (1.0, 0.0092)], segs=20, cap_ends=True,
            )
            bm_fill_unset_from(bm_tr, f0_shk_rd, MAT_RIGID_TUBE)
            for p_se in (p_shk_b, p_shk_t):
                f0_eye = len(bm_tr.faces)
                bm_add_chamfered_cyl_x(bm_tr, p_se.y, p_se.z, p_se.x - 0.015, p_se.x + 0.015, radius=0.0165, segs=20, hollow_r=0.0115)
                bm_fill_unset_from(bm_tr, f0_eye, MAT_PUMPS_OILTANK)
                f0_bsh = len(bm_tr.faces)
                bm_add_chamfered_cyl_x(bm_tr, p_se.y, p_se.z, p_se.x - 0.017, p_se.x + 0.017, radius=0.0115, segs=18, hollow_r=0.0068)
                bm_fill_unset_from(bm_tr, f0_bsh, MAT_RUBBER_DARK)
                bm_add_aerospace_bolt(bm_tr, Vector((p_se.x + sx_sp * 0.017, p_se.y, p_se.z)), Vector((sx_sp, 0, 0)), head_r=0.0064, height=0.0052)


        f0_brk_t = len(bm_tr.faces)
        _bm_add_steel_plate(bm_tr, -0.022, +0.022, y_ax + 0.056, y_ax + 0.074, -1.282, -1.250, nu=2, nv=1, chamfer=0.0010)
        bm_fill_unset_from(bm_tr, f0_brk_t, MAT_BRASS_FITTING)
        for sx_bl in (-1.0, +1.0):
            f0_blh = len(bm_tr.faces)
            bl_hard = [
                Vector((sx_bl * 0.020, y_ax + 0.065, -1.266)),
                Vector((sx_bl * 0.350, y_ax + 0.065, -1.266)),
            ]
            bm_sweep_centerline_tube(bm_tr, bl_hard, radius=0.0032, segs=14, cap_ends=True)
            bm_fill_unset_from(bm_tr, f0_blh, MAT_RIGID_TUBE)

            for x_pcl in (0.145, 0.275):
                f0_pc = len(bm_tr.faces)
                _bm_add_steel_plate(bm_tr, sx_bl * x_pcl - 0.005, sx_bl * x_pcl + 0.005, y_ax + 0.058, y_ax + 0.070, -1.273, -1.259, nu=1, nv=1, chamfer=0.0005)
                bm_fill_unset_from(bm_tr, f0_pc, MAT_STAND_STEEL)
                bm_add_aerospace_bolt(bm_tr, Vector((sx_bl * x_pcl, y_ax + 0.070, -1.266)), Vector((0, 1, 0)), head_r=0.0026, height=0.0024)
            bm_add_bnut_flare_fitting(bm_tr, Vector((sx_bl * 0.345, y_ax + 0.065, -1.266)), Vector((sx_bl, 0.0, 0.0)), tube_r=0.0032)

            f0_bfh = len(bm_tr.faces)
            bm_add_spline_tube(
                bm_tr,
                [
                    Vector((sx_bl * 0.350, y_ax + 0.065, -1.266)),
                    Vector((sx_bl * 0.420, y_ax + 0.052, -1.242)),
                    Vector((sx_bl * 0.476, y_ax + 0.018, -1.226)),
                ],
                radius=0.0044, n_samples=24, segs=16, cap_ends=True, corrugations=18,
            )
            bm_fill_unset_from(bm_tr, f0_bfh, MAT_BRAID_HOSE)

        for sx in (-1.0, +1.0):
            xc = sx * X_RAIL

            _bm_add_steel_plate(bm_tr, min(sx * 0.435, xc + sx * 0.086), max(sx * 0.435, xc + sx * 0.086), y_ax - 0.340, y_ax + 0.340, -1.022, -1.002, nu=4, nv=5)
            for dy_vb in (-0.285, +0.285):
                bm_add_aerospace_bolt(bm_tr, Vector((sx * 0.485, y_ax + dy_vb, -1.002)), Vector((0, 0, 1)), head_r=0.0052, height=0.0048)
            for dy_leg in (-0.275, +0.275):
                p_top_l = Vector((sx * 0.464, y_ax + dy_leg, -1.012))
                p_bot_l = Vector((sx * 0.448, y_ax + dy_leg * 0.30, -1.288))
                bm_add_flattened_aero_strut(
                    bm_tr, p_top_l, p_bot_l,
                    half_w_mid=0.048, half_h_mid=0.020,
                    end_half_w=0.040, end_half_h=0.018,
                    eye_r=0.040, eye_half_w=0.018, taper_frac=0.16,
                    pin_along_x=True, ribbed_pockets=2,
                )
            f0_vt_dk = len(bm_tr.faces)
            gw_lo, gw_hi = min(sx * 0.444, sx * 0.466), max(sx * 0.444, sx * 0.466)
            bm_add_clevis_plate(bm_tr, (y_ax - 0.205, -1.018), (y_ax, -1.235), 0.0480, 0.0620, gw_lo, gw_hi, arc_segs=13)
            bm_add_clevis_plate(bm_tr, (y_ax + 0.205, -1.018), (y_ax, -1.235), 0.0462, 0.0602, gw_lo + 0.0012, gw_hi - 0.0012, arc_segs=13)
            bm_add_chamfered_cyl_x(bm_tr, y_ax, -1.115, min(sx * 0.440, sx * 0.470), max(sx * 0.440, sx * 0.470), radius=0.036, segs=24, hollow_r=0.020)
            _bm_add_steel_plate(bm_tr, sx * 0.442, sx * 0.492, y_ax - 0.062, y_ax + 0.062, -1.396, -1.258, nu=2, nv=2, chamfer=0.0020)
            bm_fill_unset_from(bm_tr, f0_vt_dk, MAT_AVIONICS_BOX)

            f0_kp_st = len(bm_tr.faces)
            bm_add_segmented_tube(bm_tr, Vector((sx * 0.462, y_ax, -1.414)), Vector((sx * 0.462, y_ax, -1.256)), [(0.0, 0.020), (1.0, 0.020)], segs=24, cap_ends=True)
            bm_add_chamfered_cyl_x(bm_tr, y_ax, Z_AXLE, min(sx * 0.490, sx * 0.550), max(sx * 0.490, sx * 0.550), radius=0.038, segs=28)
            bm_fill_unset_from(bm_tr, f0_kp_st, MAT_STAND_STEEL)

        if is_front_ax:

            f0_fsh = len(bm_tr.faces)
            _bm_add_steel_plate(bm_tr, -0.092, +0.092, y_ax - 0.185, y_ax - 0.052, -1.402, -1.248, nu=3, nv=2, chamfer=0.0020)
            bm_fill_unset_from(bm_tr, f0_fsh, MAT_AVIONICS_BOX)
            f0_fsp = len(bm_tr.faces)
            bm_add_segmented_tube(bm_tr, Vector((0.0, y_ax - 0.122, -1.422)), Vector((0.0, y_ax - 0.122, -1.232)), [(0.0, 0.030), (1.0, 0.030)], segs=24, cap_ends=True)
            bm_fill_unset_from(bm_tr, f0_fsp, MAT_STAND_STEEL)
            f0_fsn = len(bm_tr.faces)
            bm_add_bevel_locked_hex(bm_tr, Vector((0.0, y_ax - 0.122, -1.232)), Vector((0.0, y_ax - 0.122, -1.215)),
                                    [(0.0, 0.024), (0.30, 0.030), (0.80, 0.030), (1.0, 0.022)], corner_bevel=0.14, cap_ends=True)
            bm_fill_unset_from(bm_tr, f0_fsn, MAT_FASTENER)
            for sx, z_link_in, z_link_out in ((-1.0, -1.342, -1.328), (+1.0, -1.384, -1.364)):
                f0_tr_arm = len(bm_tr.faces)
                _bm_add_steel_plate(bm_tr, 0.0, sx * 0.072, y_ax - 0.148, y_ax - 0.096, z_link_in - 0.014, z_link_in + 0.014, nu=2, nv=2, chamfer=0.0014)
                p_tr_in  = Vector((sx * 0.062, y_ax - 0.128, z_link_in))
                p_tr_out = Vector((sx * 0.474, y_ax - 0.088, z_link_out))
                bm_add_segmented_tube(bm_tr, p_tr_in, p_tr_out,
                                      [(0.0, 0.018), (0.08, 0.024), (0.14, 0.014), (0.22, 0.021), (0.78, 0.021), (0.86, 0.014), (0.92, 0.024), (1.0, 0.018)], segs=22, cap_ends=True)
                bm_fill_unset_from(bm_tr, f0_tr_arm, MAT_STAND_STEEL)
                f0_tr_hw = len(bm_tr.faces)
                for f_ta, f_tb, f_hn in ((0.08, 0.21, 0.18), (0.92, 0.79, 0.82)):
                    bm_add_helical_thread_cylinder(
                        bm_tr, p_tr_in.lerp(p_tr_out, min(f_ta, f_tb)), p_tr_in.lerp(p_tr_out, max(f_ta, f_tb)),
                        r_core=0.0135, thread_h=0.0018, n_turns=5, segs_turn=16, cap_core=False,
                    )
                    bm_add_bevel_locked_hex(
                        bm_tr, p_tr_in.lerp(p_tr_out, f_hn - 0.022), p_tr_in.lerp(p_tr_out, f_hn + 0.022),
                        [(0.0, 0.019), (0.25, 0.023), (0.80, 0.023), (1.0, 0.019)], corner_bevel=0.14, cap_ends=True,
                    )
                bm_fill_unset_from(bm_tr, f0_tr_hw, MAT_FASTENER)
            f0_axp_f = len(bm_tr.faces)
            for sx_rf in (-1.0, +1.0):
                _bm_add_steel_plate(bm_tr, sx_rf * 0.315, sx_rf * 0.415, y_ax - 0.063, y_ax - 0.059, -1.342, -1.288, nu=2, nv=2, chamfer=0.0006)
            bm_fill_unset_from(bm_tr, f0_axp_f, MAT_AVIONICS_BOX)


            _bm_add_steel_plate(bm_tr, -0.245, +0.245, y_ax - 0.215, y_ax - 0.165, -1.388, -1.306, nu=6, nv=2, chamfer=0.0018)
            for sx_af in (-1.0, +1.0):
                xh_af = sx_af * 0.205
                for dz_af in (-1.382, -1.312):
                    _bm_add_steel_plate(bm_tr, xh_af - 0.032, xh_af + 0.032, y_ax - 0.285, y_ax - 0.185, dz_af - 0.011, dz_af + 0.011, nu=2, nv=2, chamfer=0.0014)
                for dx_hb_a in (-0.018, +0.018):
                    bm_add_aerospace_bolt(bm_tr, Vector((xh_af + dx_hb_a, y_ax - 0.200, -1.301)), Vector((0, 0, 1)), head_r=0.0046, height=0.0042)
                bm_add_clevis_pin_assembly_x(bm_tr, y_ax - 0.252, -1.347, xh_af - 0.040, xh_af + 0.040, pin_r=0.012, segs=18, compact_ends=True)
                bm_add_flattened_aero_strut(
                    bm_tr,
                    Vector((xh_af, y_ax - 0.252, -1.347)),
                    Vector((sx_af * 0.046, -4.885, -1.347)),
                    half_w_mid=0.030, half_h_mid=0.018,
                    end_half_w=0.026, end_half_h=0.016,
                    eye_r=0.026, eye_half_w=0.016, taper_frac=0.14,
                    pin_along_x=False, ribbed_pockets=3,
                )
            _bm_add_steel_plate(bm_tr, -0.135, +0.135, -4.605, -4.485, -1.362, -1.332, nu=4, nv=2, chamfer=0.0016)
            _bm_add_steel_plate(bm_tr, -0.062, +0.062, -4.965, -4.725, -1.374, -1.320, nu=3, nv=3, chamfer=0.0018)

            _sd_stand.add_trailer_05_surge_brake_assembly(bm_tr, _bm_add_steel_plate)

            f0_tbp = len(bm_tr.faces)
            _bm_add_steel_plate(bm_tr, -0.096, +0.096, y_ax - 0.2195, y_ax - 0.2145, -1.374, -1.320, nu=3, nv=2, chamfer=0.0005)
            bm_fill_unset_from(bm_tr, f0_tbp, MAT_PLATE_BASE)
            for x_trv in (-0.088, +0.088):
                for z_trv in (-1.367, -1.327):
                    bm_add_dome_rivet(bm_tr, Vector((x_trv, y_ax - 0.2195, z_trv)), Vector((0.0, -1.0, 0.0)), radius=0.0015)

            f0_sleg = len(bm_tr.faces)
            _bm_add_steel_plate(bm_tr, +0.142, +0.188, -4.560, -4.520, -1.364, -1.330, nu=2, nv=2, chamfer=0.0012)
            bm_add_segmented_tube(
                bm_tr, Vector((+0.180, -4.540, -1.347)), Vector((+0.120, -4.835, -1.347)),
                [(0.0, 0.015), (0.85, 0.015), (0.90, 0.025), (1.0, 0.025)], segs=18, cap_ends=True,
            )
            bm_fill_unset_from(bm_tr, f0_sleg, MAT_AVIONICS_BOX)
            bm_add_aerospace_bolt(bm_tr, Vector((+0.172, -4.540, -1.330)), Vector((0, 0, 1)), head_r=0.0052, height=0.0046)

            f0_tsh = len(bm_tr.faces)
            bm_add_segmented_tube(
                bm_tr, Vector((0.0, -4.725, -1.347)), Vector((0.0, -5.005, -1.347)),
                [(0.0, 0.024), (0.12, 0.028), (0.78, 0.028), (0.84, 0.042), (0.94, 0.042), (1.0, 0.026)],
                segs=24, cap_ends=True,
            )
            bm_fill_unset_from(bm_tr, f0_tsh, MAT_STAND_STEEL)
            f0_tbt = len(bm_tr.faces)
            bm_add_spline_tube(
                bm_tr, [Vector((0.0, -4.962, -1.347)), Vector((0.0, -4.996, -1.347))],
                radius=0.036, n_samples=16, segs=22, cap_ends=True, corrugations=5,
            )
            bm_fill_unset_from(bm_tr, f0_tbt, MAT_RUBBER_DARK)
            f0_tnu = len(bm_tr.faces)
            bm_add_bevel_locked_hex(
                bm_tr, Vector((0.0, -4.725, -1.347)), Vector((0.0, -4.702, -1.347)),
                [(0.0, 0.022), (0.25, 0.027), (0.80, 0.027), (1.0, 0.021)], corner_bevel=0.14, cap_ends=True,
            )
            bm_add_helical_thread_cylinder(
                bm_tr, Vector((0.0, -4.703, -1.347)), Vector((0.0, -4.680, -1.347)),
                r_core=0.015, thread_h=0.0018, n_turns=5, segs_turn=16, cap_core=True,
            )
            bm_fill_unset_from(bm_tr, f0_tnu, MAT_FASTENER)
            b_lun = bmesh.new()
            prof_lun = [
                (0.021 * math.sin(2.0 * math.pi * ks_l / 16.0), 0.063 + 0.021 * math.cos(2.0 * math.pi * ks_l / 16.0))
                for ks_l in range(16)
            ]
            bm_add_revolved_profile(b_lun, prof_lun, segs=36, close_loop=True, bevel_r=0.0)
            bm_merge(bm_tr, b_lun, Matrix.Translation((0.0, -5.058, -1.347)) @ Matrix.Rotation(0.5 * math.pi, 4, 'X'), default_mat_id=MAT_PUMPS_OILTANK)
            for sx_ch in (-1.0, +1.0):
                for kl_c in range(10):
                    yc_lk = -4.745 - kl_c * 0.024
                    zc_lk = -1.366 - 0.024 * math.sin(math.pi * (kl_c + 0.5) / 10.0)
                    b_clk = bmesh.new()
                    lk_pts = [
                        Vector((0.0, 0.016 * math.cos(2.0 * math.pi * s_lk / 16.0), 0.009 * math.sin(2.0 * math.pi * s_lk / 16.0)))
                        for s_lk in range(17)
                    ]
                    bm_add_spline_tube(b_clk, lk_pts, radius=0.0034, n_samples=24, segs=10, cap_ends=True)
                    rot_lk = Matrix.Rotation(0.5 * math.pi if (kl_c % 2 == 1) else 0.0, 4, 'Y')
                    bm_merge(bm_tr, b_clk, Matrix.Translation((sx_ch * 0.068, yc_lk, zc_lk)) @ rot_lk, default_mat_id=MAT_PUMPS_OILTANK)

            for sx_kp in (-1.0, +1.0):
                f0_kpn = len(bm_tr.faces)
                bm_add_bevel_locked_hex(bm_tr, Vector((sx_kp * 0.462, y_ax, -1.256)), Vector((sx_kp * 0.462, y_ax, -1.242)),
                                        [(0.0, 0.014), (0.25, 0.017), (0.80, 0.017), (1.0, 0.013)], corner_bevel=0.14, cap_ends=True)
                bm_fill_unset_from(bm_tr, f0_kpn, MAT_FASTENER)
                bm_add_grease_zerk(bm_tr, Vector((sx_kp * 0.462, y_ax, -1.242)), Vector((0, 0, 1)), scale=1.0)
                f0_kpa = len(bm_tr.faces)
                bm_add_segmented_tube(bm_tr, Vector((sx_kp * 0.474, y_ax - 0.088, -1.392)), Vector((sx_kp * 0.474, y_ax - 0.088, -1.316)),
                                      [(0.0, 0.018), (0.50, 0.024), (1.0, 0.018)], segs=20, cap_ends=True)
                bm_fill_unset_from(bm_tr, f0_kpa, MAT_AVIONICS_BOX)
        else:

            f0_rsh = len(bm_tr.faces)
            _bm_add_steel_plate(bm_tr, -0.088, +0.088, y_ax + 0.052, y_ax + 0.175, -1.400, -1.252, nu=3, nv=2, chamfer=0.0020)
            bm_fill_unset_from(bm_tr, f0_rsh, MAT_AVIONICS_BOX)
            f0_rsp = len(bm_tr.faces)
            bm_add_segmented_tube(bm_tr, Vector((0.0, y_ax + 0.122, -1.418)), Vector((0.0, y_ax + 0.122, -1.224)),
                                  [(0.0, 0.026), (0.86, 0.026), (0.90, 0.032), (1.0, 0.024)], segs=24, cap_ends=True)
            bm_fill_unset_from(bm_tr, f0_rsp, MAT_STAND_STEEL)

            f0_tph = len(bm_tr.faces)
            bm_add_segmented_tube(bm_tr, Vector((-0.036, y_ax + 0.152, -1.220)), Vector((+0.036, y_ax + 0.152, -1.220)),
                                  [(0.0, 0.0070), (0.20, 0.0085), (0.80, 0.0085), (1.0, 0.0070)], segs=14, cap_ends=True)
            bm_fill_unset_from(bm_tr, f0_tph, MAT_STAND_STEEL)
            f0_tps = len(bm_tr.faces)
            bm_add_segmented_tube(bm_tr, Vector((0.0, y_ax + 0.152, -1.310)), Vector((0.0, y_ax + 0.152, -1.220)),
                                  [(0.0, 0.0085), (1.0, 0.0095)], segs=16, cap_ends=True)
            bm_fill_unset_from(bm_tr, f0_tps, MAT_FASTENER)
            f0_tpl = len(bm_tr.faces)
            bm_add_spline_tube(
                bm_tr,
                [Vector((+0.032, y_ax + 0.152, -1.220)), Vector((+0.058, y_ax + 0.168, -1.256)), Vector((+0.068, y_ax + 0.158, -1.292))],
                radius=0.0015, n_samples=16, segs=10, cap_ends=True,
            )
            bm_fill_unset_from(bm_tr, f0_tpl, MAT_LOCKWIRE)

            f0_rsp_pl = len(bm_tr.faces)
            _bm_add_steel_plate(bm_tr, -0.064, +0.064, y_ax + 0.173, y_ax + 0.178, -1.356, -1.296, nu=2, nv=2, chamfer=0.0005)
            bm_fill_unset_from(bm_tr, f0_rsp_pl, MAT_PLATE_BASE)
            for x_rrv in (-0.056, +0.056):
                for z_rrv in (-1.348, -1.304):
                    bm_add_dome_rivet(bm_tr, Vector((x_rrv, y_ax + 0.178, z_rrv)), Vector((0.0, 1.0, 0.0)), radius=0.0016)
            for sx_r, z_r_in, z_r_out in ((-1.0, -1.342, -1.330), (+1.0, -1.380, -1.362)):
                f0_rr_a = len(bm_tr.faces)
                _bm_add_steel_plate(bm_tr, 0.0, sx_r * 0.070, y_ax + 0.096, y_ax + 0.146, z_r_in - 0.013, z_r_in + 0.013, nu=2, nv=2, chamfer=0.0014)
                p_rr_in  = Vector((sx_r * 0.060, y_ax + 0.125, z_r_in))
                p_rr_out = Vector((sx_r * 0.474, y_ax + 0.088, z_r_out))
                bm_add_segmented_tube(bm_tr, p_rr_in, p_rr_out,
                                      [(0.0, 0.017), (0.08, 0.023), (0.14, 0.0135), (0.22, 0.020), (0.78, 0.020), (0.86, 0.0135), (0.92, 0.023), (1.0, 0.017)], segs=22, cap_ends=True)
                bm_fill_unset_from(bm_tr, f0_rr_a, MAT_STAND_STEEL)
                f0_rr_h = len(bm_tr.faces)
                for f_ra, f_rb, f_rh in ((0.08, 0.21, 0.18), (0.92, 0.79, 0.82)):
                    bm_add_helical_thread_cylinder(
                        bm_tr, p_rr_in.lerp(p_rr_out, min(f_ra, f_rb)), p_rr_in.lerp(p_rr_out, max(f_ra, f_rb)),
                        r_core=0.0130, thread_h=0.0018, n_turns=5, segs_turn=16, cap_core=False,
                    )
                    bm_add_bevel_locked_hex(
                        bm_tr, p_rr_in.lerp(p_rr_out, f_rh - 0.022), p_rr_in.lerp(p_rr_out, f_rh + 0.022),
                        [(0.0, 0.018), (0.25, 0.022), (0.80, 0.022), (1.0, 0.018)], corner_bevel=0.14, cap_ends=True,
                    )
                bm_add_bevel_locked_hex(bm_tr, Vector((sx_r * 0.462, y_ax, -1.256)), Vector((sx_r * 0.462, y_ax, -1.242)),
                                        [(0.0, 0.014), (0.25, 0.017), (0.80, 0.017), (1.0, 0.013)], corner_bevel=0.14, cap_ends=True)
                bm_fill_unset_from(bm_tr, f0_rr_h, MAT_FASTENER)
                bm_add_grease_zerk(bm_tr, Vector((sx_r * 0.462, y_ax, -1.242)), Vector((0, 0, 1)), scale=1.0)
                f0_rk_a = len(bm_tr.faces)
                bm_add_segmented_tube(bm_tr, Vector((sx_r * 0.474, y_ax + 0.088, -1.392)), Vector((sx_r * 0.474, y_ax + 0.088, -1.316)),
                                      [(0.0, 0.018), (0.50, 0.024), (1.0, 0.018)], segs=20, cap_ends=True)
                _bm_add_steel_plate(bm_tr, sx_r * 0.315, sx_r * 0.415, y_ax + 0.059, y_ax + 0.063, -1.342, -1.288, nu=2, nv=2, chamfer=0.0006)
                bm_fill_unset_from(bm_tr, f0_rk_a, MAT_AVIONICS_BOX)

            f0_rth = len(bm_tr.faces)
            _bm_add_steel_plate(bm_tr, -0.095, +0.095, -0.840, -0.818, -1.006, -0.914, nu=3, nv=2, chamfer=0.0016)
            for dz_rp in (-0.985, -0.935):
                _bm_add_steel_plate(bm_tr, -0.048, +0.048, -0.822, -0.752, dz_rp - 0.010, dz_rp + 0.010, nu=2, nv=2, chamfer=0.0014)
            bm_fill_unset_from(bm_tr, f0_rth, MAT_AVIONICS_BOX)
            f0_rtp = len(bm_tr.faces)
            bm_add_segmented_tube(bm_tr, Vector((0.0, -0.782, -1.008)), Vector((0.0, -0.782, -0.906)),
                                  [(0.0, 0.015), (0.84, 0.015), (0.88, 0.024), (1.0, 0.020)], segs=20, cap_ends=True)
            bm_fill_unset_from(bm_tr, f0_rtp, MAT_FASTENER)


    bm_add_bnut_flare_fitting(bm_tr, Vector((+0.046, -4.645, -1.298)), Vector((1.0, 0.0, 0.0)), tube_r=0.0042)
    f0_mbh = len(bm_tr.faces)
    bm_add_spline_tube(
        bm_tr,
        [
            Vector((+0.056, -4.645, -1.298)),
            Vector((+0.110, -4.545, -1.325)),
            Vector((+0.159, -4.350, -1.322)),
            Vector((+0.162, -4.110, -1.292)),
            Vector((+0.125, -3.965, -1.236)),
            Vector((+0.068, -3.860, -1.236)),
            Vector((+0.012, -3.815, -1.266)),
        ],
        radius=0.0044, n_samples=48, segs=16, cap_ends=True, corrugations=28,
    )
    bm_fill_unset_from(bm_tr, f0_mbh, MAT_BRAID_HOSE)
    f0_mbt = len(bm_tr.faces)
    bm_add_spline_tube(
        bm_tr,
        [
            Vector((+0.018, -3.815, -1.258)),
            Vector((+0.260, -3.720, -1.045)),
            Vector((+0.585, -3.520, -0.962)),
            Vector((+0.585, -1.780, -0.962)),
            Vector((+0.260, -1.585, -1.085)),
            Vector((+0.018, -1.430, -1.258)),
        ],
        radius=0.0034, n_samples=48, segs=14, cap_ends=True,
    )
    bm_fill_unset_from(bm_tr, f0_mbt, MAT_RIGID_TUBE)

    _sd_stand.add_trailer_05_chassis_details(bm_tr, X_RAIL, Z_AXLE, _bm_add_steel_plate)
    _hd_detail.add_trailer_05_4w_steering_and_xbracing(bm_tr, X_RAIL, Z_AXLE, _bm_add_steel_plate)
    create_mesh_obj("Stand_Trailer_05_M3000_Steerable_Axles_And_TowBar", bm_tr, coll_stand, clay)


    bm_wh = bmesh.new()
    b_wunit = bmesh.new()


    prof_rim_yz = [
        (-0.027, 0.056), (-0.027, 0.140), (-0.086, 0.152), (-0.114, 0.160), (-0.114, 0.168), (-0.098, 0.168), (-0.082, 0.150),
        (-0.014, 0.138), (-0.014, 0.056), (+0.026, 0.056), (+0.034, 0.144),
        (+0.084, 0.152), (+0.098, 0.168), (+0.114, 0.168), (+0.114, 0.158), (+0.095, 0.148), (+0.095, 0.056),
    ]
    b_rim = bmesh.new()
    bm_add_revolved_profile(b_rim, prof_rim_yz, segs=96, close_loop=True, bevel_r=0.0007)
    bm_merge(b_wunit, b_rim, Matrix.Rotation(0.5 * math.pi, 4, 'Z'), default_mat_id=MAT_TRAILER_YELLOW)

    f0_whub = len(b_wunit.faces)
    bm_add_chamfered_cyl_x(b_wunit, 0.0, 0.0, -0.100, +0.118, radius=0.059, segs=36)
    bm_add_chamfered_cyl_x(b_wunit, 0.0, 0.0, +0.116, +0.128, radius=0.032, segs=24)
    bm_fill_unset_from(b_wunit, f0_whub, MAT_STAND_STEEL)
    bm_add_grease_zerk(b_wunit, Vector((+0.128, 0.0, 0.0)), Vector((1, 0, 0)), scale=1.2)
    for ib_hc in range(6):
        ahc = 2.0 * math.pi * ib_hc / 6.0
        bm_add_aerospace_bolt(b_wunit, Vector((+0.118, 0.045 * math.cos(ahc), 0.045 * math.sin(ahc))), Vector((1, 0, 0)), head_r=0.0048, height=0.0045)
    for ib_w in range(8):
        aw = 2.0 * math.pi * ib_w / 8.0
        bm_add_aerospace_bolt(b_wunit, Vector((+0.027, 0.082 * math.cos(aw), 0.082 * math.sin(aw))), Vector((1, 0, 0)), head_r=0.0076, height=0.0072)
    for ib_r in range(10):

        ar_b = 2.0 * math.pi * ib_r / 10.0
        bm_add_aerospace_bolt(b_wunit, Vector((+0.027, 0.128 * math.cos(ar_b), 0.128 * math.sin(ar_b))), Vector((1, 0, 0)), head_r=0.0060, height=0.0058)

    f0_sch = len(b_wunit.faces)
    bm_add_segmented_tube(
        b_wunit, Vector((+0.025, 0.0, +0.138)), Vector((+0.068, 0.0, +0.146)),
        [(0.0, 0.0068), (0.22, 0.0068), (0.28, 0.0044), (0.76, 0.0044), (0.82, 0.0054), (1.0, 0.0048)],
        segs=16, cap_ends=True,
    )
    bm_fill_unset_from(b_wunit, f0_sch, MAT_BRASS_FITTING)

    prof_drum_yz = [
        (-0.148, 0.044), (-0.148, 0.146), (-0.138, 0.146), (-0.138, 0.136),
        (-0.128, 0.136), (-0.128, 0.142), (-0.118, 0.142), (-0.118, 0.136),
        (-0.108, 0.136), (-0.108, 0.142), (-0.096, 0.142), (-0.096, 0.044),
    ]
    b_drum = bmesh.new()
    bm_add_revolved_profile(b_drum, prof_drum_yz, segs=64, close_loop=True, bevel_r=0.0006)
    bm_merge(b_wunit, b_drum, Matrix.Rotation(-0.5 * math.pi, 4, 'Z'), default_mat_id=MAT_STAND_STEEL)
    f0_drm_hw = len(b_wunit.faces)
    _bm_add_steel_plate(b_wunit, -0.168, -0.146, -0.038, +0.038, +0.068, +0.118, nu=2, nv=2, chamfer=0.0014)

    bm_add_bevel_locked_hex(b_wunit, Vector((-0.168, -0.018, +0.104)), Vector((-0.180, -0.018, +0.104)),
                            [(0.0, 0.0042), (0.30, 0.0052), (0.80, 0.0052), (1.0, 0.0038)], corner_bevel=0.14, cap_ends=True)
    _bm_add_steel_plate(b_wunit, +0.102, +0.116, -0.024, +0.024, -0.166, -0.154, nu=2, nv=1, chamfer=0.0008)
    bm_fill_unset_from(b_wunit, f0_drm_hw, MAT_MACHINED_ALU)


    prof_tire_yz = [
        (-0.098, 0.160), (-0.118, 0.194), (-0.123, 0.226), (-0.126, 0.230), (-0.126, 0.236), (-0.123, 0.240),
        (-0.112, 0.268), (-0.092, 0.283), (-0.070, 0.2880), (-0.052, 0.2880),
        (-0.051, 0.2770), (-0.041, 0.2770), (-0.040, 0.2890), (-0.018, 0.2890),
        (-0.017, 0.2770), (-0.007, 0.2770), (-0.006, 0.2895), (+0.006, 0.2895),
        (+0.007, 0.2770), (+0.017, 0.2770), (+0.018, 0.2890), (+0.040, 0.2890),
        (+0.041, 0.2770), (+0.051, 0.2770), (+0.052, 0.2880), (+0.070, 0.2880),
        (+0.092, 0.283), (+0.112, 0.268), (+0.123, 0.240), (+0.126, 0.236), (+0.126, 0.230), (+0.123, 0.226),
        (+0.118, 0.194), (+0.098, 0.160), (+0.076, 0.160), (+0.084, 0.222), (0.000, 0.246), (-0.084, 0.222), (-0.076, 0.160),
    ]
    b_tire = bmesh.new()
    bm_add_revolved_profile(b_tire, prof_tire_yz, segs=160, close_loop=True, bevel_r=0.00055)
    bm_merge(b_wunit, b_tire, Matrix.Rotation(0.5 * math.pi, 4, 'Z'), default_mat_id=MAT_RUBBER_DARK)


    n_tread_lugs = 44
    for k_lg in range(n_tread_lugs):
        ang_lg = 2.0 * math.pi * k_lg / float(n_tread_lugs)
        b_lug = bmesh.new()
        x_in_lg = 0.054 if (k_lg % 2 == 0) else 0.062
        for sx_sh in (-1.0, +1.0):
            xa_l = min(sx_sh * x_in_lg, sx_sh * 0.104)
            xb_l = max(sx_sh * x_in_lg, sx_sh * 0.104)

            _bm_add_box_x_yz(b_lug, xa_l, xb_l, -0.0092, +0.0092, 0.2795, 0.2891, nx=2, ny=1, nz=1, chamfer=0.0012)
            _bm_add_box_x_yz(b_lug, min(sx_sh * 0.088, sx_sh * 0.113), max(sx_sh * 0.088, sx_sh * 0.113),
                             -0.0080, +0.0080, 0.2620, 0.2825, nx=1, ny=1, nz=1, chamfer=0.0011)
        bm_merge(b_wunit, b_lug, Matrix.Rotation(ang_lg, 4, 'X'), default_mat_id=MAT_RUBBER_DARK)


    b_sw_txt = bmesh.new()
    def _sw_x_at_r(rv: float) -> float:

        fr = max(0.0, min(1.0, (rv - 0.194) / 0.032))
        return 0.1180 + 0.0050 * fr

    for sw_str, is_top_arc in (("7.50-10 NHS", True), ("12 PLY GSE", False)):
        n_sc = len(sw_str)
        step_deg = 7.4
        tot_deg = step_deg * n_sc
        for ic_s, ch_s in enumerate(sw_str):
            strokes = AERO_GLYPH_STROKES.get(ch_s)
            if not strokes:
                continue
            if is_top_arc:

                deg_c0 = +0.5 * tot_deg - ic_s * step_deg - step_deg * 0.12
                deg_c1 = +0.5 * tot_deg - (ic_s + 1) * step_deg + step_deg * 0.12
                r_top_s, r_bot_s = 0.218, 0.200
            else:

                deg_c0 = -0.5 * tot_deg + ic_s * step_deg + step_deg * 0.12
                deg_c1 = -0.5 * tot_deg + (ic_s + 1) * step_deg - step_deg * 0.12
                r_top_s, r_bot_s = 0.200, 0.218

            def _pt_sw(u_ang: float, v_rad: float, dx_out: float, _top=is_top_arc) -> Vector:
                ar = math.radians(u_ang)
                if _top:
                    yv = -v_rad * math.sin(ar)
                    zv = +v_rad * math.cos(ar)
                else:
                    yv = +v_rad * math.sin(ar)
                    zv = -v_rad * math.cos(ar)
                return Vector((_sw_x_at_r(v_rad) + dx_out, yv, zv))

            for (su0, su1, sv0, sv1) in strokes:
                au = deg_c0 + su0 * (deg_c1 - deg_c0)
                bu = deg_c0 + su1 * (deg_c1 - deg_c0)
                rv_a = r_top_s + sv0 * (r_bot_s - r_top_s)
                rv_b = r_top_s + sv1 * (r_bot_s - r_top_s)
                _bm_emit_bevel_stroke_quad_prism(b_sw_txt, _pt_sw, au, bu, rv_a, rv_b, 0.0008, 0.0012)
    bmesh.ops.recalc_face_normals(b_sw_txt, faces=b_sw_txt.faces)
    bm_merge(b_wunit, b_sw_txt)


    rubber_verts = set()
    for f_w in b_wunit.faces:
        if f_w.material_index in (MAT_RUBBER_DARK, MAT_3D_TEXT):
            for v_w in f_w.verts:
                rubber_verts.add(v_w)
    for v_w in rubber_verts:
        vx, vy, vz = v_w.co.x, v_w.co.y, v_w.co.z
        ay = abs(vy)
        if vz < -0.205 and ay < 0.104:
            ty = 1.0 - (ay / 0.104)
            wy = ty * ty * (3.0 - 2.0 * ty)

            if vz < -0.264 and abs(vx) < 0.116:
                tz = min(1.0, max(0.0, (-0.264 - vz) / 0.0255))
                wz = tz * tz * (3.0 - 2.0 * tz)
                d_fl = max(0.0, vz + 0.2895)
                v_w.co.z = -0.2895 + d_fl * (1.0 - 0.82 * wy * wz)

            if -0.284 < vz < -0.205 and abs(vx) > 0.090:
                s_sw = (vz - (-0.284)) / 0.079
                bulge_x = 0.0036 * math.sin(math.pi * s_sw) * wy
                v_w.co.x = vx + (bulge_x if vx >= 0.0 else -bulge_x)
    _sd_stand.add_trailer_06_wheel_hub_and_rim_details(b_wunit, _bm_add_steel_plate)
    b_wunit.normal_update()

    for y_w in (-3.880, -1.495):
        for sx_w in (-1.0, +1.0):
            tf_w = Matrix.Translation((sx_w * 0.652, y_w, Z_AXLE))
            if sx_w < 0.0:
                tf_w = tf_w @ Matrix.Rotation(math.pi, 4, 'Z')
            bm_merge(bm_wh, b_wunit.copy(), tf_w)
    b_wunit.free()

    create_mesh_obj("Stand_Trailer_06_Pneumatic_GroundSupport_Wheels", bm_wh, coll_stand, clay)


    Z_FL = -1.6095
    bm_gse = bmesh.new()


    y_rear_w = -1.495
    x_rear_w = +0.652
    r_tire = 0.2895
    h_chock = 0.118
    for sy_c in (-1.0, +1.0):

        b_ck = bmesh.new()
        t_fy = subdivide_1d_with_support_loops(18, sup1=0.02, sup2=0.06)
        t_fx = subdivide_1d_with_support_loops(6, sup1=0.04, sup2=0.14)
        gb_c, gt_c = [], []
        for fy in t_fy:
            if fy <= 0.56:
                s = fy / 0.56

                dy = 0.114 + (0.2375 - 0.114) * s
                dy_t_eff = max(0.0, dy - 0.0045)
                dz_tire = r_tire - math.sqrt(max(0.0, r_tire * r_tire - dy_t_eff * dy_t_eff))
                dz = max(0.008, min(h_chock, dz_tire - 0.0110 * ((1.0 - s) ** 2)))
                rib = 0.0020 * math.sin(2.0 * math.pi * 5.0 * s) if (0.12 < s < 0.90) else 0.0
                dz_surf = dz + max(0.0, rib)
            else:
                s = (fy - 0.56) / 0.44
                dy = 0.2375 + 0.118 * s
                dz_surf = h_chock * (1.0 - 0.91 * s)
            yv = y_rear_w + sy_c * dy
            rb, rt = [], []
            for fx in t_fx:
                xv = (x_rear_w - 0.114) + 0.228 * fx
                rb.append(b_ck.verts.new((xv, yv, Z_FL)))
                rt.append(b_ck.verts.new((xv, yv, Z_FL + dz_surf)))
            gb_c.append(rb)
            gt_c.append(rt)
        bm_build_beveled_quad_shell(b_ck, gb_c, gt_c, bulge_scale=0.06, max_bevel=0.0026, inset_bevel=True)
        bm_merge(bm_gse, b_ck, default_mat_id=MAT_RUBBER_DARK)


        y_eye = y_rear_w + sy_c * 0.256
        z_eye = Z_FL + 0.046
        f0_ce = len(bm_gse.faces)
        bm_add_chamfered_cyl_x(bm_gse, y_eye, z_eye, x_rear_w - 0.116, x_rear_w + 0.116, radius=0.0130, segs=20, hollow_r=0.0076)

        ring_pts = []
        for ir in range(16):
            th_r = 2.0 * math.pi * ir / 16.0
            ring_pts.append(Vector((
                x_rear_w + 0.116 + 0.011 * (1.0 - math.cos(th_r)),
                y_eye + sy_c * 0.008 * math.sin(th_r),
                z_eye + 0.010 * math.sin(th_r),
            )))
        ring_pts.append(ring_pts[0])
        bm_add_spline_tube(bm_gse, ring_pts, radius=0.0026, n_samples=20, segs=10, cap_ends=False, closed=True)
        bm_fill_unset_from(bm_gse, f0_ce, MAT_FASTENER)


        b_hz = bmesh.new()
        g_hb, g_ht = [], []

        for s_h in (0.20, 0.24, 0.48, 0.71, 0.75):
            dy_h = 0.2375 + 0.118 * s_h
            dz_h = h_chock * (1.0 - 0.91 * s_h)
            yv_h = y_rear_w + sy_c * dy_h

            ny_h, nz_h = sy_c * 0.67, 0.74
            rhb, rht = [], []
            for xv_h in (x_rear_w - 0.088, x_rear_w - 0.082, x_rear_w, x_rear_w + 0.082, x_rear_w + 0.088):
                rhb.append(b_hz.verts.new((xv_h, yv_h + ny_h * 0.0002, Z_FL + dz_h + nz_h * 0.0002)))
                rht.append(b_hz.verts.new((xv_h, yv_h + ny_h * 0.0026, Z_FL + dz_h + nz_h * 0.0026)))
            g_hb.append(rhb)
            g_ht.append(rht)
        bm_build_beveled_quad_shell(b_hz, g_hb, g_ht, bulge_scale=0.06)
        bm_merge(bm_gse, b_hz, default_mat_id=MAT_PLATE_BASE)


    rope_pts = [
        Vector((x_rear_w + 0.128, y_rear_w - 0.256, Z_FL + 0.044)),
        Vector((x_rear_w + 0.156, y_rear_w - 0.248, Z_FL + 0.016)),
        Vector((x_rear_w + 0.195, y_rear_w - 0.222, Z_FL + 0.0054)),
        Vector((x_rear_w + 0.248, y_rear_w - 0.120, Z_FL + 0.0054)),
        Vector((x_rear_w + 0.272, y_rear_w + 0.005, Z_FL + 0.0054)),
        Vector((x_rear_w + 0.240, y_rear_w + 0.135, Z_FL + 0.0054)),
        Vector((x_rear_w + 0.188, y_rear_w + 0.225, Z_FL + 0.0054)),
        Vector((x_rear_w + 0.154, y_rear_w + 0.248, Z_FL + 0.016)),
        Vector((x_rear_w + 0.128, y_rear_w + 0.256, Z_FL + 0.044)),
    ]
    f0_rp = len(bm_gse.faces)
    bm_add_spline_tube(bm_gse, rope_pts, radius=0.0052, n_samples=56, segs=14, cap_ends=True)
    bm_fill_unset_from(bm_gse, f0_rp, MAT_RUBBER_DARK)

    for sy_fr in (-1.0, +1.0):
        f0_fr = len(bm_gse.faces)
        bm_add_segmented_tube(
            bm_gse,
            Vector((x_rear_w + 0.124, y_rear_w + sy_fr * 0.256, Z_FL + 0.045)),
            Vector((x_rear_w + 0.144, y_rear_w + sy_fr * 0.252, Z_FL + 0.028)),
            [(0.0, 0.0068), (0.15, 0.0074), (0.85, 0.0074), (1.0, 0.0064)], segs=14, cap_ends=True,
        )
        bm_fill_unset_from(bm_gse, f0_fr, MAT_BRASS_FITTING)


    y_gc = -2.015
    f0_cl = len(bm_gse.faces)

    _bm_add_steel_plate(bm_gse, +0.710, +0.788, y_gc - 0.011, y_gc + 0.011, -0.884, -0.873, nu=3, nv=2, chamfer=0.0010)
    _bm_add_steel_plate(bm_gse, +0.710, +0.788, y_gc - 0.011, y_gc + 0.011, -0.917, -0.906, nu=3, nv=2, chamfer=0.0010)

    _bm_add_steel_plate(bm_gse, +0.744, +0.766, y_gc - 0.013, y_gc + 0.013, -0.914, -0.876, nu=2, nv=2, chamfer=0.0012)
    bm_add_segmented_tube(
        bm_gse, Vector((+0.755, y_gc - 0.016, -0.895)), Vector((+0.755, y_gc + 0.016, -0.895)),
        [(0.0, 0.0045), (0.20, 0.0056), (0.80, 0.0056), (1.0, 0.0045)], segs=14, cap_ends=True,
    )
    bm_add_segmented_tube(
        bm_gse, Vector((+0.768, y_gc, -0.882)), Vector((+0.810, y_gc, -0.870)),
        [(0.0, 0.0058), (0.50, 0.0062), (1.0, 0.0048)], segs=12, cap_ends=True,
    )
    bm_add_segmented_tube(
        bm_gse, Vector((+0.768, y_gc, -0.908)), Vector((+0.810, y_gc, -0.920)),
        [(0.0, 0.0058), (0.50, 0.0062), (1.0, 0.0048)], segs=12, cap_ends=True,
    )
    bm_add_segmented_tube(
        bm_gse, Vector((+0.765, y_gc, -0.895)), Vector((+0.806, y_gc, -0.895)),
        [(0.0, 0.0048), (0.50, 0.0056), (1.0, 0.0038)], segs=14, cap_ends=True,
    )
    bm_fill_unset_from(bm_gse, f0_cl, MAT_BRASS_FITTING)


    f0_cg = len(bm_gse.faces)
    for z_g0, z_g1 in ((-0.878, -0.870), (-0.912, -0.920)):
        bm_add_segmented_tube(
            bm_gse, Vector((+0.782, y_gc, z_g0)), Vector((+0.812, y_gc, z_g1)),
            [(0.0, 0.0068), (0.85, 0.0072), (1.0, 0.0055)], segs=14, cap_ends=True,
        )
    bm_fill_unset_from(bm_gse, f0_cg, MAT_RUBBER_DARK)


    x_gr, y_gr = +1.220, -2.080
    gc_pts = [
        Vector((+0.804, y_gc,  -0.895)),
        Vector((+0.838, y_gc - 0.004, -0.975)),
        Vector((+0.868, y_gc - 0.012, -1.145)),
        Vector((+0.895, y_gc - 0.025, -1.355)),
        Vector((+0.922, y_gc - 0.042, -1.525)),
        Vector((+0.946, y_gc - 0.056, Z_FL + 0.026)),
        Vector((+0.985, y_gc - 0.082, Z_FL + 0.0038)),
        Vector((+1.078, y_gc - 0.152, Z_FL + 0.0036)),
        Vector((+1.215, y_gc - 0.215, Z_FL + 0.0036)),
        Vector((+1.335, y_gc - 0.185, Z_FL + 0.0036)),
        Vector((+1.345, y_gc - 0.105, Z_FL + 0.0036)),
        Vector((+1.280, y_gc - 0.072, Z_FL + 0.0052)),
        Vector((+1.236, y_gr + 0.002, Z_FL + 0.024)),
        Vector((x_gr,   y_gr,         Z_FL + 0.044)),
    ]
    v0_gc = len(bm_gse.verts)
    f0_gc = len(bm_gse.faces)
    bm_add_spline_tube(bm_gse, gc_pts, radius=0.0030, n_samples=84, segs=12, cap_ends=True)
    bm_gse.verts.ensure_lookup_table()
    for iv_gc in range(v0_gc, len(bm_gse.verts)):
        if bm_gse.verts[iv_gc].co.z < Z_FL + 0.0003:
            bm_gse.verts[iv_gc].co.z = Z_FL + 0.0003
    bm_fill_unset_from(bm_gse, f0_gc, MAT_TRAILER_YELLOW)


    f0_gr = len(bm_gse.faces)
    bm_add_segmented_tube(
        bm_gse, Vector((x_gr, y_gr, Z_FL)), Vector((x_gr, y_gr, Z_FL + 0.0046)),
        [(0.0, 0.040), (0.35, 0.039), (0.85, 0.037), (1.0, 0.034)], segs=32, cap_ends=True,
    )
    bm_add_segmented_tube(
        bm_gse, Vector((x_gr, y_gr, Z_FL + 0.0042)), Vector((x_gr, y_gr, Z_FL + 0.0085)),
        [(0.0, 0.0175), (0.75, 0.0165), (1.0, 0.0145)], segs=24, cap_ends=True,
    )

    bm_add_segmented_tube(
        bm_gse, Vector((x_gr, y_gr, Z_FL + 0.0082)), Vector((x_gr, y_gr, Z_FL + 0.034)),
        [(0.0, 0.0092), (0.18, 0.0108), (0.32, 0.0102), (0.48, 0.0110), (0.66, 0.0102), (0.84, 0.0108), (1.0, 0.0084)],
        segs=20, cap_ends=True,
    )
    bm_fill_unset_from(bm_gse, f0_gr, MAT_BRASS_FITTING)

    f0_gr_rb = len(bm_gse.faces)
    bm_add_segmented_tube(
        bm_gse, Vector((x_gr, y_gr, Z_FL + 0.033)), Vector((x_gr, y_gr, Z_FL + 0.048)),
        [(0.0, 0.0082), (0.40, 0.0064), (1.0, 0.0040)], segs=16, cap_ends=True,
    )
    bm_fill_unset_from(bm_gse, f0_gr_rb, MAT_RUBBER_DARK)

    f0_gr_sc = len(bm_gse.faces)
    for k_gs in range(4):
        ag = math.radians(45.0 + k_gs * 90.0)
        bm_add_aerospace_bolt(bm_gse, Vector((x_gr + 0.028 * math.cos(ag), y_gr + 0.028 * math.sin(ag), Z_FL + 0.0044)),
                              Vector((0, 0, 1)), head_r=0.0032, height=0.0014)
    bm_fill_unset_from(bm_gse, f0_gr_sc, MAT_FASTENER)


    px0, px1 = -0.354, +0.354
    py0, py1 = -2.914, -2.386
    p_h = 0.0320
    p_flare = 0.0075
    p_th = 0.0020
    f0_dp = len(bm_gse.faces)

    _bm_add_steel_plate(bm_gse, px0, px1, py0, py1, Z_FL, Z_FL + p_th, nu=8, nv=6, chamfer=0.0004)


    for sx_w, xb_w in ((-1.0, px0), (+1.0, px1)):
        b_sw = bmesh.new()
        g_in, g_out = [], []
        for fz_w in (0.0, 0.08, 0.50, 0.92, 1.0):
            zw = Z_FL + 0.0012 + (p_h - 0.0012) * fz_w
            xw_in = xb_w + sx_w * (p_flare * fz_w - p_th * 0.5)
            xw_out = xb_w + sx_w * (p_flare * fz_w + p_th * 0.5)
            rin, rout = [], []
            for fy_w in (0.0, 0.04, 0.25, 0.50, 0.75, 0.96, 1.0):
                yw = (py0 - p_flare * fz_w) + ((py1 - py0) + 2.0 * p_flare * fz_w) * fy_w
                rin.append(b_sw.verts.new((xw_in, yw, zw)))
                rout.append(b_sw.verts.new((xw_out, yw, zw)))
            g_in.append(rin)
            g_out.append(rout)
        bm_build_beveled_quad_shell(b_sw, g_in, g_out, bulge_scale=0.04)
        bm_merge(bm_gse, b_sw)

    for sy_w, yb_w in ((-1.0, py0), (+1.0, py1)):
        b_sw = bmesh.new()
        g_in, g_out = [], []
        for fz_w in (0.0, 0.08, 0.50, 0.92, 1.0):
            zw = Z_FL + 0.0012 + (p_h - 0.0012) * fz_w
            yw_in = yb_w + sy_w * (p_flare * fz_w - p_th * 0.5)
            yw_out = yb_w + sy_w * (p_flare * fz_w + p_th * 0.5)
            rin, rout = [], []
            for fx_w in (0.0, 0.04, 0.25, 0.50, 0.75, 0.96, 1.0):
                xw = (px0 - p_flare * fz_w) + ((px1 - px0) + 2.0 * p_flare * fz_w) * fx_w
                rin.append(b_sw.verts.new((xw, yw_in, zw)))
                rout.append(b_sw.verts.new((xw, yw_out, zw)))
            g_in.append(rin)
            g_out.append(rout)
        bm_build_beveled_quad_shell(b_sw, g_in, g_out, bulge_scale=0.04)
        bm_merge(bm_gse, b_sw)


    rx0, rx1 = px0 - p_flare, px1 + p_flare
    ry0, ry1 = py0 - p_flare, py1 + p_flare
    z_rim = Z_FL + p_h
    for p_r0, p_r1 in (
        (Vector((rx0, ry0, z_rim)), Vector((rx1, ry0, z_rim))),
        (Vector((rx1, ry0, z_rim)), Vector((rx1, ry1, z_rim))),
        (Vector((rx1, ry1, z_rim)), Vector((rx0, ry1, z_rim))),
        (Vector((rx0, ry1, z_rim)), Vector((rx0, ry0, z_rim))),
    ):
        bm_add_segmented_tube(bm_gse, p_r0, p_r1, [(0.0, 0.0028), (0.03, 0.0028), (0.97, 0.0028), (1.0, 0.0028)], segs=12, cap_ends=True)


    for sy_h, y_edge in ((-1.0, ry0), (+1.0, ry1)):
        y_wall_h = (py0 if sy_h < 0.0 else py1) + sy_h * (p_flare * 0.65)
        for sx_ear in (-0.078, +0.078):
            _bm_add_steel_plate(bm_gse, sx_ear - 0.0055, sx_ear + 0.0055,
                                min(y_wall_h - sy_h * 0.001, y_wall_h + sy_h * 0.009),
                                max(y_wall_h - sy_h * 0.001, y_wall_h + sy_h * 0.009),
                                z_rim - 0.0155, z_rim - 0.0065, nu=1, nv=1, chamfer=0.0005)
        bm_add_spline_tube(
            bm_gse,
            [
                Vector((-0.078, y_wall_h + sy_h * 0.003, z_rim - 0.011)),
                Vector((-0.082, y_edge + sy_h * 0.022, z_rim - 0.018)),
                Vector((-0.068, y_edge + sy_h * 0.032, z_rim - 0.022)),
                Vector((+0.068, y_edge + sy_h * 0.032, z_rim - 0.022)),
                Vector((+0.082, y_edge + sy_h * 0.022, z_rim - 0.018)),
                Vector((+0.078, y_wall_h + sy_h * 0.003, z_rim - 0.011)),
            ],
            radius=0.0022, n_samples=32, segs=10, cap_ends=True,
        )
    bm_fill_unset_from(bm_gse, f0_dp, MAT_RIGID_TUBE)


    b_op = bmesh.new()
    go_b, go_t = [], []
    tx_op = subdivide_1d_with_support_loops(8, sup1=0.03, sup2=0.10)
    ty_op = subdivide_1d_with_support_loops(6, sup1=0.03, sup2=0.10)
    for fx_o in tx_op:
        xo = (px0 + 0.008) + (px1 - px0 - 0.016) * fx_o
        rb_o, rt_o = [], []
        for fy_o in ty_op:
            yo = (py0 + 0.008) + (py1 - py0 - 0.016) * fy_o

            ed_o = min(fx_o, 1.0 - fx_o, fy_o, 1.0 - fy_o)
            zh_o = 0.0020 if ed_o > 0.04 else (0.0008 + 0.0012 * (ed_o / 0.04))
            rb_o.append(b_op.verts.new((xo, yo, Z_FL + p_th + 0.0001)))
            rt_o.append(b_op.verts.new((xo, yo, Z_FL + p_th + zh_o)))
        go_b.append(rb_o)
        go_t.append(rt_o)
    bm_build_beveled_quad_shell(b_op, go_b, go_t, bulge_scale=0.02)
    bm_merge(bm_gse, b_op, default_mat_id=MAT_PLATE_BASE)

    _hd_detail.add_trailer_07_drippan_mesh_and_drain(bm_gse, Z_FL, _bm_add_steel_plate)
    create_mesh_obj("Stand_Trailer_07_Deployed_GSE_Props", bm_gse, coll_stand, clay)


    bm_rbf = bmesh.new()
    uv_rbf = bm_rbf.loops.layers.uv.new("UVMap")


    rbf_configs = [
        (+1.0, -2.261, +24.0, 0.35),
        (-1.0, -3.784, -26.0, 1.85),
    ]
    for sx_r, y_pin, yaw_deg, ph_s in rbf_configs:

        x_anc = sx_r * 0.774
        z_anc = +0.008

        f0_kr = len(bm_rbf.faces)
        kr_pts = []
        for ik in range(20):
            th_k = 2.0 * math.pi * ik / 20.0
            kr_pts.append(Vector((
                x_anc + sx_r * 0.002 + 0.0025 * math.cos(th_k),
                y_pin + 0.0115 * math.sin(th_k),
                z_anc - 0.0135 + 0.0135 * math.cos(th_k),
            )))
        kr_pts.append(kr_pts[0])
        bm_add_spline_tube(bm_rbf, kr_pts, radius=0.0014, n_samples=24, segs=10, cap_ends=False, closed=True)
        bm_fill_unset_from(bm_rbf, f0_kr, MAT_FASTENER)


        yaw_rad = math.radians(yaw_deg)

        if sx_r > 0.0:
            w_vec0 = Vector((-math.sin(yaw_rad), math.cos(yaw_rad), 0.0))
            n_vec0 = Vector((math.cos(yaw_rad),  math.sin(yaw_rad), 0.0))
        else:
            w_vec0 = Vector((-math.sin(yaw_rad), -math.cos(yaw_rad), 0.0))
            n_vec0 = Vector((-math.cos(yaw_rad),  math.sin(yaw_rad), 0.0))

        p_grom = Vector((x_anc + sx_r * 0.003, y_pin, -0.023))
        f0_gm = len(bm_rbf.faces)
        z_vec0 = Vector((0.0, 0.0, 1.0))
        for sn_g in (-0.0007, +0.0007):
            gm_pts = []
            for ig in range(18):
                th_g = 2.0 * math.pi * ig / 18.0
                gm_pts.append(p_grom + n_vec0 * sn_g + w_vec0 * (0.0044 * math.cos(th_g)) + z_vec0 * (0.0044 * math.sin(th_g)))
            gm_pts.append(gm_pts[0])
            bm_add_spline_tube(bm_rbf, gm_pts, radius=0.00125, n_samples=20, segs=10, cap_ends=False, closed=True)
        bm_fill_unset_from(bm_rbf, f0_gm, MAT_BRASS_FITTING)


        L_rib = 0.228
        W_rib = 0.049
        th_half = 0.00032
        nu_r, nv_r = 28, 6
        grid_f, grid_b = [], []
        for iu in range(nu_r + 1):
            u = iu / float(nu_r)

            tw_ang = math.radians(15.0 * (u ** 1.35) * math.sin(1.4 * u + ph_s))
            ca_t, sa_t = math.cos(tw_ang), math.sin(tw_ang)
            w_cur = (w_vec0 * ca_t + n_vec0 * sa_t).normalized()
            n_cur = (n_vec0 * ca_t - w_vec0 * sa_t).normalized()

            sway_n = 0.0095 * (u ** 1.30) * math.sin(3.8 * u + ph_s)
            sway_w = 0.0048 * (u ** 1.45) * math.cos(2.6 * u + ph_s)
            p_ctr = Vector((x_anc + sx_r * 0.003, y_pin, -0.012 - L_rib * u)) + n_cur * sway_n + w_cur * sway_w

            row_f, row_b = [], []
            for iv in range(nv_r + 1):
                v = iv / float(nv_r)

                cup = 0.0009 * ((2.0 * v - 1.0) ** 2 - 0.35) * (0.25 + 0.75 * u)
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

    _hd_detail.add_cradle_05_inlet_probe_rbf(bm_rbf, uv_rbf)
    create_mesh_obj("Stand_Cradle_05_RBF_Streamers", bm_rbf, coll_stand, clay)


def refresh_stand_only_in_current_scene(save_blend_path: str = None):

    coll_name = "06_Maintenance_Stand_And_Trailer"
    coll_stand = bpy.data.collections.get(coll_name)
    if coll_stand is None:
        coll_stand = bpy.data.collections.new(coll_name)
        bpy.context.scene.collection.children.link(coll_stand)
    else:
        old_objs = list(coll_stand.all_objects)
        old_meshes = [o.data for o in old_objs if o.type == "MESH" and o.data is not None]
        for o in old_objs:
            bpy.data.objects.remove(o, do_unlink=True)
        for me in old_meshes:
            if me.users == 0:
                bpy.data.meshes.remove(me)

    clay_mat = bpy.data.materials.get("Mat_Clay_WhiteModel")
    mats = {"clay": clay_mat} if clay_mat is not None else build_clay_materials()
    build_maintenance_stand_and_trailer(mats, coll_stand)
    bpy.context.view_layer.update()

    stand_objs = [o for o in coll_stand.all_objects if o.type == "MESH"]
    tot_q = sum(sum(1 for p in o.data.polygons if len(p.vertices) == 4) for o in stand_objs)
    tot_f = sum(len(o.data.polygons) for o in stand_objs)
    assert tot_q == tot_f, f"Stand non-quads detected: {tot_f - tot_q}"
    print(f"[Stand Refresh] Objects={len(stand_objs)} | Quads={tot_q:,}")

    if save_blend_path:
        bpy.ops.wm.save_as_mainfile(filepath=save_blend_path)
        blend1 = save_blend_path + "1"
        if __import__("os").path.exists(blend1):
            __import__("os").remove(blend1)
        print(f"[Saved] {save_blend_path}")
