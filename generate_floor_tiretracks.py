import math
import os
import time
import numpy as np
from PIL import Image, ImageFilter


Image.MAX_IMAGE_PIXELS = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_THIS_DIR, "assets_masks_4k", "mask_4k_floor_tiretracks.png")

RES = 16384
STRIP_H = 1024
HALO = 24
X_MIN, X_MAX = -5.0, 5.0
Y_MIN, Y_MAX = -7.25, 2.75


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class _NoiseField2D:


    def __init__(self, grid_x: int, grid_y: int, seed: int):
        rng = np.random.default_rng(seed)
        base = rng.random((grid_y, grid_x), dtype=np.float32)
        self.img = Image.fromarray((base * 255.0).astype(np.uint8), mode="L")
        self.gx = grid_x
        self.gy = grid_y

    def sample_strip(self, y0: int, y1: int, total_res: int) -> np.ndarray:


        u0 = (y0 / float(total_res)) * self.gy
        u1 = (y1 / float(total_res)) * self.gy
        strip = self.img.resize(
            (total_res, y1 - y0),
            resample=Image.Resampling.BICUBIC,
            box=(0.0, u0, float(self.gx), u1),
        )
        return np.asarray(strip, dtype=np.float32) / 255.0

    def sample_full(self, res: int) -> np.ndarray:
        up = self.img.resize((res, res), resample=Image.Resampling.BICUBIC)
        return np.asarray(up, dtype=np.float32) / 255.0


class _FbmSpec:


    def __init__(self, octaves: list, mode: str = "z", scale_01: float = 0.22):

        self.octaves = [(w, _NoiseField2D(gx, gy, seed)) for (w, gx, gy, seed) in octaves]
        self.mode = mode
        self.scale_01 = scale_01

        ref = np.zeros((1024, 1024), dtype=np.float32)
        for w, nf in self.octaves:
            ref += w * nf.sample_full(1024)
        self.mean = float(ref.mean())
        self.std = float(ref.std()) + 1e-6

    def sample_strip(self, y0: int, y1: int, total_res: int) -> np.ndarray:
        acc = np.zeros((y1 - y0, total_res), dtype=np.float32)
        for w, nf in self.octaves:
            acc += w * nf.sample_strip(y0, y1, total_res)
        z = (acc - self.mean) / self.std
        if self.mode == "z":
            return z
        return np.clip(0.5 + self.scale_01 * z, 0.0, 1.0)


def _blur_2d(arr: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    out = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(out, dtype=np.float32) / 255.0


def _m3000_worn_tread_profile(
    u_signed: np.ndarray, s_along: np.ndarray,
    wobble_u: np.ndarray, wobble_s: np.ndarray,
    lateral_bias: np.ndarray, lug_erode: np.ndarray,
    lug_phase: float = 0.0,
):

    u_w = u_signed + wobble_u * 0.0050
    s_w = s_along + wobble_s * 0.0068
    u_abs = np.abs(u_w)


    rib_c = 1.0 - _smoothstep(0.0040, 0.0090, u_abs)
    rib_m = _smoothstep(0.0145, 0.0195, u_abs) * (1.0 - _smoothstep(0.0375, 0.0428, u_abs))
    rib_s = _smoothstep(0.0485, 0.0538, u_abs) * (1.0 - _smoothstep(0.0670, 0.0740, u_abs))


    pitch = 0.04134
    cyc = (s_w + lug_phase) / pitch
    frac = cyc - np.floor(cyc)
    idx_odd = (np.floor(cyc).astype(np.int32) & 1).astype(np.float32)
    lug_s = _smoothstep(0.05, 0.16, frac) * (1.0 - _smoothstep(0.50, 0.62, frac))
    u_in_lug = 0.054 + 0.008 * idx_odd
    lug_u = _smoothstep(u_in_lug - 0.0030, u_in_lug + 0.0032, u_abs) * (1.0 - _smoothstep(0.093, 0.104, u_abs))
    lug_survive = _smoothstep(0.38, 0.58, lug_erode)
    lugs = lug_s * lug_u * lug_survive * 0.92

    u_peak = (lateral_bias - 0.5) * 0.096
    asym_press = np.exp(-((u_w - u_peak) / 0.088) ** 2)
    crown_fall = 1.0 - 0.22 * _smoothstep(0.065, 0.104, u_abs)
    smear_bed = (1.0 - _smoothstep(0.052, 0.104, u_abs)) * 0.22

    tread_raw = np.maximum(np.maximum(rib_c * 0.98, rib_m * 0.95), np.maximum(rib_s * 0.82, lugs))
    tread = np.clip((tread_raw * 0.95 + smear_bed) * asym_press * crown_fall, 0.0, 1.0)
    rib_core = np.clip((np.maximum(rib_c, rib_m) * 0.88 + smear_bed * 1.12) * asym_press * crown_fall, 0.0, 1.0)
    corridor = (1.0 - _smoothstep(0.068, 0.130, u_abs)) * (0.55 + 0.45 * asym_press)
    return tread, rib_core, corridor


def _forklift_worn_tread(u_signed: np.ndarray, s_along: np.ndarray, wobble_u: np.ndarray, wobble_s: np.ndarray, erode: np.ndarray):

    u_w = u_signed + wobble_u * 0.0055
    s_w = s_along + wobble_s * 0.0080
    u_abs = np.abs(u_w)

    pitch = 0.038
    cyc = (s_w + 0.28 * u_abs) / pitch
    frac = cyc - np.floor(cyc)
    blk_s = _smoothstep(0.05, 0.19, frac) * (1.0 - _smoothstep(0.52, 0.68, frac))
    blk_u = _smoothstep(0.005, 0.011, u_abs) * (1.0 - _smoothstep(0.066, 0.080, u_abs))

    patch_keep = _smoothstep(0.15, 0.90, erode)
    lug_ghost = blk_s * blk_u * _smoothstep(0.30, 1.05, erode) * 0.92
    scuff_bed = (1.0 - _smoothstep(0.046, 0.086, u_abs)) * patch_keep * 0.46
    tread = np.clip(np.maximum(lug_ghost, scuff_bed), 0.0, 1.0)
    corr = (1.0 - _smoothstep(0.056, 0.110, u_abs)) * _smoothstep(-0.15, 0.65, erode)
    return tread, corr


def _eval_primary_curve(
    X, Y, x_of_y_fn, dxdy_fn, wheel_half_track,
    fade_in_y, fade_out_y, wobble_u, wobble_s,
    drop_l, drop_r, bias_l, bias_r, erode_l, erode_r,
):
    xc = x_of_y_fn(Y)
    slope = dxdy_fn(Y)
    inv_norm = 1.0 / np.sqrt(1.0 + slope * slope)

    d_perp = (X - xc) * inv_norm
    s_arc = Y * np.sqrt(1.0 + 0.5 * slope * slope)
    env_y = _smoothstep(fade_in_y[0], fade_in_y[1], Y) * (1.0 - _smoothstep(fade_out_y[0], fade_out_y[1], Y))

    tl, rl, cl = _m3000_worn_tread_profile(d_perp + wheel_half_track, s_arc, wobble_u, wobble_s, bias_l, erode_l, lug_phase=0.004)
    tr, rr, cr = _m3000_worn_tread_profile(d_perp - wheel_half_track, s_arc, -wobble_u, wobble_s, bias_r, erode_r, lug_phase=0.023)

    corr_keep_l = np.clip(drop_l * 0.75 + 0.25 * _smoothstep(0.18, 0.58, erode_l), 0.0, 1.0)
    corr_keep_r = np.clip(drop_r * 0.75 + 0.25 * _smoothstep(0.18, 0.58, erode_r), 0.0, 1.0)

    tread = np.maximum(tl * drop_l, tr * drop_r) * env_y
    rib_c = np.maximum(rl * drop_l, rr * drop_r) * env_y
    corr  = np.maximum(cl * corr_keep_l, cr * corr_keep_r) * env_y
    return tread, rib_c, corr


def _eval_worn_arc(X, Y, cx, cy, r_mid, half_track, ang0_deg, ang1_deg, wobble_u, wobble_s, erode_in, erode_out, fade_deg=16.0):
    dx = X - cx
    dy = Y - cy
    r = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))
    ang_rel = (ang - (ang0_deg - 40.0)) % 360.0 + (ang0_deg - 40.0)
    a_lo, a_hi = min(ang0_deg, ang1_deg), max(ang0_deg, ang1_deg)
    env_a = _smoothstep(a_lo, a_lo + fade_deg, ang_rel) * (1.0 - _smoothstep(a_hi - fade_deg, a_hi, ang_rel))

    s_in  = np.radians(ang_rel) * (r_mid - half_track)
    s_out = np.radians(ang_rel) * (r_mid + half_track)
    ti, ci = _forklift_worn_tread(r - (r_mid - half_track), s_in, wobble_u, wobble_s, erode_in)
    to, co = _forklift_worn_tread(r - (r_mid + half_track), s_out, -wobble_u, wobble_s, erode_out)
    return np.maximum(ti, to) * env_a, np.maximum(ci, co) * env_a


def generate_floor_tiretracks():
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print(f"[1/3] Initializing 16K ({RES}x{RES}, 0.61 mm/px) deterministic noise & concrete-pore fields...", flush=True)

    def _mk_patch(seed):
        return _FbmSpec([(0.38, 18, 18, seed + 1), (0.32, 42, 42, seed + 2), (0.18, 110, 110, seed + 3), (0.12, 290, 290, seed + 4)], mode="z")

    def _mk_iso(seed):
        return _FbmSpec([(0.36, 10, 10, seed + 1), (0.26, 26, 26, seed + 2), (0.18, 72, 72, seed + 3), (0.12, 210, 210, seed + 4), (0.08, 640, 640, seed + 5)], mode="01", scale_01=0.22)

    patch_l  = _mk_patch(811)
    patch_r  = _mk_patch(823)
    patch_h1 = _mk_patch(839)
    patch_h2 = _mk_patch(857)
    fbm_a    = _mk_iso(863)
    fbm_b    = _mk_iso(877)

    fine_grit_spec = _FbmSpec([
        (0.24, 56, 56, 892),
        (0.24, 180, 180, 893),
        (0.22, 560, 560, 894),
        (0.18, 1400, 1400, 895),
        (0.12, 2800, 2800, 896),
    ], mode="01", scale_01=0.25)

    mop_wipe_spec = _FbmSpec([(0.55, 10, 32, 907), (0.45, 22, 68, 911)], mode="z")
    dust_l_spec   = _FbmSpec([(0.60, 20, 12, 919), (0.40, 46, 24, 929)], mode="z")
    dust_r_spec   = _FbmSpec([(0.60, 20, 12, 937), (0.40, 46, 24, 941)], mode="z")


    ref_pl = patch_l.sample_strip(0, 1024, 1024)
    ref_pr = patch_r.sample_strip(0, 1024, 1024)
    ref_h1 = patch_h1.sample_strip(0, 1024, 1024)
    ref_h2 = patch_h2.sample_strip(0, 1024, 1024)
    ref_mw = mop_wipe_spec.sample_strip(0, 1024, 1024)
    ref_dl = dust_l_spec.sample_strip(0, 1024, 1024)
    ref_dr = dust_r_spec.sample_strip(0, 1024, 1024)
    ref_fg = fine_grit_spec.sample_strip(0, 1024, 1024)
    fg_mean, fg_std = float(ref_fg.mean()), float(ref_fg.std()) + 1e-6
    ref_fz = (ref_fg - fg_mean) / fg_std

    def _stats(a):
        return float(a.mean()), float(a.std()) + 1e-6

    zl_m, zl_s = _stats(0.54 * ref_pl + 0.28 * ref_dl + 0.18 * ref_mw)
    zr_m, zr_s = _stats(0.54 * ref_pr + 0.28 * ref_dr + 0.18 * ref_mw)
    h1a_m, h1a_s = _stats(0.56 * ref_h1 + 0.26 * ref_mw + 0.18 * ref_fz)
    h1b_m, h1b_s = _stats(0.56 * ref_h2 + 0.26 * ref_mw + 0.18 * ref_fz)
    h2a_m, h2a_s = _stats(0.56 * ref_pl + 0.26 * ref_mw + 0.18 * ref_fz)
    h2b_m, h2b_s = _stats(0.56 * ref_pr + 0.26 * ref_mw + 0.18 * ref_fz)
    del ref_pl, ref_pr, ref_h1, ref_h2, ref_mw, ref_dl, ref_dr, ref_fg, ref_fz

    xs_full = np.linspace(X_MIN, X_MAX, RES, dtype=np.float32)
    ys_full = np.linspace(Y_MAX, Y_MIN, RES, dtype=np.float32)

    def _primary_x(y):
        t = np.clip((y - (-1.45)) / 4.10, 0.0, 1.0)
        return 0.62 * (t * t * (2.2 - 1.2 * t))

    def _primary_dxdy(y):
        t = np.clip((y - (-1.45)) / 4.10, 0.0, 1.0)
        mask = (y > -1.45) & (y < 2.65)
        d = (0.62 / 4.10) * (4.4 * t - 3.6 * t * t)
        return np.where(mask, d, 0.0)


    out_rgb = np.empty((RES, RES, 3), dtype=np.uint8)
    n_strips = (RES + STRIP_H - 1) // STRIP_H

    print(f"[2/3] Streaming {n_strips} halo-padded 16K strips ({STRIP_H}x{RES}, halo={HALO}px)...", flush=True)
    for si in range(n_strips):
        y0 = si * STRIP_H
        y1 = min(RES, y0 + STRIP_H)
        hy0 = max(0, y0 - HALO)
        hy1 = min(RES, y1 + HALO)
        h_cur = hy1 - hy0

        Y = ys_full[hy0:hy1, None]
        X = np.broadcast_to(xs_full[None, :], (h_cur, RES))
        Y = np.broadcast_to(Y, (h_cur, RES))

        pl = patch_l.sample_strip(hy0, hy1, RES)
        pr = patch_r.sample_strip(hy0, hy1, RES)
        ph1 = patch_h1.sample_strip(hy0, hy1, RES)
        ph2 = patch_h2.sample_strip(hy0, hy1, RES)
        fa = fbm_a.sample_strip(hy0, hy1, RES)
        fb = fbm_b.sample_strip(hy0, hy1, RES)
        fg = fine_grit_spec.sample_strip(hy0, hy1, RES)
        mw = mop_wipe_spec.sample_strip(hy0, hy1, RES)
        dl = dust_l_spec.sample_strip(hy0, hy1, RES)
        dr = dust_r_spec.sample_strip(hy0, hy1, RES)

        wobble_u = (fa - 0.5) * 2.0
        wobble_s = (fb - 0.5) * 2.0

        near_wheels = np.zeros((h_cur, RES), dtype=np.float32)
        for yw in (-3.880, -1.495):
            for sxw in (-1.0, +1.0):
                d2 = ((X - sxw * 0.652) / 0.42) ** 2 + ((Y - yw) / 0.62) ** 2
                near_wheels = np.maximum(near_wheels, np.exp(-1.5 * d2))

        z_l = ((0.54 * pl + 0.28 * dl + 0.18 * mw) - zl_m) / zl_s + 0.55 * near_wheels
        z_r = ((0.54 * pr + 0.28 * dr + 0.18 * mw) - zr_m) / zr_s + 0.55 * near_wheels
        drop_l = _smoothstep(-0.30, 0.48, z_l)
        drop_r = _smoothstep(-0.30, 0.48, z_r)
        grit_erode = _smoothstep(0.22, 0.60, fg)


        h_tread_all, h_rib_all, h_corr_all = _eval_primary_curve(
            X, Y, _primary_x, _primary_dxdy,
            wheel_half_track=0.652,
            fade_in_y=(-4.36, -4.08),
            fade_out_y=(1.65, 2.55),
            wobble_u=wobble_u, wobble_s=wobble_s,
            drop_l=drop_l, drop_r=drop_r,
            bias_l=fb, bias_r=fa,
            erode_l=0.52 * fa + 0.48 * fg,
            erode_r=0.52 * fb + 0.48 * fg,
        )
        g_fwd = 0.52 * (1.0 - 0.35 * _smoothstep(-3.88, -4.34, Y))
        g_mid = g_fwd + (0.82 - g_fwd) * _smoothstep(-3.98, -3.76, Y)
        seg_gain = (g_mid + (1.00 - g_mid) * _smoothstep(-1.60, -1.38, Y)).astype(np.float32)
        dist_fade = 1.0 - 0.25 * _smoothstep(0.70, 2.45, Y)

        h_tread = h_tread_all * seg_gain * dist_fade
        h_rib   = h_rib_all * seg_gain * dist_fade
        h_corr  = h_corr_all * seg_gain * dist_fade

        pivot_scuff = np.zeros((h_cur, RES), dtype=np.float32)
        for yw in (-3.880, -1.495):
            for sxw in (-1.0, +1.0):
                dxw = (X - sxw * 0.652 + wobble_u * 0.015) / 0.105
                dyw = (Y - yw + wobble_s * 0.022) / 0.165
                pad = np.exp(-2.4 * (dxw * dxw + dyw * dyw)) * (0.45 + 0.55 * grit_erode)
                pivot_scuff = np.maximum(pivot_scuff, pad)


        fz = (fg - fg_mean) / fg_std
        he_1a = ((0.56 * ph1 + 0.26 * mw + 0.18 * fz) - h1a_m) / h1a_s
        he_1b = ((0.56 * ph2 + 0.26 * mw + 0.18 * fz) - h1b_m) / h1b_s
        he_2a = ((0.56 * pl  + 0.26 * mw + 0.18 * fz) - h2a_m) / h2a_s
        he_2b = ((0.56 * pr  + 0.26 * mw + 0.18 * fz) - h2b_m) / h2b_s

        hist1_t, hist1_c = _eval_worn_arc(X, Y, 6.45, -5.85, 5.42, 0.56, 118.0, 182.0, wobble_u, wobble_s, he_1a, he_1b, 18.0)
        hist2_t, hist2_c = _eval_worn_arc(X, Y, 0.60, 4.15, 3.65, 0.52, 252.0, 322.0, -wobble_u, wobble_s, he_2a, he_2b, 18.0)
        hist3_t, hist3_c = _eval_worn_arc(X, Y, -5.30, -6.40, 4.35, 0.58, -15.0, 68.0, wobble_u, -wobble_s, he_1b, he_2a, 18.0)

        hist_tread = np.clip(np.maximum(hist1_t * 0.78, np.maximum(hist2_t * 0.66, hist3_t * 0.60)), 0.0, 1.0)
        hist_corr  = np.clip(np.maximum(hist1_c * 0.68, np.maximum(hist2_c * 0.58, hist3_c * 0.52)), 0.0, 1.0)

        ch_r_raw = np.clip(h_rib * 0.85 + pivot_scuff * 0.90 + hist_tread * 0.68, 0.0, 1.0)
        ch_g_raw = np.clip(h_tread * 1.00 + hist_tread * 0.74, 0.0, 1.0)
        ch_b_raw = np.clip(np.maximum(h_corr * 0.86, np.maximum(pivot_scuff * 0.78, hist_corr * 0.72)), 0.0, 1.0)


        ch_r = _blur_2d(ch_r_raw, radius=3.6) * (0.28 + 0.72 * grit_erode)
        ch_g = _blur_2d(ch_g_raw, radius=2.2) * (0.18 + 0.82 * grit_erode)
        ch_b = _blur_2d(ch_b_raw, radius=8.0)

        ex = _smoothstep(X_MIN, X_MIN + 0.40, X) * (1.0 - _smoothstep(X_MAX - 0.40, X_MAX, X))
        ey = _smoothstep(Y_MIN, Y_MIN + 0.40, Y) * (1.0 - _smoothstep(Y_MAX - 0.40, Y_MAX, Y))
        em = ex * ey


        c0 = y0 - hy0
        c1 = c0 + (y1 - y0)
        out_rgb[y0:y1, :, 0] = (np.clip(ch_r[c0:c1] * em[c0:c1], 0.0, 1.0) * 255.0).astype(np.uint8)
        out_rgb[y0:y1, :, 1] = (np.clip(ch_g[c0:c1] * em[c0:c1], 0.0, 1.0) * 255.0).astype(np.uint8)
        out_rgb[y0:y1, :, 2] = (np.clip(ch_b[c0:c1] * em[c0:c1], 0.0, 1.0) * 255.0).astype(np.uint8)
        if (si + 1) % 4 == 0 or si == n_strips - 1:
            print(f"    strip {si + 1}/{n_strips} done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("[3/3] Encoding 16K PNG (compress_level=4 for fast high-ratio write)...", flush=True)
    img = Image.fromarray(out_rgb, mode="RGB")
    img.save(OUT_PATH, compress_level=4)
    sz_mb = os.path.getsize(OUT_PATH) / 1048576.0
    print(f"[Done] Saved floor tire-track mask: {OUT_PATH} ({RES}x{RES}, {sz_mb:.2f} MB) in {time.perf_counter() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    generate_floor_tiretracks()
