import gc
import math
import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont


Image.MAX_IMAGE_PIXELS = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_THIS_DIR, "assets_masks_4k", "mask_4k_floor_baymarkings.png")
TIRE_PATH = os.path.join(_THIS_DIR, "assets_masks_4k", "mask_4k_floor_tiretracks.png")

RES = 16384
X_MIN, X_MAX = -5.0, 5.0
Y_MIN, Y_MAX = -7.25, 2.75


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _value_noise_2d(grid_x: int, grid_y: int, out_res: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.random((grid_y, grid_x), dtype=np.float32)
    img = Image.fromarray((base * 255.0).astype(np.uint8), mode="L")
    up = img.resize((out_res, out_res), resample=Image.Resampling.BICUBIC)
    return np.asarray(up, dtype=np.float32) * (1.0 / 255.0)


def _norm_z_inplace(arr: np.ndarray) -> np.ndarray:
    m = float(arr.mean())
    s = float(arr.std()) + 1e-6
    arr -= m
    arr *= (1.0 / s)
    return arr


def _fbm_patch(res: int, seed: int) -> np.ndarray:
    f = _value_noise_2d(24, 24, res, seed + 1)
    f *= 0.36
    f += 0.30 * _value_noise_2d(64, 64, res, seed + 2)
    f += 0.20 * _value_noise_2d(180, 180, res, seed + 3)
    f += 0.14 * _value_noise_2d(520, 520, res, seed + 4)
    return _norm_z_inplace(f)


def _fbm_fine(res: int, seed: int) -> np.ndarray:
    f = _value_noise_2d(120, 120, res, seed + 11)
    f *= 0.24
    f += 0.26 * _value_noise_2d(360, 360, res, seed + 12)
    f += 0.26 * _value_noise_2d(1100, 1100, res, seed + 13)
    f += 0.24 * _value_noise_2d(3200, 3200, res, seed + 14)
    return _norm_z_inplace(f)


def _w2px(x: float, y: float):

    col = (x - X_MIN) / (X_MAX - X_MIN) * (RES - 1)
    row = (Y_MAX - y) / (Y_MAX - Y_MIN) * (RES - 1)
    return col, row


def _roi_bounds(cx: float, cy: float, half_w: float, half_h: float):

    c0, r0 = _w2px(cx - half_w, cy + half_h)
    c1, r1 = _w2px(cx + half_w, cy - half_h)
    ic0 = max(0, int(math.floor(min(c0, c1))))
    ic1 = min(RES, int(math.ceil(max(c0, c1))) + 1)
    ir0 = max(0, int(math.floor(min(r0, r1))))
    ir1 = min(RES, int(math.ceil(max(r0, r1))) + 1)
    return ir0, ir1, ic0, ic1


def _load_stencil_font(size_px: int):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\impact.ttf",
        r"C:\Windows\Fonts\bahnschrift.ttf",
        r"C:\Windows\Fonts\consolab.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size_px)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_stencil_text_layer(w: int, h: int, text: str, font_size: int, stencil_bridges: bool = True) -> Image.Image:

    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    fnt = _load_stencil_font(font_size)
    bbox = d.textbbox((0, 0), text, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (w - tw) // 2 - bbox[0]
    ty = (h - th) // 2 - bbox[1]
    d.text((tx, ty), text, fill=252, font=fnt)

    if stencil_bridges:
        arr = np.asarray(img, dtype=np.uint8).copy()
        step_b = max(24, int(font_size * 0.52))
        bw_half = max(1, int(round(font_size / 62.0)))
        y_mid0 = int(h * 0.44)
        y_mid1 = int(h * 0.56)
        for bx in range(int(tx) + step_b // 2, int(tx + tw), step_b):
            if bw_half + 1 <= bx < w - bw_half - 1:
                arr[y_mid0:y_mid1, bx - bw_half:bx + bw_half] = (
                    arr[y_mid0:y_mid1, bx - bw_half:bx + bw_half] * 0.10
                ).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    return img


def generate_floor_baymarkings():
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


    xs_1d = np.linspace(X_MIN, X_MAX, RES, dtype=np.float32)
    ys_1d = np.linspace(Y_MAX, Y_MIN, RES, dtype=np.float32)


    rgba_arr = np.zeros((RES, RES, 4), dtype=np.uint8)

    print(f"[1/5] Generating 16K ({RES}x{RES}, 0.61mm/px) multi-octave peel & concrete grain fields...", flush=True)
    z_peel = _fbm_patch(RES, seed=1401)
    z_fine = _fbm_fine(RES, seed=1409)

    tire_scuff = np.zeros((RES, RES), dtype=np.float32)
    if os.path.exists(TIRE_PATH):
        t_img = Image.open(TIRE_PATH).convert("RGB")
        if t_img.size != (RES, RES):
            t_img = t_img.resize((RES, RES), resample=Image.Resampling.BILINEAR)
        t_arr = np.asarray(t_img, dtype=np.float32) * (1.0 / 255.0)
        tire_scuff = np.clip(t_arr[:, :, 0] * 1.15 + t_arr[:, :, 1] * 0.95 + t_arr[:, :, 2] * 0.55, 0.0, 1.0)
        del t_img, t_arr


    print("[2/5] Generating floor bay markings and stencils (R)...", flush=True)
    bay_img = Image.new("L", (RES, RES), 0)
    db = ImageDraw.Draw(bay_img)

    sc_k = RES / 4096.0
    c0, r0 = _w2px(-1.36, +0.35)
    c1, r1 = _w2px(+1.36, -4.92)
    line_w = int(round(31 * sc_k))
    db.rectangle([c0, r0, c1, r1], outline=255, width=line_w)

    corner_len = int(round(135 * sc_k))
    cw = int(round(40 * sc_k))
    for cx_w, cy_w, sx_c, sy_c in (
        (-1.36, +0.35, +1, +1),
        (+1.36, +0.35, -1, +1),
        (-1.36, -4.92, +1, -1),
        (+1.36, -4.92, -1, -1),
    ):
        pc, pr = _w2px(cx_w, cy_w)
        db.line([(pc, pr), (pc + sx_c * corner_len, pr)], fill=255, width=cw)
        db.line([(pc, pr), (pc, pr + sy_c * corner_len)], fill=255, width=cw)

    for sx_m in (-1.0, +1.0):
        mc0, mr0 = _w2px(sx_m * 1.36, -2.30)
        mc1, mr1 = _w2px(sx_m * 1.21, -2.30)
        db.line([(mc0, mr0), (mc1, mr1)], fill=252, width=int(round(18 * sc_k)))


    port_w_px = int(round(1240 * sc_k))
    port_h_px = int(round(128 * sc_k))
    txt_port = _draw_stencil_text_layer(port_w_px, port_h_px, "BAY 04  •  F100-PW-220  •  FOD FREE ZONE", font_size=int(round(58 * sc_k)))
    txt_port_rot = txt_port.transpose(Image.Transpose.ROTATE_90)
    pc_paste, pr_paste = _w2px(+1.46, -0.10)
    bay_img.paste(txt_port_rot, (int(round(pc_paste)), int(round(pr_paste))), txt_port_rot)
    del txt_port, txt_port_rot


    fwd_w_px = int(round(1060 * sc_k))
    fwd_h_px = int(round(132 * sc_k))
    txt_fwd = _draw_stencil_text_layer(fwd_w_px, fwd_h_px, "BAY 04  •  FOD FREE  •  CLEAR", font_size=int(round(62 * sc_k)))
    fc_paste, fr_paste = _w2px(-1.30, -5.04)
    bay_img.paste(txt_fwd, (int(round(fc_paste)), int(round(fr_paste))), txt_fwd)
    del txt_fwd

    bay_raw = np.asarray(bay_img, dtype=np.float32) * (1.0 / 255.0)
    del bay_img


    keep_score = 0.54 * z_peel + 0.46 * z_fine
    _norm_z_inplace(keep_score)
    keep_score += 0.34

    st_x = (xs_1d > 1.43)[None, :]
    st_y = (ys_1d < -4.98)[:, None]
    keep_score += 0.96 * (st_x | st_y).astype(np.float32)
    keep_score -= 1.55 * tire_scuff
    paint_keep = _smoothstep(-0.18, 0.20, keep_score)
    del keep_score
    bay_raw *= paint_keep
    del paint_keep
    np.clip(bay_raw, 0.0, 1.0, out=bay_raw)
    bay_mean, bay_max = float(bay_raw.mean()), float(bay_raw.max())
    rgba_arr[:, :, 0] = (bay_raw * 255.0).astype(np.uint8)
    del bay_raw
    gc.collect()


    print("[3/5] Synthesizing 16K multi-drop oil spots & curved absorbent-pad wipe halo (G)...", flush=True)
    oil_g = np.zeros((RES, RES), dtype=np.float32)


    ir0, ir1, ic0, ic1 = _roi_bounds(0.16, -2.52, 1.10, 0.95)
    dx_w = xs_1d[ic0:ic1][None, :] - 0.16
    dy_w = ys_1d[ir0:ir1][:, None] - (-2.52)
    r_w = np.hypot(dx_w / 0.56, dy_w / 0.44)
    ang_w = np.arctan2(dy_w, dx_w)
    z_oil_roi = z_peel[ir0:ir1, ic0:ic1]
    wipe_ring = np.exp(-((r_w - 0.88) / 0.26) ** 2) * _smoothstep(-0.35, 0.45, np.cos(ang_w - 0.35))
    wipe_streaks = 0.58 + 0.42 * np.sin(r_w * 34.0 + 1.8 * z_oil_roi)
    wipe_halo = wipe_ring * wipe_streaks * _smoothstep(-0.4, 0.8, z_oil_roi) * 0.44
    oil_g[ir0:ir1, ic0:ic1] = np.maximum(oil_g[ir0:ir1, ic0:ic1], wipe_halo)


    oil_drops = [
        (+0.43, -2.54, 0.042, 0.036, 0.4, 0.92),
        (+0.48, -2.47, 0.022, 0.019, 1.1, 0.82),
        (+0.39, -2.31, 0.032, 0.026, -0.5, 0.85),
        (+0.16, -2.27, 0.048, 0.034, 0.2, 0.88),
        (+0.22, -2.21, 0.018, 0.016, 0.8, 0.78),
        (+0.02, -2.29, 0.026, 0.024, -0.3, 0.75),
        (-0.42, -2.64, 0.038, 0.030, 0.6, 0.84),
        (-0.46, -2.72, 0.020, 0.018, 1.4, 0.72),
        (+0.40, -2.84, 0.028, 0.022, -0.8, 0.76),
        (-0.18, -3.02, 0.034, 0.028, 0.3, 0.70),
        (+0.18, +0.16, 0.036, 0.029, 0.5, 0.68),
        (+0.25, +0.23, 0.017, 0.015, 0.1, 0.62),
        (-0.19, +0.28, 0.030, 0.025, -0.4, 0.60),
    ]
    for cx, cy, rx, ry, ang, pk in oil_drops:
        rmax = max(rx, ry) * 2.4
        ir0, ir1, ic0, ic1 = _roi_bounds(cx, cy, rmax, rmax)
        ca, sa = math.cos(ang), math.sin(ang)
        dx = xs_1d[ic0:ic1][None, :] - cx
        dy = ys_1d[ir0:ir1][:, None] - cy
        ux = (dx * ca + dy * sa) / rx
        uy = (-dx * sa + dy * ca) / ry
        d = np.hypot(ux, uy) + 0.16 * z_fine[ir0:ir1, ic0:ic1]
        core = (1.0 - _smoothstep(0.45, 0.95, d)) * pk
        rim = (1.0 - _smoothstep(0.80, 1.55, d)) * (pk * 0.34)
        oil_g[ir0:ir1, ic0:ic1] = np.maximum(oil_g[ir0:ir1, ic0:ic1], np.maximum(core, rim))

    np.clip(oil_g, 0.0, 1.0, out=oil_g)
    oil_mean, oil_max = float(oil_g.mean()), float(oil_g.max())
    rgba_arr[:, :, 1] = (oil_g * 255.0).astype(np.uint8)
    del oil_g
    gc.collect()


    print("[4/5] Synthesizing 16K circular steel jack-pad press & scuff rings (B)...", flush=True)
    jack_b = np.zeros((RES, RES), dtype=np.float32)
    jack_sites = [
        (+0.795, -4.185, 0.056, 0.90),
        (+0.768, -4.238, 0.055, 0.42),
        (-0.788, -4.192, 0.056, 0.88),
        (-0.762, -4.235, 0.055, 0.40),
        (+0.802, -0.865, 0.056, 0.92),
        (+0.770, -0.818, 0.055, 0.45),
        (-0.792, -0.858, 0.056, 0.88),
        (-0.765, -0.822, 0.055, 0.40),
    ]
    for jx, jy, jr, jstr in jack_sites:
        pad = jr + 0.045
        ir0, ir1, ic0, ic1 = _roi_bounds(jx, jy, pad, pad)
        dj = np.hypot(xs_1d[ic0:ic1][None, :] - jx, ys_1d[ir0:ir1][:, None] - jy) + 0.0028 * z_fine[ir0:ir1, ic0:ic1]
        ring = np.exp(-((dj - jr) / 0.0095) ** 2)
        disk = (1.0 - _smoothstep(jr - 0.008, jr + 0.004, dj)) * 0.36
        j_break = _smoothstep(-0.75, 0.65, 0.55 * z_peel[ir0:ir1, ic0:ic1] + 0.45 * z_fine[ir0:ir1, ic0:ic1])
        jack_b[ir0:ir1, ic0:ic1] = np.maximum(jack_b[ir0:ir1, ic0:ic1], np.maximum(ring, disk) * j_break * jstr)

    np.clip(jack_b, 0.0, 1.0, out=jack_b)
    jack_mean, jack_max = float(jack_b.mean()), float(jack_b.max())
    rgba_arr[:, :, 2] = (jack_b * 255.0).astype(np.uint8)
    del jack_b
    gc.collect()


    print("[5/5] Synthesizing 16K spalled concrete saw-cut expansion joints (A)...", flush=True)
    dist_jx_1d = (0.50 - np.abs(((xs_1d / 1.80 + 0.50) % 1.0) - 0.50)) * 1.80
    dist_jy_1d = (0.50 - np.abs(((ys_1d / 1.80 + 0.25) % 1.0) - 0.50)) * 1.80


    joint_sum = 0.0
    joint_max = 0.0
    strip_h = 1024
    for r_st in range(0, RES, strip_h):
        r_ed = min(RES, r_st + strip_h)
        dist_j = np.minimum(dist_jx_1d[None, :], dist_jy_1d[r_st:r_ed, None])
        zf_s = z_fine[r_st:r_ed, :]
        zp_s = z_peel[r_st:r_ed, :]
        ts_s = tire_scuff[r_st:r_ed, :]

        d_core = dist_j + 0.0008 * zf_s
        joint_core = 1.0 - _smoothstep(0.0020, 0.0042, d_core)

        spall_trigger = _smoothstep(0.75, 1.55, 0.62 * zp_s + 0.38 * zf_s + 0.55 * ts_s)
        spall_reach = 0.0042 + 0.0145 * spall_trigger * np.clip(0.5 + 0.5 * zf_s, 0.0, 1.0)
        spall_chip = (1.0 - _smoothstep(0.0022, spall_reach, dist_j)) * spall_trigger * 0.62

        ja = np.clip(np.maximum(joint_core * 0.96, spall_chip), 0.0, 1.0)
        joint_sum += float(ja.sum())
        j_m = float(ja.max())
        if j_m > joint_max:
            joint_max = j_m
        rgba_arr[r_st:r_ed, :, 3] = (ja * 255.0).astype(np.uint8)

    del z_peel, z_fine, tire_scuff
    gc.collect()
    joint_mean = joint_sum / float(RES * RES)

    print(f"[SAVE] Encoding 16K RGBA PNG -> {OUT_PATH} ...", flush=True)
    im_out = Image.fromarray(rgba_arr, mode="RGBA")
    im_out.save(OUT_PATH, compress_level=6)
    sz_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    dt = time.perf_counter() - t0
    print(f"[DONE] Saved {OUT_PATH} ({RES}x{RES} RGBA, {sz_mb:.2f} MB) in {dt:.1f}s", flush=True)
    print(f"       Channel means -> R(bay)={bay_mean:.4f} max={bay_max:.3f}, "
          f"G(oil)={oil_mean:.4f} max={oil_max:.3f}, "
          f"B(jack)={jack_mean:.4f} max={jack_max:.3f}, "
          f"A(joint)={joint_mean:.4f} max={joint_max:.3f}", flush=True)

if __name__ == "__main__":
    generate_floor_baymarkings()
