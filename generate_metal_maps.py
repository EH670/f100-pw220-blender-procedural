import os
import math
import numpy as np
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_THIS_DIR, "assets_masks_4k")
os.makedirs(OUT_DIR, exist_ok=True)
N = 2048


def _tileable_fbm(n, base_freq, octaves=5, persistence=0.55, seed=42, stretch=(1.0, 1.0)):

    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(n) * n
    fx = np.fft.rfftfreq(n) * n
    FX, FY = np.meshgrid(fx, fy)
    FXs = FX / stretch[0]
    FYs = FY / stretch[1]
    R = np.sqrt(FXs * FXs + FYs * FYs)

    acc = np.zeros((n, n), dtype=np.float32)
    amp = 1.0
    norm = 0.0
    for o in range(octaves):
        f0 = base_freq * (2.0 ** o)

        bw = max(1.5, f0 * 0.55)
        w = np.exp(-0.5 * ((R - f0) / bw) ** 2)
        w[0, 0] = 0.0
        phase = rng.uniform(0.0, 2.0 * np.pi, size=R.shape)
        spec = w * (np.cos(phase) + 1j * np.sin(phase))
        layer = np.fft.irfft2(spec, s=(n, n)).astype(np.float32)
        std = layer.std() + 1e-8
        acc += (layer / std) * amp
        norm += amp
        amp *= persistence
    acc /= norm

    lo, hi = np.percentile(acc, [1.5, 98.5])
    return np.clip((acc - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def gen_galvanized_crystal(n=N):

    rng = np.random.default_rng(2026)

    G = 24

    jx = rng.uniform(0.08, 0.92, size=(G, G))
    jy = rng.uniform(0.08, 0.92, size=(G, G))

    c_val = rng.uniform(0.10, 0.94, size=(G, G)).astype(np.float32)
    c_tx  = rng.uniform(-1.0, 1.0, size=(G, G)).astype(np.float32)
    c_ty  = rng.uniform(-1.0, 1.0, size=(G, G)).astype(np.float32)
    c_ang = rng.uniform(0.0, np.pi, size=(G, G)).astype(np.float32)

    u = np.linspace(0.0, G, n, endpoint=False, dtype=np.float32)
    v = np.linspace(0.0, G, n, endpoint=False, dtype=np.float32)
    U, V = np.meshgrid(u, v)
    iu = np.floor(U).astype(np.int32)
    iv = np.floor(V).astype(np.int32)

    best_d2 = np.full((n, n), 1e9, dtype=np.float32)
    best_id_x = np.zeros((n, n), dtype=np.int32)
    best_id_y = np.zeros((n, n), dtype=np.int32)
    best_dx = np.zeros((n, n), dtype=np.float32)
    best_dy = np.zeros((n, n), dtype=np.float32)


    w1 = _tileable_fbm(n, base_freq=18.0, octaves=3, seed=71) - 0.5
    w2 = _tileable_fbm(n, base_freq=18.0, octaves=3, seed=72) - 0.5
    Uw = U + w1 * 0.14
    Vw = V + w2 * 0.14

    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx = (iu + ox) % G
            cy = (iv + oy) % G
            px = (iu + ox).astype(np.float32) + jx[cy, cx]
            py = (iv + oy).astype(np.float32) + jy[cy, cx]
            dx = Uw - px
            dy = Vw - py
            d2 = dx * dx + dy * dy
            m = d2 < best_d2
            best_d2 = np.where(m, d2, best_d2)
            best_id_x = np.where(m, cx, best_id_x)
            best_id_y = np.where(m, cy, best_id_y)
            best_dx = np.where(m, dx, best_dx)
            best_dy = np.where(m, dy, best_dy)

    val = c_val[best_id_y, best_id_x]
    tx  = c_tx[best_id_y, best_id_x]
    ty  = c_ty[best_id_y, best_id_x]
    ang = c_ang[best_id_y, best_id_x]


    proj = best_dx * np.cos(ang) + best_dy * np.sin(ang)
    perp = -best_dx * np.sin(ang) + best_dy * np.cos(ang)
    radial_ang = np.arctan2(perp, proj)
    dend = 0.5 + 0.5 * np.cos(radial_ang * 12.0 + np.abs(proj) * 10.0)
    val_fin = np.clip(val * 0.88 + dend * 0.12, 0.0, 1.0)
    tx_fin  = np.clip(0.5 + 0.45 * tx + 0.04 * (dend - 0.5), 0.0, 1.0)
    ty_fin  = np.clip(0.5 + 0.45 * ty + 0.04 * (dend - 0.5), 0.0, 1.0)

    rgb = np.stack([
        (val_fin * 255.0).astype(np.uint8),
        (tx_fin  * 255.0).astype(np.uint8),
        (ty_fin  * 255.0).astype(np.uint8),
    ], axis=-1)
    p = os.path.join(OUT_DIR, "metal_2k_galvanized_crystal.png")
    Image.fromarray(rgb, mode="RGB").save(p, compress_level=6)
    print(f"[SAVED] {p} ({rgb.shape})")


def gen_aniso_grain(n=N):


    broad = _tileable_fbm(n, base_freq=8.0,  octaves=4, persistence=0.60, seed=101, stretch=(14.0, 0.7))
    med   = _tileable_fbm(n, base_freq=28.0, octaves=4, persistence=0.65, seed=102, stretch=(22.0, 0.45))
    fine  = _tileable_fbm(n, base_freq=95.0, octaves=3, persistence=0.65, seed=103, stretch=(30.0, 0.35))

    scores = np.clip((med - 0.62) / 0.38, 0.0, 1.0) ** 1.6
    dark_pass = np.clip((0.36 - broad) / 0.36, 0.0, 1.0) ** 1.5

    comb = 0.42 * broad + 0.36 * med + 0.22 * fine + 0.24 * scores - 0.22 * dark_pass
    lo, hi = np.percentile(comb, [1.0, 99.0])
    out = np.clip((comb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    p = os.path.join(OUT_DIR, "metal_2k_aniso_grain.png")
    Image.fromarray((out * 255.0).astype(np.uint8), mode="L").save(p, compress_level=6)
    print(f"[SAVED] {p} ({out.shape})")


def gen_roughness_breakup(n=N):


    wx = _tileable_fbm(n, base_freq=4.0, octaves=3, seed=201) - 0.5
    wy = _tileable_fbm(n, base_freq=4.0, octaves=3, seed=202) - 0.5
    base = _tileable_fbm(n, base_freq=5.0, octaves=6, persistence=0.62, seed=203)

    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sx = (xx + (wx * 140.0).astype(np.int32)) % n
    sy = (yy + (wy * 140.0).astype(np.int32)) % n
    warped = base[sy, sx]


    islands = np.clip((warped - 0.30) / 0.42, 0.0, 1.0)
    islands = islands * islands * (3.0 - 2.0 * islands)


    wipe_a = _tileable_fbm(n, base_freq=10.0, octaves=4, seed=204, stretch=(6.0, 1.2))
    wipe_b = _tileable_fbm(n, base_freq=16.0, octaves=4, seed=205, stretch=(1.3, 5.5))
    wipes = np.clip((wipe_a - 0.48) / 0.34, 0.0, 1.0) * 0.55 + np.clip((wipe_b - 0.55) / 0.30, 0.0, 1.0) * 0.35


    ring_f = _tileable_fbm(n, base_freq=9.0, octaves=4, seed=206)
    rings = np.exp(-((ring_f - 0.58) / 0.022) ** 2) * 0.45


    grit = _tileable_fbm(n, base_freq=64.0, octaves=4, persistence=0.68, seed=207)


    rough = islands * (0.72 + 0.28 * grit) + rings * 0.42 - wipes * 0.46 + (grit - 0.5) * 0.14
    lo, hi = np.percentile(rough, [2.0, 98.0])
    out = np.clip((rough - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    out = np.clip(0.5 + (out - 0.5) * 1.28, 0.0, 1.0)
    p = os.path.join(OUT_DIR, "metal_2k_roughness_breakup.png")
    Image.fromarray((out * 255.0).astype(np.uint8), mode="L").save(p, compress_level=6)
    print(f"[SAVED] {p} ({out.shape})")


if __name__ == "__main__":
    gen_galvanized_crystal()
    gen_aniso_grain()
    gen_roughness_breakup()
