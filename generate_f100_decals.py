import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_THIS_DIR, "assets_decals")
FONT_SEARCH_DIRS = [
    r"C:\Windows\Fonts",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/System/Library/Fonts/Supplemental",
]


S = 2


def _font(names, size: int):
    if isinstance(names, str):
        names = [names]
    for d_path in FONT_SEARCH_DIRS:
        if not os.path.isdir(d_path):
            continue
        for fn in names:
            p = os.path.join(d_path, fn)
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, int(round(size * S)))
                except Exception:
                    continue
    for fallback in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"):
        for d_path in FONT_SEARCH_DIRS:
            p = os.path.join(d_path, fallback)
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, int(round(size * S)))
                except Exception:
                    continue
    return ImageFont.load_default()


F_COND_BOLD = ["bahnschrift.ttf", "impact.ttf", "arialbd.ttf"]
F_SANS_BOLD = ["arialbd.ttf", "segoeuib.ttf", "tahomabd.ttf"]
F_MONO_BOLD = ["consolab.ttf", "courbd.ttf", "arialbd.ttf"]


def _sc(*vals):
    return tuple(int(round(v * S)) for v in vals)


def _draw_centered_text(draw, box, text, font, fill):
    x0, y0, x1, y1 = _sc(*box)
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx = x0 + (x1 - x0 - tw) * 0.5 - bb[0]
    ty = y0 + (y1 - y0 - th) * 0.5 - bb[1]
    draw.text((tx, ty), text, font=font, fill=fill)


def _draw_barcode(draw, x0, y0, x1, y1, fill=(20, 22, 26, 244)):
    x0, y0, x1, y1 = _sc(x0, y0, x1, y1)
    w = x1 - x0
    x = x0
    pattern = [3, 1, 2, 1, 4, 2, 1, 3, 1, 1, 2, 3, 1, 2, 1, 4, 1, 2, 2, 1, 3, 1, 2, 1, 1, 3, 2, 1, 4, 1, 2, 2]
    idx = 0
    while x < x1 - 4 * S:
        bw = max(2 * S, int(pattern[idx % len(pattern)] * (w / 140.0)))
        if idx % 2 == 0:
            draw.rectangle([x, y0, min(x1, x + bw), y1], fill=fill)
        x += bw + max(2 * S, int(w / 180.0))
        idx += 1


def _draw_warning_triangle(draw, cx, cy, r, fill_bg, stroke_col, symbol_col, stroke_w=12):
    cx, cy, r = cx * S, cy * S, r * S
    pts = [
        (cx, cy - r),
        (cx + r * 0.94, cy + r * 0.72),
        (cx - r * 0.94, cy + r * 0.72),
    ]
    draw.polygon(pts, fill=fill_bg)
    draw.line(pts + [pts[0]], fill=stroke_col, width=int(stroke_w * S), joint="curve")
    draw.rounded_rectangle(
        [cx - r * 0.11, cy - r * 0.42, cx + r * 0.11, cy + r * 0.22],
        radius=int(8 * S), fill=symbol_col,
    )
    draw.ellipse([cx - r * 0.12, cy + r * 0.34, cx + r * 0.12, cy + r * 0.58], fill=symbol_col)


def _draw_rivet_head(draw, cx, cy, r=14):

    cx, cy, r = int(cx * S), int(cy * S), int(r * S)
    draw.ellipse([cx - r - 2 * S, cy - r - 2 * S, cx + r + 2 * S, cy + r + 2 * S], fill=(16, 18, 22, 235))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(168, 174, 180, 250), outline=(92, 96, 102, 250), width=2 * S)

    hr = int(r * 0.62)
    draw.arc([cx - hr, cy - hr, cx + hr, cy + hr], start=200, end=340, fill=(232, 236, 240, 230), width=2 * S)

    cr = max(2, int(r * 0.24))
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(56, 60, 66, 240))


def _draw_corner_rivets(draw, box, inset=28, r=13):
    x0, y0, x1, y1 = box
    for px, py in ((x0 + inset, y0 + inset), (x1 - inset, y0 + inset),
                   (x0 + inset, y1 - inset), (x1 - inset, y1 - inset)):
        _draw_rivet_head(draw, px, py, r=r)


def _apply_micro_grain(img: Image.Image, mode: str = "stencil", seed: int = 42, strength: float = 1.0) -> Image.Image:

    arr = np.asarray(img, dtype=np.float32).copy()
    H, W, _ = arr.shape
    a = arr[:, :, 3]
    mask = a > 12.0
    if not np.any(mask):
        return img

    rng = np.random.default_rng(seed)
    if mode == "metal_plate":

        row_n = rng.normal(0.0, 4.2 * strength, size=(H, 1)).astype(np.float32)
        pix_n = rng.normal(0.0, 2.4 * strength, size=(H, W)).astype(np.float32)
        delta = (row_n + pix_n) * mask
        for c in range(3):
            arr[:, :, c] = np.clip(arr[:, :, c] + delta, 0.0, 255.0)
    else:

        hs, ws = max(8, H // 16), max(8, W // 16)
        coarse = rng.normal(0.0, 7.5 * strength, size=(hs, ws)).astype(np.float32)
        coarse_img = Image.fromarray(np.clip(coarse + 128.0, 0, 255).astype(np.uint8), mode="L").resize((W, H), Image.BILINEAR)
        c_arr = (np.asarray(coarse_img, dtype=np.float32) - 128.0)
        fine = rng.normal(0.0, 3.8 * strength, size=(H, W)).astype(np.float32)
        delta_rgb = (c_arr * 0.75 + fine) * mask
        for c in range(3):
            arr[:, :, c] = np.clip(arr[:, :, c] + delta_rgb, 0.0, 255.0)

        arr[:, :, 3] = np.where(mask, np.clip(a + (c_arr * 0.55 + fine * 0.6), 0.0, 255.0), a)

    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def gen_tk9_rail_blue():

    W, H = _sc(2048, 640)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    blue = (42, 118, 206, 248)
    dark = (34, 50, 74, 238)

    f_big = _font(F_SANS_BOLD, 340)
    f_sub = _font(F_COND_BOLD, 70)
    f_sm  = _font(F_MONO_BOLD, 52)

    _draw_centered_text(d, (30, 50, 950, 490), "TK-9", f_big, blue)
    d.line([_sc(985, 95), _sc(985, 540)], fill=blue, width=10 * S)
    d.text(_sc(1030, 102), "USAF M-4 ENGINE STAND", font=f_sub, fill=dark)
    d.text(_sc(1030, 224), "F100-PW-220 / -229", font=f_sub, fill=blue)
    d.text(_sc(1030, 355), "MAX GROSS LOAD: 6,500 LBS", font=f_sm, fill=dark)
    d.text(_sc(1030, 445), "LOCK PINS BEFORE TRANSPORT", font=f_sm, fill=(168, 34, 24, 238))
    d.rectangle(_sc(38, 520, 945, 540), fill=blue)
    img = _apply_micro_grain(img, mode="stencil", seed=101, strength=0.95)
    img.save(os.path.join(OUT_DIR, "decal_tk9_rail_blue.png"))


def gen_reflector_redwhite():

    W, H = _sc(2048, 448)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    margin = 18 * S
    inner = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    di = ImageDraw.Draw(inner)
    di.rectangle([margin + 10 * S, margin + 10 * S, W - margin - 10 * S, H - margin - 10 * S], fill=(230, 232, 236, 255))
    red_c = (196, 24, 20, 255)
    di.rectangle([margin + 10 * S, margin + 10 * S, 820 * S, H - margin - 10 * S], fill=red_c)
    di.rectangle([1228 * S, margin + 10 * S, W - margin - 10 * S, H - margin - 10 * S], fill=red_c)


    grid_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dg = ImageDraw.Draw(grid_img)

    step = 26 * S
    for x in range(-H, W + H, step):
        dg.line([(x, 0), (x + H, H)], fill=(255, 255, 255, 34), width=2 * S)
        dg.line([(x, H), (x + H, 0)], fill=(0, 0, 0, 26), width=2 * S)

    hex_r = 46 * S
    dx = hex_r * 1.5
    dy = hex_r * math.sqrt(3.0)
    col = 0
    x_pos = -hex_r
    while x_pos < W + hex_r * 2:
        y_off = (dy * 0.5) if (col % 2 == 1) else 0.0
        y_pos = -hex_r + y_off
        while y_pos < H + hex_r * 2:
            pts = [
                (x_pos + hex_r * math.cos(math.radians(60 * k)),
                 y_pos + hex_r * math.sin(math.radians(60 * k)))
                for k in range(6)
            ]
            dg.line(pts + [pts[0]], fill=(255, 255, 255, 42), width=3 * S)
            y_pos += dy
        x_pos += dx
        col += 1

    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [margin + 10 * S, margin + 10 * S, W - margin - 10 * S, H - margin - 10 * S],
        radius=16 * S, fill=255,
    )
    inner = Image.alpha_composite(inner, grid_img)
    img.paste(inner, (0, 0), mask)
    d.rounded_rectangle([margin, margin, W - margin, H - margin], radius=22 * S, outline=(42, 45, 52, 255), width=12 * S)
    f_tiny = _font(F_MONO_BOLD, 44)
    _draw_centered_text(d, (830, 448 - 115, 1218, 448 - 40), "DOT-C2  GSE", f_tiny, (130, 30, 26, 218))
    img.save(os.path.join(OUT_DIR, "decal_reflector_redwhite.png"))


def gen_forklift_chevron():

    W, H = _sc(2048, 162)
    img = Image.new("RGBA", (W, H), (228, 174, 18, 255))
    d = ImageDraw.Draw(img)
    blk = (20, 22, 25, 255)
    stripe_w = 64 * S
    for x in range(-H * 2, W + H * 2, stripe_w * 2):
        pts = [(x, H), (x + stripe_w, H), (x + stripe_w + H, 0), (x + H, 0)]
        d.polygon(pts, fill=blk)
    d.rectangle(_sc(460, 18, 1588, 162 - 18), fill=blk, outline=(235, 182, 22, 255), width=7 * S)
    f_lbl = _font(F_SANS_BOLD, 88)
    _draw_centered_text(d, (480, 22, 1568, 162 - 22), "FORKLIFT POCKET", f_lbl, (238, 188, 28, 255))
    d.rectangle([0, 0, W - 1, H - 1], outline=blk, width=10 * S)
    img = _apply_micro_grain(img, mode="metal_plate", seed=103, strength=0.75)
    img.save(os.path.join(OUT_DIR, "decal_forklift_chevron.png"))


def gen_m4_arch_placard():

    W, H = _sc(1024, 1536)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)


    d.rounded_rectangle(_sc(120, 30, 904, 280), radius=16 * S, fill=(184, 38, 66, 252), outline=(206, 202, 198, 238), width=8 * S)
    _draw_centered_text(d, (120, 36, 904, 274), "M - 4", _font(F_SANS_BOLD, 178), (216, 214, 210, 252))


    d.rounded_rectangle(_sc(40, 330, 984, 1080), radius=18 * S, fill=(24, 27, 32, 252), outline=(186, 192, 198, 245), width=10 * S)
    d.rectangle(_sc(66, 356, 958, 466), fill=(176, 182, 188, 248))
    _draw_centered_text(d, (88, 358, 936, 464), "USAF  ENGINE  CRADLE  M-4", _font(F_SANS_BOLD, 52), (18, 20, 24, 255))

    f_k = _font(F_MONO_BOLD, 42)
    f_v = _font(F_SANS_BOLD, 44)
    rows = [
        ("MODEL",   "F100-PW-220 / F-15"),
        ("PART NO", "72480-100  REV D"),
        ("SER NO",  "AF-84-0219"),
        ("MAX WT",  "6,500 LBS  (2,948 KG)"),
        ("INSP",    "NDT OK  24-06"),
    ]
    y_cur = 492
    for k, v in rows:
        d.line([_sc(66, y_cur + 72), _sc(958, y_cur + 72)], fill=(116, 124, 134, 205), width=4 * S)
        d.text(_sc(84, y_cur + 12), k, font=f_k, fill=(168, 176, 186, 245))
        d.text(_sc(340, y_cur + 10), v, font=f_v, fill=(214, 218, 224, 252))
        y_cur += 80

    d.rectangle(_sc(80, 920, 944, 1052), fill=(188, 192, 196, 250))
    _draw_barcode(d, 110, 936, 914, 1018)
    _draw_centered_text(d, (110, 1018, 914, 1050), "*72480-100-840219*", _font(F_MONO_BOLD, 28), (20, 22, 26, 252))
    _draw_corner_rivets(d, (40, 330, 984, 1080), inset=24, r=11)


    d.rounded_rectangle(_sc(40, 1130, 984, 1500), radius=18 * S, fill=(216, 168, 22, 252), outline=(22, 24, 28, 255), width=12 * S)
    d.rectangle(_sc(62, 1152, 962, 1262), fill=(22, 24, 28, 255))
    _draw_centered_text(d, (62, 1152, 962, 1262), "CAUTION", _font(F_SANS_BOLD, 82), (222, 174, 26, 255))
    _draw_centered_text(d, (66, 1282, 958, 1380), "LOCK ROLLOVER PINS BEFORE", _font(F_SANS_BOLD, 54), (20, 22, 24, 255))
    _draw_centered_text(d, (66, 1378, 958, 1474), "TOWING OR LIFTING ENGINE", _font(F_SANS_BOLD, 54), (20, 22, 24, 255))

    img = _apply_micro_grain(img, mode="metal_plate", seed=104, strength=0.85)
    img.save(os.path.join(OUT_DIR, "decal_m4_arch_placard.png"))


def gen_shroud_danger_blast():

    W, H = _sc(1600, 720)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    red = (214, 42, 24, 245)
    cream = (244, 238, 222, 245)
    dark = (22, 18, 15, 225)

    w_ln = 16 * S
    d.line([_sc(40, 40), _sc(1560, 40)], fill=red, width=w_ln)
    d.line([_sc(40, 680), _sc(1560, 680)], fill=red, width=w_ln)
    d.line([_sc(40, 40), _sc(40, 210)], fill=red, width=w_ln)
    d.line([_sc(40, 510), _sc(40, 680)], fill=red, width=w_ln)
    d.line([_sc(1560, 40), _sc(1560, 210)], fill=red, width=w_ln)
    d.line([_sc(1560, 510), _sc(1560, 680)], fill=red, width=w_ln)

    _draw_warning_triangle(d, 250, 375, 165, (28, 20, 14, 90), red, cream, stroke_w=22)
    d.text(_sc(468, 95), "DANGER", font=_font(F_SANS_BOLD, 185), fill=dark)
    d.text(_sc(465, 92), "DANGER", font=_font(F_SANS_BOLD, 185), fill=red)
    d.text(_sc(473, 317), "JET EXHAUST BLAST", font=_font(F_COND_BOLD, 102), fill=dark)
    d.text(_sc(470, 314), "JET EXHAUST BLAST", font=_font(F_COND_BOLD, 102), fill=cream)
    d.text(_sc(472, 470), "KEEP CLEAR 150 FT  •  NO STEP", font=_font(F_MONO_BOLD, 64), fill=dark)
    d.text(_sc(470, 468), "KEEP CLEAR 150 FT  •  NO STEP", font=_font(F_MONO_BOLD, 64), fill=(246, 214, 96, 248))
    img = _apply_micro_grain(img, mode="stencil", seed=105, strength=0.90)
    img.save(os.path.join(OUT_DIR, "decal_shroud_danger_blast.png"))


def gen_shroud_hot_station():

    W, H = _sc(1600, 580)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    red = (212, 40, 24, 248)
    cream = (242, 238, 224, 246)
    amber = (242, 188, 52, 244)

    d.rectangle(_sc(30, 30, 1570, 550), fill=(22, 20, 18, 220), outline=(156, 150, 138, 225), width=10 * S)
    d.rectangle(_sc(48, 48, 1552, 208), fill=red)
    _draw_centered_text(d, (88, 48, 1512, 208), "HOT SURFACE  •  DO NOT TOUCH", _font(F_SANS_BOLD, 78), (248, 246, 240, 252))
    _draw_centered_text(d, (50, 232, 1550, 372), "STA 212.0   TI-6AL-4V SHROUD", _font(F_COND_BOLD, 92), cream)
    d.line([_sc(96, 384), _sc(1504, 384)], fill=(122, 116, 106, 205), width=5 * S)
    _draw_centered_text(d, (50, 398, 1550, 522), "▲  DATUM INDEX 0°   P/N 407890", _font(F_MONO_BOLD, 62), amber)
    _draw_corner_rivets(d, (30, 30, 1570, 550), inset=32, r=14)
    img = _apply_micro_grain(img, mode="metal_plate", seed=106, strength=0.90)
    img.save(os.path.join(OUT_DIR, "decal_shroud_hot_station.png"))


def gen_belly_band_latch():

    W, H = _sc(1600, 480)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    wht = (228, 232, 236, 235)
    red = (205, 56, 38, 240)

    d.line([_sc(40, 42), _sc(1560, 42)], fill=wht, width=8 * S)
    d.line([_sc(40, 438), _sc(1560, 438)], fill=wht, width=8 * S)
    _draw_centered_text(d, (40, 62, 1560, 210), "◄  CLAMP BAND  P/N 407912  ►", _font(F_SANS_BOLD, 88), wht)
    _draw_centered_text(d, (40, 220, 1560, 330), "TORQUE 45 IN-LB  •  INSPECT SILICONE SEAL", _font(F_MONO_BOLD, 56), wht)
    _draw_centered_text(d, (40, 338, 1560, 426), "DO NOT PRY  •  ALIGN INDEX MARKS", _font(F_SANS_BOLD, 54), red)
    img = _apply_micro_grain(img, mode="stencil", seed=107, strength=0.85)
    img.save(os.path.join(OUT_DIR, "decal_belly_band_latch.png"))


def gen_intake_danger():

    W, H = _sc(1600, 600)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    red = (198, 32, 24, 246)
    wht = (240, 242, 244, 246)

    pts_arr = [_sc(45, 300), _sc(345, 70), _sc(345, 530)]
    d.polygon(pts_arr, fill=red)
    d.line(pts_arr + [pts_arr[0]], fill=wht, width=14 * S)
    d.rectangle(_sc(385, 70, 1555, 290), fill=red, outline=wht, width=10 * S)
    _draw_centered_text(d, (385, 74, 1555, 286), "DANGER", _font(F_SANS_BOLD, 168), wht)
    d.rectangle(_sc(385, 315, 1555, 530), fill=(24, 28, 34, 228), outline=red, width=10 * S)
    _draw_centered_text(d, (395, 325, 1545, 435), "JET INTAKE  •  SUCTION", _font(F_SANS_BOLD, 82), wht)
    _draw_centered_text(d, (395, 435, 1545, 520), "KEEP CLEAR 25 FT WHEN RUNNING", _font(F_MONO_BOLD, 50), (242, 198, 44, 248))
    img = _apply_micro_grain(img, mode="stencil", seed=108, strength=0.80)
    img.save(os.path.join(OUT_DIR, "decal_intake_danger.png"))


def gen_tec_firezone():

    W, H = _sc(1600, 720)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    wht = (232, 234, 236, 238)
    red = (204, 44, 28, 244)

    d.rectangle(_sc(30, 30, 1570, 690), fill=(20, 20, 22, 175), outline=wht, width=10 * S)
    d.rectangle(_sc(52, 52, 1548, 220), fill=red)
    _draw_centered_text(d, (92, 52, 1508, 220), "FIRE ZONE 3  •  HIGH TEMP", _font(F_SANS_BOLD, 92), (244, 242, 238, 250))
    _draw_centered_text(d, (52, 250, 1548, 400), "PRATT & WHITNEY  F100-PW-220", _font(F_COND_BOLD, 94), wht)
    d.line([_sc(80, 425), _sc(1520, 425)], fill=wht, width=6 * S)
    _draw_centered_text(d, (52, 445, 1548, 560), "STA 148.5   INCONEL 718 CASE", _font(F_MONO_BOLD, 66), wht)
    _draw_centered_text(d, (52, 568, 1548, 665), "▲ HOIST POINT AFT   MAX 4000 LBS", _font(F_SANS_BOLD, 56), (242, 192, 38, 246))
    _draw_corner_rivets(d, (30, 30, 1570, 690), inset=32, r=14)
    img = _apply_micro_grain(img, mode="metal_plate", seed=109, strength=0.85)
    img.save(os.path.join(OUT_DIR, "decal_tec_firezone.png"))


def gen_oiltank_spec():

    W, H = _sc(1400, 720)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    blk = (24, 26, 30, 238)
    yel = (232, 182, 24, 248)

    d.rectangle(_sc(28, 28, 1372, 692), outline=blk, width=12 * S)
    d.rectangle(_sc(48, 48, 1352, 218), fill=yel, outline=blk, width=8 * S)
    _draw_centered_text(d, (48, 48, 1352, 218), "OIL TANK  •  3.8 US GAL", _font(F_SANS_BOLD, 96), blk)
    _draw_centered_text(d, (48, 250, 1352, 380), "SPEC: MIL-PRF-7808 / -23699", _font(F_COND_BOLD, 82), blk)
    d.line([_sc(72, 405), _sc(1328, 405)], fill=blk, width=6 * S)
    _draw_centered_text(d, (48, 435, 1352, 545), "CHECK SIGHT GLASS WITHIN", _font(F_MONO_BOLD, 62), blk)
    _draw_centered_text(d, (48, 545, 1352, 655), "15 MIN AFTER ENGINE SHUTDOWN", _font(F_MONO_BOLD, 62), (176, 34, 24, 244))
    img = _apply_micro_grain(img, mode="stencil", seed=110, strength=0.85)
    img.save(os.path.join(OUT_DIR, "decal_oiltank_spec.png"))


def gen_trailer_usaf_spec():

    W, H = _sc(2048, 480)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    blk = (22, 22, 20, 242)

    d.line([_sc(36, 34), _sc(2012, 34)], fill=blk, width=12 * S)
    d.line([_sc(36, 446), _sc(2012, 446)], fill=blk, width=12 * S)
    _draw_centered_text(d, (40, 52, 2008, 238), "USAF  M3000  GSE  TRAILER", _font(F_SANS_BOLD, 134), blk)
    _draw_centered_text(d, (40, 255, 2008, 415), "MAX TOW 5 MPH   •   TIRE PRESS 90 PSI   •   CHOCK WHEELS", _font(F_MONO_BOLD, 62), blk)
    img = _apply_micro_grain(img, mode="stencil", seed=111, strength=0.95)
    img.save(os.path.join(OUT_DIR, "decal_trailer_usaf_spec.png"))


def gen_deec_caution():

    W, H = _sc(1024, 768)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    yel = (236, 190, 24, 255)
    blk = (18, 20, 24, 255)

    d.rounded_rectangle(_sc(24, 24, 1000, 744), radius=24 * S, fill=yel, outline=blk, width=18 * S)
    d.rectangle(_sc(58, 58, 966, 228), fill=blk)
    _draw_centered_text(d, (58, 58, 966, 228), "CAUTION", _font(F_SANS_BOLD, 128), yel)
    _draw_warning_triangle(d, 200, 470, 138, yel, blk, blk, stroke_w=18)
    d.text(_sc(374, 302), "ESD SENSITIVE", font=_font(F_SANS_BOLD, 66), fill=blk)
    d.text(_sc(374, 408), "DEEC-220 ECU", font=_font(F_COND_BOLD, 74), fill=blk)
    d.text(_sc(374, 525), "GROUND BEFORE", font=_font(F_MONO_BOLD, 52), fill=blk)
    d.text(_sc(374, 595), "DISCONNECTING", font=_font(F_MONO_BOLD, 52), fill=blk)
    _draw_corner_rivets(d, (24, 24, 1000, 744), inset=36, r=14)
    img = _apply_micro_grain(img, mode="metal_plate", seed=112, strength=0.75)
    img.save(os.path.join(OUT_DIR, "decal_deec_caution.png"))


def gen_micro_hyd_spec():

    W, H = _sc(1200, 480)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(_sc(16, 16, 1184, 464), radius=16 * S, fill=(24, 27, 32, 248), outline=(212, 218, 224, 240), width=8 * S)

    d.rectangle(_sc(34, 34, 1166, 98), fill=(26, 108, 206, 252))
    d.rectangle(_sc(410, 34, 790, 98), fill=(236, 188, 22, 252))
    _draw_centered_text(d, (410, 34, 790, 98), "HYDRAULIC", _font(F_SANS_BOLD, 48), (18, 20, 24, 255))
    _draw_centered_text(d, (48, 122, 1152, 256), "3000 PSI  •  MIL-PRF-83282", _font(F_SANS_BOLD, 74), (242, 244, 246, 252))
    d.line([_sc(56, 276), _sc(1144, 276)], fill=(198, 42, 32, 235), width=6 * S)
    _draw_centered_text(d, (48, 294, 1152, 376), "BLEED PRESSURE BEFORE DISCONNECT", _font(F_MONO_BOLD, 42), (240, 194, 36, 250))
    _draw_centered_text(d, (48, 382, 1152, 444), "ACTUATOR MANIFOLD  •  STA 188.5", _font(F_MONO_BOLD, 34), (176, 184, 194, 235))
    _draw_corner_rivets(d, (16, 16, 1184, 464), inset=24, r=10)
    img = _apply_micro_grain(img, mode="metal_plate", seed=113, strength=0.78)
    img.save(os.path.join(OUT_DIR, "decal_micro_hyd_spec.png"))


def gen_micro_pmel_cal():

    W, H = _sc(1024, 512)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.rounded_rectangle(_sc(20, 24, 620, 488), radius=14 * S, fill=(238, 240, 236, 252), outline=(26, 118, 64, 255), width=12 * S)
    d.rectangle(_sc(34, 38, 606, 162), fill=(24, 122, 62, 255))
    _draw_centered_text(d, (34, 40, 606, 160), "CALIBRATED", _font(F_SANS_BOLD, 74), (246, 248, 244, 255))
    _draw_centered_text(d, (42, 178, 598, 268), "USAF  PMEL  QA", _font(F_SANS_BOLD, 56), (22, 26, 24, 255))
    d.line([_sc(54, 286), _sc(586, 286)], fill=(24, 122, 62, 220), width=5 * S)
    _draw_centered_text(d, (42, 302, 598, 382), "BY: 482 MXS   OK", _font(F_MONO_BOLD, 44), (26, 30, 28, 248))
    _draw_centered_text(d, (42, 390, 598, 466), "DUE:  2026-11", _font(F_MONO_BOLD, 46), (192, 34, 26, 252))

    d.rounded_rectangle(_sc(654, 24, 1004, 488), radius=14 * S, fill=(236, 188, 20, 252), outline=(20, 22, 24, 255), width=12 * S)
    d.rectangle(_sc(668, 38, 990, 146), fill=(20, 22, 24, 255))
    _draw_centered_text(d, (668, 40, 990, 144), "GROUND", _font(F_SANS_BOLD, 62), (238, 190, 24, 255))

    cx, cy = 829, 265
    d.line([_sc(cx, 176), _sc(cx, 272)], fill=(20, 22, 24, 255), width=14 * S)
    d.line([_sc(cx - 88, 272), _sc(cx + 88, 272)], fill=(20, 22, 24, 255), width=14 * S)
    d.line([_sc(cx - 58, 312), _sc(cx + 58, 312)], fill=(20, 22, 24, 255), width=12 * S)
    d.line([_sc(cx - 28, 352), _sc(cx + 28, 352)], fill=(20, 22, 24, 255), width=12 * S)
    _draw_centered_text(d, (674, 386, 984, 464), "STATIC BOND", _font(F_COND_BOLD, 50), (20, 22, 24, 255))
    img = _apply_micro_grain(img, mode="metal_plate", seed=114, strength=0.68)
    img.save(os.path.join(OUT_DIR, "decal_micro_pmel_cal.png"))


def gen_micro_nostep_hatch():

    W, H = _sc(1600, 600)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hatch = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dh = ImageDraw.Draw(hatch)
    red = (206, 40, 24, 232)
    for x in range(-600, 2200, 88):
        dh.line([_sc(x, 24), _sc(x + 552, 576)], fill=(206, 40, 24, 165), width=10 * S)
    clip_m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(clip_m).rectangle(_sc(24, 24, 1576, 576), fill=255)
    img.paste(hatch, (0, 0), clip_m)
    d = ImageDraw.Draw(img)
    d.rectangle(_sc(24, 24, 1576, 576), outline=red, width=18 * S)
    d.rectangle(_sc(220, 128, 1380, 472), fill=(20, 18, 16, 155), outline=red, width=12 * S)
    _draw_centered_text(d, (240, 148, 1360, 344), "NO  STEP", _font(F_SANS_BOLD, 168), (222, 46, 28, 248))
    _draw_centered_text(d, (240, 352, 1360, 452), "DO NOT PUSH  •  THIN TI SKIN", _font(F_MONO_BOLD, 56), (242, 234, 218, 238))
    img = _apply_micro_grain(img, mode="stencil", seed=115, strength=0.95)
    img.save(os.path.join(OUT_DIR, "decal_micro_nostep_hatch.png"))


def gen_m4_hoist_beam():

    W, H = _sc(1800, 360)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    yel = (238, 190, 22, 250)
    blk = (18, 20, 24, 248)
    wht = (242, 244, 246, 250)

    d.rounded_rectangle(_sc(14, 14, 1786, 346), radius=14 * S, fill=blk, outline=yel, width=10 * S)

    chev = Image.new("RGBA", (W, H), blk)
    dc = ImageDraw.Draw(chev)
    for x in range(-400, 2200, 92):
        dc.polygon([_sc(x, 0), _sc(x + 46, 0), _sc(x + 46 - 360, 360), _sc(x - 360, 360)], fill=yel)
    m_lr = Image.new("L", (W, H), 0)
    dm = ImageDraw.Draw(m_lr)
    dm.rectangle(_sc(28, 28, 236, 332), fill=255)
    dm.rectangle(_sc(1564, 28, 1772, 332), fill=255)
    img.paste(chev, (0, 0), m_lr)
    d = ImageDraw.Draw(img)
    d.line([_sc(240, 26), _sc(240, 334)], fill=yel, width=6 * S)
    d.line([_sc(1560, 26), _sc(1560, 334)], fill=yel, width=6 * S)
    _draw_centered_text(d, (260, 42, 1540, 196), "HOIST POINT  •  SLING HERE", _font(F_SANS_BOLD, 78), yel)
    _draw_centered_text(d, (260, 212, 1540, 318), "MAX LIFT 6,500 LBS (2,948 KG)  •  LOCK PINS BEFORE LIFT", _font(F_MONO_BOLD, 37), wht)
    img = _apply_micro_grain(img, mode="metal_plate", seed=116, strength=0.82)
    img.save(os.path.join(OUT_DIR, "decal_m4_hoist_beam.png"))


def gen_stand_fwd_bumper():

    W, H = _sc(1920, 384)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bg  = (20, 24, 30, 248)
    rim = (196, 206, 218, 246)
    sky = (118, 194, 255, 252)
    wht = (242, 246, 250, 252)
    yel = (240, 194, 26, 250)
    blk = (16, 18, 22, 252)
    d.rounded_rectangle(_sc(16, 16, 1904, 368), radius=14 * S, fill=bg, outline=rim, width=10 * S)
    d.rectangle(_sc(34, 34, 1886, 350), outline=sky, width=4 * S)
    d.rectangle(_sc(40, 270, 1880, 342), fill=yel)
    _draw_centered_text(d, (50, 42, 1870, 174), "USAF  M-4 / TK-9  ENGINE STAND", _font(F_SANS_BOLD, 102), wht)
    _draw_centered_text(d, (50, 178, 1870, 258), "F100-PW-220 / -229   •   SER AF-84-0219   •   ASSY TARE 1,820 LBS", _font(F_MONO_BOLD, 44), sky)
    _draw_centered_text(d, (50, 274, 1870, 338), "MAX TOW 5 MPH   •   LOCK ROLLOVER PINS   •   CHOCK WHEELS", _font(F_SANS_BOLD, 44), blk)
    img = _apply_micro_grain(img, mode="metal_plate", seed=117, strength=0.68)
    img.save(os.path.join(OUT_DIR, "decal_stand_fwd_bumper.png"))


def gen_shroud_aft_index():

    W, H = _sc(1600, 320)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    wht = (238, 234, 222, 232)
    yel = (234, 192, 34, 238)
    red = (206, 44, 28, 236)

    d.line([_sc(40, 252), _sc(1560, 252)], fill=wht, width=6 * S)
    for i, tx in enumerate(range(80, 1540, 80)):
        th = 46 if (i % 3 == 0) else 24
        col = yel if (i % 3 == 0) else wht
        d.line([_sc(tx, 252), _sc(tx, 252 + th)], fill=col, width=7 * S)

    d.polygon([_sc(800, 212), _sc(768, 252), _sc(832, 252)], fill=red)
    _draw_centered_text(d, (60, 30, 1540, 134), "STA 218.5  •  MASTER FLAP INDEX", _font(F_SANS_BOLD, 74), wht)
    _draw_centered_text(d, (60, 134, 1540, 210), "LOCKWIRE CHECK  •  MATCH MARK 16-PETAL HINGE", _font(F_MONO_BOLD, 42), yel)
    img = _apply_micro_grain(img, mode="stencil", seed=118, strength=0.98)
    img.save(os.path.join(OUT_DIR, "decal_shroud_aft_index.png"))


def gen_spinner_balance_mark():

    W, H = _sc(1024, 512)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    wht = (236, 238, 240, 232)
    red = (208, 42, 28, 236)

    d.rectangle(_sc(46, 56, 978, 126), fill=wht)
    d.rectangle(_sc(478, 126, 546, 212), fill=wht)
    d.line([_sc(120, 232), _sc(904, 232)], fill=wht, width=8 * S)
    _draw_centered_text(d, (60, 254, 964, 344), "BALANCE INDEX  •  0°", _font(F_SANS_BOLD, 66), wht)
    _draw_centered_text(d, (60, 356, 964, 446), "NO PUSH  •  TORQUE 45 FT-LB", _font(F_MONO_BOLD, 44), red)
    img = _apply_micro_grain(img, mode="stencil", seed=119, strength=0.90)
    img.save(os.path.join(OUT_DIR, "decal_spinner_balance_mark.png"))


def gen_shroud_nopush_plate():

    W, H = _sc(1300, 1000)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    red = (214, 44, 26, 244)
    red_dim = (204, 58, 36, 224)


    m_x, m_y = 48, 44
    notch_x, notch_y = 148, 128
    bw = 12 * S
    d.line([_sc(m_x + notch_x, m_y), _sc(1300 - m_x - notch_x, m_y)], fill=red, width=bw)
    d.line([_sc(m_x + notch_x, 1000 - m_y), _sc(1300 - m_x - notch_x, 1000 - m_y)], fill=red, width=bw)
    d.line([_sc(m_x, m_y + notch_y), _sc(m_x, 1000 - m_y - notch_y)], fill=red, width=bw)
    d.line([_sc(1300 - m_x, m_y + notch_y), _sc(1300 - m_x, 1000 - m_y - notch_y)], fill=red, width=bw)
    for sx_c, x_e in ((1, m_x), (-1, 1300 - m_x)):
        for sy_c, y_e in ((1, m_y), (-1, 1000 - m_y)):
            x_in = x_e + sx_c * notch_x
            y_in = y_e + sy_c * notch_y
            d.line([_sc(x_in, y_e), _sc(x_in, y_in), _sc(x_e, y_in)], fill=red, width=bw, joint="curve")


    _draw_centered_text(d, (160, 150, 1140, 428), "NO PUSH", _font(F_SANS_BOLD, 184), red)

    d.line([_sc(142, 474), _sc(1158, 474)], fill=red_dim, width=8 * S)
    for cx_d in (142, 1158):
        pts_d = [_sc(cx_d, 456), _sc(cx_d + 18, 474), _sc(cx_d, 492), _sc(cx_d - 18, 474)]
        d.polygon(pts_d, fill=red)

    _draw_centered_text(d, (125, 514, 1175, 764), "DO NOT STEP", _font(F_SANS_BOLD, 136), red)
    _draw_centered_text(d, (180, 816, 1120, 908), "TI-6AL-4V SHROUD  •  MAX LOAD 0 LB", _font(F_MONO_BOLD, 40), red_dim)

    img = _apply_micro_grain(img, mode="stencil", seed=120, strength=0.88)
    img.save(os.path.join(OUT_DIR, "decal_shroud_nopush_plate.png"))


def gen_actuator_dual_band():

    W, H = _sc(1600, 640)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    ink_y = (16, 18, 22, 248)
    ink_s = (14, 16, 20, 248)
    ink_sub = (26, 30, 36, 236)


    lx0, ly0, lx1, ly1 = 22, 116, 538, 524
    lnx, lny = 62, 54
    d.line([_sc(lx0 + lnx, ly0), _sc(lx1 - lnx, ly0)], fill=ink_y, width=6 * S)
    d.line([_sc(lx0 + lnx, ly1), _sc(lx1 - lnx, ly1)], fill=ink_y, width=6 * S)
    d.line([_sc(lx0, ly0 + lny), _sc(lx0, ly1 - lny)], fill=ink_y, width=6 * S)
    d.line([_sc(lx1, ly0 + lny), _sc(lx1, ly1 - lny)], fill=ink_y, width=6 * S)
    _draw_centered_text(d, (56, 152, 504, 272), "T.S. 24-6", _font(F_SANS_BOLD, 84), ink_y)
    d.line([_sc(66, 294), _sc(494, 294)], fill=ink_y, width=5 * S)
    _draw_centered_text(d, (56, 314, 504, 424), "INSP OK", _font(F_SANS_BOLD, 78), ink_y)
    _draw_centered_text(d, (56, 444, 504, 504), "QA STAMP 04", _font(F_MONO_BOLD, 36), ink_sub)


    rx0, ry0, rx1, ry1 = 664, 42, 1576, 598
    rnx, rny = 72, 62
    bw_r = 7 * S
    d.line([_sc(rx0 + rnx, ry0), _sc(rx1 - rnx, ry0)], fill=ink_s, width=bw_r)
    d.line([_sc(rx0 + rnx, ry1), _sc(rx1 - rnx, ry1)], fill=ink_s, width=bw_r)
    d.line([_sc(rx0, ry0 + rny), _sc(rx0, ry1 - rny)], fill=ink_s, width=bw_r)
    d.line([_sc(rx1, ry0 + rny), _sc(rx1, ry1 - rny)], fill=ink_s, width=bw_r)
    for sx_c, x_e in ((1, rx0), (-1, rx1)):
        for sy_c, y_e in ((1, ry0), (-1, ry1)):
            d.line([_sc(x_e + sx_c * rnx, y_e), _sc(x_e + sx_c * rnx, y_e + sy_c * rny), _sc(x_e, y_e + sy_c * rny)],
                   fill=ink_s, width=5 * S, joint="curve")

    _draw_centered_text(d, (756, 98, 1484, 214), "CONVERGENT", _font(F_SANS_BOLD, 92), ink_s)
    _draw_centered_text(d, (756, 222, 1484, 338), "ACTUATOR", _font(F_SANS_BOLD, 92), ink_s)
    d.line([_sc(732, 368), _sc(1508, 368)], fill=ink_s, width=6 * S)
    _draw_centered_text(d, (720, 392, 1520, 476), "P/N 4048220-04  •  3000 PSI", _font(F_MONO_BOLD, 46), ink_s)
    _draw_centered_text(d, (740, 496, 1500, 562), "PRATT & WHITNEY  •  F100-PW-220", _font(F_MONO_BOLD, 34), ink_sub)

    img = _apply_micro_grain(img, mode="metal_plate", seed=121, strength=0.72)
    img.save(os.path.join(OUT_DIR, "decal_actuator_dual_band.png"))


def gen_rbf_streamer():

    W, H = 2400, 520
    img = Image.new("RGBA", (W, H), (178, 22, 18, 255))
    arr = np.asarray(img, dtype=np.float32)


    yy, xx = np.mgrid[0:H, 0:W]
    twill = np.sin((xx + yy) * 0.58) * 0.5 + np.sin((xx - yy) * 1.15) * 0.18
    weft = np.sin(xx * 0.42) * 0.15
    rng = np.random.default_rng(1422)
    grain = rng.normal(0.0, 4.2, (H, W)).astype(np.float32)

    tail_wear = np.clip((xx - 1450.0) / 950.0, 0.0, 1.0) * 14.0
    edge_v = np.minimum(yy, H - 1 - yy)
    edge_dark = np.clip(1.0 - edge_v / 36.0, 0.0, 1.0) * 26.0

    mod = twill * 8.5 + weft * 4.5 + grain - edge_dark
    arr[:, :, 0] = np.clip(arr[:, :, 0] + mod - tail_wear * 0.55, 65, 225)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + mod * 0.35 + tail_wear * 0.15, 8, 65)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + mod * 0.30 + tail_wear * 0.10, 8, 55)
    img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    d = ImageDraw.Draw(img)


    d.rectangle((0, 0, W - 1, H - 1), outline=(92, 12, 10, 255), width=18)
    for sx in range(28, W - 28, 30):
        d.line([(sx, 26), (sx + 16, 26)], fill=(230, 212, 204, 155), width=4)
        d.line([(sx, H - 27), (sx + 16, H - 27)], fill=(230, 212, 204, 155), width=4)


    d.rectangle((18, 18, 290, H - 19), fill=(154, 18, 15, 255), outline=(96, 12, 10, 255), width=8)
    for sy in range(44, H - 44, 28):
        d.line([(268, sy), (268, sy + 14)], fill=(236, 224, 216, 180), width=4)
        d.line([(244, sy), (244, sy + 14)], fill=(236, 224, 216, 165), width=4)


    f_rbf = _font(F_SANS_BOLD, 66)
    txt = "REMOVE BEFORE FLIGHT"
    bb = d.textbbox((0, 0), txt, font=f_rbf)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx = 320 + (W - 365 - tw) // 2 - bb[0]
    ty = 64 + (H - 160 - th) // 2 - bb[1]

    d.text((tx + 4, ty + 4), txt, font=f_rbf, fill=(72, 10, 8, 140))
    d.text((tx, ty), txt, font=f_rbf, fill=(244, 240, 232, 250))


    f_sub = _font(F_MONO_BOLD, 22)
    d.text((348, H - 84), "NAS1756-12  •  USAF F100-PW-220", font=f_sub, fill=(240, 224, 216, 175))


    arr2 = np.asarray(img, dtype=np.float32)
    arr2[:, :, :3] = np.clip(arr2[:, :, :3] + twill[:, :, None] * 6.0, 0, 255)
    img = Image.fromarray(arr2.astype(np.uint8), mode="RGBA")
    img.save(os.path.join(OUT_DIR, "decal_rbf_streamer.png"))


def gen_nozzle_pn407808_nostep():

    W, H = 2400, 1000
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)


    ink_dark = (12, 14, 18, 254)
    ink_sub  = (20, 22, 26, 246)

    d.rounded_rectangle((42, 52, W - 42, 448), radius=18, outline=ink_dark, width=16)
    d.rounded_rectangle((68, 76, W - 68, 424), radius=10, outline=ink_sub, width=6)

    f_pn = _font(F_SANS_BOLD, 96)
    bb = d.textbbox((0, 0), "P/N 407808-02", font=f_pn, stroke_width=3)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((W - tw) // 2 - bb[0], 98 - bb[1]), "P/N 407808-02", font=f_pn, fill=ink_dark, stroke_width=3, stroke_fill=ink_dark)
    d.line([(124, 304), (W - 124, 304)], fill=ink_dark, width=8)
    f_pnsub = _font(F_MONO_BOLD, 42)
    sub1 = "TI-6AL-4V  •  BALANCE FLAP ASSY  •  PWA"
    bb2 = d.textbbox((0, 0), sub1, font=f_pnsub, stroke_width=1)
    d.text(((W - (bb2[2] - bb2[0])) // 2 - bb2[0], 328 - bb2[1]), sub1, font=f_pnsub, fill=ink_sub, stroke_width=1, stroke_fill=ink_sub)


    red_w = (176, 26, 18, 252)
    blk_w = (12, 14, 18, 254)

    d.rectangle((176, 554, W - 176, 946), outline=red_w, width=16)
    d.line([(206, 582), (W - 206, 582)], fill=blk_w, width=7)
    d.line([(206, 918), (W - 206, 918)], fill=blk_w, width=7)
    f_ns = _font(F_SANS_BOLD, 104)
    bb3 = d.textbbox((0, 0), "NO  STEP", font=f_ns, stroke_width=3)
    d.text(((W - (bb3[2] - bb3[0])) // 2 - bb3[0], 604 - bb3[1]), "NO  STEP", font=f_ns, fill=red_w, stroke_width=3, stroke_fill=red_w)
    f_nssub = _font(F_MONO_BOLD, 40)
    sub2 = "DO NOT PUSH  •  ACTUATOR STRUT  •  TI"
    bb4 = d.textbbox((0, 0), sub2, font=f_nssub, stroke_width=1)
    d.text(((W - (bb4[2] - bb4[0])) // 2 - bb4[0], 812 - bb4[1]), sub2, font=f_nssub, fill=blk_w, stroke_width=1, stroke_fill=blk_w)

    img = _apply_micro_grain(img, mode="stencil", seed=1423, strength=0.68)
    img.save(os.path.join(OUT_DIR, "decal_nozzle_pn407808_nostep.png"))


def gen_fluid_line_bands():

    W, H = 2048, 1024
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)


    bands = [

        (10,  246, (192, 28, 22, 252),   None,                  (246, 244, 240, 255), "FUEL  >  JP-8  >"),
        (266, 502, (24, 96, 196, 252),   (234, 186, 20, 252),   (16, 18, 22, 255),    "HYD 3000  >  OIL >"),
        (522, 758, (232, 182, 18, 252),  None,                  (18, 20, 24, 255),    "LUBE OIL  >  7808 >"),
        (778, 1014,(214, 220, 226, 252), None,                  (22, 26, 32, 255),    "BLEED AIR  >  HOT >"),
    ]
    f_bd = _font(F_SANS_BOLD, 42)

    for y0, y1, bg_col, ctr_col, txt_col, label in bands:
        bh = y1 - y0
        d.rectangle((0, y0, W, y1), fill=bg_col)

        d.rectangle((0, y0, W, y0 + 16), fill=(22, 24, 28, 235))
        d.rectangle((0, y1 - 16, W, y1), fill=(22, 24, 28, 235))
        if ctr_col is not None:

            cy0 = y0 + int(bh * 0.24)
            cy1 = y1 - int(bh * 0.24)
            d.rectangle((0, cy0, W, cy1), fill=ctr_col)
            d.line([(0, cy0), (W, cy0)], fill=(18, 22, 28, 220), width=4)
            d.line([(0, cy1), (W, cy1)], fill=(18, 22, 28, 220), width=4)


        rep_txt = f"  {label}     {label}  "
        bb = d.textbbox((0, 0), rep_txt, font=f_bd)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        tx = (W - tw) // 2 - bb[0]
        ty = y0 + (bh - th) // 2 - bb[1]
        d.text((tx, ty), rep_txt, font=f_bd, fill=txt_col)

    img = _apply_micro_grain(img, mode="metal_plate", seed=1424, strength=0.62)
    img.save(os.path.join(OUT_DIR, "decal_fluid_line_bands.png"))


def gen_oilsight_graduations():

    W, H = 280, 1600
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    wht = (244, 246, 248, 246)
    wht_dim = (230, 235, 240, 212)
    red_ok = (224, 48, 30, 246)


    hoop_vs = (0.125, 0.3125, 0.500, 0.6875, 0.875)
    windows = [
        (0.015, 0.090),
        (0.160, 0.278),
        (0.348, 0.465),
        (0.535, 0.602),
        (0.722, 0.840),
        (0.910, 0.985),
    ]
    for va, vb in windows:
        ya = int(round((1.0 - vb) * H))
        yb = int(round((1.0 - va) * H))
        d.line([(36, ya), (36, yb)], fill=wht_dim, width=4)
        d.line([(W - 36, ya), (W - 36, yb)], fill=wht_dim, width=4)


    maj_specs = [
        ("ADD",  0.052, red_ok, 19),
        ("1",    0.219, wht,    28),
        ("2",    0.406, wht,    28),
        ("3",    0.566, wht,    26),
        ("4",    0.781, wht,    28),
        ("FULL", 0.948, wht,    18),
    ]
    for lbl, vf, col_tk, fsz in maj_specs:
        yc = int(round((1.0 - vf) * H))

        d.line([(16, yc), (68, yc)], fill=col_tk, width=6)
        d.line([(W - 68, yc), (W - 16, yc)], fill=col_tk, width=6)

        fnt_use = _font(F_SANS_BOLD, fsz)
        bb = d.textbbox((0, 0), lbl, font=fnt_use, stroke_width=1)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        tx = (W - tw) // 2 - bb[0]
        ty = yc - th // 2 - bb[1]

        d.text((tx, ty), lbl, font=fnt_use, fill=col_tk, stroke_width=1, stroke_fill=(24, 20, 16, 175))


    for v_sub in (0.178, 0.260, 0.365, 0.448, 0.740, 0.822):
        ys = int(round((1.0 - v_sub) * H))
        d.line([(24, ys), (56, ys)], fill=wht_dim, width=4)
        d.line([(W - 56, ys), (W - 24, ys)], fill=wht_dim, width=4)

    img = _apply_micro_grain(img, mode="stencil", seed=1425, strength=0.40)
    img.save(os.path.join(OUT_DIR, "decal_oilsight_graduations.png"))


def gen_m4_roll_lock_plate():

    W, H = 2048, 460
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)


    bg_dk = (18, 20, 25, 246)
    wht = (232, 236, 240, 248)
    wht_sub = (210, 216, 222, 232)
    yel = (234, 188, 38, 246)

    d.rounded_rectangle((170, 28, W - 170, H - 28), radius=14, fill=bg_dk, outline=wht, width=10)
    d.rounded_rectangle((190, 46, W - 190, H - 46), radius=8, outline=wht_sub, width=4)
    d.line([(210, 248), (W - 210, 248)], fill=wht_sub, width=5)

    f_main = _font(F_SANS_BOLD, 66)
    t1 = "USAF M-4 ROLL LOCK"
    bb1 = d.textbbox((0, 0), t1, font=f_main)
    d.text(((W - (bb1[2] - bb1[0])) // 2 - bb1[0], 72 - bb1[1]), t1, font=f_main, fill=wht)

    f_sub = _font(F_MONO_BOLD, 46)
    t2 = "MAX 15 DEG  •  PIN BEFORE ROLL"
    bb2 = d.textbbox((0, 0), t2, font=f_sub)
    d.text(((W - (bb2[2] - bb2[0])) // 2 - bb2[0], 278 - bb2[1]), t2, font=f_sub, fill=yel)

    img = _apply_micro_grain(img, mode="metal_plate", seed=1526, strength=0.55)
    img.save(os.path.join(OUT_DIR, "decal_m4_roll_lock.png"))


def gen_trailer_dataplate_dual():

    W, H = 2400, 1200
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    bg_dk = (18, 20, 25, 246)
    wht = (234, 238, 242, 248)
    sub = (212, 218, 224, 232)
    yel = (234, 188, 38, 246)


    d.rounded_rectangle((110, 62, W - 110, 538), radius=16, fill=bg_dk, outline=wht, width=12)
    d.rounded_rectangle((134, 84, W - 134, 516), radius=10, outline=sub, width=4)
    d.line([(160, 314), (W - 160, 314)], fill=sub, width=6)
    f_h1 = _font(F_SANS_BOLD, 82)
    t_a1 = "USAF  M3000  TRAILER"
    bb = d.textbbox((0, 0), t_a1, font=f_h1)
    d.text(((W - (bb[2] - bb[0])) // 2 - bb[0], 116 - bb[1]), t_a1, font=f_h1, fill=wht)
    f_s1 = _font(F_MONO_BOLD, 48)
    t_a2 = "4W STEERABLE  •  P/N 72480-220  •  F100"
    bb2 = d.textbbox((0, 0), t_a2, font=f_s1)
    d.text(((W - (bb2[2] - bb2[0])) // 2 - bb2[0], 356 - bb2[1]), t_a2, font=f_s1, fill=yel)


    d.rounded_rectangle((110, 662, W - 110, 1138), radius=16, fill=bg_dk, outline=wht, width=12)
    d.rounded_rectangle((134, 684, W - 134, 1116), radius=10, outline=sub, width=4)
    d.line([(160, 914), (W - 160, 914)], fill=sub, width=6)
    t_b1 = "MAX TOW 15 MPH"
    bb3 = d.textbbox((0, 0), t_b1, font=f_h1)
    d.text(((W - (bb3[2] - bb3[0])) // 2 - bb3[0], 716 - bb3[1]), t_b1, font=f_h1, fill=yel)
    t_b2 = "TIRE PRESS 85 PSI  •  CHOCK WHEELS"
    bb4 = d.textbbox((0, 0), t_b2, font=f_s1)
    d.text(((W - (bb4[2] - bb4[0])) // 2 - bb4[0], 956 - bb4[1]), t_b2, font=f_s1, fill=wht)

    img = _apply_micro_grain(img, mode="metal_plate", seed=1527, strength=0.55)
    img.save(os.path.join(OUT_DIR, "decal_trailer_dataplate_dual.png"))


def gen_stand_plates_4k_atlas():

    W, H = 4096, 2400
    CW, CH = 2048, 480
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    bg_dk = (18, 20, 25, 246)
    wht = (234, 238, 242, 248)
    sub = (210, 216, 222, 232)
    yel = (234, 186, 36, 248)
    red = (216, 50, 38, 248)
    blk = (18, 20, 24, 250)

    def _draw_cell(col, row, title, subtitle, *, accent="white", banner=None, pad_x=135):
        x0 = col * CW
        y0 = row * CH
        x1 = x0 + CW
        y1 = y0 + CH

        bcol = yel if accent == "yellow" else (red if accent == "red" else wht)
        d.rounded_rectangle((x0 + pad_x, y0 + 26, x1 - pad_x, y1 - 26), radius=16, fill=bg_dk, outline=bcol, width=11)
        d.rounded_rectangle((x0 + pad_x + 22, y0 + 46, x1 - pad_x - 22, y1 - 46), radius=10, outline=sub, width=4)

        max_tw = CW - 2 * (pad_x + 78)

        def _fit_font(fpath, start_sz, text_s):
            sz = start_sz
            while sz > 20:
                f_try = _font(fpath, sz)
                bb_try = d.textbbox((0, 0), text_s, font=f_try)
                if (bb_try[2] - bb_try[0]) <= max_tw:
                    return f_try, bb_try
                sz -= 3
            f_try = _font(fpath, 20)
            return f_try, d.textbbox((0, 0), text_s, font=f_try)

        if banner is not None:

            band_col = yel if banner == "yellow" else red
            d.rounded_rectangle((x0 + pad_x + 28, y0 + 52, x1 - pad_x - 28, y0 + 244), radius=8, fill=band_col)
            f_h, bb = _fit_font(F_SANS_BOLD, 64, title)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            tx = x0 + (CW - tw) // 2 - bb[0]
            ty = y0 + 148 - th // 2 - bb[1]
            tcol = blk if banner == "yellow" else wht
            d.text((tx, ty), title, font=f_h, fill=tcol)

            f_s, bb2 = _fit_font(F_MONO_BOLD, 34, subtitle)
            tw2, th2 = bb2[2] - bb2[0], bb2[3] - bb2[1]
            tx2 = x0 + (CW - tw2) // 2 - bb2[0]
            ty2 = y0 + 340 - th2 // 2 - bb2[1]
            d.text((tx2, ty2), subtitle, font=f_s, fill=wht)
        else:
            d.line([(x0 + pad_x + 40, y0 + 258), (x1 - pad_x - 40, y0 + 258)], fill=sub, width=5)
            f_h, bb = _fit_font(F_SANS_BOLD, 62, title)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            tx = x0 + (CW - tw) // 2 - bb[0]
            ty = y0 + 150 - th // 2 - bb[1]
            d.text((tx, ty), title, font=f_h, fill=wht)

            f_s, bb2 = _fit_font(F_MONO_BOLD, 34, subtitle)
            tw2, th2 = bb2[2] - bb2[0], bb2[3] - bb2[1]
            tx2 = x0 + (CW - tw2) // 2 - bb2[0]
            ty2 = y0 + 344 - th2 // 2 - bb2[1]
            d.text((tx2, ty2), subtitle, font=f_s, fill=yel if accent == "yellow" else sub)


    _draw_cell(0, 0, "CAUTION  •  STAND LOCK",    "ENGAGE RAIL LOCK PINS BEFORE ENGINE ROLL", accent="yellow", banner="yellow")
    _draw_cell(1, 0, "FOD CONTROL ZONE",          "INSPECT RAIL & FASTENERS BEFORE MOVE",     accent="red",    banner="red")

    _draw_cell(0, 1, "USAF M-4  •  TK-9 RAIL",    "NSN 4920-01-084-2190  •  F100-PW-220",     accent="yellow")
    _draw_cell(1, 1, "MAX LOAD  6,500 LBS",       "CRADLE TARE 1,420 LBS  •  PROOF 9,750 LBS", accent="yellow")

    _draw_cell(0, 2, "P/N 72480-220  •  F100",    "PRATT & WHITNEY GSE  •  CAGE 77445",       accent="white")
    _draw_cell(1, 2, "USAF M3000 GSE TRAILER",    "4W STEERABLE  •  SER 84-M3000-219",        accent="yellow")

    _draw_cell(0, 3, "REAR STEER TOW LOCK",       "PULL & ROTATE T-PIN 90°  •  LOCK FOR TOW", accent="yellow", banner="yellow")
    _draw_cell(1, 3, "TOW BAR  •  MAX 15 MPH",    "SURGE BRAKE  •  ATTACH SAFETY CHAINS",     accent="yellow")

    _draw_cell(0, 4, "M-4 CRADLE  •  F100-PW",    "SER 84-219  •  USAF ENGINE STAND",         accent="white")
    _draw_cell(1, 4, "WARNING  •  PIN ROLLOVER",  "VERIFY LOCK PINS & BRAKE BEFORE ROLL",     accent="red",    banner="red")

    img = _apply_micro_grain(img, mode="metal_plate", seed=1528, strength=0.52)
    img.save(os.path.join(OUT_DIR, "decal_stand_plates_4k_atlas.png"))


def gen_fodbox_techdoc_atlas():

    W, H = _sc(1024, 512)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bg_dk = (18, 20, 25, 248)
    wht   = (240, 244, 248, 252)
    yel   = (240, 192, 24, 252)
    blu   = (38, 108, 196, 252)
    blk   = (14, 16, 20, 252)
    rim   = (204, 212, 222, 248)


    d.rounded_rectangle(_sc(10, 10, 1014, 246), radius=10 * S, fill=bg_dk, outline=yel, width=6 * S)
    d.rectangle(_sc(22, 22, 1002, 92), fill=yel)
    _draw_centered_text(d, (28, 26, 996, 88), "FOD CONTROL  •  PINS & HARDWARE", _font(F_SANS_BOLD, 44), blk)
    _draw_centered_text(d, (28, 102, 996, 166), "STOW ALL LOCK PINS, SHACKLES & FASTENERS", _font(F_MONO_BOLD, 29), wht)
    _draw_centered_text(d, (28, 174, 996, 232), "LATCH & PADLOCK BEFORE TOW  •  T.O. 2J-F100-3", _font(F_MONO_BOLD, 25), yel)
    for rx, ry in ((34, 34), (990, 34), (34, 226), (990, 226)):
        _draw_rivet_head(d, rx, ry, r=9)


    d.rounded_rectangle(_sc(10, 266, 1014, 502), radius=10 * S, fill=bg_dk, outline=rim, width=6 * S)
    d.rectangle(_sc(22, 278, 1002, 348), fill=blu)
    _draw_centered_text(d, (28, 282, 996, 344), "TECH DATA  •  T.O. 2J-F100-3", _font(F_SANS_BOLD, 46), wht)
    _draw_centered_text(d, (28, 358, 996, 422), "F100-PW-220 ENGINE LOGBOOK & RIGGING CARDS", _font(F_MONO_BOLD, 28), wht)
    _draw_centered_text(d, (28, 430, 996, 488), "WATERPROOF CANISTER  •  KEEP CAP HAND-TIGHT", _font(F_MONO_BOLD, 25), yel)
    for rx, ry in ((34, 290), (990, 290), (34, 482), (990, 482)):
        _draw_rivet_head(d, rx, ry, r=9)

    img = _apply_micro_grain(img, mode="metal_plate", seed=1829, strength=0.58)
    img.save(os.path.join(OUT_DIR, "decal_fodbox_techdoc_atlas.png"))


F_HAND_BOLD = ["segoeprb.ttf", "segoescb.ttf", "Inkfree.ttf", "comicbd.ttf"]
F_HAND_PEN  = ["segoepr.ttf", "Inkfree.ttf", "segoesc.ttf", "arial.ttf"]


def _draw_hand_stroke_poly(draw, pts, fill, width_px, seed=101):

    rng = np.random.default_rng(seed)
    jpts = [(x + float(rng.normal(0.0, 1.4 * S)), y + float(rng.normal(0.0, 1.4 * S))) for (x, y) in pts]
    draw.line(jpts, fill=fill, width=int(round(width_px * S)), joint="curve")
    r_cap = max(2, int(round(width_px * S * 0.48)))
    for px, py in (jpts[0], jpts[-1]):
        draw.ellipse([px - r_cap, py - r_cap, px + r_cap, py + r_cap], fill=fill)


def _apply_paintmarker_texture(img: Image.Image, seed: int = 2101, dry_brush: float = 0.28) -> Image.Image:

    arr = np.asarray(img, dtype=np.float32).copy()
    H, W, _ = arr.shape
    a = arr[:, :, 3]
    mask = a > 12.0
    if not np.any(mask):
        return img
    rng = np.random.default_rng(seed)

    n_lo = np.asarray(
        Image.fromarray(rng.uniform(0.0, 255.0, (max(8, H // 24), max(8, W // 24))).astype(np.uint8))
        .resize((W, H), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    n_hi = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)

    alpha_mod = np.clip(1.0 - dry_brush * (0.55 * (1.0 - n_lo) + 0.22 * np.clip(n_hi, 0.0, 2.5)), 0.42, 1.0)
    a_new = np.where(mask, a * alpha_mod, a)
    lum_mod = np.clip(0.94 + 0.12 * n_lo + 0.03 * n_hi, 0.78, 1.08)
    for c in range(3):
        arr[:, :, c] = np.where(mask, np.clip(arr[:, :, c] * lum_mod, 0.0, 255.0), arr[:, :, c])
    arr[:, :, 3] = np.clip(a_new, 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def gen_paintmarker_ndt_atlas():

    W, H = 1024, 1024
    img = Image.new("RGBA", _sc(W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    wht = (234, 232, 224, 236)
    yel = (238, 202, 46, 238)
    f_big = _font(F_HAND_BOLD, 62)
    f_med = _font(F_HAND_BOLD, 48)
    f_sm  = _font(F_HAND_BOLD, 38)


    tl = Image.new("RGBA", _sc(512, 512), (0, 0, 0, 0))
    dtl = ImageDraw.Draw(tl)

    ring_pts = []
    for i in range(44):
        th = 2.0 * math.pi * (i / 40.0) - 0.35
        rx = (182.0 + 7.5 * math.sin(th * 3.0)) * S
        ry = (132.0 + 6.0 * math.cos(th * 2.0)) * S
        ring_pts.append((256 * S + rx * math.cos(th), 258 * S + ry * math.sin(th)))
    _draw_hand_stroke_poly(dtl, ring_pts, wht, 8.5, seed=2101)
    _draw_centered_text(dtl, (88, 165, 420, 252), "NDT OK", f_big, wht)
    _draw_centered_text(dtl, (105, 258, 365, 336), "04-26", f_med, wht)

    _draw_hand_stroke_poly(dtl, [(342 * S, 294 * S), (366 * S, 326 * S), (414 * S, 260 * S)], wht, 9.5, seed=2102)
    tl = tl.rotate(5.5, resample=Image.BICUBIC, expand=False)
    img.alpha_composite(tl, (0, 0))


    tr = Image.new("RGBA", _sc(512, 512), (0, 0, 0, 0))
    dtr = ImageDraw.Draw(tr)
    _draw_centered_text(dtr, (44, 118, 468, 206), "TORQ OK", f_big, yel)
    _draw_hand_stroke_poly(dtr, [(84 * S, 212 * S), (255 * S, 216 * S), (424 * S, 208 * S)], yel, 8.0, seed=2103)
    _draw_centered_text(dtr, (48, 232, 464, 312), "B-SCOP PASS", f_sm, yel)

    _draw_hand_stroke_poly(dtr, [(124 * S, 364 * S), (262 * S, 360 * S), (382 * S, 352 * S)], yel, 8.5, seed=2104)
    _draw_hand_stroke_poly(dtr, [(342 * S, 326 * S), (386 * S, 352 * S), (344 * S, 384 * S)], yel, 8.5, seed=2105)
    tr = tr.rotate(-4.5, resample=Image.BICUBIC, expand=False)
    img.alpha_composite(tr, _sc(512, 0))


    bl = Image.new("RGBA", _sc(512, 512), (0, 0, 0, 0))
    dbl = ImageDraw.Draw(bl)
    for seg in range(12):
        pts_d = []
        a0 = 2.0 * math.pi * (seg / 12.0)
        a1 = a0 + 2.0 * math.pi * 0.052
        for k in range(5):
            th = a0 + (a1 - a0) * (k / 4.0)
            pts_d.append((256 * S + 188 * S * math.cos(th), 256 * S + 132 * S * math.sin(th)))
        _draw_hand_stroke_poly(dbl, pts_d, yel, 8.0, seed=2110 + seg)
    _draw_centered_text(dbl, (78, 166, 434, 248), "HOT CHK", f_big, yel)
    _draw_centered_text(dbl, (86, 256, 426, 332), "INSP 148", f_med, yel)
    bl = bl.rotate(3.5, resample=Image.BICUBIC, expand=False)
    img.alpha_composite(bl, _sc(0, 512))


    br = Image.new("RGBA", _sc(512, 512), (0, 0, 0, 0))
    dbr = ImageDraw.Draw(br)

    _draw_hand_stroke_poly(dbr, [(56 * S, 242 * S), (256 * S, 245 * S), (456 * S, 240 * S)], wht, 26.0, seed=2130)

    _draw_hand_stroke_poly(dbr, [(82 * S, 294 * S), (258 * S, 292 * S), (430 * S, 296 * S)], yel, 13.0, seed=2131)

    _draw_hand_stroke_poly(dbr, [(256 * S, 188 * S), (256 * S, 224 * S)], wht, 9.0, seed=2132)
    img.alpha_composite(br, _sc(512, 512))

    img = _apply_paintmarker_texture(img, seed=2140, dry_brush=0.26)
    img.save(os.path.join(OUT_DIR, "decal_paintmarker_ndt_atlas.png"))


def gen_petal_numbers_atlas():

    W, H = 1024, 256
    img = Image.new("RGBA", _sc(W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    wht = (224, 222, 212, 236)
    shd = (18, 20, 24, 150)
    f_num = _font(F_COND_BOLD, 66)
    f_sub = _font(F_MONO_BOLD, 16)

    cw, ch = 128, 128
    for idx in range(16):
        col = idx % 8
        row = idx // 8
        x0, y0 = col * cw, row * ch
        x1, y1 = x0 + cw, y0 + ch
        num_txt = f"{idx + 1:02d}"

        d.line(_sc(x0 + 18, y0 + 18, x1 - 18, y0 + 18), fill=wht, width=3 * S)
        d.line(_sc(x0 + 18, y1 - 18, x1 - 18, y1 - 18), fill=wht, width=3 * S)

        _draw_centered_text(d, (x0 + 12, y0 + 22, x1 - 8, y1 - 32), num_txt, f_num, shd)
        _draw_centered_text(d, (x0 + 10, y0 + 20, x1 - 10, y1 - 34), num_txt, f_num, wht)
        _draw_centered_text(d, (x0 + 14, y1 - 36, x1 - 14, y1 - 20), f"FLAP-{idx + 1:02d}", f_sub, (210, 208, 198, 205))

    img = _apply_micro_grain(img, mode="stencil", seed=2155, strength=0.85)
    img.save(os.path.join(OUT_DIR, "decal_petal_numbers_atlas.png"))


def gen_tire_slippage_mark():

    W, H = 256, 512
    img = Image.new("RGBA", _sc(W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    ivory = (236, 234, 224, 240)

    pts = []
    for i in range(25):
        f = i / 24.0
        yv = (26.0 + f * 460.0) * S
        xv = (128.0 + 3.2 * math.sin(f * math.pi * 2.4)) * S
        pts.append((xv, yv))
    _draw_hand_stroke_poly(d, pts, ivory, 62.0, seed=2161)

    rng = np.random.default_rng(2162)
    for k in range(14):
        bx = (98.0 + k * 4.5 + float(rng.normal(0.0, 1.2))) * S
        y_top = (14.0 + float(rng.uniform(0.0, 24.0))) * S
        y_bot = (474.0 + float(rng.uniform(0.0, 24.0))) * S
        d.line([(bx, y_top), (bx, y_top + 28 * S)], fill=(236, 234, 224, 175), width=2 * S)
        d.line([(bx, y_bot - 26 * S), (bx, y_bot)], fill=(236, 234, 224, 175), width=2 * S)

    img = _apply_paintmarker_texture(img, seed=2163, dry_brush=0.30)
    img.save(os.path.join(OUT_DIR, "decal_tire_slippage_mark.png"))


def gen_dd1574_serviceable_tag():

    W, H = 800, 1200
    img = Image.new("RGBA", _sc(W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    yel_card = (228, 192, 54, 255)
    yel_dark = (202, 166, 40, 255)
    blk      = (22, 24, 28, 248)
    blk_soft = (42, 44, 48, 225)
    pen_blu  = (24, 42, 108, 232)
    red_stp  = (184, 30, 26, 210)
    brass    = (166, 134, 58, 255)
    brass_dk = (92, 72, 30, 255)


    poly = _sc(
        130, 16,  670, 16,
        784, 132, 784, 1184,
        16, 1184, 16, 132,
    )
    poly_pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
    d.polygon(poly_pts, fill=yel_card, outline=yel_dark)


    in_pts = [
        _sc(142, 34), _sc(658, 34),
        _sc(764, 142), _sc(764, 1164),
        _sc(36, 1164), _sc(36, 142),
    ]
    d.line(in_pts + [in_pts[0]], fill=blk, width=4 * S)


    cx, cy = 400 * S, 96 * S
    d.ellipse([cx - 46 * S, cy - 46 * S, cx + 46 * S, cy + 46 * S], fill=brass, outline=brass_dk, width=3 * S)
    d.ellipse([cx - 24 * S, cy - 24 * S, cx + 24 * S, cy + 24 * S], fill=(28, 26, 22, 255), outline=brass_dk, width=2 * S)


    d.rectangle(_sc(56, 168, 744, 244), fill=blk)
    _draw_centered_text(d, (64, 172, 736, 240), "SERVICEABLE TAG - MATERIEL", _font(F_COND_BOLD, 44), yel_card)
    _draw_centered_text(d, (64, 252, 736, 290), "DD FORM 1574  •  USAF ENGINE MAINTENANCE", _font(F_SANS_BOLD, 24), blk)


    rows = [
        (300, 396, "1. NATIONAL STOCK NUMBER (NSN)", "2840-01-214-8820"),
        (396, 492, "2. NOMENCLATURE / ITEM NAME",    "TURBOFAN ENGINE  F100-PW-220"),
        (492, 588, "3. PART NUMBER  /  SERIAL NO.",  "P/N 4070200-01   •   S/N P680412"),
        (588, 684, "4. QTY & UNIT   /   T.O. REF",   "1 EA  (M-4 CRADLE)   •   T.O. 2J-F100-3"),
    ]
    f_lbl = _font(F_SANS_BOLD, 17)
    f_val = _font(F_MONO_BOLD, 25)
    f_pen = _font(F_HAND_PEN, 24)
    f_pen_sm = _font(F_HAND_PEN, 21)

    for y0, y1, lbl, val in rows:
        d.rectangle(_sc(56, y0, 744, y1), outline=blk, width=3 * S)
        d.text(_sc(72, y0 + 8), lbl, font=f_lbl, fill=blk_soft)
        d.text(_sc(76, y0 + 42), val, font=f_val, fill=blk)


    d.rectangle(_sc(56, 684, 744, 964), outline=blk, width=3 * S)
    d.text(_sc(72, 694), "5. REMARKS / CONDITION / INSPECTION RECORD", font=f_lbl, fill=blk_soft)
    for ry in (768, 828, 888, 944):
        d.line(_sc(72, ry, 728, ry), fill=(80, 74, 42, 140), width=2 * S)
    d.text(_sc(80, 728), "TSO: 412.5 HRS  •  B-SCOPE & NDT PASS", font=f_pen, fill=pen_blu)
    d.text(_sc(80, 788), "NOZZLE RIGGED & OIL SERVICED (MIL-L-7808)", font=f_pen_sm, fill=pen_blu)
    d.text(_sc(80, 848), "FOD CAPS INSTALLED — READY FOR FLIGHT", font=f_pen_sm, fill=pen_blu)
    d.text(_sc(80, 904), "INSP: M. REYNOLDS TSgt   04/22/26", font=f_pen, fill=pen_blu)


    d.rectangle(_sc(56, 964, 420, 1086), outline=blk, width=3 * S)
    d.rectangle(_sc(420, 964, 744, 1086), outline=blk, width=3 * S)
    d.text(_sc(72, 974), "6. INSPECTOR STAMP / DATE", font=f_lbl, fill=blk_soft)
    d.text(_sc(78, 1020), "QA-04  /  22 APR 26", font=f_pen, fill=pen_blu)
    d.text(_sc(436, 974), "7. UNIT / WORK CENTER", font=f_lbl, fill=blk_soft)
    d.text(_sc(440, 1020), "33 FW • BAY 04", font=f_val, fill=blk)

    _draw_barcode(d, 76, 1102, 724, 1148, fill=blk)


    stp = Image.new("RGBA", _sc(280, 280), (0, 0, 0, 0))
    ds = ImageDraw.Draw(stp)
    ds.ellipse(_sc(14, 14, 266, 266), outline=red_stp, width=7 * S)
    ds.ellipse(_sc(28, 28, 252, 252), outline=red_stp, width=3 * S)
    ds.line(_sc(28, 104, 252, 104), fill=red_stp, width=3 * S)
    ds.line(_sc(28, 176, 252, 176), fill=red_stp, width=3 * S)
    _draw_centered_text(ds, (34, 42, 246, 98), "USAF QA", _font(F_COND_BOLD, 34), red_stp)
    _draw_centered_text(ds, (30, 106, 250, 174), "PASSED", _font(F_COND_BOLD, 46), red_stp)
    _draw_centered_text(ds, (34, 180, 246, 238), "33FW • BAY 04", _font(F_MONO_BOLD, 22), red_stp)
    stp = _apply_paintmarker_texture(stp, seed=2177, dry_brush=0.35)
    stp = stp.rotate(-14.0, resample=Image.BICUBIC, expand=False)
    img.alpha_composite(stp, _sc(468, 724))


    arr = np.asarray(img, dtype=np.float32).copy()
    Hh, Ww, _ = arr.shape
    rng = np.random.default_rng(2180)
    fiber = rng.normal(0.0, 5.5, (Hh, Ww, 1)).astype(np.float32)
    ys_g = np.linspace(-1.0, 1.0, Hh, dtype=np.float32)[:, None]
    xs_g = np.linspace(-1.0, 1.0, Ww, dtype=np.float32)[None, :]
    edge_grime = np.clip((np.maximum(np.abs(xs_g), np.abs(ys_g)) - 0.72) / 0.28, 0.0, 1.0) ** 1.6
    m_card = arr[:, :, 3] > 20.0
    for c, tint in enumerate((0.84, 0.80, 0.72)):
        arr[:, :, c] = np.where(
            m_card,
            np.clip(arr[:, :, c] * (1.0 - 0.16 * edge_grime * (1.0 - tint)) + fiber[:, :, 0], 0.0, 255.0),
            arr[:, :, c],
        )
    img = Image.fromarray(arr.astype(np.uint8), "RGBA")
    img.save(os.path.join(OUT_DIR, "decal_dd1574_serviceable_tag.png"))


def generate_all_decals():
    os.makedirs(OUT_DIR, exist_ok=True)
    funcs = [
        gen_tk9_rail_blue,
        gen_reflector_redwhite,
        gen_forklift_chevron,
        gen_m4_arch_placard,
        gen_shroud_danger_blast,
        gen_shroud_hot_station,
        gen_belly_band_latch,
        gen_intake_danger,
        gen_tec_firezone,
        gen_oiltank_spec,
        gen_trailer_usaf_spec,
        gen_deec_caution,
        gen_micro_hyd_spec,
        gen_micro_pmel_cal,
        gen_micro_nostep_hatch,
        gen_m4_hoist_beam,
        gen_stand_fwd_bumper,
        gen_shroud_aft_index,
        gen_spinner_balance_mark,
        gen_shroud_nopush_plate,
        gen_actuator_dual_band,
        gen_rbf_streamer,
        gen_nozzle_pn407808_nostep,
        gen_fluid_line_bands,
        gen_oilsight_graduations,
        gen_m4_roll_lock_plate,
        gen_trailer_dataplate_dual,
        gen_stand_plates_4k_atlas,
        gen_fodbox_techdoc_atlas,
        gen_paintmarker_ndt_atlas,
        gen_petal_numbers_atlas,
        gen_tire_slippage_mark,
        gen_dd1574_serviceable_tag,
    ]
    for fn in funcs:
        fn()
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".png"))
    print(f"[Decals] Generated {len(files)} decal PNGs in {OUT_DIR}:")
    for f in files:
        p = os.path.join(OUT_DIR, f)
        sz = os.path.getsize(p)
        w, h = Image.open(p).size
        print(f"  - {f:<34} {w}x{h:<5} ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    generate_all_decals()
