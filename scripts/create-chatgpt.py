#!/usr/bin/env python3

import os, math, random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np


# ---------------------------
# Labels / card definitions
# ---------------------------
SUITS = {
    "hearts":   {"sym": "♥", "color": (200, 0, 0)},
    "diamonds": {"sym": "♦", "color": (200, 0, 0)},
    "clubs":    {"sym": "♣", "color": (0, 0, 0)},
    "spades":   {"sym": "♠", "color": (0, 0, 0)},
}
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]

def label(rank, suit):
    rank_name = {
        "A": "ace", "J": "jack", "Q": "queen", "K": "king"
    }.get(rank, rank)
    return f"{rank_name}_of_{suit}"

# ---------------------------
# Simple pip layouts
# ---------------------------
# (x,y) in normalized card coordinates (0..1)
PIP_LAYOUT = {
    "A":  [(0.5, 0.5)],
    "2":  [(0.5, 0.25), (0.5, 0.75)],
    "3":  [(0.5, 0.2), (0.5, 0.5), (0.5, 0.8)],
    "4":  [(0.3,0.25),(0.7,0.25),(0.3,0.75),(0.7,0.75)],
    "5":  [(0.3,0.25),(0.7,0.25),(0.5,0.5),(0.3,0.75),(0.7,0.75)],
    "6":  [(0.3,0.2),(0.7,0.2),(0.3,0.5),(0.7,0.5),(0.3,0.8),(0.7,0.8)],
    "7":  [(0.3,0.2),(0.7,0.2),(0.3,0.45),(0.7,0.45),(0.5,0.6),(0.3,0.8),(0.7,0.8)],
    "8":  [(0.3,0.18),(0.7,0.18),(0.3,0.38),(0.7,0.38),(0.3,0.62),(0.7,0.62),(0.3,0.82),(0.7,0.82)],
    "9":  [(0.3,0.18),(0.7,0.18),(0.3,0.38),(0.7,0.38),(0.5,0.5),(0.3,0.62),(0.7,0.62),(0.3,0.82),(0.7,0.82)],
    "10": [(0.3,0.16),(0.7,0.16),(0.3,0.34),(0.7,0.34),(0.3,0.5),(0.7,0.5),(0.3,0.66),(0.7,0.66),(0.3,0.84),(0.7,0.84)],
}
# Face cards: still use rank/suit big center for recognizability
FACE_RANKS = {"J","Q","K"}


# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rand_bg(w, h):
    """Create a random background image."""
    # Option A: solid + noise
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:] = np.random.randint(0, 255, size=(1,1,3), dtype=np.uint8)
    noise = np.random.normal(0, 18, (h, w, 3)).astype(np.int16)
    out = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB").filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.2)))

def paste_with_alpha(bg, fg, xy):
    if fg.mode != "RGBA":
        fg = fg.convert("RGBA")
    bg.paste(fg, xy, fg)

def random_occlusion(draw, w, h):
    """Draw one or more occluding shapes."""
    if random.random() < 0.55:
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            x2 = x1 + random.randint(int(0.06*w), int(0.22*w))
            y2 = y1 + random.randint(int(0.06*h), int(0.22*h))
            col = tuple(int(c) for c in np.random.randint(0, 255, size=3))
            draw.rectangle([x1,y1,x2,y2], fill=col)

def jitter_color(img):
    if random.random() < 0.9:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
    if random.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.35))
    if random.random() < 0.6:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
    return img

def maybe_blur(img):
    if random.random() < 0.55:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.6)))
    return img

def add_sensor_noise(img):
    if random.random() < 0.8:
        arr = np.array(img).astype(np.int16)
        arr += np.random.normal(0, random.uniform(2, 14), arr.shape).astype(np.int16)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert(img.mode)
    return img

def perspective_warp(img):
    """Mild perspective-ish warp using QUAD mapping."""
    if random.random() < 0.75:
        w, h = img.size
        dx = w * random.uniform(0.02, 0.10)
        dy = h * random.uniform(0.02, 0.10)
        quad = (
            random.uniform(0, dx),             random.uniform(0, dy),          # top-left
            random.uniform(w-dx, w),           random.uniform(0, dy),          # top-right
            random.uniform(w-dx, w),           random.uniform(h-dy, h),        # bottom-right
            random.uniform(0, dx),             random.uniform(h-dy, h),        # bottom-left
        )
        img = img.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)
    return img

def rotate_and_scale(img):
    angle = random.uniform(-35, 35)
    scale = random.uniform(0.78, 1.15)
    w, h = img.size
    img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
    return img

def try_load_font(size):
    # Uses DejaVuSans if available (common on Linux). Falls back to default.
    for name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except:
            pass
    return ImageFont.load_default()

# ---------------------------
# Draw a clean card face
# ---------------------------
def render_card_face(rank, suit, card_w=360, card_h=540):
    suit_info = SUITS[suit]
    sym = suit_info["sym"]
    col = suit_info["color"]

    img = Image.new("RGBA", (card_w, card_h), (0,0,0,0))
    d = ImageDraw.Draw(img)

    # Card base
    radius = int(card_w * 0.06)
    # Rounded rectangle (manual)
    base = Image.new("RGBA", (card_w, card_h), (0,0,0,0))
    bd = ImageDraw.Draw(base)
    bd.rounded_rectangle([0,0,card_w-1,card_h-1], radius=radius, fill=(245,245,245,255), outline=(20,20,20,255), width=4)
    img.alpha_composite(base)

    # Fonts
    corner_font = try_load_font(int(card_h * 0.075))
    pip_font = try_load_font(int(card_h * 0.11))
    big_font = try_load_font(int(card_h * 0.22))

    # Corner rank + suit
    pad = int(card_w * 0.06)
    d.text((pad, pad), rank, font=corner_font, fill=col)
    d.text((pad, pad + int(card_h*0.06)), sym, font=corner_font, fill=col)

    # Bottom-right (rotated)
    tmp = Image.new("RGBA", (card_w, card_h), (0,0,0,0))
    td = ImageDraw.Draw(tmp)
    td.text((pad, pad), rank, font=corner_font, fill=col)
    td.text((pad, pad + int(card_h*0.06)), sym, font=corner_font, fill=col)
    tmp = tmp.rotate(180, resample=Image.BICUBIC)
    img.alpha_composite(tmp)

    # Center pips / big face marker
    if rank in FACE_RANKS:
        # Large rank + suit in center (simple, but very learnable)
        center_text = f"{rank}{sym}"
        tw, th = d.textbbox((0,0), center_text, font=big_font)[2:]
        d.text(((card_w - tw)/2, (card_h - th)/2), center_text, font=big_font, fill=col)
    else:
        for (nx, ny) in PIP_LAYOUT.get(rank, [(0.5,0.5)]):
            x = int(nx * card_w)
            y = int(ny * card_h)
            # Flip bottom-half pips for realism
            pip_img = Image.new("RGBA", (card_w, card_h), (0,0,0,0))
            pd = ImageDraw.Draw(pip_img)
            pd.text((x, y), sym, font=pip_font, fill=col, anchor="mm")
            if ny > 0.5:
                pip_img = pip_img.rotate(180, resample=Image.BICUBIC)
            img.alpha_composite(pip_img)

    return img


# ---------------------------
# Compose into a scene
# ---------------------------
def make_sample(rank, suit, out_w=512, out_h=512):
    bg = rand_bg(out_w, out_h).convert("RGBA")

    # Render card and distort it
    card = render_card_face(rank, suit)
    card = perspective_warp(card)
    card = rotate_and_scale(card)

    # Limit card size
    max_dim = int(1.25 * max(out_w, out_h))
    if max(card.size) > max_dim:
        scale = max_dim / max(card.size)
        card = card.resize((int(card.size[0]*scale), int(card.size[1]*scale)), Image.BICUBIC)

    # Optional shadow
    if random.random() < 0.8:
        shadow = card.copy()
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 8)))
        # darken shadow alpha
        sh = np.array(shadow)
        sh[..., :3] = 0
        sh[..., 3] = (sh[..., 3].astype(np.float32) * random.uniform(0.25, 0.55)).astype(np.uint8)
        shadow = Image.fromarray(sh).convert("RGBA")
    else:
        shadow = None

    # Place on background
    cw, ch = card.size

    x_min = -int(0.15 * cw)
    x_max = out_w - int(0.85 * cw)
    y_min = -int(0.15 * ch)
    y_max = out_h - int(0.85 * ch)

    if x_max  < x_min:
        x_max = x_min
    if y_max < y_min:
        y_max = y_min

    x = random.randint(x_min, x_max)
    y = random.randint(y_min, y_max)

    if shadow:
        paste_with_alpha(bg, shadow, (x + random.randint(6, 18), y + random.randint(6, 18)))
    paste_with_alpha(bg, card, (x, y))

    # Optional extra clutter (simple shapes)
    d = ImageDraw.Draw(bg)
    if random.random() < 0.35:
        for _ in range(random.randint(2, 6)):
            x1 = random.randint(0, out_w)
            y1 = random.randint(0, out_h)
            r = random.randint(6, 50)
            col = tuple(int(c) for c in np.random.randint(0, 255, size=3))
            d.ellipse([x1-r, y1-r, x1+r, y1+r], outline=col, width=random.randint(2, 6))

    random_occlusion(d, out_w, out_h)

    # Photographic-ish effects
    img = bg.convert("RGB")
    img = jitter_color(img)
    img = maybe_blur(img)
    img = add_sensor_noise(img)

    # Slight compression artifacts simulation by re-encoding to JPEG in memory
    if random.random() < 0.8:
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(55, 95))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


# ---------------------------
# Main: generate dataset
# ---------------------------
def generate_dataset(
    out_dir="cards_dataset",
    images_per_class=250,
    val_split=0.15,
    seed=1337,
    out_size=512
):
    random.seed(seed)
    np.random.seed(seed)

    out_dir = Path(out_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    ensure_dir(train_dir)
    ensure_dir(val_dir)

    for suit in SUITS.keys():
        for rank in RANKS:
            cls = label(rank, suit)
            (train_dir / cls).mkdir(parents=True, exist_ok=True)
            (val_dir / cls).mkdir(parents=True, exist_ok=True)

            n_val = int(images_per_class * val_split)
            n_train = images_per_class - n_val

            # Train
            for i in range(n_train):
                img = make_sample(rank, suit, out_w=out_size, out_h=out_size)
                img.save(train_dir / cls / f"{cls}_{i:05d}.jpg")

            # Val
            for i in range(n_val):
                img = make_sample(rank, suit, out_w=out_size, out_h=out_size)
                img.save(val_dir / cls / f"{cls}_{i:05d}.jpg")

            print(f"Done {cls}: train={n_train}, val={n_val}")

if __name__ == "__main__":
    generate_dataset(out_dir="cards_dataset", images_per_class=300, val_split=0.15, seed=42, out_size=512)
    print("Dataset created in ./cards_dataset")

