#!/usr/bin/env python
"""Generate the repo landing banner (docs/assets/banner.png).

Deterministic PIL render, committed alongside the image so the asset stays
reproducible. Design: GitHub-dark (Primer) neutrals, single emerald accent,
system grotesk for display type, system mono for data. Real measured scores
only. Run: .venv/bin/python scripts/make_banner.py
"""

from __future__ import annotations

import os
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- palette ---
BG = "#0d1117"        # canvas (Primer dark)
CARD = "#161b22"      # raised surface
BORDER = "#30363d"    # hairline
TEXT = "#e6edf3"      # primary
MUTED = "#8b949e"     # secondary
ACCENT = "#3fb950"    # emerald, the single accent (checks, hero score, logo)

S = 2                 # supersample factor (crisp on retina)
W, H = 1280 * S, 640 * S

DISPLAY = [
    ("/System/Library/Fonts/SFNS.ttf", 0, True),                     # SF, bold via variation
    ("/System/Library/Fonts/Helvetica.ttc", 1, False),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0, False),
]
SANS = [
    ("/System/Library/Fonts/SFNS.ttf", 0, False),
    ("/System/Library/Fonts/Helvetica.ttc", 0, False),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0, False),
]
MONO = [
    ("/System/Library/Fonts/Menlo.ttc", 0, False),
    ("/System/Library/Fonts/Monaco.ttf", 0, False),
    ("/System/Library/Fonts/Supplemental/Courier New.ttf", 0, False),
]


def load(candidates, size):
    for path, index, bold_var in candidates:
        if os.path.exists(path):
            try:
                f = ImageFont.truetype(path, size, index=index)
                if bold_var:
                    try:
                        f.set_variation_by_name("Bold")
                    except Exception:
                        pass
                return f
            except Exception:
                continue
    return ImageFont.load_default(size=size)


def px(v: float) -> int:
    return int(round(v * S))


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

f_title = load(DISPLAY, px(76))
f_sub = load(SANS, px(23))
f_chip = load(MONO, px(19))
f_quick = load(MONO, px(19))
f_head = load(DISPLAY, px(23))
f_row = load(MONO, px(21))
f_hero = load(DISPLAY, px(30))
f_hero_val = load(DISPLAY, px(34))
f_sub2 = load(MONO, px(16))
f_item = load(SANS, px(20))
f_meta = load(MONO, px(16))


def text(xy, s, font, fill, anchor="la"):
    d.text((px(xy[0]), px(xy[1])), s, font=font, fill=fill, anchor=anchor)


def rrect(box, radius, fill=None, outline=None, width=1):
    d.rounded_rectangle([px(box[0]), px(box[1]), px(box[2]), px(box[3])],
                        radius=px(radius), fill=fill, outline=outline,
                        width=max(1, px(width * 0.5)))


def check(x, y, size):
    """Vector checkmark, emerald, round joints."""
    pts = [(x, y + 0.48 * size), (x + 0.34 * size, y + 0.82 * size),
           (x + size, y - 0.04 * size)]
    d.line([(px(p[0]), px(p[1])) for p in pts], fill=ACCENT,
           width=max(2, px(size * 0.17)), joint="curve")


# ------------------------------------------------------------------ left ----
# Logo mark: rounded square, emerald, with a decision-tree glyph.
lx, ly, ls = 72, 72, 78
rrect((lx, ly, lx + ls, ly + ls), 18, fill=ACCENT)
root = (lx + 0.27 * ls, ly + 0.5 * ls)
outs = [(lx + 0.73 * ls, ly + 0.27 * ls), (lx + 0.73 * ls, ly + 0.5 * ls),
        (lx + 0.73 * ls, ly + 0.73 * ls)]
for o in outs:
    d.line([(px(root[0]), px(root[1])), (px(o[0]), px(o[1]))], fill=BG, width=px(4))
for c in [root] + outs:
    r = 5.5
    d.ellipse([px(c[0] - r), px(c[1] - r), px(c[0] + r), px(c[1] + r)], fill=BG)

text((176, 86), "jev-smol", f_title, TEXT)
text((178, 172), "An embedded text + image decision model,", f_sub, MUTED)
text((178, 202), "fine-tuned from SmolVLM2-500M with the", f_sub, MUTED)
text((178, 232), "Jev / CLEF System One decision API.", f_sub, MUTED)

# Question-type chips: the whole API surface in three tokens.
f_chip = load(MONO, px(17))
chips = ["noul  P(yes)", "choice  probabilities", "score  E[level]"]
cx = 72
for c in chips:
    w = d.textlength(c, font=f_chip) / S
    rrect((cx, 296, cx + w + 36, 344), 999, outline=BORDER, width=1)
    text((cx + 18, 310), c, f_chip, TEXT)
    cx += w + 36 + 12

# Quickstart bar.
q = "$ python -m jev serve --backend vlm"
qw = d.textlength(q, font=f_quick) / S
rrect((72, 540, 72 + qw + 44, 590), 999, fill=CARD, outline=BORDER, width=1)
text((94, 555), q, f_quick, TEXT)

# ------------------------------------------------------------- scoreboard ---
SX0, SY0, SX1, SY1 = 740, 72, 1208, 568
rrect((SX0, SY0, SX1, SY1), 16, fill=CARD, outline=BORDER, width=1)
text((SX0 + 32, SY0 + 26), "training progress", f_head, TEXT)

rows = [
    ("base smolvlm2-500m", "41.2%", False),
    ("lora 40 steps", "43.0%", False),
    ("lora 2 epochs", "58.2%", True),
]
ry = SY0 + 74
for label, val, hero in rows:
    if hero:
        text((SX0 + 32, ry - 4), label, f_hero, TEXT)
        check(SX0 + 300, ry + 8, 26)
        text((SX1 - 32, ry - 6), val, f_hero_val, ACCENT, anchor="ra")
        text((SX0 + 32, ry + 40), "noul 81.3%   score 47.2%   choice 39.6%", f_sub2, MUTED)
        text((SX0 + 32, ry + 64), "brier 0.457 (base 0.610)", f_sub2, MUTED)
        ry += 94
    else:
        text((SX0 + 32, ry), label, f_row, MUTED)
        text((SX1 - 32, ry - 2), val, f_row, MUTED, anchor="ra")
        ry += 42

d.line([(px(SX0 + 32), px(ry + 4)), (px(SX1 - 32), px(ry + 4))], fill=BORDER, width=S)

items = [
    "System One API server",
    "text + image decision engine",
    "synthetic data generators",
    "accuracy + Brier eval harness",
    "LoRA SFT training pipeline",
]
iy = ry + 30
for it in items:
    check(SX0 + 32, iy - 2, 20)
    text((SX0 + 64, iy - 4), it, f_item, TEXT)
    iy += 40

# ---------------------------------------------------------------- footer ----
meta = ["Apache-2.0", "500M params", "89 tests"]
mx = 72
for m in meta:
    text((mx, 608), m, f_meta, MUTED)
    mx += d.textlength(m, font=f_meta) / S + 48

os.makedirs("docs/assets", exist_ok=True)
img.save("docs/assets/banner.png", format="PNG", optimize=True)
print("wrote docs/assets/banner.png", img.size)
