"""
recolour_env.py
======================================================================
Re-render an environment screenshot into a palette that survives being printed
in black and white, and return the masks the figure needs to annotate it.

THE PROBLEM
-----------
The simulator's own palette is chosen for a colour screen and collapses in
greyscale. Converted to luminance, the predator (#FF1F8F) and the low-food
landmark (#ff4000) land 1.8 grey levels apart out of 255, the prestige landmark
and an organism 5.4 apart, and the cave sits 8.6 from the empty background — it
vanishes. Twelve of the forty-five pairs fall inside 25 levels. No amount of
labelling fixes an image whose elements are genuinely the same shade.

THE ENCODING
------------
Two channels, and the one that does the heavy lifting is not colour.

  SHAPE says which FAMILY:  scattered single cells are food, continuous lines
                            are landmarks, a filled rectangle is a cave, a
                            compact multi-cell blob is an agent.
  LUMINANCE says which TIER: default 61, low 115, medium 157, prestige 213 out
                            of 255, no two closer than 42 — and a tier's food
                            and its landmark share a grey, because shape has
                            already separated them.

That leaves only two things luminance cannot carry, and both get a mark instead:
every predator is RINGED, and every cave is hatched inside a dashed border. So a
ringed blob is a predator and an unringed one an organism, whatever the ink.

Hue is kept on top of all this — red stays low food, cyan stays prestige — so
the plate still looks like the simulator on screen. It is the luminance ladder
underneath that makes it work in print.

DISAMBIGUATING GOLD
-------------------
`organism mouth` #DEB14D and `medium food landmark` #ffbf00 are close enough
that the canvas's anti-aliasing makes a nearest-colour classifier swap them.
They are told apart structurally instead: an organism always has a mover or eye
cell against it, a landmark line never does. On this screenshot that splits
3841 gold pixels into 740 organism and 3101 landmark, which matches the pixel
counts of the other two landmark tiers.
"""

import numpy as np
from scipy import ndimage

# Every colour the renderer emits, mapped to the class it belongs to.
SRC = [
    ('background', '#0E1318'), ('default food', '#2F7AB7'), ('low food', '#ff0000'),
    ('medium food', '#ff8000'), ('prestige food', '#00ffff'), ('cave', '#3a0d00'),
    ('low landmark', '#ff4000'), ('medium landmark', '#ffbf00'),
    ('prestige landmark', '#00bfff'),
    ('organism', '#DEB14D'), ('organism', '#60D4FF'), ('organism', '#B6C1EA'),
    ('predator', '#FF1F8F'), ('predator', '#FF63B5'), ('predator', '#FFD1EA'),
    ('predator', '#FF00C8'),
]

# grey level in brackets — see THE ENCODING above
OUT = {
    'background':        '#0E1318',   # 25
    'cave':              '#66290F',   # 62
    'default food':      '#1F3C5E',   # 61
    'low food':          '#E31B1B',   # 115
    'low landmark':      '#E31B1B',   # 115  same tier, shape separates it
    'medium food':       '#E58829',   # 157
    'medium landmark':   '#E58829',   # 157
    'prestige food':     '#92E3EF',   # 213
    'prestige landmark': '#92E3EF',   # 213
    'predator':          '#FF74BE',   # 163, and ringed
    'organism':          '#FFF6DC',   # 246
}


def _rgb(h):
    return np.array([int(h[i:i+2], 16) for i in (1, 3, 5)], float)


def classify(img):
    """Per-pixel class name, with the gold ambiguity resolved structurally."""
    a = img[:, :, :3].astype(float)
    a = a * 255 if a.max() <= 1.0 else a
    pal = np.array([_rgb(h) for _n, h in SRC])
    idx = np.linalg.norm(a[:, :, None, :] - pal[None, None, :, :], axis=3).argmin(axis=2)

    gold = idx == 9                                   # the #DEB14D entry
    partner = np.isin(idx, [10, 11])                  # mover / eye cells
    beside = ndimage.binary_dilation(partner, np.ones((7, 7)))

    names = np.array([n for n, _h in SRC], dtype=object)
    lab = names[idx]
    lab[gold & ~beside] = 'medium landmark'           # a line, not a creature
    return lab


def recolour(img):
    """(RGB uint8 in the print-safe palette, class map)."""
    lab = classify(img)
    out = np.zeros(lab.shape + (3,), dtype=np.uint8)
    for name, hexc in OUT.items():
        out[lab == name] = _rgb(hexc).astype(np.uint8)
    return out, lab


def blobs(lab, name, min_px=3):
    """Centroids and radii of each connected component of a class."""
    m = lab == name
    l, n = ndimage.label(m, np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(l == i)
        if ys.size < min_px:
            continue
        out.append((xs.mean(), ys.mean(),
                    max(np.ptp(xs), np.ptp(ys)) / 2 + 1.5, ys.size))
    return out


def boxes(lab, name, min_px=40):
    """Bounding boxes of each connected component of a class."""
    m = lab == name
    l, n = ndimage.label(m, np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(l == i)
        if ys.size < min_px:
            continue
        out.append((xs.min(), ys.min(), np.ptp(xs) + 1, np.ptp(ys) + 1))
    return out
