"""
make_environment_figure.py
======================================================================
The two environments side by side, in an unlabelled and a labelled version.

    python make_environment_figure.py

Writes into output/environments/:
    environments.pdf            the two screenshots side by side, titles only
    predator_labelled.pdf       the PREDATOR screenshot alone, with numbered
                                callouts and a key
    (.png twins of both, for previewing)

The labelled plate uses the predator environment because it is the only one that
contains every element: the baseline world has no predators, so a key drawn over
it would have one entry with nothing to point at. Everything else is common to
both, so the labels transfer to the baseline panel unchanged.

WHY THE KEY IS DRAWN FROM ColorScheme.js AND NOT SAMPLED FROM THE SCREENSHOTS
------------------------------------------------------------------------------
Every swatch below is the literal hex the renderer uses (src/Rendering/
ColorScheme.js, "neon"), so the key cannot drift from the simulator. Sampling
the PNG instead would pick up whatever alpha blending the canvas applied — the
amber landmark lines, for instance, land close enough to the organism mouth
colour that a nearest-colour classifier confuses the two.

WHY NUMBERED CALLOUTS RATHER THAN ARROWS WITH TEXT ON THE IMAGE
----------------------------------------------------------------
Ten categories is more than an image this dense can carry as labelled arrows
without the text colliding with the thing it points at. A number is one glyph,
it sits in the dark background beside its feature, and the name lives in the key
underneath where it has room. The numbers are white on dark with a stroke, so
they survive greyscale printing.

COLOUR IS NOT THE ONLY CHANNEL, AND HERE IT CANNOT BE
------------------------------------------------------
Low food (#ff0000) and its landmark (#ff4000) are almost the same hue: what
separates them is SHAPE — scattered single cells against a continuous line. The
key says so in words rather than relying on the swatch.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.patheffects as pe

import recolour_env as R

INK = '#1F2933'
BASE_IMG = 'baseline environment.png'
PRED_IMG = 'predator.png'

# src/Rendering/ColorScheme.js, scheme "neon".
# Numbered in semantic groups rather than by where they happen to sit on the
# plate: 1-3 the food tiers in ascending score, 4-6 their matching landmarks in
# the same order, 7-8 the two kinds of agent, 9-10 the cave and the default food
# that has no landmark of its own. The key is laid out five to a row, so reading
# left to right and wrapping gives 1..5 then 6..10.
# (label, class name, shape family, note). The colour comes from
# recolour_env.OUT so the key cannot drift from the plate, and the icon drawn in
# the swatch is the SHAPE, because shape is what separates a food tier from its
# landmark once both are printed in the same grey.
KEY = [
    ('Low food',          'low food',          'cells', 'score 0.5'),
    ('Medium food',       'medium food',       'cells', 'score 1.0'),
    ('Prestige food',     'prestige food',     'cells', 'score 2.0'),
    ('Low landmark',      'low landmark',      'line',  'marks low food'),
    ('Medium landmark',   'medium landmark',   'line',  'marks medium food'),
    ('Prestige landmark', 'prestige landmark', 'line',  'marks prestige food'),
    ('Organism',          'organism',          'blob',  'unringed'),
    ('Predator',          'predator',          'ring',  'ringed on the plate'),
    ('Cave',              'cave',              'cave',  'hatched, dashed edge'),
    ('Default food',      'default food',      'cells', 'score 0.01'),
]

# (number, x, y, dx, dy) in each screenshot's own pixel coordinates. dx/dy push
# the glyph off the feature into neighbouring empty space so it never sits on
# top of what it is pointing at. Anchors were located by classifying pixels
# against the scheme palette and then confirmed by eye on magnified crops.
# (number, x, y, dx, dy) in the predator screenshot's 700x700 pixel space.
# dx/dy push the glyph off the feature into neighbouring empty space. Anchors
# were found by classifying pixels against the scheme palette under radius
# constraints, then confirmed on magnified crops; the food anchors additionally
# require no landmark pixels nearby, and 4 no predator or organism either, so
# each number points at an unambiguous instance of one thing.
CALLOUTS = [(1, 400, 234, -56, -44), (2, 466, 192, 56, -40),
            (3, 66, 214, 58, -42), (4, 470, 325, 62, 34),
            (5, 177, 472, -56, 44), (6, 314, 67, 56, 44),
            (7, 375, 357, 54, 46), (8, 349, 637, -60, -34),
            (9, 619, 290, -60, -48), (10, 327, 327, -66, -54)]


def panel(ax, img, title):
    ax.imshow(img, interpolation='nearest')
    if title:
        ax.set_title(title, fontsize=9, color=INK, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#9AA5B1')
        sp.set_linewidth(.6)


def callout(ax, n, x, y, dx, dy):
    """A leader from the feature to a number sitting clear of it."""
    ax.annotate(str(n), xy=(x, y), xytext=(x + dx, y + dy),
                fontsize=7.5, color='white', ha='center', va='center',
                zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground='black')],
                arrowprops=dict(arrowstyle='-', color='white', lw=.8,
                                shrinkA=3, shrinkB=1,
                                path_effects=[pe.withStroke(linewidth=2.0,
                                                            foreground='black')]))


def mark_agents(ax, lab):
    """Ring every predator and hatch every cave, on the plate itself.

    These are the two classes luminance cannot carry: the predator sits between
    two food tiers in grey, and the cave is only 37 levels off the background.
    A ring and a hatch are ink, not colour, so they read on any printer.
    """
    # Sized to sit just outside the creature and no further. An earlier pass
    # used a 6px floor with a black halo, and 58 of those read as the subject of
    # the figure rather than as a mark on it.
    for x, y, r, _n in R.blobs(lab, 'predator'):
        ax.add_patch(Circle((x, y), max(r + 1.6, 4.0), fill=False,
                            edgecolor='white', linewidth=.45, alpha=.9,
                            zorder=5))
    for x0, y0, w, h in R.boxes(lab, 'cave'):
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor='none',
                               edgecolor='white', linewidth=.8, linestyle=(0, (3, 2)),
                               hatch='///', zorder=4))


def key_icon(fig, rect, family, colour):
    """The swatch is a picture of the SHAPE, not a plain block of colour."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(R.OUT['background'])
    for sp in ax.spines.values():
        sp.set_edgecolor('#9AA5B1'); sp.set_linewidth(.4)
    if family == 'cells':
        for cx, cy in [(2, 7.4), (5.2, 8.6), (7.9, 6.2), (3.4, 3.2), (7.1, 2.2)]:
            ax.add_patch(Rectangle((cx, cy), 1.5, 1.5, color=colour))
    elif family == 'line':
        ax.plot([.8, 9.2], [2.2, 7.8], color=colour, lw=2.6,
                solid_capstyle='butt')
    elif family == 'blob':
        ax.add_patch(Rectangle((3.6, 3.6), 2.8, 2.8, color=colour))
    elif family == 'ring':
        ax.add_patch(Rectangle((3.6, 3.6), 2.8, 2.8, color=colour))
        ax.add_patch(Circle((5, 5), 3.3, fill=False, edgecolor='white', lw=.8))
    elif family == 'cave':
        ax.add_patch(Rectangle((1.4, 1.8), 7.2, 6.4, facecolor=colour,
                               edgecolor='white', lw=.8, linestyle=(0, (2.2, 1.6)),
                               hatch='///'))
    return ax


def draw_key(fig, x0, y_top, dy):
    """The ten elements in one column, read top to bottom as 1..10."""
    for i, (name, cls, family, note) in enumerate(KEY):
        y = y_top - i * dy
        key_icon(fig, [x0, y, .030, .046], family, R.OUT[cls])
        fig.text(x0 + .040, y + .034, f'{i + 1}. {name}', fontsize=7.2,
                 color=INK, va='center')
        fig.text(x0 + .040, y + .013, note, fontsize=6.0, color='#6B7280',
                 va='center')


def build_pair(out_stem):
    """Both environments side by side, no labels, print-safe palette."""
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.45))
    for ax, f, t in ((axs[0], BASE_IMG, '(a) Baseline environment'),
                     (axs[1], PRED_IMG, '(b) Predator environment')):
        rgb, lab = R.recolour(mpimg.imread(f))
        panel(ax, rgb, t)
        mark_agents(ax, lab)
    fig.subplots_adjust(left=.02, right=.98, top=.94, bottom=.02, wspace=.05)
    save(fig, out_stem)


def build_labelled(out_stem):
    """The predator environment alone, with the key beside it rather than under.

    Beside, because the plate is square: a key underneath would make the figure
    taller than it is wide and waste the column, whereas the space to the right
    of a square panel on a text-width figure is free.
    """
    rgb, lab = R.recolour(mpimg.imread(PRED_IMG))
    fig = plt.figure(figsize=(7.0, 4.35))
    ax = fig.add_axes([.015, .045, .500, .915])
    panel(ax, rgb, None)
    mark_agents(ax, lab)
    for c in CALLOUTS:
        callout(ax, *c)
    draw_key(fig, .560, .880, .0905)
    save(fig, out_stem)


def save(fig, stem):
    for ext in ('pdf', 'png'):
        p = f'{stem}.{ext}'
        fig.savefig(p, dpi=300, facecolor='white', bbox_inches='tight',
                    pad_inches=0.03)
        print(f'  wrote {p}')
    plt.close(fig)


def main():
    out = os.path.join('output', 'environments')
    os.makedirs(out, exist_ok=True)
    build_pair(os.path.join(out, 'environments'))
    build_labelled(os.path.join(out, 'predator_labelled'))


if __name__ == '__main__':
    main()
