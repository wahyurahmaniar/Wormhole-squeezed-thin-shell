"""Shared figure conventions of the manuscript: bright plot colors, STIXGeneral fonts,
non-bold panel labels centered under each panel on a common line, vector PDF plus 500 dpi PNG.
Every figure is drawn at its final printed width (7.22 in, the journal text width), so all text,
including the panel letters, prints at the point size set here."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {
    "blue": "#0000FF", "red": "#FF0000", "green": "#00A000",
    "magenta": "#E000E0", "orange": "#FF8000", "purple": "#8000C0", "cyan": "#00A8E8",
    "grey": "#808080", "lightgrey": "#D0D0D0", "black": "#000000",
    "h2": "#BFE6FF",          # hydrogen fill
    "hg": "#A8A8A8",          # mercury walls
}
TEXTWIDTH_IN = 522.0 / 72.27
DPI = 500
LABEL_PT = 8.5


def apply_style():
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "legend.fontsize": 6.9,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.3,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
    })


def panel_labels(fig, axes, labels, dy=0.030):
    """Non-bold labels on one line below the lowest panel, each centered on its own panel.

    The letter is centered on the axes box, the same horizontal center matplotlib gives the
    x-axis label, and sits a fixed distance below the lowest drawn element of any panel.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y = min(ax.get_tightbbox(renderer).transformed(inv).y0 for ax in axes) - dy
    for ax, lab in zip(axes, labels):
        pos = ax.get_position()
        fig.text(0.5 * (pos.x0 + pos.x1), y, lab, ha="center", va="top",
                 fontsize=LABEL_PT, fontweight="normal")


def save(fig, stem):
    """Write stem.pdf (vector) and stem.png (500 dpi) at the final printed size."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    fig.savefig(stem + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem + ".png", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
