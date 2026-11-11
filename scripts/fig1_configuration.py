"""Fig. 1: the junction and the apparatus. (a) Cut-and-paste junction: two flat sheets identified
along the rim R = a0 (the idealized zero-thickness shell), each carrying the annular half of the
mercury-hydrogen cavity. (b) Cross-section along the proper normal distance x for C2 and C4: mercury
walls at x = +-lambda/4 of the in-medium wavelength (3.31 cm in solid hydrogen), the filling, the
junction layer |x| <= a = sqrt(3) ell, the unloaded standing-wave fields of the mode polarized along y,
with k = n_H k0, and the reactive sheet of the passive model at x = +-d. (c) Polarization completion for a vacuum layer written out component by component."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from figstyle import COL, apply_style, panel_labels, save, TEXTWIDTH_IN
import hg_model as M

apply_style()
plt.rcParams["hatch.linewidth"] = 0.8
r = M.r_ref
s, c = np.sinh(r), np.cosh(r)
A = s ** 2 - s * c * M.G0
B = s * c - s ** 2 * M.G0
LEG = dict(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=6.6, handlelength=1.7,
           columnspacing=0.9, handletextpad=0.45, borderaxespad=0.0, labelspacing=0.3)

fig = plt.figure(figsize=(TEXTWIDTH_IN - 0.07, 2.12))
ax_a = fig.add_axes([0.000, 0.17, 0.335, 0.62])
ax_b = fig.add_axes([0.395, 0.17, 0.285, 0.62])
ax_c = fig.add_axes([0.725, 0.17, 0.275, 0.62])

# ---------------- (a) cut-and-paste junction with the apparatus ----------------
a = ax_a
a0, dcav, dl, k = 1.0, 0.62, 0.11, 0.36
zA, zB = 1.12, -1.12
Lx, Ly = 2.75, 2.15
phi = np.linspace(0, 2 * np.pi, 500)


def ellipse(rad, z):
    return rad * np.cos(phi), z + k * rad * np.sin(phi)


def ring(r1, r2, z):
    xo, yo = ellipse(r2, z); xi, yi = ellipse(r1, z)
    return np.vstack([np.column_stack([xo, yo]), np.column_stack([xi[::-1], yi[::-1]])])


def sheet(z, name):
    sh = 0.5
    corners = np.array([[-Lx - sh, z - k * Ly], [Lx - sh, z - k * Ly], [Lx + sh, z + k * Ly], [-Lx + sh, z + k * Ly]])
    a.add_patch(Polygon(corners, closed=True, fc="#F2F2F2", ec=COL["black"], lw=0.7))
    a.add_patch(Polygon(ring(a0, a0 + dcav, z), closed=True, fc=COL["h2"], ec="none"))
    xo, yo = ellipse(a0 + dcav, z)
    a.plot(xo, yo, color=COL["hg"], lw=3.2, solid_capstyle="butt")
    a.add_patch(Polygon(ring(a0, a0 + dl, z), closed=True, fc=COL["orange"], ec="none"))
    xi, yi = ellipse(a0, z)
    a.add_patch(Polygon(np.column_stack([xi, yi]), closed=True, fc="white", ec="none"))
    a.plot(xi, yi, color=COL["blue"], lw=1.8)
    a.text(-Lx + sh + 0.08, z + k * Ly + 0.05, name, fontsize=6.9, ha="left", va="bottom")


sheet(zA, "exterior " + r"$A$")
sheet(zB, "exterior " + r"$B$")
for ph in (0.0, np.pi, -np.pi / 2):
    x = a0 * np.cos(ph)
    a.plot([x, x], [zB + k * a0 * np.sin(ph), zA + k * a0 * np.sin(ph)], color=COL["blue"], lw=0.7, ls=":")
a.annotate("", xy=(a0, zA), xytext=(0.0, zA), arrowprops=dict(arrowstyle="<->", color=COL["black"], lw=0.7, shrinkA=0, shrinkB=0))
a.text(0.5 * a0, zA + 0.06, r"$a_0$", ha="center", va="bottom", fontsize=7.6)
a.annotate("", xy=(-(a0 + dcav), zB), xytext=(-a0, zB), arrowprops=dict(arrowstyle="<->", color=COL["black"], lw=0.7, shrinkA=0, shrinkB=0))
a.text(-(a0 + 0.5 * dcav), zB + 0.07, r"$\lambda/4$", ha="center", va="bottom", fontsize=7.4)
a.set_xlim(-Lx - 0.62, Lx + 0.62); a.set_ylim(zB - k * Ly - 0.12, zA + k * Ly + 0.33)
a.set_aspect("equal"); a.axis("off")
ha = [Line2D([], [], color=COL["blue"], lw=1.8), Line2D([], [], color=COL["blue"], lw=0.7, ls=":"),
      Patch(fc=COL["orange"], ec="none"), Patch(fc=COL["h2"], ec="none"), Line2D([], [], color=COL["hg"], lw=3.2)]
la = ["thin shell " + r"$\Sigma$" + " (idealized junction)", "identified rims", "junction layer, width " + r"$\ell$",
      "hydrogen filling", "mercury wall"]
a.legend(ha, la, ncol=2, **LEG)

# ---------------- (b) cross-section along the normal ----------------
b = ax_b
nH = M.H2["solid"]["nH"]
lam = 2 * np.pi / (nH * M.k0) * 100     # in-medium wavelength in cm (solid hydrogen)
xw = lam / 4
n = np.linspace(-xw, xw, 800)
b.add_patch(Rectangle((-xw, -1.32), 2 * xw, 2.64, fc=COL["h2"], ec="none", zorder=0))
for sgn in (-1, 1):
    x0 = xw if sgn > 0 else -xw - 0.55
    b.add_patch(Rectangle((x0, -1.32), 0.55, 2.64, fc=COL["hg"], ec=COL["black"], lw=0.5, hatch="////", zorder=1))
lay = np.sqrt(3) * M.ell * 100          # compact junction layer |x| <= a
b.add_patch(Rectangle((-lay, -1.32), 2 * lay, 2.64, fc=COL["orange"], ec="none", zorder=1))
b.plot(n, np.cos(2 * np.pi * n / lam), color=COL["blue"], lw=1.3, label="electric field " + r"$E_y$", zorder=3)
b.plot(n, np.sin(2 * np.pi * n / lam), color=COL["red"], lw=1.2, ls="--", label="magnetic field " + r"$B_z$", zorder=3)
b.axvline(0, color=COL["black"], lw=0.9, zorder=2)
# reactive sheet of the passive loaded cavity at x = +-d, d = 5 mm, outside the junction layer (assumption vi)
xe = M.res["loaded"]["d"] * 100        # cm
for sgn in (-1, 1):
    b.add_patch(Rectangle((sgn * xe - 0.055, -1.32), 0.11, 2.64, fc=COL["purple"], ec="none", alpha=0.85, zorder=2))
# direction triad
ox, oy = 1.75, -1.0
b.add_patch(FancyArrowPatch((ox, oy), (ox + 1.2, oy), arrowstyle="-|>", mutation_scale=7, color=COL["black"], lw=0.8, zorder=4))
b.add_patch(FancyArrowPatch((ox, oy), (ox, oy + 0.5), arrowstyle="-|>", mutation_scale=7, color=COL["black"], lw=0.8, zorder=4))
b.text(ox + 1.28, oy, r"$x$", ha="left", va="center", fontsize=7.4)
b.text(ox, oy + 0.55, r"$y$", ha="center", va="bottom", fontsize=7.4)
b.plot([ox - 0.42], [oy + 0.02], marker="o", ms=4.2, mfc="white", mec=COL["black"], mew=0.7, zorder=4)
b.plot([ox - 0.42], [oy + 0.02], marker=".", ms=2.2, color=COL["black"], zorder=5)
b.text(ox - 0.42, oy - 0.13, r"$z$", ha="center", va="top", fontsize=7.4)
b.set_xlim(-xw - 0.55, xw + 0.55); b.set_ylim(-1.32, 1.32)
b.set_xticks([-3, -2, -1, 0, 1, 2, 3]); b.set_yticks([-1, 0, 1])
b.set_xlabel(r"proper normal distance $x$ (cm)"); b.set_ylabel("mode function")
hb, lb = b.get_legend_handles_labels()
hb += [Line2D([], [], color=COL["black"], lw=0.9), Patch(fc=COL["orange"], ec="none"),
       Patch(fc=COL["h2"], ec="none"), Patch(fc=COL["hg"], ec=COL["black"], lw=0.5, hatch="////")]
hb += [Patch(fc=COL["purple"], ec="none", alpha=0.85)]
lb += ["thin shell " + r"$\Sigma$", "junction layer", "hydrogen", "mercury", "passive sheet"]
b.legend(hb, lb, ncol=3, **LEG)

# ---------------- (c) polarization completion ----------------
d = ax_c
NEG, POS, ZERO = (COL["cyan"], 0.55), (COL["red"], 0.40), ("white", 1.0)


def cellcolor(v):
    return NEG if v < -1e-9 else (POS if v > 1e-9 else ZERO)


rows = [("mode along " + r"$y$", (A, B, -B)), ("mode along " + r"$z$", (A, -B, B)), ("both modes", (2 * A, 0.0, 0.0))]
# (c) is the vacuum reference configuration C1; (b) shows the hydrogen-filled geometry of C2 and C4
cw, chh = 1.25, 0.62
ys = [2.0, 1.0, -0.15]
for (name, vals), y in zip(rows, ys):
    for jx, v in enumerate(vals):
        fc, al = cellcolor(v)
        d.add_patch(Rectangle((jx * cw, y), cw, chh, fc=fc, alpha=al, ec=COL["black"], lw=0.6))
        d.text((jx + 0.5) * cw, y + 0.5 * chh, ("%+.3f" % v).replace("-", "\u2212") if abs(v) > 1e-9 else "0", ha="center", va="center", fontsize=7.4)
    d.text(-0.12, y + 0.5 * chh, name, ha="right", va="center", fontsize=6.9)
d.text(-0.12, 1.0 + chh + 0.19, "+", ha="right", va="center", fontsize=10)
d.plot([-0.05, 3 * cw + 0.05], [0.72, 0.72], color=COL["black"], lw=0.9)
d.text(-0.12, 0.72, "=", ha="right", va="center", fontsize=10)
for jx, lab in enumerate([r"$\sigma$", r"$p_y$", r"$p_z$"]):
    d.text((jx + 0.5) * cw, 2.0 + chh + 0.1, lab, ha="center", va="bottom", fontsize=8.5)
d.set_xlim(-1.75, 3 * cw + 0.12); d.set_ylim(-0.3, 3.05)
d.axis("off")
d.legend([Patch(fc=NEG[0], alpha=NEG[1], ec=COL["black"], lw=0.5), Patch(fc=POS[0], alpha=POS[1], ec=COL["black"], lw=0.5),
          Patch(fc="white", ec=COL["black"], lw=0.5)],
         ["negative", "positive", "vanishing"], ncol=3, **LEG)

panel_labels(fig, (ax_a, ax_b, ax_c), ("(a)", "(b)", "(c)"))
save(fig, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "fig1"))
print("A=%.4f B=%.4f 2A=%.4f lambda/4=%.3f cm" % (A, B, 2 * A, xw))
