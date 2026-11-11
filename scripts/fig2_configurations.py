"""Fig. 2: negative-energy generation across the four configurations, from the model.
(a) Completed surface null combination per unit 2 L_eff hbar omega / V against the squeezing amplitude
for C1-C4 at their operating temperatures, with the optima and the reference point r = 2 marked.
(b) NEC-violating region in the (r, T) plane for the bare and hydrogen-dressed localization factors,
against the temperatures fixed by the materials."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from figstyle import COL, apply_style, panel_labels, save, TEXTWIDTH_IN
import hg_model as M

apply_style()
LEG = dict(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=6.7, handlelength=1.9,
           columnspacing=1.0, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.3)

fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.33))
fig.subplots_adjust(wspace=0.25, left=0.085, right=0.985, top=0.735, bottom=0.175)

# ---------------- (a) null combination against r ----------------
rr = np.linspace(1e-3, 3.4, 1200)
spec = [("C1", COL["blue"], "-", "C1: ideal walls, " + r"$T=0$"),
        ("C2", COL["red"], "--", "C2: solid H" + r"$_2$" + " in layer, " + r"$T=0$"),
        ("C3", COL["green"], "-.", "C3: Hg walls, 10 mK"),
        ("C4", COL["magenta"], ":", "C4: Hg and solid H" + r"$_2$" + ", 10 mK")]
for key, col, ls, lab in spec:
    cf = M.out[key]
    y = cf["amp"] * M.H_T(rr, cf["T"], cf["G"])
    a.plot(rr, y, color=col, ls=ls, lw=1.35, label=lab)
    a.plot([cf["r_opt"]], [cf["amp"] * cf["H_opt"]], marker="o", ms=4.2, color=col, mec=COL["black"], mew=0.5, ls="none")
    a.plot([M.r_ref], [cf["amp"] * cf["H_r2"]], marker="s", ms=3.8, color=col, mec=COL["black"], mew=0.5, ls="none")
a.axhline(0, color=COL["black"], lw=0.8)
a.set_xlim(0, 3.4); a.set_ylim(-0.56, 0.12)
a.set_xlabel(r"squeezing amplitude $r$")
a.set_ylabel("electromagnetic-sector " + r"$(\sigma+p)_\perp/(2L_{\rm eff}\hbar\omega/V)$")
h, l = a.get_legend_handles_labels()
h += [Line2D([], [], marker="o", ms=4.2, color=COL["grey"], mec=COL["black"], mew=0.5, ls="none"),
      Line2D([], [], marker="s", ms=3.8, color=COL["grey"], mec=COL["black"], mew=0.5, ls="none"),
      Line2D([], [], color=COL["black"], lw=0.8)]
l += ["optimum over " + r"$r$", "reference point " + r"$r=2$", "NEC boundary"]
order = [0, 1, 2, 3, 4, 5, 6]
a.legend([h[i] for i in order], [l[i] for i in order], ncol=2, **LEG)

# ---------------- (b) the (r, T) plane ----------------
rb = np.linspace(0.02, 3.4, 1500)
Tb = np.array([M.T_nec(x, M.G0) for x in rb])
Ts = np.array([M.T_nec(x, M.G_sol) for x in rb])
Tlo, Thi = 1e-3, 1e3
b.fill_between(rb, Tlo, np.where(Tb > 0, Tb, Tlo), color=COL["cyan"], alpha=0.30, lw=0)
b.plot(rb[Tb > 0], Tb[Tb > 0], color=COL["blue"], lw=1.3)
b.plot(rb[Ts > 0], Ts[Ts > 0], color=COL["red"], lw=1.2, ls="--")
b.axhspan(13.8, 20.3, color=COL["orange"], alpha=0.35, lw=0)
b.axhspan(234.3, Thi, color=COL["grey"], alpha=0.25, lw=0)
b.axhline(M.Tc_Hg, color=COL["purple"], lw=1.0, ls="-.")
b.axhline(M.T_op, color=COL["green"], lw=1.0, ls=":")
b.plot([M.rTmax_bare], [M.Tmax_bare], marker="*", ms=8, color=COL["blue"], mec=COL["black"], mew=0.5, ls="none")
b.plot([M.out["C3"]["r_opt"]], [M.T_op], marker="o", ms=4.2, color=COL["green"], mec=COL["black"], mew=0.5, ls="none")
b.set_yscale("log")
b.set_xlim(0, 3.4); b.set_ylim(Tlo, Thi)
b.set_xlabel(r"squeezing amplitude $r$"); b.set_ylabel(r"temperature $T$ (K)")
hb = [(Patch(fc=COL["cyan"], alpha=0.30, ec="none"), Line2D([], [], color=COL["blue"], lw=1.3)),
      Line2D([], [], color=COL["red"], lw=1.2, ls="--"),
      Line2D([], [], marker="*", ms=8, color=COL["blue"], mec=COL["black"], mew=0.5, ls="none"),
      Line2D([], [], marker="o", ms=4.2, color=COL["green"], mec=COL["black"], mew=0.5, ls="none"),
      Line2D([], [], color=COL["green"], lw=1.0, ls=":"),
      Line2D([], [], color=COL["purple"], lw=1.0, ls="-."),
      Patch(fc=COL["orange"], alpha=0.35, ec="none"), Patch(fc=COL["grey"], alpha=0.25, ec="none")]
lb = ["NEC violated, vacuum layer", "boundary, solid H" + r"$_2$" + " layer",
      "maximum temperature " + r"$T_{\max}$", "C3 operating point", "operating temperature",
      "Hg superconducting transition", "liquid H" + r"$_2$" + " at 1 atm", "liquid Hg"]
b.legend(hb, lb, ncol=2, **LEG)

panel_labels(fig, (a, b), ("(a)", "(b)"))
save(fig, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "fig2"))
print("Tmax bare %.2f mK at r=%.3f" % (M.Tmax_bare * 1e3, M.rTmax_bare))
