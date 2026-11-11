"""Fig. 3: feasibility and the complete junction requirement, from the model.
(a) Conditional electromagnetic-sector null combination supplied by C1-C4 (single mode, optimal squeezing)
and the illustrative 10^6-mode scaling, the Ford-Roman free-field benchmark, the Israel requirement, and the
complete boundary-vacuum null combination of the passive loaded cavity (Lifshitz calculation, positive).
(b) Null combinations sigma + p of the computed sectors in the junction layer of the stationary-model
configurations: squeezed sector; boundary vacuum (Lifshitz: superconducting walls of C3 and the hydrogen surfaces
of the primed variants; exactly zero for the ideal vacuum cavity C1, marked 0; bounded by 2|sigma_C| for the
hydrogen-filled cavities, open squares with arrows); thermal radiation and hydrogen rest energy; sign of the sum. The sum omits the interaction of the
stationary model, so it is not the complete stress-energy.
(c) Reissner-Nordstrom exteriors: required pressure p/p_I and energy density |sigma|/|sigma_I| relative to the
flat-exterior values, against a0/r_+ - 1, for |Q|/M = 0, 0.9, 1 (Eq. (4); every solid curve stays >= 1)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import COL, apply_style, panel_labels, save, TEXTWIDTH_IN
import hg_model as M
import casimir_layer as CL
import rn_thin_shell as RN

apply_style()
LEG = dict(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=6.7, handlelength=1.5,
           columnspacing=1.0, handletextpad=0.45, borderaxespad=0.0, labelspacing=0.40)
lg = np.log10
req = lg(M.israel_nec)
CR = CL.results()

W, H = TEXTWIDTH_IN - 0.05, 2.26
fig = plt.figure(figsize=(W, H))
def axin(x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])
y0, h0 = 0.384, 1.403                               # axes bottom and height of revision 7
a = axin(0.80, y0, 1.60, h0)
b = axin(3.22, y0, 1.60, h0)
c = axin(5.42, y0, 1.70, h0)

# ---------------- (a) supplied against required ----------------
O = M.out
rows = [("Israel requirement", req, dict(marker="^", ms=6.5, color=COL["blue"])),
        ("Ford" + "–" + "Roman", lg(M.qei_sigma), dict(marker="D", ms=5, mfc="white", color=COL["green"])),
        ("C1, " + r"$10^6$" + " modes", lg(1e6 * abs(O["C1"]["em_nec_opt"])), dict(marker="s", ms=5, color=COL["purple"])),
        ("C1", lg(abs(O["C1"]["em_nec_opt"])), dict(marker="o", ms=5, color=COL["blue"])),
        ("C2", lg(abs(O["C2"]["em_nec_opt"])), dict(marker="o", ms=5, color=COL["red"])),
        ("C3", lg(abs(O["C3"]["em_nec_opt"])), dict(marker="o", ms=5, color=COL["green"])),
        ("C4", lg(abs(O["C4"]["em_nec_opt"])), dict(marker="o", ms=5, color=COL["magenta"])),
        ("passive cavity", lg(CR["P_nec"]), dict(marker="o", ms=5.5, mfc="white", color=COL["red"], mec=COL["red"], mew=1.1))]
x0 = -34                                            # both panels reach the superconducting-wall vacuum
for i, (name, v, st) in enumerate(rows):
    y = len(rows) - 1 - i
    a.plot([x0, v], [y, y], color=COL["lightgrey"], lw=1.0, zorder=1)
    kw = dict(mec=COL["black"], mew=0.5)
    kw.update(st)
    a.plot([v], [y], ls="none", zorder=3, **kw)
a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows][::-1])
a.tick_params(axis="y", which="both", length=0)
yC1 = len(rows) - 1 - 3
a.annotate("", xy=(req, yC1 + 0.45), xytext=(rows[3][1], yC1 + 0.45),
           arrowprops=dict(arrowstyle="<->", color=COL["black"], lw=0.7, shrinkA=0, shrinkB=0))
a.set_xlim(x0, 48); a.set_ylim(-0.6, len(rows) - 0.4)
a.set_xlabel(r"$\log_{10}\left(|\sigma+p|\,/\,{\rm J\,m^{-2}}\right)$")
a.annotate("%.1f decades" % (req - rows[3][1]), xy=(0.5 * (req + rows[3][1]), yC1 + 0.62),
           ha="center", va="bottom", fontsize=6.9)
a.text(14.0, 2.5, "conditional estimates,\ninteraction excluded",
       ha="center", va="center", fontsize=6.6, color=COL["black"], linespacing=1.25)
a.text(lg(CR["P_nec"]) + 3.0, 0.0, "complete vacuum, positive", ha="left", va="center", fontsize=6.6,
       color=COL["black"])

# ---------------- (b) computed sectors in the layer (interaction excluded) ----------------
cfgs = [("C1", "C1"), ("C2", "C2"), ("C2o", "C2" + r"$'$"), ("C3", "C3"), ("C4", "C4"), ("C4o", "C4" + r"$'$"),
        ("C4liq", "C4, liquid H" + r"$_2$")]
xb0 = x0
for i, (key, name) in enumerate(cfgs):
    y = len(cfgs) - 1 - i
    cf = O[key]
    b.plot([xb0, 48], [y, y], color=COL["lightgrey"], lw=0.8, zorder=1)
    em = cf["em_nec_opt"]
    if em < 0:
        b.plot([lg(abs(em))], [y], marker="o", ms=5, color=COL["blue"], mec=COL["black"], mew=0.5, ls="none", zorder=3)
    else:
        b.plot([lg(em)], [y], marker="o", ms=5, mfc="white", color=COL["red"], mec=COL["red"], mew=1.0, ls="none", zorder=3)
    # boundary vacuum on the null-combination axis: computed (filled), zero (C1), or bounded (hydrogen-filled)
    if key == "C1":
        assert CR["vac"]["C1"] < 1e-35                       # ideal walls in vacuum: sigma_C + p_C = 0
        b.text(xb0 + 0.9, y - 0.27, "0", fontsize=7.0, color=COL["grey"], ha="left", va="center", zorder=3)
    elif key in CR["vac"]:
        yv = y if key in ("C2o", "C4o") else y - 0.26
        b.plot([lg(CR["vac"][key])], [yv], marker="s", ms=4.8, color=COL["grey"], mec=COL["black"], mew=0.5, ls="none", zorder=3)
    else:
        vb = lg(2 * abs(cf["casimir_sigma"]))                # |sigma_C + p_C| <= 2|sigma_C| if |p_C| <= |sigma_C|
        b.annotate("", xy=(vb - 4.5, y - 0.26), xytext=(vb - 0.9, y - 0.26),
                   arrowprops=dict(arrowstyle="-|>", color=COL["grey"], lw=0.8, mutation_scale=5, shrinkA=0, shrinkB=0), zorder=3)
        b.plot([vb], [y - 0.26], marker="s", ms=4.4, mfc="white", mec=COL["grey"], mew=1.0, ls="none", zorder=3)
    if cf["thermal_nec"] > 0:
        b.plot([lg(cf["thermal_nec"])], [y + 0.26], marker="v", ms=5.0, color=COL["orange"], mec=COL["black"], mew=0.5, ls="none", zorder=3)
    if cf["matter_nec"] > 0:
        b.plot([lg(cf["matter_nec"])], [y], marker="D", ms=4.8, color=COL["red"], mec=COL["black"], mew=0.5, ls="none", zorder=3)
    net = CR["layer_sum"][key]
    b.plot([lg(abs(net))], [y], marker="o", ms=9.5, mfc="none", mec=COL["blue"] if net < 0 else COL["red"], mew=1.3, ls="none", zorder=4)
b.axvline(req, color=COL["blue"], lw=1.0, ls="--", zorder=2)
b.set_yticks(range(len(cfgs))); b.set_yticklabels([c[1] for c in cfgs][::-1])
b.tick_params(axis="y", which="both", length=0)
b.set_xlim(xb0, 48); b.set_ylim(-0.6, len(cfgs) - 0.4)
b.set_xlabel(r"$\log_{10}\left(|\sigma+p|\,/\,{\rm J\,m^{-2}}\right)$")
hb = [Line2D([], [], marker="o", ms=5, color=COL["blue"], mec=COL["black"], mew=0.5, ls="none"),
      Line2D([], [], marker="o", ms=5, mfc="white", color=COL["red"], mec=COL["red"], mew=1.0, ls="none"),
      Line2D([], [], marker="s", ms=4.8, color=COL["grey"], mec=COL["black"], mew=0.5, ls="none"),
      Line2D([], [], marker="s", ms=4.4, mfc="white", mec=COL["grey"], mew=1.0, ls="none"),
      Line2D([], [], marker="v", ms=5.0, color=COL["orange"], mec=COL["black"], mew=0.5, ls="none"),
      Line2D([], [], marker="D", ms=4.8, color=COL["red"], mec=COL["black"], mew=0.5, ls="none"),
      Line2D([], [], marker="o", ms=7.2, mfc="none", mec=COL["blue"], mew=1.2, ls="none"),
      Line2D([], [], marker="o", ms=7.2, mfc="none", mec=COL["red"], mew=1.2, ls="none"),
      Line2D([], [], color=COL["blue"], lw=1.0, ls="--")]
lb = ["squeezed sector, negative", "squeezed sector, positive", "boundary vacuum, positive",
      "boundary vacuum, upper bound", "thermal radiation, positive", "H" + r"$_2$" + " rest energy, positive",
      "sum of sectors, negative", "sum of sectors, positive", "Israel requirement"]
b.legend(hb, lb, ncol=2, **LEG)

# ---------------- (c) charged exteriors: complete Israel tensor relative to flat exteriors ----------------
xs = np.logspace(-4, 2, 600)
for qq, col in ((0.0, COL["blue"]), (0.9, COL["red"]), (1.0, COL["green"])):
    rp = 1 + np.sqrt(max(1 - qq ** 2, 0.0))
    aa = (1 + xs) * rp                              # throat radius in units of M
    sg, pg = RN.israel(aa, 1.0, qq)
    c.plot(xs, 4 * np.pi * aa * pg, color=col, lw=1.3, zorder=3)                    # p / p_I
    c.plot(xs, 2 * np.pi * aa * np.abs(sg), color=col, lw=1.3, ls="--", zorder=3)   # |sigma| / |sigma_I|
c.set_xscale("log"); c.set_yscale("log")
c.set_xlim(1e-4, 1e2); c.set_ylim(1e-2, 2e2)
c.set_xticks([1e-4, 1e-2, 1e0, 1e2]); c.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
c.set_xlabel(r"$a_0/r_+-1$")
c.set_ylabel("ratio to flat exterior")
blank = Line2D([], [], color="none", lw=0)
hc = [Line2D([], [], color=COL["blue"], lw=1.3), Line2D([], [], color=COL["red"], lw=1.3),
      Line2D([], [], color=COL["green"], lw=1.3),
      Line2D([], [], color=COL["black"], lw=1.3), Line2D([], [], color=COL["black"], lw=1.3, ls="--"), blank]
lc = [r"$|Q|/M=0$", r"$|Q|/M=0.9$", r"$|Q|/M=1$", r"$p/p_{\rm I}$", r"$|\sigma|/|\sigma_{\rm I}|$", ""]
c.legend(hc, lc, ncol=2, **LEG)

panel_labels(fig, (a, b, c), ("(a)", "(b)", "(c)"))
save(fig, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "fig3"))
print("separation C1 = %.2f decades; passive vacuum = %.2e; C2' sum = %.2e" % (req - rows[3][1], CR["P_nec"], CR["layer_sum"]["C2o"]))
