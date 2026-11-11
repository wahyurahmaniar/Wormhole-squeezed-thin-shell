"""Charged (Reissner-Nordstrom) exteriors for the symmetric thin-shell wormhole.

Derivations in geometric units G = c = 1 (lengths), converted to SI at the end.
Exterior on each side: ds^2 = -f dt^2 + dr^2/f + r^2 dOmega^2, f = 1 - 2M/r + Q^2/r^2,
shell at r = a(tau).  Lanczos: S^i_j = -(1/8 pi)([K^i_j] - delta^i_j [K]), [K] = 2K (mirror symmetry).

Results checked here (symbolically, then numerically):
  (C1) sigma = -sqrt(f)/(2 pi a),  p = (sqrt(f)/a + f'/(2 sqrt f))/(4 pi)          (static Israel tensor)
  (C2) RN: p = (1 - M/a)/(4 pi a sqrt f),  (1 - M/a)^2 - f = (M^2 - Q^2)/a^2
       => p >= 1/(4 pi a) for every exterior with a horizon or extremal (M >= |Q|), equality iff |Q| = M
  (C3) sigma + p = (a f' - 2 f)/(8 pi a sqrt f): zero on the photon sphere, positive inside it
  (C4) ANEC through the shell along the radial null geodesic of energy E:  int T_kk dlambda = -E/(2 pi a),
       for every f (the charge cannot reduce it)
  (C5) a source with sigma < 0 and |p| < 1/(4 pi a) needs |Q| > M, with the exact relation
       (Q^2 - M^2)/a^2 = (2 pi a sigma)^2 [1 - (4 pi a p)^2]
  (C6) over-extremal minimum of f at a_m = Q^2/M: f_min = 1 - M^2/Q^2 = eps^2,
       sigma(a_m) = -eps/(2 pi a_m), p(a_m) = eps/(4 pi a_m); p(M) = 0
  (C8) the over-extremal throat is long: radial light crossing takes t = pi M^2/(c sqrt(Q^2 - M^2)) ~ pi a0/(eps c)
  (C7) linear stability: V = f - (2 pi a sigma)^2 with sigma' = -2(sigma + p)/a, dp/dsigma = beta^2:
       V(a0) = V'(a0) = 0 and V''(a0) = f'' - 8 pi^2 [(sigma + 2p)^2 + sigma (sigma + p)(2 + 4 beta^2)]
       flat exteriors: V'' = -(2 + 4 beta^2)/a^2, stable iff beta^2 < -1/2 (Poisson-Visser)
       at a_m: V'' = (2 M^2/Q^4)(1 - 2 eps^2 (1 + beta^2))
  (C9) the p = 0 shell of an over-extremal exterior, Q^2 = M^2 (1 + eps^2): a0 = M, f(M) = eps^2,
       sigma = eps sigma_I, and V''(M) = (2 - 8 eps^2 beta^2)/M^2, stable iff beta^2 < (sigma_I/sigma)^2/4
  (C10) energy accounting with redshift. The first junction condition at an instant of rest,
       -4 pi a sigma = sqrt f_+ + sqrt f_-, gives for a local energy dm added to the shell at fixed a
       dm = dM_+/sqrt f_+ + dM_-/sqrt f_-  (first order), so dM_+ + dM_- = sqrt f dm for equal exteriors:
       a quantum of local energy counts with its Killing (redshifted) energy.
  (C11) energy from infinity: Killing energy E is conserved during infall and, by Birkhoff, the exterior
       between the shell and infinity of the side it came from has mass M + E after absorption. At the
       p = 0 shell, f_+(M) = eps^2 - 2E/M, so for E > eps^2 M/2 = |Q| - M + O(eps^4) the shell lies inside
       the new outer horizon r_+ = M + E + sqrt((M+E)^2 - Q^2) > a0 of that side.
  (C12) exact static family at fixed a: M(sigma) = (a/2)[1 + Q^2/a^2 - (2 pi a sigma)^2]; the margin closes
       only when 2 pi a |sigma| falls to |1 - |Q|/a|, i.e. when added local energy nearly cancels the shell.
"""
import os, sys, json
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hg_model as M

HBAR, C, G, EPS0, MU0, KB = M.HBAR, M.C, M.G, M.EPS0, M.MU0, M.KB
H = 2 * np.pi * HBAR
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))


# ------------------------------------------------------------------ symbolic part
def symbolic():
    r, a, Mm, Q, E, beta2 = sp.symbols("r a M Q E beta2", real=True)
    a_p = sp.symbols("a", positive=True)
    ffun = sp.Function("f")
    # static extrinsic curvature for normal n^r = sqrt(f): K_ij = (1/2) n^r d_r g_ij
    f = ffun(r)
    g_tt, g_th = -f, r ** 2
    nr = sp.sqrt(f)
    Ktt_up = (1 / g_tt) * sp.Rational(1, 2) * nr * sp.diff(g_tt, r)
    Kth_up = (1 / g_th) * sp.Rational(1, 2) * nr * sp.diff(g_th, r)
    Kjump_tt, Kjump_th = 2 * Ktt_up, 2 * Kth_up
    Ktr = Kjump_tt + 2 * Kjump_th
    S_tt = -(Kjump_tt - Ktr) / (8 * sp.pi)
    S_th = -(Kjump_th - Ktr) / (8 * sp.pi)
    sigma_g = sp.simplify(-S_tt)
    p_g = sp.simplify(S_th)
    sig_ref = -sp.sqrt(f) / (2 * sp.pi * r)
    p_ref = (sp.sqrt(f) / r + sp.diff(f, r) / (2 * sp.sqrt(f))) / (4 * sp.pi)
    check("C1 Israel sigma (static, general f)", sp.simplify(sigma_g - sig_ref) == 0)
    check("C1 Israel p (static, general f)", sp.simplify(p_g - p_ref) == 0)
    check("C1 flat limit sigma+p = -1/(4 pi a)",
          sp.simplify((sig_ref + p_ref).subs(ffun(r), 1).doit() + 1 / (4 * sp.pi * r)) == 0)

    # RN
    fRN = 1 - 2 * Mm / r + Q ** 2 / r ** 2
    sig = -sp.sqrt(fRN) / (2 * sp.pi * r)
    p = (sp.sqrt(fRN) / r + sp.diff(fRN, r) / (2 * sp.sqrt(fRN))) / (4 * sp.pi)
    check("C2 f + r f'/2 = 1 - M/r", sp.simplify(fRN + r * sp.diff(fRN, r) / 2 - (1 - Mm / r)) == 0)
    check("C2 p = (1 - M/r)/(4 pi r sqrt f)", sp.simplify(p - (1 - Mm / r) / (4 * sp.pi * r * sp.sqrt(fRN))) == 0)
    check("C2 (1 - M/r)^2 - f = (M^2 - Q^2)/r^2", sp.simplify((1 - Mm / r) ** 2 - fRN - (Mm ** 2 - Q ** 2) / r ** 2) == 0)
    fext = (1 - Mm / r) ** 2
    pext = (sp.sqrt(fext) / r + sp.diff(fext, r) / (2 * sp.sqrt(fext))) / (4 * sp.pi)
    pext = sp.simplify(pext.subs(sp.sqrt((1 - Mm / r) ** 2), 1 - Mm / r))
    check("C2 extremal: p = 1/(4 pi r) identically", sp.simplify(pext - 1 / (4 * sp.pi * r)) == 0)

    # NEC on the shell and photon sphere
    snp = sp.simplify(sig + p)
    check("C3 sigma+p = (r f' - 2 f)/(8 pi r sqrt f)",
          sp.simplify(snp - (r * sp.diff(fRN, r) - 2 * fRN) / (8 * sp.pi * r * sp.sqrt(fRN))) == 0)
    check("C3 sigma+p = -(r^2 - 3 M r + 2 Q^2)/(4 pi r^3 sqrt f)",
          sp.simplify(snp + (r ** 2 - 3 * Mm * r + 2 * Q ** 2) / (4 * sp.pi * r ** 3 * sp.sqrt(fRN))) == 0)
    ps = sp.simplify(sp.numer(sp.together(sp.diff(fRN / r ** 2, r) * r ** 5 / 2)))
    check("C3 photon sphere d(f/r^2)/dr = 0 <-> r^2 - 3 M r + 2 Q^2 = 0",
          sp.simplify(ps + (r ** 2 - 3 * Mm * r + 2 * Q ** 2)) == 0 or sp.simplify(ps - (r ** 2 - 3 * Mm * r + 2 * Q ** 2)) == 0,
          str(ps))

    # ANEC through the shell, radial null geodesic: k_t = -E, k.u = -E/sqrt(f) (u^t = 1/sqrt f),
    # T_kk = S_ab k^a k^b delta(l) = sigma (k.u)^2 delta(l); dl = (E/sqrt f) dlambda  (dr/dlambda = E, dl = dr/sqrt f)
    ku = -E / sp.sqrt(ffun(r))
    dl_dlam = E / sp.sqrt(ffun(r))
    anec = sig_ref * ku ** 2 / dl_dlam
    check("C4 int T_kk dlambda = -E/(2 pi a) for every f", sp.simplify(anec + E / (2 * sp.pi * r)) == 0)

    # C5: given (sigma, p) of the source
    sg, pp = sp.symbols("sigma_s p_s", real=True)
    u, w = sp.symbols("u w", real=True)        # u = M/a, w = Q/a
    f_uw = 1 - 2 * u + w ** 2
    # sqrt f = -2 pi a sigma, 1 - u = 4 pi a p sqrt f  (a = 1 by scaling)
    sqf = -2 * sp.pi * sg
    u_sol = 1 - 4 * sp.pi * pp * sqf
    w2_sol = sp.expand(sqf ** 2 - 1 + 2 * u_sol)
    lhs = sp.expand(w2_sol - u_sol ** 2)
    rhs = sp.expand((2 * sp.pi * sg) ** 2 * (1 - (4 * sp.pi * pp) ** 2))
    check("C5 (Q^2 - M^2)/a^2 = (2 pi a sigma)^2 [1 - (4 pi a p)^2]", sp.simplify(lhs - rhs) == 0)

    # C6: over-extremal minimum
    am = Q ** 2 / Mm
    check("C6 f'(Q^2/M) = 0", sp.simplify(sp.diff(fRN, r).subs(r, am)) == 0)
    check("C6 f(Q^2/M) = 1 - M^2/Q^2", sp.simplify(fRN.subs(r, am) - (1 - Mm ** 2 / Q ** 2)) == 0)
    eps = sp.symbols("epsilon", positive=True)
    Qe = Mm / sp.sqrt(1 - eps ** 2)
    Mpos = sp.symbols("M", positive=True)
    sub = {Mm: Mpos, Q: Mpos / sp.sqrt(1 - eps ** 2)}
    sig_m = sp.simplify(sig.subs(r, am).subs(sub))
    p_m = sp.simplify(p.subs(r, am).subs(sub))
    am_s = sp.simplify(am.subs(sub))
    check("C6 sigma(a_m) = -eps/(2 pi a_m)", sp.simplify(sig_m + eps / (2 * sp.pi * am_s)) == 0, str(sig_m))
    check("C6 p(a_m) = eps/(4 pi a_m)", sp.simplify(p_m - eps / (4 * sp.pi * am_s)) == 0, str(p_m))
    check("C6 p(M) = 0", sp.simplify(((1 - Mm / r) / (4 * sp.pi * r * sp.sqrt(fRN))).subs(r, Mm)) == 0)

    # C7: stability potential with an equation of state p(sigma), dp/dsigma = beta^2
    x = sp.symbols("x", positive=True)
    s0, p0, f0, f1, f2 = sp.symbols("s0 p0 f0 f1 f2", real=True)
    sfun, pfun = sp.Function("s")(x), sp.Function("P")(x)
    V = ffun(x) - (2 * sp.pi * x * sfun) ** 2
    ds = -2 * (sfun + pfun) / x
    dV = sp.diff(V, x).subs(sp.Derivative(sfun, x), ds)
    d2V = sp.diff(sp.diff(V, x), x)
    d2s = sp.diff(ds, x).subs(sp.Derivative(pfun, x), beta2 * sp.Derivative(sfun, x)).subs(sp.Derivative(sfun, x), ds)
    d2V = d2V.subs(sp.Derivative(sfun, (x, 2)), d2s).subs(sp.Derivative(sfun, x), ds)
    rep = {sp.Derivative(ffun(x), (x, 2)): f2, sp.Derivative(ffun(x), x): f1, ffun(x): f0, sfun: s0, pfun: p0}
    dV_s = dV.subs(rep)
    d2V_s = sp.expand(d2V.subs(rep))
    target = f2 - 8 * sp.pi ** 2 * ((s0 + 2 * p0) ** 2 + s0 * (s0 + p0) * (2 + 4 * beta2))
    check("C7 V'' formula", sp.simplify(d2V_s - sp.expand(target)) == 0)
    stat = {s0: -sp.sqrt(f0) / (2 * sp.pi * x), p0: (sp.sqrt(f0) / x + f1 / (2 * sp.sqrt(f0))) / (4 * sp.pi)}
    check("C7 V'(a0) = 0 at the static Israel values", sp.simplify(dV_s.subs(stat)) == 0)
    check("C7 V(a0) = 0 at the static Israel values",
          sp.simplify((f0 - (2 * sp.pi * x * s0) ** 2).subs(stat)) == 0)
    flat = sp.simplify(target.subs(stat).subs({f0: 1, f1: 0, f2: 0}))
    check("C7 flat exteriors: V'' = -(2 + 4 beta^2)/a^2 (stable iff beta^2 < -1/2)",
          sp.simplify(flat + (2 + 4 * beta2) / x ** 2) == 0, str(flat))
    # V'' at the over-extremal minimum a_m
    f2m = sp.diff(fRN, r, 2).subs(r, am)
    Vm = target.subs({f2: f2m, s0: sig.subs(r, am), p0: p.subs(r, am)}).subs(sub)
    Vm_ref = (2 * Mpos ** 2 / (Mpos / sp.sqrt(1 - eps ** 2)) ** 4) * (1 - 2 * eps ** 2 * (1 + beta2))
    check("C7 V''(a_m) = (2 M^2/Q^4)(1 - 2 eps^2 (1 + beta^2))", sp.simplify(Vm - Vm_ref) == 0)
    # (C9) p = 0 shell: a0 = M for Q^2 = M^2 (1 + eps^2)
    sub9 = {Mm: Mpos, Q: Mpos * sp.sqrt(1 + eps ** 2)}
    check("C9 f(M) = eps^2", sp.simplify(fRN.subs(r, Mm).subs(sub9) - eps ** 2) == 0)
    sig9 = sp.simplify(sig.subs(r, Mm).subs(sub9))
    check("C9 sigma(M) = -eps/(2 pi M) = eps sigma_I", sp.simplify(sig9 + eps / (2 * sp.pi * Mpos)) == 0, str(sig9))
    p9 = sp.simplify(p.subs(r, Mm).subs(sub9))
    check("C9 p(M) = 0", sp.simplify(p9) == 0, str(p9))
    f2M = sp.simplify(sp.diff(fRN, r, 2).subs(r, Mm).subs(sub9))
    V9 = sp.simplify(target.subs({f2: f2M, s0: sig9, p0: 0}))
    check("C9 V''(M) = (2 - 8 eps^2 beta^2)/M^2", sp.simplify(V9 - (2 - 8 * eps ** 2 * beta2) / Mpos ** 2) == 0, str(V9))
    # independent route: integrate sigma' = -2 (sigma + p)/a with p = beta2 (sigma - sigma0), then V = f - (2 pi a sigma)^2
    aa_ = sp.symbols("aa", positive=True)
    s0_9 = -eps / (2 * sp.pi * Mpos)
    sig_a = s0_9 * (beta2 + (aa_ / Mpos) ** (-2 * (1 + beta2))) / (1 + beta2)
    p_a = beta2 * (sig_a - s0_9)
    check("C9 sigma(a) solves sigma' = -2 (sigma + p)/a",
          sp.simplify(sp.diff(sig_a, aa_) + 2 * (sig_a + p_a) / aa_) == 0)
    V_a = (fRN.subs(sub9)).subs(r, aa_) - (2 * sp.pi * aa_ * sig_a) ** 2
    check("C9 V(M) = V'(M) = 0", sp.simplify(V_a.subs(aa_, Mpos)) == 0 and sp.simplify(sp.diff(V_a, aa_).subs(aa_, Mpos)) == 0)
    check("C9 V''(M) from the integrated equation of state",
          sp.simplify(sp.diff(V_a, aa_, 2).subs(aa_, Mpos) - (2 - 8 * eps ** 2 * beta2) / Mpos ** 2) == 0)
    # (C10) first-order energy accounting from the first junction condition at an instant of rest
    Mp, Mm_, aS = sp.symbols("M_p M_m a_s", positive=True)
    fp = 1 - 2 * Mp / aS + Q ** 2 / aS ** 2
    fm = 1 - 2 * Mm_ / aS + Q ** 2 / aS ** 2
    sig2 = -(sp.sqrt(fp) + sp.sqrt(fm)) / (4 * sp.pi * aS)
    dm_dMp = sp.simplify(4 * sp.pi * aS ** 2 * sp.diff(sig2, Mp))
    dm_dMm = sp.simplify(4 * sp.pi * aS ** 2 * sp.diff(sig2, Mm_))
    check("C10 dm = dM_+/sqrt f_+ + dM_-/sqrt f_- (first junction condition, fixed a)",
          sp.simplify(dm_dMp - 1 / sp.sqrt(fp)) == 0 and sp.simplify(dm_dMm - 1 / sp.sqrt(fm)) == 0,
          "%s, %s" % (dm_dMp, dm_dMm))
    # (C11) energy E from infinity raises M_+ to M + E; f_+(M) at the p = 0 shell
    EE = sp.symbols("E", positive=True)
    fplus = (1 - 2 * (Mpos + EE) / r + Q ** 2 / r ** 2).subs(Q, Mpos * sp.sqrt(1 + eps ** 2)).subs(r, Mpos)
    check("C11 f_+(M) = eps^2 - 2E/M after absorbing E", sp.simplify(fplus - (eps ** 2 - 2 * EE / Mpos)) == 0, str(sp.simplify(fplus)))
    marg = sp.series(Mpos * sp.sqrt(1 + eps ** 2) - Mpos, eps, 0, 5).removeO()
    check("C11 |Q| - M = eps^2 M/2 - eps^4 M/8 + ...", sp.simplify(marg - (eps ** 2 * Mpos / 2 - eps ** 4 * Mpos / 8)) == 0, str(marg))
    # (C12) exact static symmetric family at fixed a: M(sigma), and the closing point of the margin
    Mfam = (aS / 2) * (1 + Q ** 2 / aS ** 2 - (2 * sp.pi * aS * sg) ** 2)
    check("C12 static family: f(a) = (2 pi a sigma)^2 <-> M(sigma)",
          sp.simplify((1 - 2 * Mfam / aS + Q ** 2 / aS ** 2) - (2 * sp.pi * aS * sg) ** 2) == 0)
    # at M = |Q|: (2 pi a sigma)^2 = (1 - |Q|/a)^2
    Qp = sp.symbols("Q_p", positive=True)
    closing = sp.solve(sp.Eq(Mfam.subs(Q, Qp), Qp), sg)
    check("C12 margin closes at |2 pi a sigma| = |1 - |Q|/a|",
          any(sp.simplify((2 * sp.pi * aS * c_) ** 2 - (1 - Qp / aS) ** 2) == 0 for c_ in closing), str(closing))
    # (C8) radial light-crossing time of the over-extremal well: dt = dr/(c f), f = ((r-M)^2 + d^2)/r^2, d^2 = Q^2 - M^2
    xx, dd, LL = sp.symbols("x d L", positive=True)
    integrand = (xx + Mpos) ** 2 / (xx ** 2 + dd ** 2)
    anti = xx + Mpos * sp.log(xx ** 2 + dd ** 2) + (Mpos ** 2 - dd ** 2) / dd * sp.atan(xx / dd)
    check("C8 antiderivative of r^2/((r-M)^2 + d^2)", sp.simplify(sp.diff(anti, xx) - integrand) == 0)
    full = sp.simplify(anti.subs(xx, LL) - anti.subs(xx, -LL))
    check("C8 crossing integral = 2L + 2 (M^2 - d^2)/d atan(L/d) -> pi M^2/d for d << L",
          sp.simplify(full - (2 * LL + 2 * (Mpos ** 2 - dd ** 2) / dd * sp.atan(LL / dd))) == 0)


# ------------------------------------------------------------------ numerical part
def f_RN(a, Mg, Qg):
    """f = 1 - 2M/a + Q^2/a^2 in a form free of cancellation near a horizon."""
    a, Mg, Qg = np.broadcast_arrays(np.asarray(a, float), np.asarray(Mg, float), np.abs(np.asarray(Qg, float)))
    sub = Mg >= Qg
    disc = np.sqrt(np.abs((Mg - Qg) * (Mg + Qg)))
    f_sub = (a - (Mg + disc)) * (a - (Mg - disc)) / a ** 2
    f_over = ((a - Mg) / a) ** 2 + (Qg - Mg) * (Qg + Mg) / a ** 2
    return np.where(sub, f_sub, f_over)


def israel(a, Mg, Qg):
    """Static Israel (sigma, p) in geometric units (1/length) for RN exteriors."""
    sq = np.sqrt(f_RN(a, Mg, Qg))
    return -sq / (2 * np.pi * a), ((a - Mg) / a) / (4 * np.pi * a * sq)


def numeric():
    out = {}
    rng = np.random.default_rng(1)
    # (C2) sweep over every exterior with a horizon or extremal, shell outside r_+
    q = rng.uniform(0, 1, 200000); q[:2000] = 1.0
    x = 1 + 10 ** rng.uniform(-8, 3, q.size)               # a/r_+
    Mg = 1.0; Qg = q * Mg
    rp = Mg + np.sqrt(np.maximum(Mg ** 2 - Qg ** 2, 0))
    a = x * rp
    s, p = israel(a, Mg, Qg)
    pn = 4 * np.pi * a * p
    out["C2_min_4pi_a_p"] = float(pn.min())
    check("C2 numeric: 4 pi a p >= 1 for M >= |Q| (2e5 samples)", pn.min() >= 1 - 1e-9, "min = %.12f" % pn.min())
    ext = q == 1.0
    check("C2 numeric: extremal 4 pi a p = 1", np.allclose(pn[ext], 1, rtol=1e-9))
    # (C4) ANEC invariance numerically
    anec = s * 1.0 / np.sqrt(f_RN(a, Mg, Qg))
    check("C4 numeric: sigma/sqrt f = -1/(2 pi a)", np.allclose(anec * 2 * np.pi * a, -1, rtol=1e-10))
    # (C3) tangential NEC satisfied inside the photon sphere
    rps = (3 * Mg + np.sqrt(9 * Mg ** 2 - 8 * Qg ** 2)) / 2
    inside = a < rps * (1 - 1e-9); outside = a > rps * (1 + 1e-9)
    check("C3 numeric: sigma+p > 0 inside the photon sphere", np.all((s + p)[inside] > 0))
    check("C3 numeric: sigma+p < 0 outside the photon sphere", np.all((s + p)[outside] < 0))

    # SI conversions
    a0 = M.a0
    Pl = C ** 4 / G                                 # N; sigma_SI = sigma_geo * Pl
    out["c4_over_4piGa0"] = Pl / (4 * np.pi * a0)
    out["anec_SI"] = -Pl / (2 * np.pi * a0)          # J/m^2 per unit E
    # example table: sub-extremal / extremal
    rows = []
    for qq in (0.0, 0.9, 1.0):
        for xx in (1.001, 1.5, 3.0, 10.0):
            rpp = 1 + np.sqrt(max(1 - qq ** 2, 0))
            aa = xx * rpp                            # in units of M
            sg, pg = israel(aa, 1.0, qq)
            # rescale to a0 = 1 m: geometric quantities scale as 1/length
            scale = aa / a0                          # a0 (m) = aa * M  -> M = a0/aa
            rows.append(dict(QoverM=qq, a_over_rplus=xx, sigma_SI=sg * scale * Pl, p_SI=pg * scale * Pl,
                             nec_SI=(sg + pg) * scale * Pl, p_over_flat=4 * np.pi * aa * pg,
                             p_over_abs_sigma=pg / abs(sg)))
    out["table_subextremal"] = rows

    # (C5)/(C6) over-extremal tuning for the per-mode squeezed source
    src = abs(M.out["C1"]["em_nec_opt"])                 # J/m^2, single-mode C1 at optimal squeezing
    out["source_per_mode"] = src
    eps = 2 * np.pi * G * a0 * src / C ** 4               # = 2 pi a sigma (geometric) for |sigma| = src
    out["eps_required"] = eps
    out["QQ_minus_MM_over_a2"] = eps ** 2
    out["Q_over_M_minus_1"] = eps ** 2 / 2
    Mc2 = a0 * C ** 4 / G                                # M c^2 for M_geo = a0
    dE = 0.5 * eps ** 2 * Mc2                            # energy that closes the margin |Q| - M (horizon forms)
    out["Mc2"] = Mc2
    out["dE_crit"] = dE
    hnu = H * M.f0
    out["photon_2GHz"] = hnu
    out["photon_over_dE"] = hnu / dE
    out["kT_10mK_over_dE"] = KB * 0.010 / dE
    out["hubble_photon"] = H * C / 1.37e26              # photon of wavelength ~ Hubble radius (J)
    out["hubble_photon_over_dE"] = out["hubble_photon"] / dE
    out["dt_uncertainty"] = HBAR / (2 * dE)
    out["age_universe_s"] = 13.80e9 * 3.15576e7          # 13.80 Gyr in s
    # throat field for |Q| ~ M at a0 ~ M (magnetic charge): B a0 = c/sqrt(4 pi eps0 G)
    Ba = C / np.sqrt(4 * np.pi * EPS0 * G)
    out["B_times_a"] = Ba
    out["B_at_a0"] = Ba / a0
    out["E_at_a0"] = Ba * C / a0
    out["magnetic_pressure_a0"] = (Ba / a0) ** 2 / (2 * MU0)
    out["c4_over_8piG_a0sq"] = C ** 4 / (8 * np.pi * G * a0 ** 2)
    Bc_Hg = 0.0412                                       # T, Kittel (412 G)
    out["Bc_Hg"] = Bc_Hg
    out["a0_min_Hg"] = Ba / Bc_Hg
    out["a0_min_Hg_ly"] = Ba / Bc_Hg / 9.4607e15
    out["M_min_Hg_kg"] = (Ba / Bc_Hg) * C ** 2 / G
    out["M_min_Hg_Msun"] = out["M_min_Hg_kg"] / 1.98847e30
    B_mat = 1.0e3                                        # T, generous material limit (destructive pulsed fields)
    out["B_material"] = B_mat
    out["a0_min_material"] = Ba / B_mat
    out["a0_min_material_ly"] = Ba / B_mat / 9.4607e15
    out["E_schwinger"] = (M.ME * C ** 2) ** 2 / (M.QE * HBAR * C)
    out["a0_min_schwinger"] = Ba * C / out["E_schwinger"]
    # light-crossing time and blueshift of the over-extremal throat (C8): t ~ pi M^2/(c sqrt(Q^2 - M^2)) ~ pi a0/(eps c)
    out["throat_length_eff"] = a0 / eps
    out["t_cross"] = np.pi * a0 / (eps * C)
    out["t_cross_over_age"] = out["t_cross"] / out["age_universe_s"]
    out["blueshift"] = 1 / eps
    # numerical check of the crossing formula at a moderate over-extremality
    from scipy.integrate import quad
    Mt, et = 1.0, 1e-3
    dt_ = et * Mt
    num = quad(lambda x_: (x_ + Mt) ** 2 / (x_ ** 2 + dt_ ** 2), -0.5, 0.5, points=[0.0], limit=400)[0]
    ana = 2 * 0.5 + 2 * (Mt ** 2 - dt_ ** 2) / dt_ * np.arctan(0.5 / dt_)
    check("C8 numeric crossing integral (eps = 1e-3)", abs(num / ana - 1) < 1e-8, "%.10e vs %.10e" % (num, ana))
    # stability at the over-extremal minimum: V'' > 0 unless beta^2 > (1 - 2 eps^2)/(2 eps^2)
    out["beta2_max_stable"] = (1 - 2 * eps ** 2) / (2 * eps ** 2)
    check("C7 numeric: over-extremal minimum stable for beta^2 < 1/(2 eps^2)", out["beta2_max_stable"] > 1e100)
    # (C9) p = 0 shell of the single mode at a0 = M: stable for zeta < (sigma_I/sigma)^2/4 = 1/(4 eps^2)
    out["zeta_max_p0"] = 1 / (4 * eps ** 2)
    check("C9 numeric: p = 0 shell stable for zeta < 1/(4 eps^2)", out["zeta_max_p0"] > 1e100)
    # photon-robust source: one photon closes the margin (eps^2/2) M c^2 for every |sigma| below |sigma_I| sqrt(2 h nu G/(a0 c^4))
    out["y_photon"] = 2 * hnu * G / (a0 * C ** 4)
    out["sigma_photon_robust"] = C ** 4 / (2 * np.pi * G * a0) * np.sqrt(out["y_photon"])
    # (C10)-(C12) redshift-consistent energy accounting for the single-mode p = 0 shell
    out["sqrt_f_throat"] = eps                                    # sqrt f = |sigma/sigma_I| at a0 = M
    out["margin_inf"] = dE                                        # (|Q| - M) c^4/G to leading order, Killing energy
    out["cavity_photon_killing"] = eps * hnu                      # Killing energy of one local 2 GHz quantum
    out["cavity_photon_over_margin"] = eps * hnu / dE
    out["local_to_close_one_sided"] = dE / eps                    # all of sqrt f dm on one side
    out["local_to_close_symmetric"] = 2 * dE / eps                # split equally between the sides
    out["quanta_one_sided"] = dE / eps / hnu
    out["quanta_symmetric"] = 2 * dE / eps / hnu
    out["shell_local_energy"] = 4 * np.pi * a0 ** 2 * src         # |m_s| = 4 pi a0^2 |sigma|
    out["shell_quanta"] = out["shell_local_energy"] / hnu
    check("C10 numeric: |m_s| = 2 eps M c^2 (shell energy = twice the symmetric closing energy)",
          abs(out["shell_local_energy"] / (2 * eps * Mc2) - 1) < 1e-12 and abs(out["local_to_close_symmetric"] / out["shell_local_energy"] - 0.5) < 1e-12)
    R_H = 1.37e26                                                 # Hubble radius c/H0 (m)
    out["lambda_max_from_infinity"] = H * C / dE
    out["lambda_max_over_RH"] = H * C / dE / R_H
    T_cmb = 2.725
    out["cmb_photon_mean"] = 2.701 * KB * T_cmb                   # mean energy of a CMB photon (J)
    out["cmb_photon_over_margin"] = out["cmb_photon_mean"] / dE
    # capture rate of CMB photons by a near-extremal mouth: critical impact parameter 4M, cross-section 16 pi M^2
    n_cmb = 4.11e8                                                # photons per m^3 at 2.725 K
    out["cmb_capture_rate"] = n_cmb * C * 16 * np.pi * a0 ** 2
    # static matter in the well: a static shell of rest mass m at radius r needs sqrt f_out = sqrt f_in - G m/(c^2 r) >= 0
    out["wall_mass_max"] = eps * a0 * C ** 2 / G                  # kg, with sqrt f_in = eps near the throat
    out["gram_killing_over_margin"] = eps * 1e-3 * C ** 2 / dE    # Killing energy of 1 g resting in the well / margin
    check("C11 numeric: static matter in the well is limited to eps a0 c^2/G (below 1e-40 kg)", out["wall_mass_max"] < 1e-40)
    return out


def results():
    CHECKS.clear()
    symbolic()
    res = numeric()
    return res, list(CHECKS)


if __name__ == "__main__":
    res, chk = results()
    nf = 0
    for n, ok, d in chk:
        print(("PASS " if ok else "FAIL ") + n + (("   [" + d + "]") if (d and not ok) else ""))
        nf += (not ok)
    for k, v in res.items():
        if k != "table_subextremal":
            print("%-24s %s" % (k, v))
    print("\nQ/M  a/r+    sigma(J/m2)   p(J/m2)    sigma+p     p/p_flat  p/|sigma|")
    for r_ in res["table_subextremal"]:
        print("%.1f %7.3f %11.3e %11.3e %11.3e %9.4f %10.3f" % (r_["QoverM"], r_["a_over_rplus"], r_["sigma_SI"], r_["p_SI"],
                                                          r_["nec_SI"], r_["p_over_flat"], r_["p_over_abs_sigma"]))
    print("failures:", nf)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_rn.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
