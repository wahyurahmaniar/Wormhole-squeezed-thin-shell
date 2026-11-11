"""Complete renormalized electromagnetic vacuum stress in the junction layer of planar layered cavities.

Lifshitz (scattering) theory at T = 0, imaginary frequency xi and in-plane wavenumber k,
kappa = sqrt(k^2 + xi^2/c^2), xi = c kappa t, k^2 = kappa^2 (1 - t^2).  In a vacuum gap |x| < g between two
identical planar stacks of reflection R_s(xi,k), R_p(xi,k) (seen from the gap, reference planes x = -g, +g),
the fluctuation-dissipation theorem with the Green tensor of the stack gives (derivation in the module notes):

  T00(x) = (hbar c/4 pi^2) int dkappa kappa^3 int_0^1 dt sum_s [(1 - t^2) L_s(x) - t^2 U_s]
  Txx    = -(hbar c/4 pi^2) int dkappa kappa^3 int_0^1 dt sum_s U_s                      (uniform)
  N(x)   = T00 + Tyy = (3 T00 - Txx)/2
         = (hbar c/8 pi^2) int dkappa kappa^3 int_0^1 dt sum_s [3 (1 - t^2) L_s(x) + (1 - 3 t^2) U_s]
  L_s(x) = R_s [e^{-2 kappa (g + x)} + e^{-2 kappa (g - x)}] / D_s,
  U_s    = 2 R_s^2 e^{-4 kappa g} / D_s,   D_s = 1 - R_s^2 e^{-4 kappa g}.

The surface stress follows from the top-hat window of the paper, half-width a = sqrt(3) ell, L_eff = ell:
  sigma = L_eff <T00>,  p = L_eff <Tyy>,  sigma + p = L_eff <N>,
with <e^{-2 kappa (g -+ x)}> = e^{-2 kappa g} sinh(2 kappa a)/(2 kappa a).

Stacks are described from the outer end toward the gap and reduced with bounded Moebius maps of the local
reflection rho (|rho| <= 1 for passive media at imaginary frequency), so perfect conductors are exact limits.

Material responses at imaginary frequency:
  mercury, superconducting: plasma model eps = 1 + (c/lam_L)^2 / xi^2 (lam_L = London depth);
  mercury, normal (check):  Drude eps = 1 + w_p^2 / (xi (xi + gamma)), gamma = eps0 w_p^2 / sigma_dc;
  solid hydrogen: Clausius-Mossotti with a single-oscillator polarizability alpha0/(1 + xi^2/w0^2);
  reactive sheet: in-plane surface conductivity sigma_s(i xi) = xi C_sq/(1 + xi^2/w_s^2) (Lorentz cutoff w_s).
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hg_model as M

HBAR, C, EPS0, MU0 = M.HBAR, M.C, M.EPS0, M.MU0
QE = M.QE


# ---------------------------------------------------------------- materials
def eps_plasma(lam):
    wp = C / lam
    return lambda xi: 1.0 + wp ** 2 / xi ** 2


def eps_drude(wp, sigma_dc):
    gam = EPS0 * wp ** 2 / sigma_dc
    return lambda xi: 1.0 + wp ** 2 / (xi * (xi + gam))


def eps_h2(phase="solid", hw0_eV=15.0):
    n = M.H2[phase]["n"]
    w0 = hw0_eV * QE / HBAR
    def f(xi):
        x = n * M.alphaH2 / (3 * EPS0) / (1.0 + (xi / w0) ** 2)
        return (1 + 2 * x) / (1 - x)
    return f


def sheet_C(Csq, ws=np.inf):
    if np.isinf(ws):
        return lambda xi: Csq + 0.0 * xi
    return lambda xi: Csq / (1.0 + (xi / ws) ** 2)


# ---------------------------------------------------------------- stack reflection
def stack_R(xi, k, stack):
    """Reflection coefficients (R_s, R_p) seen from the vacuum gap.

    stack: list from the outer end to the gap.  First element is ('perfect',) or ('half', eps_fn);
    then any sequence of ('slab', eps_fn or None for vacuum, thickness) and ('sheet', Csq_fn);
    the last medium must be vacuum (the gap).  xi and k are broadcastable arrays."""
    xi = np.asarray(xi, float); k = np.asarray(k, float)
    kap = lambda e: np.sqrt(k ** 2 + e * xi ** 2 / C ** 2)
    one = lambda xi_: 1.0 + 0.0 * xi_
    head = stack[0]
    if head[0] == "perfect":
        eps_cur = one(xi); rs = -np.ones(np.broadcast(xi, k).shape); rp = np.ones_like(rs)
        started_in_vacuum = True
    elif head[0] == "half":
        eps_cur = head[1](xi) * one(xi); rs = np.zeros(np.broadcast(xi, k).shape); rp = np.zeros_like(rs)
        started_in_vacuum = False
    else:
        raise ValueError(head)
    for el in stack[1:]:
        if el[0] == "slab":
            e_new = (el[1](xi) if el[1] is not None else one(xi)) * one(xi)
            # interface from current medium (eps_cur) into the slab medium (e_new)
            k1, k2 = kap(eps_cur), kap(e_new)
            rs = (k2 * (1 + rs) - k1 * (1 - rs)) / (k2 * (1 + rs) + k1 * (1 - rs))
            rp = (eps_cur * k2 * (1 + rp) - e_new * k1 * (1 - rp)) / (eps_cur * k2 * (1 + rp) + e_new * k1 * (1 - rp))
            eps_cur = e_new
            h = el[2]
            if h > 0:
                ph = np.exp(-2 * kap(eps_cur) * h)
                rs = rs * ph; rp = rp * ph
        elif el[0] == "sheet":
            Cs = el[1](xi)
            kk = kap(eps_cur)
            s = MU0 * Cs * xi ** 2                       # TE: jump of psi'/psi
            rs = (2 * kk * rs - s * (1 + rs)) / (2 * kk + s * (1 + rs))
            cm = Cs / (EPS0 * eps_cur)                   # TM: jump of phi / phi'
            rp = (2 * rp + cm * kk * (1 - rp)) / (2 + cm * kk * (1 - rp))
        else:
            raise ValueError(el)
    # final interface into the vacuum gap if the last medium is not vacuum
    e_gap = one(xi)
    if not np.allclose(eps_cur, 1.0):
        k1, k2 = kap(eps_cur), kap(e_gap)
        rs = (k2 * (1 + rs) - k1 * (1 - rs)) / (k2 * (1 + rs) + k1 * (1 - rs))
        rp = (eps_cur * k2 * (1 + rp) - e_gap * k1 * (1 - rp)) / (eps_cur * k2 * (1 + rp) + e_gap * k1 * (1 - rp))
    return rs, rp


# ---------------------------------------------------------------- quadrature
def _gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def _kappa_nodes(kmin, kmax, per_decade=24, order=12):
    edges = np.logspace(np.log10(kmin), np.log10(kmax), int(np.ceil(np.log10(kmax / kmin) * per_decade / order)) + 1)
    xg, wg = _gl(order)
    nodes, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        u0, u1 = np.log(lo), np.log(hi)
        u = 0.5 * (u1 - u0) * xg + 0.5 * (u1 + u0)
        nodes.append(np.exp(u)); weights.append(0.5 * (u1 - u0) * wg * np.exp(u))   # dkappa = kappa du
    return np.concatenate(nodes), np.concatenate(weights)


def _t_nodes(n_panels=24, order=12, power=4):
    """Nodes on t in (0,1), clustered at t = 0 by t = s^power (resolves low-frequency structure)."""
    xg, wg = _gl(order)
    edges = np.linspace(0, 1, n_panels + 1)
    s = np.concatenate([0.5 * (b - a) * xg + 0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])])
    ws = np.concatenate([0.5 * (b - a) * wg for a, b in zip(edges[:-1], edges[1:])])
    return s ** power, ws * power * s ** (power - 1)


def layer_stress(stack, g, a=None, Leff=None, x=None, kmin=None, kmax=None, res=1):
    """Projected (sigma, p, sigma+p) in J/m^2 over |x| <= a, or local (T00, Tyy, N) in J/m^3 at x.

    g: half-width of the vacuum gap; the stack describes each (identical) side."""
    a = M.a_layer if a is None else a
    Leff = M.Leff if Leff is None else Leff
    near = g - (a if x is None else np.max(np.abs(np.atleast_1d(x))))
    kmin = 1e-4 / g if kmin is None else kmin
    kmax = 45.0 / max(near, 1e-12) if kmax is None else kmax
    kn, kw = _kappa_nodes(kmin, kmax, per_decade=24 * res)
    tn, tw = _t_nodes(n_panels=24 * res)
    K, T = np.meshgrid(kn, tn, indexing="ij")
    XI = C * K * T; KP = K * np.sqrt(1 - T ** 2)
    rs, rp = stack_R(XI, KP, stack)
    e4 = np.exp(-4 * K * g)
    out = {}
    tot_U = 0.0; tot_L = {}
    for R in (rs, rp):
        D = 1 - R ** 2 * e4
        U = 2 * R ** 2 * e4 / D
        tot_U = tot_U + U
    def integ(F):
        return np.einsum("i,ij,j->", kw * kn ** 3, F, tw)
    Txx = -HBAR * C / (4 * np.pi ** 2) * integ(tot_U)
    if x is None:
        # window average of e^{-2 kappa (g -+ x)} over |x| <= a
        av = (np.exp(-2 * K * (g - a)) - np.exp(-2 * K * (g + a))) / (4 * K * a)
        Lsum = 0.0
        for R in (rs, rp):
            D = 1 - R ** 2 * e4
            Lsum = Lsum + 2 * R * av / D
        T00 = HBAR * C / (4 * np.pi ** 2) * integ((1 - T ** 2) * Lsum - T ** 2 * tot_U)
        Tyy = 0.5 * (T00 - Txx)
        return dict(sigma=Leff * T00, p=Leff * Tyy, nec=Leff * (T00 + Tyy), T00=T00, Txx=Txx, Tyy=Tyy,
                    N=T00 + Tyy)
    xs = np.atleast_1d(x)
    res_ = []
    for xv in xs:
        loc = np.exp(-2 * K * (g + xv)) + np.exp(-2 * K * (g - xv))
        Lsum = 0.0
        for R in (rs, rp):
            D = 1 - R ** 2 * e4
            Lsum = Lsum + R * loc / D
        T00 = HBAR * C / (4 * np.pi ** 2) * integ((1 - T ** 2) * Lsum - T ** 2 * tot_U)
        res_.append((T00, 0.5 * (T00 - Txx), T00 + 0.5 * (T00 - Txx)))
    return np.array(res_), Txx


# ---------------------------------------------------------------- configurations of the paper
LD = M.res["loaded"]
Csq0 = LD["Cs"]; d_sheet = LD["d"]
L_vac = M.Lc
nH_s = M.H2["solid"]["nH"]
L_H2 = M.Lc / nH_s                      # hydrogen-filled half-wave cavity, walls at +-3.31 cm
gap_H2 = 2 * M.ell                      # vacuum gap half-width of the primed variants
HG = eps_plasma(M.lam_L)
H2s = eps_h2("solid")


def cfg_passive_vacuum(ws=np.inf, Csq=Csq0, d=d_sheet, wall=HG):
    head = ("perfect",) if wall is None else ("half", wall)
    return [head, ("slab", None, L_vac / 2 - d), ("sheet", sheet_C(Csq, ws))], d


def cfg_passive_h2(ws=np.inf, Csq=Csq0, d=d_sheet, wall=HG, h2=H2s):
    head = ("perfect",) if wall is None else ("half", wall)
    return [head, ("slab", h2, L_H2 / 2 - d), ("sheet", sheet_C(Csq, ws)), ("slab", h2, d - gap_H2)], gap_H2


def cfg_cavity_vacuum(wall=None):
    head = ("perfect",) if wall is None else ("half", wall)
    return [head, ("slab", None, 0.0)], L_vac / 2


def cfg_cavity_h2_gap(wall=None, h2=H2s):
    head = ("perfect",) if wall is None else ("half", wall)
    return [head, ("slab", h2, L_H2 / 2 - gap_H2)], gap_H2


def run_all(res=1):
    out = {}
    for name, (st, g) in dict(P=cfg_passive_vacuum(), P_H2=cfg_passive_h2(), C1=cfg_cavity_vacuum(None),
                              C3=cfg_cavity_vacuum(HG), C2p=cfg_cavity_h2_gap(None), C4p=cfg_cavity_h2_gap(HG)).items():
        out[name] = layer_stress(st, g, res=res)
        out[name]["g"] = g
    return out


def results():
    """Every number of the boundary-vacuum calculation quoted in the manuscript."""
    from scipy.optimize import brentq
    o = run_all()
    P = o["P"]
    xs = np.linspace(-M.a_layer, M.a_layer, 41)
    locP, _ = layer_stress(*cfg_passive_vacuum(), x=xs)
    C_scan = {f: layer_stress(*cfg_passive_vacuum(Csq=Csq0 * f))["nec"] for f in (0.1, 1.0, 10.0, 100.0)}
    C_scan_min_local = min(layer_stress(*cfg_passive_vacuum(Csq=Csq0 * f), x=xs)[0][:, 2].min()
                           for f in (0.1, 1.0, 10.0, 100.0))
    ws_change = abs(layer_stress(*cfg_passive_vacuum(ws=1e12))["nec"] / P["nec"] - 1)
    hw_change = abs(layer_stress([("perfect",), ("slab", eps_h2("solid", 7.5), L_H2 / 2 - gap_H2)], gap_H2)["nec"]
                    / layer_stress([("perfect",), ("slab", eps_h2("solid", 30.0), L_H2 / 2 - gap_H2)], gap_H2)["nec"] - 1)
    drude_change = abs(layer_stress(*cfg_passive_vacuum(wall=eps_drude(M.omega_p, M.sig_Hg_293)))["nec"] / P["nec"] - 1)

    def vac_gap(g):
        return layer_stress([("perfect",), ("slab", H2s, L_H2 / 2 - g)], g)["nec"]
    cross_C1 = brentq(lambda g: np.log(vac_gap(g) / abs(M.out["C1"]["em_nec_opt"])), 5e-3, 3e-2, xtol=1e-7)
    cross_C2o = brentq(lambda g: np.log(vac_gap(g) / abs(M.out["C2o"]["em_nec_opt"])), 5e-3, 3e-2, xtol=1e-7)
    vac = dict(C1=o["C1"]["nec"], C3=o["C3"]["nec"], C2o=o["C2p"]["nec"], C4o=o["C4p"]["nec"])
    layer_sum = {k: M.out[k]["total_nec_opt"] + vac.get(k, 0.0) for k in M.out}
    return dict(
        P_sigma=P["sigma"], P_p=P["p"], P_nec=P["nec"], P_T00=P["T00"], P_Txx=P["Txx"],
        P_ratio=P["nec"] / M.israel_nec, P_N_min=locP[:, 2].min(), P_N_max=locP[:, 2].max(),
        P_T00_min=locP[:, 0].min(), P_T00_max=locP[:, 0].max(),
        P_over_single_mode=P["nec"] / (2 * abs(M.res["loaded"]["sigma_vac"])),
        C_scan=C_scan, C_scan_min_local=C_scan_min_local, ws_change=ws_change, hw_change=hw_change,
        drude_change=drude_change, P_H2_nec=o["P_H2"]["nec"], vac=vac, layer_sum=layer_sum,
        dec_C2o=np.log10(vac["C2o"] / abs(M.out["C2o"]["em_nec_opt"])),
        dec_C4o=np.log10(vac["C4o"] / abs(M.out["C4o"]["em_nec_opt"])),
        cross_C1=cross_C1, cross_C2o=cross_C2o, edge_gap=gap_H2 - M.a_layer, edge_P=d_sheet - M.a_layer,
        ideal_check=layer_stress([("perfect",), ("slab", None, 0.0)], d_sheet))


if __name__ == "__main__":
    import json
    R = results()
    keep = {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items() if not isinstance(b, dict)})
            for k, v in R.items()}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "casimir_results.json"), "w") as fh:
        json.dump(keep, fh, indent=1, default=float)
    for k, v in R.items():
        if not isinstance(v, dict):
            print("%-22s %+.6e" % (k, v))
        else:
            print("%-22s %s" % (k, {a: ("%+.3e" % b if not isinstance(b, dict) else "...") for a, b in v.items()}))
    o = run_all()
    for k_, v in o.items():
        print("%-5s g=%.2f mm  sigma=%+.3e  p=%+.3e  sigma+p=%+.3e J/m^2   T00=%+.3e Txx=%+.3e J/m^3"
              % (k_, v["g"] * 1e3, v["sigma"], v["p"], v["nec"], v["T00"], v["Txx"]))
