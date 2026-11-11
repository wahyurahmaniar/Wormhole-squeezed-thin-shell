"""Symbolic (sympy), operator (truncated Fock space) and field (1D mode solver) verification of the
identities used in the manuscript."""
import sys, os
import numpy as np
import sympy as sp
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hg_model as M

fails = []


def report(name, ok):
    print("%-66s %s" % (name, "ok" if ok else "FAIL"))
    if not ok:
        fails.append(name)


def numzero(expr, syms, n=6, lo=0.2, hi=2.5):
    rng = np.random.default_rng(1)
    for _ in range(n):
        subs = {q: float(rng.uniform(0.1, 0.95)) if str(q) in ("G", "chi") else float(rng.uniform(lo, hi)) for q in syms}
        if abs(complex(expr.subs(subs))) > 1e-9:
            return False
    return True


r, G, a0, ell, rho, F, Gc, hbar, c, tau = sp.symbols("r G a_0 ell rho F G_N hbar c tau", positive=True)
s, ch = sp.sinh(r), sp.cosh(r)
A = s**2 - s * ch * G
B = s * ch - s**2 * G

# ---------------- single pair, completion, sign condition ----------------
report("single pair sigma+p = sinh r e^r (1-G)", numzero(A + B - s * sp.exp(r) * (1 - G), [r, G]))
report("single pair sigma+p_z = -sinh r e^-r (1+G)", numzero(A - B + s * sp.exp(-r) * (1 + G), [r, G]))
pair1 = sp.Matrix([A, B, -B]); pair2 = sp.Matrix([A, -B, B])
report("completion: (A,B,-B)+(A,-B,B) = (2A,0,0)", (pair1 + pair2 - sp.Matrix([2 * A, 0, 0])).equals(sp.zeros(3, 1)))
report("sign: A<0 iff tanh r < G", sp.simplify(A / (s * ch) - (sp.tanh(r) - G)) == 0)
k0l = sp.symbols("k_0l", positive=True)
report("window inversion k0 l = sqrt(-ln tanh r / 2)",
       sp.simplify(sp.solve(sp.Eq(sp.exp(-2 * k0l**2), sp.tanh(r)), k0l)[0] - sp.sqrt(-sp.log(sp.tanh(r)) / 2)) == 0)
H = A
report("H = (cosh2r - 1)/2 - (G/2) sinh 2r", numzero(H - ((sp.cosh(2 * r) - 1) / 2 - G * sp.sinh(2 * r) / 2), [r, G]))
report("optimum: dH/dr = sinh2r - G cosh2r", sp.simplify(sp.diff(H, r) - (sp.sinh(2 * r) - G * sp.cosh(2 * r))) == 0)
rstar = sp.atanh(G) / 2
report("optimum: H(r*) = -(1 - sqrt(1-G^2))/2", sp.simplify(sp.simplify(H.subs(r, rstar)) + (1 - sp.sqrt(1 - G**2)) / 2) == 0)
report("optimum: H(artanh G) = 0", sp.simplify(H.subs(r, sp.atanh(G))) == 0)

# ---------------- ratio identities ----------------
RNEC = 2 * ell * rho * F * sp.Abs(A) / (c**4 / (4 * sp.pi * Gc * a0))
report("R = (8 pi G a0/c^4) L rho F |H|", sp.simplify(RNEC - 8 * sp.pi * Gc * a0 / c**4 * ell * rho * F * sp.Abs(A)) == 0)
Rq = (4 * sp.pi * Gc * a0 / c**4) * ell * 3 * hbar / (16 * sp.pi**2 * c**3 * tau**4)
lP2 = Gc * hbar / c**3
report("QEI ceiling at tau = L/c: (3/4pi)(a0/L)(lP/L)^2",
       sp.simplify(Rq.subs(tau, ell / c) - sp.Rational(3, 4) / sp.pi * (a0 / ell) * lP2 / ell**2) == 0)
chi = sp.symbols("chi", positive=True)
report("Clausius-Mossotti Xi = 3chi/(chi+3)", sp.simplify(3 * (chi / (chi + 3)) - 3 * chi / (chi + 3)) == 0)
report("dielectric amplitude bound 1 + Xi/2 < 5/2 for all chi", sp.limit(1 + sp.Rational(3, 2) * chi / (chi + 3), chi, sp.oo) == sp.Rational(5, 2))

# ---------------- thermal bracket ----------------
u = sp.symbols("u", positive=True)
th = sp.symbols("theta", positive=True)
# NEC iff coth(theta/u) g < 1 with g = u - G sqrt(u^2-1)  <=>  theta > u artanh g (0<g<1)
Gv = M.G0
for rv in (0.4, 0.833, 1.2, 2.0):
    Tn = M.T_nec(rv, Gv)
    ok = abs(float(M.H_T(rv, Tn, Gv))) < 1e-10 and float(M.H_T(rv, 0.98 * Tn, Gv)) < 0 < float(M.H_T(rv, 1.02 * Tn, Gv))
    report("thermal window: H_T(T_nec(r)) = 0 and sign change, r=%.3f" % rv, ok)
# brute-force T_max
Tgrid = np.linspace(0.05, 0.12, 1401)
rgrid = np.linspace(0.01, np.arctanh(Gv) - 1e-4, 3001)
viol = [np.any(M.H_T(rgrid, T, Gv) < 0) for T in Tgrid]
Tmax_grid = Tgrid[np.where(viol)[0].max()]
report("T_max brute force %.2f mK vs %.2f mK" % (Tmax_grid * 1e3, M.Tmax_bare * 1e3), abs(Tmax_grid - M.Tmax_bare) < 1e-4)
report("zero-temperature limit H_T -> s^2 - s c G", abs(float(M.H_T(2.0, 1e-6, Gv)) - (np.sinh(2)**2 - np.sinh(2) * np.cosh(2) * Gv)) < 1e-12)


# ---------------- Fock-space check of the Gibbs state of H_eff ----------------
def annihilation(D):
    return np.diag(np.sqrt(np.arange(1, D)), 1)


D = 34
I = np.eye(D)
a1 = np.kron(annihilation(D), I); a2 = np.kron(I, annihilation(D))
ad1, ad2 = a1.conj().T, a2.conj().T
rv, chiv, wv = 0.45, 0.0, 1.0
kap = wv * np.tanh(2 * rv)
Hint = kap * (np.exp(1j * chiv) * ad1 @ ad2 + np.exp(-1j * chiv) * a1 @ a2)
Heff = wv * (ad1 @ a1 + ad2 @ a2) + Hint
wb = wv / np.cosh(2 * rv)
a_s = (a1 + a2) / np.sqrt(2)
X = a_s + a_s.conj().T; P = -1j * (a_s - a_s.conj().T)
for beta in (np.inf, 6.0, 2.5):
    if np.isinf(beta):
        vals, vecs = np.linalg.eigh(Heff); rho_st = np.outer(vecs[:, 0], vecs[:, 0].conj()); fac = 1.0
    else:
        Hs = Heff - np.eye(D * D) * np.linalg.eigvalsh(Heff)[0]
        rho_st = expm(-beta * Hs); rho_st /= np.trace(rho_st); fac = 1 / np.tanh(beta * wb / 2)
    vX = np.real(np.trace(rho_st @ X @ X)) - 1
    vP = np.real(np.trace(rho_st @ P @ P)) - 1
    eX = fac * (np.cosh(2 * rv) - np.sinh(2 * rv)) - 1
    eP = fac * (np.cosh(2 * rv) + np.sinh(2 * rv)) - 1
    report("Gibbs state of H_eff, beta=%s: <:X^2:> = coth(b w_b/2) e^{-2r} - 1" % beta, abs(vX - eX) < 2e-3)
    report("Gibbs state of H_eff, beta=%s: <:P^2:> = coth(b w_b/2) e^{+2r} - 1" % beta, abs(vP - eP) < 5e-3)
# single standing mode: H = w b^dag b + (kappa/2)(e^{i chi} b^dag^2 + h.c.), the cavity form used in the paper
D1 = 120
b1 = annihilation(D1); bd1 = b1.conj().T
kap1 = wv * np.tanh(2 * rv)
Hint1 = 0.5 * kap1 * (bd1 @ bd1 + b1 @ b1)
H1 = wv * bd1 @ b1 + Hint1
X1 = b1 + bd1; P1 = -1j * (b1 - bd1)
for beta in (np.inf, 6.0, 2.5):
    if np.isinf(beta):
        vals1, vecs1 = np.linalg.eigh(H1); rho1 = np.outer(vecs1[:, 0], vecs1[:, 0].conj()); fac = 1.0
    else:
        Hs1 = H1 - np.eye(D1) * np.linalg.eigvalsh(H1)[0]
        rho1 = expm(-beta * Hs1); rho1 /= np.trace(rho1); fac = 1 / np.tanh(beta * wb / 2)
    vX1 = np.real(np.trace(rho1 @ X1 @ X1)) - 1; vP1 = np.real(np.trace(rho1 @ P1 @ P1)) - 1
    report("single-mode Gibbs state, beta=%s: <:X^2:> = Theta e^{-2r} - 1" % beta, abs(vX1 - (fac * np.exp(-2 * rv) - 1)) < 1e-6)
    report("single-mode Gibbs state, beta=%s: <:P^2:> = Theta e^{+2r} - 1" % beta, abs(vP1 - (fac * np.exp(2 * rv) - 1)) < 1e-6)
vals1, vecs1 = np.linalg.eigh(H1); psi1 = vecs1[:, 0]
report("single mode: <H_int> = -(w/2) tanh 2r sinh 2r", abs(np.real(psi1.conj() @ Hint1 @ psi1) + 0.5 * wv * np.tanh(2 * rv) * np.sinh(2 * rv)) < 1e-8)
report("single mode: ground energy (w_b - w)/2", abs(vals1[0] - 0.5 * (wb - wv)) < 1e-8)

# ---------------- Eqs. (9), (11), (25) and (26) ----------------
# Eq. (9): beta = b cosh r + e^{i chi} b^dag sinh r annihilates the ground state of H_eff, [beta, beta^dag] = 1
# on low-lying states, and the excitation gap is (w^2 - kappa^2)^{1/2} = w/cosh 2r
for chiv1 in (0.0, 0.7):
    Hc = wv * bd1 @ b1 + 0.5 * kap1 * (np.exp(1j * chiv1) * bd1 @ bd1 + np.exp(-1j * chiv1) * b1 @ b1)
    ev, evec = np.linalg.eigh(Hc)
    bet = np.cosh(rv) * b1 + np.exp(1j * chiv1) * np.sinh(rv) * bd1
    g0 = evec[:, 0]
    report("Bogoliubov, chi=%.1f: beta annihilates the ground state of H_eff" % chiv1,
           np.linalg.norm(bet @ g0) < 1e-9)
    comm = (bet @ bet.conj().T - bet.conj().T @ bet)
    report("Bogoliubov, chi=%.1f: [beta, beta^dag] = 1 on the low-lying states" % chiv1,
           abs(np.real(g0.conj() @ comm @ g0) - 1) < 1e-9)
    report("Bogoliubov, chi=%.1f: gap = (w^2 - kappa^2)^{1/2} = w/cosh 2r" % chiv1,
           abs((ev[1] - ev[0]) - np.sqrt(wv**2 - kap1**2)) < 1e-9 and abs(np.sqrt(wv**2 - kap1**2) - wv / np.cosh(2 * rv)) < 1e-12)
# Eq. (11): Boltzmann weights in the dressed number basis are nbar^n/(1+nbar)^{n+1}, and Theta = 2 nbar + 1
xx, nn = sp.symbols("x n", positive=True)
nbar = 1 / (sp.exp(xx) - 1)
report("Gibbs weights (1 - e^{-x}) e^{-n x} = nbar^n/(1+nbar)^{n+1}",
       sp.simplify((1 - sp.exp(-xx)) * sp.exp(-nn * xx) - nbar**nn / (1 + nbar)**(nn + 1)) == 0
       or numzero((1 - sp.exp(-xx)) * sp.exp(-nn * xx) - nbar**nn / (1 + nbar)**(nn + 1), [xx, nn]))
report("Theta = coth(x/2) = 2 nbar + 1", sp.simplify(sp.coth(xx / 2) - (2 * nbar + 1)) == 0
       or numzero(sp.coth(xx / 2) - (2 * nbar + 1), [xx]))
rho_g = expm(-2.5 * (H1 - np.eye(D1) * vals1[0])); rho_g /= np.trace(rho_g)
bet0 = np.cosh(rv) * b1 + np.sinh(rv) * bd1
nb_num = np.real(np.trace(rho_g @ bet0.conj().T @ bet0)); nb_th = 1 / np.expm1(2.5 * wb)
report("Gibbs state: Tr rho_T beta^dag beta = 1/(e^{w_b/kT} - 1) (Fock space)", abs(nb_num - nb_th) < 1e-8)
# Eq. (25): the width optimum of u |A_0|_max with the Gaussian window
uu = np.linspace(1e-4, 1.5, 300001)
fu = uu / 2 * (1 - np.sqrt(1 - np.exp(-4 * uu**2)))
report("cap: f_* = max_u (u/2)(1 - sqrt(1 - e^{-4u^2})) = 0.0675 at u = 0.297, r = 0.61",
       abs(fu.max() - 0.0675) < 5e-5 and abs(uu[fu.argmax()] - 0.297) < 1e-3
       and abs(0.5 * np.arctanh(np.exp(-2 * uu[fu.argmax()]**2)) - 0.61) < 5e-3)
# Eq. (26): k_B T_NEC < hbar w wherever the window is open, and -> hbar w only as G -> 1, r -> infinity
worst = 0.0
for Gq in np.linspace(0.05, 1.0, 96):
    rq = np.linspace(1e-3, 8.0, 4000)
    gq = np.exp(-2 * rq) + (1 - Gq) * np.sinh(2 * rq)       # cosh2r - G sinh2r without cancellation
    m = gq < 1
    if m.any():
        worst = max(worst, np.max(1 / (2 * np.cosh(2 * rq[m]) * np.arctanh(gq[m]))))
report("thermal: k_B T_NEC/(hbar w) < 1 over the open window (max %.12f)" % worst, worst < 1)
lim = [1 / (2 * np.cosh(2 * q) * np.arctanh(np.exp(-2 * q))) for q in (1.0, 2.0, 4.0, 6.0)]
report("thermal: at G = 1 the ratio rises to 1 as r -> infinity (%s)" % ", ".join("%.6f" % v for v in lim),
       all(np.diff(lim) > 0) and 1 - lim[-1] < 1e-9 and all(v < 1 for v in lim))
report("single mode: excitation gap w_b = w / cosh 2r", abs(vals1[1] - vals1[0] - wb) < 1e-8)

# interaction (binding) energy of the ground state
vals, vecs = np.linalg.eigh(Heff); psi = vecs[:, 0]
Eint = np.real(psi.conj() @ Hint @ psi); Eem = np.real(psi.conj() @ (wv * (ad1 @ a1 + ad2 @ a2)) @ psi)
report("coupling-sector energy <H_int> = -w tanh2r sinh2r", abs(Eint + wv * np.tanh(2 * rv) * np.sinh(2 * rv)) < 2e-3)
report("EM-sector energy <H_0> = w (cosh 2r - 1)", abs(Eem - wv * (np.cosh(2 * rv) - 1)) < 2e-3)
report("total ground energy w (sech 2r - 1) = w_b - w", abs(vals[0] - (wb - wv)) < 2e-3)


# ---------------- Leontovich resonance and Q ----------------
w293, Q293, dL = M.resonance(M.Zs_normal(M.sig_Hg_293))
report("Q exact root vs pi Z0/(4 Rs) to first order in Rs/Z0: %.1f vs %.1f" % (Q293, M.Q293_closed), abs(Q293 / M.Q293_closed - 1) < 5 * M.Rs_293 / M.Z0)
report("normal-metal wall extension = delta/2", abs(dL / (M.delta_293 / 2) - 1) < 1e-3)
_, Qd, _ = M.resonance(M.Zs_normal(M.sig_Hg_293), nH=np.sqrt(M.H2["solid"]["eps_r"]))
report("dielectric-filled Q = pi Z0/(4 nH Rs) to first order", abs(Qd / (np.pi * M.Z0 / (4 * np.sqrt(M.H2['solid']['eps_r']) * M.Rs_293)) - 1) < 5 * M.Rs_293 / M.Z0)
_, Qs, dLs = M.resonance(M.Zs_super(1e-8, M.lam_L))
report("superconducting wall extension = London depth", abs(dLs / M.lam_L - 1) < 1e-3)


# direct field matching (independent of the closed form): transfer across the wall
def wall_mismatch(kL2, nH, Zs):
    # cavity field E = cos(kx), H = i k sin(kx)/(w mu0); wall requires E/H = Zs at x = L/2
    Ew = np.cos(kL2); Hw = 1j * nH / M.Z0 * np.sin(kL2)
    return Ew - M.Zs_normal(M.sig_Hg_293) * Hw if Zs is None else Ew - Zs * Hw


z = 0.5 * np.pi - np.arctan(1j * M.Zs_normal(M.sig_Hg_293) / M.Z0)
report("wall impedance condition satisfied by the resonance root", abs(wall_mismatch(z, 1.0, M.Zs_normal(M.sig_Hg_293))) < 1e-12)


# ---------------- Casimir tensor of ideal plates ----------------
eta = sp.diag(-1, 1, 1, 1)
nvec = sp.Matrix([0, 0, 0, 1])
Tc = eta - 4 * nvec * nvec.T            # times pi^2 hbar c /(720 L^4), contravariant (t,y,z,n)
trace = sum(eta[i, i] * Tc[i, i] for i in range(4))
report("Casimir tensor traceless", sp.simplify(trace) == 0)
report("Casimir tangential null combination rho + p_par = 0", sp.simplify(Tc[0, 0] + Tc[1, 1]) == 0)
report("Casimir normal null combination rho + p_n = 4 rho < 0", sp.simplify(Tc[0, 0] + Tc[3, 3]) == -4)
# boost invariance of the tangential block: Lambda^T T Lambda = T for a boost along y
v = sp.symbols("v", real=True)
gam = 1 / sp.sqrt(1 - v**2)
Lam = sp.Matrix([[gam, -gam * v, 0], [-gam * v, gam, 0], [0, 0, 1]])
Tt = Tc[:3, :3]
report("tangential block invariant under boosts parallel to the plates", sp.simplify(Lam * Tt * Lam.T - Tt) == sp.zeros(3, 3))


# ---------------- dielectric outside the layer: amplitude 1/eps_r at fixed geometry ----------------
def throat_scale(eps_r, gap):
    """1D TE mode at the reference frequency in a cavity with perfect walls, dielectric eps_r for
    |x| > gap/2 and vacuum for |x| < gap/2 (gap = 0: fully filled). Returns the throat energy density
    (field plus polarization, eps(0) E(0)^2) per unit total mode energy."""
    n = np.sqrt(eps_r); k0 = M.k0
    E_g = np.cos(k0 * gap / 2); dE_g = -k0 * np.sin(k0 * gap / 2)
    Acoef = E_g; Bcoef = dE_g / (n * k0)
    d = (np.arctan2(-Acoef, Bcoef) / (n * k0)) % (np.pi / (n * k0))   # first node of E outside the gap
    x = np.linspace(0, gap / 2 + d, 400001)
    inside = x < gap / 2
    E = np.where(inside, np.cos(k0 * x), Acoef * np.cos(n * k0 * (x - gap / 2)) + Bcoef * np.sin(n * k0 * (x - gap / 2)))
    dE = np.gradient(E, x)
    epsx = np.where(inside, 1.0, eps_r)
    U = np.trapezoid(epsx * E**2 + (dE / k0) ** 2, x)
    eps0 = 1.0 if gap > 0 else eps_r
    return eps0 * E[0] ** 2 / U


er = M.H2["solid"]["eps_r"]
ratio = throat_scale(er, 4 * M.ell) / throat_scale(er, 0.0)       # vacuum gap of half-width 2 ell > a = sqrt(3) ell
report("H2 outside the layer: throat scale / filled = 1/eps_r (%.4f vs %.4f)" % (ratio, 1 / er), abs(ratio * er - 1) < 5e-3)
ratio_vac = throat_scale(1.0, 0.0)
report("fixed geometry: filled / vacuum throat scale = n_H (%.4f vs %.4f)" % (throat_scale(er, 0.0) / ratio_vac, np.sqrt(er)),
       abs(throat_scale(er, 0.0) / ratio_vac / np.sqrt(er) - 1) < 1e-3)

# ---------------- parametrically driven alternative ----------------
from scipy.optimize import minimize_scalar as _ms
Gd = M.G0
Sfun = lambda e: (1 + Gd) / (2 * (1 + e)) + (1 - Gd) / (2 * (1 - e))
_r = _ms(Sfun, bounds=(0, 0.9999999), method="bounded", options=dict(xatol=1e-12))
qd = np.sqrt((1 - Gd) / (1 + Gd))
report("driven: min_eps S = (1+G)(1+q)^2/4", abs(_r.fun - (1 + Gd) * (1 + qd) ** 2 / 4) < 1e-9)
report("driven: optimal pump eps* = (1-q)/(1+q)", abs(_r.x - (1 - qd) / (1 + qd)) < 1e-6)
# Langevin steady state of the damped degenerate parametric oscillator: Lyapunov check
g_, gam, nbar = 0.3, 1.0, 0.7
A_ = np.array([[-(gam / 2 - g_), 0], [0, -(gam / 2 + g_)]])
Dn = gam * (2 * nbar + 1) * np.eye(2)
from scipy.linalg import solve_continuous_lyapunov
Cv = solve_continuous_lyapunov(A_, -Dn)
eps_ = 2 * g_ / gam
report("driven: steady variances Theta/(1 -+ eps)", abs(Cv[0, 0] - (2 * nbar + 1) / (1 - eps_)) < 1e-9 and abs(Cv[1, 1] - (2 * nbar + 1) / (1 + eps_)) < 1e-9)
report("QEI constants: 27/1024 = (9/64)(3/16)", abs(27 / 1024 - (9 / 64) * (3 / 16)) < 1e-15)

# ---------------- loaded-resonator consistency (Sec. 3.3) ----------------
_r, _w = sp.symbols("r omega", positive=True)
_k = _w * sp.tanh(2 * _r)                       # kappa from tanh 2r = kappa/omega
_wb = sp.sqrt(_w ** 2 - _k ** 2)                # loaded resonance
report("loaded resonance omega_b = omega / cosh 2r",
       sp.simplify(_wb - _w / sp.cosh(2 * _r)) == 0)
report("Faraday: magnetic amplitude omega_b/(omega + kappa) = e^{-2r}",
       sp.simplify((_wb / (_w + _k) - sp.exp(-2 * _r)).rewrite(sp.exp)) == 0)
report("equipartition in the loaded mode: e^{-4r} <P^2> = <X^2>",
       sp.simplify(sp.exp(-4 * _r) * sp.exp(2 * _r) - sp.exp(-2 * _r)) == 0)
_G = sp.Symbol("G", positive=True)
_A = (sp.cosh(2 * _r) - 1 - _G * sp.sinh(2 * _r)) / 2
report("Ghat = 1 limit of the bracket is (e^{-2r} - 1)/2",
       sp.simplify(_A.subs(_G, 1) - (sp.exp(-2 * _r) - 1) / 2) == 0)
report("bracket deepens as Ghat grows", sp.simplify(sp.diff(_A, _G)) == -sp.sinh(2 * _r) / 2)

# ---------------- loaded cavity (Sec. 3.4) ----------------
_LD = M.res["loaded"]
report("loaded resonance condition is satisfied", abs(_LD["residual"]) < 1e-10)
report("loaded fundamental equals omega / cosh 2r",
       abs(_LD["wb_over_w"] - 1 / np.cosh(4.0)) < 1e-9)
report("unloaded limit Cs -> 0 returns k = pi/L",
       abs(M.loaded_k(5e-3, 1e-20) / M.k0 - 1) < 1e-7)
report("loading lowers the fundamental monotonically",
       M.loaded_k(5e-3, 1e-11) > M.loaded_k(5e-3, 1e-10) > M.loaded_k(5e-3, 1e-9))
report("mode function is continuous at the element",
       abs(np.cos(_LD["k_b"] * _LD["d"]) - (np.cos(_LD["k_b"] * _LD["d"])
            / np.sin(_LD["k_b"] * (M.Lc / 2 - _LD["d"])))
           * np.sin(_LD["k_b"] * (M.Lc / 2 - _LD["d"]))) < 1e-12)
_jump = (-(np.cos(_LD["k_b"] * _LD["d"]) / np.sin(_LD["k_b"] * (M.Lc / 2 - _LD["d"])))
         * _LD["k_b"] * np.cos(_LD["k_b"] * (M.Lc / 2 - _LD["d"]))
         + _LD["k_b"] * np.sin(_LD["k_b"] * _LD["d"]))
_src = -(_LD["k_b"] * M.C) ** 2 * (4e-7 * np.pi) * _LD["Cs"] * np.cos(_LD["k_b"] * _LD["d"])
report("derivative jump equals the sheet polarization current",
       abs(_jump - _src) < 1e-6 * abs(_src))

# ---------------- Eq. (18): energy normalization of the loaded mode ----------------
# quadrature convention X = b + b^dag, P = -i(b - b^dag): [X,P] = 2i and H = (hbar w/4)(X^2+P^2)
_bq = annihilation(40)
_Xq = _bq + _bq.T; _Pq = -1j * (_bq - _bq.T)
_cut = slice(0, 30)                     # stay away from the truncation edge
report("quadratures obey [X,P] = 2i",
       np.allclose((_Xq @ _Pq - _Pq @ _Xq)[_cut, _cut], 2j * np.eye(40)[_cut, _cut]))
report("(X^2+P^2)/4 = b^dag b + 1/2",
       np.allclose(((_Xq @ _Xq + _Pq @ _Pq) / 4)[_cut, _cut], (_bq.T @ _bq + 0.5 * np.eye(40))[_cut, _cut]))
# dimensions as exponent vectors (kg, m, s, A)
_dim = dict(E=(1, 1, -3, -1), area=(0, 2, 0, 0), eps=(-1, -3, 4, 2), length=(0, 1, 0, 0),
            Csq=(-1, -2, 4, 2), J=(1, 2, -2, 0))
_add = lambda *v: tuple(int(sum(x)) for x in zip(*v))
report("Eq. (18) field term E0^2 A eps dx has dimension of energy",
       _add(_dim["E"], _dim["E"], _dim["area"], _dim["eps"], _dim["length"]) == _dim["J"])
report("Eq. (18) element term E0^2 A C_sq has dimension of energy",
       _add(_dim["E"], _dim["E"], _dim["area"], _dim["Csq"]) == _dim["J"])
_E0, _Aa, _eps, _L, _w, _xx = sp.symbols("E_0 A epsilon L omega x", positive=True)
_WE = sp.Rational(1, 2) * _E0**2 * _Aa * _eps * sp.integrate(sp.cos(sp.pi * _xx / _L)**2, (_xx, -_L / 2, _L / 2))
_E0sq = sp.solve(sp.Eq(_WE, hbar * _w / 4), _E0**2)
_E0sq = _E0sq[0] if _E0sq else sp.solve(sp.Eq(_WE, hbar * _w / 4), _E0)[0]**2
report("Eq. (18) at C_sq = 0 gives eps E0^2 = hbar w / V with V = A L",
       sp.simplify(_eps * _E0sq - hbar * _w / (_Aa * _L)) == 0)

# ---------------- QEI constants (Sec. 4.3) ----------------
_t = sp.symbols("t", real=True)
_f = tau / (sp.pi * (_t**2 + tau**2))   # Lorentzian sampling, unit area
report("Lorentzian sampling function has unit area", sp.simplify(sp.integrate(_f, (_t, -sp.oo, sp.oo)) - 1) == 0)
_g2 = sp.diff(sp.sqrt(_f), _t, 2)
_FE_scalar = sp.simplify(sp.integrate(sp.simplify(_g2**2), (_t, -sp.oo, sp.oo)) / (16 * sp.pi**2))
report("massless scalar, Fewster-Eveson form: 27/(2048 pi^2 tau^4)", sp.simplify(_FE_scalar - sp.Rational(27, 2048) / (sp.pi**2 * tau**4)) == 0)
report("electromagnetic = two scalar polarizations: 27/(1024 pi^2 tau^4)", sp.simplify(2 * _FE_scalar - sp.Rational(27, 1024) / (sp.pi**2 * tau**4)) == 0)
report("Ford-Roman electromagnetic 3/(16 pi^2) = 2 x scalar 3/(32 pi^2)", sp.Rational(3, 16) == 2 * sp.Rational(3, 32))
report("Pfenning over Ford-Roman = 9/64", sp.Rational(27, 1024) / sp.Rational(3, 16) == sp.Rational(9, 64))

# ---------------- complete boundary vacuum, Eq. (19) (Lifshitz, planar stacks) ----------------
import casimir_layer as CL
_HB, _C = M.HBAR, M.C
for _g in (5e-3, 0.0375):
    _r = CL.layer_stress([("perfect",), ("slab", None, 0.0)], _g)
    _T0 = -np.pi ** 2 * _HB * _C / (720 * (2 * _g) ** 4)
    report("ideal plates, gap %.1f mm: T00 = Brown-Maclay, Txx = 3 T00, N = 0" % (2e3 * _g),
           abs(_r["T00"] / _T0 - 1) < 1e-9 and abs(_r["Txx"] / (3 * _T0) - 1) < 1e-9 and abs(_r["N"] / _T0) < 1e-12)
for _chi in (1e-3, 1e-4):
    _loc, _ = CL.layer_stress([("half", lambda xi, c=_chi: 1.0 + c + 0 * xi)], 1.0, x=[1.0 - 1e-3])
    _an = _HB * _C * _chi / (40 * np.pi ** 2 * 1e-3 ** 4)
    report("dilute half-space: T00 -> hbar c chi/(40 pi^2 z^4) at chi=%.0e, N = 3 T00/2" % _chi,
           abs(_loc[0][0] / _an - 1) < 1.2 * _chi and abs(_loc[0][2] / _loc[0][0] - 1.5) < 1e-9)
for _ag in (1e-5, 1e-6):
    _r = CL.layer_stress([("half", CL.eps_plasma(M.lam_L))], _ag / 2, a=_ag / 4, kmax=200 / _ag)
    _dl = M.lam_L / _ag
    _exp = (1 - 16 / 3 * _dl + 24 * _dl ** 2 - 640 / 7 * (1 - np.pi ** 2 / 210) * _dl ** 3
            + 2800 / 9 * (1 - 163 * np.pi ** 2 / 7350) * _dl ** 4)
    report("plasma-model Lifshitz pressure vs fourth-order expansion at a = %.0e m" % _ag,
           abs(_r["Txx"] / (-3 * np.pi ** 2 * _HB * _C / (720 * _ag ** 4)) - _exp) < 1000 * _dl ** 5)
_rinf = CL.layer_stress(*CL.cfg_passive_vacuum(Csq=1e3))
report("sheet C -> infinity: ideal plates at the sheets", abs(_rinf["T00"] / (-np.pi ** 2 * _HB * _C / (720 * (2 * CL.d_sheet) ** 4)) - 1) < 1e-5)
_r0 = CL.layer_stress(*CL.cfg_passive_vacuum(Csq=1e-40)); _r3 = CL.layer_stress(*CL.cfg_cavity_vacuum(CL.HG))
report("sheet C -> 0: the bare superconducting cavity", abs(_r0["nec"] / _r3["nec"] - 1) < 1e-8 and abs(_r0["sigma"] / _r3["sigma"] - 1) < 1e-8)
_rng = np.random.default_rng(3)
_xi = _rng.uniform(1e9, 1e12, 50); _k = _rng.uniform(1, 1e4, 50); _e = 1.284; _h = 3e-3
_q = np.sqrt(_k ** 2 + _xi ** 2 / _C ** 2); _qe = np.sqrt(_k ** 2 + _e * _xi ** 2 / _C ** 2)
_rs12 = (_q - _qe) / (_q + _qe); _rp12 = (_e * _q - _qe) / (_e * _q + _qe); _ex = np.exp(-2 * _qe * _h)
_rs, _rp = CL.stack_R(_xi, _k, [("perfect",), ("slab", lambda x: _e + 0 * x, _h)])
report("dielectric slab on a perfect wall: Airy formula",
       np.max(abs(_rs - (_rs12 - _ex) / (1 - _rs12 * _ex))) < 1e-14 and np.max(abs(_rp - (_rp12 + _ex) / (1 + _rp12 * _ex))) < 1e-14)
_rs, _rp = CL.stack_R(_xi, _k, [("half", lambda x: _e + 0 * x)])
report("dielectric half-space: Fresnel coefficients", np.max(abs(_rs - _rs12)) < 1e-15 and np.max(abs(_rp - _rp12)) < 1e-15)
_Cs = 1e-10
_rs, _rp = CL.stack_R(_xi, _k, [("perfect",), ("slab", None, 1.0), ("sheet", CL.sheet_C(_Cs))])
_sTE = M.MU0 * _Cs * _xi ** 2 / (2 * _q); _sTM = _Cs * _q / (2 * M.EPS0)
report("isolated capacitive sheet: conductivity-sheet coefficients",
       np.max(abs(_rs - (-_sTE / (1 + _sTE)))) < 1e-12 and np.max(abs(_rp - _sTM / (1 + _sTM))) < 1e-12)
# energy density integrand equals Sopova-Ford Eq. (50) (hbar = c = 1)
_u, _t, _a, _z, _R = sp.symbols("u t a z R", positive=True)
_D = 1 - _R ** 2 * sp.exp(-2 * _u * _a)
_ours = ((1 - _t ** 2) * _R * (sp.exp(-2 * _u * _z) + sp.exp(-2 * _u * (_a - _z))) / _D - _t ** 2 * 2 * _R ** 2 * sp.exp(-2 * _u * _a) / _D) / (4 * sp.pi ** 2)
_sf = (_t ** 2 * _R ** 2 / (_R ** 2 - sp.exp(2 * _u * _a)) + (1 - _t ** 2) * _R / (1 - _R ** 2 * sp.exp(-2 * _u * _a)) * sp.exp(-_u * _a) * sp.cosh(_u * (2 * _z - _a))) / (2 * sp.pi ** 2)
report("energy density integrand equals Sopova-Ford Eq. (50), per polarization",
       numzero(sp.simplify((_ours - _sf).rewrite(sp.exp)), [_u, _t, _a, _z, _R], lo=0.1, hi=0.9))
# the recursion printed as Eq. (20), implemented from the manuscript text alone
def _paper_R(_kap, _t, _stack):
    _Z0 = np.sqrt(M.MU0 / M.EPS0); _xi = M.C * _kap * _t
    _kj = lambda e: _kap * np.sqrt(1 + (e - 1) * _t ** 2)
    _eps = np.ones_like(_kap); _Rs = -np.ones_like(_kap); _Rp = np.ones_like(_kap)
    for _el in _stack:
        if _el[0] == "slab":
            _e2 = (_el[1](_xi) if _el[1] is not None else 1.0) * np.ones_like(_kap)
            _k1, _k2 = _kj(_eps), _kj(_e2)
            _rs = (_k2 - _k1) / (_k2 + _k1)
            _rp = (_eps * _k2 - _e2 * _k1) / (_eps * _k2 + _e2 * _k1)
            _Rs = (_rs + _Rs) / (1 + _rs * _Rs); _Rp = (_rp + _Rp) / (1 + _rp * _Rp)
            _eps = _e2
            _Rs = _Rs * np.exp(-2 * _kj(_eps) * _el[2]); _Rp = _Rp * np.exp(-2 * _kj(_eps) * _el[2])
        else:                                    # capacitive sheet in the current medium
            _sig = _el[1](_xi)
            _es = _Z0 * _sig * _kap * _t / (2 * _kj(_eps))
            _ep = _Z0 * _sig * _kj(_eps) / (2 * _eps * _kap * _t)
            _Rs = (_Rs - _es * (1 + _Rs)) / (1 + _es * (1 + _Rs))
            _Rp = (_Rp + _ep * (1 - _Rp)) / (1 + _ep * (1 - _Rp))
    _k1, _k2 = _kj(_eps), _kap                   # final interface into the vacuum gap
    _rs = (_k2 - _k1) / (_k2 + _k1); _rp = (_eps * _k2 - _k1) / (_eps * _k2 + _k1)
    return (_rs + _Rs) / (1 + _rs * _Rs), (_rp + _Rp) / (1 + _rp * _Rp)


_rng2 = np.random.default_rng(11)
_kp = 10 ** _rng2.uniform(-1, 4, 200); _tt = _rng2.uniform(1e-3, 1 - 1e-6, 200)
_sig_sheet = lambda xi: xi * CL.Csq0
for _nm, _st in (("passive cavity", [("slab", None, CL.L_vac / 2 - CL.d_sheet), ("sheet", _sig_sheet)]),
                 ("hydrogen gap", [("slab", CL.H2s, CL.L_H2 / 2 - CL.gap_H2)])):
    _code = [("perfect",)] + [(e[0], e[1], e[2]) if e[0] == "slab" else ("sheet", CL.sheet_C(CL.Csq0)) for e in _st]
    _a1, _b1 = _paper_R(_kp, _tt, _st)
    _a2, _b2 = CL.stack_R(M.C * _kp * _tt, _kp * np.sqrt(1 - _tt ** 2), _code)
    report("Eq. (20) as printed reproduces the computed reflection, %s" % _nm,
           np.max(np.abs(_a1 - _a2)) < 1e-12 and np.max(np.abs(_b1 - _b2)) < 1e-12)
_r = CL.results()
report("passive cavity: N_C > 0 at every point and for C_sq x 0.1..100", _r["P_N_min"] > 0 and _r["C_scan_min_local"] > 0)

# ---------------- Reissner-Nordstrom exteriors (Sec. 3.1, Eq. (4)), geometric units G = c = 1 ----------------
# static symmetric shell: sigma = -sqrt(f)/(2 pi a), p = (sqrt(f)/a + f'/(2 sqrt f))/(4 pi)  (Israel, mirror symmetry)
rr, Mq, Qq, zq, eq_ = sp.symbols("r M Q zeta epsilon", positive=True)
fq = 1 - 2 * Mq / rr + Qq**2 / rr**2
sgq = -sp.sqrt(fq) / (2 * sp.pi * rr)
pq = (sp.sqrt(fq) / rr + sp.diff(fq, rr) / (2 * sp.sqrt(fq))) / (4 * sp.pi)
sI, pI = -1 / (2 * sp.pi * rr), 1 / (4 * sp.pi * rr)
report("RN: flat limit gives Eq. (3)", sp.simplify(sgq.subs({Mq: 0, Qq: 0}) - sI) == 0 and sp.simplify(pq.subs({Mq: 0, Qq: 0}) - pI) == 0)
report("RN: sigma = sigma_I sqrt f", sp.simplify(sgq - sI * sp.sqrt(fq)) == 0)
report("RN: p = p_I (1 - M/a)/sqrt f, Eq. (4)", sp.simplify(pq - pI * (1 - Mq / rr) / sp.sqrt(fq)) == 0)
report("RN: (1 - M/a)^2 - f = (M^2 - Q^2)/a^2, Eq. (4)", sp.simplify((1 - Mq / rr)**2 - fq - (Mq**2 - Qq**2) / rr**2) == 0)
report("RN: null-energy integral sigma/sqrt f = sigma_I", sp.simplify(sgq / sp.sqrt(fq) - sI) == 0)
report("RN: (Q^2 - M^2)/a^2 = (sigma/sigma_I)^2 [1 - (p/p_I)^2]",
       sp.simplify((Qq**2 - Mq**2) / rr**2 - (sgq / sI)**2 * (1 - (pq / pI)**2)) == 0)
report("RN: sigma + p = -(a^2 - 3 M a + 2 Q^2)/(4 pi a^3 sqrt f) (photon sphere)",
       sp.simplify(sgq + pq + (rr**2 - 3 * Mq * rr + 2 * Qq**2) / (4 * sp.pi * rr**3 * sp.sqrt(fq))) == 0)
# p >= p_I for M >= |Q| and a > r_+, equality at |Q| = M (sweep in units M = 1)
_ok, _eqok = True, True
for _q in np.linspace(0.0, 1.0, 101):
    _rp = 1 + np.sqrt(max(1 - _q**2, 0.0))
    _a = _rp * (1 + np.logspace(-9, 4, 400))
    _f = (_a - _rp) * (_a - (1 - np.sqrt(max(1 - _q**2, 0.0)))) / _a**2
    _ratio = ((_a - 1) / _a) / np.sqrt(_f)         # p/p_I, with 1 - M/a written without cancellation
    _ok &= bool(np.all(_ratio >= 1 - 1e-12))
    if _q == 1.0:
        _eqok &= bool(np.allclose(_ratio, 1.0, rtol=1e-9))
report("RN: p/p_I >= 1 for every M >= |Q|, a > r_+ (sweep)", _ok)
report("RN: p/p_I = 1 identically at extremality", _eqok)
# p = 0 shell of an over-extremal exterior, Q^2 = M^2 (1 + eps^2): a0 = M; equation of state p = zeta (sigma - sigma0)
sub9 = {Qq: Mq * sp.sqrt(1 + eq_**2)}
report("RN: p(M) = 0 and sigma(M) = eps sigma_I for Q^2 = M^2(1+eps^2)",
       sp.simplify(pq.subs(sub9).subs(rr, Mq)) == 0 and sp.simplify((sgq / sI).subs(sub9).subs(rr, Mq) - eq_) == 0)
_s0 = -eq_ / (2 * sp.pi * Mq)
_sa = _s0 * (zq + (rr / Mq)**(-2 * (1 + zq))) / (1 + zq)              # solves sigma' = -2(sigma + p)/a
report("RN: sigma(a) solves the conservation law with dp/dsigma = zeta",
       sp.simplify(sp.diff(_sa, rr) + 2 * (_sa + zq * (_sa - _s0)) / rr) == 0)
_V = fq.subs(sub9) - (2 * sp.pi * rr * _sa)**2                          # a'^2 + V(a) = 0
_V2 = sp.simplify(sp.diff(_V, rr, 2).subs(rr, Mq))
report("RN: V''(M) = (2 - 8 eps^2 zeta)/M^2, stable iff zeta < (sigma_I/sigma)^2/4",
       sp.simplify(_V2 - (2 - 8 * eq_**2 * zq) / Mq**2) == 0)
_Vn = sp.lambdify((rr,), _V.subs({Mq: 1, eq_: sp.Rational(3, 10), zq: sp.Rational(1, 2)}))
_h = 1e-4
report("RN: V''(M) by finite differences (eps = 0.3, zeta = 0.5)",
       abs((_Vn(1 + _h) - 2 * _Vn(1.0) + _Vn(1 - _h)) / _h**2 - (2 - 8 * 0.09 * 0.5)) < 1e-5)

# static wall in the redshift well (Sec. 3.1): m = r (sqrt f_in - sqrt f_out) for an ordinary static shell
mw, rw, fin, fout = sp.symbols("m r f_in f_out", positive=True)
Mi = sp.symbols("M_in", positive=True)
f_in = 1 - 2 * Mi / rw + Qq**2 / rw**2
Mo = Mi + mw * sp.sqrt(f_in) - mw**2 / (2 * rw)          # claimed exterior mass beyond the wall
f_out = 1 - 2 * Mo / rw + Qq**2 / rw**2
report("Wall: f_out = (sqrt f_in - m/r)^2, so a static wall needs m <= r sqrt f_in",
       sp.simplify(f_out - (sp.sqrt(f_in) - mw / rw)**2) == 0)
mstar = sp.solve(sp.diff(Mo - Mi, mw), mw)[0]
report("Wall: the Killing energy added is at most r f_in/2 (the over-extremal margin when f_in = eps^2)",
       sp.simplify((Mo - Mi).subs(mw, mstar) - rw * f_in / 2) == 0)
report("Wall: m_max = a0 |sigma/sigma_I| = 2 pi a0^2 |sigma| in geometric units",
       sp.simplify(rw * (2 * sp.pi * rw * sp.Symbol("s_abs", positive=True)) - 2 * sp.pi * rw**2 * sp.Symbol("s_abs", positive=True)) == 0)

print("\n%d failures" % len(fails))
sys.exit(1 if fails else 0)
