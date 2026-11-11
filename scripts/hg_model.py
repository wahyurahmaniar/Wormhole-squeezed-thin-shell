"""Mercury-hydrogen junction-layer model of the manuscript.

Computes, at one common normalization (omega, V, ell, a0, N_src):
  1. Hg surface impedance (normal liquid at 293 K, superconducting at T << Tc), the Leontovich
     resonance condition of the half-wave cavity, its Q, and the penetration correction to the mode;
  2. the stationary (Gibbs) state of H_eff at the wall temperature and the field correlations;
  3. the stress decomposition: boundary vacuum (Casimir), squeezed excess, thermal radiation,
     material rest energy, coupling-sector binding energy;
  4. the NEC condition at finite temperature and the thermal window T_max;
  5. magnitudes and the ratio R_NEC for the four configurations C1-C4, plus the liquid-H2 variant.
Writes model_results.json next to this script."""
import json, os
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

# ---------------- constants (SI, CODATA 2018) ----------------
HBAR = 1.054571817e-34; C = 2.99792458e8; G = 6.67430e-11
EPS0 = 8.8541878128e-12; MU0 = 1.25663706212e-6; KB = 1.380649e-23
QE = 1.602176634e-19; ME = 9.1093837015e-31; NA = 6.02214076e23; AMU = 1.66053906660e-27
AB = 5.29177210903e-11
Z0 = MU0 * C

# ---------------- reference source and junction ----------------
f0 = 2e9
omega = 2 * np.pi * f0
k0 = omega / C
ell = 1e-3                   # layer width (m); L_eff = ell
Leff = ell
a0 = 1.0
r_ref = 2.0
k0l = k0 * ell
G0 = np.exp(-2 * k0l ** 2)
israel_nec = C ** 4 / (4 * np.pi * G * a0)   # |sigma+p| required (J/m^2)
israel_sigma = 2 * israel_nec
israel_p = israel_nec
lam0 = 2 * np.pi / k0        # free-space wavelength at the reference frequency (m)
Lc = np.pi / k0              # half-wave cavity length in vacuum (m)
x_wall = Lc / 2              # throat-to-wall distance
# Reference mode volume: the smallest patch that carries an approximately planar standing wave of
# both polarizations, one wavelength across and half a wavelength long.  A cubic cavity resonant at
# the same frequency (TE101 and TE011 degenerate, side c/(sqrt2 f)) has V within 0.15 decades.
V = lam0 ** 2 * Lc
a_cube = C / (np.sqrt(2) * f0)
V_cube = a_cube ** 3
A_patch = lam0 ** 2
rho1 = HBAR * omega / V      # single-mode energy scale (J/m^3)

# ---------------- hydrogen (NPB values; Clausius-Mossotti) ----------------
alphaH2 = 8.93e-41           # C m^2 / V  (5.41 a.u.)
H2 = {}
for phase, rho_m, Trange in (("liquid", 70.8, (13.8, 20.3)), ("solid", 86.0, (0.0, 13.8))):
    n = rho_m / (2.016 * AMU)
    x = n * alphaH2 / (3 * EPS0)
    er = (1 + 2 * x) / (1 - x)
    chi = er - 1
    Xi = n * alphaH2 / EPS0
    H2[phase] = dict(rho_m=rho_m, n=n, eps_r=er, chi=chi, Xi=Xi, nH=np.sqrt(er), T_range=Trange)
C_uniform = (2 / np.pi) ** 2         # phase-alignment fraction, uniform filling of the half wavelength

# ---------------- mercury ----------------
rho_Hg_293 = 95.8e-8                 # Ohm m, liquid Hg at 20 C (CRC)
sig_Hg_293 = 1 / rho_Hg_293
Tc_Hg = 4.15                         # K
Delta_Hg = 0.83e-3 * QE              # J, tunneling gap (Bermon-Ginsberg)
dens_Hg = 13546.0                    # kg/m^3 at 20 C
n_e = 2 * dens_Hg / (200.59e-3) * NA # two conduction electrons per atom
omega_p = np.sqrt(n_e * QE ** 2 / (EPS0 * ME))
lam_L = C / omega_p                  # free-electron London depth (m)
R_res_range = (1e-8, 1e-6)           # residual surface resistance range assumed (Ohm)


def Zs_normal(sig):
    """Normal-skin-effect surface impedance, e^{-i w t} convention: Z = (1 - i) sqrt(w mu0 / 2 sigma)."""
    Rs = np.sqrt(omega * MU0 / (2 * sig))
    return Rs * (1 - 1j)


def Zs_super(Rs, lam):
    """Superconducting wall: Z = Rs - i Xs with Xs = w mu0 lambda (inductive)."""
    return Rs - 1j * omega * MU0 * lam


def resonance(Zs, nH=1.0):
    """Solve cot(k L/2) = i nH Zs / Z0 for the complex wavenumber at the vacuum cavity length scaled
    to the medium, L = pi/(nH k0); return complex omega, Q, and the effective-length shift per wall."""
    L = np.pi / (nH * k0)
    rhs = 1j * nH * Zs / Z0
    # kL/2 = arccot(rhs) near pi/2
    z = 0.5 * np.pi - np.arctan(rhs)          # cot(z) = rhs  <=>  z = pi/2 - arctan(rhs)
    k = 2 * z / L
    w = k * C / nH
    Q = w.real / (2 * abs(w.imag))
    dL_wall = 0.5 * (np.pi / k.real - L)      # effective extension per wall
    return w, Q, dL_wall


# normal liquid Hg at 293 K
Zn = Zs_normal(sig_Hg_293)
Rs_293 = Zn.real
delta_293 = np.sqrt(2 / (omega * MU0 * sig_Hg_293))
w293, Q293, dL293 = resonance(Zn)
Q293_closed = np.pi * Z0 / (4 * Rs_293)
# superconducting Hg, residual-limited
Qsc = []
for Rres in R_res_range:
    w_, Q_, dL_ = resonance(Zs_super(Rres, lam_L))
    Qsc.append(Q_)
_, _, dL_sc = resonance(Zs_super(R_res_range[0], lam_L))
frac_wall_energy_sc = 2 * lam_L / Lc          # fraction of mode energy in the two London layers
frac_wall_energy_n = delta_293 / Lc           # normal skin layers (order of magnitude)
gamma_sc = [omega / q for q in Qsc]           # energy decay rate (1/s)


# ---------------- stationary state and NEC bracket ----------------
def coth(x):
    return 1.0 / np.tanh(x)


def H_T(r, T, Gl):
    """Projected bracket of the polarization-completed pair in the Gibbs state of H_eff at temperature T:
    (1/2)[coth(hbar w_b / 2 kB T)(cosh 2r - G sinh 2r) - 1], w_b = w / cosh 2r."""
    r = np.asarray(r, float)
    u = np.cosh(2 * r)
    g = u - Gl * np.sinh(2 * r)
    if T <= 0:
        th = np.ones_like(u)
    else:
        th = coth(HBAR * omega / (2 * KB * T * u))
    return 0.5 * (th * g - 1)


def T_nec(r, Gl):
    """Temperature below which the completed pair violates the NEC at squeezing r."""
    u = np.cosh(2 * r)
    g = u - Gl * np.sinh(2 * r)
    if g >= 1:
        return 0.0
    return HBAR * omega / (2 * KB * u * np.arctanh(g))


def T_max(Gl):
    rmax = np.arctanh(Gl)
    res = minimize_scalar(lambda r: -T_nec(r, Gl), bounds=(1e-3, rmax - 1e-6), method="bounded",
                          options=dict(xatol=1e-10))
    return -res.fun, res.x


def opt_r(T, Gl):
    rmax = np.arctanh(Gl)
    res = minimize_scalar(lambda r: H_T(r, T, Gl), bounds=(1e-4, rmax), method="bounded",
                          options=dict(xatol=1e-10))
    return res.x, float(H_T(res.x, T, Gl))


def R_of(H, amp=1.0, N=1):
    """Feasibility ratio for the completed pair: 2 L_eff N rho1 amp |H| / israel_nec."""
    return 2 * Leff * N * rho1 * amp * abs(H) / israel_nec


# ---------------- configurations ----------------
T_op = 0.010                         # dilution-refrigerator operating temperature (K)
sol = H2["solid"]; liq = H2["liquid"]
G_sol = G0 ** sol["eps_r"]
G_liq = G0 ** liq["eps_r"]
F_sol = 1 + 0.5 * sol["Xi"] * C_uniform
F_liq = 1 + 0.5 * liq["Xi"] * C_uniform
F_sol_max = 1 + 0.5 * sol["Xi"]
F_liq_max = 1 + 0.5 * liq["Xi"]
Phi_Hg = 1 / (1 + frac_wall_energy_sc)

configs = {
    "C1": dict(label="squeezed field, ideal walls", T=0.0, Gl=G0, amp=1.0, matter=0.0, nH=1.0),
    "C2": dict(label="+ solid H2 in the layer (ideal walls)", T=0.0, Gl=G_sol, amp=F_sol_max, matter=sol["rho_m"] * C ** 2 * Leff, nH=sol["nH"]),
    "C2o": dict(label="+ solid H2 outside the layer (ideal walls)", T=0.0, Gl=G0, amp=1 / sol["eps_r"], matter=0.0, nH=1.0),
    "C3": dict(label="+ superconducting Hg walls", T=T_op, Gl=G0, amp=Phi_Hg, matter=0.0, nH=1.0),
    "C4": dict(label="+ Hg walls + solid H2 in the layer", T=T_op, Gl=G_sol, amp=F_sol_max * Phi_Hg, matter=sol["rho_m"] * C ** 2 * Leff, nH=sol["nH"]),
    "C4o": dict(label="+ Hg walls + solid H2 outside the layer", T=T_op, Gl=G0, amp=Phi_Hg / sol["eps_r"], matter=0.0, nH=1.0),
    "C4liq": dict(label="Hg walls + liquid H2 at 20 K", T=20.0, Gl=G_liq, amp=F_liq_max, matter=liq["rho_m"] * C ** 2 * Leff, nH=liq["nH"]),
}
# hydrogen amplitude factors at fixed (omega, V), relative to the bare bracket at r = 2
Hb = abs(float(np.sinh(r_ref) ** 2 - np.sinh(r_ref) * np.cosh(r_ref) * G0))
hfac = {}
for ph, Gh in (("solid", G_sol), ("liquid", G_liq)):
    Hh = abs(float(np.sinh(r_ref) ** 2 - np.sinh(r_ref) * np.cosh(r_ref) * Gh))
    hfac[ph] = dict(lower=Hh / Hb, upper=(1 + 0.5 * H2[ph]["Xi"]) * Hh / Hb, outside=1 / H2[ph]["eps_r"],
                    fixed_geometry=np.sqrt(H2[ph]["eps_r"]),
                    upper_fixed_geometry=np.sqrt(H2[ph]["eps_r"]) * (1 + 0.5 * H2[ph]["Xi"]) * Hh / Hb)

def casimir_sigma(nH=1.0):
    """Surface energy density of the ideal half-wave cavity vacuum in the layer (J/m^2); p = -sigma."""
    K = np.pi ** 2 * HBAR * C / 720
    return -K * nH ** 3 / Lc ** 4 * Leff


# ---------------------------------------------------------------- loaded cavity
# Shunt reactive element at x = +-d, outside the junction layer, with surface (per-square)
# capacitance C_box in farads: the sheet admittance is Y = -i omega C_box (siemens per square),
# so K = Y E_t has units A/m and omega C_box Z0 is dimensionless.
# Inner |x|<d: u = cos(kx).  Outer d<|x|<L/2: u = B sin(k(L/2-|x|)), zero at the walls.
# E continuous, u' jumps by -w^2 mu Cs u(d), giving the resonance condition below.
Z0 = 376.730313668
EPS0 = 8.8541878128e-12


def loaded_residual(k, d, Cs, nH=1.0):
    return np.tan(k * d) - 1 / np.tan(k * (Lc / 2 - d)) + k * (C / nH) * Cs * Z0 * nH


def loaded_k(d, Cs, nH=1.0):
    return brentq(lambda k: loaded_residual(k, d, Cs, nH), 1e-9, k0 * (1 - 1e-12), xtol=1e-15, rtol=8.9e-16)


def loaded_Cs(d, ratio, nH=1.0):
    """Shunt capacitance per unit area that puts the fundamental at ratio times the unloaded one."""
    return brentq(lambda Cs: loaded_k(d, Cs, nH) - ratio * k0, 1e-14, 1e-6, xtol=1e-22, rtol=8.9e-16)


def loaded_mode(d, r, nH=1.0):
    """Exact fundamental of the loaded cavity at omega_b = omega / cosh 2r.

    Energy normalization (Eq. 17): the electric energy at unit quadrature amplitude,
        W_E = (1/2) E0^2 A [ eps int |u|^2 dx + 2 C_box |u(d)|^2 ] = hbar omega_b / 4,
    the coefficient of X^2 in H = (hbar omega_b / 4)(X^2 + P^2) for [X, P] = 2i; at C_box = 0 it
    gives eps E0^2 = hbar omega / V.  rho below is eps E0^2 per unit transverse area, so only the
    ratio loaded/unloaded is used.  The element fraction is the second term over the whole bracket,
    i.e. the share of the electric energy held by the element.  Returns the layer wavenumber, that fraction, and the throat
    energy-density scale per quantum relative to the unloaded half-wave mode."""
    ratio = 1 / np.cosh(2 * r)
    Cs = loaded_Cs(d, ratio, nH)
    k = loaded_k(d, Cs, nH)
    B = np.cos(k * d) / np.sin(k * (Lc / 2 - d))
    inner = 2 * quad(lambda x: np.cos(k * x) ** 2, 0, d)[0]
    outer = 2 * quad(lambda x: (B * np.sin(k * (Lc / 2 - x))) ** 2, d, Lc / 2)[0]
    N_field = EPS0 * nH ** 2 * (inner + outer)
    N_cap = 2 * Cs * np.cos(k * d) ** 2
    rho_loaded = HBAR * (k * C / nH) * EPS0 * nH ** 2 / (2 * (N_field + N_cap))
    rho_unloaded = HBAR * omega * EPS0 * nH ** 2 / (2 * EPS0 * nH ** 2 * Lc / 2)
    # vacuum shift of the fundamental under the loading: Delta E = (hbar/2)(omega_b - omega)
    dE = 0.5 * HBAR * omega * (k / k0 - 1.0)
    sigma_vac = dE / V * Leff
    return dict(d=d, Cs=Cs, Cs_over_eps0L=Cs / (EPS0 * Lc), k_b=k, wb_over_w=k / k0,
                N_field=N_field, N_cap=N_cap, dE_vac=dE, sigma_vac=sigma_vac,
                R_vac_bound=2 * abs(sigma_vac) / israel_nec,
                decades_vac=np.log10(israel_nec / (2 * abs(sigma_vac))),
                f_b=k * C / (2 * np.pi), Ghat_b=np.exp(-2 * (k * ell) ** 2),
                energy_fraction_element=N_cap / (N_field + N_cap),
                rho_ratio=rho_loaded / rho_unloaded,
                decades=np.log10(rho_unloaded / rho_loaded),
                residual=loaded_residual(k, d, Cs, nH))


def planck_nec(T, nH=1.0):
    """(4/3) u_Planck L_eff: continuum estimate of the thermal-radiation null combination."""
    if T <= 0:
        return 0.0
    u = np.pi ** 2 / 15 * (KB * T) ** 4 / (HBAR * C) ** 3 * nH ** 3
    return 4 / 3 * u * Leff


def cavity_thermal_nec(T, nH=1.0, N=60):
    """Discrete-mode thermal null combination of the finite resonator: (4/3) L_eff times the energy
    density summed over the modes of a lam0 x lam0 x Lc box, indices with at most one zero, two
    polarizations when all three are nonzero.  Converged while kT stays well below the cutoff at N."""
    if T <= 0:
        return 0.0
    idx = np.arange(N)
    l = idx[:, None, None]; m = idx[None, :, None]; p_ = idx[None, None, :]
    z = (l == 0).astype(np.int8) + (m == 0).astype(np.int8) + (p_ == 0).astype(np.int8)
    w = (C / nH) * np.pi * np.sqrt((l / lam0) ** 2 + (m / lam0) ** 2 + (p_ / Lc) ** 2)
    pol = np.where(z == 1, 1.0, 2.0)
    x = np.where((z < 2) & (w > 0), HBAR * w / (KB * T), 1e3)
    nb = np.where(x < 300, 1.0 / np.expm1(np.clip(x, 1e-12, 300)), 0.0)
    u = float(np.sum(np.where((z < 2) & (w > 0), pol * HBAR * w * nb, 0.0))) / V
    return 4 / 3 * u * Leff


def thermal_nec(T, nH=1.0):
    """The discrete sum below 1 K, where the resonator holds only a few occupied modes, and the
    continuum estimate above it, where the sum has converged to Planck."""
    return cavity_thermal_nec(T, nH) if T < 1.0 else planck_nec(T, nH)


out = {}
for name, cfg in configs.items():
    H2r = float(H_T(r_ref, cfg["T"], cfg["Gl"]))
    ropt, Hopt = opt_r(cfg["T"], cfg["Gl"])
    nH = cfg["nH"]
    sigC = casimir_sigma(nH)
    th = thermal_nec(cfg["T"], nH)
    em_r2 = 2 * Leff * rho1 * cfg["amp"] * H2r
    em_opt = 2 * Leff * rho1 * cfg["amp"] * Hopt
    out[name] = dict(
        label=cfg["label"], T=cfg["T"], G=cfg["Gl"], amp=cfg["amp"],
        H_r2=H2r, r_opt=ropt, H_opt=Hopt,
        em_nec_r2=em_r2, em_nec_opt=em_opt,
        R_r2=R_of(H2r, cfg["amp"]) if H2r < 0 else 0.0,
        R_opt=R_of(Hopt, cfg["amp"]) if Hopt < 0 else 0.0,
        R6_opt=R_of(Hopt, cfg["amp"], 1e6) if Hopt < 0 else 0.0,
        casimir_sigma=sigC, casimir_nec=0.0, thermal_nec=th, matter_nec=cfg["matter"],
        total_nec_opt=(em_opt if Hopt < 0 else em_r2) + th + cfg["matter"],
        Tmax=T_max(cfg["Gl"])[0], r_at_Tmax=T_max(cfg["Gl"])[1],
    )

R1 = out["C1"]["R_r2"]
synergy_r2 = out["C4"]["R_r2"] * R1 / (out["C2"]["R_r2"] * out["C3"]["R_r2"])
synergy_opt = out["C4"]["R_opt"] * out["C1"]["R_opt"] / (out["C2"]["R_opt"] * out["C3"]["R_opt"])
synergy_o_r2 = out["C4o"]["R_r2"] * R1 / (out["C2o"]["R_r2"] * out["C3"]["R_r2"])
synergy_o_opt = out["C4o"]["R_opt"] * out["C1"]["R_opt"] / (out["C2o"]["R_opt"] * out["C3"]["R_opt"])

# thermal factors and other diagnostics
s, c = np.sinh(r_ref), np.cosh(r_ref)
wb_ref = omega / np.cosh(2 * r_ref)
Tb_ref = HBAR * wb_ref / KB
nb_op = 1 / np.expm1(HBAR * wb_ref / (KB * T_op))
Tnec_r2_bare = T_nec(r_ref, G0)
Tnec_r2_sol = T_nec(r_ref, G_sol)
Tmax_bare, rTmax_bare = T_max(G0)
Tmax_sol, rTmax_sol = T_max(G_sol)
n_liq_min = 1 / np.expm1(HBAR * omega / (KB * 13.8))
coupling_binding = -0.5 * HBAR * omega * np.tanh(2 * r_ref) * np.sinh(2 * r_ref)   # J per squeezed standing mode
em_energy_pair = 0.5 * HBAR * omega * (np.cosh(2 * r_ref) - 1)                     # J per squeezed standing mode
# QEI benchmark
lP2 = G * HBAR / C ** 3
R_qei = 3 / (4 * np.pi) * (a0 / Leff) * lP2 / Leff ** 2
qei_sigma = R_qei * israel_nec
ctau_over_xwall = Leff / x_wall
# cap (Corollary) and zero-temperature optimum
rstar0 = 0.5 * np.arctanh(G0); rmax0 = np.arctanh(G0); Hmax0 = 0.5 * (1 - np.sqrt(1 - G0 ** 2))

# width optimization at fixed (omega, V): maximize x |H|max(e^{-2x^2}), x = k0 ell
def _fw(x):
    return x * 0.5 * (1 - np.sqrt(1 - np.exp(-4 * x * x)))


_wres = minimize_scalar(lambda x: -_fw(x), bounds=(1e-3, 1.5), method="bounded", options=dict(xatol=1e-12))
x_w = _wres.x; f_w = -_wres.fun
G_w = np.exp(-2 * x_w ** 2); r_w = 0.5 * np.arctanh(G_w); ell_w = x_w / k0
R_wmax = 8 * np.pi * G * a0 / C ** 4 * rho1 / k0 * f_w
R_qei_w = 3 / (4 * np.pi) * (a0 / ell_w) * (G * HBAR / C ** 3) / ell_w ** 2
Tmax_w = T_max(G_w)[0]


# parametrically driven alternative (rotating frame, walls as the only loss): intracavity variances
# Theta_w/(1 +- eps); the projected bracket is minimized at eps* = (1-q)/(1+q), q = sqrt((1-G)/(1+G)),
# where Theta_w (1+G)(1+q)^2/4 < 1 is required.
q_drv = np.sqrt((1 - G0) / (1 + G0))
S_drv = (1 + G0) * (1 + q_drv) ** 2 / 4
A_drv0 = 0.5 * (S_drv - 1)                        # T = 0 bracket at the optimal pump
Theta_star = 1 / S_drv
n_star = (Theta_star - 1) / 2
Tmax_drv = HBAR * omega / (KB * np.log(1 + 1 / n_star))
Tmax_drv_limit = HBAR * omega / (KB * np.log(3.0))


def dielectric_width_factor(chi):
    """Most favorable dressing factor of the width-optimized ratio, (1 + Xi/2)/sqrt(1 + chi)."""
    Xi = 3 * chi / (chi + 3)
    return (1 + 0.5 * Xi) / np.sqrt(1 + chi)


# dressed-mode damping: the walls couple through the tangential H, proportional to P = e^r P_beta, and
# sample the bath at omega_b: gamma_beta = gamma e^{2r} cosh 2r R_s(omega_b)/R_s(omega)
enh_beta = np.exp(2 * r_ref) * np.cosh(2 * r_ref)
gb_sc = [enh_beta * g for g in gamma_sc]                       # residual R_s taken frequency independent (upper bound)
gb_over_wb_sc = [g / wb_ref for g in gb_sc]
gb_over_wb_293 = enh_beta * np.sqrt(wb_ref / omega) * (omega / Q293) / wb_ref
# QEI locality: the stationary antinode density must obey the free-field bound for every tau < d/c
# completed source: both polarizations, rho(0) = (hbar w / V)(Theta e^{-2r} - 1) per mode pair, T = 0
def rho_antinode(r):
    return 2 * rho1 * (np.sinh(r) ** 2 - np.sinh(r) * np.cosh(r))


def d_qei(rho):
    return (3 * HBAR * C / (16 * np.pi ** 2 * abs(rho))) ** 0.25


rho_anti = rho_antinode(r_ref)                    # J/m^3 at the antinode, C1, r = 2, one mode pair
d_qei_fr = d_qei(rho_anti)
d_qei_pf = d_qei_fr * (9 / 64) ** 0.25
a_layer = np.sqrt(3) * ell
N_close_fr = (d_qei_fr / a_layer) ** 4
N_close_pf = (d_qei_pf / a_layer) ** 4
rho_anti_w = rho_antinode(r_w)                    # width optimum, r = 0.61
d_qei_w = d_qei(rho_anti_w)
a_layer_w = np.sqrt(3) * ell_w
qei_excess_6 = np.log10(1e6 * out["C1"]["R_r2"] / R_qei)
qei_excess_6_pf = np.log10(1e6 * out["C1"]["R_r2"] / (R_qei * 9 / 64))
qei_excess_w = R_wmax / R_qei_w
qei_excess_w_pf = R_wmax / (R_qei_w * 9 / 64)
R_qei_pf = R_qei * 9 / 64
casimir_filled_bound = 2 * abs(casimir_sigma(H2["solid"]["nH"]))
# top-hat (compact) window: G = sin(2 k a)/(2 k a), a = sqrt(3) ell
G_tophat = np.sin(2 * np.sqrt(3) * k0l) / (2 * np.sqrt(3) * k0l)
_th = minimize_scalar(lambda u: -u * 0.5 * (1 - np.sqrt(1 - (np.sin(2 * np.sqrt(3) * u) / (2 * np.sqrt(3) * u)) ** 2)),
                      bounds=(1e-3, 0.8), method="bounded", options=dict(xatol=1e-12))
f_tophat = -_th.fun
# hydrogen at fixed transverse area: half-wave cavity shortens by n_H, V by n_H
h_fixed_area_opt = sol["nH"] * out["C2"]["R_opt"] / out["C1"]["R_opt"]
bcs_at_100mK = np.exp(-Delta_Hg / (KB * 0.1))

res = dict(
    extra=dict(enh_beta=enh_beta, gb_over_wb_sc=gb_over_wb_sc, gb_over_wb_293=gb_over_wb_293,
               t_relax_min=[1 / g for g in gb_sc], rho_anti=rho_anti, d_qei_fr=d_qei_fr, d_qei_pf=d_qei_pf,
               a_layer=a_layer, N_close_fr=N_close_fr, N_close_pf=N_close_pf, R_qei_pf=R_qei_pf,
               rho_anti_w=rho_anti_w, d_qei_w=d_qei_w, a_layer_w=a_layer_w, qei_excess_6=qei_excess_6,
               qei_excess_6_pf=qei_excess_6_pf, qei_excess_w=qei_excess_w, qei_excess_w_pf=qei_excess_w_pf,
               R_qei_w_pf=R_qei_w * 9 / 64, casimir_sigma_solid=casimir_sigma(H2["solid"]["nH"]),
               casimir_decades_below=np.log10(israel_nec / abs(casimir_sigma(1.0))),
               Gw_tophat_cap=8 * np.pi * G * a0 / C ** 4 * rho1 / k0 * f_tophat,
               qei_sigma_pf=qei_sigma * 9 / 64, casimir_filled_bound=casimir_filled_bound, G_tophat=G_tophat,
               f_tophat=f_tophat, h_fixed_area_opt=h_fixed_area_opt, bcs_at_100mK=bcs_at_100mK,
               a_cube=a_cube, V_cube=V_cube, cube_decades=np.log10(V / V_cube)),
    width=dict(x_w=x_w, f_w=f_w, G_w=G_w, r_w=r_w, ell_w=ell_w, R_wmax=R_wmax, R_qei_w=R_qei_w, Tmax_w=Tmax_w,
               gain_over_C1opt=R_wmax / out["C1"]["R_opt"], diel_solid=dielectric_width_factor(H2["solid"]["chi"]),
               diel_liquid=dielectric_width_factor(H2["liquid"]["chi"])),
    driven=dict(q=q_drv, S=S_drv, A0=A_drv0, Tmax=Tmax_drv, Tmax_limit=Tmax_drv_limit),
    constants=dict(Z0=Z0),
    reference=dict(omega=omega, k0=k0, k0l=k0l, G0=G0, rho1=rho1, Lc=Lc, x_wall=x_wall, lam0=lam0,
                   israel_nec=israel_nec, israel_sigma=israel_sigma, r_ref=r_ref, T_op=T_op,
                   rstar0=rstar0, rmax0=rmax0, Hmax0=Hmax0, H_ref=float(H_T(r_ref, 0, G0)),
                   tanh_r=np.tanh(r_ref), window0=np.sqrt(-0.5 * np.log(np.tanh(r_ref)))),
    hydrogen=H2, C_uniform=C_uniform, F_sol=F_sol, F_liq=F_liq, F_sol_max=F_sol_max, F_liq_max=F_liq_max,
    G_sol=G_sol, G_liq=G_liq,
    mercury=dict(rho_293=rho_Hg_293, sigma_293=sig_Hg_293, Rs_293=Rs_293, delta_293=delta_293,
                 Q293=Q293, Q293_closed=Q293_closed, dL293=dL293, Tc=Tc_Hg, Delta=Delta_Hg,
                 omega_p=omega_p, hbar_omega_p_eV=HBAR * omega_p / QE, lam_L=lam_L, Qsc=Qsc,
                 dL_sc=dL_sc, frac_wall_energy_sc=frac_wall_energy_sc, frac_wall_energy_n=frac_wall_energy_n,
                 gamma_sc=gamma_sc, Phi_Hg=Phi_Hg,
                 bcs_suppression_at_Tmax=np.exp(-Delta_Hg / (KB * T_max(G0)[0]))),
    thermal=dict(wb_ref=wb_ref, Tb_ref=Tb_ref, nb_op=nb_op, Tnec_r2_bare=Tnec_r2_bare,
                 Tnec_r2_sol=Tnec_r2_sol, Tmax_bare=Tmax_bare, rTmax_bare=rTmax_bare,
                 Tmax_sol=Tmax_sol, rTmax_sol=rTmax_sol, n_liq_min=n_liq_min,
                 Thbar=HBAR * omega / KB),
    sectors=dict(coupling_binding_per_pair=coupling_binding, em_energy_pair=em_energy_pair,
                 casimir_sigma_vac=casimir_sigma(1.0)),
    qei=dict(R_qei=R_qei, qei_sigma=qei_sigma, ctau_over_xwall=ctau_over_xwall),
    loaded=loaded_mode(5e-3, r_ref),
    geometry=dict(V=V, A_patch=A_patch, lam0=lam0, Lc=Lc, a_cube=a_cube, V_cube=V_cube,
                  decades_vs_cube=np.log10(V / V_cube)),
    configs=out, synergy_r2=synergy_r2, synergy_opt=synergy_opt, synergy_o_r2=synergy_o_r2,
    synergy_o_opt=synergy_o_opt, hfac=hfac,
)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    return o


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_results.json"), "w") as fh:
    json.dump(_clean(res), fh, indent=1)

if __name__ == "__main__":
    print("k0 ell = %.4f  G0 = %.5f  rho1 = %.3e  Lc = %.4f m" % (k0l, G0, rho1, Lc))
    print("r* = %.3f  rmax = %.3f  Hmax = %.4f  H(2) = %.4f  frac = %.3f" % (rstar0, rmax0, Hmax0, H_T(2, 0, G0), abs(H_T(2, 0, G0)) / Hmax0))
    print("Hg 293 K: Rs = %.4f Ohm, delta = %.2f um, Q = %.0f (closed %.0f), dL = %.2f um" % (Rs_293, delta_293 * 1e6, Q293, Q293_closed, dL293 * 1e6))
    print("Hg SC: lam_L = %.1f nm, hbar w_p = %.2f eV, Q = %.2e .. %.2e, gamma = %.2f .. %.3g 1/s, wall energy frac %.2e" % (lam_L * 1e9, HBAR * omega_p / QE, Qsc[0], Qsc[1], gamma_sc[0], gamma_sc[1], frac_wall_energy_sc))
    for ph in ("liquid", "solid"):
        h = H2[ph]
        print("H2 %s: n=%.3e eps=%.4f chi=%.4f Xi=%.4f" % (ph, h["n"], h["eps_r"], h["chi"], h["Xi"]))
    print("G_sol=%.5f G_liq=%.5f F_sol=%.4f (max %.4f) F_liq=%.4f (max %.4f) C_unif=%.4f" % (G_sol, G_liq, F_sol, F_sol_max, F_liq, F_liq_max, C_uniform))
    print("w_b(r=2)/2pi = %.1f MHz, hbar w_b/kB = %.2f mK, n_b(10 mK) = %.3f" % (wb_ref / 2 / np.pi / 1e6, Tb_ref * 1e3, nb_op))
    print("T_NEC(r=2): bare %.2f mK, solid %.2f mK" % (Tnec_r2_bare * 1e3, Tnec_r2_sol * 1e3))
    print("T_max: bare %.2f mK at r=%.3f; solid %.2f mK at r=%.3f" % (Tmax_bare * 1e3, rTmax_bare, Tmax_sol * 1e3, rTmax_sol))
    print("n_th at 13.8 K = %.1f; hbar w/kB = %.4f K" % (n_liq_min, HBAR * omega / KB))
    for k, v in out.items():
        print("%-6s T=%7.3f G=%.5f amp=%.5f H(2)=%+.4f r_opt=%.3f H_opt=%+.4f R(2)=%.3e R_opt=%.3e R6=%.3e casimir=%.2e thermal=%.2e matter=%.2e total=%.2e Tmax=%.1f mK" % (
            k, v["T"], v["G"], v["amp"], v["H_r2"], v["r_opt"], v["H_opt"], v["R_r2"], v["R_opt"], v["R6_opt"], v["casimir_sigma"], v["thermal_nec"], v["matter_nec"], v["total_nec_opt"], v["Tmax"] * 1e3))
    print("synergy in-layer (r=2) = %.4f, (opt) = %.4f; outside (r=2) = %.4f, (opt) = %.4f" % (synergy_r2, synergy_opt, synergy_o_r2, synergy_o_opt))
    for ph, h in hfac.items():
        print("hydrogen %s factors: lower %.4f upper %.4f outside %.4f fixed-geom %.4f upper-fixed-geom %.4f" % (ph, h["lower"], h["upper"], h["outside"], h["fixed_geometry"], h["upper_fixed_geometry"]))
    print("coupling binding per mode = %.2f hbar w; EM energy per mode = %.2f hbar w" % (coupling_binding / (HBAR * omega), em_energy_pair / (HBAR * omega)))
    print("QEI: R = %.3e, sigma = %.3e, ctau/x_wall = %.4f" % (R_qei, qei_sigma, ctau_over_xwall))
    _lm = loaded_mode(5e-3, r_ref)
    print("loaded cavity d=5 mm: C_box=%.4e F/square (%.0f eps0 L), f_b=%.3f MHz, k_b=%.4f rad/m, Ghat=%.7f, element holds %.4f of the energy, throat scale %.3e (%.2f decades), residual %.1e" % (
        _lm["Cs"], _lm["Cs_over_eps0L"], _lm["f_b"] / 1e6, _lm["k_b"], _lm["Ghat_b"], _lm["energy_fraction_element"], _lm["rho_ratio"], _lm["decades"], _lm["residual"]))
    print("passive vacuum shift: dE=%.4e J, sigma_C=%.4e J/m^2, R bound %.3e (%.2f decades)" % (
        _lm["dE_vac"], _lm["sigma_vac"], _lm["R_vac_bound"], _lm["decades_vac"]))
    print("conditional ratio with the loaded normalization: %.3e (separation %.2f decades)" % (
        out["C1"]["R_opt"] * _lm["rho_ratio"], np.log10(1 / (out["C1"]["R_opt"] * _lm["rho_ratio"]))))
    print("reference V = %.3e m^3 (patch %.1f x %.1f cm, L = %.2f cm); cube side %.1f cm, V_cube = %.3e (%.2f decades)" % (V, lam0*100, lam0*100, Lc*100, a_cube*100, V_cube, np.log10(V/V_cube)))
    print("gamma_beta enhancement %.0f; gb/wb SC %s; gb/wb 293 K %.2f; min relaxation times %s s" % (enh_beta, gb_over_wb_sc, gb_over_wb_293, [1 / g for g in gb_sc]))
    print("antinode rho (pair) %.3e J/m^3; d_QEI FR %.2f mm, Pfenning %.2f mm; layer half-width %.2f mm; N closing FR %.1f, Pf %.2f" % (rho_anti, d_qei_fr * 1e3, d_qei_pf * 1e3, a_layer * 1e3, N_close_fr, N_close_pf))
    print("width optimum: rho(0) %.3e, d_QEI %.2f mm vs a %.2f mm; QEI excess 1e6 modes %.2f (Pf %.2f) decades; width-opt excess %.2f (Pf %.1f); tophat cap %.2e" % (rho_anti_w, d_qei_w * 1e3, a_layer_w * 1e3, qei_excess_6, qei_excess_6_pf, qei_excess_w, qei_excess_w_pf, 8 * np.pi * G * a0 / C ** 4 * rho1 / k0 * f_tophat))
    print("QEI Pfenning ceiling %.3e, sigma %.3e; Casimir filled bound %.2e; G tophat %.6f vs %.6f; f tophat %.5f" % (R_qei_pf, qei_sigma * 9 / 64, casimir_filled_bound, G_tophat, G0, f_tophat))
    print("hydrogen fixed-area factor at optimum %.4f (%.3f decades); BCS factor at 0.1 K %.2e" % (h_fixed_area_opt, np.log10(h_fixed_area_opt), bcs_at_100mK))
    print("Casimir below requirement: %.1f decades" % np.log10(israel_nec / abs(out["C1"]["casimir_sigma"])))
    print("width optimum: k0 ell = %.4f, f = %.5f, G = %.4f, r = %.4f, ell = %.3f mm, R = %.3e (x%.3f over C1 opt), QEI at ell* = %.3e, Tmax = %.2f mK" % (x_w, f_w, G_w, r_w, ell_w * 1e3, R_wmax, R_wmax / out["C1"]["R_opt"], R_qei_w, Tmax_w * 1e3))
    print("dielectric width factor: solid %.5f liquid %.5f" % (dielectric_width_factor(H2["solid"]["chi"]), dielectric_width_factor(H2["liquid"]["chi"])))
    print("driven: q=%.4f S=%.4f |A0|=%.4f Tmax=%.2f mK (thin-layer limit %.2f mK)" % (q_drv, S_drv, abs(A_drv0), Tmax_drv * 1e3, Tmax_drv_limit * 1e3))
    for kl in (0.001, 0.01, 0.042, 0.1, 0.13):
        Gk = np.exp(-2 * kl ** 2); Tk, rk = T_max(Gk)
        print("  table: k0l=%.3f Tmax=%.2f mK r=%.3f ratio=%.3f" % (kl, Tk * 1e3, rk, Tk / (HBAR * omega / KB)))
    print("israel nec %.3e, sigma %.3e; decades C1: %.2f" % (israel_nec, israel_sigma, np.log10(israel_nec / abs(out['C1']['em_nec_r2']))))
