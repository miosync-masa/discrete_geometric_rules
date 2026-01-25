# =============================================================================
# CKM GEOMETRIC SELECTION — PUBLICATION-GRADE FINAL SCRIPT (Reviewer-Resistant)
# =============================================================================
# Features:
#  [A] chi2 uses moduli directly: |V_us|,|V_ub|,|V_cb|,|V_td|,|J| (optionally |J|)
#  [B] kappa_v lattice stored as Fractions (exact rationals)
#  [C] Multi-start local fits per lattice point + bounded Powell
#  [D] Evidence weights: w(theta)=exp(-(chi2_profile(theta)-chi2_min)/2)
#  [E] Full-lattice toys (few) + leave-one-out cross-validation (cos30 vs cos35)
#
# Output:
#  - Profile best chi2 per ratio angle (+best kv, phi, params)
#  - Delta-chi2 and evidence weights
#  - Toy winners + toy evidence summary
#  - Leave-one-out table: cos30 vs cos35 (prediction/pull)
# =============================================================================

import numpy as np
from scipy.optimize import minimize
from fractions import Fraction
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG (edit here)
# -----------------------------
ANGLES_DEG = list(range(5, 86, 5))          # ratio angles (deg)
DENOM_MAX = 25                               # Fraction denom bound for kappa_v
KV_MIN, KV_MAX = 1.20, 1.28                  # range for kappa_v
PHI_N_MIN, PHI_N_MAX = 40, 50                # phi candidates: -2pi/n

# Continuous parameters bounds (reviewer-friendly: explicit)
EPS_BOUNDS = (0.0, 6.0)                      # eps_12
K23U_BOUNDS = (4.0, 12.0)                    # kappa_23_u

# Optimizer settings
METHOD = "Powell"                            # bounded Powell is robust for 2D
N_STARTS_REAL = 20                           # multi-start per lattice point (real data)
N_STARTS_TOY  = 4                            # cheaper multi-start per lattice point (toy)
MAXITER = 2500

# Toy study
N_TOYS = 10                                  # full-lattice toys (keep small)
TOY_SEED0 = 12345

# Leave-one-out (cos30 vs cos35)
DO_LOO = True
LOO_ANGLES = [30, 35]                        # compare cos(30°) vs cos(35°)
N_STARTS_LOO = 12

# Convention handling
USE_ABS_J = True                             # phase convention safe
CLIP_TOY_MODULI_POSITIVE = True              # avoid negative toy draws for moduli-like obs

# -----------------------------
# Physics / model core
# -----------------------------
PI = np.pi
t_r = 1.0
t_i = (180.0 / PI) * t_r
v_base = np.sqrt(15.0) * t_i

def build_H_complex_t23(t_12, t_23_mag, t_23_phase, t_13_imag, v):
    t_23 = t_23_mag * np.exp(1j * t_23_phase)
    return np.array([
        [0,       t_12,         1j * t_13_imag],
        [t_12,    0,            t_23],
        [-1j*t_13_imag, np.conj(t_23), -v]
    ], dtype=complex)

def diagonalize_and_sort(H):
    w, U = np.linalg.eigh(H)
    idx = np.abs(w).argsort()
    return w[idx], U[:, idx]

def compute_jarlskog(V):
    return np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0]))

def model_moduli(eps_12, kappa_23_u, ratio, kv, phi_d):
    """Return (V_us, V_ub, V_cb, V_td, J) from the model."""
    kappa_23_d = kappa_23_u * ratio

    H_u = build_H_complex_t23(t_r, kappa_23_u*t_r, 0.0, t_i, v_base)
    _, U_u = diagonalize_and_sort(H_u)

    H_d = build_H_complex_t23(t_r*(1.0+eps_12), kappa_23_d*t_r, phi_d, t_i, v_base*kv)
    _, U_d = diagonalize_and_sort(H_d)

    V = U_u.conj().T @ U_d

    V_us = np.abs(V[0,1])
    V_ub = np.abs(V[0,2])
    V_cb = np.abs(V[1,2])
    V_td = np.abs(V[2,0])

    J = compute_jarlskog(V)
    if USE_ABS_J:
        J = abs(J)

    return V_us, V_ub, V_cb, V_td, J

def chi2_from_moduli(obs, exp):
    """obs=(Vus,Vub,Vcb,Vtd,J), exp dict key->(mean,sigma)"""
    V_us, V_ub, V_cb, V_td, J = obs
    return (
        (V_us-exp["V_us"][0])**2/exp["V_us"][1]**2 +
        (V_ub-exp["V_ub"][0])**2/exp["V_ub"][1]**2 +
        (V_cb-exp["V_cb"][0])**2/exp["V_cb"][1]**2 +
        (V_td-exp["V_td"][0])**2/exp["V_td"][1]**2 +
        (J   -exp["J"][0])**2   /exp["J"][1]**2
    )

def chi2_excluding_key(obs, exp, drop_key):
    """same as chi2_from_moduli but excluding one observable (leave-one-out training)."""
    V_us, V_ub, V_cb, V_td, J = obs
    terms = []
    if drop_key != "V_us":
        terms.append((V_us-exp["V_us"][0])**2/exp["V_us"][1]**2)
    if drop_key != "V_ub":
        terms.append((V_ub-exp["V_ub"][0])**2/exp["V_ub"][1]**2)
    if drop_key != "V_cb":
        terms.append((V_cb-exp["V_cb"][0])**2/exp["V_cb"][1]**2)
    if drop_key != "V_td":
        terms.append((V_td-exp["V_td"][0])**2/exp["V_td"][1]**2)
    if drop_key != "J":
        terms.append((J-exp["J"][0])**2/exp["J"][1]**2)
    return float(np.sum(terms))

# -----------------------------
# Lattice generation (Fractions)
# -----------------------------
def make_kv_fractions(denom_max=DENOM_MAX, vmin=KV_MIN, vmax=KV_MAX):
    fracs = set()
    for d in range(1, denom_max+1):
        # numer search window: a bit wide, but cheap
        n_min = int(np.floor(vmin*d)) - 2
        n_max = int(np.ceil(vmax*d)) + 2
        for n in range(max(1,n_min), max(2,n_max)+1):
            f = Fraction(n, d)
            val = float(f)
            if vmin < val < vmax:
                fracs.add(f)  # reduced automatically
    return sorted(fracs, key=lambda x: float(x))

def make_phi_candidates(nmin=PHI_N_MIN, nmax=PHI_N_MAX):
    return [(-2.0*PI/n, n) for n in range(nmin, nmax+1)]

def make_ratio_candidates(angles_deg=ANGLES_DEG):
    return [(np.cos(np.radians(d)), d) for d in angles_deg]

# -----------------------------
# Optimization helpers
# -----------------------------
def rng_from_tuple(*items):
    # stable seed from lattice point identity (reviewer-friendly determinism)
    s = 0
    for it in items:
        if isinstance(it, Fraction):
            s ^= (it.numerator*1000003 + it.denominator*9176)
        elif isinstance(it, (int, np.integer)):
            s ^= int(it)*2654435761
        else:
            s ^= int(abs(float(it))*1e6) * 97531
    return np.random.default_rng(s & 0xFFFFFFFF)

def optimize_point(ratio, kv, phi, exp, n_starts, drop_key=None):
    """
    Fit continuous params (eps_12, kappa_23_u) at fixed (ratio,kv,phi).
    - If drop_key is not None: train chi2 excludes that key (LOO training).
    Returns: best_chi2, best_params (eps_12, kappa_23_u)
    """
    bounds = [EPS_BOUNDS, K23U_BOUNDS]
    best = (np.inf, None)

    local_rng = rng_from_tuple(ratio, kv, phi, n_starts, 777)

    for s in range(n_starts):
        # jittered starts within a reasonable basin
        eps0 = local_rng.uniform(2.2, 3.9)
        k230 = local_rng.uniform(5.2, 9.0)
        x0 = np.array([eps0, k230], dtype=float)

        def obj(x):
            eps, k23u = float(x[0]), float(x[1])
            obs = model_moduli(eps, k23u, ratio, float(kv), phi)
            if drop_key is None:
                return chi2_from_moduli(obs, exp)
            return chi2_excluding_key(obs, exp, drop_key)

        try:
            res = minimize(
                obj, x0,
                method=METHOD,
                bounds=bounds,
                options={"maxiter": MAXITER, "ftol": 1e-10, "xtol": 1e-10}
            )
            val = float(res.fun)
            if np.isfinite(val) and val < best[0]:
                best = (val, res.x.copy())
        except Exception:
            pass

    return best

def profile_scan(exp, n_starts, ratios, kv_fracs, phis, drop_key=None):
    """
    Full profile scan over angles:
      chi2_profile(theta) = min_{kv,phi,eps,k23u} chi2
    Returns list of dict per angle with best info.
    """
    out = []
    for ratio, deg in ratios:
        best_for_angle = {"deg": deg, "ratio": ratio, "chi2": np.inf}
        for kv in kv_fracs:
            for phi, n in phis:
                chi2, params = optimize_point(ratio, kv, phi, exp, n_starts, drop_key=drop_key)
                if chi2 < best_for_angle["chi2"]:
                    best_for_angle.update({
                        "chi2": chi2,
                        "kv": kv,
                        "phi": phi,
                        "phi_n": n,
                        "params": params
                    })
        out.append(best_for_angle)
    return out

def evidence_weights(profile):
    """
    Convert profile chi2 into normalized evidence-like weights:
      w_i ∝ exp(-(chi2_i - chi2_min)/2)
    """
    chi2s = np.array([p["chi2"] for p in profile], dtype=float)
    chi2_min = np.min(chi2s)
    w = np.exp(-0.5*(chi2s - chi2_min))
    w = w / np.sum(w)
    return chi2_min, w

# -----------------------------
# PDG inputs (edit if needed)
# -----------------------------
PDG = {
    "V_us": (0.2243, 0.0008),
    "V_ub": (0.00382, 0.00020),
    "V_cb": (0.0408, 0.0014),
    "V_td": (0.0086, 0.0002),
    "J":    (3.08e-5, 0.15e-5),
}

def toy_experiment(pdg, rng):
    toy = {}
    for k, (mu, sig) in pdg.items():
        x = mu + rng.normal(0.0, sig)
        if CLIP_TOY_MODULI_POSITIVE and k != "J":
            x = max(1e-12, x)
        if CLIP_TOY_MODULI_POSITIVE and k == "J" and USE_ABS_J:
            x = abs(x)
        toy[k] = (x, sig)
    return toy

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*90)
    print("CKM GEOMETRIC SELECTION — PUBLICATION-GRADE FINAL SCRIPT")
    print("="*90)
    print("\nKey reviewer-resistance choices:")
    print("  [A] chi2 uses moduli directly: |V_us|,|V_ub|,|V_cb|,|V_td|,|J|")
    print("  [B] kappa_v lattice stored as Fractions (exact rationals)")
    print("  [C] Multi-start local fits per lattice point + bounded Powell")
    print("  [D] Evidence weights from profile: w(theta)=exp(-(Δchi2)/2)")
    print("  [E] Full-lattice toys (few) + leave-one-out (cos30 vs cos35)")

    ratios = make_ratio_candidates()
    kv_fracs = make_kv_fractions()
    phis = make_phi_candidates()

    print("\n[LATTICE]")
    print(f"  Ratio angles: {len(ratios)} candidates (5° steps)")
    print(f"  kappa_v rationals: {len(kv_fracs)} candidates (Fraction, denom≤{DENOM_MAX}, {KV_MIN}<kv<{KV_MAX})")
    print(f"  phi_d: {len(phis)} candidates (-2π/n, n={PHI_N_MIN}..{PHI_N_MAX})")
    print(f"  TOTAL lattice points overall: {len(ratios)*len(kv_fracs)*len(phis)}")
    print(f"  Starts (real): {N_STARTS_REAL}  |  Starts (toy): {N_STARTS_TOY}")

    # -------------------------------------------------------------------------
    # PART 1: Real-data full profile scan
    # -------------------------------------------------------------------------
    print("\n" + "="*90)
    print("PART 1 — FULL PROFILE SCAN (REAL DATA)")
    print("="*90)

    profile = profile_scan(PDG, N_STARTS_REAL, ratios, kv_fracs, phis, drop_key=None)
    profile_sorted = sorted(profile, key=lambda d: d["deg"])

    # Print best per angle
    print("\n[Best per angle] (profile chi2_min(theta))")
    for p in profile_sorted:
        kv = p.get("kv", None)
        phi_n = p.get("phi_n", None)
        star = " ⭐" if p["chi2"] == min([x["chi2"] for x in profile_sorted]) else ""
        if kv is not None:
            print(f"  cos({p['deg']:>2}°): chi2={p['chi2']:.3f}   kv={str(kv):>7}  phi=-2π/{phi_n:<2}{star}")
        else:
            print(f"  cos({p['deg']:>2}°): chi2={p['chi2']:.3f}{star}")

    # Global best
    best = min(profile_sorted, key=lambda d: d["chi2"])
    eps_best, k23u_best = best["params"]
    obs_best = model_moduli(eps_best, k23u_best, best["ratio"], float(best["kv"]), best["phi"])

    print("\n[GLOBAL BEST]")
    print(f"  Winner angle: {best['deg']}° (cos({best['deg']}°))")
    print(f"  chi2 = {best['chi2']:.3f}")
    print(f"  kv = {best['kv']}  (={float(best['kv']):.6f})")
    print(f"  phi = -2π/{best['phi_n']} (={np.degrees(best['phi']):.4f}°)")
    print(f"  eps_12 = {eps_best:.6f}")
    print(f"  kappa_23_u = {k23u_best:.6f}")
    print(f"  model: |V_us|={obs_best[0]:.6f}, |V_ub|={obs_best[1]:.6f}, |V_cb|={obs_best[2]:.6f}, |V_td|={obs_best[3]:.6f}, |J|={obs_best[4]:.3e}")

    # Evidence weights
    chi2_min, w = evidence_weights(profile_sorted)
    degs = [p["deg"] for p in profile_sorted]
    w_map = {d: float(wi) for d, wi in zip(degs, w)}

    # Delta chi2 & evidence summary (cos30 vs cos35 if present)
    def find_deg(d):
        for p in profile_sorted:
            if p["deg"] == d:
                return p
        return None

    p30 = find_deg(30)
    p35 = find_deg(35)
    if p30 and p35:
        dchi2 = p35["chi2"] - p30["chi2"]
        bf_like = np.exp(+0.5*dchi2)  # w30/w35 = exp((chi2_35-chi2_30)/2)
        print("\n[Δχ² & Evidence] (profile)")
        print(f"  chi2(30°) = {p30['chi2']:.3f}")
        print(f"  chi2(35°) = {p35['chi2']:.3f}")
        print(f"  Δχ²(35-30) = {dchi2:.3f}")
        print(f"  Bayes-factor-like w30/w35 = exp(Δχ²/2) = {bf_like:.3f}")
        print(f"  w_norm(30°) = {w_map[30]:.4f}")
        print(f"  w_norm(35°) = {w_map[35]:.4f}")

    # -------------------------------------------------------------------------
    # PART 2: Full-lattice Toy Study (few toys)
    # -------------------------------------------------------------------------
    print("\n" + "="*90)
    print("PART 2 — FULL-LATTICE TOY STUDY (FEW TOYS, EACH TOY RUNS FULL LATTICE)")
    print("="*90)
    print(f"  N_TOYS = {N_TOYS}")
    print(f"  N_STARTS_TOY = {N_STARTS_TOY} (cheap multi-start per lattice point)")

    toy_winners = []
    toy_w30 = []
    toy_w35 = []
    toy_dchi2_35_30 = []

    rng = np.random.default_rng(TOY_SEED0)

    for t in range(N_TOYS):
        toy_exp = toy_experiment(PDG, rng)

        prof_toy = profile_scan(toy_exp, N_STARTS_TOY, ratios, kv_fracs, phis, drop_key=None)
        best_toy = min(prof_toy, key=lambda d: d["chi2"])
        toy_winners.append(best_toy["deg"])

        # Evidence weights for toys (useful when 30/35 are close)
        prof_toy_sorted = sorted(prof_toy, key=lambda d: d["deg"])
        _, w_toy = evidence_weights(prof_toy_sorted)
        w_toy_map = {d: float(wi) for d, wi in zip([p["deg"] for p in prof_toy_sorted], w_toy)}

        toy_w30.append(w_toy_map.get(30, 0.0))
        toy_w35.append(w_toy_map.get(35, 0.0))

        # Δχ²(35-30) for each toy (if both exist)
        p30t = next((p for p in prof_toy_sorted if p["deg"] == 30), None)
        p35t = next((p for p in prof_toy_sorted if p["deg"] == 35), None)
        if p30t and p35t:
            toy_dchi2_35_30.append(p35t["chi2"] - p30t["chi2"])

        print(f"  Toy {t+1:>2}/{N_TOYS}: winner={best_toy['deg']}°, chi2={best_toy['chi2']:.3f}")

    cnt = Counter(toy_winners)
    print("\n[Toy winners]")
    for d in sorted(cnt.keys()):
        print(f"  {d:>2}°: {cnt[d]}/{N_TOYS} = {100*cnt[d]/N_TOYS:.1f}%")

    if len(toy_dchi2_35_30) > 0:
        arr = np.array(toy_dchi2_35_30, dtype=float)
        print("\n[Toy Δχ² diagnostics]  (Δχ² = chi2(35)-chi2(30))")
        print(f"  mean Δχ² = {arr.mean():.3f}")
        print(f"  median Δχ² = {np.median(arr):.3f}")
        print(f"  P(Δχ²>0) = {100*np.mean(arr>0):.1f}%  (30° preferred)")
        print(f"  Avg evidence weight: <w30>={np.mean(toy_w30):.3f}, <w35>={np.mean(toy_w35):.3f}")

    # -------------------------------------------------------------------------
    # PART 3: Leave-one-out extrapolation table (cos30 vs cos35)
    # -------------------------------------------------------------------------
    if DO_LOO:
        print("\n" + "="*90)
        print("PART 3 — LEAVE-ONE-OUT EXTRAPOLATION (cos30 vs cos35)")
        print("="*90)
        keys = ["V_us", "V_ub", "V_cb", "V_td", "J"]

        def fit_fixed_angle(deg, exp, drop_key):
            """Profile-min over kv,phi,eps,k23u at fixed ratio angle with one observable removed."""
            ratio = np.cos(np.radians(deg))
            best_here = {"deg": deg, "chi2_train": np.inf}
            for kv in kv_fracs:
                for phi, n in phis:
                    chi2, params = optimize_point(ratio, kv, phi, exp, N_STARTS_LOO, drop_key=drop_key)
                    if chi2 < best_here["chi2_train"]:
                        best_here.update({
                            "chi2_train": chi2,
                            "kv": kv,
                            "phi": phi,
                            "phi_n": n,
                            "params": params
                        })
            return best_here

        print("\nColumns:")
        print("  left_out | angle | chi2_train | pred(left_out) | pull | (optional) chi2_full")
        print("-"*90)

        # store summary for paper-ready delta comparisons
        loo_rows = []

        for drop in keys:
            for deg in LOO_ANGLES:
                fit = fit_fixed_angle(deg, PDG, drop_key=drop)
                eps, k23u = fit["params"]
                pred = model_moduli(eps, k23u, np.cos(np.radians(deg)), float(fit["kv"]), fit["phi"])

                pred_map = {"V_us": pred[0], "V_ub": pred[1], "V_cb": pred[2], "V_td": pred[3], "J": pred[4]}
                mu, sig = PDG[drop]
                pull = (pred_map[drop] - mu) / sig

                # also compute chi2_full for reference
                chi2_full = chi2_from_moduli(pred, PDG)

                print(f"{drop:<7} | {deg:>2}° | {fit['chi2_train']:>9.3f} | {pred_map[drop]:>12.6g} | {pull:>+6.2f} | {chi2_full:>8.3f}")

                loo_rows.append({
                    "drop": drop, "deg": deg,
                    "chi2_train": fit["chi2_train"],
                    "pred": pred_map[drop],
                    "pull": pull,
                    "chi2_full": chi2_full
                })

            # delta line (35 - 30) for this left-out
            r30 = next(r for r in loo_rows if r["drop"] == drop and r["deg"] == 30)
            r35 = next(r for r in loo_rows if r["drop"] == drop and r["deg"] == 35)
            d_train = r35["chi2_train"] - r30["chi2_train"]
            d_full  = r35["chi2_full"]  - r30["chi2_full"]
            d_pull2 = (r35["pull"]**2) - (r30["pull"]**2)

            print(f"Δ(35-30) |      | train Δχ²={d_train:+.3f} | full Δχ²={d_full:+.3f} | Δpull²={d_pull2:+.3f}")
            print("-"*90)

    # -------------------------------------------------------------------------
    # FINAL SUMMARY (paper-ready)
    # -------------------------------------------------------------------------
    print("\n" + "="*90)
    print("FINAL SUMMARY (paper-ready)")
    print("="*90)
    print(f"Real-data global best: {best['deg']}° with chi2={best['chi2']:.3f}")
    if p30 and p35:
        print(f"Profile Δχ²(35-30) = {p35['chi2']-p30['chi2']:.3f}")
        print(f"Evidence weights: w30={w_map[30]:.4f}, w35={w_map[35]:.4f}")
    print("Done.")
