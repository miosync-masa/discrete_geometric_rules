import numpy as np
from dataclasses import dataclass
from scipy.stats import norm

# =============================================================================
# PDG quark masses (from PDG 2025 quark summary table)
# Units: GeV
# Notes:
#   u,d,s are MSbar at mu=2 GeV; c,b are MSbar at mu=m; t is "direct"
#   u,d,s,c,b uncertainties are quoted with CL=90% in the PDG table.
# =============================================================================

@dataclass(frozen=True)
class PDGMass:
    name: str
    mean_GeV: float
    err_GeV: float
    cl: float | None  # confidence level for "err" if applicable (e.g., 0.90). None means treat as 1σ.

PDG = {
    # light quarks: MeV -> GeV
    "mu": PDGMass("mu", 2.16e-3, 0.07e-3, 0.90),
    "md": PDGMass("md", 4.70e-3, 0.07e-3, 0.90),
    "ms": PDGMass("ms", 93.5e-3, 0.8e-3,  0.90),

    # heavy quarks: GeV
    "mc": PDGMass("mc", 1.2730, 0.0046, 0.90),
    "mb": PDGMass("mb", 4.183,  0.007,  0.90),

    # top: PDG table gives "direct measurements" 172.56 ± 0.31 GeV (no CL label on the line)
    # We'll treat this as 1σ by default.
    "mt": PDGMass("mt", 172.56, 0.31, None),
}

# =============================================================================
# Sampling helpers
# =============================================================================

def cl_to_sigma(err: float, cl: float) -> float:
    """
    Convert a symmetric ±err quoted at confidence level 'cl' into a 1σ equivalent,
    assuming a Gaussian distribution.
    For two-sided CL: z = Phi^{-1}((1+cl)/2), so sigma = err / z.
    """
    z = norm.ppf((1.0 + cl) / 2.0)
    return err / z

def sample_positive_normal(mean: float, sigma: float, rng: np.random.Generator) -> float:
    """Sample from N(mean, sigma) but reject non-positive draws."""
    while True:
        x = rng.normal(mean, sigma)
        if x > 0:
            return x

def draw_pdg_masses(
    rng: np.random.Generator,
    use_cl_to_sigma: bool = True,
    use_lognormal: bool = False
) -> dict:
    """
    Draw one toy set of quark masses.
    - use_cl_to_sigma: if True, convert CL=90% errors to 1σ.
    - use_lognormal: if True, sample in log-space (better positivity; approx for small relative errors).
    """
    out = {}
    for k, p in PDG.items():
        if p.cl is not None and use_cl_to_sigma:
            sigma = cl_to_sigma(p.err_GeV, p.cl)
        else:
            sigma = p.err_GeV

        if not use_lognormal:
            out[k] = sample_positive_normal(p.mean_GeV, sigma, rng)
        else:
            # lognormal with matched mean~p.mean and std~sigma (approx; good when sigma/mean small)
            # Convert (mean, sigma) in linear space to (mu, s) in log space:
            # s^2 = ln(1 + (sigma/mean)^2), mu = ln(mean) - 0.5*s^2
            rel = sigma / p.mean_GeV
            s2 = np.log(1.0 + rel * rel)
            mu = np.log(p.mean_GeV) - 0.5 * s2
            s = np.sqrt(s2)
            out[k] = float(np.exp(rng.normal(mu, s)))
    return out

def compute_ratios(m: dict) -> dict:
    """
    Ratios used in your hypothesis:
      down: ms/md, mb/md
      up:   mc/mu, mt/mu
      implied boosts:
        f2_hat = (mc/mu)/(ms/md)
        f3_hat = (mt/mu)/(mb/md)
        Nc_hat = f3_hat/f2_hat  (should be ~3)
    """
    ms_md = m["ms"] / m["md"]
    mb_md = m["mb"] / m["md"]
    mc_mu = m["mc"] / m["mu"]
    mt_mu = m["mt"] / m["mu"]

    f2_hat = (mc_mu) / (ms_md)
    f3_hat = (mt_mu) / (mb_md)
    nc_hat = f3_hat / f2_hat

    return {
        "ms_md": ms_md,
        "mb_md": mb_md,
        "mc_mu": mc_mu,
        "mt_mu": mt_mu,
        "f2_hat": f2_hat,
        "f3_hat": f3_hat,
        "nc_hat": nc_hat,
    }

# =============================================================================
# Candidate selection statistics
# =============================================================================

def select_best_integer_boost(
    f2_hat: float,
    f3_hat: float,
    f2_candidates: list[int],
    enforce_f3_eq_3f2: bool = True
) -> tuple[int, int, float]:
    """
    Pick the (f2, f3) integer hypothesis that best matches the toy's implied boosts.
    Metric: squared log-distance (scale-invariant):
      chi2 = (log f2_hat - log f2)^2 + (log f3_hat - log f3)^2
    If enforce_f3_eq_3f2: use f3 = 3*f2 only.
    """
    best = None
    for f2 in f2_candidates:
        f3 = 3 * f2 if enforce_f3_eq_3f2 else None
        if enforce_f3_eq_3f2:
            chi2 = (np.log(f2_hat) - np.log(f2))**2 + (np.log(f3_hat) - np.log(f3))**2
            cand = (f2, f3, chi2)
        else:
            raise NotImplementedError("Set enforce_f3_eq_3f2=True for your current hypothesis.")
        if best is None or cand[2] < best[2]:
            best = cand
    return best  # (f2, f3, chi2)

def run_mc_boost_test(
    n_toys: int = 200000,
    seed: int = 42,
    f2_candidates: list[int] = None,
    enforce_f3_eq_3f2: bool = True,
    use_cl_to_sigma: bool = True,
    use_lognormal: bool = False,
    tol_abs_f2: float = 0.5,
    tol_abs_f3: float = 1.0,
    tol_abs_nc: float = 0.2,
):
    """
    Monte Carlo:
      1) sample PDG masses with uncertainties
      2) compute implied boosts (f2_hat, f3_hat, nc_hat)
      3) compute selection frequency among integer candidates (default: multiples of 3 around 30)
      4) report robust summary stats + "how often 30/90 wins"
    """
    if f2_candidates is None:
        # a sensible local hypothesis set around 30 (edit as you like)
        f2_candidates = [24, 27, 30, 33, 36]

    rng = np.random.default_rng(seed)

    # storage
    f2_hats = np.empty(n_toys)
    f3_hats = np.empty(n_toys)
    nc_hats = np.empty(n_toys)

    winners = {f2: 0 for f2 in f2_candidates}

    hit_30 = 0
    hit_90 = 0
    hit_both = 0
    hit_nc3 = 0

    for i in range(n_toys):
        masses = draw_pdg_masses(rng, use_cl_to_sigma=use_cl_to_sigma, use_lognormal=use_lognormal)
        r = compute_ratios(masses)

        f2_hat = r["f2_hat"]
        f3_hat = r["f3_hat"]
        nc_hat = r["nc_hat"]

        f2_hats[i] = f2_hat
        f3_hats[i] = f3_hat
        nc_hats[i] = nc_hat

        # integer model selection
        f2_win, f3_win, _ = select_best_integer_boost(
            f2_hat, f3_hat,
            f2_candidates=f2_candidates,
            enforce_f3_eq_3f2=enforce_f3_eq_3f2
        )
        winners[f2_win] += 1

        # "direct" proximity counts (tunable)
        if abs(f2_hat - 30.0) <= tol_abs_f2:
            hit_30 += 1
        if abs(f3_hat - 90.0) <= tol_abs_f3:
            hit_90 += 1
        if abs(nc_hat - 3.0) <= tol_abs_nc:
            hit_nc3 += 1
        if abs(f2_hat - 30.0) <= tol_abs_f2 and abs(f3_hat - 90.0) <= tol_abs_f3:
            hit_both += 1

    def summarize(x: np.ndarray, name: str):
        q = np.quantile(x, [0.005, 0.025, 0.16, 0.50, 0.84, 0.975, 0.995])
        return {
            "name": name,
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)),
            "q005": float(q[0]),
            "q025": float(q[1]),
            "q16":  float(q[2]),
            "q50":  float(q[3]),
            "q84":  float(q[4]),
            "q975": float(q[5]),
            "q995": float(q[6]),
        }

    s_f2 = summarize(f2_hats, "f2_hat = (mc/mu)/(ms/md)")
    s_f3 = summarize(f3_hats, "f3_hat = (mt/mu)/(mb/md)")
    s_nc = summarize(nc_hats, "Nc_hat = f3_hat/f2_hat")

    # Print summary
    print("="*80)
    print("PDG-uncertainty Monte Carlo for integer boost hypothesis (2Nf, Nc×2Nf)")
    print("="*80)
    print(f"toys = {n_toys:,}  seed = {seed}")
    print(f"sampling: {'lognormal' if use_lognormal else 'normal'}; CL->σ = {use_cl_to_sigma}")
    print(f"f2 candidates = {f2_candidates} (f3=3*f2 enforced={enforce_f3_eq_3f2})")
    print()

    for s in [s_f2, s_f3, s_nc]:
        print(f"[{s['name']}]")
        print(f"  mean ± std = {s['mean']:.4f} ± {s['std']:.4f}")
        print(f"  median [16%,84%] = {s['q50']:.4f} [{s['q16']:.4f}, {s['q84']:.4f}]")
        print(f"  95% CI ~ [{s['q025']:.4f}, {s['q975']:.4f}]")
        print()

    print("[Direct proximity counts]")
    print(f"  P(|f2_hat-30| <= {tol_abs_f2}) = {hit_30/n_toys*100:.2f}%")
    print(f"  P(|f3_hat-90| <= {tol_abs_f3}) = {hit_90/n_toys*100:.2f}%")
    print(f"  P(|Nc_hat-3| <= {tol_abs_nc})  = {hit_nc3/n_toys*100:.2f}%")
    print(f"  P(both f2~30 and f3~90)         = {hit_both/n_toys*100:.2f}%")
    print()

    print("[Integer-model selection frequency]")
    for f2 in f2_candidates:
        print(f"  f2={f2:2d}, f3={3*f2:3d}:  {winners[f2]/n_toys*100:6.2f}%")
    print("="*80)

    # return raw arrays for plotting if needed
    return {
        "f2_hat": f2_hats,
        "f3_hat": f3_hats,
        "nc_hat": nc_hats,
        "winners": winners,
        "summary": {"f2": s_f2, "f3": s_f3, "nc": s_nc},
    }

# -----------------------------------------------------------------------------
# Run example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    _ = run_mc_boost_test(
        n_toys=200000,   # bump to 1e6 if you want ultra-stable tails
        seed=42,
        f2_candidates=[24, 27, 30, 33, 36],
        use_cl_to_sigma=True,
        use_lognormal=False,  # try True as a robustness check
        tol_abs_f2=0.5,
        tol_abs_f3=1.0,
        tol_abs_nc=0.2,
    )
