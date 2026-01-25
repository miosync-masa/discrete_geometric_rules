"""
PDG-uncertainty Monte Carlo for integer boost hypothesis (A+B+C) + figures

A) Expand candidate set (multiples of 3 around 30) and re-run integer selection.
B) Robustness checks: normal vs lognormal sampling, and CL(90%)->1σ conversion on/off.
C) Remove enforced relation f3=3*f2 and select from an independent integer grid.

Outputs:
  - summary_scenarios.csv
  - freq_f2_*.csv, freq_f3_*.csv
  - fig_*.png (histograms + vertical lines)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

OUT_DIR = Path("quark_pdg_mc_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class PDGMass:
    mean_GeV: float
    err_GeV: float
    cl90: bool

PDG = {
    "mu": PDGMass(2.16e-3, 0.07e-3, True),
    "md": PDGMass(4.70e-3, 0.07e-3, True),
    "ms": PDGMass(93.5e-3, 0.8e-3,  True),
    "mc": PDGMass(1.2730, 0.0046,  True),
    "mb": PDGMass(4.183,  0.007,   True),
    "mt": PDGMass(172.56, 0.31,    False),
}

Z_90_TWO_SIDED = 1.6448536269514722

def sigma_from_err(err, cl90: bool, use_cl_to_sigma: bool) -> float:
    if cl90 and use_cl_to_sigma:
        return err / Z_90_TWO_SIDED
    return err

def draw_positive_normal(mean, sigma, n, rng):
    x = rng.normal(mean, sigma, size=n)
    bad = x <= 0
    while np.any(bad):
        x[bad] = rng.normal(mean, sigma, size=int(np.sum(bad)))
        bad = x <= 0
    return x

def draw_lognormal(mean, sigma, n, rng):
    rel = sigma / mean
    s2 = np.log(1.0 + rel * rel)
    mu = np.log(mean) - 0.5 * s2
    s = np.sqrt(s2)
    return np.exp(rng.normal(mu, s, size=n))

def draw_masses(n, rng, *, use_cl_to_sigma: bool, use_lognormal: bool):
    m = {}
    for key, p in PDG.items():
        sig = sigma_from_err(p.err_GeV, p.cl90, use_cl_to_sigma)
        if use_lognormal:
            m[key] = draw_lognormal(p.mean_GeV, sig, n, rng)
        else:
            m[key] = draw_positive_normal(p.mean_GeV, sig, n, rng)
    return m

def compute_boosts(m):
    ms_md = m["ms"] / m["md"]
    mb_md = m["mb"] / m["md"]
    mc_mu = m["mc"] / m["mu"]
    mt_mu = m["mt"] / m["mu"]
    f2_hat = mc_mu / ms_md
    f3_hat = mt_mu / mb_md
    nc_hat = f3_hat / f2_hat
    return f2_hat, f3_hat, nc_hat

def select_constrained(f2_hat, f3_hat, f2_candidates):
    log_f2h = np.log(f2_hat)[:, None]
    log_f3h = np.log(f3_hat)[:, None]
    f2c = f2_candidates[None, :]
    chi2 = (log_f2h - np.log(f2c))**2 + (log_f3h - np.log(3.0*f2c))**2
    idx = np.argmin(chi2, axis=1)
    return f2_candidates[idx]

def select_free_grid(f2_hat, f3_hat, f2_candidates, f3_candidates):
    log_f2h = np.log(f2_hat)
    log_f3h = np.log(f3_hat)
    log_f3c = np.log(f3_candidates)

    best = np.full(f2_hat.shape[0], np.inf)
    best_f2 = np.empty_like(f2_hat)
    best_f3 = np.empty_like(f3_hat)

    for f2 in f2_candidates:
        d2 = (log_f2h - np.log(f2))**2
        chi = d2[:, None] + (log_f3h[:, None] - log_f3c[None, :])**2
        idx3 = np.argmin(chi, axis=1)
        chi_min = chi[np.arange(chi.shape[0]), idx3]
        better = chi_min < best
        if np.any(better):
            best[better] = chi_min[better]
            best_f2[better] = f2
            best_f3[better] = f3_candidates[idx3[better]]
    return best_f2.astype(int), best_f3.astype(int)

def summarize(x):
    q = np.quantile(x, [0.025, 0.16, 0.5, 0.84, 0.975])
    return dict(mean=float(np.mean(x)), std=float(np.std(x, ddof=1)),
                q025=float(q[0]), q16=float(q[1]), q50=float(q[2]), q84=float(q[3]), q975=float(q[4]))

def save_hist_overlay(data_by_scenario, key, vline, xlabel, title, fname, bins=90, xlim=None):
    plt.figure()
    for name, vals in data_by_scenario.items():
        plt.hist(vals[key], bins=bins, density=True, alpha=0.35, label=name)
    plt.axvline(vline)
    plt.xlabel(xlabel); plt.ylabel("density"); plt.title(title)
    if xlim is not None: plt.xlim(*xlim)
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=200)
    plt.close()

def main():
    N_TOYS = 200_000
    SEED = 42
    f2_candidates = np.arange(18, 49, 3)
    f3_candidates = np.arange(60, 126, 3)

    scenarios = [
        ("normal_CL2sigma",     dict(use_lognormal=False, use_cl_to_sigma=True)),
        ("normal_noCL2sigma",   dict(use_lognormal=False, use_cl_to_sigma=False)),
        ("lognormal_CL2sigma",  dict(use_lognormal=True,  use_cl_to_sigma=True)),
        ("lognormal_noCL2sigma",dict(use_lognormal=True,  use_cl_to_sigma=False)),
    ]

    rows = []
    plot_data = {}

    for i, (name, cfg) in enumerate(scenarios):
        rng = np.random.default_rng(SEED + i)
        m = draw_masses(N_TOYS, rng, **cfg)
        f2_hat, f3_hat, nc_hat = compute_boosts(m)

        f2_sel_con = select_constrained(f2_hat, f3_hat, f2_candidates)
        f2_sel_free, f3_sel_free = select_free_grid(f2_hat, f3_hat, f2_candidates, f3_candidates)

        con_counts = pd.Series(f2_sel_con).value_counts().reindex(f2_candidates, fill_value=0)
        free_f2_counts = pd.Series(f2_sel_free).value_counts().reindex(f2_candidates, fill_value=0)
        free_f3_counts = pd.Series(f3_sel_free).value_counts().reindex(f3_candidates, fill_value=0)

        s2, s3, sn = summarize(f2_hat), summarize(f3_hat), summarize(nc_hat)
        rows.append({
            "scenario": name,
            "sampling": "lognormal" if cfg["use_lognormal"] else "normal",
            "CL_to_sigma": cfg["use_cl_to_sigma"],
            "N": N_TOYS,
            "f2_hat_mean": s2["mean"], "f2_hat_std": s2["std"], "f2_hat_95lo": s2["q025"], "f2_hat_95hi": s2["q975"],
            "f3_hat_mean": s3["mean"], "f3_hat_std": s3["std"], "f3_hat_95lo": s3["q025"], "f3_hat_95hi": s3["q975"],
            "Nc_hat_mean": sn["mean"], "Nc_hat_std": sn["std"], "Nc_hat_95lo": sn["q025"], "Nc_hat_95hi": sn["q975"],
            "P_constrained_select_f2_30": float(con_counts.loc[30] / N_TOYS),
            "P_freegrid_select_(30,90)": float(np.mean((f2_sel_free == 30) & (f3_sel_free == 90))),
        })

        pd.DataFrame({
            "f2_candidate": f2_candidates,
            "freq_constrained": con_counts.values / N_TOYS,
            "freq_freegrid_f2": free_f2_counts.values / N_TOYS,
        }).to_csv(OUT_DIR / f"freq_f2_{name}.csv", index=False)

        pd.DataFrame({
            "f3_candidate": f3_candidates,
            "freq_freegrid_f3": free_f3_counts.values / N_TOYS,
        }).to_csv(OUT_DIR / f"freq_f3_{name}.csv", index=False)

        plot_data[name] = {
            "f2_hat": f2_hat, "f3_hat": f3_hat, "nc_hat": nc_hat,
        }

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary_scenarios.csv", index=False)

    save_hist_overlay(plot_data, "f2_hat", 30.0, r"$\hat f_2=(m_c/m_u)/(m_s/m_d)$",
                      "Distribution of $\\hat f_2$ (vertical line: 30)", "fig_f2_hat_overlay.png",
                      bins=90, xlim=(24,36))
    save_hist_overlay(plot_data, "f3_hat", 90.0, r"$\hat f_3=(m_t/m_u)/(m_b/m_d)$",
                      "Distribution of $\\hat f_3$ (vertical line: 90)", "fig_f3_hat_overlay.png",
                      bins=90, xlim=(78,102))
    save_hist_overlay(plot_data, "nc_hat", 3.0, r"$\hat N_c=\hat f_3/\hat f_2$",
                      "Distribution of $\\hat N_c$ (vertical line: 3)", "fig_nc_hat_overlay.png",
                      bins=90, xlim=(2.92,3.14))

if __name__ == "__main__":
    main()
