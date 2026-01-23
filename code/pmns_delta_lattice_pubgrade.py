#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMNS GEOMETRIC SELECTION — Publication-grade (delta lattice + NuFIT 1D Δχ² curves)
=================================================================================

Goal
----
Reproduce a CKM-style "profile scan + evidence weights" analysis for PMNS,
but using ONLY Platonic base angles and a *discrete* δ_CP lattice.

Key reviewer-resistance choices
-------------------------------
[A] Use NuFIT 1D Δχ² curves (CSV) instead of Gaussian approximations.
[B] δ_CP is discrete (a priori) on a geometric lattice:
    - full equal-division lattice from divisors of 360 (n ∈ {12,15,18,20,24,30,36,40,45,60,72})
    - plus explicit "Platonic δ candidates":
        δ₁ = 180° + asin(1/√12)
        δ₂ = 180° + asin(√10/4)
[C] Continuous parameters are only small symmetry-breaking corrections:
    θ_ij = θ_ij^0 * (1 + ε_ij), with Platonic θ^0 fixed:
      θ23^0 = asin(1/√2), θ12^0 = asin(1/√3), θ13^0 = asin(1/√45)
[D] Profile scan over δ candidates (discrete) with evidence weights:
    w(δ) = exp(-(χ²_profile(δ) - χ²_min)/2)
[E] Model comparison includes χ², AIC, BIC, and PPC (posterior predictive checks).

IMPORTANT CAVEAT (state in paper)
---------------------------------
Summing 1D Δχ² curves approximates a factorized likelihood; correlations are neglected.
This is still a strict upgrade over Gaussian fallback and is useful for robustness / selection-rule tests.

Inputs
------
NuFIT 1D Δχ² curves as CSV in a directory, default:
  ./pmns_nufit_curves/
with files:
  s12sq.csv   columns: x,dchi2   (x = sin^2 θ12)
  s13sq.csv   columns: x,dchi2   (x = sin^2 θ13)
  s23sq.csv   columns: x,dchi2   (x = sin^2 θ23)
  delta_cp.csv columns: x,dchi2  (x = δ in degrees)

Outputs
-------
Creates ./pmns_outputs_delta_lattice/ with:
  - delta_grid.csv
  - pmns_profile_by_delta.csv           (best model at each δ)
  - pmns_profile_by_model_and_delta.csv (all models × δ)
  - pmns_models_summary.csv             (χ², AIC, BIC for each model)
  - pmns_delta_evidence_weights.csv     (per-model and global δ weights)
  - pmns_ppc_toy_winners.csv
  - pmns_ppc_summary.json

Run
---
python pmns_delta_lattice_pubgrade.py \
  --curves_dir ./pmns_nufit_curves \
  --n_starts_real 40 --n_starts_toy 6 --n_toys 200

"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# -----------------------------------------------------------------------------
# Constants: Platonic base angles (radians)
# -----------------------------------------------------------------------------
PI = math.pi
TH23_0 = math.asin(1.0 / math.sqrt(2.0))    # 45°
TH12_0 = math.asin(1.0 / math.sqrt(3.0))    # 35.264°
TH13_0 = math.asin(1.0 / math.sqrt(45.0))   # 8.573°

# Conservative bounds for symmetry-breaking epsilons
EPS12_BOUNDS = (-0.20, 0.20)
EPS23_BOUNDS = (-0.30, 0.30)
EPS13_BOUNDS = (-0.20, 0.20)

# Edge penalty settings for curve interpolation outside domain
EDGE_SCALE_SIN2 = 0.02    # sin^2 scale
EDGE_SCALE_DELTA = 10.0   # degrees
EDGE_PENALTY_MULT = 10.0  # how fast penalty grows outside provided curve support


# -----------------------------------------------------------------------------
# Utility: robust interpolation of 1D Δχ² curves
# -----------------------------------------------------------------------------
def load_curve_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if not {"x", "dchi2"}.issubset(df.columns):
        raise ValueError(f"Curve CSV must contain columns x,dchi2: {path}")
    x = df["x"].to_numpy(dtype=float)
    y = df["dchi2"].to_numpy(dtype=float)
    # ensure strictly increasing for interp
    order = np.argsort(x)
    x, y = x[order], y[order]
    return x, y


def interp_dchi2(
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    x: float,
    *,
    periodic: bool = False,
    period: float = 360.0,
    edge_scale: float = 1.0,
) -> float:
    """
    Linear interpolation on (xgrid,ygrid), with a strong penalty outside range.

    For periodic variables (δ), wrap into [0,period).
    """
    if periodic:
        x = x % period

    xmin, xmax = float(xgrid[0]), float(xgrid[-1])
    if x < xmin:
        # penalty grows quadratically with distance outside
        dx = (xmin - x) / edge_scale
        return float(ygrid[0] + EDGE_PENALTY_MULT * dx * dx)
    if x > xmax:
        dx = (x - xmax) / edge_scale
        return float(ygrid[-1] + EDGE_PENALTY_MULT * dx * dx)

    return float(np.interp(x, xgrid, ygrid))


# -----------------------------------------------------------------------------
# PMNS predictions from Platonic base + eps corrections
# -----------------------------------------------------------------------------
def predict_from_eps(eps12: float, eps23: float, eps13: float, delta_deg: float) -> Dict[str, float]:
    th12 = TH12_0 * (1.0 + eps12)
    th23 = TH23_0 * (1.0 + eps23)
    th13 = TH13_0 * (1.0 + eps13)

    s12sq = math.sin(th12) ** 2
    s23sq = math.sin(th23) ** 2
    s13sq = math.sin(th13) ** 2
    ddeg = delta_deg % 360.0

    return {
        "theta12_deg": math.degrees(th12),
        "theta23_deg": math.degrees(th23),
        "theta13_deg": math.degrees(th13),
        "s12sq": s12sq,
        "s23sq": s23sq,
        "s13sq": s13sq,
        "delta_deg": ddeg,
    }


# -----------------------------------------------------------------------------
# Models (Platonic-only) = which eps are free + octant constraints
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelClass:
    name: str
    free_eps12: bool = True
    free_eps23: bool = True
    free_eps13: bool = False
    fix_eps23: Optional[float] = None
    fix_eps13: Optional[float] = 0.0
    eps23_sign: int = 0  # 0 none, +1 require eps23>=0, -1 require eps23<=0

    @property
    def k_params(self) -> int:
        k = 0
        if self.free_eps12:
            k += 1
        if self.free_eps23:
            k += 1
        if self.free_eps13:
            k += 1
        return k

    def bounds(self) -> List[Tuple[float, float]]:
        b = []
        if self.free_eps12:
            b.append(EPS12_BOUNDS)
        if self.free_eps23:
            lo, hi = EPS23_BOUNDS
            if self.eps23_sign > 0:
                lo = max(lo, 0.0)
            if self.eps23_sign < 0:
                hi = min(hi, 0.0)
            b.append((lo, hi))
        if self.free_eps13:
            b.append(EPS13_BOUNDS)
        return b

    def unpack(self, x: np.ndarray) -> Tuple[float, float, float]:
        i = 0
        eps12 = float(x[i]) if self.free_eps12 else 0.0
        i += 1 if self.free_eps12 else 0

        if self.fix_eps23 is not None:
            eps23 = float(self.fix_eps23)
        else:
            eps23 = float(x[i]) if self.free_eps23 else 0.0
            i += 1 if self.free_eps23 else 0

        if self.fix_eps13 is not None and (not self.free_eps13):
            eps13 = float(self.fix_eps13)
        else:
            eps13 = float(x[i]) if self.free_eps13 else float(self.fix_eps13 or 0.0)

        return eps12, eps23, eps13


MODELS: List[ModelClass] = [
    ModelClass(name="M_maximal23", free_eps12=True, free_eps23=False, free_eps13=False, fix_eps23=0.0, fix_eps13=0.0),
    ModelClass(name="M_free23", free_eps12=True, free_eps23=True, free_eps13=False, fix_eps13=0.0),
    ModelClass(name="M_upper_octant", free_eps12=True, free_eps23=True, free_eps13=False, fix_eps13=0.0, eps23_sign=+1),
    ModelClass(name="M_lower_octant", free_eps12=True, free_eps23=True, free_eps13=False, fix_eps13=0.0, eps23_sign=-1),
    ModelClass(name="M_free23+free13", free_eps12=True, free_eps23=True, free_eps13=True, fix_eps13=None),
]


# -----------------------------------------------------------------------------
# Delta lattice (360 divisors + explicit Platonic candidates)
# -----------------------------------------------------------------------------
def generate_delta_lattice_deg(
    ns: List[int],
    include_all_k: bool = True,
    include_platonic: bool = True,
) -> pd.DataFrame:
    rows = []
    seen = set()

    for n in ns:
        step = 360.0 / n
        ks = range(n) if include_all_k else range(0, n, max(1, n // 12))
        for k in ks:
            d = (k * step) % 360.0
            # these are exact integers for n|360; represent as int
            dint = int(round(d))
            if dint not in seen:
                seen.add(dint)
                rows.append({"delta_deg": float(dint), "source": "360_divisor", "n": int(n), "k": int(k)})

    if include_platonic:
        d1 = 180.0 + math.degrees(math.asin(1.0 / math.sqrt(12.0)))  # 196.78...
        d2 = 180.0 + math.degrees(math.asin(math.sqrt(10.0) / 4.0))  # 232.24...
        for name, d in [("platonic_pi+asin(1/sqrt12)", d1), ("platonic_pi+asin(sqrt10/4)", d2)]:
            d = d % 360.0
            key = ("P", round(d, 6))
            if key not in seen:
                seen.add(key)
                rows.append({"delta_deg": float(round(d, 6)), "source": name, "n": None, "k": None})

    df = pd.DataFrame(rows).sort_values("delta_deg").reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Objective: total χ² from NuFIT Δχ² curves (factorized approximation)
# -----------------------------------------------------------------------------
@dataclass
class Curves:
    s12sq: Tuple[np.ndarray, np.ndarray]
    s13sq: Tuple[np.ndarray, np.ndarray]
    s23sq: Tuple[np.ndarray, np.ndarray]
    delta: Tuple[np.ndarray, np.ndarray]


def chi2_from_curves(pred: Dict[str, float], curves: Curves, include_delta: bool = True) -> float:
    x12, y12 = curves.s12sq
    x13, y13 = curves.s13sq
    x23, y23 = curves.s23sq
    xd, yd = curves.delta

    chi2 = 0.0
    chi2 += interp_dchi2(x12, y12, pred["s12sq"], edge_scale=EDGE_SCALE_SIN2)
    chi2 += interp_dchi2(x13, y13, pred["s13sq"], edge_scale=EDGE_SCALE_SIN2)
    chi2 += interp_dchi2(x23, y23, pred["s23sq"], edge_scale=EDGE_SCALE_SIN2)
    if include_delta:
        chi2 += interp_dchi2(xd, yd, pred["delta_deg"], periodic=True, period=360.0, edge_scale=EDGE_SCALE_DELTA)
    return float(chi2)


# -----------------------------------------------------------------------------
# Multi-start bounded optimization per (model, delta)
# -----------------------------------------------------------------------------
def random_in_bounds(rng: np.random.Generator, bounds: List[Tuple[float, float]]) -> np.ndarray:
    x0 = []
    for lo, hi in bounds:
        x0.append(rng.uniform(lo, hi))
    return np.array(x0, dtype=float)


def optimize_model_for_delta(
    model: ModelClass,
    delta_deg: float,
    curves: Curves,
    *,
    n_starts: int,
    seed: int,
    include_delta: bool = True,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Returns: (chi2_min, best_params_dict, best_pred_dict)
    """
    bounds = model.bounds()
    rng = np.random.default_rng(seed)

    def obj(x: np.ndarray) -> float:
        eps12, eps23, eps13 = model.unpack(x)
        pred = predict_from_eps(eps12, eps23, eps13, delta_deg)
        return chi2_from_curves(pred, curves, include_delta=include_delta)

    # deterministic "central" start + random starts
    starts = []
    if bounds:
        mid = np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=float)
        starts.append(mid)
        for _ in range(max(0, n_starts - 1)):
            starts.append(random_in_bounds(rng, bounds))
    else:
        starts = [np.array([], dtype=float)]

    best = (float("inf"), None)

    for x0 in starts:
        try:
            res = minimize(
                obj,
                x0,
                method="Powell",
                bounds=bounds if bounds else None,
                options={"maxiter": 4000, "xtol": 1e-10, "ftol": 1e-10},
            )
            if float(res.fun) < best[0]:
                best = (float(res.fun), np.array(res.x, dtype=float))
        except Exception:
            continue

    chi2_min, xbest = best
    if xbest is None:
        # fallback (should be rare)
        xbest = starts[0]
        chi2_min = obj(xbest)

    eps12, eps23, eps13 = model.unpack(xbest)
    params = {"eps12": eps12, "eps23": eps23, "eps13": eps13, "delta_deg": float(delta_deg)}
    pred = predict_from_eps(eps12, eps23, eps13, delta_deg)
    return chi2_min, params, pred


# -----------------------------------------------------------------------------
# Evidence weights and information criteria
# -----------------------------------------------------------------------------
def evidence_weights(chi2_list: np.ndarray) -> np.ndarray:
    chi2_min = float(np.min(chi2_list))
    w = np.exp(-0.5 * (chi2_list - chi2_min))
    wsum = float(np.sum(w))
    return w / wsum if wsum > 0 else w


def aic_bic(chi2: float, k: int, n_obs: int) -> Tuple[float, float, float]:
    """
    Returns (AIC, BIC, AICc). For small n_obs, AICc may be undefined.
    """
    aic = chi2 + 2.0 * k
    bic = chi2 + k * math.log(max(n_obs, 1))
    # AICc: only defined if n_obs > k+1
    if n_obs > (k + 1):
        aicc = aic + (2.0 * k * (k + 1)) / (n_obs - k - 1)
    else:
        aicc = float("nan")
    return float(aic), float(bic), float(aicc)


# -----------------------------------------------------------------------------
# Posterior Predictive Check (PPC) using 1D curves as likelihoods
# -----------------------------------------------------------------------------
def sample_from_curve(
    rng: np.random.Generator,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    *,
    periodic: bool = False,
    period: float = 360.0,
) -> float:
    """
    Sample x ~ exp(-Δχ²(x)/2) using a discrete approximation on the provided grid.
    """
    w = np.exp(-0.5 * (ygrid - np.min(ygrid)))
    wsum = float(np.sum(w))
    if wsum <= 0:
        idx = rng.integers(0, len(xgrid))
        return float(xgrid[idx])
    p = w / wsum
    idx = int(rng.choice(len(xgrid), p=p))
    x = float(xgrid[idx])
    if periodic:
        x = x % period
    return x


def run_ppc(
    curves: Curves,
    delta_candidates: np.ndarray,
    *,
    n_toys: int,
    n_starts_toy: int,
    seed: int,
    models: List[ModelClass],
) -> Tuple[pd.DataFrame, Dict]:
    """
    PPC: draw synthetic "observations" from NuFIT 1D likelihoods (factorized),
    then re-run model selection over the *same* delta lattice.
    """
    rng = np.random.default_rng(seed)

    winners = []
    for t in range(n_toys):
        # toy "central values" sampled from curves (factorized)
        toy = {}
        toy["s12sq"] = sample_from_curve(rng, *curves.s12sq)
        toy["s13sq"] = sample_from_curve(rng, *curves.s13sq)
        toy["s23sq"] = sample_from_curve(rng, *curves.s23sq)
        toy["delta_deg"] = sample_from_curve(rng, *curves.delta, periodic=True)

        # Build toy curves as *shifted* Δχ² by re-centering on toy draw:
        # We keep original curve shapes but subtract value at toy point so toy has Δχ²=0.
        # This preserves non-Gaussian shapes while defining a toy dataset.
        def recenter(curve: Tuple[np.ndarray, np.ndarray], x0: float, periodic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
            xg, yg = curve
            y0 = interp_dchi2(xg, yg, x0, periodic=periodic, edge_scale=EDGE_SCALE_DELTA if periodic else EDGE_SCALE_SIN2)
            return xg, yg - y0

        toy_curves = Curves(
            s12sq=recenter(curves.s12sq, toy["s12sq"]),
            s13sq=recenter(curves.s13sq, toy["s13sq"]),
            s23sq=recenter(curves.s23sq, toy["s23sq"]),
            delta=recenter(curves.delta, toy["delta_deg"], periodic=True),
        )

        best = {"chi2": float("inf")}
        # model selection: minimize chi2 over (model, delta, eps...)
        for model in models:
            for d in delta_candidates:
                chi2, params, pred = optimize_model_for_delta(
                    model, float(d), toy_curves, n_starts=n_starts_toy, seed=seed + 1000 * t + int(10 * d), include_delta=True
                )
                if chi2 < best["chi2"]:
                    best = {"toy": t + 1, "model": model.name, "delta_deg": float(d), "chi2": float(chi2)}

        winners.append(best)

    df = pd.DataFrame(winners)

    # summary counts
    counts = df["model"].value_counts().to_dict()
    summary = {
        "n_toys": n_toys,
        "n_starts_toy": n_starts_toy,
        "winner_counts": counts,
    }
    return df, summary


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves_dir", type=str, default="./pmns_nufit_curves", help="Directory containing NuFIT 1D CSV curves")
    ap.add_argument("--out_dir", type=str, default="./pmns_outputs_delta_lattice", help="Output directory")
    ap.add_argument("--n_starts_real", type=int, default=40, help="Multi-start count per lattice point (real data)")
    ap.add_argument("--n_starts_toy", type=int, default=6, help="Multi-start count per lattice point (toys)")
    ap.add_argument("--n_toys", type=int, default=200, help="Number of PPC toys")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load curves
    cdir = args.curves_dir
    req = ["s12sq.csv", "s13sq.csv", "s23sq.csv", "delta_cp.csv"]
    for f in req:
        if not os.path.exists(os.path.join(cdir, f)):
            raise FileNotFoundError(f"Missing curve file: {os.path.join(cdir, f)}")

    curves = Curves(
        s12sq=load_curve_csv(os.path.join(cdir, "s12sq.csv")),
        s13sq=load_curve_csv(os.path.join(cdir, "s13sq.csv")),
        s23sq=load_curve_csv(os.path.join(cdir, "s23sq.csv")),
        delta=load_curve_csv(os.path.join(cdir, "delta_cp.csv")),
    )

    # Delta lattice (a priori)
    ns = [12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72]
    delta_df = generate_delta_lattice_deg(ns, include_all_k=True, include_platonic=True)
    delta_df.to_csv(os.path.join(args.out_dir, "delta_grid.csv"), index=False)

    delta_candidates = delta_df["delta_deg"].to_numpy(dtype=float)

    # -------------------------
    # PART 1: Full profile scan
    # -------------------------
    rows = []
    best_global = {"chi2": float("inf")}
    for model in MODELS:
        for d in delta_candidates:
            chi2, params, pred = optimize_model_for_delta(
                model, float(d), curves, n_starts=args.n_starts_real, seed=args.seed + int(10 * d), include_delta=True
            )
            row = {
                "model": model.name,
                "chi2": float(chi2),
                "k": int(model.k_params),
                **params,
                **pred,
            }
            rows.append(row)
            if chi2 < best_global["chi2"]:
                best_global = row

    prof_all = pd.DataFrame(rows)
    prof_all.to_csv(os.path.join(args.out_dir, "pmns_profile_by_model_and_delta.csv"), index=False)

    # Best model at each delta
    prof_by_delta = prof_all.sort_values("chi2").groupby("delta_deg", as_index=False).first()
    prof_by_delta.to_csv(os.path.join(args.out_dir, "pmns_profile_by_delta.csv"), index=False)

    # Per-model summary (best over delta)
    best_by_model = prof_all.sort_values("chi2").groupby("model", as_index=False).first()

    # AIC/BIC (treating delta as discrete *index*, not a continuous parameter)
    n_obs = 4  # s12sq, s13sq, s23sq, delta
    aic_list, bic_list, aicc_list = [], [], []
    for _, r in best_by_model.iterrows():
        aic, bic, aicc = aic_bic(float(r["chi2"]), int(r["k"]), n_obs)
        aic_list.append(aic)
        bic_list.append(bic)
        aicc_list.append(aicc)
    best_by_model["AIC"] = aic_list
    best_by_model["BIC"] = bic_list
    best_by_model["AICc"] = aicc_list
    best_by_model.to_csv(os.path.join(args.out_dir, "pmns_models_summary.csv"), index=False)

    # Evidence weights over delta
    ew_rows = []
    for model in MODELS:
        sub = prof_all[prof_all["model"] == model.name].copy()
        sub = sub.sort_values("delta_deg")
        w = evidence_weights(sub["chi2"].to_numpy(dtype=float))
        for d, chi2v, wv in zip(sub["delta_deg"], sub["chi2"], w):
            ew_rows.append({"model": model.name, "delta_deg": float(d), "chi2": float(chi2v), "w_norm": float(wv)})

    ew = pd.DataFrame(ew_rows)

    # "global" delta evidence by marginalizing over models (discrete model index)
    # w_global(delta) ∝ Σ_model exp(-(chi2_model(delta) - chi2_global_min)/2)
    chi2_global_min = float(prof_all["chi2"].min())
    global_rows = []
    for d in np.unique(ew["delta_deg"].to_numpy(dtype=float)):
        sub = ew[ew["delta_deg"] == d]
        wsum = float(np.sum(np.exp(-0.5 * (sub["chi2"].to_numpy(dtype=float) - chi2_global_min))))
        global_rows.append({"model": "GLOBAL_MARGINAL", "delta_deg": float(d), "chi2": float(sub["chi2"].min()), "w_unnorm": wsum})
    global_df = pd.DataFrame(global_rows).sort_values("delta_deg")
    wglob = global_df["w_unnorm"].to_numpy(dtype=float)
    wglob = wglob / float(np.sum(wglob)) if float(np.sum(wglob)) > 0 else wglob
    global_df["w_norm"] = wglob
    global_df = global_df.drop(columns=["w_unnorm"])

    ew_out = pd.concat([ew, global_df], ignore_index=True)
    ew_out.to_csv(os.path.join(args.out_dir, "pmns_delta_evidence_weights.csv"), index=False)

    # -------------------------
    # PART 2: delta-only holdout (angles-only fit; then add δ curve on lattice)
    # -------------------------
    # Fit eps params ignoring delta curve, then score each delta candidate with delta curve.
    hold_rows = []
    for model in MODELS:
        # optimize eps once (delta irrelevant when include_delta=False)
        chi2_angles, params, pred = optimize_model_for_delta(
            model, float(delta_candidates[0]), curves, n_starts=args.n_starts_real, seed=args.seed + 999, include_delta=False
        )
        # now add delta curve across lattice
        xd, yd = curves.delta
        for d in delta_candidates:
            delta_pen = interp_dchi2(xd, yd, float(d), periodic=True, period=360.0, edge_scale=EDGE_SCALE_DELTA)
            total = float(chi2_angles + delta_pen)
            hold_rows.append({
                "model": model.name,
                "chi2_angles_only": float(chi2_angles),
                "delta_deg": float(d),
                "delta_dchi2": float(delta_pen),
                "chi2_total": total,
                **params,
                **pred,
            })
    hold = pd.DataFrame(hold_rows)
    hold.to_csv(os.path.join(args.out_dir, "pmns_delta_holdout.csv"), index=False)

    # -------------------------
    # PART 3: PPC
    # -------------------------
    ppc_df, ppc_summary = run_ppc(
        curves, delta_candidates,
        n_toys=args.n_toys,
        n_starts_toy=args.n_starts_toy,
        seed=args.seed + 2024,
        models=MODELS,
    )
    ppc_df.to_csv(os.path.join(args.out_dir, "pmns_ppc_toy_winners.csv"), index=False)
    with open(os.path.join(args.out_dir, "pmns_ppc_summary.json"), "w") as f:
        json.dump(ppc_summary, f, indent=2)

    # Final console summary
    print("=" * 92)
    print("PMNS GEOMETRIC SELECTION — delta lattice + NuFIT 1D curves (DONE)")
    print("=" * 92)
    print("\n[Platonic base angles]")
    print(f"  theta23^0 = asin(1/sqrt2)  = {math.degrees(TH23_0):.4f} deg")
    print(f"  theta12^0 = asin(1/sqrt3)  = {math.degrees(TH12_0):.4f} deg")
    print(f"  theta13^0 = asin(1/sqrt45) = {math.degrees(TH13_0):.4f} deg")
    print("\n[Global best (chi2)]")
    print(f"  model={best_global['model']}  delta={best_global['delta_deg']:.6g} deg  chi2={best_global['chi2']:.6g}")
    print(f"  eps12={best_global['eps12']:.6g}  eps23={best_global['eps23']:.6g}  eps13={best_global['eps13']:.6g}")
    print("\n[Per-model best: chi2 / AIC / BIC]")
    for _, r in best_by_model.sort_values("chi2").iterrows():
        print(f"  {r['model']:<16} chi2={r['chi2']:.6g}  AIC={r['AIC']:.6g}  BIC={r['BIC']:.6g}  delta={r['delta_deg']:.6g}")
    print(f"\nOutputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
