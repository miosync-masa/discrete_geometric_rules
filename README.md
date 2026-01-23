# Geometric Selection of the Leptonic CP Phase from Platonic Base Angles

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the analysis code and data for the paper:

> **Geometric Selection of the Leptonic CP Phase from Platonic Base Angles and a Discrete δ<sub>CP</sub> Lattice**  
> Masamichi Iizumi (Miosync, Inc.)

## Overview

We present a discrete-geometric selection principle for the PMNS mixing parameters, emphasizing the leptonic Dirac CP phase δ<sub>CP</sub>. The approach uses only **Platonic base angles** and an ***a priori* discrete lattice** for δ<sub>CP</sub>, rather than continuously fitting δ<sub>CP</sub>.

### Key Results

| Parameter | Platonic (*a priori*) | Best fit | ε |
|-----------|----------------------|----------|---|
| θ₁₂ | arcsin(1/√3) = 35.26° | 33.52° | −0.049 |
| θ₂₃ | arcsin(1/√2) = 45.00° | 42.19° | −0.062 |
| θ₁₃ | arcsin(1/√45) = 8.57° | 8.59° | +0.002 |
| δ<sub>CP</sub> | π + arcsin(√10/4) = 232.24° | 232° | (discrete) |

**Main finding:** The global best fit selects δ<sub>CP</sub> ≈ 232°, lying within **0.24°** of the Platonic candidate:

```
δ₂ = π + arcsin(√10/4) = 232.2388°

sin(δ₂) = −√10/4,  cos(δ₂) = −√6/4
```

This yields a sharp, **falsifiable target** for ongoing and upcoming long-baseline neutrino programs (T2K, Hyper-Kamiokande).

## Repository Structure

```
Pmns_geometric/
├── README.md
├── pmns_delta_lattice_pubgrade.py    # Main analysis script 
├── pmns_result_analytics.py          # Result CSV analysis script
├── pmns_nufit_curves/                # NuFIT v5.2 1D Δχ² profiles
│   ├── README.md
│   ├── s12sq.csv
│   ├── s13sq.csv
│   ├── s23sq.csv
│   └── delta_cp.csv
├── pmns_outputs_delta_lattice/       # Analysis outputs
│   ├── delta_grid.csv
│   ├── pmns_profile_by_delta.csv
│   ├── pmns_profile_by_model_and_delta.csv
│   ├── pmns_models_summary.csv
│   ├── pmns_delta_evidence_weights.csv
│   ├── pmns_delta_holdout.csv
│   ├── pmns_ppc_toy_winners.csv
│   └── pmns_ppc_summary.json
└── paper/
    └── pmns_geometric_selection.pdf
    
```

## Installation

```bash
git clone https://github.com/miosync-masa/Pmns_geometric.git
cd Pmns_geometric
pip install numpy pandas scipy
```

## Usage

### Run the full analysis

```bash
python pmns_delta_lattice_pubgrade.py \
  --curves_dir ./pmns_nufit_curves \
  --out_dir ./pmns_outputs_delta_lattice \
  --n_starts_real 40 \
  --n_starts_toy 6 \
  --n_toys 200
```

### Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `--curves_dir` | `./pmns_nufit_curves` | Directory containing NuFIT 1D CSV curves |
| `--out_dir` | `./pmns_outputs_delta_lattice` | Output directory |
| `--n_starts_real` | 40 | Multi-start count per lattice point (real data) |
| `--n_starts_toy` | 6 | Multi-start count per lattice point (PPC toys) |
| `--n_toys` | 200 | Number of posterior predictive check toys |
| `--seed` | 12345 | Random seed |

## Methodology

### Platonic Base Angles

The mixing angles are parameterized as small deformations around fixed Platonic values:

```
θᵢⱼ = θᵢⱼ⁰ × (1 + εᵢⱼ)
```

where the Platonic base angles are:
- θ₂₃⁰ = arcsin(1/√2) = 45°
- θ₁₂⁰ = arcsin(1/√3) = 35.2644°
- θ₁₃⁰ = arcsin(1/√45) = 8.5731°

### Discrete δ<sub>CP</sub> Lattice

δ<sub>CP</sub> is selected from an *a priori* discrete lattice:
- **Equal-division lattice:** δ = k×(360°/n) with n ∈ {12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72}
- **Explicit Platonic candidates:**
  - δ₁ = π + arcsin(1/√12) = 196.7787°
  - δ₂ = π + arcsin(√10/4) = 232.2388°

### Likelihood Approximation

We use NuFIT v5.2 (November 2022) one-dimensional Δχ² profiles for normal ordering, approximating the global likelihood as a sum of 1D profiles:

```
χ²(θ₁₂, θ₁₃, θ₂₃, δ) = Δχ²(sin²θ₁₂) + Δχ²(sin²θ₁₃) + Δχ²(sin²θ₂₃) + Δχ²(δ)
```

**Caveat:** This factorized approximation neglects correlations among oscillation parameters.

### Robustness Tests

- **δ-holdout:** Fit angles only (exclude Δχ²<sub>δ</sub>), then score each discrete δ
- **Posterior predictive checks (PPC):** 100/100 toy experiments select the geometric model

## Output Files

| File | Description |
|------|-------------|
| `delta_grid.csv` | The *a priori* discrete δ lattice |
| `pmns_profile_by_delta.csv` | Best model at each δ |
| `pmns_profile_by_model_and_delta.csv` | All models × δ combinations |
| `pmns_models_summary.csv` | χ², AIC, BIC for each model class |
| `pmns_delta_evidence_weights.csv` | Evidence weights w(δ) per model |
| `pmns_delta_holdout.csv` | δ-holdout test results |
| `pmns_ppc_toy_winners.csv` | PPC toy experiment winners |
| `pmns_ppc_summary.json` | PPC summary statistics |

## Related Work

This study is part of a broader program applying geometric selection rules to Standard Model flavor parameters:

1. **Lepton masses:** [A parameter-free geometric Hamiltonian reproducing charged-lepton mass ratios](https://doi.org/10.5281/zenodo.18337936)  
   Zenodo (2025), doi:10.5281/zenodo.18337936

2. **CKM matrix:** [A Geometric Selection Rule for CKM Mixing from a Minimal Complex Hermitian Hamiltonian](https://doi.org/10.5281/zenodo.18347917)  
   Zenodo (2026), doi:10.5281/zenodo.18347917

## Data Source

NuFIT v5.2 (November 2022) oscillation data:  
http://www.nu-fit.org/

> I. Esteban, M.C. Gonzalez-Garcia, M. Maltoni, T. Schwetz, A. Zhou,  
> "The fate of hints: updated global analysis of three-flavor neutrino oscillations,"  
> JHEP 09, 178 (2020)

## Citation

If you use this code or data, please cite:

```bibtex
@misc{Iizumi2026pmns,
    author = "Iizumi, Masamichi",
    title = "{Geometric Selection of the Leptonic CP Phase from Platonic Base Angles and a Discrete $\delta_{CP}$ Lattice}",
    howpublished = "GitHub",
    year = "2026",
    url = "https://github.com/miosync-masa/Pmns_geometric"
}
```

## License

MIT License

## Author

**Masamichi Iizumi****Tamaki Iizumi** 
Miosync, Inc., Tokyo, Japan
