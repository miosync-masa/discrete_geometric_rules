import numpy as np
from scipy.optimize import minimize
from itertools import product

def build_H_complex_t23(t_12, t_23_mag, t_23_phase, t_13_imag, v):
    t_23 = t_23_mag * np.exp(1j * t_23_phase)
    return np.array([
        [0, t_12, 1j * t_13_imag],
        [t_12, 0, t_23],
        [-1j * t_13_imag, np.conj(t_23), -v]
    ], dtype=complex)

def diagonalize_and_sort(H):
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    idx = np.abs(eigenvalues).argsort()
    return eigenvalues[idx], eigenvectors[:, idx]

def extract_CKM_angles_from_moduli(V):
    s13 = np.abs(V[0, 2])
    c13 = np.sqrt(1 - s13**2) if s13 < 1 else 1e-10
    s12 = np.abs(V[0, 1]) / c13 if c13 > 1e-10 else 0
    s12 = np.clip(s12, 0, 1)
    c12 = np.sqrt(1 - s12**2)
    s23 = np.abs(V[1, 2]) / c13 if c13 > 1e-10 else 0
    s23 = np.clip(s23, 0, 1)
    c23 = np.sqrt(1 - s23**2)
    return np.arcsin(s12), np.arcsin(s13), np.arcsin(s23), s12, c12, s13, c13, s23, c23

def compute_jarlskog(V):
    return np.imag(V[0,0] * V[1,1] * np.conj(V[0,1]) * np.conj(V[1,0]))

def extract_delta_from_Vtd(V, s12, c12, s13, c13, s23, c23):
    J = compute_jarlskog(V)
    denom_sin = s12 * s23 * s13 * c12 * c23 * c13**2
    sin_delta = J / denom_sin if abs(denom_sin) > 1e-12 else 0
    sin_delta = np.clip(sin_delta, -1, 1)

    V_td_sq = np.abs(V[2, 0])**2
    term1 = s12**2 * s23**2
    term2 = c12**2 * c23**2 * s13**2
    denom_cos = 2 * s12 * c12 * s23 * c23 * s13
    cos_delta = (term1 + term2 - V_td_sq) / denom_cos if abs(denom_cos) > 1e-12 else 1
    cos_delta = np.clip(cos_delta, -1, 1)

    return np.arctan2(sin_delta, cos_delta), J

# Base values
PI = np.pi
t_r = 1.0
ti = (180 / PI) * t_r
v_base = np.sqrt(15) * ti

# PDG 2024 experimental values with uncertainties
exp_data = {
    'theta12': (13.04, 0.05),
    'theta13': (0.20, 0.01),
    'theta23': (2.38, 0.06),
    'J': (3.0e-5, 0.2e-5),
    'V_td': (0.0086, 0.0002),
}

print("="*80)
print("A PRIORI DISCRETE LATTICE SEARCH")
print("Pre-defined candidate space (declared BEFORE fitting)")
print("="*80)

# ========================================
# 事前定義された離散候補空間
# ========================================

print("\n[PRE-DEFINED CANDIDATE SPACE]")

# 1. κ_ratio: cos(角度) で 15°刻み
ratio_candidates = []
for deg in [15, 30, 45, 60, 75]:
    ratio_candidates.append((np.cos(np.radians(deg)), f"cos({deg}°)"))
print(f"\nκ₂₃(d)/κ₂₃(u) candidates (cos of 15° multiples):")
for val, name in ratio_candidates:
    print(f"  {name} = {val:.6f}")

# 2. κ_v: 分母 ≤ 50 の有理数で 1.1 < κ_v < 1.4
kv_candidates = []
for denom in range(1, 51):
    for numer in range(denom + 1, int(denom * 1.5) + 1):
        val = numer / denom
        if 1.15 < val < 1.35:
            kv_candidates.append((val, f"{numer}/{denom}"))
# 重複除去してソート
kv_candidates = list(set(kv_candidates))
kv_candidates.sort(key=lambda x: x[0])
print(f"\nκ_v candidates (rationals with denom ≤ 50, 1.15 < κ_v < 1.35):")
print(f"  Total: {len(kv_candidates)} candidates")
print(f"  Range: {kv_candidates[0][0]:.4f} to {kv_candidates[-1][0]:.4f}")

# 3. φ_d: -2π/n for n = 20 to 50
phi_candidates = []
for n in range(20, 51):
    phi_candidates.append((-2*PI/n, f"-2π/{n}"))
print(f"\nφ_d candidates (-2π/n for n = 20 to 50):")
print(f"  Total: {len(phi_candidates)} candidates")
print(f"  Range: {np.degrees(phi_candidates[0][0]):.2f}° to {np.degrees(phi_candidates[-1][0]):.2f}°")

# 総候補数
total_candidates = len(ratio_candidates) * len(kv_candidates) * len(phi_candidates)
print(f"\n[TOTAL SEARCH SPACE]: {len(ratio_candidates)} × {len(kv_candidates)} × {len(phi_candidates)} = {total_candidates} combinations")

# ========================================
# 格子探索
# ========================================

print("\n" + "="*80)
print("LATTICE SEARCH (minimizing χ² over 2 continuous params: ε₁₂, κ₂₃)")
print("="*80)

def compute_chi2(eps_12, kappa_23_u, ratio, kv, phi_d):
    kappa_23_d = kappa_23_u * ratio

    H_u = build_H_complex_t23(t_r, kappa_23_u * t_r, 0, ti, v_base)
    _, U_u = diagonalize_and_sort(H_u)

    H_d = build_H_complex_t23(t_r * (1 + eps_12), kappa_23_d * t_r, phi_d, ti, v_base * kv)
    _, U_d = diagonalize_and_sort(H_d)

    V = U_u.conj().T @ U_d
    th12, th13, th23, *_ = extract_CKM_angles_from_moduli(V)
    J = compute_jarlskog(V)
    V_td = np.abs(V[2, 0])

    chi2 = ((np.degrees(th12) - exp_data['theta12'][0])**2 / exp_data['theta12'][1]**2 +
            (np.degrees(th13) - exp_data['theta13'][0])**2 / exp_data['theta13'][1]**2 +
            (np.degrees(th23) - exp_data['theta23'][0])**2 / exp_data['theta23'][1]**2 +
            (J - exp_data['J'][0])**2 / exp_data['J'][1]**2 +
            (V_td - exp_data['V_td'][0])**2 / exp_data['V_td'][1]**2)
    return chi2

results = []
count = 0

for ratio_val, ratio_name in ratio_candidates:
    for kv_val, kv_name in kv_candidates:
        for phi_val, phi_name in phi_candidates:
            def objective(params):
                return compute_chi2(params[0], params[1], ratio_val, kv_val, phi_val)

            # 最適化
            res = minimize(objective, [3.0, 6.5], method='Nelder-Mead',
                          options={'maxiter': 1000})

            results.append({
                'ratio': ratio_val,
                'ratio_name': ratio_name,
                'kv': kv_val,
                'kv_name': kv_name,
                'phi': phi_val,
                'phi_name': phi_name,
                'eps_12': res.x[0],
                'kappa_23_u': res.x[1],
                'chi2': res.fun
            })

            count += 1
            if count % 500 == 0:
                print(f"  Progress: {count}/{total_candidates} ({100*count/total_candidates:.1f}%)")

# χ²でソート
results.sort(key=lambda x: x['chi2'])

print(f"\nSearch completed: {len(results)} combinations evaluated")

# Top 10
print("\n" + "="*80)
print("TOP 10 RESULTS (lowest χ²)")
print("="*80)
print(f"\n{'Rank':<5} {'χ²':<8} {'ratio':<12} {'κ_v':<10} {'φ_d':<12} {'ε₁₂':<8} {'κ₂₃(u)':<8}")
print("-"*75)

for i, r in enumerate(results[:10]):
    print(f"{i+1:<5} {r['chi2']:<8.3f} {r['ratio_name']:<12} {r['kv_name']:<10} {r['phi_name']:<12} {r['eps_12']:<8.3f} {r['kappa_23_u']:<8.3f}")

# Best result の詳細
best = results[0]
print("\n" + "="*80)
print("BEST RESULT DETAILS")
print("="*80)

print(f"\n🎯 Optimal discrete constants:")
print(f"  κ₂₃(d)/κ₂₃(u) = {best['ratio_name']} = {best['ratio']:.6f}")
print(f"  κ_v = {best['kv_name']} = {best['kv']:.6f}")
print(f"  φ_d = {best['phi_name']} = {np.degrees(best['phi']):.4f}°")

print(f"\n🎯 Optimized continuous parameters:")
print(f"  ε₁₂    = {best['eps_12']:.4f}")
print(f"  κ₂₃(u) = {best['kappa_23_u']:.4f}")

# 最終計算
kappa_23_d = best['kappa_23_u'] * best['ratio']
H_u = build_H_complex_t23(t_r, best['kappa_23_u'] * t_r, 0, ti, v_base)
_, U_u = diagonalize_and_sort(H_u)
H_d = build_H_complex_t23(t_r * (1 + best['eps_12']), kappa_23_d * t_r, best['phi'], ti, v_base * best['kv'])
_, U_d = diagonalize_and_sort(H_d)
V_best = U_u.conj().T @ U_d

th12, th13, th23, s12, c12, s13, c13, s23, c23 = extract_CKM_angles_from_moduli(V_best)
delta, J = extract_delta_from_Vtd(V_best, s12, c12, s13, c13, s23, c23)
V_td = np.abs(V_best[2, 0])

print(f"\n🎯 Resulting CKM parameters:")
print(f"  θ₁₂   = {np.degrees(th12):.2f}° (exp: {exp_data['theta12'][0]}±{exp_data['theta12'][1]}°)")
print(f"  θ₁₃   = {np.degrees(th13):.2f}° (exp: {exp_data['theta13'][0]}±{exp_data['theta13'][1]}°)")
print(f"  θ₂₃   = {np.degrees(th23):.2f}° (exp: {exp_data['theta23'][0]}±{exp_data['theta23'][1]}°)")
print(f"  J     = {J:.2e} (exp: {exp_data['J'][0]:.1e})")
print(f"  |V_td|= {V_td:.4f} (exp: {exp_data['V_td'][0]})")
print(f"  δ     = {np.degrees(delta):.2f}° (exp: ~70°)")
print(f"\n  χ²    = {best['chi2']:.3f}")

print(f"\n|V_CKM|:")
print(np.array2string(np.abs(V_best), precision=4))

# Look-elsewhere effect の評価
print("\n" + "="*80)
print("LOOK-ELSEWHERE EFFECT ANALYSIS")
print("="*80)
chi2_threshold = 1.0  # χ² < 1 を「良いフィット」とする
good_fits = [r for r in results if r['chi2'] < chi2_threshold]
print(f"\nNumber of combinations with χ² < {chi2_threshold}: {len(good_fits)} / {total_candidates}")
print(f"Fraction: {100*len(good_fits)/total_candidates:.2f}%")

if len(good_fits) > 0:
    print(f"\nAll good fits (χ² < {chi2_threshold}):")
    for r in good_fits:
        print(f"  {r['ratio_name']}, {r['kv_name']}, {r['phi_name']}: χ² = {r['chi2']:.3f}")
