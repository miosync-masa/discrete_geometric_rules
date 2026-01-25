import numpy as np
from scipy.optimize import minimize, differential_evolution

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

    return np.arctan2(sin_delta, cos_delta), J, sin_delta, cos_delta

# Base values
PI = np.pi
t_r = 1.0
ti = (180 / PI) * t_r
v_base = np.sqrt(15) * ti

# PDG 2024 experimental values with uncertainties
exp_data = {
    'theta12': (13.04, 0.05),   # degrees
    'theta13': (0.20, 0.01),
    'theta23': (2.38, 0.06),
    'J': (3.0e-5, 0.2e-5),
    'V_td': (0.0086, 0.0002),
}

print("="*80)
print("REDUCING FREE PARAMETERS: Geometric constant constraints")
print("="*80)

# 幾何学的定数候補の確認
print("\n[Analysis of previous optimization]")
print(f"  κ₂₃(d)/κ₂₃(u) = 5.7923/6.6734 = {5.7923/6.6734:.6f}")
print(f"  √3/2 = cos(30°) = {np.sqrt(3)/2:.6f}")
print(f"  Difference: {abs(5.7923/6.6734 - np.sqrt(3)/2):.6f} ({abs(5.7923/6.6734 - np.sqrt(3)/2)/(np.sqrt(3)/2)*100:.2f}%)")
print(f"\n  κ_v = 1.2389")
print(f"  31/25 = {31/25:.4f}")
print(f"  Difference: {abs(1.2389 - 31/25):.4f}")

# STEP 1: κ₂₃(d)/κ₂₃(u) = √3/2 を固定
print("\n" + "="*80)
print("STEP 1: Fix κ₂₃(d)/κ₂₃(u) = √3/2 (cos 30°)")
print("Free parameters: 4 (ε₁₂, κ₂₃, κ_v, φ_d)")
print("="*80)

RATIO_FIXED = np.sqrt(3) / 2  # ≈ 0.866

def objective_ratio_fixed(params):
    eps_12, kappa_23_u, kappa_v, phi_d = params
    kappa_23_d = kappa_23_u * RATIO_FIXED  # 固定比！

    H_u = build_H_complex_t23(t_r, kappa_23_u * t_r, 0, ti, v_base)
    _, U_u = diagonalize_and_sort(H_u)

    t_12_d = t_r * (1 + eps_12)
    v_d = v_base * kappa_v
    H_d = build_H_complex_t23(t_12_d, kappa_23_d * t_r, phi_d, ti, v_d)
    _, U_d = diagonalize_and_sort(H_d)

    V = U_u.conj().T @ U_d
    th12, th13, th23, *_ = extract_CKM_angles_from_moduli(V)
    J = compute_jarlskog(V)
    V_td = np.abs(V[2, 0])

    # PDG誤差で正規化したχ²
    chi2 = ((np.degrees(th12) - exp_data['theta12'][0])**2 / exp_data['theta12'][1]**2 +
            (np.degrees(th13) - exp_data['theta13'][0])**2 / exp_data['theta13'][1]**2 +
            (np.degrees(th23) - exp_data['theta23'][0])**2 / exp_data['theta23'][1]**2 +
            (J - exp_data['J'][0])**2 / exp_data['J'][1]**2 +
            (V_td - exp_data['V_td'][0])**2 / exp_data['V_td'][1]**2)
    return chi2

bounds = [(2, 4), (5, 10), (1.1, 1.4), (-0.2, 0.0)]
result = differential_evolution(objective_ratio_fixed, bounds, maxiter=2000, seed=42, polish=True)

eps_12, kappa_23_u, kappa_v, phi_d = result.x
kappa_23_d = kappa_23_u * RATIO_FIXED

print(f"\nOptimized (4 free params, ratio fixed):")
print(f"  ε₁₂      = {eps_12:.4f}")
print(f"  κ₂₃(u)   = {kappa_23_u:.4f}")
print(f"  κ₂₃(d)   = {kappa_23_d:.4f} [= κ₂₃(u) × √3/2]")
print(f"  κ_v      = {kappa_v:.4f}")
print(f"  φ_d      = {np.degrees(phi_d):.4f}°")
print(f"  χ²       = {result.fun:.2f}")

# 結果計算
H_u = build_H_complex_t23(t_r, kappa_23_u * t_r, 0, ti, v_base)
_, U_u = diagonalize_and_sort(H_u)
H_d = build_H_complex_t23(t_r * (1 + eps_12), kappa_23_d * t_r, phi_d, ti, v_base * kappa_v)
_, U_d = diagonalize_and_sort(H_d)
V_CKM = U_u.conj().T @ U_d

th12, th13, th23, s12, c12, s13, c13, s23, c23 = extract_CKM_angles_from_moduli(V_CKM)
delta, J, sin_d, cos_d = extract_delta_from_Vtd(V_CKM, s12, c12, s13, c13, s23, c23)
V_td = np.abs(V_CKM[2, 0])

print(f"\nResults:")
print(f"  θ₁₂ = {np.degrees(th12):.2f}° (exp: {exp_data['theta12'][0]}±{exp_data['theta12'][1]}°)")
print(f"  θ₁₃ = {np.degrees(th13):.2f}° (exp: {exp_data['theta13'][0]}±{exp_data['theta13'][1]}°)")
print(f"  θ₂₃ = {np.degrees(th23):.2f}° (exp: {exp_data['theta23'][0]}±{exp_data['theta23'][1]}°)")
print(f"  J   = {J:.2e} (exp: {exp_data['J'][0]:.1e})")
print(f"  |V_td| = {V_td:.4f} (exp: {exp_data['V_td'][0]})")
print(f"  δ   = {np.degrees(delta):.2f}° (exp: ~70°)")

# STEP 2: さらに κ_v = 31/25 も固定
print("\n" + "="*80)
print("STEP 2: Also fix κ_v = 31/25 = 1.24")
print("Free parameters: 3 (ε₁₂, κ₂₃, φ_d)")
print("="*80)

KV_FIXED = 31/25  # = 1.24

def objective_two_fixed(params):
    eps_12, kappa_23_u, phi_d = params
    kappa_23_d = kappa_23_u * RATIO_FIXED
    kappa_v = KV_FIXED

    H_u = build_H_complex_t23(t_r, kappa_23_u * t_r, 0, ti, v_base)
    _, U_u = diagonalize_and_sort(H_u)

    H_d = build_H_complex_t23(t_r * (1 + eps_12), kappa_23_d * t_r, phi_d, ti, v_base * kappa_v)
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

bounds2 = [(2, 4), (5, 10), (-0.2, 0.0)]
result2 = differential_evolution(objective_two_fixed, bounds2, maxiter=2000, seed=42, polish=True)

eps_12_2, kappa_23_u_2, phi_d_2 = result2.x
kappa_23_d_2 = kappa_23_u_2 * RATIO_FIXED

print(f"\nOptimized (3 free params):")
print(f"  ε₁₂      = {eps_12_2:.4f}")
print(f"  κ₂₃(u)   = {kappa_23_u_2:.4f}")
print(f"  κ₂₃(d)   = {kappa_23_d_2:.4f} [FIXED: × √3/2]")
print(f"  κ_v      = {KV_FIXED:.4f} [FIXED: 31/25]")
print(f"  φ_d      = {np.degrees(phi_d_2):.4f}°")
print(f"  χ²       = {result2.fun:.2f}")

# 結果
H_u = build_H_complex_t23(t_r, kappa_23_u_2 * t_r, 0, ti, v_base)
_, U_u = diagonalize_and_sort(H_u)
H_d = build_H_complex_t23(t_r * (1 + eps_12_2), kappa_23_d_2 * t_r, phi_d_2, ti, v_base * KV_FIXED)
_, U_d = diagonalize_and_sort(H_d)
V2 = U_u.conj().T @ U_d

th12_2, th13_2, th23_2, s12, c12, s13, c13, s23, c23 = extract_CKM_angles_from_moduli(V2)
delta_2, J_2, _, _ = extract_delta_from_Vtd(V2, s12, c12, s13, c13, s23, c23)
V_td_2 = np.abs(V2[2, 0])

print(f"\nResults:")
print(f"  θ₁₂ = {np.degrees(th12_2):.2f}° ({'✓' if abs(np.degrees(th12_2)-exp_data['theta12'][0])<0.5 else '✗'})")
print(f"  θ₁₃ = {np.degrees(th13_2):.2f}° ({'✓' if abs(np.degrees(th13_2)-exp_data['theta13'][0])<0.15 else '✗'})")
print(f"  θ₂₃ = {np.degrees(th23_2):.2f}° ({'✓' if abs(np.degrees(th23_2)-exp_data['theta23'][0])<0.5 else '✗'})")
print(f"  J   = {J_2:.2e} ({'✓' if abs(J_2-exp_data['J'][0])/exp_data['J'][0]<0.3 else '✗'})")
print(f"  |V_td| = {V_td_2:.4f} ({'✓' if abs(V_td_2-exp_data['V_td'][0])/exp_data['V_td'][0]<0.3 else '✗'})")
print(f"  δ   = {np.degrees(delta_2):.2f}° ({'✓' if abs(np.degrees(delta_2)-70)<20 else '✗'})")

# φ_d の離散スキャン
print("\n" + "="*80)
print("STEP 3: Discrete scan of φ_d (geometric candidates)")
print("Free parameters: 2 (ε₁₂, κ₂₃) with φ_d discrete")
print("="*80)

phi_candidates = [
    (-PI/24, "-π/24 = -7.5°"),
    (-PI/20, "-π/20 = -9°"),
    (-PI/30, "-π/30 = -6°"),
    (-PI/36, "-π/36 = -5°"),
    (-2*PI/45, "-2π/45 = -8°"),
]

print(f"\n{'φ_d':<20} {'ε₁₂':<8} {'κ₂₃(u)':<8} {'χ²':<8} {'θ₁₂':<8} {'θ₁₃':<8} {'θ₂₃':<8} {'δ':<8}")
print("-"*90)

best_discrete = None
best_chi2 = float('inf')

for phi_val, phi_name in phi_candidates:
    def obj_discrete(params):
        eps_12, kappa_23_u = params
        kappa_23_d = kappa_23_u * RATIO_FIXED

        H_u = build_H_complex_t23(t_r, kappa_23_u * t_r, 0, ti, v_base)
        _, U_u = diagonalize_and_sort(H_u)
        H_d = build_H_complex_t23(t_r * (1 + eps_12), kappa_23_d * t_r, phi_val, ti, v_base * KV_FIXED)
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

    res = minimize(obj_discrete, [3.0, 7.0], method='Nelder-Mead')

    # 結果計算
    eps, kap = res.x
    H_u = build_H_complex_t23(t_r, kap * t_r, 0, ti, v_base)
    _, U_u = diagonalize_and_sort(H_u)
    H_d = build_H_complex_t23(t_r * (1 + eps), kap * RATIO_FIXED * t_r, phi_val, ti, v_base * KV_FIXED)
    _, U_d = diagonalize_and_sort(H_d)
    V = U_u.conj().T @ U_d

    th12, th13, th23, s12, c12, s13, c13, s23, c23 = extract_CKM_angles_from_moduli(V)
    delta, J, _, _ = extract_delta_from_Vtd(V, s12, c12, s13, c13, s23, c23)

    print(f"{phi_name:<20} {eps:<8.3f} {kap:<8.3f} {res.fun:<8.2f} {np.degrees(th12):<8.2f} {np.degrees(th13):<8.2f} {np.degrees(th23):<8.2f} {np.degrees(delta):<8.1f}")

    if res.fun < best_chi2:
        best_chi2 = res.fun
        best_discrete = (phi_name, phi_val, eps, kap, np.degrees(delta))

print(f"\n✅ Best discrete φ_d: {best_discrete[0]}")
print(f"   χ² = {best_chi2:.2f}, δ = {best_discrete[4]:.1f}°")
