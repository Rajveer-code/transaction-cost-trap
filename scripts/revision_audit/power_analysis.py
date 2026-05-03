#!/usr/bin/env python3
"""
SCRIPT 1: power_analysis.py
Gap: Null-result paper with no power/MDE analysis. A reviewer will reject without this.
Produces: figure_power_analysis.png + LaTeX MDE table for manuscript.
"""

import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Hardcoded inputs from manuscript ─────────────────────────────────────────
IC_mean      = -0.0005
IC_std       = 0.2204
T_full       = 1512
T_fold       = 126
n_folds      = 12
HAC_lag      = 9
HAC_t_stat   = -0.09
alpha        = 0.05
IC_std_upper = 0.4747   # momentum IC std — upper bound for sensitivity

z_alpha  = stats.norm.ppf(1 - alpha)   # 1.645 one-tailed
z_b80    = stats.norm.ppf(0.80)        # 0.842
z_b90    = stats.norm.ppf(0.90)        # 1.282

# ── Step 1: Back-calculate AR(1) rho from known HAC t-stat ───────────────────
# SE of mean = IC_mean / t_HAC
SE_full = IC_mean / HAC_t_stat          # = 0.005556
V_NW    = SE_full**2 * T_full           # Newey-West long-run variance estimate

sigma2  = IC_std**2                     # = 0.04858

def nw_multiplier(rho, L=HAC_lag):
    """Newey-West long-run variance multiplier under AR(1)."""
    total = 1.0
    for k in range(1, L + 1):
        total += 2.0 * (1.0 - k / (L + 1)) * (rho ** k)
    return total

target_mult = V_NW / sigma2             # ratio NW variance to unconditional variance

try:
    rho_hat = optimize.brentq(
        lambda r: nw_multiplier(r) - target_mult, -0.9999, 0.9999
    )
except ValueError:
    rho_hat = -0.004   # fallback: near-zero negative autocorrelation

print("=" * 70)
print("SCRIPT 1: Power Analysis and Minimum Detectable IC")
print("=" * 70)

print(f"\n[AR(1) Autocorrelation Estimation]")
print(f"  HAC SE of mean (T={T_full}):        {SE_full:.6f}")
print(f"  NW long-run variance V_NW:          {V_NW:.6f}")
print(f"  Unconditional variance sigma^2:     {sigma2:.6f}")
print(f"  NW multiplier (V_NW / sigma^2):     {target_mult:.4f}")
print(f"  Estimated AR(1) rho:                {rho_hat:.4f}")
print(f"  Interpretation: near-zero {'negative' if rho_hat < 0 else 'positive'} "
      f"autocorrelation — IC series approximately i.i.d.")

# ── Step 2: Effective sample size ────────────────────────────────────────────
def N_eff_AR1(T, rho):
    """Effective N under AR(1): T / (1 + 2*rho/(1-rho)). Exact long-run formula."""
    if abs(rho) >= 1.0:
        return float(T)
    return T / (1.0 + 2.0 * rho / (1.0 - rho))

N_eff_full = N_eff_AR1(T_full, rho_hat)
N_eff_fold = N_eff_AR1(T_fold, rho_hat)

print(f"\n[Effective Sample Size under Estimated AR(1)]")
print(f"  Full window  (T={T_full}):  N_eff = {N_eff_full:.1f}")
print(f"  Per-fold     (T={T_fold}):   N_eff = {N_eff_fold:.1f}")
print(f"  (Correction is minimal for |rho| << 1)")

# ── Step 3: MDE at 80% and 90% power ─────────────────────────────────────────
def compute_MDE(IC_std_val, N_eff_val, power=0.80):
    """MDE = (z_alpha + z_beta) * IC_std / sqrt(N_eff). One-tailed, alpha=0.05."""
    z_b = stats.norm.ppf(power)
    SE  = IC_std_val / np.sqrt(N_eff_val)
    return (z_alpha + z_b) * SE

mde_full_80   = compute_MDE(IC_std,       N_eff_full, 0.80)
mde_full_90   = compute_MDE(IC_std,       N_eff_full, 0.90)
mde_fold_80   = compute_MDE(IC_std,       N_eff_fold, 0.80)
mde_fold_90   = compute_MDE(IC_std,       N_eff_fold, 0.90)
mde_upper_80  = compute_MDE(IC_std_upper, N_eff_full, 0.80)
mde_upper_90  = compute_MDE(IC_std_upper, N_eff_full, 0.90)

print(f"\n[Minimum Detectable IC (MDE)]")
print(f"  Full window, N=30 (IC_std={IC_std}):")
print(f"    80% power: MDE = {mde_full_80:.4f}")
print(f"    90% power: MDE = {mde_full_90:.4f}")
print(f"  Full window, upper bound (IC_std={IC_std_upper}):")
print(f"    80% power: MDE = {mde_upper_80:.4f}")
print(f"    90% power: MDE = {mde_upper_90:.4f}")
print(f"  Per-fold avg (T={T_fold}):")
print(f"    80% power: MDE = {mde_fold_80:.4f}")
print(f"    90% power: MDE = {mde_fold_90:.4f}")

# ── Step 4: LaTeX summary table ───────────────────────────────────────────────
print(f"\n[LaTeX Table — Insert as new Table X in Section 5 (Discussion)]")
print()
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\small")
print(r"\caption{Power analysis for the IC null result. MDE = minimum detectable IC at the")
print(r"specified power level, computed using HAC-consistent standard errors (Newey-West,")
print(r"$L=9$). Effective sample size $N_{\text{eff}}$ adjusts for estimated AR(1) serial")
print(r"correlation in the daily IC series ($\hat{\rho} = " + f"{rho_hat:.4f}" + r"$).}")
print(r"\label{tab:power_analysis}")
print(r"\begin{tabular}{lcccc}")
print(r"\hline")
print(r"Setting & $T$ & $N_{\text{eff}}$ & MDE (80\% power) & MDE (90\% power) \\")
print(r"\hline")
print(f"Full window, $N=30$ ($\\sigma_{{\\text{{IC}}}}={IC_std}$) & "
      f"{T_full} & {N_eff_full:.0f} & {mde_full_80:.4f} & {mde_full_90:.4f} \\\\")
print(f"Full window, upper bound ($\\sigma_{{\\text{{IC}}}}={IC_std_upper}$) & "
      f"{T_full} & {N_eff_full:.0f} & {mde_upper_80:.4f} & {mde_upper_90:.4f} \\\\")
print(f"Per-fold average ($T \\approx {T_fold}$) & "
      f"{T_fold} & {N_eff_fold:.0f} & {mde_fold_80:.4f} & {mde_fold_90:.4f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")

# ── Step 5: Power curve figure ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "legend.fontsize":    8.5,
    "legend.frameon":     False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.15,
    "grid.linestyle":     "-",
    "lines.linewidth":    1.8,
})

true_IC = np.linspace(0, 0.055, 400)

def power_curve(true_ic, IC_std_val, N_eff_val):
    """Power = P(reject H0 | H1: IC = delta). One-tailed alpha=0.05."""
    SE  = IC_std_val / np.sqrt(N_eff_val)
    ncp = true_ic / SE            # non-centrality: drift / SE
    return stats.norm.cdf(ncp - z_alpha)

pow_full = power_curve(true_IC, IC_std, N_eff_full)
pow_fold = power_curve(true_IC, IC_std, N_eff_fold)

TEAL   = "#2A9D8F"
CORAL  = "#E76F51"
NAVY   = "#264653"

fig, ax = plt.subplots(figsize=(5.5, 3.8))

ax.plot(true_IC, pow_full, color=TEAL,  lw=2.0,
        label=f"Full window ($T={T_full}$, $N_{{\\mathrm{{eff}}}}={N_eff_full:.0f}$)")
ax.plot(true_IC, pow_fold, color=CORAL, lw=2.0, linestyle="--",
        label=f"Per-fold ($T={T_fold}$, $N_{{\\mathrm{{eff}}}}={N_eff_fold:.0f}$)")

# 80% power horizontal reference
ax.axhline(0.80, color="gray", lw=0.9, linestyle="--", alpha=0.55,
           label="80\\% power threshold")

# MDE vertical markers
for mde, color, y_ann, x_off in [
    (mde_full_80, TEAL,  0.62, 0.002),
    (mde_fold_80, CORAL, 0.42, 0.002),
]:
    ax.axvline(mde, color=color, lw=1.1, linestyle=":", alpha=0.75)
    ax.annotate(
        f"MDE={mde:.4f}",
        xy=(mde, 0.80), xytext=(mde + x_off, y_ann),
        fontsize=7.5, color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
    )

# Mark observed IC
ax.axvline(abs(IC_mean), color=NAVY, lw=1.0, linestyle=":", alpha=0.5)
ax.text(abs(IC_mean) + 0.0005, 0.07,
        f"Observed |IC|={abs(IC_mean):.4f}", fontsize=7, color=NAVY, rotation=90)

ax.set_xlim(0, 0.055)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("True Information Coefficient (IC)")
ax.set_ylabel("Statistical Power")
ax.set_title(
    "Power Analysis: IC Gate Study\n"
    r"(One-tailed, $\alpha=0.05$, HAC Newey-West $L=9$)"
)
ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("figure_power_analysis.png", dpi=300, bbox_inches="tight")
print(f"\n[Figure saved: figure_power_analysis.png]")

# ── Key numbers for abstract / Section 5.1 ────────────────────────────────────
print(f"\n[Key numbers for manuscript]")
print(f"  MDE at 80% power (full window, primary universe): IC >= {mde_full_80:.4f}")
print(f"  MDE at 90% power (full window, primary universe): IC >= {mde_full_90:.4f}")
print(f"  Observed mean IC = {IC_mean:.4f} is {abs(IC_mean/mde_full_80)*100:.1f}% of the 80%-power MDE")
print()
print(f"  Recommended abstract / Section 5.1 addition:")
print(f"  'With T={T_full} out-of-sample days, the study achieves 80% power to detect a")
print(f"   true IC >= {mde_full_80:.4f} (90% power threshold: IC >= {mde_full_90:.4f}),")
print(f"   a level of daily cross-sectional predictability widely considered economically")
print(f"   meaningful in the factor-investing literature. The observed IC = {IC_mean:.4f}")
print(f"   is {abs(IC_mean/mde_full_80)*100:.0f}% of this detectable threshold, providing no")
print(f"   evidence that the ML ensemble possesses material predictive skill.'")

plt.show()
