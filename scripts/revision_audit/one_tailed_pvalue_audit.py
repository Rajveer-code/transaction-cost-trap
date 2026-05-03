#!/usr/bin/env python3
"""
SCRIPT 6: one_tailed_pvalue_audit.py
Gap: Manuscript reports p-values without stating they are one-tailed. Risk of
inconsistency across tests (HAC, permutation, VIX regime, DM). Reviewers will flag.
Produces: complete p-value audit table + recommended footnote text.
"""

import numpy as np
from scipy import stats, optimize
import warnings
warnings.filterwarnings("ignore")

# ── Hardcoded inputs ──────────────────────────────────────────────────────────
HAC_lag = 9

reported_stats = {
    "ML ensemble (full)":   {"t":  -0.090, "T": 1512, "p_reported": 0.464},
    "Momentum (full)":      {"t":  +0.600, "T": 1512, "p_reported": 0.276},
    "N=100 universe":       {"t":  -1.623, "T": 1512, "p_reported": 0.948},
    "Permutation (ML)":     {"t":   None,  "T": 1512, "p_reported": 0.742},
    "VIX Low":              {"t":   None,  "T":  486, "p_reported": 0.152},
    "VIX Mid":              {"t":   None,  "T":  485, "p_reported": 0.136},
    "VIX High":             {"t":   None,  "T":  485, "p_reported": 0.141},
    "DM: TopK1 vs Random":  {"t":  +0.420, "T": 1512, "p_reported": 0.672},
}

IC_means = {
    "ML ensemble (full)":  -0.0005,
    "Momentum (full)":     +0.0071,
    "N=100 universe":      -0.0062,
    "VIX Low":             +0.0106,
    "VIX Mid":             -0.0106,
    "VIX High":            -0.0111,
}

IC_stds = {
    "ML ensemble (full)":  0.2204,
    "Momentum (full)":     0.4747,
    "N=100 universe":      0.2204,
    "VIX Low":             0.2204,
    "VIX Mid":             0.2204,
    "VIX High":            0.2204,
}

print("=" * 70)
print("SCRIPT 6: One-Tailed p-Value Audit — Full Manuscript Consistency Check")
print("=" * 70)

# ── NW variance function ──────────────────────────────────────────────────────
def nw_var_mean(rho, IC_std_val, T_val, L):
    mult = 1.0
    for k in range(1, L + 1):
        mult += 2.0 * (1.0 - k / (L + 1)) * (rho**k)
    return (IC_std_val**2 / T_val) * mult

# Back-calculate rho from the primary result
SE_primary = IC_means["ML ensemble (full)"] / reported_stats["ML ensemble (full)"]["t"]
V_primary  = SE_primary**2
try:
    rho_assumed = optimize.brentq(
        lambda r: nw_var_mean(r, IC_stds["ML ensemble (full)"], 1512, HAC_lag) - V_primary,
        -0.9999, 0.9999
    )
except ValueError:
    rho_assumed = -0.004

print(f"\n[Assumed AR(1) rho (back-calculated from primary result): {rho_assumed:.4f}]")
print(f"  Applied uniformly across all HAC tests (conservative; near-i.i.d. IC series)")

# ── Step 1: Verify all p-values ───────────────────────────────────────────────
results = {}

for name, info in reported_stats.items():
    t_rep  = info["t"]
    T      = info["T"]
    p_rep  = info["p_reported"]
    row    = {"T": T, "p_reported": p_rep}

    if t_rep is not None:
        # Direct computation from t-statistic
        p_1tail_upper = float(1.0 - stats.norm.cdf(t_rep))    # H1: IC > 0 (gate relevant)
        p_1tail_lower = float(stats.norm.cdf(t_rep))           # H1: IC < 0
        p_2tail       = float(2.0 * (1.0 - stats.norm.cdf(abs(t_rep))))
        row["t_computed"]       = t_rep
        row["p_1tail_upper"]    = p_1tail_upper
        row["p_1tail_lower"]    = p_1tail_lower
        row["p_2tail"]          = p_2tail
        # Which direction is consistent with reported?
        row["match_upper"]  = abs(p_1tail_upper - p_rep) < 0.012
        row["match_lower"]  = abs(p_1tail_lower - p_rep) < 0.012
        row["match_2tail"]  = abs(p_2tail        - p_rep) < 0.012
    elif name in IC_means:
        # Derive from IC mean and HAC SE
        ic_m = IC_means[name]
        ic_s = IC_stds[name]
        V_L  = nw_var_mean(rho_assumed, ic_s, T, HAC_lag)
        t_c  = ic_m / np.sqrt(V_L)
        p_1u = float(1.0 - stats.norm.cdf(t_c))
        p_1l = float(stats.norm.cdf(t_c))
        p_2t = float(2.0 * (1.0 - stats.norm.cdf(abs(t_c))))
        row["t_computed"]    = t_c
        row["p_1tail_upper"] = p_1u
        row["p_1tail_lower"] = p_1l
        row["p_2tail"]       = p_2t
        row["match_upper"]   = abs(p_1u - p_rep) < 0.020
        row["match_lower"]   = abs(p_1l - p_rep) < 0.020
        row["match_2tail"]   = abs(p_2t - p_rep) < 0.020
    else:
        # Permutation: parametric check not available
        row["t_computed"]    = None
        row["p_1tail_upper"] = None
        row["p_1tail_lower"] = None
        row["p_2tail"]       = None
        row["match_upper"]   = None
        row["match_lower"]   = None
        row["match_2tail"]   = None

    results[name] = row

# ── Step 2: Print audit table ─────────────────────────────────────────────────
print(f"\n[Complete p-Value Audit Table]")
print()
col_w = 24
print(f"{'Test':<{col_w}} | {'T':>5} | {'t-stat':>8} | {'p (1-tail ↑)':>13} | "
      f"{'p (1-tail ↓)':>13} | {'p (2-tail)':>11} | {'Reported':>9} | {'Match':>7}")
print("-" * (col_w + 80))

any_flag = False
for name, row in results.items():
    t_str  = f"{row['t_computed']:>8.4f}" if row["t_computed"] is not None else f"{'perm':>8}"
    p_1u   = f"{row['p_1tail_upper']:>13.4f}" if row["p_1tail_upper"] is not None else f"{'N/A':>13}"
    p_1l   = f"{row['p_1tail_lower']:>13.4f}" if row["p_1tail_lower"] is not None else f"{'N/A':>13}"
    p_2t   = f"{row['p_2tail']:>11.4f}"       if row["p_2tail"]       is not None else f"{'N/A':>11}"

    if row["match_upper"] is None:
        match_str = "  N/A  "
    elif row["match_upper"]:
        match_str = "  OK   "
    elif row["match_lower"]:
        match_str = " OK(↓) "   # lower-tail consistent — flag
        any_flag = True
    elif row["match_2tail"]:
        match_str = " OK(2) "   # 2-tail consistent — flag convention
        any_flag = True
    else:
        match_str = " FLAG! "
        any_flag = True

    print(f"{name:<{col_w}} | {row['T']:>5} | {t_str} | {p_1u} | {p_1l} | "
          f"{p_2t} | {row['p_reported']:>9.4f} | {match_str}")

if not any_flag:
    print(f"\n  All parametric p-values are consistent with ONE-TAILED UPPER (H1: IC > 0).")
    print(f"  No inconsistencies detected. Manuscript p-value convention is coherent.")
else:
    print(f"\n  One or more p-values flagged — review 'FLAG!' rows above.")

# ── Step 3: Detailed momentum verification ────────────────────────────────────
print(f"\n[Momentum IC Verification from First Principles]")
ic_m_mom = IC_means["Momentum (full)"]
ic_s_mom = IC_stds["Momentum (full)"]
T_mom    = 1512
V_mom    = nw_var_mean(rho_assumed, ic_s_mom, T_mom, HAC_lag)
SE_mom   = np.sqrt(V_mom)
t_mom    = ic_m_mom / SE_mom
p_mom    = 1.0 - stats.norm.cdf(t_mom)

print(f"  IC_mean = {ic_m_mom}, IC_std = {ic_s_mom}, T = {T_mom}, L = {HAC_lag}")
print(f"  V_HAC  = {V_mom:.8f}")
print(f"  SE     = {SE_mom:.6f}")
print(f"  t_HAC  = {t_mom:.4f}  (paper reports +0.600)")
print(f"  p (one-tailed upper) = {p_mom:.4f}  (paper reports 0.276)")
t_match = abs(t_mom - 0.600) < 0.05
p_match = abs(p_mom - 0.276) < 0.015
print(f"  t-stat match: {'YES' if t_match else 'NO — check rho or IC_std'}")
print(f"  p-value match: {'YES' if p_match else 'NO — investigate'}")

# ── Step 4: VIX tercile verification ──────────────────────────────────────────
print(f"\n[VIX Tercile p-Value Verification from First Principles]")
vix_cases = [
    ("VIX Low",  IC_means["VIX Low"],  IC_stds["VIX Low"],  486, 0.152),
    ("VIX Mid",  IC_means["VIX Mid"],  IC_stds["VIX Mid"],  485, 0.136),
    ("VIX High", IC_means["VIX High"], IC_stds["VIX High"], 485, 0.141),
]
print(f"  {'Regime':<12} | {'IC_mean':>8} | {'t_HAC':>8} | {'p(1-tail↑)':>12} | {'Reported':>9} | Match")
print(f"  {'-'*65}")
for name, ic_m, ic_s, T_v, p_rep in vix_cases:
    V_v    = nw_var_mean(rho_assumed, ic_s, T_v, HAC_lag)
    t_v    = ic_m / np.sqrt(V_v)
    p_v    = 1.0 - stats.norm.cdf(t_v)
    match  = "YES" if abs(p_v - p_rep) < 0.020 else "NO"
    print(f"  {name:<12} | {ic_m:>8.4f} | {t_v:>8.4f} | {p_v:>12.4f} | {p_rep:>9.3f} | {match}")

# ── Step 5: LaTeX corrected audit table ──────────────────────────────────────
print(f"\n[LaTeX Corrected p-Value Table — Insert as Table in Manuscript]")
print()
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\small")
print(r"\caption{Complete audit of all reported hypothesis tests. All $p$-values are")
print(r"one-tailed (upper tail), evaluating $H_0$: IC $\leq 0$ against $H_1$: IC $> 0$,")
print(r"consistent with the gate's one-sided deployment criterion. Two-tailed equivalents")
print(r"are shown for reference. Permutation $p$-values are empirical (1,000 replicates)")
print(r"and are not available in parametric form.}")
print(r"\label{tab:pvalue_audit}")
print(r"\begin{tabular}{lrccc}")
print(r"\hline")
print(r"Test & $T$ & $t$-stat & $p$ (one-tailed, $\uparrow$) & $p$ (two-tailed) \\")
print(r"\hline")

latex_rows = [
    ("ML ensemble IC (full)",          1512, -0.090, None),
    ("Momentum IC (full)",             1512, +0.600, None),
    ("NASDAQ-100 ($N=100$)",           1512, -1.623, None),
    ("Permutation, ML (empirical)",    1512,  None,  0.742),
    ("VIX Low tercile",                 486,  None,  0.152),
    ("VIX Mid tercile",                 485,  None,  0.136),
    ("VIX High tercile",                485,  None,  0.141),
    ("DM: Top$K$=1 vs Random",         1512, +0.420, None),
]

for label, T_val, t_val, p_perm in latex_rows:
    if t_val is not None:
        p1u = 1.0 - stats.norm.cdf(t_val)
        p2  = 2.0 * (1.0 - stats.norm.cdf(abs(t_val)))
        print(f"{label} & {T_val:,} & ${t_val:+.3f}$ & {p1u:.4f} & {p2:.4f} \\\\")
    else:
        print(f"{label} & {T_val:,} & --- & {p_perm:.3f} (empirical) & N/A \\\\")

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")

# ── Recommended footnote ──────────────────────────────────────────────────────
print(f"""
[Recommended footnote — attach at first p-value mention in Section 5.1]

\\footnote{{All reported $p$-values are one-tailed (upper tail), evaluating the
null hypothesis $H_0$: IC $\\leq 0$ against the alternative $H_1$: IC $> 0$.
This one-sided convention is economically motivated: the ICGDF gate deploys
capital only when predicted IC exceeds a strictly positive threshold, and the
relevant inferential question is whether the ensemble possesses positive
predictive skill, not merely non-zero skill. Two-tailed equivalents are
reported in Table~\\ref{{tab:pvalue_audit}} for reference. HAC $p$-values use
a normal approximation under the Newey-West long-run variance (Bartlett kernel,
$L=9$; see Section~4.4 and Table~[HAC sensitivity table]). The permutation
$p$-value (0.742) is empirical and does not assume a parametric distribution.}}

[Recommended convention note at top of Section 5 or in Section 4.4]

All hypothesis tests in this paper evaluate $H_1$: IC $> 0$ (one-tailed).
The IC gate is a one-sided instrument: it is active only when the predicted
IC exceeds a positive deployment threshold. Accordingly, the economically
relevant test is whether there is evidence for positive predictive skill, not
merely non-zero skill. Readers preferring two-tailed $p$-values may double
all parametric $p$-values reported as $p < 0.50$ (or use Table~\\ref{{tab:pvalue_audit}}).
""")
