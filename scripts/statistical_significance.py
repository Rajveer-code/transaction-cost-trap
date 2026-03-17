"""
Statistical significance test comparing ML ensemble vs momentum baseline.

Per user request this uses hardcoded counts from previous runs:
 - N = 15120
 - ML_Wins = 8679
 - Momentum_Wins = 8632

Constructs contingency table [[ML_Wins, ML_Losses], [Mom_Wins, Mom_Losses]]
and runs scipy.stats.chi2_contingency. Prints chi2, p-value and interpretation.
"""
from math import isfinite
try:
    from scipy.stats import chi2_contingency
except Exception as e:
    chi2_contingency = None


def main():
    N = 15120
    ML_Wins = 8679
    Mom_Wins = 8632

    ML_Losses = N - ML_Wins
    Mom_Losses = N - Mom_Wins

    table = [[ML_Wins, ML_Losses], [Mom_Wins, Mom_Losses]]

    print("Contingency Table:")
    print(f"ML:       Wins={ML_Wins}, Losses={ML_Losses}")
    print(f"Momentum: Wins={Mom_Wins}, Losses={Mom_Losses}\n")

    if chi2_contingency is None:
        print("scipy is not available in the environment. Install scipy to run the test:")
        print("pip install scipy")
        return

    chi2, p, dof, expected = chi2_contingency(table)

    print(f"Chi2 statistic: {chi2:.6f}")
    print(f"p-value: {p:.6e}")
    print(f"Degrees of freedom: {dof}")
    print("Expected frequencies:")
    print(expected)

    alpha = 0.05
    if p < alpha:
        print("\nInterpretation: p < 0.05 -> Statistically Significant difference.")
    else:
        print("\nInterpretation: p >= 0.05 -> Not Statistically Significant.")


if __name__ == '__main__':
    main()
