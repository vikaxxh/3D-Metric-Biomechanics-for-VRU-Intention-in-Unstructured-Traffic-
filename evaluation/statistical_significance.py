"""
Statistical Significance and Hypothesis Testing
===============================================
Computes:
1. Paired Student's t-test and Wilcoxon signed-rank test between model variants
2. Bootstrap 95% Confidence Intervals for F1-macro and Accuracy
3. Formatted LaTeX table rows for IEEE IROS/ICRA paper submissions
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, List


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes metric mean and 95% bootstrap confidence interval.
    Returns:
        (mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    bootstrapped_scores = np.array(bootstrapped_scores)
    mean_val = float(np.mean(bootstrapped_scores))
    lower_pct = ((1.0 - ci) / 2.0) * 100.0
    upper_pct = (ci + ((1.0 - ci) / 2.0)) * 100.0
    ci_lower = float(np.percentile(bootstrapped_scores, lower_pct))
    ci_upper = float(np.percentile(bootstrapped_scores, upper_pct))

    return mean_val, ci_lower, ci_upper


def paired_significance_test(
    scores_baseline: np.ndarray,
    scores_proposed: np.ndarray
) -> Dict[str, float]:
    """
    Runs paired t-test and Wilcoxon signed-rank test.
    """
    t_stat, t_pval = stats.ttest_rel(scores_proposed, scores_baseline)
    try:
        w_stat, w_pval = stats.wilcoxon(scores_proposed, scores_baseline)
    except Exception:
        w_stat, w_pval = float("nan"), float("nan")

    return {
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pval),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "is_significant_p05": bool(t_pval < 0.05),
        "is_significant_p01": bool(t_pval < 0.01),
    }


def format_latex_table_row(
    method_name: str,
    acc_stats: Tuple[float, float, float],
    f1_stats: Tuple[float, float, float],
    auc_stats: Tuple[float, float, float],
    mpjpe: float = None
) -> str:
    """Formats a row for IEEE table with mean and margin of error."""
    acc_mean, acc_low, acc_high = acc_stats
    acc_err = (acc_high - acc_low) / 2.0

    f1_mean, f1_low, f1_high = f1_stats
    f1_err = (f1_high - f1_low) / 2.0

    auc_mean, auc_low, auc_high = auc_stats
    auc_err = (auc_high - auc_low) / 2.0

    row = f"{method_name} & ${acc_mean:.1f} \\pm {acc_err:.1f}$ & ${f1_mean:.1f} \\pm {f1_err:.1f}$ & ${auc_mean:.3f} \\pm {auc_err:.3f}$"
    if mpjpe is not None:
        row += f" & ${mpjpe:.1f}$"
    row += " \\\\"
    return row
