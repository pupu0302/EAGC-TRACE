"""Holm-Bonferroni step-down p-value correction.

This self-contained pure-Python implementation is used by the offline
statistical reproduction workflow. It has no external dependencies."""
from __future__ import annotations

from typing import List


def holm(pvals: List[float]) -> List[float]:
    """
    Holm step-down correction for multiple comparisons.

    Given a list of raw p-values (one per hypothesis in the family), returns
    the Holm-adjusted p-values in the SAME order as the input list.

    Algorithm (verbatim from the frozen scoring pipeline's
    run_stage6_stats.py, Stage 11):
      1. Sort hypotheses by ascending raw p-value.
      2. For the i-th smallest (rank starting at 0), multiply its p-value by
         (n - rank) and cap at 1.0.
      3. Enforce monotonicity by taking the running maximum over increasing
         rank (the standard Holm step-down guarantee that adjusted p-values
         are non-decreasing in rank).
      4. Return adjusted p-values realigned to the original input order.

    Args:
        pvals: raw p-values, one per hypothesis in the family.

    Returns:
        Holm-adjusted p-values, same length and order as `pvals`.
    """
    # --- BEGIN verbatim algorithm body (from run_stage6_stats.py::holm) ---
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = n - rank
        val = min(1.0, pvals[idx] * factor)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj
    # --- END verbatim algorithm body ---


__all__ = ["holm"]
