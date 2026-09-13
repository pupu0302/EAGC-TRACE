"""
Deterministic empty-evidence negative control.

The paper's frozen 7-comparison Holm family (`FROZEN_ORIGINAL_COMPARISON_FAMILY`
in results/bootstrap/table1_bootstrap_results.json) includes three comparisons
against a legacy empty-evidence control (GPT-4o and GPT-4o-mini backbones):
G_vs_B3_4o, TRACE_vs_B3_4o, TRACE_vs_B3_mini. That control's historical
per-instance model predictions are intentionally not shipped in this package
(see reproduce/README.md) -- it is a
negative control, not a reported Table 1 row, and shipping an old model run
is not required to reproduce it.

The control's defining property is not empirical: when no evidence pool is
ever populated for a claim, ECS_strict's report-internal-pointer admissibility
check cannot be satisfied for ANY instance, by construction of the metric
(see evaluation/FROZEN_SCORING_CONTRACT.yaml). ECS_strict@full is therefore
0.0 for every instance under this control, independent of backbone, document,
or sector -- a fact the maintainers independently verified over the full
789-instance extended population before writing this module, not merely
assumed. See reproduce/README.md for the full justification.

This module provides that deterministic zero vector directly, so the full
7-comparison Holm family can be reproduced from a design-defined constant
plus the public G/TRACE/B3 Flat-Evidence predictions already shipped -- WITHOUT
shipping, re-running, or reconstructing historical model outputs for that control.

build_empty_evidence_control() takes no model name, reads no prediction
file, and makes no network call: it is pure function of the instance_id
list you pass it.
"""
from __future__ import annotations

from typing import Dict, List

PROVENANCE = "DESIGN_DEFINED_EMPTY_EVIDENCE_NEGATIVE_CONTROL"


def build_empty_evidence_control(instance_ids: List[str]) -> List[Dict]:
    """
    Return one record per entry in `instance_ids`, in the same order, each
    with ecs_strict_at_full fixed at 0.0 by definition.

    This is NOT a replay of a historical model run: it does not read any
    prediction file, does not accept a model/backbone name, and performs no
    I/O beyond consuming the `instance_ids` argument already in memory.
    Records are keyed by instance_id (not position), so callers that join by
    instance_id -- as reproduce/reproduce_table1.py's run_full_stats() does
    for every other row -- get correct alignment even if `instance_ids` is
    reordered relative to some other list of the same ids.

    Args:
        instance_ids: the instance ids this control covers (any order,
            duplicates not expected but not deduplicated here -- the caller
            owns membership correctness, exactly as for every other row's
            per-instance records).

    Returns:
        List of {"instance_id": iid, "ecs_strict_at_full": 0.0,
        "provenance": "DESIGN_DEFINED_EMPTY_EVIDENCE_NEGATIVE_CONTROL"}
        dicts, len(...) == len(instance_ids), in input order.
    """
    return [
        {
            "instance_id": iid,
            "ecs_strict_at_full": 0.0,
            "provenance": PROVENANCE,
        }
        for iid in instance_ids
    ]


__all__ = ["build_empty_evidence_control", "PROVENANCE"]
