"""Production-wrapper scorer for the Evidence Closure Score (ECS).

The full-set score delegates to the released production evaluator in
``evaluation/e/e_evaluation.py``. ECS@1, ECS@2, and ECS@3 use the fixed,
deduplicated citation traversal defined in
``evaluation/FROZEN_SCORING_CONTRACT.yaml``.

Full-set scoring supports coverage-map matching, in which a citation to a
containing retrieval unit can cover an associated sentence or table cell.
The K-truncated variants use exact evidence-identifier matching and may
therefore be conservative for predictions that cite only containing units.
This behavior is model-blind and is cross-checked against the independent
reference implementation for every released prediction.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

K_VALUES: Tuple[Any, ...] = (1, 2, 3, "full")

# Resolve the released evaluation package used by the Table 1 scoring path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALUATION_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_EVALUATION_DIR,):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _import_production_evaluator():
    from e.e_evaluation import ModuleE_Evaluator  # noqa: WPS433 -- deliberate lazy, read-only import
    return ModuleE_Evaluator


# ---------------------------------------------------------------------------
# Structural gate -- kept as an EXPLICIT wrapper-level check (not read from
# e_evaluation.py, which does not enforce node_type/edge_type presence as
# part of acs_strict today) so that ecs_strict_at_k_definition's "schema-
# conformant nodes/edges/evidence_sets, node_type/edge_type fields present"
# clause is actually enforced, per the frozen contract's own text -- this is
# additive wrapper logic layered on TOP of the real evaluator's v_passed
# result, never a modification of e_evaluation.py itself.
# ---------------------------------------------------------------------------

def is_structurally_valid(prediction_graph: Optional[Dict]) -> bool:
    if not isinstance(prediction_graph, dict):
        return False
    nodes = prediction_graph.get("nodes")
    edges = prediction_graph.get("edges")
    evidence_sets = prediction_graph.get("evidence_sets")
    if not isinstance(nodes, list) or len(nodes) == 0:
        return False
    if not isinstance(edges, list) or not isinstance(evidence_sets, list):
        return False
    for n in nodes:
        if not isinstance(n, dict) or not n.get("node_type"):
            return False
    for e in edges:
        if not isinstance(e, dict) or not e.get("edge_type"):
            return False
    for es in evidence_sets:
        if not isinstance(es, dict) or not isinstance(es.get("links"), list):
            return False
    return True


def unique_evidence_traversal(prediction_graph: Dict) -> List[str]:
    """Same rule as reference_implementation's traversal (node-array order,
    then within-node evidence_sets link order, dedup on first occurrence,
    orphaned evidence_sets traversed last) -- independently coded here, not
    imported from reference_implementation."""
    nodes = prediction_graph.get("nodes") or []
    evidence_sets = prediction_graph.get("evidence_sets") or []
    node_ids_in_order = [n.get("node_id") for n in nodes if isinstance(n, dict)]

    grouped: Dict[Any, List[Dict]] = {nid: [] for nid in node_ids_in_order}
    orphans: List[Dict] = []
    for es in evidence_sets:
        if not isinstance(es, dict):
            continue
        node = es.get("attached_to_node")
        if node in grouped:
            grouped[node].append(es)
        else:
            orphans.append(es)

    seen: Set[str] = set()
    out: List[str] = []
    for nid in node_ids_in_order:
        for es in grouped[nid]:
            for link in es.get("links") or []:
                eid = link.get("evid_id") if isinstance(link, dict) else None
                if eid and eid not in seen:
                    seen.add(eid)
                    out.append(eid)
    for es in orphans:
        for link in es.get("links") or []:
            eid = link.get("evid_id") if isinstance(link, dict) else None
            if eid and eid not in seen:
                seen.add(eid)
                out.append(eid)
    return out


def _covers_any_gold_set(cited: Set[str], gold_sets: List[Dict]) -> bool:
    if not gold_sets:
        return True
    for gs in gold_sets:
        ids = {l.get("evid_id") for l in (gs.get("links") or []) if isinstance(l, dict) and l.get("evid_id")}
        if ids and ids.issubset(cited):
            return True
    return False


def score_all(
    gold_data: List[Dict],
    pred_data: List[Dict],
    denominator_instance_ids: Optional[List[str]] = None,
    oracle_candidates_by_claim: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    ECS_full (K="full") comes DIRECTLY from the real ModuleE_Evaluator's
    acs_strict column (v_passed AND best_recall>=0.99, full untruncated cited
    set -- order-independent, so this already equals what a full,
    non-truncated traversal-and-dedup would produce). K=1/2/3 are computed by
    this wrapper's own additive traversal+truncation logic, ANDed with the
    real evaluator's v_passed (recovered as acs_noev==1.0) AND this wrapper's
    own is_structurally_valid() check (the schema-conformance clause the
    contract requires but e_evaluation.py does not itself enforce as part of
    acs_strict).

    Model/backbone identity is never read by ANY branch below -- gold_data/
    pred_data carry no field this function inspects for that purpose, and
    denominator_instance_ids/oracle_candidates_by_claim are id-keyed only.
    """
    ModuleE_Evaluator = _import_production_evaluator()
    evaluator = ModuleE_Evaluator()
    report, df = evaluator.run(gold_data, pred_data, oracle_candidates_by_claim=oracle_candidates_by_claim)

    row_by_id = {row["instance_id"]: row for _, row in df.iterrows()}
    pred_map = {p["instance_id"]: p for p in pred_data}
    gold_map = {g["instance_id"]: g for g in gold_data}
    ids = denominator_instance_ids or sorted(gold_map.keys())

    per_instance = []
    for iid in ids:
        row = row_by_id.get(iid)
        pred_record = pred_map.get(iid)
        gold_record = gold_map.get(iid, {"instance_id": iid, "gold_evidence_sets": []})

        if row is None or pred_record is None:
            per_instance.append({
                "instance_id": iid, "structurally_valid": False, "validator_passed": False,
                "unique_evidence_count": 0,
                **{f"ecs_strict_at_{k}": 0 for k in K_VALUES},
            })
            continue

        pred_graph = pred_record.get("prediction") or {}
        struct_ok = bool(is_structurally_valid(pred_graph))
        v_passed = bool(row["acs_noev"] == 1.0)  # acs_noev IS v0&v1&v2 pass, per e_evaluation.py:206
        gate_ok = struct_ok and v_passed

        traversal = unique_evidence_traversal(pred_graph) if struct_ok else []
        gold_sets = gold_record.get("gold_evidence_sets", [])

        scores = {}
        for k in K_VALUES:
            if k == "full":
                # ECS_full: taken directly from the real evaluator's own acs_strict,
                # additionally gated by this wrapper's structural check (production's
                # acs_strict does not itself enforce node_type/edge_type presence).
                scores["ecs_strict_at_full"] = 1 if (struct_ok and bool(row["acs_strict"] == 1.0)) else 0
                continue
            if not gate_ok:
                scores[f"ecs_strict_at_{k}"] = 0
                continue
            subset = set(traversal[:k])
            scores[f"ecs_strict_at_{k}"] = 1 if _covers_any_gold_set(subset, gold_sets) else 0

        per_instance.append({
            "instance_id": iid,
            "structurally_valid": struct_ok,
            "validator_passed": v_passed,
            "unique_evidence_count": len(traversal),
            **scores,
        })

    n = len(per_instance)
    summary = {}
    for k in K_VALUES:
        key = f"ecs_strict_at_{k}"
        summary[key] = {"numerator": sum(r[key] for r in per_instance), "denominator": n}

    return {"n_instances": n, "per_instance": per_instance, "summary": summary}


__all__ = ["is_structurally_valid", "unique_evidence_traversal", "score_all", "K_VALUES"]
