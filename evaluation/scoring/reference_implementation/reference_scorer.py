"""Independent reference implementation of the frozen scoring contract.

The implementation reads only the released record shapes and written scoring
definitions; it does not import the production evaluator. It is deterministic,
model-blind, and has no network or filesystem side effects.

For the full-set score, citation matching supports ``meta.coverage_map``: a
citation to a containing unit can cover an associated sentence or table cell.
For ECS@1, ECS@2, and ECS@3, the scorer applies the contract's exact-ID,
deduplicated traversal. The latter can be conservative for predictions that
cite only containing units. These semantics mirror the production wrapper and
are checked against it for every released prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

K_VALUES: Tuple[Any, ...] = (1, 2, 3, "full")


# ---------------------------------------------------------------------------
# Structural / validator gate
# (FROZEN_SCORING_CONTRACT.yaml::ecs_strict_at_k_definition: "valid JSON,
#  schema-conformant nodes/edges/evidence_sets, node_type/edge_type fields
#  present")
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
            return False  # a node without node_type (e.g. mock fallback's bare "type") is not schema-conformant
    for e in edges:
        if not isinstance(e, dict) or not e.get("edge_type"):
            return False
    for es in evidence_sets:
        if not isinstance(es, dict) or not isinstance(es.get("links"), list):
            return False
    return True


def validator_gate_passed(validator_report: Optional[Dict]) -> bool:
    vr = validator_report or {}
    return bool(
        vr.get("v0_gate", {}).get("pass") is True
        and vr.get("v1_gate", {}).get("pass") is True
        and vr.get("v2_gate", {}).get("pass") is True
    )


# ---------------------------------------------------------------------------
# unique_evidence_traversal_order
# "traverse the final predicted JSON's nodes array in array order ... and
#  within each node traverse its evidence_sets[*].links array in array
#  order" -- interpreted per the EAGC schema (evidence_sets is a TOP-LEVEL
#  array of {set_id, attached_to_node, links}, not nested inside each node)
#  as: for node in nodes (array order): for es in evidence_sets (array
#  order) where es.attached_to_node == node.node_id: for link in es.links
#  (array order): evid_id. Dedup on first occurrence across this whole
#  traversal. evidence_sets with no attached_to_node match to any node
#  (orphaned) are traversed LAST, in evidence_sets array order, so no
#  evid_id is silently dropped from the traversal.
# ---------------------------------------------------------------------------

def unique_evidence_traversal(prediction_graph: Dict) -> List[str]:
    nodes = prediction_graph.get("nodes") or []
    evidence_sets = prediction_graph.get("evidence_sets") or []

    node_order = [n.get("node_id") for n in nodes if isinstance(n, dict)]
    es_by_node: Dict[Any, List[Dict]] = {}
    orphan_es: List[Dict] = []
    for es in evidence_sets:
        if not isinstance(es, dict):
            continue
        attached = es.get("attached_to_node")
        if attached in node_order:
            es_by_node.setdefault(attached, []).append(es)
        else:
            orphan_es.append(es)

    seen: Set[str] = set()
    ordered: List[str] = []

    def _consume(es_list: List[Dict]) -> None:
        for es in es_list:
            for link in es.get("links") or []:
                if not isinstance(link, dict):
                    continue
                eid = link.get("evid_id")
                if eid and eid not in seen:
                    seen.add(eid)
                    ordered.append(eid)

    for node_id in node_order:
        _consume(es_by_node.get(node_id, []))
    _consume(orphan_es)  # orphaned evidence_sets last, in array order

    return ordered


# ---------------------------------------------------------------------------
# ECS_strict@K
# ---------------------------------------------------------------------------

def _covers_any_gold_set(cited: Set[str], gold_sets: List[Dict]) -> bool:
    """True if `cited` fully covers (recall == 1.0) AT LEAST ONE gold-admissible
    evidence set. No gold sets at all => vacuously covered (mirrors production's
    treatment of an instance with no gold evidence requirement).

    Used ONLY for K=1/2/3 (exact-evid_id-match, order-truncated) -- see
    _best_recall_with_coverage_map below for the coverage-map-aware "full"
    computation this module's v0.2 fix adds.
    """
    if not gold_sets:
        return True
    for gs in gold_sets:
        gold_ids = {
            link.get("evid_id") for link in (gs.get("links") or [])
            if isinstance(link, dict) and link.get("evid_id")
        }
        if not gold_ids:
            continue
        if gold_ids.issubset(cited):
            return True
    return False


# ---------------------------------------------------------------------------
# Coverage-map-aware "full" computation (v0.2 fix for defect D2).
# Independently reimplements ModuleE_Evaluator._evidence_metrics's _is_cited/
# best_recall semantics (scripts/pipeline/e/e_evaluation.py:375-484, read but
# NOT imported/called) using only the shapes FROZEN_SCORING_CONTRACT.yaml
# already documents: pred["meta"]["coverage_map"] as {unit_id: [sentence_id]}.
# ---------------------------------------------------------------------------

def _build_sent_to_unit(coverage_map: Optional[Dict[str, List[str]]]) -> Dict[str, str]:
    """Reverse index: sentence_id -> unit_id, from {unit_id: [sentence_id]}.
    If the same sentence_id appears under multiple units (a malformed/
    overlapping map), the LAST unit encountered in dict-iteration order wins
    -- an explicit, documented tie-break, not silent nondeterminism, since
    real coverage_map dicts are not expected to have this shape but a
    fixture might probe it."""
    sent_to_unit: Dict[str, str] = {}
    if coverage_map:
        for unit_id, sent_ids in coverage_map.items():
            if not isinstance(sent_ids, list):
                continue
            for sid in sent_ids:
                if isinstance(sid, str):
                    sent_to_unit[sid] = unit_id
    return sent_to_unit


def _is_cited_with_coverage(gold_id: str, cited: Set[str], sent_to_unit: Dict[str, str]) -> bool:
    """True if gold_id is cited exactly, OR gold_id is a sentence_id whose
    containing unit_id (per coverage_map) is cited. Mirrors the real
    evaluator's _is_cited exactly: exact match first, then one-hop lookup
    through the reverse coverage index -- no transitive/multi-hop chaining,
    since the real function performs none either (a single dict.get)."""
    if gold_id in cited:
        return True
    unit_id = sent_to_unit.get(gold_id)
    return unit_id is not None and unit_id in cited


def _best_recall_with_coverage_map(
    cited: Set[str], gold_sets: List[Dict], sent_to_unit: Dict[str, str],
) -> float:
    """max over gold_evidence_sets of (fraction of that set's evid_ids that
    are _is_cited_with_coverage), mirroring ModuleE_Evaluator._evidence_metrics's
    best_recall computation exactly, including its no-gold-sets vacuous-pass
    (1.0) and its per-set empty-links skip (a gold set with zero evid_id
    links contributes nothing and is skipped, never scored as recall=0)."""
    if not gold_sets:
        return 1.0
    best = 0.0
    for gs in gold_sets:
        gold_ids = [
            link.get("evid_id") for link in (gs.get("links") or [])
            if isinstance(link, dict) and link.get("evid_id")
        ]
        if not gold_ids:
            continue
        recall = sum(1 for gid in gold_ids if _is_cited_with_coverage(gid, cited, sent_to_unit)) / len(gold_ids)
        if recall > best:
            best = recall
    return best


def score_instance(pred_record: Optional[Dict], gold_record: Dict) -> Dict[str, Any]:
    """
    pred_record: {instance_id, prediction: {nodes, edges, evidence_sets}, validator_report}
    gold_record: {instance_id, gold_evidence_sets: [...]}
    Returns {ecs_strict_at_1, ecs_strict_at_2, ecs_strict_at_3, ecs_strict_full,
             unique_evidence_count, structurally_valid, validator_passed}
    Model/backbone identity is NEVER read -- pred_record carries no such field
    that this function inspects, and none of the branches below key on one.
    """
    gold_sets = gold_record.get("gold_evidence_sets", [])

    if not pred_record:
        return {
            "instance_id": gold_record.get("instance_id"),
            "structurally_valid": False, "validator_passed": False,
            "unique_evidence_count": 0,
            **{f"ecs_strict_at_{k}": 0 for k in K_VALUES},
        }

    pred_graph = pred_record.get("prediction") or {}
    struct_ok = is_structurally_valid(pred_graph)
    val_ok = validator_gate_passed(pred_record.get("validator_report"))
    gate_ok = struct_ok and val_ok

    traversal = unique_evidence_traversal(pred_graph) if struct_ok else []
    coverage_map = (pred_record.get("meta") or {}).get("coverage_map") or {}
    sent_to_unit = _build_sent_to_unit(coverage_map)

    scores: Dict[str, int] = {}
    for k in K_VALUES:
        if not gate_ok:
            scores[f"ecs_strict_at_{k}"] = 0
            continue
        if k == "full":
            # Coverage-map-aware, order-independent -- matches the real
            # evaluator's own untruncated, non-traversal-based semantics.
            best_recall = _best_recall_with_coverage_map(set(traversal), gold_sets, sent_to_unit)
            scores["ecs_strict_at_full"] = 1 if best_recall >= 0.99 else 0
            continue
        # K=1/2/3: unchanged exact-evid_id-match traversal+subset logic
        # (frozen K-truncation contract; NONMATERIAL_CONSERVATIVE_DIFFERENCE).
        subset = set(traversal[: int(k)])
        scores[f"ecs_strict_at_{k}"] = 1 if _covers_any_gold_set(subset, gold_sets) else 0

    return {
        "instance_id": pred_record.get("instance_id", gold_record.get("instance_id")),
        "structurally_valid": struct_ok,
        "validator_passed": val_ok,
        "unique_evidence_count": len(traversal),
        **scores,
    }


def score_all(
    gold_data: List[Dict],
    pred_data: List[Dict],
    denominator_instance_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Scores every instance in `denominator_instance_ids` (defaults to the gold
    set's instance_ids -- the frozen manifest's instance_id set in real use).
    A gold or prediction record missing for an id in the denominator scores 0
    for every K (FROZEN_SCORING_CONTRACT.yaml::failure_handling: fixed
    denominator, never complete-case deletion).
    """
    gold_map = {g["instance_id"]: g for g in gold_data}
    pred_map = {p["instance_id"]: p for p in pred_data}
    ids = denominator_instance_ids or sorted(gold_map.keys())

    per_instance = []
    for iid in ids:
        gold_record = gold_map.get(iid, {"instance_id": iid, "gold_evidence_sets": []})
        pred_record = pred_map.get(iid)
        per_instance.append(score_instance(pred_record, gold_record))

    n = len(per_instance)
    summary = {}
    for k in K_VALUES:
        key = f"ecs_strict_at_{k}"
        summary[key] = {
            "numerator": sum(r[key] for r in per_instance),
            "denominator": n,
        }
    return {
        "n_instances": n,
        "per_instance": per_instance,
        "summary": summary,
    }


__all__ = [
    "is_structurally_valid", "validator_gate_passed", "unique_evidence_traversal",
    "score_instance", "score_all", "K_VALUES",
    "_build_sent_to_unit", "_is_cited_with_coverage", "_best_recall_with_coverage_map",
]
