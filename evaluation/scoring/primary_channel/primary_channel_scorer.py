"""Primary-only (gold ``importance == "primary"``) text / table / definition
channel-recovery scorer.

This is the production-style implementation used to reproduce the released
primary-only channel metrics. Text, table, and definition recovery all use
gold evidence links with ``importance == "primary"``
(``hit_at_k_table_primary`` in ``evaluation/e/e_evaluation.py``, denominator
119); this module extends that same primary-only definition to the text and
definition channels, as a new standalone module -- it does not read, import,
or modify the frozen ``evaluation/e/e_evaluation.py`` evaluator.

Contract (the primary-only channel scientific contract, gold importance ==
"primary"):
  1. Channel eligibility is decided on GOLD provenance only, after filtering
     every evidence-set link to ``importance == "primary"``. This makes the
     denominator fixed and method-independent: computed once from gold data
     and shared unchanged across every prediction file.
  2. A missing/unscoreable prediction counts as a miss for an otherwise
     eligible instance -- it is never dropped from the fixed denominator.
     (Handled here simply by omission: an instance absent from
     ``pred_by_instance`` yields an empty cited-id set, which can never
     satisfy an eligible channel's hit condition.)
  3. A citation counts as a hit via an exact ``evid_id`` match OR via the
     prediction's own unit-to-sentence ``coverage_map`` -- the identical
     matching rule already used by the frozen evaluator's
     ``ModuleE_Evaluator._evidence_metrics._is_cited``, reimplemented here
     independently (read-only reference, not imported).

Reads only public, released files: the caller passes gold records built
from ``data/eagc_trace_789/gold/*.json`` and predictions built from
``predictions/table1/*.json``. No model or network call is made anywhere in
this module.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Set

CHANNELS = ("text", "table_cell", "definition")


def resolve_pool_type(prov_pool: Optional[str], evid_id: str) -> str:
    """Resolve a gold link to one of "text" / "table_cell" / "definition".

    Mirrors the frozen evaluator's ``_resolve_pool_type`` priority order
    (provenance field first, then a token heuristic on the evid_id) minus
    the task-level ``pool_membership`` fallback, which only matters for
    classifying a *predicted* evidence id's pool -- irrelevant here since
    this function classifies GOLD links only.
    """
    if prov_pool:
        if prov_pool == "table":
            return "table_cell"
        if prov_pool in ("text", "table_cell", "definition"):
            return prov_pool
    for token in evid_id.lower().split(":"):
        if token in ("sent", "sec", "para", "sentence"):
            return "text"
        if token in ("cell", "table"):
            return "table_cell"
        if token in ("def", "definition", "glossary"):
            return "definition"
    return "text"


def load_public_gold(gold_dir: str) -> List[Dict]:
    """[{"instance_id": ..., "gold_evidence_sets": [...]}] from the public
    released gold directory. Identical record shape to the one
    ``reproduce/reproduce_table1.py::load_gold`` builds for the same files."""
    gold_data = []
    for fn in sorted(os.listdir(gold_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(gold_dir, fn)) as f:
            g = json.load(f)
        pred = g.get("prediction", {}) or {}
        gold_data.append({
            "instance_id": g["canonical_claim_id"],
            "gold_evidence_sets": pred.get("evidence_sets", []) or [],
        })
    return gold_data


def primary_gold_ids_by_channel(gold_evidence_sets: List[Dict]) -> Dict[str, Set[str]]:
    """{"text": {evid_id, ...}, "table_cell": {...}, "definition": {...}} --
    the PRIMARY-only gold evidence ids for one instance, grouped by pool."""
    out: Dict[str, Set[str]] = {c: set() for c in CHANNELS}
    for gs in gold_evidence_sets:
        for link in gs.get("links", []) or []:
            if link.get("importance") != "primary":
                continue
            eid = link.get("evid_id")
            if not eid:
                continue
            prov = link.get("provenance")
            prov_pool = prov.get("pool") if isinstance(prov, dict) else None
            pool = resolve_pool_type(prov_pool, eid)
            out[pool].add(eid)
    return out


def compute_gold_eligibility(gold_data: List[Dict]) -> Dict[str, Set[str]]:
    """Fixed, method-independent per-channel eligible instance_id sets: an
    instance is eligible for a channel iff it carries >=1 PRIMARY gold link
    resolving to that channel's pool. Computed once from gold data alone,
    identical for every prediction file scored against it."""
    eligible: Dict[str, Set[str]] = {c: set() for c in CHANNELS}
    for g in gold_data:
        primary_by_channel = primary_gold_ids_by_channel(g["gold_evidence_sets"])
        for c in CHANNELS:
            if primary_by_channel[c]:
                eligible[c].add(g["instance_id"])
    return eligible


def extract_cited_ids(pred_graph: Optional[Dict]) -> Set[str]:
    """Mirrors the frozen evaluator's ``_extract_evidence_ids`` exactly."""
    ids: Set[str] = set()
    if not pred_graph:
        return ids
    for es in pred_graph.get("evidence_sets", []) or []:
        for link in es.get("links", []) or []:
            eid = link.get("evid_id")
            if eid:
                ids.add(eid)
    return ids


def build_sent_to_unit(coverage_map: Optional[Dict[str, List[str]]]) -> Dict[str, str]:
    sent_to_unit: Dict[str, str] = {}
    if coverage_map:
        for uid, sids in coverage_map.items():
            for sid in sids:
                sent_to_unit[sid] = uid
    return sent_to_unit


def is_cited(gold_id: str, pred_evid_ids: Set[str], sent_to_unit: Dict[str, str]) -> bool:
    """True if gold_id is cited exactly OR via unit coverage -- mirrors the
    frozen evaluator's ``_evidence_metrics._is_cited`` exactly."""
    if gold_id in pred_evid_ids:
        return True
    unit_id = sent_to_unit.get(gold_id)
    return unit_id is not None and unit_id in pred_evid_ids


def score_instance_primary_hits(
    gold_evidence_sets: List[Dict],
    pred_record: Optional[Dict],
) -> Dict[str, Optional[float]]:
    """Per-channel decision for ONE instance:
      None -- not eligible (instance has no primary gold link of that pool)
      1.0  -- eligible and the primary gold evidence was cited
      0.0  -- eligible and it was not (includes a missing/None pred_record,
              which yields an empty cited-id set and therefore a miss on
              every otherwise-eligible channel, per contract point 2)."""
    primary_by_channel = primary_gold_ids_by_channel(gold_evidence_sets)

    pred_graph = None
    coverage_map = None
    if pred_record is not None:
        pred_graph = pred_record.get("prediction")
        coverage_map = (pred_record.get("meta") or {}).get("coverage_map")
    pred_evid_ids = extract_cited_ids(pred_graph)
    sent_to_unit = build_sent_to_unit(coverage_map)

    out: Dict[str, Optional[float]] = {}
    for c in CHANNELS:
        ids = primary_by_channel[c]
        if not ids:
            out[c] = None
            continue
        out[c] = 1.0 if any(is_cited(gid, pred_evid_ids, sent_to_unit) for gid in ids) else 0.0
    return out


def score_all(gold_data: List[Dict], pred_by_instance: Dict[str, Dict]) -> Dict:
    """pred_by_instance: {instance_id: raw "ok"-status prediction record}.
    An instance absent from this dict is treated as missing/unscoreable.

    Returns {"eligibility": {channel: {instance_id,...}},
             "per_instance": {instance_id: {channel: 1.0/0.0/None}},
             "summary": {channel: {"eligible_n", "hit_n", "pct"}}}."""
    eligibility = compute_gold_eligibility(gold_data)
    per_instance: Dict[str, Dict[str, Optional[float]]] = {}
    for g in gold_data:
        iid = g["instance_id"]
        rec = pred_by_instance.get(iid)
        per_instance[iid] = score_instance_primary_hits(g["gold_evidence_sets"], rec)

    summary = {}
    for c in CHANNELS:
        elig_ids = eligibility[c]
        hit_n = sum(1 for iid in elig_ids if per_instance[iid][c] == 1.0)
        summary[c] = {
            "eligible_n": len(elig_ids),
            "hit_n": hit_n,
            "pct": round(100.0 * hit_n / len(elig_ids), 2) if elig_ids else None,
        }
    return {"eligibility": eligibility, "per_instance": per_instance, "summary": summary}
