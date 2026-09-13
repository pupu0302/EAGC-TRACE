"""Independent second implementation of primary-only channel scoring.

Written from scratch against the same public gold/prediction file shapes
and the same contract as ``primary_channel_scorer.py`` (the primary-only
channel scientific contract, gold importance == "primary"), but
deliberately structured differently so the two implementations can
disagree if either has a transcription bug:

  - gold is flattened into per-instance ``(evid_id, importance, pool)``
    tuples up front, instead of per-channel sets built while looping;
  - pool resolution dispatches through lookup tables instead of an
    if/elif chain;
  - eligibility is derived from the flattened tuples with a set
    comprehension instead of accumulated inside the gold-loading loop;
  - citation matching is still the exact-id-or-covered-unit rule (the
    frozen evaluator's own definition), but built via a dict comprehension
    rather than the
    other module's nested for-loop.

``reproduce/reproduce_primary_channel_metrics.py`` runs both modules on
every prediction file and requires their outputs to match on all 789
instances x 3 channels before accepting any aggregate.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Set, Tuple

_PROV_POOL_TABLE = {"text": "text", "table_cell": "table_cell", "table": "table_cell", "definition": "definition"}
_TOKEN_POOL_TABLE = {
    "sent": "text", "sec": "text", "para": "text", "sentence": "text",
    "cell": "table_cell", "table": "table_cell",
    "def": "definition", "definition": "definition", "glossary": "definition",
}
CHANNELS = ("text", "table_cell", "definition")


def _pool_of(evid_id: str, prov: Optional[dict]) -> str:
    prov_pool = prov.get("pool") if isinstance(prov, dict) else None
    if prov_pool in _PROV_POOL_TABLE:
        return _PROV_POOL_TABLE[prov_pool]
    for token in evid_id.lower().split(":"):
        if token in _TOKEN_POOL_TABLE:
            return _TOKEN_POOL_TABLE[token]
    return "text"


def load_public_gold_flat(gold_dir: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """{instance_id: [(evid_id, importance, pool), ...]} flattened across
    every gold_evidence_sets/links entry for that instance."""
    out: Dict[str, List[Tuple[str, str, str]]] = {}
    for fn in sorted(os.listdir(gold_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(gold_dir, fn)) as f:
            g = json.load(f)
        iid = g["canonical_claim_id"]
        flat: List[Tuple[str, str, str]] = []
        for gs in (g.get("prediction") or {}).get("evidence_sets", []) or []:
            for link in gs.get("links", []) or []:
                eid = link.get("evid_id")
                if not eid:
                    continue
                flat.append((eid, link.get("importance"), _pool_of(eid, link.get("provenance"))))
        out[iid] = flat
    return out


def channel_eligibility(flat_gold: Dict[str, List[Tuple[str, str, str]]]) -> Dict[str, Set[str]]:
    elig: Dict[str, Set[str]] = {c: set() for c in CHANNELS}
    for iid, links in flat_gold.items():
        pools_with_primary = {pool for (_eid, imp, pool) in links if imp == "primary"}
        for pool in pools_with_primary:
            elig[pool].add(iid)
    return elig


def _sentence_to_unit(coverage_map: Optional[Dict[str, List[str]]]) -> Dict[str, str]:
    return {sid: uid for uid, sids in (coverage_map or {}).items() for sid in sids}


def _cited_ids(pred_record: Optional[dict]) -> Set[str]:
    pred_graph = (pred_record or {}).get("prediction") or {}
    return {
        link["evid_id"]
        for es in pred_graph.get("evidence_sets", []) or []
        for link in es.get("links", []) or []
        if link.get("evid_id")
    }


def score_all(flat_gold: Dict[str, List[Tuple[str, str, str]]], pred_by_instance: Dict[str, dict]) -> Dict:
    eligibility = channel_eligibility(flat_gold)
    per_instance: Dict[str, Dict[str, Optional[float]]] = {}

    for iid, links in flat_gold.items():
        rec = pred_by_instance.get(iid)
        cited = _cited_ids(rec)
        coverage_map = ((rec or {}).get("meta") or {}).get("coverage_map")
        sent_to_unit = _sentence_to_unit(coverage_map)

        row: Dict[str, Optional[float]] = {}
        for pool in CHANNELS:
            primary_ids = {eid for (eid, imp, p) in links if imp == "primary" and p == pool}
            if not primary_ids:
                row[pool] = None
                continue
            hit = any((eid in cited) or (sent_to_unit.get(eid) in cited) for eid in primary_ids)
            row[pool] = 1.0 if hit else 0.0
        per_instance[iid] = row

    summary = {}
    for pool in CHANNELS:
        ids = eligibility[pool]
        hit_n = sum(1 for iid in ids if per_instance[iid][pool] == 1.0)
        summary[pool] = {
            "eligible_n": len(ids),
            "hit_n": hit_n,
            "pct": round(100.0 * hit_n / len(ids), 2) if ids else None,
        }
    return {"eligibility": eligibility, "per_instance": per_instance, "summary": summary}
