# baselines/b2_rulegraph.py

"""
2) B2: RuleGraph baseline (minimal GraphPrediction, sufficient to run the
scoring pipeline end to end)

Design principles:
 - Minimal nodes: always emit one Claim node (all others are optional).
 - Minimal evidence_sets: at least 1 set with at least 1 primary evidence
   link (otherwise ECS C1 fails outright).
 - relation: defaults to qualitative_support; when numeric/definition
   content is detected, uses numeric_support / definition_support to try
   to satisfy ECS C2/C3.
 - closed-candidate: the primary evid must be in reachable_set.


"""

# baselines/b2_rulegraph.py
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

from baselines.b1_reachable_set import build_reachable_set, Provenance

NUM_REGEX = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)")
DEF_KWS = ("methodology", "scope", "boundary", "assurance", "standard", "sasb", "gri", "tcfd", "issb")

def _norm_num(s: str) -> Tuple[Optional[float], Optional[str]]:
    m = NUM_REGEX.search(s or "")
    if not m:
        return None, None
    tok = m.group(0)
    unit = "%" if tok.endswith("%") else None
    tok2 = tok.replace(",", "").replace("%", "")
    try:
        return float(tok2), unit
    except Exception:
        return None, unit

def _claim_flags(claim_text: str) -> Tuple[bool, bool]:
    t = (claim_text or "").lower()
    numeric_needed = bool(NUM_REGEX.search(t))
    definition_needed = any(kw in t for kw in DEF_KWS)
    return numeric_needed, definition_needed

def _pick_best_evidence(
    reachable: set,
    prov_map: Dict[str, Provenance],
    prefer_pools: List[str],
) -> Optional[str]:
    if not reachable:
        return None
    pool_rank = {p: i for i, p in enumerate(prefer_pools)}

    def key(eid: str):
        p = prov_map.get(eid)
        pool = p.pool if p else "unknown"
        r = p.retrieval_rank if p else 10**9
        return (pool_rank.get(pool, 10**6), r, eid)

    return sorted(list(reachable), key=key)[0]

def _prov_obj(prov: Optional[Provenance]) -> Optional[Dict[str, Any]]:
    if prov is None:
        return None
    return {
        "from_candidate_unit_id": prov.from_candidate_unit_id,
        "retrieval_rank": prov.retrieval_rank,
        "pool": prov.pool,
    }

def predict_rulegraph(instance: Dict[str, Any], run_meta: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = instance["doc_id"]
    claim_id = instance["claim_id"]
    claim_text = instance.get("claim_text", "") or ""
    claim_span_evid_id = instance.get("claim_span_evid_id")  # input field (task spec)

    reachable, prov_map = build_reachable_set(instance)
    numeric_needed, definition_needed = _claim_flags(claim_text)

    warnings: List[str] = []
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # ---- Claim node (schema: requires text + attrs) ----
    claim_node_id = f"{claim_id}:n:claim"
    primary_type = "D" if definition_needed else ("O" if numeric_needed else None)  # conservative
    nodes.append({
        "node_id": claim_node_id,
        "node_type": "Claim",
        "text": claim_text,
        "source_span_evid_id": claim_span_evid_id,
        "attrs": {},  # Claim attrs allows additionalProperties
        "primary_type": primary_type,
        "secondary_type": None,
        "numeric_needed": numeric_needed,
        "definition_needed": definition_needed,
        "table_likely": None,
        "claim_mention_id": None,
        "canonical_claim_id": None,
        "topic": None,
    })

    # ---- Optional Outcome node (numeric grounding) ----
    outcome_node_id = None
    if numeric_needed:
        val, unit = _norm_num(claim_text)
        outcome_node_id = f"{claim_id}:n:outcome:0"
        nodes.append({
            "node_id": outcome_node_id,
            "node_type": "Outcome",
            "text": None,
            "source_span_evid_id": None,
            "attrs": {
                "year": None,
                "value": val,
                "value_type": "unknown",
                "unit": unit,
                "baseline_year": None,
                "polarity": "unknown",
            },
            "primary_type": None,
            "secondary_type": None,
            "numeric_needed": None,
            "definition_needed": None,
            "table_likely": None,
            "claim_mention_id": None,
            "canonical_claim_id": None,
            "topic": None,
        })
        edges.append({
            "edge_type": "associated_with",
            "src": claim_node_id,
            "dst": outcome_node_id,
        })

    # ---- Optional Definition + Metric nodes (definition grounding) ----
    metric_node_id = None
    definition_node_id = None
    if definition_needed:
        metric_node_id = f"{claim_id}:n:metric:0"
        nodes.append({
            "node_id": metric_node_id,
            "node_type": "Metric",
            "text": None,
            "source_span_evid_id": None,
            "attrs": {
                "metric_name": "unspecified_metric",  # REQUIRED by schema
                "scope": None,
                "unit": None,
                "denominator": None,
            },
            "primary_type": None,
            "secondary_type": None,
            "numeric_needed": None,
            "definition_needed": None,
            "table_likely": None,
            "claim_mention_id": None,
            "canonical_claim_id": None,
            "topic": None,
        })
        edges.append({
            "edge_type": "associated_with",
            "src": claim_node_id,
            "dst": metric_node_id,
        })

        definition_node_id = f"{claim_id}:n:def:0"
        nodes.append({
            "node_id": definition_node_id,
            "node_type": "Definition",
            "text": None,
            "source_span_evid_id": None,
            "attrs": {
                "definition_type": "methodology",
                "standard": None,
                "scope_covered": [],
                "notes": None,
            },
            "primary_type": None,
            "secondary_type": None,
            "numeric_needed": None,
            "definition_needed": None,
            "table_likely": None,
            "claim_mention_id": None,
            "canonical_claim_id": None,
            "topic": None,
        })
        edges.append({
            "edge_type": "defines",
            "src": definition_node_id,
            "dst": metric_node_id if metric_node_id else (outcome_node_id or claim_node_id),
        })

    # ---- Evidence links (schema: evidence_sets[].links minItems=1) ----
    # Choose evid for numeric_support (prefer text/table), definition_support (prefer definition)
    e_num = _pick_best_evidence(reachable, prov_map, prefer_pools=["text", "table", "definition"]) if numeric_needed else None
    e_def = _pick_best_evidence(reachable, prov_map, prefer_pools=["definition", "text", "table"]) if definition_needed else None

    links: List[Dict[str, Any]] = []

    def _add_link(evid_id: str, relation: str, importance: str, is_manual_add: bool):
        prov = prov_map.get(evid_id)
        link = {
            "evid_id": evid_id,
            "relation": relation,
            "importance": importance,
        }
        if is_manual_add:
            link["is_manual_add"] = True
            link["provenance"] = {"from_candidate_unit_id": None, "retrieval_rank": None, "pool": "unknown"}
        else:
            p = _prov_obj(prov)
            if p is not None:
                link["provenance"] = p
        links.append(link)

    if reachable:
        # Always ensure at least one primary anchor
        primary = e_num or e_def or _pick_best_evidence(reachable, prov_map, prefer_pools=["text", "definition", "table"])
        if primary is None:
            warnings.append("NO_PRIMARY_EVID_IN_REACHABLE")
        else:
            # base primary qualitative anchor
            _add_link(primary, "qualitative_support", "primary", is_manual_add=False)

        # numeric-needed: ensure a numeric_support primary exists (ECS C2) — can reuse same evid_id
        if numeric_needed:
            _add_link(e_num or primary, "numeric_support", "primary", is_manual_add=False)

        # definition-needed: ensure definition_support primary exists + defines edge already added (ECS C3)
        if definition_needed:
            _add_link(e_def or primary, "definition_support", "primary", is_manual_add=False)

    else:
        # schema requires >=1 link: fallback to claim span evid as manual add
        fallback = claim_span_evid_id or f"{doc_id}:sent:UNKNOWN"
        warnings.append("EMPTY_REACHABLE_SET_FALLBACK_TO_MANUAL_ADD")
        _add_link(fallback, "background", "primary", is_manual_add=True)

    pred = {
        "doc_id": doc_id,
        "claim_id": claim_id,
        "nodes": nodes,
        "edges": edges,
        "evidence_sets": [{
            "set_id": "S0",
            "links": links,
        }],
        "meta": {
            "model": run_meta.get("model", "b2_rulegraph_v0"),
            "seed": int(run_meta.get("seed", 0)),
            **{k: v for k, v in (run_meta or {}).items() if k not in ("model", "seed")},
        }
    }
    if warnings:
        pred["warnings"] = warnings
    return pred
