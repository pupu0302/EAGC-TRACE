# baselines/b1_reachable_set.py

"""
B1: reachable_set (candidate units -> reachable atomic evid_id)

Produces two outputs:
(a) reachable_set: Set[evid_id]
(b) provenance_map[evid_id] = {pool, from_candidate_unit_id, retrieval_rank}
Used to enforce the closed-candidate constraint and to populate
EvidenceLink provenance.

"""



# baselines/b1_reachable_set.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Optional

@dataclass(frozen=True)
class Provenance:
    pool: str  # "text" | "definition" | "table"
    from_candidate_unit_id: str
    retrieval_rank: int  # 1-based

def _safe_list(x) -> List[Any]:
    return x if isinstance(x, list) else []

def build_reachable_set(instance: Dict[str, Any]) -> Tuple[Set[str], Dict[str, Provenance]]:
    """
    Closed-candidate reachable set builder:
      - text_units / definition_units: covered_evidence_ids (atomic sentence evid_id)
      - table_units: recommended_cells[*].evid_id (atomic cell evid_id)
    """
    cand = instance.get("candidate_units", {}) or {}
    reachable: Set[str] = set()
    prov_map: Dict[str, Provenance] = {}

    def _add(pool: str, evid_id: str, unit_id: str, rank: int):
        if not evid_id:
            return
        reachable.add(evid_id)
        prev = prov_map.get(evid_id)
        cur = Provenance(pool=pool, from_candidate_unit_id=unit_id, retrieval_rank=rank)
        if prev is None:
            prov_map[evid_id] = cur
        else:
            # deterministic: keep lower rank; tie-break by unit_id then pool
            if (cur.retrieval_rank, cur.from_candidate_unit_id, cur.pool) < (prev.retrieval_rank, prev.from_candidate_unit_id, prev.pool):
                prov_map[evid_id] = cur

    # text pool
    for rank0, u in enumerate(_safe_list(cand.get("text_units"))):
        unit_id = str(u.get("unit_id", ""))
        for evid_id in _safe_list(u.get("covered_evidence_ids")):
            _add("text", str(evid_id), unit_id, rank0 + 1)

    # definition pool
    for rank0, u in enumerate(_safe_list(cand.get("definition_units"))):
        unit_id = str(u.get("unit_id", ""))
        for evid_id in _safe_list(u.get("covered_evidence_ids")):
            _add("definition", str(evid_id), unit_id, rank0 + 1)

    # table pool (optional)
    for rank0, u in enumerate(_safe_list(cand.get("table_units"))):
        unit_id = str(u.get("unit_id", ""))
        for cell in _safe_list(u.get("recommended_cells")):
            evid_id = str((cell or {}).get("evid_id", ""))
            _add("table", evid_id, unit_id, rank0 + 1)

    return reachable, prov_map
