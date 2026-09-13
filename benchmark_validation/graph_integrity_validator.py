"""
Structural and provenance integrity checks for EAGC GraphPrediction
instances: node/edge reference validity, duplicate detection, and
evidence-link provenance consistency (evid_id format, pool/relation
alignment, manual-add flagging).
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

SENT_RE = re.compile(r"^(?P<doc>[^:]+):sent:(?P<idx>\d{6})$")
CELL_RE = re.compile(
    r"^(?P<doc>[^:]+):cell:tbl_(?P<hash>[0-9a-f]{16}):r(?P<row>\d{3})c(?P<col>\d{2})$"
)

@dataclass
class Finding:
    code: str
    message: str
    path: str = ""

@dataclass
class SampleReport:
    doc_id: str
    claim_id: str
    source_path: str
    errors: List[Finding]
    warnings: List[Finding]
    stats: Dict[str, Any]

def _parse_evid_id(evid_id: str) -> Tuple[str, str]:
    """
    Returns (kind, doc_id_from_evid) where kind in {"sent","cell","unknown"}.
    """
    m = SENT_RE.match(evid_id)
    if m:
        return ("sent", m.group("doc"))
    m = CELL_RE.match(evid_id)
    if m:
        return ("cell", m.group("doc"))
    return ("unknown", "")

def _count_node_types(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for n in nodes:
        t = n.get("node_type", "UNKNOWN")
        out[t] = out.get(t, 0) + 1
    return out

def validate_graph_prediction(
    gp: Dict[str, Any],
    source_path: str,
    strict: bool = True,
    require_canonical_claim_id: bool = True,
    require_primary: bool = True,
    require_provenance_non_manual: bool = True,
) -> SampleReport:
    errors: List[Finding] = []
    warnings: List[Finding] = []

    doc_id = gp.get("doc_id", "")
    claim_id = gp.get("claim_id", "")

    nodes = gp.get("nodes") or []
    edges = gp.get("edges") or []
    ev_sets = gp.get("evidence_sets") or []
    meta = gp.get("meta") or {}

    # Basic presence (schema should handle, but keep defensive)
    if not doc_id:
        errors.append(Finding("E_TOP_DOC_ID", "Missing or empty doc_id", "doc_id"))
    if not claim_id:
        errors.append(Finding("E_TOP_CLAIM_ID", "Missing or empty claim_id", "claim_id"))

    # E1. node_id uniqueness
    node_ids: List[str] = []
    for i, n in enumerate(nodes):
        nid = n.get("node_id")
        if not nid:
            errors.append(Finding("E_NODE_ID", "Missing node_id", f"nodes[{i}].node_id"))
            continue
        node_ids.append(nid)

    dup = {x for x in node_ids if node_ids.count(x) > 1}
    for d in sorted(dup):
        errors.append(Finding("E_NODE_DUP", f"Duplicate node_id: {d}", "nodes"))

    node_id_set = set(node_ids)

    # E2. Exactly one Claim node (recommended default)
    claim_nodes = [n for n in nodes if n.get("node_type") == "Claim"]
    if len(claim_nodes) != 1:
        errors.append(
            Finding(
                "E_CLAIM_COUNT",
                f"Expected exactly 1 Claim node, got {len(claim_nodes)}",
                "nodes",
            )
        )
        claim_node = None
    else:
        claim_node = claim_nodes[0]

    # Canonical claim id requirement (to compute OracleRecall robustly)
    if require_canonical_claim_id and claim_node is not None:
        ccid = claim_node.get("canonical_claim_id")
        if ccid is None or (isinstance(ccid, str) and ccid.strip() == ""):
            # If schema allows null, treat as error in strict mode
            (errors if strict else warnings).append(
                Finding(
                    "E_CANONICAL_CLAIM_ID" if strict else "W_CANONICAL_CLAIM_ID",
                    "Claim node missing canonical_claim_id (needed for stable Gate A/oracle recall mapping)",
                    "nodes[Claim].canonical_claim_id",
                )
            )

    # E1. Edge references
    seen_edges = set()
    for i, e in enumerate(edges):
        src, dst = e.get("src"), e.get("dst")
        et = e.get("edge_type")
        if not src or not dst or not et:
            errors.append(Finding("E_EDGE_FIELDS", "Edge missing required fields", f"edges[{i}]"))
            continue
        if src == dst:
            errors.append(Finding("E_EDGE_SELF", "Self-loop edge (src==dst) not allowed", f"edges[{i}]"))
        if src not in node_id_set:
            errors.append(Finding("E_EDGE_SRC_REF", f"Edge src not found in nodes: {src}", f"edges[{i}].src"))
        if dst not in node_id_set:
            errors.append(Finding("E_EDGE_DST_REF", f"Edge dst not found in nodes: {dst}", f"edges[{i}].dst"))
        key = (et, src, dst)
        if key in seen_edges:
            warnings.append(Finding("W_EDGE_DUP", f"Duplicate edge triple: {key}", f"edges[{i}]"))
        seen_edges.add(key)

        # Optional: light type compatibility checks
        # measures should point to Metric
        if et == "measures":
            dst_node = next((n for n in nodes if n.get("node_id") == dst), None)
            if dst_node and dst_node.get("node_type") != "Metric":
                (errors if strict else warnings).append(
                    Finding(
                        "E_MEASURES_DST" if strict else "W_MEASURES_DST",
                        f"Edge 'measures' dst should be Metric, got {dst_node.get('node_type')}",
                        f"edges[{i}]",
                    )
                )

    # Evidence checks
    primary_links = 0
    manual_links = 0
    missing_pool = 0
    definition_pool_mismatch = 0
    evid_doc_mismatch = 0
    evid_format_unknown = 0

    for si, s in enumerate(ev_sets):
        links = s.get("links") or []
        for li, link in enumerate(links):
            evid_id = link.get("evid_id", "")
            importance = link.get("importance")
            relation = link.get("relation")
            is_manual = link.get("is_manual_add", None)
            prov = link.get("provenance", None) or {}
            pool = prov.get("pool", None)

            if importance == "primary":
                primary_links += 1

            if is_manual is True:
                manual_links += 1

            # doc_id consistency
            kind, doc_from_evid = _parse_evid_id(evid_id)
            if kind == "unknown":
                evid_format_unknown += 1
                (errors if strict else warnings).append(
                    Finding(
                        "E_EVID_FORMAT" if strict else "W_EVID_FORMAT",
                        f"Unknown evid_id format: {evid_id}",
                        f"evidence_sets[{si}].links[{li}].evid_id",
                    )
                )
            else:
                if doc_id and doc_from_evid and doc_from_evid != doc_id:
                    evid_doc_mismatch += 1
                    (errors if strict else warnings).append(
                        Finding(
                            "E_EVID_DOC_MISMATCH" if strict else "W_EVID_DOC_MISMATCH",
                            f"evid_id doc prefix {doc_from_evid} != doc_id {doc_id}",
                            f"evidence_sets[{si}].links[{li}].evid_id",
                        )
                    )

            # pool requiredness / consistency
            if pool is None:
                missing_pool += 1
                warnings.append(
                    Finding(
                        "W_PROV_POOL_MISSING",
                        "provenance.pool missing (ambiguous for sent IDs; harms pool-level diagnostics)",
                        f"evidence_sets[{si}].links[{li}].provenance.pool",
                    )
                )
            else:
                # If evid_id is cell, pool should be table
                if kind == "cell" and pool not in ("table", "unknown"):
                    (errors if strict else warnings).append(
                        Finding(
                            "E_POOL_CELL_MISMATCH" if strict else "W_POOL_CELL_MISMATCH",
                            f"cell evid_id should have pool=table, got pool={pool}",
                            f"evidence_sets[{si}].links[{li}]",
                        )
                    )

                # If relation indicates definition, pool should be definition
                if relation == "definition_support" and pool != "definition":
                    definition_pool_mismatch += 1
                    warnings.append(
                        Finding(
                            "W_DEF_POOL_MISMATCH",
                            f"relation=definition_support but provenance.pool={pool} (definition/text share sent IDs; require pool=definition for interpretability)",
                            f"evidence_sets[{si}].links[{li}]",
                        )
                    )

            # require provenance for non-manual links (strictly recommended)
            if require_provenance_non_manual and is_manual is False:
                if not prov:
                    (errors if strict else warnings).append(
                        Finding(
                            "E_PROV_MISSING" if strict else "W_PROV_MISSING",
                            "Non-manual evidence link missing provenance (pool/rank/unit_id).",
                            f"evidence_sets[{si}].links[{li}].provenance",
                        )
                    )
                else:
                    # rank should exist in most cases
                    if prov.get("retrieval_rank", None) in (None, 0):
                        warnings.append(
                            Finding(
                                "W_RANK_MISSING",
                                "Non-manual link missing retrieval_rank (diagnostics weakened).",
                                f"evidence_sets[{si}].links[{li}].provenance.retrieval_rank",
                            )
                        )

            # is_manual_add missing -> error in strict mode (Gate B consistency critical)
            if is_manual is None:
                (errors if strict else warnings).append(
                    Finding(
                        "E_IS_MANUAL_MISSING" if strict else "W_IS_MANUAL_MISSING",
                        "is_manual_add missing; required for Gate B (manual_add_rate) consistency. Must be explicitly set to true (manual) or false (from candidates).",
                        f"evidence_sets[{si}].links[{li}].is_manual_add",
                    )
                )

    if require_primary and primary_links == 0:
        (errors if strict else warnings).append(
            Finding(
                "E_NO_PRIMARY" if strict else "W_NO_PRIMARY",
                "No primary evidence links found (Gate A/oracle recall becomes undefined).",
                "evidence_sets",
            )
        )

    # Soft drift warnings
    if primary_links > 3:
        warnings.append(
            Finding(
                "W_PRIMARY_TOO_MANY",
                f"Primary evidence links unusually high: {primary_links} (risk: primary semantics drift; consider minimal sufficient set guidance).",
                "evidence_sets",
            )
        )

    stats = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "evidence_set_count": len(ev_sets),
        "primary_link_count": primary_links,
        "manual_link_count": manual_links,
        "node_type_counts": _count_node_types(nodes),
        "missing_pool_count": missing_pool,
        "definition_pool_mismatch_count": definition_pool_mismatch,
        "evid_doc_mismatch_count": evid_doc_mismatch,
        "evid_format_unknown_count": evid_format_unknown,
        "meta_keys": sorted(list(meta.keys())) if isinstance(meta, dict) else [],
    }

    return SampleReport(
        doc_id=str(doc_id),
        claim_id=str(claim_id),
        source_path=source_path,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )
