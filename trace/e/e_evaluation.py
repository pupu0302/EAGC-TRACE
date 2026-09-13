"""
Module E: Evaluation Harness for EAGC predictions.

Computes the evaluation metric suite:
  Field-level  : Parseable, F1_role, F1_fields
  Evidence-level: Evidence_Recall, Hit@K_text, Hit@K_table, Hit@K_def
  System-level : ECS_noev, ECS_strict, Consistency

ECS_oracle requires a separate oracle-retrieval experiment and is not
computed here (it is left as "--" in the paper table until that run is done).
"""

import json
import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Set, Optional, Tuple
from collections import Counter

# ---------------------------------------------------------------------------
# Canonical ECS core: evidence-recall and closure logic lives in
# evaluation/scoring/ecs_metric.py, imported here as a bare
# top-level module via a self-contained sys.path addition (this file's own
# directory chain, not a caller-established path), so this import resolves
# both when this module is imported by a runner and when it is executed
# directly (`python trace/e/e_evaluation.py`, the self-test below).
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../trace/e
_TRACE_DIR = os.path.dirname(_THIS_DIR)                          # .../trace
_REPO_ROOT = os.path.dirname(_TRACE_DIR)                         # repo root
_ECS_SCORING_DIR = os.path.join(_REPO_ROOT, "evaluation", "scoring")
if _ECS_SCORING_DIR not in sys.path:
    sys.path.insert(0, _ECS_SCORING_DIR)
import ecs_metric  # noqa: E402

logger = logging.getLogger(__name__)


# =============================================================================
# Gold Annotation Format Adapters
# =============================================================================

def adapt_gold_annotation(raw: Dict) -> Dict:
    """
    Convert a raw M4 gold annotation file (batch_runner output) to the
    reference format expected by ModuleE_Evaluator.run(gold_data, ...).

    Raw format (dryrun gpt4o files):
        {task_id, canonical_claim_id, prediction: {nodes, edges, evidence_sets}, validation: {after_repair: {...}}}

    Output format (gold reference):
        {instance_id, nodes, edges, gold_evidence_sets}
    """
    pred = raw.get("prediction", {})
    return {
        "instance_id": raw.get("canonical_claim_id", raw.get("task_id", "")),
        "nodes": pred.get("nodes", []),
        "edges": pred.get("edges", []),
        "gold_evidence_sets": pred.get("evidence_sets", []),
    }


def adapt_gold_as_prediction(raw: Dict) -> Dict:
    """
    Convert a raw M4 gold annotation file to the prediction format expected
    by ModuleE_Evaluator.run(..., pred_data).  Used for gold-self evaluation
    (ECS_strict(gold→gold) ≈ 1.0).

    Output format (prediction):
        {instance_id, prediction, validator_report}
    """
    val = raw.get("validation", {}).get("after_repair", {})
    validator_report = {
        "v0_gate": {"pass": val.get("v0_pass", False), "errors": []},
        "v1_gate": {"pass": val.get("v1_pass", False), "errors": []},
        "v2_gate": {"pass": val.get("v2_pass", False), "errors": []},
    }
    return {
        "instance_id": raw.get("canonical_claim_id", raw.get("task_id", "")),
        "prediction": raw.get("prediction", {}),
        "validator_report": validator_report,
        "meta": raw.get("llm_cost", {}),
    }


def load_gold_dir(dir_path: str) -> List[Dict]:
    """
    Load all gold annotation JSON files from a directory (skips batch_summary.json).
    Returns a list of raw dicts (not yet adapted).
    """
    items = []
    for fname in sorted(os.listdir(dir_path)):
        if fname == "batch_summary.json" or not fname.endswith(".json"):
            continue
        fpath = os.path.join(dir_path, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            items.append(json.load(f))
    return items


# =============================================================================
# Module E: Evaluation Harness
# =============================================================================

class ModuleE_Evaluator:
    """
    Computes evaluation metrics for a set of gold/pred pairs.

    Input formats
    -------------
    gold_data : List[Dict], each item:
        {instance_id, nodes, edges, gold_evidence_sets}

    pred_data : List[Dict], each item:
        {
          instance_id,
          prediction: {nodes, edges, evidence_sets},
          validator_report: {v0_gate, v1_gate, v2_gate},
          pool_membership: {evid_id → "text"|"table_cell"|"definition"},  # optional
          meta: {...}
        }
    """

    # Numeric attributes: match within 1% relative error OR 0.1 absolute.
    _NUMERIC_ATTRS = {"value", "baseline_year", "target_year", "year"}
    # Node types to skip when computing F1_role (always present / trivial).
    _SKIP_TYPES = {"Claim"}

    def __init__(self):
        self.error_taxonomy: Counter = Counter()
        self.efficiency_stats: Dict[str, list] = {
            "latency": [], "tokens": [], "repair_iters": []
        }

    def run(
        self,
        gold_data: List[Dict],
        pred_data: List[Dict],
        oracle_candidates_by_claim: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[Dict, pd.DataFrame]:
        """Execute full evaluation; returns (report_dict, per-instance DataFrame).

        oracle_candidates_by_claim: maps claim_id → list of all covered_evidence_ids
        from the candidate pool (text+table+def).  When provided, ECS_oracle is
        computed as ECS_strict under hypothetical perfect retrieval.
        """
        gold_map = {item["instance_id"]: item for item in gold_data}
        pred_map = {item["instance_id"]: item for item in pred_data}

        logger.info(f"Starting evaluation: {len(gold_map)} gold, {len(pred_map)} preds")

        rows = []
        for inst_id, gold_inst in gold_map.items():
            pred_inst = pred_map.get(inst_id)
            oracle_evid_ids: Optional[Set[str]] = None
            if oracle_candidates_by_claim is not None:
                oracle_evid_ids = set(oracle_candidates_by_claim.get(inst_id, []))
            row = self._evaluate_instance(inst_id, gold_inst, pred_inst, oracle_evid_ids)
            rows.append(row)
            if pred_inst:
                self._analyze_errors(pred_inst)
                self._analyze_efficiency(pred_inst)

        df = pd.DataFrame(rows)
        report = self._finalize_report(df)
        return report, df

    # ------------------------------------------------------------------
    # Per-instance evaluation
    # ------------------------------------------------------------------

    def _evaluate_instance(
        self,
        inst_id: str,
        gold: Dict,
        pred: Optional[Dict],
        oracle_evid_ids: Optional[Set[str]] = None,
    ) -> Dict:
        if not pred:
            logger.debug(f"[{inst_id}] No prediction — all metrics = 0")
            return self._zero_row(inst_id)

        pred_graph = pred.get("prediction", {})
        gold_nodes = gold.get("nodes", [])
        pred_nodes = pred_graph.get("nodes", [])
        pool_membership: Dict[str, str] = pred.get("pool_membership", {})

        res: Dict[str, Any] = {"instance_id": inst_id}

        # 1. Parseable: prediction produced at least one node
        res["parse_ok"] = 1.0 if pred_nodes else 0.0

        # 2. F1_role: node-type set micro-F1 (excluding Claim)
        res["f1_role"] = self._f1_role(gold_nodes, pred_nodes)

        # 3. F1_fields: attribute-value micro-F1 across matched node types
        res["f1_fields"] = self._f1_fields(gold_nodes, pred_nodes)

        # 4. Evidence metrics
        pred_evid_ids = self._extract_evidence_ids(pred_graph)
        gold_sets = gold.get("gold_evidence_sets", [])
        coverage_map: Dict[str, List[str]] = pred.get("meta", {}).get("coverage_map") or {}
        ev = self._evidence_metrics(
            gold_sets, pred_evid_ids, pool_membership, coverage_map, oracle_evid_ids)

        best_recall = ev["best_recall"]
        res["evidence_recall"]       = best_recall
        res["hit_at_k_text"]         = ev["hit_at_k_text"]
        res["hit_at_k_table"]        = ev["hit_at_k_table"]          # ANY-HIT
        res["hit_at_k_table_primary"] = ev["hit_at_k_table_primary"]  # PRIMARY only
        res["hit_at_k_def"]          = ev["hit_at_k_def"]
        res["gold_primary_pool"]     = ev["gold_primary_pool"]

        # 5. ECS
        report_v = pred.get("validator_report", {})
        v0_pass = report_v.get("v0_gate", {}).get("pass") is True
        v1_pass = report_v.get("v1_gate", {}).get("pass") is True
        v2_pass = report_v.get("v2_gate", {}).get("pass") is True
        v_passed = v0_pass and v1_pass and v2_pass

        res["ecs_noev"]   = ecs_metric.no_evidence_closure(v_passed)
        res["ecs_strict"] = ecs_metric.strict_closure(v_passed, best_recall)

        # ECS_oracle: ECS_strict under hypothetical perfect retrieval
        oracle_recall = ev.get("oracle_recall")
        res["ecs_oracle"] = ecs_metric.oracle_closure(v_passed, oracle_recall)

        # Failure reason for ECS_strict=0 instances
        if res["parse_ok"] == 0.0:
            res["failure_reason"] = "parse_fail"
        elif res["ecs_strict"] == 1.0:
            res["failure_reason"] = "pass"
        elif best_recall < 0.99 and not v_passed:
            res["failure_reason"] = "both_fail"
        elif best_recall < 0.99:
            res["failure_reason"] = "evidence_miss"
        else:
            res["failure_reason"] = "closure_fail"

        # 6. Consistency: macro-F1 of V1 + V2 per instance
        #    (for G baseline V1/V2 are not actually computed → consistency = 1.0,
        #     indicating "unchecked"; drops to real value once validators are wired in)
        res["consistency"] = (float(v1_pass) + float(v2_pass)) / 2.0

        # Legacy alias kept for backward compatibility
        res["field_value_match"] = res["f1_fields"]

        # 7. Table 3 diagnostics (graph-specific structural quality)
        gold_edges = gold.get("edges", [])
        pred_edges = pred_graph.get("edges", [])
        diag = self._compute_diagnostics(gold_nodes, gold_edges, pred_nodes, pred_edges, gold_sets, pred_evid_ids)
        res.update(diag)

        logger.debug(
            f"[{inst_id}] parse={res['parse_ok']:.0f} "
            f"f1_role={res['f1_role']:.3f} f1_fields={res['f1_fields']:.3f} "
            f"ev_recall={best_recall:.3f} "
            f"ecs_noev={res['ecs_noev']:.1f} ecs_strict={res['ecs_strict']:.1f}"
        )
        return res

    @staticmethod
    def _zero_row(inst_id: str) -> Dict:
        return {
            "instance_id":   inst_id,
            "parse_ok":      0.0,
            "f1_role":       0.0,
            "f1_fields":     0.0,
            "field_value_match": 0.0,
            "evidence_recall": 0.0,
            "hit_at_k_text":  None,
            "hit_at_k_table": None,
            "hit_at_k_table_primary": None,
            "hit_at_k_def":   None,
            "gold_primary_pool": "none",
            "ecs_noev":      0.0,
            "ecs_strict":    0.0,
            "ecs_oracle":    None,
            "consistency":   0.0,
            "failure_reason": "no_pred",
            # Table 3 diagnostics
            "f1_measures":        None,
            "f1_defines":         None,
            "cov_defines_to_D":   None,
            "to_align":           0.0,
            "gold_to_align":      0.0,
            "rate_temporal_valid":None,
            "miss_D":             0.0,
            "miss_primary_evid":  0.0,
        }

    # ------------------------------------------------------------------
    # F1_role
    # ------------------------------------------------------------------

    def _f1_role(self, gold_nodes: List, pred_nodes: List) -> float:
        """
        Micro-F1 over node-type multisets (excluding Claim).
        TP = min(gold_count, pred_count) per type; FP/FN from the remainder.
        """
        g_types = Counter(
            self._node_type(n) for n in gold_nodes
            if self._node_type(n) and self._node_type(n) not in self._SKIP_TYPES
        )
        p_types = Counter(
            self._node_type(n) for n in pred_nodes
            if self._node_type(n) and self._node_type(n) not in self._SKIP_TYPES
        )

        if not g_types and not p_types:
            return 1.0
        if not g_types or not p_types:
            return 0.0

        tp = sum(min(g_types[t], p_types[t]) for t in g_types)
        fp = sum(p_types.values()) - sum(min(g_types[t], p_types[t]) for t in p_types)
        fn = sum(g_types.values()) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ------------------------------------------------------------------
    # F1_fields
    # ------------------------------------------------------------------

    def _f1_fields(self, gold_nodes: List, pred_nodes: List) -> float:
        """
        Attribute-value micro-F1 across node types present in gold.

        For each non-Claim node type in gold, find the matching pred node
        (first occurrence), then compare all attrs key-by-key.
        TP: attr key present in both with matching value.
        FN: attr key present in gold but missing/wrong in pred.
        FP: attr key present in pred but missing/wrong in gold (or extra node type).
        """
        gold_by_type = self._nodes_by_type(gold_nodes, skip=self._SKIP_TYPES)
        pred_by_type = self._nodes_by_type(pred_nodes, skip=self._SKIP_TYPES)

        tp = fn = fp = 0

        for nt, g_node in gold_by_type.items():
            g_attrs = g_node.get("attrs", {}) or {}
            p_node  = pred_by_type.get(nt)

            if p_node is None:
                # All gold attrs are FN
                fn += sum(1 for v in g_attrs.values() if v is not None)
                continue

            p_attrs = p_node.get("attrs", {}) or {}
            all_keys = set(g_attrs) | set(p_attrs)

            for k in all_keys:
                gv = g_attrs.get(k)
                pv = p_attrs.get(k)
                if gv is None and pv is None:
                    continue
                if gv is None:        # pred has extra attr → FP
                    if pv is not None:
                        fp += 1
                    continue
                if pv is None:        # gold has attr, pred missing → FN
                    fn += 1
                    continue
                # Both present
                if self._attr_match(gv, pv, k in self._NUMERIC_ATTRS):
                    tp += 1
                else:
                    fp += 1
                    fn += 1

        # Extra pred node types not in gold → FP
        for nt, p_node in pred_by_type.items():
            if nt not in gold_by_type:
                p_attrs = p_node.get("attrs", {}) or {}
                fp += sum(1 for v in p_attrs.values() if v is not None)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ------------------------------------------------------------------
    # Evidence metrics
    # ------------------------------------------------------------------

    def _evidence_metrics(
        self,
        gold_sets: List[Dict],
        pred_evid_ids: Set[str],
        pool_membership: Dict[str, str],
        coverage_map: Optional[Dict[str, List[str]]] = None,
        oracle_evid_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Returns a dict with evidence metrics:
          best_recall           — max over gold sets of fraction cited
          hit_at_k_text         — 1.0 if any gold TEXT evidence cited (ANY-HIT)
          hit_at_k_table        — 1.0 if any gold TABLE evidence cited (ANY-HIT)
          hit_at_k_table_primary — 1.0 if any PRIMARY gold TABLE evidence cited
          hit_at_k_def          — 1.0 if any gold DEF evidence cited (ANY-HIT)
          oracle_recall         — recall under oracle candidates (None if not provided)
          gold_primary_pool     — most common pool type among primary gold links

        hit_at_k_* is None when no gold evidence of that pool type exists in
        this instance (so the metric is N/A and excluded from the mean).

        coverage_map: optional {unit_id → [sentence_ids]} mapping.  When the
        LLM cites a paragraph-level unit_id but the gold cites a sentence_id
        within that unit, coverage matching counts it as a hit.

        oracle_evid_ids: all covered_evidence_ids from the task candidate pool.
        oracle_recall = max over gold sets of fraction whose IDs appear in oracle pool.
        """
        if not gold_sets:
            return {
                "best_recall": 1.0,
                "hit_at_k_text": 1.0,
                "hit_at_k_table": 1.0,
                "hit_at_k_table_primary": 1.0,
                "hit_at_k_def": 1.0,
                "oracle_recall": 1.0 if oracle_evid_ids is not None else None,
                "gold_primary_pool": "none",
            }

        # Canonical citation-matching primitive (evaluation/scoring/ecs_metric.py):
        # exact evid_id match, or coverage-map resolution to a cited unit_id.
        sent_to_unit: Dict[str, str] = ecs_metric.build_sentence_to_unit_map(coverage_map)

        def _is_cited(gold_id: str) -> bool:
            """True if gold_id is cited exactly OR via unit coverage."""
            return ecs_metric.is_cited(gold_id, pred_evid_ids, sent_to_unit)

        gold_by_pool: Dict[str, Set[str]] = {"text": set(), "table_cell": set(), "definition": set()}
        gold_table_primary_ids: Set[str] = set()
        primary_pool_counts: Counter = Counter()

        for gs in gold_sets:
            for link in gs.get("links", []):
                eid = link.get("evid_id")
                if not eid:
                    continue
                ptype = self._resolve_pool_type(
                    eid,
                    prov_pool=link.get("provenance", {}).get("pool"),
                    pool_membership=pool_membership,
                )
                gold_by_pool[ptype].add(eid)
                if link.get("importance") == "primary":
                    primary_pool_counts[ptype] += 1
                    if ptype == "table_cell":
                        gold_table_primary_ids.add(eid)

        # Canonical Definition-A recall/oracle-recall (max over gold sets, all
        # importance levels included; see evaluation/scoring/ecs_metric.py).
        best_recall = ecs_metric.best_recall_over_gold_sets(gold_sets, pred_evid_ids, sent_to_unit)
        best_oracle_recall = ecs_metric.oracle_recall_over_gold_sets(gold_sets, oracle_evid_ids)

        def _hit_any(pool_key: str) -> Optional[float]:
            ids = gold_by_pool[pool_key]
            if not ids:
                return None
            return 1.0 if any(_is_cited(gid) for gid in ids) else 0.0

        def _hit_table_primary() -> Optional[float]:
            if not gold_table_primary_ids:
                return None
            return 1.0 if any(_is_cited(gid) for gid in gold_table_primary_ids) else 0.0

        gold_primary_pool = (
            primary_pool_counts.most_common(1)[0][0]
            if primary_pool_counts else "none"
        )

        return {
            "best_recall": best_recall,
            "hit_at_k_text": _hit_any("text"),
            "hit_at_k_table": _hit_any("table_cell"),
            "hit_at_k_table_primary": _hit_table_primary(),
            "hit_at_k_def": _hit_any("definition"),
            "oracle_recall": best_oracle_recall,
            "gold_primary_pool": gold_primary_pool,
        }

    @staticmethod
    def _resolve_pool_type(
        evid_id: str,
        prov_pool: Optional[str],
        pool_membership: Dict[str, str],
    ) -> str:
        """Resolve pool type string to one of: text, table_cell, definition."""
        # Priority 1: provenance field from gold annotation
        if prov_pool:
            if prov_pool == "table":
                return "table_cell"
            if prov_pool in ("text", "table_cell", "definition"):
                return prov_pool

        # Priority 2: pool_membership from task JSON (passed by caller)
        if pool_membership and evid_id in pool_membership:
            pt = pool_membership[evid_id]
            if pt == "table":
                return "table_cell"
            return pt

        # Priority 3: heuristic from evid_id token
        parts = evid_id.lower().split(":")
        for p in parts:
            if p in ("sent", "sec", "para", "sentence"):
                return "text"
            if p in ("cell", "table"):
                return "table_cell"
            if p in ("def", "definition", "glossary"):
                return "definition"
        return "text"  # default

    # ------------------------------------------------------------------
    # Table 3: Graph-Specific Diagnostics
    # ------------------------------------------------------------------

    def _compute_diagnostics(
        self,
        gold_nodes: List[Dict],
        gold_edges: List[Dict],
        pred_nodes: List[Dict],
        pred_edges: List[Dict],
        gold_sets: List[Dict],
        pred_evid_ids: Set[str],
    ) -> Dict:
        """
        Compute Table 3 per-instance diagnostics.

        Returns dict with keys:
          f1_measures, f1_defines, cov_defines_to_D,
          to_align, gold_to_align, rate_temporal_valid,
          miss_D, miss_primary_evid
        """
        gold_id_to_type = {n.get("node_id", ""): self._node_type(n) for n in gold_nodes}
        pred_id_to_type = {n.get("node_id", ""): self._node_type(n) for n in pred_nodes}
        gold_by_type = self._nodes_by_type(gold_nodes, skip=set())
        pred_by_type = self._nodes_by_type(pred_nodes, skip=set())

        res: Dict = {}

        # --- F1_measures / F1_defines ---
        res["f1_measures"] = self._edge_type_f1(
            gold_edges, gold_id_to_type, pred_edges, pred_id_to_type, "measures")
        res["f1_defines"] = self._edge_type_f1(
            gold_edges, gold_id_to_type, pred_edges, pred_id_to_type, "defines")

        # --- Cov_defines→D: fraction of gold defines-edges reproduced in pred ---
        gold_defines = [e for e in gold_edges if e.get("edge_type") == "defines"]
        if not gold_defines:
            res["cov_defines_to_D"] = None  # N/A if no defines edges in gold
        else:
            # Build gold define pairs as (src_type, dst_type)
            gold_define_pairs = set()
            for e in gold_defines:
                st = gold_id_to_type.get(e.get("src", ""), "")
                dt = gold_id_to_type.get(e.get("dst", ""), "")
                if st and dt:
                    gold_define_pairs.add((st, dt))
            pred_define_pairs = set()
            for e in pred_edges:
                if e.get("edge_type") == "defines":
                    st = pred_id_to_type.get(e.get("src", ""), "")
                    dt = pred_id_to_type.get(e.get("dst", ""), "")
                    if st and dt:
                        pred_define_pairs.add((st, dt))
            if gold_define_pairs:
                hits = gold_define_pairs & pred_define_pairs
                res["cov_defines_to_D"] = len(hits) / len(gold_define_pairs)
            else:
                res["cov_defines_to_D"] = None

        # --- F1_T-O align: binary per-instance alignment label ---
        # 1 if graph has both Target and Outcome with same unit, defined direction, non-null values
        res["gold_to_align"] = self._has_to_align(gold_by_type)
        res["to_align"]      = self._has_to_align(pred_by_type)

        # --- Rate_temporal_valid: Target.target_year > Outcome.year in prediction ---
        p_target  = pred_by_type.get("Target")
        p_outcome = pred_by_type.get("Outcome")
        if p_target and p_outcome:
            try:
                ty = float(p_target.get("attrs", {}).get("target_year") or 0)
                oy = float(p_outcome.get("attrs", {}).get("year") or 0)
                res["rate_temporal_valid"] = 1.0 if (ty > 0 and oy > 0 and ty > oy) else 0.0
            except (TypeError, ValueError):
                res["rate_temporal_valid"] = None
        else:
            res["rate_temporal_valid"] = None  # N/A if no T-O pair

        # --- Miss_D: gold has Definition node but pred omits it ---
        gold_has_def = "Definition" in gold_by_type
        pred_has_def = "Definition" in pred_by_type
        res["miss_D"] = 1.0 if (gold_has_def and not pred_has_def) else 0.0

        # --- Miss_primary_evid: no primary evidence link in prediction ---
        has_primary = False
        for es in []:  # pred evidence sets are in pred_graph, accessed via pred_evid_ids
            pass
        # We detect "primary" via gold_sets: if gold has primary links, check pred cites them
        gold_primary_ids: Set[str] = set()
        for gs in gold_sets:
            for link in gs.get("links", []):
                if link.get("importance") == "primary":
                    eid = link.get("evid_id")
                    if eid:
                        gold_primary_ids.add(eid)
        if not gold_primary_ids:
            res["miss_primary_evid"] = 0.0  # no gold primary to miss
        else:
            # pred misses primary if none of the gold primary IDs are cited
            res["miss_primary_evid"] = 0.0 if (gold_primary_ids & pred_evid_ids) else 1.0

        return res

    @staticmethod
    def _has_to_align(nodes_by_type: Dict[str, Dict]) -> float:
        """
        Returns 1.0 if graph has both Target and Outcome with:
          - same unit (case-insensitive)
          - Target.direction is defined (not None/unknown)
          - Target.value and Outcome.value are non-null
        """
        t_node = nodes_by_type.get("Target")
        o_node = nodes_by_type.get("Outcome")
        if not t_node or not o_node:
            return 0.0
        t_attrs = t_node.get("attrs", {}) or {}
        o_attrs = o_node.get("attrs", {}) or {}
        t_unit = str(t_attrs.get("unit") or "").strip().lower()
        o_unit = str(o_attrs.get("unit") or "").strip().lower()
        t_dir = t_attrs.get("direction")
        t_val = t_attrs.get("value")
        o_val = o_attrs.get("value")
        unit_match = (t_unit == o_unit) and t_unit != ""
        dir_ok = t_dir and t_dir not in ("unknown", "neutral", None)
        vals_ok = t_val is not None and o_val is not None
        return 1.0 if (unit_match and dir_ok and vals_ok) else 0.0

    def _edge_type_f1(
        self,
        gold_edges: List[Dict],
        gold_id_to_type: Dict[str, str],
        pred_edges: List[Dict],
        pred_id_to_type: Dict[str, str],
        edge_type_filter: str,
    ) -> Optional[float]:
        """
        Micro-F1 for edges of edge_type_filter, matched by (src_node_type, edge_type, dst_node_type) triplets.
        Returns None if no gold edges of this type exist.
        """
        def _triplets(edges: List[Dict], id_to_type: Dict) -> Counter:
            c: Counter = Counter()
            for e in edges:
                if e.get("edge_type") != edge_type_filter:
                    continue
                st = id_to_type.get(e.get("src", ""), "")
                dt = id_to_type.get(e.get("dst", ""), "")
                if st and dt:
                    c[(st, edge_type_filter, dt)] += 1
            return c

        gold_c = _triplets(gold_edges, gold_id_to_type)
        pred_c = _triplets(pred_edges, pred_id_to_type)

        if not gold_c:
            return None  # N/A when gold has no edges of this type

        tp = sum(min(gold_c[t], pred_c[t]) for t in gold_c)
        fp = sum(pred_c.values()) - sum(min(gold_c[t], pred_c[t]) for t in pred_c)
        fn = sum(gold_c.values()) - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_type(node: Dict) -> Optional[str]:
        """Return node type, supporting both EAGC node_type and pipeline type fields."""
        return node.get("node_type") or node.get("type")

    @classmethod
    def _nodes_by_type(cls, nodes: List[Dict], skip: Set[str]) -> Dict[str, Dict]:
        """Return dict {node_type: first_node} for non-skip types."""
        result: Dict[str, Dict] = {}
        for n in nodes:
            nt = cls._node_type(n)
            if nt and nt not in skip and nt not in result:
                result[nt] = n
        return result

    @staticmethod
    def _attr_match(gold_val: Any, pred_val: Any, is_numeric: bool) -> bool:
        if is_numeric:
            try:
                gf, pf = float(gold_val), float(pred_val)
                rel_err = abs(gf - pf) / max(abs(gf), 1e-9)
                return rel_err <= 0.01 or abs(gf - pf) <= 0.1
            except (TypeError, ValueError):
                pass
        return str(gold_val).strip().lower() == str(pred_val).strip().lower()

    def _get_node(self, data: Dict, type_name: str) -> Optional[Dict]:
        """Legacy helper: find first node of given type (supports both node_type / type fields)."""
        if not data:
            return None
        for n in data.get("nodes", []):
            if self._node_type(n) == type_name:
                return n
        return None

    @staticmethod
    def _extract_evidence_ids(pred_graph: Dict) -> Set[str]:
        ids: Set[str] = set()
        if not pred_graph:
            return ids
        for es in pred_graph.get("evidence_sets", []):
            for link in es.get("links", []):
                eid = link.get("evid_id")
                if eid:
                    ids.add(eid)
        return ids

    # ------------------------------------------------------------------
    # Error & efficiency bookkeeping
    # ------------------------------------------------------------------

    def _analyze_errors(self, pred: Dict):
        report = pred.get("validator_report", {})
        for err in report.get("v0_gate", {}).get("errors", []):
            self.error_taxonomy["Hallucination" if "HALLUCINATION" in err else "Format_Error"] += 1
        for err in report.get("v1_gate", {}).get("errors", []):
            if "NUMERIC" in err:
                self.error_taxonomy["Numeric_Mismatch"] += 1
        for err in report.get("v2_gate", {}).get("errors", []):
            if "SCOPE" in err:
                self.error_taxonomy["Scope_Mismatch"] += 1
            elif "DEFINITION" in err:
                self.error_taxonomy["Definition_Missing"] += 1

    def _analyze_efficiency(self, pred: Dict):
        meta = pred.get("meta", {})
        if "latency_ms" in meta:
            self.efficiency_stats["latency"].append(meta["latency_ms"])
        if "repair_iterations" in meta:
            self.efficiency_stats["repair_iters"].append(meta["repair_iterations"])

    # ------------------------------------------------------------------
    # Report finalization
    # ------------------------------------------------------------------

    @staticmethod
    def _finalize_report(df: pd.DataFrame) -> Dict:
        """Compute per-column means from DataFrame (skipna for optional columns)."""
        def col_mean(col: str) -> Optional[float]:
            if col not in df.columns:
                return None
            v = df[col].astype(float).mean(skipna=True)
            return float(v) if not np.isnan(v) else None

        # Table 3: F1_T-O align as dataset-level F1 (not mean of per-instance binary labels)
        # TP: gold_to_align=1 AND to_align=1; FP: gold=0 AND pred=1; FN: gold=1 AND pred=0
        def _f1_to_align() -> Optional[float]:
            if "gold_to_align" not in df.columns or "to_align" not in df.columns:
                return None
            gold = df["gold_to_align"].astype(float)
            pred = df["to_align"].astype(float)
            tp = float(((gold == 1) & (pred == 1)).sum())
            fp = float(((gold == 0) & (pred == 1)).sum())
            fn = float(((gold == 1) & (pred == 0)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        def _pool_breakdown() -> Dict:
            """ECS_strict + primary pool Hit@K, grouped by gold_primary_pool."""
            if "gold_primary_pool" not in df.columns:
                return {}
            result: Dict = {}
            for pool in ("text", "table_cell", "definition"):
                mask = df["gold_primary_pool"] == pool
                sub = df[mask]
                if len(sub) == 0:
                    continue
                entry: Dict = {"n": int(len(sub))}
                ecs_v = sub["ecs_strict"].astype(float).mean(skipna=True)
                entry["ECS_strict"] = float(ecs_v) if not np.isnan(ecs_v) else None
                if pool == "text" and "hit_at_k_text" in df.columns:
                    v = sub["hit_at_k_text"].astype(float).mean(skipna=True)
                    entry["Hit@K_text"] = float(v) if not np.isnan(v) else None
                elif pool == "table_cell" and "hit_at_k_table_primary" in df.columns:
                    v = sub["hit_at_k_table_primary"].astype(float).mean(skipna=True)
                    entry["Hit@K_table_primary"] = float(v) if not np.isnan(v) else None
                elif pool == "definition" and "hit_at_k_def" in df.columns:
                    v = sub["hit_at_k_def"].astype(float).mean(skipna=True)
                    entry["Hit@K_def"] = float(v) if not np.isnan(v) else None
                result[pool] = entry
            return result

        def _failure_dist() -> Dict:
            """Failure reason breakdown among ECS_strict=0 instances."""
            if "failure_reason" not in df.columns or "ecs_strict" not in df.columns:
                return {}
            fail_df = df[df["ecs_strict"].astype(float) == 0.0]
            if len(fail_df) == 0:
                return {}
            total = len(fail_df)
            return {
                reason: float((fail_df["failure_reason"] == reason).sum()) / total
                for reason in ("parse_fail", "evidence_miss", "closure_fail", "both_fail", "no_pred")
                if (fail_df["failure_reason"] == reason).sum() > 0
            }

        return {
            "summary": {
                # Field-level metrics
                "Parseable":   col_mean("parse_ok"),
                "F1_role":     col_mean("f1_role"),
                "F1_fields":   col_mean("f1_fields"),
                # Evidence-level metrics
                "Evidence_Recall":     col_mean("evidence_recall"),
                "Hit@K_text":          col_mean("hit_at_k_text"),
                "Hit@K_table":         col_mean("hit_at_k_table"),          # ANY-HIT (kept for compat)
                "Hit@K_table_primary": col_mean("hit_at_k_table_primary"),  # PRIMARY only
                "Hit@K_def":           col_mean("hit_at_k_def"),
                # System-level metrics
                "ECS_noev":    col_mean("ecs_noev"),
                "ECS_strict":  col_mean("ecs_strict"),
                "ECS_oracle":  col_mean("ecs_oracle"),  # None if oracle_candidates not provided
                "Consistency": col_mean("consistency"),
                # Table 3: Graph-specific diagnostics
                "F1_measures":        col_mean("f1_measures"),
                "F1_defines":         col_mean("f1_defines"),
                "Cov_defines_to_D":   col_mean("cov_defines_to_D"),
                "F1_T-O_align":       _f1_to_align(),
                "Rate_temporal_valid": col_mean("rate_temporal_valid"),
                "Miss_D":             col_mean("miss_D"),
                "Miss_primary_evid":  col_mean("miss_primary_evid"),
                # Legacy
                "Field_Match": col_mean("f1_fields"),
            },
            "breakdown_by_primary_pool": _pool_breakdown(),
            "failure_reason_dist":       _failure_dist(),
            "error_distribution": {},  # populated by _analyze_errors
            "efficiency": {},          # populated by _analyze_efficiency
            "n_instances": len(df),
        }


# =============================================================================
# Quick sanity test (run directly: python e_evaluation.py)
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gold_data = [
        {
            "instance_id": "TEST:cc:000001",
            "nodes": [
                {"node_type": "Outcome", "text": "Emissions reduced",
                 "attrs": {"value": 12.86, "unit": "tCO2e", "year": 2023, "polarity": "improved"}},
                {"node_type": "Target",  "text": "Net zero by 2030",
                 "attrs": {"direction": "reduce", "value": 0.0, "unit": "tCO2e", "target_year": 2030}},
            ],
            "edges": [{"src": "outcome_1", "dst": "target_1", "edge_type": "supports"}],
            "gold_evidence_sets": [
                {
                    "set_id": "es_outcome",
                    "links": [
                        {"evid_id": "TEST:sent:000100", "relation": "numeric_support",
                         "provenance": {"pool": "text"}},
                        {"evid_id": "TEST:cell:T1:R2:C3", "relation": "numeric_support",
                         "provenance": {"pool": "table"}},
                    ]
                }
            ]
        }
    ]

    # Perfect prediction
    pred_perfect = [
        {
            "instance_id": "TEST:cc:000001",
            "prediction": {
                "nodes": [
                    {"node_type": "Outcome", "text": "Emissions reduced",
                     "attrs": {"value": 12.9, "unit": "tCO2e", "year": 2023, "polarity": "improved"}},
                    {"node_type": "Target",  "text": "Net zero by 2030",
                     "attrs": {"direction": "reduce", "value": 0.0, "unit": "tCO2e", "target_year": 2030}},
                ],
                "edges": [],
                "evidence_sets": [
                    {"set_id": "es1", "links": [
                        {"evid_id": "TEST:sent:000100"},
                        {"evid_id": "TEST:cell:T1:R2:C3"},
                    ]}
                ]
            },
            "validator_report": {
                "v0_gate": {"pass": True, "errors": []},
                "v1_gate": {"pass": True, "errors": []},
                "v2_gate": {"pass": True, "errors": []},
            },
            "pool_membership": {
                "TEST:sent:000100": "text",
                "TEST:cell:T1:R2:C3": "table_cell",
            }
        }
    ]

    evaluator = ModuleE_Evaluator()
    report, df = evaluator.run(gold_data, pred_perfect)

    print("=== Sanity test (perfect pred) ===")
    for k, v in report["summary"].items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    expected = {"Parseable": 1.0, "F1_role": 1.0, "F1_fields": 1.0,
                "Evidence_Recall": 1.0, "Hit@K_text": 1.0, "Hit@K_table": 1.0,
                "ECS_noev": 1.0, "ECS_strict": 1.0, "Consistency": 1.0}
    for k, v in expected.items():
        assert abs(report["summary"][k] - v) < 1e-6, f"FAIL: {k}={report['summary'][k]:.4f}, expected {v}"
    print("All assertions passed ✓")
