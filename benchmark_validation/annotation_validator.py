"""
Three-layer validator for EAGC GraphPrediction annotations.

Checks a GraphPrediction dict (a task's gold or predicted graph) against
schemas/eagc.schema.json and the task's evidence candidate pool, then runs
two further consistency layers:

Layers:
  V0 — Blocking: schema validity + evidence closed-set (evid_id ∈ candidate pool)
  V1 — Blocking: field consistency (year/unit/scope/baseline)
  V2 — Non-blocking: T-M-D-O closure diagnostics (Target--measures-->Metric,
        Def--defines-->Metric, Metric--supports-->Outcome). Under the current
        released behavior these checks emit warnings, not errors, for missing
        measures/defines edges or missing definition_support evidence, so a
        V2 warning does not make v2_pass or all_pass False; see
        data/DATA_CARD.md for the 789-record diagnostic-warning counts.

Schema edge types: supports, defines, measures, associated_with, contradicts
(Note: progress_toward is NOT a valid schema edge type)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .graph_integrity_validator import validate_graph_prediction
from .schema_validator import SchemaValidator


# ─── Unit normalization table ─────────────────────────────────────────────────

_UNIT_ALIASES: Dict[str, str] = {
    # GHG
    "tco2e": "tCO2e",
    "tco2eq": "tCO2e",
    "t co2e": "tCO2e",
    "tonnes co2e": "tCO2e",
    "mtco2e": "MtCO2e",
    "mt co2e": "MtCO2e",
    "million tonnes co2e": "MtCO2e",
    "ktco2e": "ktCO2e",
    "kgco2e": "kgCO2e",
    # Energy
    "mwh": "MWh",
    "gwh": "GWh",
    "twh": "TWh",
    "mj": "MJ",
    "gj": "GJ",
    "tj": "TJ",
    # Percent
    "%": "%",
    "percent": "%",
    "percentage": "%",
    # Water
    "m3": "m³",
    "cubic meters": "m³",
    "cubic metres": "m³",
    "ml": "ML",
    "megalitres": "ML",
    "megalitres (ml)": "ML",
    # Dimensionless / ratio
    "ratio": "ratio",
    "index": "index",
}


def normalize_unit(unit: str) -> str:
    if not unit or unit in ("unknown", "Unknown"):
        return unit
    return _UNIT_ALIASES.get(unit.strip().lower(), unit.strip())


def normalize_scope(scope: str) -> str:
    if not scope:
        return scope
    s = scope.strip()
    # "scope 1" → "Scope 1"
    return re.sub(r"(?i)\bscope\s+(\d+)", lambda m: f"Scope {m.group(1)}", s)


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class ValidationError:
    gate: str        # "V0" | "V1" | "V2"
    error_code: str  # e.g. "UNIT_MISMATCH"
    node_id: str
    message: str


@dataclass
class ValidationWarning:
    gate: str
    warning_code: str
    node_id: str
    message: str


@dataclass
class ValidatorReport:
    v0_pass: bool = False
    v1_pass: bool = False
    v2_pass: bool = False
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return self.v0_pass and self.v1_pass and self.v2_pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v0_pass": self.v0_pass,
            "v1_pass": self.v1_pass,
            "v2_pass": self.v2_pass,
            "all_pass": self.all_pass,
            "errors": [
                {"gate": e.gate, "error_code": e.error_code,
                 "node_id": e.node_id, "message": e.message}
                for e in self.errors
            ],
            "warnings": [
                {"gate": w.gate, "warning_code": w.warning_code,
                 "node_id": w.node_id, "message": w.message}
                for w in self.warnings
            ],
        }


# ─── Main Validator ───────────────────────────────────────────────────────────

class AnnotationValidator:
    """
    Three-layer validator for EAGC GraphPrediction annotations.

    Usage:
        validator = AnnotationValidator(schema_path="schemas/eagc.schema.json")
        pool_ids = {"DOC:sent:000001", "DOC:cell:tbl_001r001c001", ...}
        report = validator.validate(prediction_dict, pool_ids)
    """

    def __init__(self, schema_path: Optional[Path] = None):
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "schemas" / "eagc.schema.json"
        self._schema_validator = SchemaValidator(schema_path)

    def validate(
        self,
        prediction: Dict[str, Any],
        candidate_pool_evid_ids: Set[str],
        strict: bool = True,
    ) -> ValidatorReport:
        """
        Run all three validation layers.

        Args:
            prediction: GraphPrediction dict (7 top-level fields)
            candidate_pool_evid_ids: Set of all valid evid_ids from the task's
                                     candidate_pools (flattened covered_evidence_ids)
            strict: If True, treat integrity warnings as errors (recommended)

        Returns:
            ValidatorReport with v0/v1/v2_pass flags and error list
        """
        report = ValidatorReport()
        errors: List[ValidationError] = []
        warnings: List[ValidationWarning] = []

        # ── V0: Schema + Pointer ─────────────────────────────────────────────
        v0_errors = self._run_v0(prediction, candidate_pool_evid_ids, strict)
        errors.extend(v0_errors)
        report.v0_pass = len(v0_errors) == 0

        # ── V1: Field Consistency ────────────────────────────────────────────
        # Only check if V0 passed (otherwise node structure may be broken)
        if report.v0_pass:
            v1_errors, v1_warnings = self._run_v1(prediction)
            errors.extend(v1_errors)
            warnings.extend(v1_warnings)
            report.v1_pass = len(v1_errors) == 0
        else:
            report.v1_pass = False

        # ── V2: T-M-D-O Closure Diagnostics (non-blocking; see module docstring) ─
        if report.v0_pass:
            v2_errors, v2_warnings = self._run_v2(prediction)
            errors.extend(v2_errors)
            warnings.extend(v2_warnings)
            report.v2_pass = len(v2_errors) == 0
        else:
            report.v2_pass = False

        report.errors = errors
        report.warnings = warnings
        return report

    # ─── V0 Helpers ──────────────────────────────────────────────────────────

    def _run_v0(
        self,
        prediction: Dict[str, Any],
        pool_ids: Set[str],
        strict: bool,
    ) -> List[ValidationError]:
        errors: List[ValidationError] = []

        # Schema validation via existing SchemaValidator
        schema_result = self._schema_validator.validate_single(prediction)
        for err_msg in schema_result.errors:
            errors.append(ValidationError(
                gate="V0", error_code="SCHEMA_ERROR",
                node_id="<schema>", message=err_msg,
            ))

        # Structural integrity via existing graph_integrity_validator
        integrity_report = validate_graph_prediction(
            prediction, source_path="<in-memory>", strict=strict
        )
        for finding in integrity_report.errors:
            errors.append(ValidationError(
                gate="V0", error_code=finding.code,
                node_id=finding.path, message=finding.message,
            ))
        # In strict mode, integrity warnings become V0 errors too
        if strict:
            for finding in integrity_report.warnings:
                errors.append(ValidationError(
                    gate="V0", error_code=finding.code,
                    node_id=finding.path, message=f"[warning-as-error] {finding.message}",
                ))

        # Closed-set check: all evid_ids must be in candidate pool
        ev_sets = prediction.get("evidence_sets") or []
        doc_id = prediction.get("doc_id", "")
        # W_SELF_CITE: derive claim's own sentence ID from claim_id (cc:XXXXX → sent:XXXXX)
        claim_id = prediction.get("claim_id", "")
        cc_num = claim_id.split(":")[-1] if claim_id else ""
        self_evid_id = f"{doc_id}:sent:{cc_num}" if doc_id and cc_num else ""
        for si, ev_set in enumerate(ev_sets):
            for li, link in enumerate(ev_set.get("links") or []):
                evid_id = link.get("evid_id", "")
                # Warn if evidence cites the claim's own sentence
                if self_evid_id and evid_id == self_evid_id:
                    warnings.append(ValidationWarning(
                        gate="V0",
                        warning_code="W_SELF_CITE",
                        node_id=ev_set.get("set_id", f"evidence_sets[{si}]"),
                        message=(
                            f"Evidence link cites claim's own sentence "
                            f"(evid_id='{evid_id}' matches {claim_id})"
                        ),
                    ))
                is_manual = link.get("is_manual_add", None)
                if is_manual is True:
                    # Milestone 3 doc-local constraint: manual_add evidence must
                    # belong to the same document (evid_id format: {doc_id}:{type}:...)
                    # Cross-doc citations must use explicit external_evidence, not evid_id.
                    evid_doc = evid_id.split(":")[0] if evid_id else ""
                    if evid_id and evid_doc and doc_id and evid_doc != doc_id:
                        errors.append(ValidationError(
                            gate="V0", error_code="E_MANUAL_CROSS_DOC",
                            node_id=f"evidence_sets[{si}].links[{li}]",
                            message=(
                                f"is_manual_add=True but evid_id doc '{evid_doc}' "
                                f"!= task doc '{doc_id}' — cross-doc manual adds "
                                f"are not allowed; use external_evidence instead"
                            ),
                        ))
                    continue
                if pool_ids and evid_id and evid_id not in pool_ids:
                    errors.append(ValidationError(
                        gate="V0", error_code="EVIDENCE_OUT_OF_POOL",
                        node_id=f"evidence_sets[{si}].links[{li}]",
                        message=f"evid_id '{evid_id}' not in candidate pool (pool size={len(pool_ids)})",
                    ))

        return errors

    # ─── V1 Helpers ──────────────────────────────────────────────────────────

    def _run_v1(
        self, prediction: Dict[str, Any]
    ) -> tuple[List[ValidationError], List[ValidationWarning]]:
        errors: List[ValidationError] = []
        warnings: List[ValidationWarning] = []
        nodes = prediction.get("nodes") or []

        targets = [n for n in nodes if n.get("node_type") == "Target"]
        outcomes = [n for n in nodes if n.get("node_type") == "Outcome"]
        metrics = [n for n in nodes if n.get("node_type") == "Metric"]
        definitions = [n for n in nodes if n.get("node_type") == "Definition"]

        # V1.1 Year consistency: target_year > baseline_year
        for t in targets:
            attrs = t.get("attrs") or {}
            ty = attrs.get("target_year")
            by = attrs.get("baseline_year")
            nid = t.get("node_id", "target")
            if ty and by and isinstance(ty, (int, float)) and isinstance(by, (int, float)):
                if ty <= by:
                    errors.append(ValidationError(
                        gate="V1", error_code="YEAR_INCONSISTENCY", node_id=nid,
                        message=f"target_year ({ty}) must be > baseline_year ({by})",
                    ))

        # V1.2 Unit consistency: normalize(metric.unit) == normalize(outcome.unit)
        # Correct edge direction: metric --supports--> outcome (src=metric, dst=outcome)
        edges = prediction.get("edges") or []
        metric_to_outcomes: Dict[str, List[str]] = {}
        for e in edges:
            if e.get("edge_type") == "supports":
                mid = e.get("src")
                oid = e.get("dst")
                # Only consider supports edges where src is Metric and dst is Outcome
                nodes_by_id = {n.get("node_id"): n.get("node_type") for n in nodes}
                if (mid and oid and
                        nodes_by_id.get(mid) == "Metric" and
                        nodes_by_id.get(oid) == "Outcome"):
                    metric_to_outcomes.setdefault(mid, []).append(oid)

        metric_map = {n["node_id"]: n for n in metrics if "node_id" in n}
        outcome_map = {n["node_id"]: n for n in outcomes if "node_id" in n}

        for mid, oids in metric_to_outcomes.items():
            m_node = metric_map.get(mid)
            if not m_node:
                continue
            m_unit = normalize_unit((m_node.get("attrs") or {}).get("unit", ""))
            for oid in oids:
                o_node = outcome_map.get(oid)
                if not o_node:
                    continue
                o_unit = normalize_unit((o_node.get("attrs") or {}).get("unit", ""))
                if (m_unit and o_unit and
                        m_unit not in ("unknown", "Unknown") and
                        o_unit not in ("unknown", "Unknown") and
                        m_unit != o_unit):
                    errors.append(ValidationError(
                        gate="V1", error_code="UNIT_MISMATCH", node_id=oid,
                        message=f"outcome.unit '{o_unit}' != metric.unit '{m_unit}' (metric={mid})",
                    ))

        # V1.3 Scope consistency: metric.scope ⊆ definition.scope_covered
        # Find defines edges: Definition → Metric
        def_to_metric: Dict[str, List[str]] = {}
        for e in edges:
            if e.get("edge_type") == "defines":
                did = e.get("src")
                mid = e.get("dst")
                if did and mid:
                    def_to_metric.setdefault(did, []).append(mid)

        def_map = {n["node_id"]: n for n in definitions if "node_id" in n}

        for did, mids in def_to_metric.items():
            d_node = def_map.get(did)
            if not d_node:
                continue
            d_scope_covered = (d_node.get("attrs") or {}).get("scope_covered") or []
            d_scopes = {normalize_scope(s) for s in d_scope_covered}
            for mid in mids:
                m_node = metric_map.get(mid)
                if not m_node:
                    continue
                m_scope = (m_node.get("attrs") or {}).get("scope")
                if m_scope and m_scope not in ("unknown", "Unknown"):
                    m_scope_norm = normalize_scope(str(m_scope))
                    if d_scopes and m_scope_norm not in d_scopes:
                        warnings.append(ValidationWarning(
                            gate="V1", warning_code="SCOPE_MISMATCH",
                            node_id=mid,
                            message=(f"metric.scope '{m_scope_norm}' not in "
                                     f"definition.scope_covered {sorted(d_scopes)} "
                                     f"(def={did})"),
                        ))

        # V1.4 Baseline consistency: if polarity is set, baseline_year should exist
        for o in outcomes:
            attrs = o.get("attrs") or {}
            nid = o.get("node_id", "outcome")
            polarity = attrs.get("polarity")
            if polarity in ("improved", "worsened") and not attrs.get("baseline_year"):
                warnings.append(ValidationWarning(
                    gate="V1", warning_code="BASELINE_MISSING",
                    node_id=nid,
                    message=f"outcome.polarity='{polarity}' but baseline_year is missing",
                ))

        return errors, warnings

    # ─── V2 Helpers ──────────────────────────────────────────────────────────

    def _run_v2(
        self, prediction: Dict[str, Any]
    ) -> tuple[List[ValidationError], List[ValidationWarning]]:
        errors: List[ValidationError] = []
        warnings: List[ValidationWarning] = []
        nodes = prediction.get("nodes") or []
        edges = prediction.get("edges") or []
        ev_sets = prediction.get("evidence_sets") or []

        node_types = {n.get("node_id"): n.get("node_type") for n in nodes}

        # Build edge index
        edges_by_type: Dict[str, List[tuple]] = {}
        for e in edges:
            et = e.get("edge_type", "")
            edges_by_type.setdefault(et, []).append((e.get("src"), e.get("dst")))

        # Flatten all evidence links for primary evidence check
        all_links = [lk for es in ev_sets for lk in (es.get("links") or [])]

        # Helper: does any evidence link have definition_support?
        def has_any_def_support() -> bool:
            return any(lk.get("relation") == "definition_support" for lk in all_links)

        # V2.1: T-M chain — any Target should connect to at least one Metric via
        #        measures edge (correct direction: target --measures--> metric)
        target_ids = [n.get("node_id") for n in nodes if n.get("node_type") == "Target"]
        if target_ids:
            targets_with_metric = {s for s, d in edges_by_type.get("measures", [])
                                   if node_types.get(s) == "Target"}
            for tid in target_ids:
                if tid not in targets_with_metric:
                    warnings.append(ValidationWarning(
                        gate="V2", warning_code="TARGET_NO_METRIC",
                        node_id=tid,
                        message=f"Target '{tid}' has no measures edge to a Metric",
                    ))

        # V2.2: For Metrics with a Target via measures, check Definition exists
        #        (defines edge: definition → metric)
        metric_ids = [n.get("node_id") for n in nodes if n.get("node_type") == "Metric"]
        if metric_ids:
            # Metrics that are measured by a Target
            target_measured_metrics = {d for s, d in edges_by_type.get("measures", [])
                                       if node_types.get(s) == "Target"}
            # Metrics that have a defines edge from a Definition
            defined_metrics = {d for s, d in edges_by_type.get("defines", [])
                               if node_types.get(s) == "Definition"}
            for mid in target_measured_metrics:
                if mid not in defined_metrics:
                    warnings.append(ValidationWarning(
                        gate="V2", warning_code="DEFINITION_MISSING",
                        node_id=mid,
                        message=(f"Metric '{mid}' is measured by a Target but has no "
                                 f"Definition (missing defines edge)"),
                    ))

        # V2.3: Primary evidence completeness — prediction should have some primary evidence
        has_primary_overall = any(
            lk.get("importance") == "primary" for lk in all_links
        )
        if all_links and not has_primary_overall:
            errors.append(ValidationError(
                gate="V2", error_code="PRIMARY_EVIDENCE_MISSING",
                node_id="<evidence_sets>",
                message="No primary evidence links found in any evidence set",
            ))

        # V2.4: If definition node present, check definition_support evidence exists
        def_ids = [n.get("node_id") for n in nodes if n.get("node_type") == "Definition"]
        if def_ids and not has_any_def_support():
            warnings.append(ValidationWarning(
                gate="V2", warning_code="DEFINITION_EVIDENCE_MISSING",
                node_id="<evidence_sets>",
                message="Definition node(s) present but no definition_support evidence found",
            ))

        return errors, warnings
