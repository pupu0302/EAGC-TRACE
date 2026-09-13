import copy
import json
import os
import logging
from typing import Dict, List, Any, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# Module V class definition (full version: V0+V1+V2)
# ==========================================
class ModuleV_Validator:
    """
    Module V: The Grand Auditor
    Runs the triple gate: V0 (hallucination/format), V1 (numeric), V2 (logic).
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Tolerance: allow 5% relative error (e.g. 10% vs 10.5% is OK)
        self.tolerance = self.config.get("numeric_tolerance", 0.05)

    def run(self, g_output: Dict, r1_evidence_pool: List[Dict]) -> Dict:
        """
        Run the full audit
        """
        instance_id = g_output.get("instance_id", "unknown")
        logger.info(f"[{instance_id}] Starting Module V Validation...")

        # ============================================================
        # Use a deep copy to avoid mutating the original data
        # ============================================================
        if "prediction" in g_output and isinstance(g_output["prediction"], dict):
            # Nested format: prediction.nodes
            prediction = copy.deepcopy(g_output["prediction"])  # <- deep copy
            is_nested = True
            logger.debug(f"[{instance_id}] Using nested format (prediction.nodes)")
        else:
            # Flat format: fields live at the top level
            prediction = {
                "claim_id": g_output.get("claim_id"),
                "doc_id": g_output.get("doc_id"),
                "nodes": g_output.get("nodes", []),
                "edges": g_output.get("edges", []),
                "evidence_sets": g_output.get("evidence_sets", [])
            }
            is_nested = False
            logger.debug(f"[{instance_id}] Using flat format (top-level nodes)")
        # ============================================================

        # ============================================================
        # Debug logging
        # ============================================================
        logger.info(f"[{instance_id}] Extracted prediction: "
                    f"nodes={len(prediction.get('nodes', []))}, "
                    f"edges={len(prediction.get('edges', []))}, "
                    f"evidence_sets={len(prediction.get('evidence_sets', []))}")
        # ============================================================

        # 1. Build the evidence lookup table (ID -> content) and the set of valid IDs
        evidence_map = {item['evid_id']: item for item in r1_evidence_pool if item.get('evid_id')}
        valid_pool_ids = set(evidence_map.keys())

        # 2. Initialize the report structure
        validator_report = {
            "v0_gate": {"pass": False, "errors": [], "cited_count": 0},
            "v1_gate": {"pass": True, "errors": [], "audit_trail": []},
            "v2_gate": {"pass": True, "errors": []}
        }

        # ==========================================
        # 🚪 Gate V0: Structure & Hallucination
        # ==========================================
        if not self._audit_v0(prediction, valid_pool_ids, validator_report["v0_gate"]):
            logger.warning(f"[{instance_id}] V0 Failed. Stopping validation.")
            return self._finalize(g_output, validator_report, passed=False, is_nested=is_nested, prediction=prediction)

        # ==========================================
        # 🚪 Gate V1: Numeric Consistency
        # ==========================================
        self._audit_v1(prediction, evidence_map, validator_report["v1_gate"])

        # ==========================================
        # 🚪 Gate V2: Scope & Logic Completeness
        # ==========================================
        self._audit_v2(prediction, evidence_map, validator_report["v2_gate"])

        # 3. Combined verdict
        all_passed = (validator_report["v0_gate"]["pass"] and
                      validator_report["v1_gate"]["pass"] and
                      validator_report["v2_gate"]["pass"])

        status_msg = "PASSED" if all_passed else "FAILED"
        logger.info(f"[{instance_id}] Validation Finished. Status: {status_msg}")

        return self._finalize(g_output, validator_report, passed=all_passed, is_nested=is_nested, prediction=prediction)

    def _finalize(self, g_output: Dict, report: Dict, passed: bool, is_nested: bool, prediction: Dict) -> Dict:
        """
        Construct the final output format
        """
        instance_id = g_output.get("instance_id", "unknown")

        # ============================================================
        # Ensure evidence_sets is present
        # ============================================================
        if "evidence_sets" not in prediction:
            # Try to recover it from the original output
            if "prediction" in g_output and "evidence_sets" in g_output["prediction"]:
                prediction["evidence_sets"] = g_output["prediction"]["evidence_sets"]
                logger.warning(f"[{instance_id}] Restored evidence_sets from original g_output")
            else:
                prediction["evidence_sets"] = []
                logger.warning(f"[{instance_id}] No evidence_sets found, using empty list")
        # ============================================================

        result = {
            "module": "Module_V",
            "instance_id": instance_id,
            "status": "audit_passed" if passed else "audit_failed",
            "prediction": prediction,
            "validator_report": report,
            "meta": g_output.get("meta", {})
        }

        # ============================================================
        # Debug logging
        # ============================================================
        logger.info(f"[{instance_id}] Module V output: "
                    f"prediction.nodes={len(result['prediction'].get('nodes', []))}, "
                    f"prediction.evidence_sets={len(result['prediction'].get('evidence_sets', []))}, "
                    f"status={result['status']}")
        # ============================================================

        return result

    # --- V0 Logic ---
    def _audit_v0(self, prediction: Dict, valid_ids: Set[str], report: Dict) -> bool:
        errors = []
        if not prediction or not isinstance(prediction, dict):
            errors.append("Invalid JSON structure")
            report["errors"] = errors
            return False

        cited_ids = set()
        hallucinations = []

        # Iterate over all evidence sets
        for es in prediction.get("evidence_sets", []):
            for link in es.get("links", []):
                eid = link.get("evid_id")
                if eid:
                    cited_ids.add(eid)
                    if eid not in valid_ids:
                        hallucinations.append(eid)

        if hallucinations:
            errors.append(f"HALLUCINATION_ERROR: Found {len(hallucinations)} invalid IDs: {hallucinations}")

        report["pass"] = len(errors) == 0
        report["errors"] = errors
        report["cited_count"] = len(cited_ids)
        return report["pass"]

    # --- V1 Logic ---
    def _audit_v1(self, prediction: Dict, evidence_map: Dict, report: Dict):
        nodes = prediction.get("nodes", [])
        evidence_sets = prediction.get("evidence_sets", [])

        outcome_node = next((n for n in nodes
                            if (n.get("node_type") or n.get("type")) == "Outcome"), None)
        if not outcome_node:
            return  # No numeric check without an Outcome node

        # Get the claimed value
        try:
            claimed_val = float(outcome_node["attrs"].get("value", 0))
        except (ValueError, TypeError):
            return

        # Get the evidence value
        es_outcome = next((es for es in evidence_sets if es["attached_to_node"] == outcome_node["node_id"]), None)
        if not es_outcome:
            return

        numeric_vals = []
        for link in es_outcome["links"]:
            evid = evidence_map.get(link["evid_id"])
            if evid and evid["type"] == "table_cell":
                try:
                    # Strip commas and percent signs
                    clean_content = evid["content"].replace(",", "").replace("%", "")
                    val = float(clean_content)
                    numeric_vals.append({"val": val, "year": evid.get("row_label")})
                except ValueError:
                    continue

        if len(numeric_vals) >= 2:
            # Sort by year
            sorted_vals = sorted(numeric_vals, key=lambda x: str(x["year"]))
            v_start = sorted_vals[0]["val"]
            v_end = sorted_vals[-1]["val"]

            if v_start == 0:
                return

            actual_change_pct = ((v_end - v_start) / v_start) * 100
            actual_abs_change = abs(actual_change_pct)

            report["audit_trail"].append(
                f"Evidence Calc: {v_start} -> {v_end} = {actual_change_pct:.2f}% (Abs: {actual_abs_change:.2f}%)")
            report["audit_trail"].append(f"Claimed Value: {claimed_val}%")

            diff = abs(actual_abs_change - claimed_val)

            # Verdict criterion: absolute error > 1.0 (i.e. more than 1 percentage point off)
            if diff > 1.0:
                report["pass"] = False
                report["errors"].append(
                    f"NUMERIC_MISMATCH: Evidence implies {actual_abs_change:.1f}%, Claim says {claimed_val}%. Diff: {diff:.2f}")
            else:
                report["audit_trail"].append("Numeric verification passed.")

    # --- V2 Logic ---
    def _audit_v2(self, prediction: Dict, evidence_map: Dict, report: Dict):
        nodes = prediction.get("nodes", [])
        evidence_sets = prediction.get("evidence_sets", [])

        metric_node = next((n for n in nodes
                           if (n.get("node_type") or n.get("type")) == "Metric"), None)
        if not metric_node:
            return

        metric_text = metric_node.get("text", "").lower()

        # Check whether the Outcome's supporting-evidence headers cover the Metric's scope
        outcome_node = next((n for n in nodes
                            if (n.get("node_type") or n.get("type")) == "Outcome"), None)
        if outcome_node:
            es_outcome = next((es for es in evidence_sets if es["attached_to_node"] == outcome_node["node_id"]), None)
            if es_outcome:
                headers = []
                for link in es_outcome["links"]:
                    evid = evidence_map.get(link["evid_id"])
                    if evid and "col_header" in evid:
                        headers.append(evid["col_header"].lower())

                headers_str = " ".join(headers)

                # Simple keyword-check logic
                if "scope 1+2" in metric_text:
                    has_scope1 = "scope 1" in headers_str
                    has_scope2 = "scope 2" in headers_str

                    if has_scope1 and not has_scope2:
                        report["pass"] = False
                        report["errors"].append(
                            f"SCOPE_MISMATCH: Metric '{metric_node['text']}' requires Scope 2, but evidence headers only contain Scope 1.")


# ==========================================
# Main execution logic
# ==========================================
def main():
    # 1. Define paths
    base_dir = "quickstart"

    r1_path = os.path.join(base_dir, "r1_output.json")
    g_path = os.path.join(base_dir, "g_output.json")
    v_path = os.path.join(base_dir, "v_output.json")

    # 2. Check that the files exist
    if not os.path.exists(r1_path):
        logger.error(f"R1 Output not found at: {r1_path}")
        return
    if not os.path.exists(g_path):
        logger.error(f"G Output not found at: {g_path}")
        return

    logger.info("Loading input files...")

    # 3. Read the R1 data (evidence pool)
    with open(r1_path, 'r', encoding='utf-8') as f:
        r1_data = json.load(f)
        # Compatibility handling: keep it as-is if it is a list, wrap it in one if it is a single object
        r1_instances = r1_data if isinstance(r1_data, list) else [r1_data]
        # Build the lookup table: instance_id -> evidence_pool
        r1_map = {}
        for inst in r1_instances:
            iid = inst.get("instance_id")
            pool = inst.get("evidence_pool", [])
            if iid:
                r1_map[iid] = pool

    # 4. Read the G data (graph prediction)
    with open(g_path, 'r', encoding='utf-8') as f:
        g_data = json.load(f)
        g_instances = g_data if isinstance(g_data, list) else [g_data]

    # 5. Initialize Module V
    validator = ModuleV_Validator()
    v_results = []

    # 6. Batch process
    for g_inst in g_instances:
        inst_id = g_inst.get("instance_id")

        # Look up the matching evidence pool
        evidence_pool = r1_map.get(inst_id)

        if not evidence_pool:
            logger.warning(f"[{inst_id}] No matching evidence pool found in R1 output. Skipping.")
            # Alternatively, a failed report could be generated instead
            failed_res = {
                "module": "Module_V",
                "instance_id": inst_id,
                "status": "error",
                "error_message": "Evidence pool not found",
                "prediction": {"nodes": [], "edges": [], "evidence_sets": []},
                "validator_report": {
                    "v0_gate": {"pass": False, "errors": ["Evidence pool not found"], "cited_count": 0},
                    "v1_gate": {"pass": False, "errors": [], "audit_trail": []},
                    "v2_gate": {"pass": False, "errors": []}
                },
                "meta": {}
            }
            v_results.append(failed_res)
            continue

        # Run the audit
        v_output = validator.run(g_inst, evidence_pool)
        v_results.append(v_output)

    # 7. Save the results
    logger.info(f"Saving validation results to: {v_path}")
    with open(v_path, 'w', encoding='utf-8') as f:
        # If there is only one result, save it as an object; otherwise save a list
        if len(v_results) == 1:
            json.dump(v_results[0], f, indent=2)
        else:
            json.dump(v_results, f, indent=2)

    logger.info("✅ Done! Module V output saved.")


if __name__ == "__main__":
    main()
