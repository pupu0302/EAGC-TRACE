import json
import copy
import re
import logging
from typing import Dict, List, Any
from llm.g_llm_client import create_llm_client
from llm.config_loader import get_config_loader

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Module SR-D: Deterministic Self-Repair (rule-based repair)
# =============================================================================
class ModuleSR_D:
    """
    Handles deterministic, unambiguous errors.
    Examples: formatting errors, small numeric rounding errors, orphan-edge cleanup.
    """

    def __init__(self):
        self.numeric_threshold = 1.0  # tolerance for auto-repair (1.0 percentage point)

    def run(self, v_output: Dict, evidence_pool: List[Dict] = None) -> Dict:
        print("\n--- [SR-D] Starting Deterministic Repair ---")

        # Deep-copy to avoid mutating the original data
        repaired_output = copy.deepcopy(v_output)
        repaired_output["module"] = "Module_SR_D"
        repaired_output["repair_trace"] = []

        report = v_output.get("validator_report", {})
        prediction = repaired_output.get("prediction", {})

        # 0. Fix hallucinated evid_id (V0 failure: cites an ID not in the candidate pool)
        if not report.get("v0_gate", {}).get("pass", True) and evidence_pool:
            self._fix_hallucinated_ids(prediction, evidence_pool, repaired_output["repair_trace"])

        # 1. Try to fix V1 (numeric) errors
        if not report.get("v1_gate", {}).get("pass", True):
            self._try_fix_numeric(prediction, report["v1_gate"]["errors"], repaired_output["repair_trace"])

        # 2. Try to fix V2 (logic) errors
        if not report.get("v2_gate", {}).get("pass", True):
            self._try_fix_scope(prediction, report["v2_gate"]["errors"], repaired_output["repair_trace"])

        # 3. Clean up orphan edges (general rule)
        self._cleanup_orphans(prediction, repaired_output["repair_trace"])

        # Status flag
        if repaired_output["repair_trace"]:
            repaired_output["status"] = "partially_repaired"
            print(f"[SR-D] Applied fixes: {repaired_output['repair_trace']}")
        else:
            repaired_output["status"] = "repair_skipped"
            print("[SR-D] No deterministic rules matched. Passing to SR-L.")

        return repaired_output

    def _fix_hallucinated_ids(self, prediction: Dict, evidence_pool: List[Dict], trace: List[str]):
        """
        Deterministic repair: replace a hallucinated evid_id (an ID not in the
        candidate pool) with the best valid ID from the candidate pool.
        Replacement strategy: for each evidence set, use its attached_to_node's
        text as the query, and within the pool pick the candidate with the
        highest token overlap, preferring type order (text > definition >
        table_cell). If the query is empty, fall back to rank-1 (the original
        logic).
        """
        valid_ids = {item["evid_id"] for item in evidence_pool}
        if not valid_ids:
            return

        # node_id -> node text, used for semantic matching
        node_texts: Dict[str, str] = {
            n["node_id"]: n.get("text", "")
            for n in prediction.get("nodes", [])
        }

        # set_id -> attached_to_node
        set_to_node: Dict[str, str] = {
            es.get("set_id", ""): es.get("attached_to_node", "")
            for es in prediction.get("evidence_sets", [])
        }

        # Build a per-type candidate list: [(evid_id, content), ...], preserving original rank order
        pool_by_type: Dict[str, List[tuple]] = {}
        for item in evidence_pool:
            t = item.get("type", "text")
            pool_by_type.setdefault(t, []).append(
                (item["evid_id"], item.get("content", ""))
            )

        def _token_overlap(a: str, b: str) -> float:
            """Simple token-level Jaccard overlap, case-insensitive."""
            ta = set(a.lower().split())
            tb = set(b.lower().split())
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / max(len(ta), len(tb))

        def _best_replacement(set_id: str) -> str:
            node_id = set_to_node.get(set_id, "")
            query = node_texts.get(node_id, "")
            for pool_type in ("text", "definition", "table_cell"):
                candidates = pool_by_type.get(pool_type, [])
                if not candidates:
                    continue
                if not query:
                    return candidates[0][0]  # fall back to rank-1 when there is no query
                # Take the candidate with the highest token overlap
                best_id = max(candidates, key=lambda c: _token_overlap(query, c[1]))[0]
                return best_id
            return next(iter(valid_ids))

        n_fixed = 0
        for es in prediction.get("evidence_sets", []):
            set_id = es.get("set_id", "")
            for link in es.get("links", []):
                eid = link.get("evid_id", "")
                if eid and eid not in valid_ids:
                    replacement = _best_replacement(set_id)
                    link["evid_id"] = replacement
                    link["_sr_d_substituted"] = True
                    n_fixed += 1

        if n_fixed > 0:
            prediction.setdefault("meta", {})["sr_d_hallucination_fix"] = True
            trace.append(f"SR-D hallucination fix: substituted {n_fixed} invalid evid_id(s) with semantically-matched valid pool candidates.")

    def _try_fix_numeric(self, prediction: Dict, errors: List[str], trace: List[str]):
        """
        Rule: extract the evidence-implied value from the error message and
        update the Outcome node. Auto-repair applies only when
        diff <= numeric_threshold (1.0 percentage point); larger differences
        must go to SR-L.
        """
        for error in errors:
            # Error format: "NUMERIC_MISMATCH: Evidence implies X%, Claim says Y%. Diff: Z"
            m = re.search(r'Evidence implies ([\d.]+)%.*?Diff:\s*([\d.]+)', error)
            if not m:
                continue
            try:
                evidence_val = float(m.group(1))
                diff_val = float(m.group(2))
            except ValueError:
                continue

            if diff_val <= self.numeric_threshold:
                # Find the Outcome node and update its value
                nodes = prediction.get("nodes", [])
                for node in nodes:
                    if node.get("type") == "Outcome":
                        old_val = node.get("attrs", {}).get("value")
                        node.setdefault("attrs", {})["value"] = evidence_val
                        trace.append(
                            f"SR-D numeric fix: Outcome.value {old_val} → {evidence_val} (diff={diff_val:.3f})"
                        )
                        break

    def _try_fix_scope(self, prediction: Dict, errors: List[str], trace: List[str]):
        """
        Rule: SR-D cannot handle scope mismatches, since these require
        renaming nodes or deleting definitions.
        """
        pass  # left to SR-L

    def _cleanup_orphans(self, prediction: Dict, trace: List[str]):
        """
        General rule: remove edges pointing to nonexistent nodes
        """
        node_ids = {n["node_id"] for n in prediction.get("nodes", [])}
        original_edges = prediction.get("edges", [])
        valid_edges = [e for e in original_edges if e["src"] in node_ids and e["dst"] in node_ids]

        if len(valid_edges) < len(original_edges):
            diff = len(original_edges) - len(valid_edges)
            prediction["edges"] = valid_edges
            trace.append(f"Removed {diff} orphan edges.")


# =============================================================================
# 2. Module SR-L: LLM-based Self-Repair (intelligent repair)
# =============================================================================
class ModuleSR_L:
    """
    Handles errors that require reasoning, such as semantic conflicts and
    hallucination correction.
    """

    def __init__(self):
        pass

    def run(self, sr_d_output: Dict) -> Dict:
        print("\n--- [SR-L] Starting LLM-based Repair ---")

        # If SR-D had already fully repaired things (hypothetically), SR-L
        # would not be needed. In this case, though, SR-D skipped repair, so
        # we continue.

        repaired_output = copy.deepcopy(sr_d_output)
        repaired_output["module"] = "Module_SR_L"

        # 1. Build the prompt
        prompt = self._construct_prompt(sr_d_output)
        print(f"[SR-L] Prompt Constructed (Length: {len(prompt)} chars)")
        # print(prompt) # Debug usage

        # 2. Call the LLM (a mock stands in for the LLM's response here)
        # In real production, this would be: response = openai.ChatCompletion.create(...)
        llm_response_json = self._mock_llm_inference(prompt)

        # 3. Parse and apply the repair
        if llm_response_json:
            repaired_output["prediction"] = llm_response_json
            repaired_output["status"] = "repaired"

            # Auto-generate the repair log (diff before/after)
            trace = self._generate_repair_trace(sr_d_output["prediction"], llm_response_json)
            repaired_output["repair_trace"] = trace
            print(f"[SR-L] Repair Success. Trace: {trace}")
        else:
            repaired_output["status"] = "repair_failed"
            print("[SR-L] LLM failed to generate valid JSON.")

        return repaired_output

    def _construct_prompt(self, data: Dict) -> str:
        """Build the repair-instruction prompt"""
        pred = data["prediction"]
        report = data["validator_report"]

        errors = []
        if not report["v1_gate"]["pass"]: errors.extend(report["v1_gate"]["errors"])
        if not report["v2_gate"]["pass"]: errors.extend(report["v2_gate"]["errors"])

        prompt = f"""
You are an expert ESG Auditor. Fix the Knowledge Graph based on the validation errors.

Current Graph Nodes: {json.dumps(pred['nodes'], indent=2)}
Current Graph Edges: {json.dumps(pred['edges'], indent=2)}

Validation Errors:
{json.dumps(errors, indent=2)}

Instructions:
1. For NUMERIC_MISMATCH: If the evidence calculation is robust, update the Outcome node value to match the evidence.
2. For SCOPE_MISMATCH: Narrow down the Metric node definition to match the available evidence. Remove unsupported definitions.
3. Return ONLY the fixed JSON object with keys: "nodes", "edges", "evidence_sets".
"""
        return prompt

    def _mock_llm_inference(self, prompt: str) -> Dict:
        """
        [Mock] LLM output.
        The expected repair result is hand-coded here, standing in for what
        an LLM would produce after reasoning about the errors.
        """
        # Expected behavior:
        # 1. Metric: "Scope 1+2" -> "Scope 1"
        # 2. Outcome: value 10 -> 12.9
        # 3. Remove the Scope 2 Definition node and its edge

        return {
            "claim_id": "JPM_2023:claim:000000",
            "doc_id": "JPM_2023",
            "nodes": [
                {
                    "node_id": "n_metric",
                    "type": "Metric",
                    "text": "Scope 1 emissions",  # <--- FIXED: Renamed
                    "attrs": {"unit": "tCO2e"}
                },
                {
                    "node_id": "n_outcome",
                    "type": "Outcome",
                    "text": "reduced by 10%",
                    "attrs": {
                        "direction": "decrease",
                        "value": 12.9,  # <--- FIXED: Updated Value
                        "unit": "percent",
                        "year": 2023
                    }
                },
                {
                    "node_id": "n_def_s1",
                    "type": "Definition",
                    "text": "Scope 1 Definition",
                    "attrs": {"type": "standard"}
                }
                # <--- FIXED: Removed n_def_s2
            ],
            "edges": [
                {"src": "n_outcome", "dst": "n_metric", "edge_type": "measures"},
                {"src": "n_def_s1", "dst": "n_metric", "edge_type": "defines"}
                # <--- FIXED: Removed edge to n_def_s2
            ],
            "evidence_sets": [
                {
                    "set_id": "es_outcome",
                    "attached_to_node": "n_outcome",
                    "links": [
                        {"evid_id": "JPM_2023:cell:emissions:000:001", "relation": "numeric_support"},
                        {"evid_id": "JPM_2023:cell:emissions:002:001", "relation": "numeric_support"}
                    ]
                },
                {
                    "set_id": "es_def",
                    "attached_to_node": "n_metric",
                    "links": [
                        {"evid_id": "JPM_2023:def:000000", "relation": "definition_support"}
                    ]
                }
            ]
        }

    def _generate_repair_trace(self, old_pred: Dict, new_pred: Dict) -> List[str]:
        """Simple diff generator, used to produce repair_trace"""
        trace = []

        # Check Metric text changes
        old_metric = next((n for n in old_pred["nodes"] if n["type"] == "Metric"), {})
        new_metric = next((n for n in new_pred["nodes"] if n["type"] == "Metric"), {})
        if old_metric.get("text") != new_metric.get("text"):
            trace.append(f"Metric renamed: '{old_metric.get('text')}' -> '{new_metric.get('text')}'")

        # Check Outcome value changes
        old_out = next((n for n in old_pred["nodes"] if n["type"] == "Outcome"), {})
        new_out = next((n for n in new_pred["nodes"] if n["type"] == "Outcome"), {})
        if old_out.get("attrs", {}).get("value") != new_out.get("attrs", {}).get("value"):
            trace.append(
                f"Outcome value updated: {old_out.get('attrs', {}).get('value')} -> {new_out.get('attrs', {}).get('value')}")

        # Check node-count changes
        if len(old_pred["nodes"]) > len(new_pred["nodes"]):
            trace.append(f"Removed {len(old_pred['nodes']) - len(new_pred['nodes'])} nodes (Scope narrowing).")

        return trace


# =============================================================================
# 2. Module SR-L: LLM-based Self-Repair (intelligent repair)
# =============================================================================
class ModuleSR_L_LLM:
    """
    Handles errors that require reasoning, such as semantic conflicts and
    hallucination correction. Uses an LLM to fix the graph structure based
    on the validator's error report.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the SR-L module and load the LLM client
        """
        # 1. Load configuration (reuses Module G's config logic; the repair
        # model and the generation model are usually the same)
        if config is None:
            config_loader = get_config_loader()
            # Try to use module_g's config as the default
            module_conf = config_loader.get_module_config("module_g")
            model_ref = module_conf.get("model")

            # Get the detailed model configuration
            all_models = config_loader.config.get("models", {})
            model_details = all_models.get(model_ref, {})

            self.config = {**module_conf, **model_details}
        else:
            self.config = config

        # 2. Initialize the LLM client
        provider = self.config.get("provider", "mock")
        model_name = self.config.get("model_name")

        self.llm_client = create_llm_client(provider, model_name, self.config)
        logger.info(f"[SR-L] Initialized with Provider: {provider}, Model: {model_name}")

    def run(self, sr_d_output: Dict, evidence_pool: List[Dict] = None) -> Dict:
        """
        Execute the repair
        :param sr_d_output: SR-D's output (contains the current prediction and validator_report)
        :param evidence_pool: (optional) the original evidence pool, provided for the LLM's reference
        """
        print("\n--- [SR-L] Starting LLM-based Repair ---")

        repaired_output = copy.deepcopy(sr_d_output)
        repaired_output["module"] = "Module_SR_L"

        # 1. Build the prompt
        prompt = self._construct_prompt(sr_d_output, evidence_pool)
        logger.info(f"[SR-L] Prompt Constructed (Length: {len(prompt)} chars)")

        # 2. Call the LLM
        try:
            llm_response_json = self._call_llm(prompt)

            # 3. Parse and apply the repair
            if llm_response_json and "nodes" in llm_response_json:
                # Simple structural validation
                repaired_output["prediction"]["nodes"] = llm_response_json.get("nodes", [])
                repaired_output["prediction"]["edges"] = llm_response_json.get("edges", [])
                repaired_output["prediction"]["evidence_sets"] = llm_response_json.get("evidence_sets", [])

                repaired_output["status"] = "repaired"

                # Generate the repair log
                trace = self._generate_repair_trace(sr_d_output["prediction"], repaired_output["prediction"])
                repaired_output["repair_trace"] = trace

                print(f"[SR-L] Repair Success. Trace: {trace}")
                logger.info(f"[SR-L] Repair Success. Trace: {trace}")
            else:
                repaired_output["status"] = "repair_failed"
                logger.error("[SR-L] LLM returned invalid JSON structure.")
                print("[SR-L] LLM returned invalid JSON structure.")

        except Exception as e:
            repaired_output["status"] = "repair_error"
            logger.error(f"[SR-L] Critical Error: {e}")
            print(f"[SR-L] Critical Error: {e}")

        return repaired_output

    def _construct_prompt(self, data: Dict, evidence_pool: List[Dict] = None) -> str:
        """Build the repair-instruction prompt"""
        pred = data["prediction"]
        report = data["validator_report"]

        # Extract error information (includes all V0/V1/V2 errors)
        errors = []
        if not report.get("v0_gate", {}).get("pass", True):
            errors.extend(report["v0_gate"].get("errors", []))
        if not report.get("v1_gate", {}).get("pass", True):
            errors.extend(report["v1_gate"].get("errors", []))
        if not report.get("v2_gate", {}).get("pass", True):
            errors.extend(report["v2_gate"].get("errors", []))

        # Prepare the evidence text (includes valid evid_id values, for
        # HALLUCINATION_ERROR repair reference)
        evidence_text = "Not provided."
        if evidence_pool:
            evidence_items = []
            for item in evidence_pool[:15]:  # cap the count to avoid overflowing the context
                evid_id = item.get("evid_id", item.get("id", "unknown"))
                content = item.get("content") or item.get("text", "")
                evidence_items.append(f"evid_id: {evid_id}\nContent: {content[:200]}")
            evidence_text = "\n---\n".join(evidence_items)

        prompt = f"""
You are an expert ESG Auditor and Knowledge Graph Repair Agent.
Your task is to FIX a Knowledge Graph based on specific validation errors.

=== 1. CONTEXT (Valid Evidence Pool) ===
{evidence_text}

=== 2. CURRENT BROKEN GRAPH ===
Nodes: {json.dumps(pred.get('nodes', []), indent=2)}
Edges: {json.dumps(pred.get('edges', []), indent=2)}
Evidence Sets: {json.dumps(pred.get('evidence_sets', []), indent=2)}

=== 3. VALIDATION ERRORS (Must Fix) ===
{json.dumps(errors, indent=2)}

=== 4. INSTRUCTIONS ===
1. Analyze the errors and apply the appropriate fix:
   - If "HALLUCINATION_ERROR": The cited evid_id values do not exist in the evidence pool.
     Fix: For each invalid evid_id in evidence_sets[].links, replace it with the most
     semantically relevant evid_id from the CONTEXT above. Do NOT delete the link —
     always substitute a valid evid_id. If uncertain, use the first evid_id listed in CONTEXT.
   - If "NUMERIC_MISMATCH": The Claim value differs from Evidence. Trust the EVIDENCE.
     Update the Outcome node value to match the evidence calculation.
   - If "SCOPE_MISMATCH": The Metric definition is too broad/wrong. Rename the Metric node
     or remove incorrect Definition nodes to match the available evidence.
   - If "ORPHAN_EDGE": Remove edges pointing to non-existent nodes.
2. Maintain the JSON structure exactly. All evid_id values in the output must exist in CONTEXT.
3. Output the FULL corrected graph (nodes, edges, evidence_sets).

=== 5. OUTPUT FORMAT ===
Return ONLY a valid JSON object. No markdown.
{{
  "nodes": [...],
  "edges": [...],
  "evidence_sets": [...]
}}
"""
        return prompt

    def _call_llm(self, user_prompt: str) -> Dict:
        """
        Call the real LLM client
        """
        system_prompt = "You are a helpful assistant that outputs valid JSON only."

        # Uses G_LLMClient's generate method
        # Note: this relies on the previously fixed retry logic
        response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format="json_object",
            temperature=0.1,  # repair tasks need low temperature for determinism
            max_tokens=4096
        )

        if response.get("parsed_json"):
            return response["parsed_json"]
        else:
            raise ValueError(f"Failed to parse LLM response: {response.get('content')}")

    def _generate_repair_trace(self, old_pred: Dict, new_pred: Dict) -> List[str]:
        """Simple diff generator"""
        trace = []

        # Helper to find node by ID (assuming IDs match, otherwise by type)
        def get_node(nodes, type_name):
            return next((n for n in nodes if n.get("type") == type_name), {})

        # 1. Check Metric Name
        old_metric = get_node(old_pred.get("nodes", []), "Metric")
        new_metric = get_node(new_pred.get("nodes", []), "Metric")
        if old_metric.get("text") != new_metric.get("text"):
            trace.append(f"Metric renamed: '{old_metric.get('text')}' -> '{new_metric.get('text')}'")

        # 2. Check Outcome Value
        old_out = get_node(old_pred.get("nodes", []), "Outcome")
        new_out = get_node(new_pred.get("nodes", []), "Outcome")
        old_val = old_out.get("attrs", {}).get("value")
        new_val = new_out.get("attrs", {}).get("value")
        if old_val != new_val:
            trace.append(f"Outcome value updated: {old_val} -> {new_val}")

        # 3. Check Node Count
        diff = len(old_pred.get("nodes", [])) - len(new_pred.get("nodes", []))
        if diff > 0:
            trace.append(f"Removed {diff} nodes.")
        elif diff < 0:
            trace.append(f"Added {abs(diff)} nodes.")

        if not trace:
            trace.append("Structure adjusted (edges/attributes changed).")

        return trace
# =============================================================================
# 3. Main run script
# =============================================================================
if __name__ == "__main__":
    # Simulated Module V output (input data)
    v_output_data = {
        "module": "Module_V",
        "instance_id": "JPM_2023:cc:000000",
        "status": "audit_failed",
        "prediction": {
            "claim_id": "JPM_2023:claim:000000",
            "nodes": [
                {"node_id": "n_metric", "type": "Metric", "text": "Scope 1+2 emissions", "attrs": {"unit": "tCO2e"}},
                {"node_id": "n_outcome", "type": "Outcome", "text": "reduced by 10%",
                 "attrs": {"direction": "decrease", "value": 10, "unit": "percent", "year": 2023}},
                {"node_id": "n_def_s1", "type": "Definition", "text": "Scope 1 Definition",
                 "attrs": {"type": "standard"}},
                {"node_id": "n_def_s2", "type": "Definition", "text": "Scope 2 Definition",
                 "attrs": {"type": "standard"}}
            ],
            "edges": [
                {"src": "n_outcome", "dst": "n_metric", "edge_type": "measures"},
                {"src": "n_def_s1", "dst": "n_metric", "edge_type": "defines"},
                {"src": "n_def_s2", "dst": "n_metric", "edge_type": "defines"}
            ],
            "evidence_sets": []  # simplified/omitted
        },
        "validator_report": {
            "v0_gate": {"pass": True},
            "v1_gate": {"pass": False,
                        "errors": ["NUMERIC_MISMATCH: Evidence implies 12.9%, Claim says 10.0%. Diff: 2.86"]},
            "v2_gate": {"pass": False, "errors": [
                "SCOPE_MISMATCH: Metric 'Scope 1+2 emissions' requires Scope 2, but evidence headers only contain Scope 1."]}
        }
    }

    # 1. Run SR-D
    sr_d = ModuleSR_D()
    d_result = sr_d.run(v_output_data)

    # 2. Run SR-L (if SR-D did not fully repair it)
    if d_result["status"] != "repaired":
        sr_l = ModuleSR_L()
        final_result = sr_l.run(d_result)

        print("\n=== Final Repaired Output ===")
        print(json.dumps(final_result, indent=2))
