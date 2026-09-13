import json
import sys
import time
import os
import logging
from typing import List, Dict, Any, Set



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Use relative imports
# ============================================================
# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Relative imports
from llm.g_llm_client import create_llm_client
from llm.config_loader import get_config_loader
# ============================================================


class GenerationParseError(Exception):
    """Raised when the LLM response cannot be parsed into a valid graph
    (missing 'nodes' key, or an empty/falsy parsed_json). Carries the
    truncated raw response content for diagnosis."""

    def __init__(self, message: str, raw_content: Any = None):
        super().__init__(message)
        self.raw_content = raw_content


class GenerationAPIError(Exception):
    """Raised when the underlying LLM API call itself raises an exception
    (network error, auth error, rate limit, etc.). Wraps the original
    exception for diagnosis."""

    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception


class ModuleG_Generator:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Module G
        """
        # 1. Load configuration
        if config is None:
            config_loader = get_config_loader()

            # A. Get module config (e.g.: {'model': '<a model key from configs/models.yaml>'})
            module_conf = config_loader.get_module_config("module_g")

            # B. Get the referenced model ID (note: the yaml key is 'model', not 'model_name')
            model_ref = module_conf.get("model")

            # C. Get that model's detailed configuration
            # Access config_loader's internal dict directly to get the models section
            all_models = config_loader.config.get("models", {})
            model_details = all_models.get(model_ref, {})

            # if not model_details:
            #     logger.warning(f"Model reference '{model_ref}' not found in 'models' section! Using Mock.")

            # D. Merge configuration: model config (provider, api_key) + module config
            self.config = {**module_conf, **model_details}

            logger.info(f"Loaded config for module_g. Ref: {model_ref} -> Provider: {self.config.get('provider')}")
        else:
            self.config = config

        # 2. Extract parameters (self.config now has provider set)
        provider = self.config.get("provider", "mock")
        model_name = self.config.get("model_name")  # the specific API model name, as configured in configs/models.yaml

        # 3. Initialize the client
        self.llm_client = create_llm_client(provider, model_name, self.config)
        logger.info(f"Initialized LLM Client: {provider} / {model_name}")

        # 4. Mock-fallback-on-error gate (default OFF — fail closed).
        # Only when explicitly set True (via config["allow_mock_fallback_on_error"],
        # threaded from run_g_baseline.py's --allow-mock-fallback-on-error flag)
        # does a parse failure or API exception route into the mock generator.
        # Default behavior is to raise GenerationParseError/GenerationAPIError.
        self.allow_mock_fallback_on_error = bool(
            self.config.get("allow_mock_fallback_on_error", False)
        )

    def run(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Single entry point called by the pipeline"""
        start_time = time.time()
        instance_id = instance_data.get("instance_id", "unknown")

        try:
            # 1. Extract the set of valid IDs (used for V0 validation)
            evidence_pool = instance_data.get('evidence_pool', [])
            valid_pool_ids = self._extract_valid_ids(evidence_pool)

            # 2. Build the prompt
            prompt = self._construct_prompt(instance_data)

            # ============================================================
            # Pass through all required arguments
            # ============================================================
            # 3. Call the LLM via the shared client (implemented below;
            # supersedes the earlier mock-only call kept commented out here)
            # prediction = self._call_llm_mock(prompt, evidence_pool, instance_data)
            prediction = self._call_llm(prompt, evidence_pool, instance_data)
            # ============================================================

            # 4. V0 gate check
            gate_res = self._v0_gate_check(prediction, valid_pool_ids)

            # 5. Construct the standard output
            result = {
                "module": "Module_G",
                "instance_id": instance_id,
                "status": "success" if gate_res['pass'] else "flagged",
                "prediction": prediction,
                "validator_report": {
                    "v0_gate": gate_res
                },
                "meta": {
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "model_used": self.config.get("model")
                }
            }

            # ============================================================
            # Debug logging
            # ============================================================
            logger.info(f"[{instance_id}] Module G completed. Prediction nodes: {len(prediction.get('nodes', []))}")
            # ============================================================

            return result

        except (GenerationParseError, GenerationAPIError):
            # Fail-closed generation failures propagate to the caller
            # (run_g_baseline.py's per-instance loop) so they are recorded
            # as an explicit status="generation_failed" record, never
            # silently absorbed here as a generic status="error".
            raise

        except Exception as e:
            logger.error(f"[{instance_id}] Module G Failed: {str(e)}")
            import traceback
            traceback.print_exc()

            # ============================================================
            # Return a standard-format error result
            # ============================================================
            return {
                "module": "Module_G",
                "instance_id": instance_id,
                "status": "error",
                "error_message": str(e),
                "prediction": {
                    "claim_id": None,
                    "doc_id": None,
                    "nodes": [],
                    "edges": [],
                    "evidence_sets": []
                },
                "validator_report": {
                    "v0_gate": {"pass": False, "errors": [str(e)], "cited_count": 0}
                },
                "meta": {
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "error": str(e)
                }
            }

    def _extract_valid_ids(self, evidence_pool: List[Dict]) -> Set[str]:
        return set(item['evid_id'] for item in evidence_pool if item.get('evid_id') is not None)

    def _construct_prompt(self, instance: Dict) -> str:
        claim = instance.get('canonical_text', '')
        pool = instance.get('evidence_pool', [])

        # Partition by type for clearer inductive bias
        text_items  = [i for i in pool if i.get('evid_id') and i.get('type') == 'text']
        table_items = [i for i in pool if i.get('evid_id') and i.get('type') == 'table_cell']
        def_items   = [i for i in pool if i.get('evid_id') and i.get('type') == 'definition']

        prompt = f"CLAIM: {claim}\n"

        if text_items:
            prompt += "\nTEXT EVIDENCE:\n"
            for item in text_items:
                prompt += f"[{item['evid_id']}] {item.get('content', '')}\n"

        if table_items:
            prompt += "\nTABLE EVIDENCE:\n"
            for item in table_items:
                prompt += f"[{item['evid_id']}] {item.get('content', '')}\n"

        if def_items:
            prompt += "\nDEFINITION EVIDENCE:\n"
            for item in def_items:
                prompt += f"[{item['evid_id']}] {item.get('content', '')}\n"

        # Check if claim is numeric to trigger targeted table citation hint
        import re
        claim_text = instance.get('canonical_text', '')
        is_numeric_claim = bool(re.search(r'\d+\s*%|\$\s*\d|\d+\s*(tonnes|MWh|GWh|tCO2|MW|kWh|GJ|MT)', claim_text, re.IGNORECASE))
        table_ids = [i['evid_id'] for i in pool if i.get('type') == 'table_cell']
        numeric_table_hint = (
            f"\n\nNUMERIC CLAIM DETECTED: This claim contains specific numbers/percentages. "
            f"TABLE EVIDENCE has {len(table_ids)} items available. "
            "For the OUTCOME node: you MUST include a table_cell link if any TABLE EVIDENCE item "
            "directly shows the specific value or percentage stated in the claim. "
            "Keep text links for other nodes as appropriate."
        ) if (is_numeric_claim and table_ids) else ""

        prompt += (
            f"{numeric_table_hint}"
            "\n\nIMPORTANT: Before writing the JSON, mentally identify for EACH node "
            "the single most directly supporting evid_id. That evid_id must appear first "
            "in that node's evidence_set links."
        )

        return prompt

    def _call_llm(self, prompt: str, evidence_pool: List[Dict], instance_data: Dict) -> Dict:
        """
        Call the LLM to generate the graph structure (supports Mock / real LLM)
        """
        instance_id = instance_data.get("instance_id", "UNKNOWN")

        # 1. Build the system prompt
        system_prompt = """You are an ESG auditor assistant.
    Given a claim and evidence pool, generate a structured graph following the EAGC schema.

    Rules:
    1. Only use evid_id from the evidence_pool (closed-world constraint)
    2. If information is missing, use null for numeric fields or "unknown" for enum fields
    3. Output valid JSON with keys: claim_id, doc_id, nodes, edges, evidence_sets
    4. Use "node_type" (not "type") for the node type field
    5. For numeric or year-specific claims, consider TABLE evidence first when it directly contains the reported value; otherwise use TEXT evidence.
    6. Use DEFINITION evidence when the claim depends on methodology, boundary, or term definition.
    7. Always cite the specific evid_id exactly as shown in the pool (do not invent IDs).
    8. For each node's evidence_set, the FIRST link must be the single most directly supporting evidence — the item that literally contains the claim's specific value, term, or statement (not a paraphrase or background sentence).
    9. For numeric claims (containing specific numbers, %, $, or year+value pairs): the first link MUST be a table_cell that explicitly states that exact value; if no table_cell matches, use the text passage that directly quotes it. Do NOT default to a generic text context passage when a more specific item exists.
    10. For definitional claims (referencing a methodology, standard, or term): the first link MUST be a DEFINITION evidence item. Use TEXT only if no definition item covers the term.
    11. TABLE EVIDENCE FOR OUTCOME NODE: For the Outcome node specifically — if the claim states a specific numeric value, percentage, or measurement (contains %, $, tonnes, MWh, or a number with a unit), the Outcome node's evidence_set MUST include at least one table_cell link in addition to any text links. Search TABLE EVIDENCE for the item whose numeric content directly shows that specific value. Do NOT force table citations on Metric, Target, Definition, or Action nodes unless a table item clearly defines that node's content directly.

    Node types: Metric, Outcome, Target, Definition, Action
    Edge types (field name: edge_type): supports, defines, measures, associated_with, contradicts
    Evidence relations: numeric_support, definition_support, qualitative_support

    Node attrs schema (include ONLY these fields inside attrs):
    - Target: {"direction": "reduce|increase|maintain|neutral|unknown", "value": number|null, "unit": string|null, "target_year": int|null, "baseline_year": int|null}
    - Outcome: {"year": int|null, "value": number|null, "value_type": "absolute|delta|ratio|boolean|ordinal|unknown", "unit": string|null, "baseline_year": int|null, "polarity": "improved|worsened|neutral|unknown"}
    - Metric: {"metric_name": string, "scope": string|null, "unit": string|null, "denominator": string|null}
    - Definition: {"definition_type": string, "standard": string|null, "scope_covered": string|null, "notes": string|null}
    - Action: {"status": "implemented|planning|indeterminate", "action_category": string, "time_anchor_year": int|null}"""

        # 2. Build the user prompt (reuses existing logic)
        user_prompt = self._construct_prompt(instance_data)

        # Append the schema description
        user_prompt += "\n\nOutput the graph as JSON with this structure:\n"
        user_prompt += json.dumps({
            "claim_id": "string",
            "doc_id": "string",
            "nodes": [{"node_id": "string", "node_type": "Metric|Outcome|Target|Definition|Action", "text": "string", "attrs": {}}],
            "edges": [{"src": "string", "dst": "string", "edge_type": "string"}],
            "evidence_sets": [{
                "set_id": "string",
                "attached_to_node": "string",
                "links": [{"evid_id": "string", "relation": "string"}]
            }]
        }, indent=2)

        # 3. Call the LLM
        try:
            response = self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format="json_object",
                temperature=self.config.get("temperature", 0.0),
                max_tokens=self.config.get("max_tokens", 4096)
            )
        except Exception as e:
            logger.error(f"[{instance_id}] LLM call failed: {e}")
            if self.allow_mock_fallback_on_error:
                # Explicit opt-in only (--allow-mock-fallback-on-error). Never
                # the default path — see class docstrings above.
                return self._fallback_mock_generation(prompt, evidence_pool, instance_data)
            raise GenerationAPIError(
                f"[{instance_id}] LLM API call failed: {type(e).__name__}: {e}",
                original_exception=e,
            ) from e

        # 4. Extract the result
        prediction = response.get("parsed_json")
        if not prediction or "nodes" not in prediction:
            truncated = str(response.get('content', ''))[:200]
            logger.error(f"[{instance_id}] Invalid LLM response: {truncated}")
            if self.allow_mock_fallback_on_error:
                # Explicit opt-in only (--allow-mock-fallback-on-error). Never
                # the default path — see class docstrings above.
                return self._fallback_mock_generation(prompt, evidence_pool, instance_data)
            raise GenerationParseError(
                f"[{instance_id}] LLM response missing 'nodes' key or empty "
                f"parsed_json. Truncated content: {truncated}",
                raw_content=response.get('content'),
            )

        # 5. Log meta information
        logger.info(f"[{instance_id}] LLM call successful: "
                    f"{response['usage']['total_tokens']} tokens, "
                    f"{response['latency_ms']:.0f}ms, "
                    f"${response['cost_usd']:.4f}")

        return prediction

    def _fallback_mock_generation(self, prompt: str, evidence_pool: List[Dict], instance_data: Dict) -> Dict:
    #     """Fallback path used when the LLM call fails (reuses the original mock code)."""
    #
        return self._call_llm_mock(prompt, evidence_pool, instance_data)

    def _call_llm_mock(self, prompt: str, evidence_pool: List[Dict], instance_data: Dict) -> Dict:
        """
        Mock LLM response.

        Behavior:
        1. Dynamically selects evidence IDs from evidence_pool.
        2. Dynamically generates the graph structure based on the claim.
        3. Ensures evidence_sets are generated correctly.
        """
        # 1. Extract instance metadata
        instance_id = instance_data.get("instance_id", "UNKNOWN")
        doc_id = instance_id.split(':')[0] if ':' in instance_id else "UNKNOWN"
        claim_id = instance_data.get("canonical_claim_id", f"{doc_id}:claim:000000")
        canonical_text = instance_data.get("canonical_text", "")

        # ============================================================
        # Debug logging
        # ============================================================
        logger.info(f"[{instance_id}] === Module G: _call_llm_mock ===")
        logger.info(f"[{instance_id}] Canonical text: {canonical_text}")
        logger.info(f"[{instance_id}] Evidence pool size: {len(evidence_pool)}")
        # ============================================================

        # 2. Intelligently select evidence IDs from evidence_pool
        table_cells = []
        definitions = []
        sentences = []

        for item in evidence_pool:
            evid_id = item.get('evid_id')
            if not evid_id:  # Skip evidence with evid_id == None
                continue

            evid_type = item.get('type', '')
            if evid_type == 'table_cell':
                table_cells.append(evid_id)
            elif evid_type == 'definition':
                definitions.append(evid_id)
            elif evid_type in ['sentence', 'textual']:
                sentences.append(evid_id)

        # ============================================================
        # Debug logging
        # ============================================================
        logger.info(f"[{instance_id}] Classified evidence:")
        logger.info(f"  - Table cells: {table_cells}")
        logger.info(f"  - Definitions: {definitions}")
        logger.info(f"  - Sentences: {sentences}")
        # ============================================================

        # Select evidence IDs
        numeric_support_ids = table_cells[:2] if table_cells else []
        definition_support_ids = definitions[:2] if definitions else []

        # If numeric evidence is insufficient, fall back to sentences
        if len(numeric_support_ids) < 2 and sentences:
            needed = 2 - len(numeric_support_ids)
            numeric_support_ids.extend(sentences[:needed])

        # If there is no definition evidence, fall back to sentences
        if not definition_support_ids and sentences:
            definition_support_ids = sentences[:1]

        logger.info(f"[{instance_id}] Selected evidence IDs:")
        logger.info(f"  - Numeric support: {numeric_support_ids}")
        logger.info(f"  - Definition support: {definition_support_ids}")

        # 3. Build the graph structure
        nodes = []
        edges = []
        evidence_sets = []

        # ============================================================
        # Dynamically generate nodes based on canonical_text
        # ============================================================
        text_lower = canonical_text.lower()

        # Detect whether a numeric claim is present (e.g. "reduced by 10%")
        has_numeric_claim = any(
            keyword in text_lower for keyword in ['reduced', 'increased', 'decreased', '%', 'percent'])

        # Detect whether a metric is present (e.g. "emissions", "carbon")
        has_metric = any(keyword in text_lower for keyword in ['emission', 'carbon', 'ghg', 'scope'])

        # Detect whether a target/action is present (e.g. "committed", "target", "goal")
        has_target = any(keyword in text_lower for keyword in ['committed', 'target', 'goal', 'achieve', 'plan'])

        # Build nodes based on the detection results
        if has_metric:
            # Extract the metric name
            if 'scope 1+2' in text_lower or 'scope 1 and 2' in text_lower:
                metric_text = "Scope 1+2 emissions"
            elif 'scope 1' in text_lower:
                metric_text = "Scope 1 emissions"
            elif 'scope 2' in text_lower:
                metric_text = "Scope 2 emissions"
            elif 'carbon' in text_lower:
                metric_text = "Carbon neutrality"
            else:
                metric_text = "GHG emissions"

            nodes.append({
                "node_id": "n_metric",
                "type": "Metric",
                "text": metric_text,
                "attrs": {"unit": "tCO2e"}
            })

            # ============================================================
            # Add a definition evidence set for the Metric node
            # ============================================================
            if definition_support_ids:
                nodes.append({
                    "node_id": "n_def",
                    "type": "Definition",
                    "text": f"{metric_text} Definition",
                    "attrs": {"type": "standard"}
                })
                edges.append({
                    "src": "n_def",
                    "dst": "n_metric",
                    "edge_type": "defines"
                })

                # Add the definition evidence set
                evidence_sets.append({
                    "set_id": "es_def",
                    "attached_to_node": "n_metric",
                    "links": [{"evid_id": eid, "relation": "definition_support"}
                              for eid in definition_support_ids]
                })
                logger.info(f"[{instance_id}] Added definition evidence set with {len(definition_support_ids)} links")

        if has_numeric_claim:
            # Extract the numeric value
            import re
            value_match = re.search(r'(\d+(?:\.\d+)?)\s*%', canonical_text)
            value = float(value_match.group(1)) if value_match else 10.0

            # Extract the direction
            if any(word in text_lower for word in ['reduced', 'decreased', 'decline']):
                direction = "decrease"
            elif any(word in text_lower for word in ['increased', 'grew', 'rise']):
                direction = "increase"
            else:
                direction = "change"

            # Extract the year
            year_match = re.search(r'(20\d{2})', canonical_text)
            year = int(year_match.group(1)) if year_match else 2023

            nodes.append({
                "node_id": "n_outcome",
                "type": "Outcome",
                "text": f"{direction} by {value}%",
                "attrs": {
                    "direction": direction,
                    "value": value,
                    "unit": "percent",
                    "year": year
                }
            })

            if has_metric:
                edges.append({
                    "src": "n_outcome",
                    "dst": "n_metric",
                    "edge_type": "measures"
                })

            # ============================================================
            # Add a numeric evidence set for the Outcome node
            # ============================================================
            if numeric_support_ids:
                evidence_sets.append({
                    "set_id": "es_outcome",
                    "attached_to_node": "n_outcome",
                    "links": [{"evid_id": eid, "relation": "numeric_support"}
                              for eid in numeric_support_ids]
                })
                logger.info(f"[{instance_id}] Added numeric evidence set with {len(numeric_support_ids)} links")

        if has_target and not has_numeric_claim:
            # If this is a target/commitment claim rather than a numeric outcome
            nodes.append({
                "node_id": "n_target",
                "type": "Target",
                "text": canonical_text[:100],  # first 100 characters of the source text
                "attrs": {"type": "commitment"}
            })

            # Add a text evidence set (if sentence evidence is available)
            if sentences:
                evidence_sets.append({
                    "set_id": "es_target",
                    "attached_to_node": "n_target",
                    "links": [{"evid_id": sentences[0], "relation": "textual_support"}]
                })
                logger.info(f"[{instance_id}] Added target evidence set with 1 link")

        # ============================================================
        # If no nodes were generated, return the minimal structure
        # ============================================================
        if not nodes:
            logger.warning(f"[{instance_id}] Could not generate any nodes from claim text")
            nodes.append({
                "node_id": "n_claim",
                "type": "Claim",
                "text": canonical_text[:100],
                "attrs": {}
            })

            # If any evidence is available, add a generic evidence set
            if sentences:
                evidence_sets.append({
                    "set_id": "es_claim",
                    "attached_to_node": "n_claim",
                    "links": [{"evid_id": sentences[0], "relation": "textual_support"}]
                })

        # ============================================================
        # Final debug logging
        # ============================================================
        logger.info(f"[{instance_id}] Generated graph:")
        logger.info(f"  - Nodes: {len(nodes)}")
        logger.info(f"  - Edges: {len(edges)}")
        logger.info(f"  - Evidence sets: {len(evidence_sets)}")
        # ============================================================

        return {
            "claim_id": claim_id,
            "doc_id": doc_id,
            "nodes": nodes,
            "edges": edges,
            "evidence_sets": evidence_sets
        }

    def _v0_gate_check(self, prediction: Dict, valid_pool_ids: Set[str]) -> Dict:
        errors = []
        cited_ids = set()
        if not prediction: return {"pass": False, "errors": ["No prediction"]}

        for es in prediction.get("evidence_sets", []):
            for link in es.get("links", []):
                eid = link.get("evid_id")
                if eid:
                    cited_ids.add(eid)
                    if eid not in valid_pool_ids:
                        errors.append(f"Hallucination: {eid}")
        return {"pass": len(errors) == 0, "errors": errors, "cited_count": len(cited_ids)}


# ==========================================
# Main execution logic
# ==========================================
def main():
    # 1. Define paths
    base_dir = "quickstart"
    input_path = os.path.join(base_dir, "r1_output.json")
    output_path = os.path.join(base_dir, "g_output.json")

    # Ensure the directory exists (create it if missing, though it normally should exist)
    os.makedirs(base_dir, exist_ok=True)

    # 2. Read the R1 output data
    if not os.path.exists(input_path):
        # If the file does not exist, create a mock file for testing
        logger.warning(f"Input file not found at {input_path}. Creating mock data...")
        mock_data = {
            "instance_id": "JPM_2023:cc:000000",
            "canonical_claim_id": "JPM_2023:claim:000000",
            "doc_id": "JPM_2023",
            "canonical_text": "JPM reduced Scope 1+2 emissions by 10% in 2023.",
            "evidence_pool": [
                {"evid_id": "JPM_2023:cell:emissions:000:001", "type": "table_cell", "content": "35000",
                 "row_label": "2021", "col_header": "Scope 1 (tCO2e)", "table_title": "JPM GHG Emissions"},
                {"evid_id": "JPM_2023:cell:emissions:002:001", "type": "table_cell", "content": "30500",
                 "row_label": "2023", "col_header": "Scope 1 (tCO2e)", "table_title": "JPM GHG Emissions"},
                {"evid_id": "JPM_2023:def:000000", "type": "definition", "content": "Scope 1: Direct emissions..."},
                {"evid_id": "JPM_2023:def:000001", "type": "definition", "content": "Scope 2: Indirect emissions..."},
                {"evid_id": None, "type": "table_row_view", "content": "Context Row...",
                 "provenance_note": "Context only."}
            ]
        }
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(mock_data, f, indent=2)
        logger.info(f"Created mock input file at {input_path}")

    logger.info(f"Reading input from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        # The input may be a single JSON object or JSONL (one object per line).
        # For simplicity, assume a single JSON object; if it is a list, loop over it.
        try:
            input_data = json.load(f)
            # If it is a list, take all entries and loop over them
            if isinstance(input_data, list):
                instances = input_data
            else:
                instances = [input_data]
        except json.JSONDecodeError:
            logger.error("Invalid JSON file.")
            return

    # 3. Initialize Module G
    module_g = ModuleG_Generator(config={"model": "mock-model"})

    results = []

    # 4. Batch process
    for instance in instances:
        logger.info(f"Processing instance: {instance.get('instance_id')}")
        output = module_g.run(instance)
        results.append(output)

    # 5. Save the results
    logger.info(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        # If there is only one result, save it as an object; otherwise save a list
        if len(results) == 1:
            json.dump(results[0], f, indent=2)
        else:
            json.dump(results, f, indent=2)

    logger.info("✅ Done!")


if __name__ == "__main__":
    main()
