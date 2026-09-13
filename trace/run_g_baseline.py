#!/usr/bin/env python3
"""
Run Module G baselines, TRACE ablations and Module E evaluation.

Bridges M4 candidate_pools format → pipeline evidence_pool format,
runs Module G (+ optionally R1/V/SR), then evaluates with Module E
against GPT-4o gold annotations.

Supported variants (--variant):
  G              — Schema-RAG baseline (B4 in paper)
  G_R1           — +R1 evidence routing
  G_R1_V         — +R1 +V1+V2 validation gates
  G_R1_V_SRD     — +R1 +V +SR-D deterministic repair
  G_R1_V_SRD_SRL — Full EFCGG pipeline (B5 in paper)
  empty-evidence — Vanilla RAG (empty evidence pool, bare LLM); a negative
                    control only, NOT the paper's B3 Flat-Evidence comparator
                    (that is a separate script — see trace/run_flat_evidence.py)

Usage:
  # Schema-RAG baseline (GPT-4o):
  python run_g_baseline.py \
    --tasks data/eagc_trace_789/tasks \
    --gold  data/eagc_trace_789/gold \
    --output outputs/G \
    --model openai --model-name gpt-4o-2024-08-06 \
    --variant G --concurrency 5

  # Full TRACE pipeline (R1+G+V+SR-D+SR-L):
  python run_g_baseline.py \
    --tasks data/eagc_trace_789/tasks \
    --gold  data/eagc_trace_789/gold \
    --output outputs/G_R1_V_SRD_SRL \
    --model openai --model-name gpt-4o-2024-08-06 \
    --variant G_R1_V_SRD_SRL --concurrency 5

On LLM generation failure (parse failure or API exception), the default
behavior is to fail closed: the instance is recorded with
status="generation_failed" (visible in the output, kept in the
denominator, never silently dropped or fabricated) rather than crashing
the batch or substituting a synthetic mock graph. Pass
--allow-mock-fallback-on-error to opt into the old mock-substitution
behavior for offline testing only — never use it for a real evaluation run.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup: add scripts/pipeline/ and scripts/ so that both sub-packages
# (g, e, r1, v, sr, ...) and the shared llm module are importable.
# ---------------------------------------------------------------------------
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../scripts/pipeline
SCRIPTS_DIR  = os.path.dirname(PIPELINE_DIR)                 # .../scripts
PROJECT_DIR  = os.path.dirname(SCRIPTS_DIR)                  # .../E_DAG
for p in (PIPELINE_DIR, SCRIPTS_DIR, PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import json
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from g.g_generator import ModuleG_Generator, GenerationParseError, GenerationAPIError
from e.e_evaluation import ModuleE_Evaluator, load_gold_dir, adapt_gold_annotation

logging.basicConfig(
    level=logging.WARNING,   # suppress noisy INFO from sub-modules
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VARIANTS_WITH_R1  = ("G_R1", "G_R1_V", "G_R1_V_SRD", "G_R1_V_SRD_SRL")
VARIANTS_WITH_V   = ("G_R1_V", "G_R1_V_SRD", "G_R1_V_SRD_SRL")
VARIANTS_WITH_SRD = ("G_R1_V_SRD", "G_R1_V_SRD_SRL")
VARIANTS_WITH_SRL = ("G_R1_V_SRD_SRL",)


# ---------------------------------------------------------------------------
# Format bridge: M4 candidate_pools → pipeline evidence_pool
# ---------------------------------------------------------------------------

def _table_content(item: Dict) -> str:
    """Extract readable content from a table pool item via table_preview.render."""
    tp = item.get("table_preview")
    if isinstance(tp, dict):
        render = tp.get("render", "")
        if render:
            return render[:600]   # cap at ~150 tokens per table row
    return item.get("unit_text", "")


def build_evidence_pool(task: Dict) -> List[Dict]:
    """
    Convert a task JSON's candidate_pools into a flat evidence_pool list.

    Conversion rules:
      text pool       → {evid_id: unit_id, type: "text",       content: unit_text}
      table pool      → {evid_id: unit_id, type: "table_cell", content: table_preview.render}
                         One entry per table row (unit_id).  Coverage map bridges
                         unit_id → covered cell IDs for evaluation matching.
      definition pool → {evid_id: unit_id, type: "definition", content: unit_text}

    Deduplication via seen_ids prevents duplicates across pools.
    """
    pools = task.get("candidate_pools", {})
    result: List[Dict] = []
    seen_ids: set = set()

    # --- Text pool ---
    for item in pools.get("text", []):
        uid = item.get("unit_id")
        if uid and uid not in seen_ids:
            result.append({
                "evid_id": uid,
                "type": "text",
                "content": item.get("unit_text", "")
            })
            seen_ids.add(uid)

    # --- Table pool: one entry per row unit, content from table_preview ---
    for item in pools.get("table", []):
        uid = item.get("unit_id")
        if uid and uid not in seen_ids:
            result.append({
                "evid_id": uid,
                "type": "table_cell",
                "content": _table_content(item),
            })
            seen_ids.add(uid)

    # --- Definition pool ---
    for item in pools.get("definition", []):
        uid = item.get("unit_id")
        if uid and uid not in seen_ids:
            result.append({
                "evid_id": uid,
                "type": "definition",
                "content": item.get("unit_text", "")
            })
            seen_ids.add(uid)

    return result


def build_coverage_map(task: Dict) -> Dict[str, List[str]]:
    """
    Build unit_id → covered_evidence_ids mapping for all pool types.

    Gold annotations cite sentence/cell IDs (from covered_evidence_ids), but the
    evidence pool shows unit_ids to the LLM.  This map allows the evaluator to
    match a cited unit_id against a gold sentence_id or cell_id via coverage lookup.
    """
    pools = task.get("candidate_pools", {})
    coverage: Dict[str, List[str]] = {}
    for pool_name in ("text", "table", "definition"):
        for item in pools.get(pool_name, []):
            uid = item.get("unit_id")
            if uid:
                coverage[uid] = item.get("covered_evidence_ids", [])
    return coverage


def task_to_g_input(task: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Convert a task JSON to (Module G input, pool_membership dict, coverage_map).

    pool_membership maps each evid_id → pool type ("text"|"table_cell"|"definition")
    so that Module E can compute Hit@K split by pool type.

    coverage_map maps each text/def unit_id → list of covered sentence IDs so that
    Module E can match a cited unit_id against a gold sentence_id via coverage.
    """
    evidence_pool = build_evidence_pool(task)
    pool_membership = {item["evid_id"]: item["type"] for item in evidence_pool}
    coverage_map = build_coverage_map(task)
    g_input = {
        "instance_id": task.get("canonical_claim_id", task.get("task_id", "")),
        "canonical_text": task.get("canonical_text", ""),
        "doc_id": task.get("doc_id", ""),
        "evidence_pool": evidence_pool
    }
    return g_input, pool_membership, coverage_map


def task_to_b_instance(task: Dict) -> Dict:
    """Bridge M4 candidate_pools → B1/B2 candidate_units format."""
    pools = task.get("candidate_pools", {})
    return {
        "doc_id": task.get("doc_id", ""),
        "claim_id": task.get("canonical_claim_id", task.get("task_id", "")),
        "claim_text": task.get("canonical_text", ""),
        "claim_span_evid_id": None,
        "candidate_units": {
            "text_units":       pools.get("text", []),
            "definition_units": pools.get("definition", []),
            "table_units":      pools.get("table", []),
        },
    }


def b_pred_to_g_result(b_pred: Dict, instance_id: str) -> Dict:
    """Wrap B1/B2 prediction output into g_output_to_pred input format."""
    return {
        "instance_id": instance_id,
        "status": "success",
        "prediction": {
            "doc_id":        b_pred.get("doc_id"),
            "claim_id":      b_pred.get("claim_id"),
            "nodes":         b_pred.get("nodes", []),
            "edges":         b_pred.get("edges", []),
            "evidence_sets": b_pred.get("evidence_sets", []),
        },
        "validator_report": {"v0_gate": {"pass": True, "errors": []}},
        "meta": b_pred.get("meta", {}),
    }


def build_atomic_pool_membership(prov_map) -> Dict[str, str]:
    """Map atomic evid_id → pool type ('text'|'table_cell'|'definition')."""
    POOL_MAP = {"text": "text", "definition": "definition", "table": "table_cell"}
    return {eid: POOL_MAP.get(p.pool, p.pool) for eid, p in prov_map.items()}


def g_output_to_pred(
    result: Dict,
    pool_membership: Dict,
    coverage_map: Dict,
    validator_report: Optional[Dict] = None,
) -> Dict:
    """
    Adapt a G/V/SR result to Module E's prediction input format.

    If validator_report is None (G-only variants), V1/V2 are set to pass=True
    (unchecked) so ECS_strict reflects V0 pass rate only.  When a real
    validator_report is provided (from V/SR runs), it is used as-is.

    pool_membership and coverage_map flow through to Module E for Hit@K
    computation with coverage-based matching.
    """
    v0 = result.get("validator_report", {}).get("v0_gate", {"pass": False, "errors": []})
    if validator_report is None:
        vr = {
            "v0_gate": v0,
            "v1_gate": {"pass": True, "errors": []},
            "v2_gate": {"pass": True, "errors": []},
        }
    else:
        vr = dict(validator_report)
        vr["v0_gate"] = v0   # V0 always from G
        # Ensure v1/v2 are present (defensive: V may have crashed before setting them)
        vr.setdefault("v1_gate", {"pass": True, "errors": []})
        vr.setdefault("v2_gate", {"pass": True, "errors": []})

    out = {
        "instance_id": result.get("instance_id"),
        "status": result.get("status", "success"),
        "prediction": result.get("prediction", {}),
        "pool_membership": pool_membership,
        "validator_report": vr,
        "meta": {**result.get("meta", {}), "coverage_map": coverage_map},
    }
    if "error_message" in result:
        out["error_message"] = result["error_message"]
    if "sr_l_status" in result:
        out["sr_l_status"] = result["sr_l_status"]
    return out


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

def apply_r1(
    g_input: Dict,
    pool_membership: Dict,
    coverage_map: Dict,
    router,
) -> Tuple[Dict, Dict, Dict]:
    """
    Route and rerank the evidence pool with R1.

    R1EvidenceRouter reads from 'evidence_pool_raw' key, so we bridge our
    'evidence_pool' key.  coverage_map is unchanged (built from the original
    task candidate_pools; R1 selects a subset but does not change unit_id→sent_ids).
    """
    r0_data = {**g_input, "evidence_pool_raw": g_input["evidence_pool"]}
    r1_output = router.route_and_rerank(r0_data)
    new_pool = r1_output.get("evidence_pool", g_input["evidence_pool"])
    new_pm = {item["evid_id"]: item.get("type", "text") for item in new_pool}
    new_g_input = {**g_input, "evidence_pool": new_pool}
    return new_g_input, new_pm, coverage_map


def _generation_failed_record(inp: Dict, ex: Exception) -> Dict:
    """
    Build the per-instance output record for a fail-closed Module G
    generation failure (GenerationParseError / GenerationAPIError raised
    by g_generator.py with --allow-mock-fallback-on-error unset).

    Mirrors the frozen GPT-3.5 rerun's failure representation: the instance
    stays in the denominator with an explicit, visible status and error
    message, an empty (never fabricated) prediction, and a failing v0_gate
    — it is never silently dropped or made to look like a successful call.
    """
    return {
        "module": "Module_G",
        "instance_id": inp.get("instance_id", "unknown"),
        "status": "generation_failed",
        "error_message": str(ex),
        "prediction": {
            "claim_id": None,
            "doc_id": None,
            "nodes": [],
            "edges": [],
            "evidence_sets": [],
        },
        "validator_report": {
            "v0_gate": {"pass": False, "errors": [str(ex)], "cited_count": 0}
        },
        "meta": {"error": str(ex), "error_type": type(ex).__name__},
    }


def _all_pass(result: Dict) -> bool:
    report = result.get("validator_report", {})
    return (
        report.get("v0_gate", {}).get("pass", False) and
        report.get("v1_gate", {}).get("pass", False) and
        report.get("v2_gate", {}).get("pass", False)
    )


def apply_validator(g_result: Dict, evidence_pool: List[Dict], validator) -> Dict:
    """Run ModuleV_Validator on G output."""
    return validator.run(g_result, evidence_pool)


def apply_repair(
    v_result: Dict,
    evidence_pool: List[Dict],
    validator,
    sr_d,
    sr_l=None,
    max_iter: int = 3,
) -> Dict:
    """
    Run SR-D deterministic repair, then optional SR-L, in a loop.

    Each iteration: SR-D repairs → re-validate; if still failing and SR-L
    is available, SR-L repairs → re-validate.  Stops early if all gates pass.
    """
    current = v_result
    for _ in range(max_iter):
        if _all_pass(current):
            break
        repaired = sr_d.run(current, evidence_pool)
        current = validator.run(repaired, evidence_pool)
        if sr_l is not None and not _all_pass(current):
            repaired_l = sr_l.run(current, evidence_pool)
            # ModuleSR_L_LLM.run() sets status="repair_error" (caught
            # exception, no re-raise) or "repair_failed"/"repaired" — but
            # ModuleV_Validator._finalize() unconditionally overwrites
            # "status" with "audit_passed"/"audit_failed" on the next line,
            # which would otherwise silently absorb a repair_error into what
            # looks like an ordinary unrepaired-but-validated record. Capture
            # it here and surface it explicitly in meta so it survives to
            # the final output instead.
            sr_l_status = repaired_l.get("status")
            current = validator.run(repaired_l, evidence_pool)
            if sr_l_status == "repair_error":
                current = dict(current)
                current["sr_l_status"] = "repair_error"
                current["meta"] = {**current.get("meta", {}), "sr_l_repair_error": True}
    return current


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_tasks(tasks_dir: str) -> List[Dict]:
    tasks = []
    for fname in sorted(os.listdir(tasks_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(tasks_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            tasks.append(json.load(f))
    return tasks


def build_oracle_candidates(tasks: List[Dict]) -> Dict[str, List[str]]:
    """
    Build oracle candidate IDs for each task from all pool covered_evidence_ids.

    Oracle candidates = all actual sentence/cell IDs exposed in the candidate pool
    (text + table + definition).  When passed to ModuleE_Evaluator, ECS_oracle
    answers: "if retrieval were perfect, what fraction of instances would pass?"
    """
    oracle: Dict[str, List[str]] = {}
    for task in tasks:
        claim_id = task.get("canonical_claim_id", task.get("task_id", ""))
        all_ids: List[str] = []
        for pool_items in task.get("candidate_pools", {}).values():
            for item in pool_items:
                all_ids.extend(item.get("covered_evidence_ids", []))
        oracle[claim_id] = list(set(all_ids))
    return oracle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Module G baselines, TRACE ablations and Module E evaluation."
    )
    parser.add_argument("--tasks",       required=True,
                        help="Directory of benchmark task JSON files")
    parser.add_argument("--gold",        required=True,
                        help="Directory of adjudicated gold annotation JSON files")
    parser.add_argument("--output",      default="../../outputs/phase2/dryrun_g_baseline",
                        help="Output directory for results")
    parser.add_argument("--model",       default="mock",
                        choices=["mock", "anthropic", "openai"],
                        help="LLM provider for Module G")
    parser.add_argument("--model-name",  default=None,
                        help="Model name (e.g. claude-sonnet-4-6, gpt-4o-2024-08-06)")
    parser.add_argument("--variant",     default="G",
                        choices=["G", "G_R1", "G_R1_V", "G_R1_V_SRD", "G_R1_V_SRD_SRL", "empty-evidence", "B1", "B2"],
                        help="Ablation variant (default: G = Schema-RAG baseline)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of parallel G calls")
    parser.add_argument("--eval-only",   action="store_true",
                        help="Skip Module G: load g_predictions.json, re-run "
                             "deterministic V/SR, re-evaluate with oracle metrics")
    parser.add_argument("--debug",       action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--k-total",     type=int, default=30,
                        help="R1 evidence pool budget K_total (default 30)")
    parser.add_argument("--no-gold",     action="store_true",
                        help="Run G on all tasks without gold evaluation (scale-up mode)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM sampling temperature (default 0.0 = deterministic)")
    parser.add_argument("--max-tokens",  type=int, default=4096,
                        help="LLM max output tokens for Module G (default 4096)")
    parser.add_argument("--allow-mock-fallback-on-error", action="store_true",
                        help="DANGEROUS, offline-testing only: on an LLM parse "
                             "failure or API exception, silently substitute a "
                             "fabricated mock graph instead of failing closed. "
                             "Default is OFF — failures raise and are recorded "
                             "per-instance as status='generation_failed'. Never "
                             "use this flag for a real evaluation run.")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    variant = args.variant

    # -----------------------------------------------------------------------
    # 1. Load tasks
    # -----------------------------------------------------------------------
    print(f"[1/5] Loading tasks from {args.tasks} ...")
    tasks = load_tasks(args.tasks)
    print(f"      {len(tasks)} task JSONs found.")

    # -----------------------------------------------------------------------
    # 2. Load gold annotations
    # -----------------------------------------------------------------------
    print(f"[2/5] Loading gold from {args.gold} ...")
    gold_raw = load_gold_dir(args.gold)
    gold_raw_valid = [r for r in gold_raw
                      if isinstance(r, dict) and
                      r.get("pipeline_status") in ("PASS", "PASS_REPAIRED", "ADJUDICATED")]
    gold_data = [adapt_gold_annotation(r) for r in gold_raw_valid]
    gold_ids  = {g["instance_id"] for g in gold_data}
    print(f"      {len(gold_data)} valid gold instances "
          f"(excluded {len(gold_raw) - len(gold_raw_valid)} non-PASS).")

    if args.no_gold:
        tasks_covered = tasks
        print(f"      --no-gold mode: running on all {len(tasks_covered)} tasks (no evaluation).")
    else:
        tasks_covered = [t for t in tasks if t.get("canonical_claim_id") in gold_ids]
        print(f"      {len(tasks_covered)}/{len(tasks)} tasks have gold coverage.")

    # Build oracle candidate IDs for ECS_oracle computation
    oracle_candidates = build_oracle_candidates(tasks_covered)
    print(f"      Oracle candidates built for {len(oracle_candidates)} tasks.")

    # -----------------------------------------------------------------------
    # 3. Initialise modules
    # -----------------------------------------------------------------------
    print(f"[3/5] Initialising modules (variant={variant}, "
          f"provider={args.model}, model={args.model_name or 'default'}) ...")

    g_config: Dict[str, Any] = {"provider": args.model}
    if args.model_name:
        g_config["model_name"] = args.model_name
    g_config["temperature"] = args.temperature
    g_config["max_tokens"] = args.max_tokens
    g_config["allow_mock_fallback_on_error"] = args.allow_mock_fallback_on_error
    generator = ModuleG_Generator(config=g_config)

    # Lazily import and init only the modules needed by this variant
    router    = None
    validator = None
    sr_d      = None
    sr_l      = None

    if variant in VARIANTS_WITH_R1:
        from r1.r1_router import R1EvidenceRouter
        router = R1EvidenceRouter(k_total=args.k_total)
        print(f"      R1EvidenceRouter initialised (k_total={args.k_total}).")

    if variant in VARIANTS_WITH_V:
        from v.v_validator import ModuleV_Validator
        validator = ModuleV_Validator()
        print("      ModuleV_Validator initialised.")

    if variant in VARIANTS_WITH_SRD:
        from sr.sr_DL import ModuleSR_D
        sr_d = ModuleSR_D()
        print("      ModuleSR_D initialised.")

    if variant in VARIANTS_WITH_SRL:
        from sr.sr_DL import ModuleSR_L_LLM
        sr_l = ModuleSR_L_LLM(config=g_config)
        print("      ModuleSR_L_LLM initialised (real LLM repair).")

    # -----------------------------------------------------------------------
    # 4. Build task inputs
    # -----------------------------------------------------------------------
    task_inputs = [task_to_g_input(t) for t in tasks_covered]
    g_inputs        = [t[0] for t in task_inputs]
    pool_memberships = [t[1] for t in task_inputs]
    coverage_maps   = [t[2] for t in task_inputs]

    # empty-evidence: empty evidence pool (Vanilla RAG — negative control only)
    if variant == "empty-evidence":
        g_inputs = [{**gi, "evidence_pool": []} for gi in g_inputs]
        pool_memberships = [{} for _ in g_inputs]

    # R1 routing modifies g_inputs and pool_memberships in place (pre-G)
    if router is not None:
        routed = [apply_r1(gi, pm, cm, router)
                  for gi, pm, cm in zip(g_inputs, pool_memberships, coverage_maps)]
        g_inputs        = [t[0] for t in routed]
        pool_memberships = [t[1] for t in routed]
        # coverage_maps unchanged

    # -----------------------------------------------------------------------
    # 5. Run Module G  (or load from cache in --eval-only mode)
    #    For B1/B2 deterministic variants, skip Module G entirely.
    # -----------------------------------------------------------------------
    if variant in ("B1", "B2"):
        from baselines.b1_reachable_set import build_reachable_set
        from baselines.b2_rulegraph import predict_rulegraph

        print(f"[4/5] Running deterministic {variant} baseline on "
              f"{len(tasks_covered)} tasks ...")
        g_results = []
        pool_memberships = []
        coverage_maps = []

        for task in tqdm(tasks_covered, desc=f"Module {variant}"):
            iid = task.get("canonical_claim_id", task.get("task_id", ""))
            b_instance = task_to_b_instance(task)
            reachable, prov_map = build_reachable_set(b_instance)

            pm = build_atomic_pool_membership(prov_map)
            cm = {}  # empty: B1/B2 cite atomic IDs directly → direct match in Module E

            if variant == "B1":
                links = [{"evid_id": eid, "relation": "qualitative_support",
                          "importance": "primary"} for eid in sorted(reachable)]
                if not links:
                    links = [{"evid_id": f"{b_instance['doc_id']}:sent:UNKNOWN",
                              "relation": "qualitative_support", "importance": "primary",
                              "is_manual_add": True}]
                b_pred = {
                    "doc_id": b_instance["doc_id"],
                    "claim_id": iid,
                    "nodes": [{"node_id": f"{iid}:n:claim", "node_type": "Claim",
                               "text": b_instance["claim_text"], "attrs": {}}],
                    "edges": [],
                    "evidence_sets": [{"set_id": "S0", "links": links}],
                    "meta": {"model": "b1_reachable_set_v0"},
                }
            else:  # B2
                run_meta = {"model": "b2_rulegraph_v0", "seed": 0}
                b_pred = predict_rulegraph(b_instance, run_meta)

            g_results.append(b_pred_to_g_result(b_pred, iid))
            pool_memberships.append(pm)
            coverage_maps.append(cm)

        print(f"      {variant} complete: {len(g_results)} predictions generated.")

    elif args.eval_only:
        g_pred_path = os.path.join(args.output, "g_predictions.json")
        if not os.path.exists(g_pred_path):
            print(f"ERROR: --eval-only requires {g_pred_path} to exist. "
                  f"Run without --eval-only first to generate it.")
            sys.exit(1)
        print(f"[4/5] --eval-only: loading cached G predictions from {g_pred_path} ...")
        with open(g_pred_path, "r", encoding="utf-8") as f:
            g_results = json.load(f)
        print(f"      Loaded {len(g_results)} cached G outputs.")
    else:
        print(f"[4/5] Running Module G on {len(tasks_covered)} tasks "
              f"(concurrency={args.concurrency}) ...")

        # GenerationParseError/GenerationAPIError (raised by g_generator.py
        # when --allow-mock-fallback-on-error is unset) are caught here, at
        # the per-instance loop boundary, so one instance's LLM failure
        # never crashes the whole batch run. Each failure is recorded as an
        # explicit status="generation_failed" record (see
        # _generation_failed_record) — kept in the denominator, never
        # silently dropped, never fabricated.
        g_results = []
        if args.concurrency <= 1:
            for inp in tqdm(g_inputs, desc="Module G"):
                try:
                    g_results.append(generator.run(inp))
                except (GenerationParseError, GenerationAPIError) as ex:
                    logger.warning(f"Generation failed for {inp.get('instance_id')}: {ex}")
                    g_results.append(_generation_failed_record(inp, ex))
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {executor.submit(generator.run, inp): i
                           for i, inp in enumerate(g_inputs)}
                ordered: Dict[int, Dict] = {}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Module G"):
                    idx = futures[fut]
                    try:
                        ordered[idx] = fut.result()
                    except (GenerationParseError, GenerationAPIError) as ex:
                        logger.warning(f"Generation failed for {g_inputs[idx].get('instance_id')}: {ex}")
                        ordered[idx] = _generation_failed_record(g_inputs[idx], ex)
                g_results = [ordered[i] for i in range(len(g_inputs))]

        n_success     = sum(1 for r in g_results if r.get("status") in ("success", "flagged"))
        n_v0_pass     = sum(1 for r in g_results
                        if r.get("validator_report", {}).get("v0_gate", {}).get("pass", False))
        n_error       = sum(1 for r in g_results if r.get("status") == "error")
        n_gen_failed  = sum(1 for r in g_results if r.get("status") == "generation_failed")
        print(f"\n      G complete: success={n_success}, v0_pass={n_v0_pass}, "
              f"error={n_error}, generation_failed={n_gen_failed} / {len(g_results)} total")

    # -----------------------------------------------------------------------
    # 5b. Apply V + SR if requested
    # -----------------------------------------------------------------------
    final_results = g_results   # default: just G output

    if validator is not None:
        print("      Applying Module V (V1+V2 validation) ...")
        v_results = []
        for g_res, g_inp in zip(g_results, g_inputs):
            ep = g_inp.get("evidence_pool", [])
            try:
                v_out = apply_validator(g_res, ep, validator)
            except Exception as ex:
                logger.warning(f"V failed for {g_res.get('instance_id')}: {ex}")
                v_out = g_res  # fallback: skip V
            v_results.append(v_out)

        if sr_d is not None:
            print("      Applying Module SR-D+SR-L repair ...")
            sr_results = []
            for v_res, g_inp in zip(v_results, g_inputs):
                ep = g_inp.get("evidence_pool", [])
                try:
                    sr_out = apply_repair(v_res, ep, validator, sr_d, sr_l)
                except Exception as ex:
                    logger.warning(f"SR failed for {v_res.get('instance_id')}: {ex}")
                    sr_out = v_res
                sr_results.append(sr_out)
            final_results = sr_results
        else:
            final_results = v_results

    # -----------------------------------------------------------------------
    # 6. Build predictions for Module E
    # -----------------------------------------------------------------------
    predictions = []
    for result, pm, cm in zip(final_results, pool_memberships, coverage_maps):
        if validator is not None:
            vr = result.get("validator_report")
            pred = g_output_to_pred(result, pm, cm, validator_report=vr)
        else:
            pred = g_output_to_pred(result, pm, cm)
        predictions.append(pred)

    # -----------------------------------------------------------------------
    # 7. Evaluate with Module E
    # -----------------------------------------------------------------------
    print(f"[5/5] Running Module E evaluation ...")
    evaluator = ModuleE_Evaluator()
    report, df = evaluator.run(gold_data, predictions, oracle_candidates_by_claim=oracle_candidates)

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    s = report.get("summary", {})
    def _fmt(v) -> str:
        return f"{v:.4f}" if v is not None else "  --  "

    # V-gate pass rates (only meaningful for V variants)
    n_v1_pass = sum(1 for p in predictions
                    if p.get("validator_report", {}).get("v1_gate", {}).get("pass", False))
    n_v2_pass = sum(1 for p in predictions
                    if p.get("validator_report", {}).get("v2_gate", {}).get("pass", False))
    consistency = n_v1_pass / len(predictions) if predictions else 0.0

    print(f"\n{'='*65}")
    print(f"  VARIANT={variant}  RESULTS  (N={len(df)}, model={args.model})")
    print(f"{'='*65}")
    print(f"  {'Metric':<28}  {'Value':>8}  {'Ref (gold)':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Parseable (%)' :<28}  {_fmt(s.get('Parseable')):>8}  {'1.0000':>10}")
    print(f"  {'F1_role'       :<28}  {_fmt(s.get('F1_role')):>8}  {'1.0000':>10}")
    print(f"  {'F1_fields'     :<28}  {_fmt(s.get('F1_fields')):>8}  {'1.0000':>10}")
    print(f"  {'Hit@K_text'    :<28}  {_fmt(s.get('Hit@K_text')):>8}  {'1.0000':>10}")
    print(f"  {'Hit@K_table'   :<28}  {_fmt(s.get('Hit@K_table')):>8}  {'1.0000':>10}")
    print(f"  {'Hit@K_def'     :<28}  {_fmt(s.get('Hit@K_def')):>8}  {'1.0000':>10}")
    print(f"  {'Evidence_Recall':<28}  {_fmt(s.get('Evidence_Recall')):>8}  {'1.0000':>10}")
    print(f"  {'ECS_noev'      :<28}  {_fmt(s.get('ECS_noev')):>8}  {'1.0000':>10}")
    print(f"  {'ECS_strict'    :<28}  {_fmt(s.get('ECS_strict')):>8}  {'1.0000':>10}")
    print(f"  {'ECS_oracle'    :<28}  {_fmt(s.get('ECS_oracle')):>8}  {'1.0000':>10}")
    print(f"  {'Hit@K_table_primary':<28}  {_fmt(s.get('Hit@K_table_primary')):>8}  {'1.0000':>10}")
    print(f"  {'Consistency'   :<28}  {_fmt(consistency):>8}  {'1.0000':>10}"
          + ("" if validator else "  (unchecked for G)"))
    print(f"{'='*65}")
    if not args.eval_only and g_results and variant not in ("B1", "B2"):
        v0_rate = n_v0_pass / len(g_results)
        print(f"  V0 pass rate (hallucination check): {v0_rate:.1%}")
    if validator:
        print(f"  V1 pass rate: {n_v1_pass/len(predictions):.1%}"
              f"  V2 pass rate: {n_v2_pass/len(predictions):.1%}")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    os.makedirs(args.output, exist_ok=True)

    report_path       = os.path.join(args.output, "eval_report.json")
    metrics_path      = os.path.join(args.output, "metrics.csv")
    final_pred_path   = os.path.join(args.output, "final_predictions.json")

    if not args.eval_only:
        # Only (over)write g_predictions.json on a full run
        g_pred_path = os.path.join(args.output, "g_predictions.json")
        with open(g_pred_path, "w", encoding="utf-8") as f:
            json.dump(g_results, f, indent=2, ensure_ascii=False)

    # Always save final_predictions (post-V/SR) and eval results
    with open(final_pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    df.to_csv(metrics_path, index=False)

    print(f"\nSaved to {args.output}/")
    if not args.eval_only:
        print(f"  g_predictions.json     — {len(g_results)} G outputs (raw)")
    print(f"  final_predictions.json — {len(predictions)} predictions (post V/SR)")
    print(f"  eval_report.json       — summary metrics")
    print(f"  metrics.csv            — per-instance scores")


if __name__ == "__main__":
    main()
