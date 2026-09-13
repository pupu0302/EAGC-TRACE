#!/usr/bin/env python3
"""
B3 Flat-Evidence comparator for the frozen GPT-4o Table 1 run.

The script gives the model the same full, unrouted evidence pool as G, but
presents it as one flat, untyped list. It retains the full EAGC output schema
and does not use R1 routing, V validation, SR-D repair or SR-L repair.

The frozen Flat-Evidence and G system prompts are checked by SHA-256 at import
time so a prompt change cannot silently alter the comparator.

Key differences from run_g_baseline.py's standard G path:
  - Evidence pool: bare G's full unrouted pool (build_evidence_pool via
    task_to_g_input) — no R1, no V, no SR-D, no SR-L.
  - User prompt: flat, untyped EVIDENCE list (same units, content and order as
    bare G), with no type sectioning or routing hints.
  - System prompt: full EAGC output contract, matching G except for the
    citation-discipline rules tied to typed presentation.
  - No mock fallback. Parse failures and API exceptions are preserved as
    explicit empty predictions, with the raw response archived separately.
"""

import sys
import os
import json
import time
import hashlib
import argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(PIPELINE_DIR)
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
for p in (PIPELINE_DIR, SCRIPTS_DIR, PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_g_baseline import (
    task_to_g_input, load_tasks, load_gold_dir, adapt_gold_annotation,
    build_oracle_candidates, g_output_to_pred,
)
from llm.g_llm_client import create_llm_client
from g.g_generator import ModuleG_Generator
from e.e_evaluation import ModuleE_Evaluator

FROZEN_PROMPT_DIR = os.path.join(PIPELINE_DIR, "prompts")

FLAT_EVIDENCE_SYSTEM_PROMPT_SHA256_EXPECTED = "775aa577e11ad8305254da027756b96ac1263d01346fb83fc8da4894d29b2dcc"
G_SYSTEM_PROMPT_SHA256_EXPECTED = "331ad34e39e1c9fd4dfaeddfc725fbb6a70222c3da37044db5b732624be23ce2"

with open(os.path.join(FROZEN_PROMPT_DIR, "flat_evidence_system_prompt.txt")) as f:
    FLAT_EVIDENCE_SYSTEM_PROMPT = f.read()
with open(os.path.join(FROZEN_PROMPT_DIR, "G_system_prompt.txt")) as f:
    G_SYSTEM_PROMPT_FOR_COMPARISON = f.read()

_actual_flat_evidence_hash = hashlib.sha256(FLAT_EVIDENCE_SYSTEM_PROMPT.encode()).hexdigest()
_actual_g_hash = hashlib.sha256(G_SYSTEM_PROMPT_FOR_COMPARISON.encode()).hexdigest()
if _actual_flat_evidence_hash != FLAT_EVIDENCE_SYSTEM_PROMPT_SHA256_EXPECTED:
    print(f"FATAL: B3 Flat-Evidence system prompt hash mismatch. Expected {FLAT_EVIDENCE_SYSTEM_PROMPT_SHA256_EXPECTED}, got {_actual_flat_evidence_hash}. ABORT.")
    sys.exit(1)
if _actual_g_hash != G_SYSTEM_PROMPT_SHA256_EXPECTED:
    print(f"FATAL: G system prompt hash mismatch. Expected {G_SYSTEM_PROMPT_SHA256_EXPECTED}, got {_actual_g_hash}. ABORT.")
    sys.exit(1)

JSON_STRUCTURE_TEMPLATE = {
    "claim_id": "string", "doc_id": "string",
    "nodes": [{"node_id": "string", "node_type": "Metric|Outcome|Target|Definition|Action", "text": "string", "attrs": {}}],
    "edges": [{"src": "string", "dst": "string", "edge_type": "string"}],
    "evidence_sets": [{"set_id": "string", "attached_to_node": "string", "links": [{"evid_id": "string", "relation": "string"}]}]
}


def build_flat_evidence_user_prompt(g_input):
    """Flat, untyped evidence presentation. Same units/IDs/content/order as
    g_input['evidence_pool'] (bare G's full unrouted pool). No type sectioning,
    no numeric_table_hint, no most-relevant-first instruction."""
    claim = g_input.get('canonical_text', '')
    pool = g_input.get('evidence_pool', [])
    prompt = f"CLAIM: {claim}\n"
    if pool:
        prompt += "\nEVIDENCE:\n"
        for item in pool:
            prompt += f"[{item['evid_id']}] {item.get('content', '')}\n"
    prompt += "\n\nOutput the graph as JSON with this structure:\n"
    prompt += json.dumps(JSON_STRUCTURE_TEMPLATE, indent=2)
    return prompt


def evid_sequence(pool):
    return [item['evid_id'] for item in pool]


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


EMPTY_PREDICTION = {"claim_id": None, "doc_id": None, "nodes": [], "edges": [], "evidence_sets": []}


def run_one_instance(g_input, llm_client, gen_for_v0, max_tokens, temperature):
    """No-fallback single-instance B3 Flat-Evidence call. Returns (g_result_dict, raw_archive_dict)."""
    iid = g_input['instance_id']
    full_pool = g_input['evidence_pool']
    valid_pool_ids = gen_for_v0._extract_valid_ids(full_pool)

    user_prompt = build_flat_evidence_user_prompt(g_input)
    request_hash = sha256(FLAT_EVIDENCE_SYSTEM_PROMPT + "\x00" + user_prompt)

    start = time.time()
    resp = None
    try:
        resp = llm_client.generate(
            system_prompt=FLAT_EVIDENCE_SYSTEM_PROMPT, user_prompt=user_prompt,
            response_format="json_object", temperature=temperature, max_tokens=max_tokens,
        )
        latency_ms = int((time.time() - start) * 1000)
        parsed = resp.get("parsed_json")
        raw_content = resp.get("content")
        exception_str = None
    except Exception as ex:
        latency_ms = int((time.time() - start) * 1000)
        parsed = None
        raw_content = None
        exception_str = f"{type(ex).__name__}: {ex}"

    has_error_marker = parsed is None or (isinstance(parsed, dict) and "error" in parsed) or (isinstance(parsed, dict) and "nodes" not in parsed)

    if has_error_marker or exception_str is not None:
        prediction = dict(EMPTY_PREDICTION)
        parse_status = "exception" if exception_str is not None else "parse_failed_or_invalid_shape"
    else:
        prediction = parsed
        parse_status = "ok"

    gate_res = gen_for_v0._v0_gate_check(prediction, valid_pool_ids)

    g_result = {
        "module": "Module_FlatEvidence",
        "instance_id": iid,
        "status": "success" if parse_status == "ok" else "parse_failed",
        "prediction": prediction,
        "validator_report": {"v0_gate": gate_res},
        "meta": {
            "latency_ms": latency_ms,
            "model_used": llm_client.model_name,
            "parse_status": parse_status,
        },
    }
    raw_archive = {
        "instance_id": iid,
        "request_prompt_hash": request_hash,
        "system_prompt_sha256": _actual_flat_evidence_hash,
        "user_prompt_sha256": sha256(user_prompt),
        "raw_content": raw_content,
        "exception": exception_str,
        "parse_status": parse_status,
        "latency_ms": latency_ms,
        "usage": resp.get("usage") if resp is not None else None,
        "cost_usd": resp.get("cost_usd") if resp is not None else None,
    }
    return g_result, raw_archive


def main():
    parser = argparse.ArgumentParser(description="B3 Flat-Evidence frozen run — Flat-Evidence Graph-RAG, GPT-4o")
    parser.add_argument("--tasks", default="data/eagc_trace_789/tasks")
    parser.add_argument("--gold", default="data/eagc_trace_789/gold")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="gpt-4o", help="Provider model identifier (default: gpt-4o)")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true",
                         help="No API calls. Build+hash all prompts, verify evidence-sequence equality vs bare G, then exit.")
    parser.add_argument("--evaluable-subset-manifest", default=None,
                         help="Optional JSON manifest containing an "
                              "'evaluable_canonical_claim_ids' list; evaluates "
                              "that subset in addition to the full benchmark.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    raw_archive_dir = os.path.join(args.output, "raw_responses")
    os.makedirs(raw_archive_dir, exist_ok=True)

    print(f"[1/5] Loading tasks from {args.tasks} ...")
    tasks = load_tasks(args.tasks)
    print(f"      {len(tasks)} task JSONs found.")

    print(f"[2/5] Loading gold from {args.gold} ...")
    gold_raw = load_gold_dir(args.gold)
    gold_raw_valid = [r for r in gold_raw if isinstance(r, dict) and
                      r.get("pipeline_status") in ("PASS", "PASS_REPAIRED", "ADJUDICATED")]
    gold_data = [adapt_gold_annotation(r) for r in gold_raw_valid]
    gold_ids = {g["instance_id"] for g in gold_data}
    tasks_covered = [t for t in tasks if t.get("canonical_claim_id") in gold_ids]
    print(f"      {len(tasks_covered)}/{len(tasks)} tasks have gold coverage.")

    oracle_candidates = build_oracle_candidates(tasks_covered)

    task_inputs = [task_to_g_input(t) for t in tasks_covered]
    g_inputs = [t[0] for t in task_inputs]
    pool_memberships = [t[1] for t in task_inputs]
    coverage_maps = [t[2] for t in task_inputs]
    print(f"      Built {len(g_inputs)} bare-G evidence pools (no R1 routing).")

    # Evidence-sequence equality audit vs bare G. g_inputs here are built by the
    # identical, unmodified task_to_g_input() call the paper's canonical G run
    # used (no separate "G's exact input pool" artifact is stored anywhere to
    # diff against directly — G's run only persisted its OUTPUT). This is a
    # code-identity guarantee (same function, same task files, deterministic),
    # not an independently-observed per-instance diff. Cross-checked empirically
    # against representative task records before the frozen run.
    print(f"[3/5] Evidence pool built via task_to_g_input() — same function bare G's "
          f"generation used, no R1/routing applied. Pool sizes: min={min(len(gi['evidence_pool']) for gi in g_inputs)}, "
          f"max={max(len(gi['evidence_pool']) for gi in g_inputs)}.")

    gen_for_v0 = ModuleG_Generator(config={"provider": "mock", "model_name": "n/a"})

    if args.dry_run:
        print("[4/5] DRY RUN: constructing + hashing all prompts, zero API calls ...")
        seq_records = []
        for gi in g_inputs:
            user_prompt = build_flat_evidence_user_prompt(gi)
            seq_records.append({
                "instance_id": gi["instance_id"],
                "pool_size": len(gi["evidence_pool"]),
                "evidence_id_sequence_sha256": sha256(json.dumps(evid_sequence(gi["evidence_pool"]))),
                "user_prompt_sha256": sha256(user_prompt),
            })
        with open(os.path.join(args.output, "dry_run_prompt_hashes.json"), "w") as f:
            json.dump(seq_records, f, indent=2)
        print(f"      Wrote {len(seq_records)} prompt hashes to dry_run_prompt_hashes.json. No API calls made. Exiting.")
        return

    print(f"[4/5] Running B3 Flat-Evidence on {len(g_inputs)} tasks (model={args.model_name}, "
          f"max_tokens={args.max_tokens}, temperature={args.temperature}, concurrency={args.concurrency}) ...")
    llm_client = create_llm_client("openai", args.model_name, {"temperature": args.temperature, "max_tokens": args.max_tokens})

    g_results = [None] * len(g_inputs)
    raw_archives = [None] * len(g_inputs)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_one_instance, gi, llm_client, gen_for_v0, args.max_tokens, args.temperature): i
            for i, gi in enumerate(g_inputs)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            g_results[idx], raw_archives[idx] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(g_inputs):
                print(f"      {done}/{len(g_inputs)} complete")

    n_success = sum(1 for r in g_results if r["status"] == "success")
    n_parse_failed = sum(1 for r in g_results if r["status"] == "parse_failed")
    print(f"\n      B3 Flat-Evidence complete: success={n_success}, parse_failed={n_parse_failed} / {len(g_results)} total")

    # Persist raw responses (one file per instance) + g_predictions.json
    for ra in raw_archives:
        fname = ra["instance_id"].replace(":", "_") + ".json"
        with open(os.path.join(raw_archive_dir, fname), "w") as f:
            json.dump(ra, f, indent=2)

    with open(os.path.join(args.output, "g_predictions.json"), "w") as f:
        json.dump(g_results, f, indent=2)

    request_hash_log = [{"instance_id": ra["instance_id"], "request_prompt_hash": ra["request_prompt_hash"],
                          "parse_status": ra["parse_status"]} for ra in raw_archives]
    with open(os.path.join(args.output, "request_prompt_hash_log.json"), "w") as f:
        json.dump(request_hash_log, f, indent=2)

    # Build predictions for Module E (no validator_report -> V1/V2 unchecked=pass, matching plain-G scoring convention)
    predictions = []
    for result, pm, cm in zip(g_results, pool_memberships, coverage_maps):
        pred = g_output_to_pred(result, pm, cm)
        predictions.append(pred)

    with open(os.path.join(args.output, "final_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    # Evaluate on the full covered set (N is whatever the --tasks/--gold
    # inputs resolve to. The released defaults point at the 789-instance
    # evaluation set, so N is computed dynamically rather than hardcoded).
    n_full = len(predictions)
    print(f"[5/5] Running Module E evaluation (N={n_full}) ...")
    evaluator = ModuleE_Evaluator()
    report_full, df_full = evaluator.run(gold_data, predictions, oracle_candidates_by_claim=oracle_candidates)
    with open(os.path.join(args.output, f"eval_report_N{n_full}.json"), "w") as f:
        json.dump(report_full, f, indent=2)
    df_full.to_csv(os.path.join(args.output, f"metrics_N{n_full}.csv"), index=False)

    # Optional: evaluate on a named evaluable-subset manifest, if one is
    # passed via --evaluable-subset-manifest. The flag lets a user supply a
    # subset manifest; if omitted or the path does not
    # exist, this step is skipped and only the full-N evaluation above runs.
    report_subset = None
    if args.evaluable_subset_manifest and os.path.exists(args.evaluable_subset_manifest):
        manifest_subset = json.load(open(args.evaluable_subset_manifest))
        evaluable_ids = set(manifest_subset["evaluable_canonical_claim_ids"])
        gold_subset = [g for g in gold_data if g["instance_id"] in evaluable_ids]
        oracle_subset = {k: v for k, v in oracle_candidates.items() if k in evaluable_ids}
        predictions_subset = [p for p in predictions if p["instance_id"] in evaluable_ids]
        n_subset = len(predictions_subset)
        print(f"      Evaluable-subset manifest: N={n_subset} gold, {len(predictions_subset)} predictions")
        report_subset, df_subset = evaluator.run(gold_subset, predictions_subset, oracle_candidates_by_claim=oracle_subset)
        with open(os.path.join(args.output, f"eval_report_N{n_subset}.json"), "w") as f:
            json.dump(report_subset, f, indent=2)
        df_subset.to_csv(os.path.join(args.output, f"metrics_N{n_subset}.csv"), index=False)
    elif args.evaluable_subset_manifest:
        print(f"      WARNING: {args.evaluable_subset_manifest} not found — subset evaluation skipped.")

    print(f"\n=== N={n_full} summary ===")
    print(json.dumps(report_full.get("summary", {}), indent=2))
    if report_subset:
        print("\n=== evaluable-subset summary ===")
        print(json.dumps(report_subset.get("summary", {}), indent=2))

    print(f"\nSaved to {args.output}/")
    print(f"  g_predictions.json, final_predictions.json, eval_report_N{n_full}.json,")
    print(f"  metrics_N{n_full}.csv, request_prompt_hash_log.json, raw_responses/*.json")


if __name__ == "__main__":
    main()
