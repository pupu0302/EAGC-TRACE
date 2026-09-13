#!/usr/bin/env python3
"""`make validate-data` entry point: re-run the frozen validator against the
public data/eagc_trace_789/ tasks+gold, entirely within this package (no
external paths, no network). The validator applies three layers: V0
(blocking — schema + evidence closed-set), V1 (blocking — field
consistency), and V2 (non-blocking — T-M-D-O closure diagnostics; emits
warnings, not errors, under the current released behavior)."""
import json
import os
import sys
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DSET = os.path.join(_ROOT, "data", "eagc_trace_789")
TASKS = os.path.join(DSET, "tasks")
GOLD = os.path.join(DSET, "gold")
SCHEMA = os.path.join(_ROOT, "schemas", "eagc.schema.json")


def gold_name_for(task_fn):
    base = task_fn[:-5]
    return base.replace("_formal_", "_task_formal_", 1) + ".json"


def main():
    if not os.path.isdir(TASKS) or not os.path.isdir(GOLD):
        print(
            "Full benchmark data is not installed. Download and extract "
            "EAGC-TRACE_data.tar.gz as described in data/README.md, "
            "then rerun this command.",
            file=sys.stderr,
        )
        return 2

    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from benchmark_validation import AnnotationValidator

    validator = AnnotationValidator(schema_path=SCHEMA)
    task_files = sorted(os.listdir(TASKS))
    n_ok = 0
    v0_fail, v1_fail, v2_fail, missing = [], [], [], []

    for tfn in task_files:
        gfn = gold_name_for(tfn)
        gpath = os.path.join(GOLD, gfn)
        if not os.path.exists(gpath):
            missing.append((tfn, gfn))
            continue
        task = json.load(open(os.path.join(TASKS, tfn)))
        gold = json.load(open(gpath))
        pool_ids = set()
        for items in task.get("candidate_pools", {}).values():
            for it in items:
                pool_ids.update(it.get("covered_evidence_ids", []))
        report = validator.validate(gold["prediction"], pool_ids, strict=True)
        if report.v0_pass and report.v1_pass and report.v2_pass:
            n_ok += 1
        else:
            entry = {"instance_id": gold.get("canonical_claim_id"), "errors": [f"{e.gate}:{e.error_code}:{e.message}" for e in report.errors]}
            (v0_fail if not report.v0_pass else v1_fail if not report.v1_pass else v2_fail).append(entry)

    n_total = len(task_files)
    blocking = bool(missing or v0_fail)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_instances": n_total,
        "n_pass_all_gates": n_ok,
        "n_missing_gold_pairs": len(missing),
        "n_v0_fail": len(v0_fail), "n_v1_fail": len(v1_fail), "n_v2_fail": len(v2_fail),
        "blocking_error": blocking,
    }
    print(json.dumps(result, indent=2))
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
