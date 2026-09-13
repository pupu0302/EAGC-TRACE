#!/usr/bin/env python3
"""
EAGC-TRACE public package quickstart.

Default mode: a source-only synthetic smoke check. It makes no model or
network call and reads no real task/gold/prediction data. It loads every
fixture returned by
evaluation/scoring/fixtures/synthetic_fixtures.py::build_all_fixtures,
scores each with BOTH the production scorer
(evaluation/scoring/production_wrapper/production_scorer.py) and the
independently-written reference scorer
(evaluation/scoring/reference_implementation/reference_scorer.py), and
requires the two implementations to agree with each other and with each
fixture's own documented expectation for every declared K/full value
(FROZEN_SCORING_CONTRACT.yaml::cross_validation_requirement). This is a
smoke check of the scoring code path only -- it never reads or reports a
real ECS_strict number and is NOT a Table 1 result.

--release-sample mode: scores the same five real BAX_2020 instances the
quickstart previously shipped bundled copies of, but reads them from the
normal composed release layout (data/eagc_trace_789/{tasks,gold}/ and
predictions/table1/row02_G_GPT_4o.json) instead of a duplicated local copy.
It requires the full benchmark and results to already be present in this
checkout; if either is missing, it exits nonzero with an actionable message
rather than silently degrading to load-only or skipping scoring.

Neither mode is the full benchmark reproduction. For the full, cross-checked,
789-instance Table 1 reproduction, see reproduce/reproduce_table1.py at the
repository root; this quickstart does not depend on it and does not
replace it.

Usage:
    python quickstart/run_quickstart.py                   # synthetic smoke check (default)
    python quickstart/run_quickstart.py --release-sample   # real 5-instance sample (needs data/ + predictions/)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

# This script establishes every import path it needs itself, up front --
# it does not rely on any imported module's own sys.path mutation as a
# side effect to make a later import resolve (in particular, it does not
# rely on production_scorer.py's own internal bootstrap, which separately
# points sys.path at evaluation/ for its own lazy `from e.e_evaluation
# import ...`; evaluation/ is added below independently of that).
_EVALUATION_DIR = os.path.join(_REPO_ROOT, "evaluation")
_SCORING_DIR = os.path.join(_EVALUATION_DIR, "scoring")
_FIXTURES_DIR = os.path.join(_SCORING_DIR, "fixtures")
_PRODUCTION_DIR = os.path.join(_SCORING_DIR, "production_wrapper")
_REFERENCE_DIR = os.path.join(_SCORING_DIR, "reference_implementation")
for _p in (_EVALUATION_DIR, _SCORING_DIR, _FIXTURES_DIR, _PRODUCTION_DIR, _REFERENCE_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SAMPLE_INSTANCE_IDS = [
    "BAX_2020:cc:000059",
    "BAX_2020:cc:000075",
    "BAX_2020:cc:000081",
    "BAX_2020:cc:000096",
    "BAX_2020:cc:000098",
]


def run_synthetic_smoke_check() -> int:
    """Default mode. Source-only: no model/network call, no real data read."""
    import production_scorer  # noqa: E402
    import reference_scorer  # noqa: E402
    from synthetic_fixtures import build_all_fixtures  # noqa: E402

    print("EAGC-TRACE quickstart -- SYNTHETIC SMOKE CHECK")
    print("(fabricated fixtures only; makes no model or network call; "
          "NOT a Table 1 result)\n")

    fixtures = build_all_fixtures()
    n_fixtures = len(fixtures)
    n_checks = 0
    mismatches = []            # production vs. reference disagree with each other
    expectation_failures = []  # either scorer disagrees with the fixture's own documented expectation

    for name, fx in sorted(fixtures.items()):
        gold_data = [fx["gold"]]
        # A fixture with pred=None (e.g. a missing-prediction case) is kept
        # in the denominator via denominator_instance_ids below and scored
        # 0 by both scorers -- never silently dropped.
        pred_data = [fx["pred"]] if fx["pred"] is not None else []
        denom_ids = [fx["gold"]["instance_id"]]

        prod = production_scorer.score_all(gold_data, pred_data, denominator_instance_ids=denom_ids)
        ref = reference_scorer.score_all(gold_data, pred_data, denominator_instance_ids=denom_ids)
        prod_rec = prod["per_instance"][0]
        ref_rec = ref["per_instance"][0]

        for k, expected in fx["expect"].items():
            key = f"ecs_strict_at_{k}"
            n_checks += 1
            pv, rv = prod_rec[key], ref_rec[key]
            if pv != rv:
                mismatches.append((name, key, pv, rv))
            if pv != expected:
                expectation_failures.append((name, key, "production", pv, expected))
            if rv != expected:
                expectation_failures.append((name, key, "reference", rv, expected))

    print(f"fixtures loaded:                 {n_fixtures}")
    print(f"K/full values checked:           {n_checks}")
    print(f"production/reference mismatches: {len(mismatches)}")
    print(f"expectation failures:            {len(expectation_failures)}")

    if mismatches:
        print("\nproduction/reference disagreements:")
        for name, key, pv, rv in mismatches:
            print(f"  [{name}] {key}: production={pv} reference={rv}")
    if expectation_failures:
        print("\nfixture expectation failures:")
        for name, key, which, got, expected in expectation_failures:
            print(f"  [{name}] {key} ({which}): got={got} expected={expected}")

    if mismatches or expectation_failures:
        print("\nFAIL -- synthetic smoke check found a scorer mismatch or an "
              "unmet fixture expectation. This checks the scoring code path "
              "only; it is not a Table 1 result.")
        return 1

    print(f"\nPASS -- production and reference scorers agree with each other "
          f"and with every fixture's own documented expectation on all "
          f"{n_fixtures} synthetic fixtures ({n_checks} checked values). "
          f"This is a synthetic smoke check, not a Table 1 result. For the "
          f"full 789-instance reproduction, see "
          f"reproduce/reproduce_table1.py at the repository root.")
    return 0


def _task_path_for(base_dir: str, instance_id: str) -> str:
    doc_id, _, suffix = instance_id.partition(":cc:")
    return os.path.join(base_dir, "tasks", f"{doc_id}_formal_{suffix}.json")


def _gold_path_for(base_dir: str, instance_id: str) -> str:
    doc_id, _, suffix = instance_id.partition(":cc:")
    return os.path.join(base_dir, "gold", f"{doc_id}_task_formal_{suffix}.json")


def run_release_sample() -> int:
    """--release-sample mode. Real 5-instance sample read from the normal
    composed release layout. Requires the full benchmark and results to be
    present; fails loudly (nonzero, actionable message) rather than
    degrading to load-only or skipping scoring."""
    data_dir = os.path.join(_REPO_ROOT, "data", "eagc_trace_789")
    tasks_dir = os.path.join(data_dir, "tasks")
    gold_dir = os.path.join(data_dir, "gold")
    pred_file = os.path.join(_REPO_ROOT, "predictions", "table1", "row02_G_GPT_4o.json")

    missing = [os.path.relpath(p, _REPO_ROOT) for p in (tasks_dir, gold_dir, pred_file)
               if not os.path.exists(p)]
    if missing:
        print(
            "ERROR: --release-sample requires the full benchmark and results to "
            "already be present in this checkout. Missing required path(s): "
            f"{missing}. This source package does not bundle the 789-instance "
            "benchmark or predictions -- see data/README.md to obtain the full "
            "release, then re-run this command from the composed checkout. "
            "(The default synthetic smoke check does not need any of this.)",
            file=sys.stderr,
        )
        return 1

    import production_scorer  # noqa: E402
    from e.e_evaluation import adapt_gold_annotation  # noqa: E402

    tasks_and_gold = {}
    for iid in SAMPLE_INSTANCE_IDS:
        task_path = _task_path_for(data_dir, iid)
        gold_path = _gold_path_for(data_dir, iid)
        if not os.path.isfile(task_path) or not os.path.isfile(gold_path):
            print(
                f"ERROR: --release-sample instance {iid} is missing its task or "
                f"gold file ({os.path.relpath(task_path, _REPO_ROOT)} / "
                f"{os.path.relpath(gold_path, _REPO_ROOT)}) in this checkout.",
                file=sys.stderr,
            )
            return 1
        with open(task_path, "r", encoding="utf-8") as f:
            task = json.load(f)
        with open(gold_path, "r", encoding="utf-8") as f:
            gold = json.load(f)
        tasks_and_gold[iid] = (task, gold)

    print(f"Loaded {len(tasks_and_gold)} real sample instances from "
          f"{os.path.relpath(data_dir, _REPO_ROOT)}/ ...\n")

    with open(pred_file, "r", encoding="utf-8") as f:
        all_predictions = json.load(f)
    pred_by_id = {p["instance_id"]: p for p in all_predictions}

    gold_records = []
    pred_records = []
    for iid, (task, gold) in tasks_and_gold.items():
        if iid not in pred_by_id:
            print(
                f"ERROR: --release-sample instance {iid} has no matching "
                f"prediction in {os.path.relpath(pred_file, _REPO_ROOT)}.",
                file=sys.stderr,
            )
            return 1
        gold_records.append(adapt_gold_annotation(gold))
        pred_records.append(pred_by_id[iid])

    ids = [g["instance_id"] for g in gold_records]
    result = production_scorer.score_all(gold_records, pred_records, denominator_instance_ids=ids)

    print(f"{'instance_id':<28} {'ecs_strict_at_full':>20}")
    for rec in result["per_instance"]:
        print(f"{rec['instance_id']:<28} {rec.get('ecs_strict_at_full', 'n/a'):>20}")

    n = result["n_instances"]
    summary = result["summary"]["ecs_strict_at_full"]
    print(f"\n5-instance sample summary (illustration only, not the full "
          f"Table 1 result): {summary}")

    if n != 5 or summary["denominator"] != 5 or summary["numerator"] != 5:
        print(
            f"\nFAIL -- expected exactly 5/5 (all five sample instances scored "
            f"ecs_strict_at_full=1), got n_instances={n}, "
            f"numerator={summary['numerator']}, denominator={summary['denominator']}.",
            file=sys.stderr,
        )
        return 1

    print("\nPASS -- 5/5 real sample instances scored with the production "
          "scorer. For the full 789-instance Table 1 reproduction "
          "(0-mismatch production/reference cross-check against "
          "results/EXPECTED_RESULTS.json), see reproduce/reproduce_table1.py "
          "at the repository root.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--release-sample", action="store_true",
        help="Score the 5 real BAX_2020 sample instances read from the normal "
             "composed data/eagc_trace_789 + predictions/table1 layout, instead "
             "of the default synthetic smoke check. Requires the full benchmark "
             "and results to already be present; fails nonzero otherwise.",
    )
    args = parser.parse_args()
    return run_release_sample() if args.release_sample else run_synthetic_smoke_check()


if __name__ == "__main__":
    sys.exit(main())
