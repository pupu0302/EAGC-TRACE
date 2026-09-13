#!/usr/bin/env python3
"""Offline reproduction of primary-only (gold ``importance == "primary"``)
text / table / definition channel-recovery metrics, from the public data
archive + frozen predictions only.

The script reads only the released data, predictions, and result artifacts:

  - data/eagc_trace_789/gold/*.json (public gold)
  - predictions/prediction_provenance_map.json (13-row -> 11-unique-file
    map, read dynamically, no hardcoded row/file list)
  - predictions/table1/*.json (public frozen predictions)
  - results/primary_channel_summary.json (the frozen reference this script
    checks its recomputation against)
  - results/figure_data/r2_fig2__fig2_channel_recovery.csv (the frozen
    Figure 2 source data this script also checks its recomputation against)

It makes no network or model-API calls and has no dependency outside the
released code, data, and results tree.

Runs TWO independently-written scorer implementations
(evaluation/scoring/primary_channel/primary_channel_scorer.py and
..._crosscheck.py) on every one of the 11 unique prediction files. All
per-instance, per-channel decisions from those two implementations must
match exactly -- 789 instances x 3 channels x 11 files = 26,037 decisions,
zero exceptions -- exactly mirroring the cross-validation discipline
reproduce_table1.py already applies to ECS_strict. This script adds no
third scorer and does not change either existing one: every aggregate
below (method-independent AND scoreable-conditional) is built purely by
filtering/summing the SAME already-cross-checked per-instance decisions
returned by primary_channel_scorer.score_all().

Definition of "scoreable" (for the scoreable-conditional variant): a
prediction record is scoreable for an instance iff ``status == "ok"``,
``prediction`` is present (non-null), AND the predicted graph has at least
one node (i.e. it is not an empty/degenerate graph). This is the exact
convention used to build the released Figure 2
scoreable-conditional column -- reproduced here, not redefined. The
scoreable-conditional denominator for a channel is the intersection of
those scoreable instance IDs with that channel's fixed gold-eligible ID
set; the numerator is the count of hits inside that intersection, using
the same per-instance hit decisions already computed for the
method-independent variant (never rescored).

overall_status is "PASS" -- and the process exits 0 -- only if ALL of the
following hold (see reproduce/README.md for the same list):
  1. Zero cross-check mismatches over all 26,037 per-instance decisions.
  2. Gold-side denominators are exactly text=750, table_cell=119,
     definition=190 (this condition actually gates the result -- it is not
     merely reported).
  3. Every one of the 11 unique prediction files' method-independent
     text/table/definition summaries (eligible_n, hit_n, pct) match
     results/primary_channel_summary.json exactly.
  4. Every one of the 11 unique prediction files' scoreable-conditional
     summaries match the same frozen file exactly.
  5. All 13 released result rows match the frozen summary.
  6. All 24 rows of results/figure_data/r2_fig2__fig2_channel_recovery.csv
     match the recomputation (method-independent and scoreable-conditional
     numerator/denominator/percentage fields) exactly.

Usage:
    python reproduce/reproduce_primary_channel_metrics.py [--output-dir PATH]

--output-dir defaults to ./reproduced_results under the CALLER's current
working directory (never inside this package's own source tree), matching
reproduce_table1.py's convention -- both scripts can write into the same
output directory without clobbering each other's files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SCORING_DIR = os.path.join(_ROOT, "evaluation", "scoring", "primary_channel")
if _SCORING_DIR not in sys.path:
    sys.path.insert(0, _SCORING_DIR)

import primary_channel_scorer as prod  # noqa: E402
import primary_channel_scorer_crosscheck as xchk  # noqa: E402

GOLD_DIR = os.path.join(_ROOT, "data", "eagc_trace_789", "gold")
PRED_MAP_PATH = os.path.join(_ROOT, "predictions", "prediction_provenance_map.json")
FROZEN_SUMMARY_PATH = os.path.join(_ROOT, "results", "primary_channel_summary.json")
R2_FIG2_PATH = os.path.join(_ROOT, "results", "figure_data", "r2_fig2__fig2_channel_recovery.csv")
DEFAULT_OUT_DIR = os.path.join(os.getcwd(), "reproduced_results")

TOLERANCE_PCT_POINTS = 1e-6

# CSV channel name <-> internal channel key used by the scorer modules.
R2_FIG2_CHANNEL_TO_INTERNAL = {"text": "text", "table_primary": "table_cell", "definition": "definition"}

EXPECTED_DENOMINATORS = {"text": 750, "table_cell": 119, "definition": 190}


def denominators_match(denom_text: int, denom_table: int, denom_def: int) -> bool:
    """True iff the recomputed gold-side denominators are exactly
    text=750, table_cell=119, definition=190. Kept as a standalone,
    independently unit-testable function -- see
    tests/test_primary_channel_scorer.py -- rather than an inline
    comparison, specifically so a caller (or a test) can verify this
    condition actually participates in compute_overall_pass() below and is
    not merely computed-and-reported."""
    return (
        denom_text == EXPECTED_DENOMINATORS["text"]
        and denom_table == EXPECTED_DENOMINATORS["table_cell"]
        and denom_def == EXPECTED_DENOMINATORS["definition"]
    )


def compute_overall_pass(checks: "dict[str, bool]") -> bool:
    """overall_status is PASS iff every named check is True. A single
    False anywhere -- including 'denominators_match_expected' -- makes the
    whole run FAIL. Kept as a standalone function so a test can prove this
    by flipping one entry at a time (see
    tests/test_primary_channel_scorer.py::test_denominator_check_gates_overall_pass)."""
    return all(checks.values())


def load_raw_predictions(public_file: str):
    with open(os.path.join(_ROOT, public_file)) as f:
        return json.load(f)


def filter_ok_predictions(raw):
    """{instance_id: record} for status=="ok" and prediction is not None --
    the fixed scoreable-for-method-independent-scoring set. An instance
    absent here is "missing/unscoreable" and is scored as a miss on every
    otherwise-eligible channel by the scorer modules (it is never removed
    from the fixed gold-side denominator)."""
    out = {}
    n_filtered_out = 0
    for r in raw:
        if r.get("status") == "ok" and r.get("prediction") is not None:
            out[r["instance_id"]] = r
        else:
            n_filtered_out += 1
    return out, n_filtered_out


def compute_scoreable_ids(raw):
    """instance_ids that are scoreable per the definition above: status=="ok",
    non-null prediction, AND the predicted graph has >=1 node."""
    ids = set()
    for r in raw:
        if r.get("status") == "ok" and r.get("prediction") is not None:
            nodes = (r.get("prediction") or {}).get("nodes", [])
            if nodes:
                ids.add(r["instance_id"])
    return ids


def compute_scoreable_conditional(eligibility, per_instance, scoreable_ids):
    """Scoreable-conditional {channel: {eligible_n, hit_n, pct}}, built
    purely from the already-cross-checked per_instance hit decisions
    (from primary_channel_scorer.score_all) restricted to the intersection
    of gold eligibility and scoreable_ids -- no new scoring, no third
    implementation."""
    out = {}
    for c in prod.CHANNELS:
        elig_scoreable = eligibility[c] & scoreable_ids
        hit_n = sum(1 for iid in elig_scoreable if per_instance[iid][c] == 1.0)
        out[c] = {
            "eligible_n": len(elig_scoreable),
            "hit_n": hit_n,
            "pct": round(100.0 * hit_n / len(elig_scoreable), 2) if elig_scoreable else None,
        }
    return out


def _summary_matches(recomputed: dict, frozen: dict) -> bool:
    if recomputed["eligible_n"] != frozen["eligible_n"]:
        return False
    if recomputed["hit_n"] != frozen["hit_n"]:
        return False
    r_pct, f_pct = recomputed["pct"], frozen["pct"]
    if r_pct is None or f_pct is None:
        return r_pct == f_pct
    return abs(r_pct - f_pct) <= TOLERANCE_PCT_POINTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUT_DIR,
                     help="Directory to write reproduction output files to. "
                          "Defaults to ./reproduced_results under the CALLER's "
                          "current working directory (never inside this "
                          "package's own source tree).")
    args = ap.parse_args()
    OUT_DIR = args.output_dir

    if not os.path.isdir(GOLD_DIR):
        ap.error(
            "full benchmark data is not installed: missing "
            f"{GOLD_DIR}. Download and extract the data archive as described "
            "in data/README.md, then rerun this command."
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    gold_data = prod.load_public_gold(GOLD_DIR)
    flat_gold = xchk.load_public_gold_flat(GOLD_DIR)
    assert len(gold_data) == 789, f"expected 789 gold instances, found {len(gold_data)}"
    assert len(flat_gold) == 789

    elig_prod = prod.compute_gold_eligibility(gold_data)
    elig_x = xchk.channel_eligibility(flat_gold)
    for c in prod.CHANNELS:
        if elig_prod[c] != elig_x[c]:
            print(f"FATAL: gold eligibility mismatch on channel {c} between the two "
                  f"independent implementations before any prediction was even scored.")
            sys.exit(1)

    denom_text = len(elig_prod["text"])
    denom_table = len(elig_prod["table_cell"])
    denom_def = len(elig_prod["definition"])
    denominators_match_expected = denominators_match(denom_text, denom_table, denom_def)

    pred_map = json.load(open(PRED_MAP_PATH))
    frozen = json.load(open(FROZEN_SUMMARY_PATH))

    unique_files = sorted({row["public_prediction_file"] for row in pred_map["rows"]})

    all_mismatches = []
    per_file_mi = {}
    per_file_sc = {}
    per_file_n_ok = {}
    per_file_per_instance = {}

    for fname in unique_files:
        raw = load_raw_predictions(fname)
        pred_by_instance, n_filtered_out = filter_ok_predictions(raw)
        r_prod = prod.score_all(gold_data, pred_by_instance)
        r_xchk = xchk.score_all(flat_gold, pred_by_instance)

        for g in gold_data:
            iid = g["instance_id"]
            for c in prod.CHANNELS:
                v_prod = r_prod["per_instance"][iid][c]
                v_xchk = r_xchk["per_instance"][iid].get(c)
                if v_prod != v_xchk:
                    all_mismatches.append({
                        "file": fname, "instance_id": iid, "channel": c,
                        "production": v_prod, "crosscheck": v_xchk,
                    })

        scoreable_ids = compute_scoreable_ids(raw)
        per_file_mi[fname] = r_prod["summary"]
        per_file_sc[fname] = compute_scoreable_conditional(r_prod["eligibility"], r_prod["per_instance"], scoreable_ids)
        per_file_n_ok[fname] = 789 - n_filtered_out
        per_file_per_instance[fname] = r_prod["per_instance"]

    n_files = len(unique_files)
    n_decisions_total = 789 * 3 * n_files
    n_mismatches = len(all_mismatches)
    crosscheck_pass = (n_mismatches == 0)

    with open(os.path.join(OUT_DIR, "primary_channel_crosscheck_mismatches.json"), "w") as f:
        json.dump({"n_mismatches": n_mismatches, "mismatches": all_mismatches}, f, indent=2)

    # ---- Check 3 & 4: per-unique-file MI and SC summaries vs. the frozen
    # reference (results/primary_channel_summary.json). ----
    frozen_per_file = frozen["per_unique_prediction_file"]
    per_file_check_rows = []
    per_file_mi_all_match = True
    per_file_sc_all_match = True
    for fname in unique_files:
        frow = frozen_per_file.get(fname, {})
        frow_mi = frow.get("method_independent", {})
        frow_sc = frow.get("scoreable_conditional", {})
        for c in prod.CHANNELS:
            mi_match = _summary_matches(per_file_mi[fname][c], frow_mi.get(c, {}))
            sc_match = _summary_matches(per_file_sc[fname][c], frow_sc.get(c, {}))
            per_file_mi_all_match = per_file_mi_all_match and mi_match
            per_file_sc_all_match = per_file_sc_all_match and sc_match
            per_file_check_rows.append({
                "public_prediction_file": fname, "channel": c,
                "method_independent_recomputed": per_file_mi[fname][c],
                "method_independent_frozen": frow_mi.get(c),
                "method_independent_match": mi_match,
                "scoreable_conditional_recomputed": per_file_sc[fname][c],
                "scoreable_conditional_frozen": frow_sc.get(c),
                "scoreable_conditional_match": sc_match,
            })

    with open(os.path.join(OUT_DIR, "primary_channel_per_file_audit.json"), "w") as f:
        json.dump({"rows": per_file_check_rows,
                    "method_independent_all_match": per_file_mi_all_match,
                    "scoreable_conditional_all_match": per_file_sc_all_match}, f, indent=2)

    # ---- Check 5: the 13 released result rows vs. the frozen reference. ----
    row_results = []
    rows_all_match = True
    frozen_rows_by_id = {r["row_id"]: r for r in frozen["table1_rows"]}

    with open(os.path.join(OUT_DIR, "primary_channel_numerator_denominator_audit.csv"), "w") as f:
        f.write("row_id,method,backbone,public_prediction_file,"
                "text_eligible_n,text_hit_n,text_pct,text_match_frozen,"
                "def_eligible_n,def_hit_n,def_pct,def_match_frozen,"
                "table_eligible_n,table_hit_n,table_pct,table_match_frozen\n")
        for row in pred_map["rows"]:
            rid = row["row_id"]
            fname = row["public_prediction_file"]
            summ = per_file_mi[fname]
            frow = frozen_rows_by_id.get(rid, {})

            def _match(recomputed_pct, expected_key):
                exp = frow.get(expected_key)
                return (exp is not None) and abs(recomputed_pct - exp) <= TOLERANCE_PCT_POINTS

            text_match = _match(summ["text"]["pct"], "H_cite_text_primary_pct")
            def_match = _match(summ["definition"]["pct"], "H_cite_def_primary_pct")
            table_match = _match(summ["table_cell"]["pct"], "H_cite_table_primary_pct")
            row_pass = text_match and def_match and table_match
            rows_all_match = rows_all_match and row_pass

            row_results.append({
                "row_id": rid, "method": row["method"], "backbone": row["backbone"],
                "public_prediction_file": fname,
                "text": summ["text"], "definition": summ["definition"], "table_cell": summ["table_cell"],
                "match_frozen": row_pass,
            })
            f.write(f"{rid},{row['method']},{row['backbone']},{fname},"
                    f"{summ['text']['eligible_n']},{summ['text']['hit_n']},{summ['text']['pct']},{text_match},"
                    f"{summ['definition']['eligible_n']},{summ['definition']['hit_n']},{summ['definition']['pct']},{def_match},"
                    f"{summ['table_cell']['eligible_n']},{summ['table_cell']['hit_n']},{summ['table_cell']['pct']},{table_match}\n")

    # ---- Check 6: all 24 rows of r2_fig2__fig2_channel_recovery.csv. ----
    upper_rows = [r for r in pred_map["rows"] if r["section"] == "upper" and r["method"] in ("G", "TRACE")]
    backbone_method_to_file = {(r["backbone"], r["method"]): r["public_prediction_file"] for r in upper_rows}

    r2_rows_all_match = True
    r2_check_rows = []
    with open(R2_FIG2_PATH, newline="") as f:
        for csv_row in csv.DictReader(f):
            key = (csv_row["backbone"], csv_row["method"])
            fname = backbone_method_to_file.get(key)
            internal_channel = R2_FIG2_CHANNEL_TO_INTERNAL.get(csv_row["channel"])
            if fname is None or internal_channel is None:
                r2_rows_all_match = False
                r2_check_rows.append({"csv_row": csv_row, "match": False, "reason": "no recomputation mapping found"})
                continue

            mi = per_file_mi[fname][internal_channel]
            sc = per_file_sc[fname][internal_channel]

            def _int_eq(recomputed, csv_val):
                return recomputed == int(csv_val)

            def _pct_eq(recomputed, csv_val):
                if recomputed is None:
                    return csv_val == "" or csv_val is None
                return abs(recomputed - float(csv_val)) <= TOLERANCE_PCT_POINTS

            row_match = (
                _int_eq(mi["eligible_n"], csv_row["gold_eligible_n_method_independent"])
                and _int_eq(mi["hit_n"], csv_row["hit_n_method_independent"])
                and _pct_eq(mi["pct"], csv_row["pct_method_independent"])
                and _int_eq(sc["eligible_n"], csv_row["gold_eligible_n_scoreable_conditional"])
                and _int_eq(sc["hit_n"], csv_row["hit_n_scoreable_conditional"])
                and _pct_eq(sc["pct"], csv_row["pct_scoreable_conditional"])
            )
            r2_rows_all_match = r2_rows_all_match and row_match
            r2_check_rows.append({
                "backbone": csv_row["backbone"], "method": csv_row["method"], "channel": csv_row["channel"],
                "method_independent_recomputed": mi, "scoreable_conditional_recomputed": sc,
                "match": row_match,
            })

    with open(os.path.join(OUT_DIR, "primary_channel_r2_fig2_audit.json"), "w") as f:
        json.dump({"n_rows_checked": len(r2_check_rows), "all_match": r2_rows_all_match, "rows": r2_check_rows}, f, indent=2)

    # ---- Final PASS/FAIL: every one of the 6 checks above must hold. ----
    checks_for_gate = {
        "crosscheck_zero_mismatches": crosscheck_pass,
        "denominators_match_expected": denominators_match_expected,
        "per_file_method_independent_matches_frozen": per_file_mi_all_match,
        "per_file_scoreable_conditional_matches_frozen": per_file_sc_all_match,
        "table1_13_rows_match_frozen": rows_all_match,
        "r2_fig2_24_rows_match_recomputation": r2_rows_all_match,
    }
    overall_pass = compute_overall_pass(checks_for_gate)

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "n_unique_prediction_files": n_files,
        "n_instances": 789,
        "n_channels": 3,
        "n_decisions_total": n_decisions_total,
        "checks": {
            "1_crosscheck_zero_mismatches": {"pass": crosscheck_pass, "n_mismatches": n_mismatches, "n_decisions": n_decisions_total},
            "2_gold_denominators_exact": {
                "pass": denominators_match_expected,
                "recomputed": {"text": denom_text, "table_cell": denom_table, "definition": denom_def},
                "expected": {"text": 750, "table_cell": 119, "definition": 190},
            },
            "3_per_file_method_independent_matches_frozen": {"pass": per_file_mi_all_match, "n_files": n_files, "n_channel_checks": n_files * 3},
            "4_per_file_scoreable_conditional_matches_frozen": {"pass": per_file_sc_all_match, "n_files": n_files, "n_channel_checks": n_files * 3},
            "5_table1_13_rows_match_frozen": {"pass": rows_all_match, "n_rows": len(row_results)},
            "6_r2_fig2_24_rows_match_recomputation": {"pass": r2_rows_all_match, "n_rows": len(r2_check_rows)},
        },
        "gold_side_denominators": {"text": denom_text, "table_cell": denom_table, "definition": denom_def},
        "expected_gold_side_denominators": {"text": 750, "table_cell": 119, "definition": 190},
        "denominators_match_expected": denominators_match_expected,
        "rows": row_results,
    }
    with open(os.path.join(OUT_DIR, "PRIMARY_CHANNEL_REPRODUCTION_RESULT.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"overall_status={result['overall_status']}  files={n_files}  "
          f"decisions={n_decisions_total}  crosscheck_mismatches={n_mismatches}  "
          f"denominators(text/table/def)={denom_text}/{denom_table}/{denom_def} "
          f"(expected 750/119/190, match={denominators_match_expected})  "
          f"per_file_MI_match={per_file_mi_all_match}  per_file_SC_match={per_file_sc_all_match}  "
          f"table1_rows_match={rows_all_match}  r2_fig2_match={r2_rows_all_match}")
    for r in row_results:
        flag = "OK" if r["match_frozen"] else "FAIL"
        print(f"  [{flag}] row{r['row_id']:02d} {r['method']}/{r['backbone']}: "
              f"text={r['text']['pct']} def={r['definition']['pct']} table={r['table_cell']['pct']}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
