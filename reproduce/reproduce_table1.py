#!/usr/bin/env python3
"""Offline reproduction of the released Table 1 and Fig. 3 ablation rows.

No network access, no model API calls. Reads:
  - data/eagc_trace_789/gold/*.json (public gold)
  - predictions/prediction_provenance_map.json (row list -- read dynamically,
    no hardcoded row count) + predictions/table1/*.json (public predictions)
  - results/EXPECTED_RESULTS.json (frozen values to check against)

Runs BOTH the production and reference scorer implementations independently
and asserts 0 instance-level mismatch between them for every row, per K, per
the frozen scoring contract (evaluation/FROZEN_SCORING_CONTRACT.yaml). Any
mismatch is a hard failure (non-zero exit), never averaged away.

Usage:
    python reproduce/reproduce_table1.py [--full-stats] [--output-dir PATH]

--output-dir defaults to ./reproduced_results under the CALLER's current
working directory, never inside this package's own source tree, so this
script runs cleanly from a read-only checkout with output redirected
elsewhere (e.g. `make reproduce-table1 OUTPUT_DIR=/tmp/eagc-reproduced`).

--full-stats additionally recomputes bootstrap CIs + Holm-corrected
comparisons (see run_full_stats() below) using evaluation/scoring/
bootstrap_lib.py + holm_correction.py, and compares them against the frozen
reference in results/bootstrap/table1_bootstrap_results.json. This includes
the full frozen seven-comparison family (Holm n=7): the four members
computable directly from shipped public predictions, plus three members
members compared against the OLD empty-evidence B3 control (G_vs_B3_4o,
TRACE_vs_B3_4o, TRACE_vs_B3_mini), reproduced using a design-defined
deterministic zero vector (evaluation/scoring/empty_evidence_control.py) --
not a replay of that control's historical model predictions, which are not
shipped in this package by design (see docs/METHOD_OVERVIEW.md). See
reproduce/README.md for why an all-zero vector is metric-equivalent to
that control under ECS_strict.

Exit code 0 only if: (a) all rows' recomputed ECS_strict match
EXPECTED_RESULTS.json within float tolerance, (b) 0 production/reference
mismatches across every instance x K x row, and (c) if --full-stats was
passed, every recomputed bootstrap/Holm value -- across both the primary
n=4 TRACE-vs-G family and the full n=7 FROZEN_ORIGINAL_COMPARISON_FAMILY,
all 3 resampling schemes -- matches the frozen reference exactly (p_raw and
p_holm equal after rounding both sides to 4 decimals; observed delta equal
within 1e-12 percentage points; comparison_id, family size, bootstrap
method, and significance flags equal exactly). No tolerance band is applied
to absorb disagreement -- see STRICT_MATCH_NDIGITS / STRICT_DELTA_ABS_TOL
below.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_EVAL_DIR = os.path.join(_ROOT, "evaluation")
for _p in (
    os.path.join(_EVAL_DIR, "scoring", "production_wrapper"),
    os.path.join(_EVAL_DIR, "scoring", "reference_implementation"),
    os.path.join(_EVAL_DIR, "scoring"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import production_scorer  # noqa: E402
import reference_scorer  # noqa: E402
# Note: production_scorer.py / reference_scorer.py already expose their
# results under "ecs_strict_at_*" keys (the public naming) -- the ACS-named
# internals (acs_strict/acs_noev/acs_oracle) live only inside the deeper
# frozen e_evaluation.py::ModuleE_Evaluator, which this script never touches
# directly. evaluation/scoring/acs_ecs_adapter.py is for callers that DO
# consume ModuleE_Evaluator's raw output; not needed in this script.

from empty_evidence_control import build_empty_evidence_control  # noqa: E402

DATA_DIR = os.path.join(_ROOT, "data", "eagc_trace_789")
GOLD_DIR = os.path.join(DATA_DIR, "gold")
PRED_MAP_PATH = os.path.join(_ROOT, "predictions", "prediction_provenance_map.json")
PRED_DIR = os.path.join(_ROOT, "predictions", "table1")
EXPECTED_PATH = os.path.join(_ROOT, "results", "EXPECTED_RESULTS.json")
DEFAULT_OUT_DIR = os.path.join(os.getcwd(), "reproduced_results")

TOLERANCE_PCT_POINTS = 1e-6

# Statistical-reproduction acceptance thresholds (--full-stats only).
# No absolute p-value tolerance band: both sides are rounded to the same
# precision the public results are reported at, then compared for exact
# equality. STRICT_DELTA_ABS_TOL bounds only floating-point summation-order
# noise on the observed-delta statistic itself (same seed/B/algorithm/data),
# never a "close enough" allowance for a genuinely different number.
STRICT_MATCH_NDIGITS = 4
STRICT_DELTA_ABS_TOL = 1e-12


def _p_equal(a: float, b: float, ndigits: int = STRICT_MATCH_NDIGITS) -> bool:
    return round(float(a), ndigits) == round(float(b), ndigits)


def _delta_equal(a: float, b: float, tol: float = STRICT_DELTA_ABS_TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def _bool_str_equal(recomputed_bool: bool, frozen_str: str) -> bool:
    return str(bool(recomputed_bool)) == str(frozen_str)


def load_gold():
    """Build {instance_id, nodes, gold_evidence_sets} records from the
    public gold files -- the exact shape ModuleE_Evaluator/the scorers
    expect, derived mechanically from each file's prediction.{nodes,
    evidence_sets}."""
    gold_data = []
    for fn in sorted(os.listdir(GOLD_DIR)):
        g = json.load(open(os.path.join(GOLD_DIR, fn)))
        pred = g.get("prediction", {})
        gold_data.append({
            "instance_id": g["canonical_claim_id"],
            "nodes": pred.get("nodes", []),
            "gold_evidence_sets": pred.get("evidence_sets", []),
        })
    return gold_data


def load_predictions_for_scoring(public_file):
    """The public prediction file keeps a full 789-length left-join array,
    with explicit null-prediction placeholder records for every failed/
    missing instance (for transparency/auditability -- see
    predictions/README.md). The frozen scorer chain (e_evaluation.py's
    ModuleE_Evaluator._evaluate_instance / production_scorer.py /
    reference_scorer.py), however, expects a failed instance to be simply
    ABSENT from pred_data -- a present-but-null "prediction" key crashes
    ModuleE_Evaluator (AttributeError on None.get(...)). Both scorers already
    handle an absent pred_map entry correctly (score 0 for that instance,
    via the fixed denominator_instance_ids), so filtering here changes
    nothing about which instances score 0 -- it only avoids the crash."""
    raw = json.load(open(os.path.join(_ROOT, public_file)))
    filtered = []
    n_filtered_out = 0
    for r in raw:
        if r.get("status") == "ok" and r.get("prediction") is not None:
            filtered.append(r)
        else:
            n_filtered_out += 1
    return filtered, n_filtered_out


def run_full_stats(denom_ids, per_instance_by_row, out_dir, root):
    """Real (non-stub) bootstrap + Holm-correction reproduction.

    Fully recomputes, from public data/predictions/scorer plus a
    design-defined deterministic control (never a historical model replay):
      - PRIMARY_WITHIN_BACKBONE_TRACE_VS_G_FAMILY (n=4: TRACE-vs-G on each of
        GPT-4o, GPT-4o-mini, GPT-3.5, Haiku), all 3 bootstrap schemes.
      - The full frozen seven-comparison family, all 3 bootstrap schemes. The 3
        members compared against the OLD empty-evidence B3 control
        (G_vs_B3_4o, TRACE_vs_B3_4o, TRACE_vs_B3_mini) use
        evaluation/scoring/empty_evidence_control.build_empty_evidence_control()
        -- a pure, deterministic, all-zero vector over denom_ids, not a
        reconstruction, simulation, or replay of that control's historical
        model predictions (which are not shipped in this package by design;
        see docs/METHOD_OVERVIEW.md). This substitution is licensed by
        ECS_strict's own admissibility check: with no evidence pool ever
        populated, every instance scores 0 by construction, independent of
        backbone -- see reproduce/README.md.

    Both families, all 3 resampling schemes, are checked against the frozen
    reference in results/bootstrap/table1_bootstrap_results.json with NO
    absolute-tolerance band: p_raw and p_holm must be equal after rounding
    both sides to STRICT_MATCH_NDIGITS decimals; observed delta must match
    within STRICT_DELTA_ABS_TOL percentage points; comparison_id, family
    size, bootstrap method, and significance flags must match exactly.
    """
    import numpy as np
    import bootstrap_lib
    import holm_correction

    manifest = json.load(open(os.path.join(root, "data", "eagc_trace_789", "extended_manifest.json")))
    manifest_by_id = {it["instance_id"]: it for it in manifest["items"]}
    doc_ids = np.array([manifest_by_id[iid]["doc_id"] for iid in denom_ids])
    sectors = np.array([manifest_by_id[iid]["sector"] for iid in denom_ids])

    def arr_from_records(records):
        by_id = {r["instance_id"]: r for r in records}
        return np.array([by_id[iid]["ecs_strict_at_full"] for iid in denom_ids], dtype=float)

    def arr(row_id):
        return arr_from_records(per_instance_by_row[row_id])

    row_by_backbone = {
        "GPT-4o": {"G": 2, "TRACE": 6},
        "GPT-4o-mini": {"G": 3, "TRACE": 7},
        "GPT-3.5": {"G": 4, "TRACE": 8},
        "Haiku": {"G": 5, "TRACE": 9},
    }
    B3_FLAT_EVIDENCE_ROW = 1

    methods = [
        ("instance_bootstrap", lambda y1, y2: bootstrap_lib.paired_instance_bootstrap(y1, y2)),
        ("document_cluster_bootstrap", lambda y1, y2: bootstrap_lib.paired_document_cluster_bootstrap(y1, y2, doc_ids)),
        ("sector_stratified_bootstrap", lambda y1, y2: bootstrap_lib.paired_sector_stratified_document_bootstrap(y1, y2, doc_ids, sectors)),
    ]

    frozen = json.load(open(os.path.join(root, "results", "bootstrap", "table1_bootstrap_results.json")))

    # ---- Family 1: PRIMARY_WITHIN_BACKBONE_TRACE_VS_G (n=4) ----
    family1_backbones = ["GPT-3.5", "GPT-4o", "GPT-4o-mini", "Haiku"]  # matches frozen row order
    family1_raw = {mname: [] for mname, _ in methods}
    for backbone in family1_backbones:
        y1 = arr(row_by_backbone[backbone]["G"])
        y2 = arr(row_by_backbone[backbone]["TRACE"])
        for mname, mfunc in methods:
            family1_raw[mname].append(mfunc(y1, y2))

    family1_holm = {mname: holm_correction.holm([r["p_value"] for r in family1_raw[mname]]) for mname, _ in methods}

    frozen_family1_rows = frozen["PRIMARY_WITHIN_BACKBONE_TRACE_VS_G_FAMILY"]["rows"]
    frozen_family1_by_key = {(r["backbone"], r["bootstrap_method"]): r for r in frozen_family1_rows}

    family1_comparisons = []
    family1_all_match = True
    for mname, _ in methods:
        for i, backbone in enumerate(family1_backbones):
            rec_p_raw = family1_raw[mname][i]["p_value"]
            rec_p_holm = family1_holm[mname][i]
            rec_delta_pct = family1_raw[mname][i]["obs_delta"] * 100.0
            frow = frozen_family1_by_key[(backbone, mname)]
            f_p_raw, f_p_holm = float(frow["p_raw"]), float(frow["p_holm"])
            f_delta_pct = float(frow["obs_delta_pct_points"])
            f_family_size = int(frow["n_tests_in_family"])
            rec_sig_raw = rec_p_raw < 0.05
            rec_sig_holm = rec_p_holm < 0.05

            match = (
                _p_equal(rec_p_raw, f_p_raw)
                and _p_equal(rec_p_holm, f_p_holm)
                and _delta_equal(rec_delta_pct, f_delta_pct)
                and f_family_size == 4
                and _bool_str_equal(rec_sig_raw, frow["sig_raw_0.05"])
                and _bool_str_equal(rec_sig_holm, frow["sig_holm_0.05"])
            )
            family1_all_match = family1_all_match and match
            family1_comparisons.append({
                "family": "PRIMARY_WITHIN_BACKBONE_TRACE_VS_G_FAMILY", "comparison_id": f"TRACE_vs_G_{backbone}",
                "backbone": backbone, "bootstrap_method": mname, "family_size": f_family_size,
                "recomputed_p_raw": round(rec_p_raw, STRICT_MATCH_NDIGITS), "frozen_p_raw": round(f_p_raw, STRICT_MATCH_NDIGITS),
                "recomputed_p_holm": round(rec_p_holm, STRICT_MATCH_NDIGITS), "frozen_p_holm": round(f_p_holm, STRICT_MATCH_NDIGITS),
                "recomputed_obs_delta_pct_points": rec_delta_pct, "frozen_obs_delta_pct_points": f_delta_pct,
                "recomputed_sig_raw_0.05": rec_sig_raw, "recomputed_sig_holm_0.05": rec_sig_holm,
                "frozen_sig_raw_0.05": frow["sig_raw_0.05"], "frozen_sig_holm_0.05": frow["sig_holm_0.05"],
                "match": match,
            })

    # ---- Family 2: full frozen FROZEN_ORIGINAL_COMPARISON_FAMILY (n=7) ----
    # 4 members from shipped public predictions + 3 members against the
    # design-defined deterministic empty-evidence control (see docstring).
    y_g4o = arr(row_by_backbone["GPT-4o"]["G"])
    y_trace4o = arr(row_by_backbone["GPT-4o"]["TRACE"])
    y_gmini = arr(row_by_backbone["GPT-4o-mini"]["G"])
    y_tracemini = arr(row_by_backbone["GPT-4o-mini"]["TRACE"])
    y_b3_flat_evidence = arr(B3_FLAT_EVIDENCE_ROW)
    y_b3_old_4o = arr_from_records(build_empty_evidence_control(denom_ids))
    y_b3_old_mini = arr_from_records(build_empty_evidence_control(denom_ids))

    # Order matches the frozen family_definition's comparison set; Holm
    # correction is order-independent (rank-based), so this ordering only
    # affects report readability, not the resulting p_holm values.
    family2_def = [
        ("TRACE_vs_G_4o", y_g4o, y_trace4o),
        ("TRACE_vs_B3_4o", y_b3_old_4o, y_trace4o),
        ("G_vs_B3_4o", y_b3_old_4o, y_g4o),
        # These two comparison_id strings are retained because they are
        # serialized keys in the released frozen-results JSON.
        ("TRACE_vs_B3R1_4o", y_b3_flat_evidence, y_trace4o),
        ("G_vs_B3R1_4o", y_b3_flat_evidence, y_g4o),
        ("TRACE_vs_G_mini", y_gmini, y_tracemini),
        ("TRACE_vs_B3_mini", y_b3_old_mini, y_tracemini),
    ]
    FAMILY2_SIZE = len(family2_def)
    assert FAMILY2_SIZE == 7

    # The outer key is a legacy serialized identifier retained for exact
    # compatibility with the released results artifact. Its B3R1 token refers
    # to the B3 Flat-Evidence condition.
    frozen_family2_all_rows = frozen["B3R1_VS_G_AND_TRACE_FAMILY__FROZEN_ORIGINAL_COMPARISON_FAMILY"]["rows"]
    frozen_family2_rows = [r for r in frozen_family2_all_rows if r["eval_set"] == "extended_789"]
    frozen_family2_by_key = {(r["comparison_id"], r["bootstrap_method"]): r for r in frozen_family2_rows}

    family2_comparisons = []
    family2_all_match = True
    for mname, mfunc in methods:
        raw_results = [mfunc(y1, y2) for (_, y1, y2) in family2_def]
        p_raws = [r["p_value"] for r in raw_results]
        p_holms = holm_correction.holm(p_raws)
        for i, (comparison_id, _, _) in enumerate(family2_def):
            rec_p_raw = p_raws[i]
            rec_p_holm = p_holms[i]
            rec_delta_pct = raw_results[i]["obs_delta"] * 100.0
            frow = frozen_family2_by_key.get((comparison_id, mname))
            member_present = frow is not None
            f_family_size = int(frow["n_tests_in_family"]) if member_present else None
            f_p_raw = float(frow["p_raw"]) if member_present else None
            f_p_holm = float(frow["p_holm"]) if member_present else None
            f_delta_pct = float(frow["paired_abs_diff_pct_points"]) if member_present else None
            rec_sig_raw = rec_p_raw < 0.05
            rec_sig_holm = rec_p_holm < 0.05

            match = (
                member_present
                and f_family_size == FAMILY2_SIZE
                and _p_equal(rec_p_raw, f_p_raw)
                and _p_equal(rec_p_holm, f_p_holm)
                and _delta_equal(rec_delta_pct, f_delta_pct)
                and _bool_str_equal(rec_sig_raw, frow["sig_raw_0.05"])
                and _bool_str_equal(rec_sig_holm, frow["sig_holm_0.05"])
            )
            family2_all_match = family2_all_match and match
            family2_comparisons.append({
                "family": "FROZEN_ORIGINAL_COMPARISON_FAMILY", "comparison_id": comparison_id,
                "eval_set": "extended_789", "bootstrap_method": mname,
                "family_size": FAMILY2_SIZE,
                "recomputed_p_raw": round(rec_p_raw, STRICT_MATCH_NDIGITS),
                "frozen_p_raw": round(f_p_raw, STRICT_MATCH_NDIGITS) if member_present else None,
                "recomputed_p_holm": round(rec_p_holm, STRICT_MATCH_NDIGITS),
                "frozen_p_holm": round(f_p_holm, STRICT_MATCH_NDIGITS) if member_present else None,
                "recomputed_obs_delta_pct_points": rec_delta_pct, "frozen_obs_delta_pct_points": f_delta_pct,
                "recomputed_sig_raw_0.05": rec_sig_raw, "recomputed_sig_holm_0.05": rec_sig_holm,
                "frozen_sig_raw_0.05": frow["sig_raw_0.05"] if member_present else None,
                "frozen_sig_holm_0.05": frow["sig_holm_0.05"] if member_present else None,
                "match": match,
            })

    n_family1_units = len(family1_comparisons)
    n_family2_units = len(family2_comparisons)
    assert n_family1_units == 12
    assert n_family2_units == 21
    overall_full_stats_pass = family1_all_match and family2_all_match

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_B": bootstrap_lib.B_DEFAULT,
        "bootstrap_seed": bootstrap_lib.SEED_DEFAULT,
        "strict_match_ndigits": STRICT_MATCH_NDIGITS,
        "strict_delta_abs_tol_pct_points": STRICT_DELTA_ABS_TOL,
        "PRIMARY_WITHIN_BACKBONE_TRACE_VS_G_FAMILY_n4": {
            "family_definition": "4 within-backbone TRACE-vs-G comparisons on ECS_strict@full, Holm n=4, "
                                  "3 resampling schemes -- fully recomputed from public data/predictions/scorer.",
            "n_comparisons": n_family1_units,
            "comparisons_vs_frozen": family1_comparisons,
            "all_match_frozen": family1_all_match,
        },
        "FROZEN_ORIGINAL_COMPARISON_FAMILY_n7": {
            "family_definition": "7-comparison family (TRACE_vs_G_4o, TRACE_vs_B3_4o, G_vs_B3_4o, "
                                  "TRACE_vs_B3R1_4o, G_vs_B3R1_4o, TRACE_vs_G_mini, TRACE_vs_B3_mini), "
                                  "Holm n=7, 3 resampling schemes -- fully recomputed from public "
                                  "data/predictions/scorer plus the design-defined deterministic "
                                  "empty-evidence control (evaluation/scoring/empty_evidence_control.py) "
                                  "for the 3 members compared against the OLD (unshipped) empty-evidence "
                                  "B3 control. Not a replay of that control's historical model run.",
            "empty_evidence_control_provenance": "DESIGN_DEFINED_EMPTY_EVIDENCE_NEGATIVE_CONTROL",
            "n_comparisons": n_family2_units,
            "comparisons_vs_frozen": family2_comparisons,
            "all_match_frozen": family2_all_match,
        },
    }

    with open(os.path.join(out_dir, "reproduced_bootstrap_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    with open(os.path.join(out_dir, "reproduced_holm_comparisons.csv"), "w") as f:
        f.write("family,comparison_id,bootstrap_method,family_size,recomputed_p_raw,frozen_p_raw,"
                "recomputed_p_holm,frozen_p_holm,recomputed_obs_delta_pct_points,frozen_obs_delta_pct_points,match\n")
        for c in (family1_comparisons + family2_comparisons):
            f.write(f"{c['family']},{c['comparison_id']},{c['bootstrap_method']},{c['family_size']},"
                    f"{c['recomputed_p_raw']},{c['frozen_p_raw']},{c['recomputed_p_holm']},{c['frozen_p_holm']},"
                    f"{c['recomputed_obs_delta_pct_points']},{c['frozen_obs_delta_pct_points']},{c['match']}\n")

    report = {
        "generated_at_utc": result["generated_at_utc"],
        "overall_status": "PASS" if overall_full_stats_pass else "FAIL",
        "OFFLINE_FULL_STATS_REPRODUCTION_STATUS": "PASS" if overall_full_stats_pass else "FAIL",
        "PRIMARY_WITHIN_BACKBONE_TRACE_VS_G_FAMILY_n4_all_match_frozen": family1_all_match,
        "n_comparisons_checked_family1": n_family1_units,
        "FROZEN_ORIGINAL_COMPARISON_FAMILY_n7_all_match_frozen": family2_all_match,
        "n_comparisons_checked_family2": n_family2_units,
        "n_comparisons_checked_total": n_family1_units + n_family2_units,
        "strict_match_ndigits": STRICT_MATCH_NDIGITS,
        "strict_delta_abs_tol_pct_points": STRICT_DELTA_ABS_TOL,
        "outputs": ["reproduced_bootstrap_results.json", "reproduced_holm_comparisons.csv"],
    }
    with open(os.path.join(out_dir, "FULL_STATS_REPRODUCTION_REPORT.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"--full-stats: PRIMARY_WITHIN_BACKBONE_TRACE_VS_G_FAMILY_n4 all_match_frozen={family1_all_match} "
          f"({n_family1_units} cells); FROZEN_ORIGINAL_COMPARISON_FAMILY_n7 all_match_frozen={family2_all_match} "
          f"({n_family2_units} cells); strict match (round={STRICT_MATCH_NDIGITS}dp exact, "
          f"delta_tol={STRICT_DELTA_ABS_TOL}) -- see FULL_STATS_REPRODUCTION_REPORT.json")
    return overall_full_stats_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-stats", action="store_true", help="also recompute bootstrap CIs + Holm correction (slower, opt-in)")
    ap.add_argument("--output-dir", default=DEFAULT_OUT_DIR,
                     help="Directory to write reproduction output files to. "
                          "Defaults to ./reproduced_results under the CALLER's current "
                          "working directory (never inside this package's own source "
                          "tree), so this script can be run from a read-only checkout. "
                          "Created if it does not already exist.")
    args = ap.parse_args()
    OUT_DIR = args.output_dir

    if not os.path.isdir(GOLD_DIR):
        ap.error(
            "full benchmark data is not installed: missing "
            f"{GOLD_DIR}. Download and extract EAGC-TRACE_data.tar.gz "
            "as described in data/README.md, then rerun this command."
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    gold_data = load_gold()
    denom_ids = sorted(r["instance_id"] for r in gold_data)
    assert len(denom_ids) == 789, f"expected 789 gold instances, found {len(denom_ids)}"

    pred_map = json.load(open(PRED_MAP_PATH))
    expected = json.load(open(EXPECTED_PATH))
    expected_by_row = {r["row_id"]: r for r in expected["rows"]}

    all_mismatches = []
    row_results = []
    per_instance_by_row = {}
    overall_pass = True

    for row in pred_map["rows"]:
        rid = row["row_id"]
        pred_data, n_filtered_out = load_predictions_for_scoring(row["public_prediction_file"])

        prod = production_scorer.score_all(gold_data, pred_data, denominator_instance_ids=denom_ids)
        ref = reference_scorer.score_all(gold_data, pred_data, denominator_instance_ids=denom_ids)
        per_instance_by_row[rid] = prod["per_instance"]

        prod_by_id = {r["instance_id"]: r for r in prod["per_instance"]}
        ref_by_id = {r["instance_id"]: r for r in ref["per_instance"]}

        row_mismatches = []
        for iid in denom_ids:
            p, r = prod_by_id.get(iid), ref_by_id.get(iid)
            for k in production_scorer.K_VALUES:
                key = f"ecs_strict_at_{k}"
                pv, rv = (p or {}).get(key), (r or {}).get(key)
                if pv != rv:
                    row_mismatches.append({"row_id": rid, "instance_id": iid, "k": k, "production": pv, "reference": rv})
        all_mismatches.extend(row_mismatches)

        numerator = prod["summary"]["ecs_strict_at_full"]["numerator"]
        denominator = prod["summary"]["ecs_strict_at_full"]["denominator"]
        recomputed_pct = round(100.0 * numerator / denominator, 2)

        exp_row = expected_by_row.get(rid, {})
        exp_pct = exp_row.get("ECS_strict_pct")
        pct_match = (exp_pct is not None) and abs(recomputed_pct - exp_pct) <= TOLERANCE_PCT_POINTS
        if not pct_match:
            overall_pass = False

        row_results.append({
            "row_id": rid, "method": row["method"], "backbone": row["backbone"],
            "numerator": numerator, "denominator": denominator,
            "n_filtered_out_of_789_before_scoring": n_filtered_out,
            "recomputed_ECS_strict_pct": recomputed_pct,
            "expected_ECS_strict_pct": exp_pct,
            "match": pct_match,
            "n_production_reference_mismatches": len(row_mismatches),
        })

    if all_mismatches:
        overall_pass = False

    with open(os.path.join(OUT_DIR, "numerator_denominator_audit.csv"), "w") as f:
        f.write("row_id,method,backbone,numerator,denominator,recomputed_ECS_strict_pct,expected_ECS_strict_pct,match\n")
        for r in row_results:
            f.write(f"{r['row_id']},{r['method']},{r['backbone']},{r['numerator']},{r['denominator']},{r['recomputed_ECS_strict_pct']},{r['expected_ECS_strict_pct']},{r['match']}\n")

    with open(os.path.join(OUT_DIR, "production_reference_mismatches.json"), "w") as f:
        json.dump({"n_mismatches": len(all_mismatches), "mismatches": all_mismatches}, f, indent=2)

    n_comparisons_total = len(denom_ids) * len(production_scorer.K_VALUES) * len(row_results)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "n_rows": len(row_results),
        "n_instances": len(denom_ids),
        "n_k_values": len(production_scorer.K_VALUES),
        "n_production_reference_comparisons_total": n_comparisons_total,
        "n_production_reference_mismatches_total": len(all_mismatches),
        "rows": row_results,
    }
    with open(os.path.join(OUT_DIR, "REPRODUCTION_RESULT.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"overall_status={result['overall_status']}  rows={len(row_results)}  "
          f"comparisons={n_comparisons_total}  mismatches={len(all_mismatches)}")
    for r in row_results:
        flag = "OK" if r["match"] and r["n_production_reference_mismatches"] == 0 else "FAIL"
        print(f"  [{flag}] row{r['row_id']:02d} {r['method']}/{r['backbone']}: recomputed={r['recomputed_ECS_strict_pct']} expected={r['expected_ECS_strict_pct']} mismatches={r['n_production_reference_mismatches']}")

    full_stats_pass = True
    if args.full_stats:
        full_stats_pass = run_full_stats(denom_ids, per_instance_by_row, OUT_DIR, _ROOT)

    sys.exit(0 if (overall_pass and full_stats_pass) else 1)


if __name__ == "__main__":
    main()
