# Reproducing the reported results offline

`reproduce_table1.py` recomputes every row of `results/table1_final_789.csv` from scratch, reading only:

- `data/eagc_trace_789/gold/` (installed from the separate data archive; see `data/README.md`)
- `predictions/prediction_provenance_map.json` + `predictions/table1/*.json` (public frozen predictions)
- `results/EXPECTED_RESULTS.json` (the values it checks against)

**No network access, no model API calls.** The row list is read dynamically from `predictions/prediction_provenance_map.json` — this script does not hardcode "13 rows" or any filename, so it stays correct if that map is regenerated.

The release result table has 13 rows: its first nine rows correspond to the
paper's Table 1, and its final four rows reproduce the GPT-4o ablation used in
Fig. 3b.

## Run it

Clone this repository, then install the separate data and results downloads
as described in the root `README.md`. From the repository root, run:

```bash
python reproduce/validate_data.py
python reproduce/reproduce_table1.py --output-dir reproduced_results
# or, with a Makefile-provided default output dir:
make reproduce-table1 OUTPUT_DIR=reproduced_results
```

`--output-dir` defaults to `./reproduced_results` under your current working
directory (never inside this package's own source tree), so this script runs
cleanly against a read-only checkout.

Exit code `0` only if **both** of the following hold:
1. Every row's recomputed `ECS_strict` matches `results/EXPECTED_RESULTS.json` within floating-point tolerance.
2. The production scorer (`evaluation/scoring/production_wrapper/production_scorer.py`) and the independently-written reference scorer (`evaluation/scoring/reference_implementation/reference_scorer.py`) agree on every single instance, for every K value, for every row — 789 instances x 4 K values x 13 rows = **41,028** individual instance-K-row comparisons, zero exceptions. Any single mismatch is treated as a hard failure and reported, never averaged away, per `evaluation/FROZEN_SCORING_CONTRACT.yaml`'s cross-validation requirement.

The production scorer's `ecs_strict_at_full` value comes directly from the frozen `evaluation/e/e_evaluation.py` evaluator; that evaluator's own citation-matching, recall and closure computation is the canonical, plainly located `evaluation/scoring/ecs_metric.py` module — see its docstring for the exact Definition-A contract it implements. The reference scorer does not import this module (its independence from the production evaluator is itself part of the frozen contract's cross-validation requirement).

## What it writes

Under `--output-dir` (see above):
- `numerator_denominator_audit.csv` — per-row numerator/denominator and recomputed-vs-expected comparison.
- `production_reference_mismatches.json` — full detail on any scorer disagreement (empty list if none, which is the expected/required state).
- `REPRODUCTION_RESULT.json` — the overall PASS/FAIL verdict plus per-row detail.
- With `--full-stats` (see below), also `reproduced_bootstrap_results.json`, `reproduced_holm_comparisons.csv`, and `FULL_STATS_REPRODUCTION_REPORT.json`.

## A note on failed/missing predictions

Each public prediction file in `predictions/table1/` is a fixed 789-length array (one entry per public benchmark instance), including explicit `status`/`failure_code` entries for instances where generation failed or the instance is missing from the source run — this is intentional and matches the frozen scoring contract's fixed-denominator rule (a failure counts as 0, it is never dropped from the denominator). Before handing predictions to the scorers, this script filters out those null-prediction placeholder entries — the scorers themselves treat an *absent* instance_id as "score 0 for this instance" via the fixed `denominator_instance_ids` list, so this filtering changes nothing about the final score; it only matches the input shape the frozen scorer code expects (a present-but-null `"prediction"` field is not a shape it was written to handle).

## `--full-stats` (bootstrap / Holm correction)

The default run above only reproduces the point-estimate `ECS_strict` values (fast, seconds). Passing `--full-stats` additionally recomputes bootstrap confidence intervals and Holm-corrected significance tests using `evaluation/scoring/bootstrap_lib.py` and `evaluation/scoring/holm_correction.py` (slower — 10,000 bootstrap resamples per comparison), and checks the result against the frozen reference in `results/bootstrap/table1_bootstrap_results.json`. This fully reproduces **both** frozen Holm families:

- the primary within-backbone TRACE-vs-G family (n=4 comparisons, all 3 resampling schemes — 12 statistical units), and
- the frozen 7-comparison `FROZEN_ORIGINAL_COMPARISON_FAMILY` (n=7 comparisons, all 3 resampling schemes — 21 statistical units), including 3 members compared against the design-defined empty-evidence control.

Those 3 members use a **design-defined deterministic zero vector** (`evaluation/scoring/empty_evidence_control.py::build_empty_evidence_control`), not a replay or reconstruction of historical model predictions. Their identifiers are retained verbatim from the frozen results file for compatibility. With no populated evidence pool, the control scores exactly 0 on every instance by construction of ECS_strict.

Every one of the 33 recomputed statistical units (12 + 21) is checked against the frozen reference with **no absolute p-value tolerance band**: `p_raw`/`p_holm` must be equal once both sides are rounded to 4 decimals, the observed delta must match within `1e-12` percentage points, and `comparison_id`/family size/bootstrap method/significance flags must match exactly. `FULL_STATS_REPRODUCTION_REPORT.json`'s `OFFLINE_FULL_STATS_REPRODUCTION_STATUS` is `PASS` only if all 33 units match under this strict rule.

## Reproducing the primary-only channel-recovery metrics offline

`reproduce_primary_channel_metrics.py` recomputes the H@K_txt / H@K_tbl / definition-channel recovery metrics reported in `results/table1_final_789.csv`, `results/EXPECTED_RESULTS.json`, `results/figure_data/r2_fig2__fig2_channel_recovery.csv`, `results/figure_data/r3_fig3ab__fig3b_ablations.csv`, and `public_assets/main_results.csv`, using gold evidence links filtered to `importance == "primary"` only. It reads only:

- `data/eagc_trace_789/gold/` (public gold)
- `predictions/prediction_provenance_map.json` + `predictions/table1/*.json` (public frozen predictions)
- `results/primary_channel_summary.json` (the frozen reference values it checks against)

**No network access, no model API calls, no dependency on any internal `_audit` file or governance directory.** Channel eligibility is decided on gold provenance alone (an instance is eligible for a channel iff it carries at least one PRIMARY gold link of that pool), giving fixed, method-independent denominators of **750** text-bearing, **119** primary-table-bearing, and **190** definition-bearing instances — identical across every prediction file. A missing or unscoreable prediction on an otherwise-eligible instance counts as a miss; it is never dropped from the denominator.

**Method-independent vs. scoreable-conditional.** Every channel is reported two ways (matching `results/figure_data/r2_fig2__fig2_channel_recovery.csv`'s existing columns): the **method-independent** denominator is the full fixed gold-eligible set (750/119/190) regardless of what any model produced; the **scoreable-conditional** denominator additionally restricts to instances where that specific prediction file was *scoreable* — `status == "ok"`, a non-null `prediction`, and a predicted graph with at least one node — intersected with the same gold-eligible set. The numerator in both cases is the count of hits inside that denominator, computed from the identical, already-cross-checked per-instance decisions (no separate scoring pass, no third implementation).

```bash
# Run from the repository root:
python reproduce/reproduce_primary_channel_metrics.py --output-dir reproduced_results
# or:
make reproduce-primary-channel OUTPUT_DIR=reproduced_results
```

Exit code `0` (`overall_status == "PASS"`) only if **all six** of the following hold:
1. The two independently-written scorer implementations (`evaluation/scoring/primary_channel/primary_channel_scorer.py` and `..._crosscheck.py`) agree on every per-instance, per-channel decision for every one of the 11 unique prediction files — 789 instances x 3 channels x 11 files = **26,037** individual decisions, zero exceptions.
2. The recomputed gold-side denominators are exactly text=750, table_cell=119, definition=190 — this is a real gate on the exit code, not a display-only diagnostic.
3. Every one of the 11 unique prediction files' **method-independent** text/table/definition summaries (`eligible_n`, `hit_n`, `pct`) match `results/primary_channel_summary.json` exactly.
4. Every one of the 11 unique prediction files' **scoreable-conditional** summaries match the same frozen file exactly.
5. All 13 released result rows' recomputed text/definition/table percentages match the frozen reference in `results/primary_channel_summary.json`.
6. All 24 rows of `results/figure_data/r2_fig2__fig2_channel_recovery.csv` (method-independent and scoreable-conditional numerator/denominator/percentage, for all 4 backbones x {G, TRACE} x {text, table_primary, definition}) match the recomputation.
