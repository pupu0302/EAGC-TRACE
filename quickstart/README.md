# Quickstart

Two modes, both against this package's actual scoring code:

## 1. Default: source-only synthetic smoke check

```bash
python quickstart/run_quickstart.py
```

Makes **no model or network call** and reads **no real task/gold/prediction
data**. It loads every fabricated fixture from
`evaluation/scoring/fixtures/synthetic_fixtures.py::build_all_fixtures`,
scores each one with **both** the production scorer
(`evaluation/scoring/production_wrapper/production_scorer.py`) and the
independently-written reference scorer
(`evaluation/scoring/reference_implementation/reference_scorer.py`), and
requires the two implementations to agree with each other and with every
fixture's own documented K/full expectation
(`evaluation/FROZEN_SCORING_CONTRACT.yaml`'s cross-validation requirement).
It establishes its own import paths and does not depend on any other
module's `sys.path` side effect to resolve its imports.

This is a **smoke check of the scoring code path only** — it never computes
or prints a real `ecs_strict` number and is **not** a Table 1 result. It
exits nonzero if the two scorers disagree with each other or with a
fixture's documented expectation.

## 2. `--release-sample`: real 5-instance sample from the full release

```bash
python quickstart/run_quickstart.py --release-sample
```

Scores the same five real `BAX_2020` instances
(`BAX_2020:cc:000059/000075/000081/000096/000098`) this quickstart has
always used as its illustration, but reads them from the **normal composed
release layout** — `data/eagc_trace_789/tasks/`, `data/eagc_trace_789/gold/`,
and `predictions/table1/row02_G_GPT_4o.json` — rather than a bundled
duplicate copy. It requires the full benchmark and results to already be
present in this checkout (see `data/README.md`); if either is missing, it
**exits nonzero with an actionable message** — it never silently falls back
to load-only or skips scoring. On the full release it reproduces the
package's accepted 5/5 sample summary (`ecs_strict_at_full=1` for all five
instances).

## Where the real numbers live

- `results/EXPECTED_RESULTS.json` — the frozen metric values
  (ECS_strict/oracle/noev, CIs, F1_role, H_cite, etc.) for the nine Table 1
  rows and four GPT-4o ablation rows, with their provenance.
- `results/table1_final_789.csv` / `.tex` — the same 13-row release result
  table in spreadsheet/LaTeX form.
- `reproduce/reproduce_table1.py` — the full public, zero-network
  reproduction: production + reference scorer, 0-mismatch cross-check,
  compared against `EXPECTED_RESULTS.json` within float tolerance.

Reproducing Table 1 correctly end to end (the full 789-instance benchmark,
0-mismatch production/reference cross-check) is the job of
**`reproduce/reproduce_table1.py`** at the repository root — neither
quickstart mode attempts that itself, and neither depends on it.

The full benchmark and frozen results are provided as separate data and
results downloads. See the root `README.md` for download and extraction
instructions before running `--release-sample` or the full reproduction
command.

## Full pipeline usage

For running the actual TRACE inference pipeline (R1→G→V→SR-D→SR-L) rather
than just scoring existing predictions, see `trace/README.md` and
`trace/run_g_baseline.py --help`.
