# TRACE pipeline code

This directory holds the R1→G→V→SR-D→SR-L inference pipeline evaluated in
the paper: the five stage modules plus the two entry-point scripts that run
them. Run everything from the repository root (e.g.
`python trace/run_g_baseline.py ...`) — the entry-point scripts add this
directory to `sys.path` themselves.

The scripts that reproduce specific paper figures/tables/statistical tests
and the gold-benchmark construction/audit tooling are not included in this
release — this directory is scoped to the TRACE system itself.

## Canonical entry point

**`run_g_baseline.py`** produced the G/TRACE ablation results and the B1/B2
controls reported in the paper (the paper's actual B3 comparator was run
separately — see `run_flat_evidence.py` below). It runs a single stage or the
full TRACE pipeline on a task directory against a gold directory, and
evaluates with Module E:

```bash
python trace/run_g_baseline.py \
  --tasks data/eagc_trace_789/tasks \
  --gold  data/eagc_trace_789/gold \
  --output outputs/my_run \
  --model openai --model-name gpt-4o-2024-08-06 \
  --variant G_R1_V_SRD_SRL --concurrency 5
```

`--variant` selects the ablation condition:

| `--variant` | Paper condition |
|---|---|
| `empty-evidence` | **Not** the paper's B3 — this is a separate diagnostic negative control with an empty evidence pool (`ECS_strict=0.00%` by construction), retained for testing and statistical reproduction. It is not a reported paper condition. See below for the actual B3. |
| `G` | G — Schema-RAG baseline |
| `G_R1` | +R1 evidence routing |
| `G_R1_V` | +R1 +V (three-gate validation) |
| `G_R1_V_SRD` | +R1 +V +SR-D (deterministic repair) |
| `G_R1_V_SRD_SRL` | **TRACE** (full pipeline, +SR-L LLM repair) |
| `B1`, `B2` | routes to `baselines/b1_reachable_set.py` / `b2_rulegraph.py` |

**`run_flat_evidence.py` produces the paper's B3 Flat-Evidence row in
Table 1.** It receives the same full evidence pool as G without R1 routing,
presents that pool as a flat untyped list, and requests the same typed EAGC
output. It does not apply V, SR-D or SR-L. The script verifies the frozen
`flat_evidence_system_prompt.txt` and `G_system_prompt.txt` files by SHA-256
before execution. Frozen comparison identifiers in the results package remain
unchanged so the offline reproduction scripts can look them up exactly.

## Fail-closed generation failures (no silent mock substitution)

`g_generator.py`'s `_call_llm()` fails closed by default: if the LLM
response can't be parsed into a valid graph, or the underlying API call
itself raises, the instance is NOT silently completed with a fabricated
mock graph. Instead:

- `_call_llm()` raises `GenerationParseError` or `GenerationAPIError`
  (both carry diagnostic detail — truncated response content or the
  original exception).
- `run_g_baseline.py`'s per-instance loop (serial and
  `ThreadPoolExecutor` paths alike) catches these at the per-instance
  boundary — one instance's failure never crashes the batch — and records
  an explicit `status="generation_failed"` output record (empty
  prediction, visible error message, failing V0 gate), exactly the way the
  frozen GPT-3.5 rerun represents its own failures: kept in the
  denominator, never silently dropped, never fabricated.

The old mock-substitution behavior (`_fallback_mock_generation` /
`_call_llm_mock`) still exists for offline testing, but is now gated behind
an explicit, default-`False` `--allow-mock-fallback-on-error` flag on
`run_g_baseline.py`. **Never pass this flag for a real evaluation run.**
`MockLLMClient` in `llm/g_llm_client.py` is unrelated and unaffected — it's
an explicit opt-in provider selected via `--model mock`, not a
silent-failure path.

SR-L (`sr/sr_DL.py`'s `ModuleSR_L_LLM.run()`) similarly sets an explicit
`status="repair_error"` if its own LLM repair call raises. Because
`run_g_baseline.py`'s `apply_repair()` re-validates with Module V
immediately after SR-L (and Module V always overwrites `status` with
`audit_passed`/`audit_failed`), that repair-error signal would otherwise be
silently absorbed into what looks like an ordinary unrepaired-but-validated
record. `apply_repair()` now captures it explicitly and surfaces it as
`sr_l_status="repair_error"` plus `meta.sr_l_repair_error=True` on the final
output record, so it survives to `final_predictions.json` instead of
disappearing.

## Pipeline stages

| Directory | Paper module | Contents |
|---|---|---|
| `r1/` | **Module R1** (evidence routing) | `r1_router.py` |
| `g/` | **Module G** (schema-grounded generation) | `g_generator.py` |
| `v/` | **Module V** (three-gate validation) | `v_validator.py` — an independent TRACE-inference-time component; it does not import or share validator code with the benchmark's own public re-validator (`benchmark_validation/annotation_validator.py`) |
| `sr/` | **Module SR-D / SR-L** (repair) | `sr_DL.py` |
| `e/` | **Module E** (evaluation, trace-local copy) | `e_evaluation.py` — computes ECS strict/no-evidence/oracle and the field/evidence metrics. This trace-local copy is used by the TRACE/G/Flat-Evidence runner path (`run_g_baseline.py`, `run_flat_evidence.py`, both of which import it as `e.e_evaluation` via this directory's own relative-import layout) and exposes `ecs_*`/`ECS_*` identifiers in code. The separate copy used by the Table 1 scoring path, `evaluation/e/e_evaluation.py`, exposes `acs_*`/`ACS_*` identifiers instead — the two files are **not byte-identical**; their differences are limited to this naming convention (see `evaluation/e/e_evaluation.py`'s own docstring). The public display layer reconciles the two naming conventions via `evaluation/scoring/acs_ecs_adapter.py` where applicable. Both copies are retained because each is imported by a different public entry point; neither is unused or merely historical. |
| `llm/` | Shared LLM client | `g_llm_client.py` (Mock/OpenAI/Anthropic adapters), `config_loader.py` |
| `dvi/` | Disclosure Verifiability Index (Methods, Eq. 5) — a separate report-level metric, not part of the R1-G-V-SR-E claim pipeline | `build_doc_profile.py` — see `dvi/README.md` |
| `prompts/` | Frozen, hash-verified system prompts for `run_flat_evidence.py` | `flat_evidence_system_prompt.txt`, `G_system_prompt.txt` |
