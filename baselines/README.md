# Baselines B1 / B2

These two structural baselines are **axis-specific deterministic controls**
reported in **Extended Data Table 1** (not the paper's main Table 1). They
probe individual task axes rather than complete EAGC closure: neither
instantiates the full generator-based output interface, and neither
independently computes the V0-V2 validity gates, so no ECS value is
assigned to either. They do not call an LLM at all — both are pure Python,
deterministic, and reproducible offline.

| File | Paper baseline | What it does |
|---|---|---|
| `b1_reachable_set.py` | **B1 (Reachable-Set)** | Returns every atomic evidence id reachable from the frozen candidate pools, with no semantic evidence graph construction. Recovers evidence almost perfectly (Extended Data Table 1) but produces no graph — an evidence-recall upper bound. |
| `b2_rulegraph.py` | **B2 (RuleGraph)** | Builds a minimal, deterministic rule-based `GraphPrediction` (one Claim node, at least one evidence link) using pattern matching, with heuristic (not model-generated) evidence recovery. Its defining result is that this heuristic recovery is *much worse* than B1's reachable-set recall (Extended Data Table 1), illustrating that a deterministic graph alone doesn't solve evidence grounding. Depends on `b1_reachable_set.py` for its closed-candidate constraint. |

## B3 Flat-Evidence

B3 Flat-Evidence is implemented under `trace/` because it uses the same
model-calling and evidence-loading infrastructure as G and TRACE:

- `../trace/run_flat_evidence.py` produces the B3 Flat-Evidence row in
  Table 1. It receives the same complete evidence pool as G, presented as a
  flat untyped list, and produces the full EAGC output schema.
- `../trace/run_g_baseline.py --variant empty-evidence` runs a separate
  design-defined negative control with no evidence pool. It is not the
  Table 1 B3 comparator.

See `../trace/README.md` for usage and
`../results/EXPECTED_RESULTS.json` for the reported values.

Both `b1_reachable_set.py` and `b2_rulegraph.py` are invoked from
`trace/run_g_baseline.py --variant B1` / `--variant B2`.
