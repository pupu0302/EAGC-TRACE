# Method Overview

A short orientation to the TRACE pipeline and the released results. The first
nine rows of `results/table1_final_789.csv` correspond to the paper's Table 1;
four additional GPT-4o rows support the Fig. 3 ablation. For code-level detail
see `trace/README.md`; for offline reproduction see `reproduce/README.md`.

## The TRACE pipeline: R1 -> G -> V -> SR-D -> SR-L

TRACE constructs a typed, citation-bearing evidence graph for a claim in five
stages. The paper reports standalone G and full TRACE in Table 1 and examines
the intervening GPT-4o configurations in Fig. 3b:

1. **R1 — Evidence routing** (`trace/r1/r1_router.py`): narrows the full
   per-claim evidence pool (prose sentences, table cells, definitions) down
   to a routed candidate set before generation.
2. **G — Schema-grounded generation** (`trace/g/g_generator.py`): given the
   claim and its (routed, if R1 ran; otherwise full) evidence pool, an LLM
   produces a graph of typed nodes (Target / Metric / Definition / Action /
   Outcome), typed edges (`supports`, `defines`, `measures`,
   `associated_with`, `contradicts`), and evidence citations, closed against
   the visible evidence pool (closed-world: no `evid_id` may be invented).
   **G alone (no R1/V/SR-D/SR-L) is itself a reported Table 1 row**, not
   just an internal stage.
3. **V — Protocol validation** (`trace/v/v_validator.py::ModuleV_Validator`):
   validates a generated graph and flags failing predictions for repair
   rather than silently accepting them.
4. **SR-D — Deterministic repair** (`trace/sr/sr_DL.py`, `ModuleSR_D`):
   rule-based, non-LLM repair of specific, mechanically-detectable gate
   failures.
5. **SR-L — Selective LLM repair** (`trace/sr/sr_DL.py`, `ModuleSR_L_LLM`):
   a second, targeted LLM call to repair whatever SR-D could not, only on
   instances still failing after SR-D.

## B3 Flat-Evidence, G, and TRACE

| Name | What it is | Paper placement |
|---|---|---|
| **B3 Flat-Evidence** | The paper's B3 comparator (`trace/run_flat_evidence.py`). Receives the same full evidence pool as G (no R1 routing), presented as a flat, untyped list instead of the typed EAGC schema categories, but still asked to produce the full typed EAGC output — isolating the effect of schema-grounded structure/typing while holding evidence access fixed. | Row 1, GPT-4o only |
| **G** | The pipeline's generation stage alone (no R1/V/SR-D/SR-L). | Table 1, four backbones |
| **TRACE** | The full pipeline, `G+R1+V+SR-D+SR-L`. | Table 1, four backbones |

## Scoring: ECS

Every prediction is scored by the frozen evaluator
(`evaluation/e/e_evaluation.py::ModuleE_Evaluator`), whose citation-matching
and closure decisions are implemented in the canonical
`evaluation/scoring/ecs_metric.py` module, through two independently written
wrapper implementations (`evaluation/scoring/production_wrapper/` and
`evaluation/scoring/reference_implementation/`) required to agree on every
instance — see `evaluation/FROZEN_SCORING_CONTRACT.yaml`. The shipped
results implement **Definition A**: recall against the best single
gold-admissible evidence set, at a 0.99 threshold.

## More detail

- Code map and pipeline internals: `trace/README.md`
- Offline reproduction of Table 1 and the full-statistics workflow:
  `reproduce/README.md`
