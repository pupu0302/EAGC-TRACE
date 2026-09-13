# EAGC-TRACE

Code and reproducibility materials for **“Auditable Machine Reading
for Grounding Corporate Sustainability Claims with Report Evidence.”**

EAGC formalises claim-level evidence closure for sustainability-report claims.
TRACE is a training-free framework that constructs, validates and selectively
repairs typed, citation-bearing evidence graphs.

Mengpu Liu, Yiyang Duan, Qi Yang, Yi Wu, Yilei Zhao, Lei Xiao, Wei Liu and
Wei Yang Bryan Lim.

## Data and results

The benchmark data and frozen model outputs are distributed separately from
the code repository:

| Download | Contents |
|---|---|
| [**Data — EAGC-TRACE_data.tar.gz**](https://drive.google.com/file/d/1PtnmqtcNmGY27P7-JKBsBn3HbYjboDPF/view?usp=drive_link) | 789 tasks, adjudicated gold evidence graphs and benchmark metadata |
| [**Predictions and results — EAGC-TRACE_results.tar.gz**](https://drive.google.com/file/d/1PpbQ3iKl2HKcTz7UEhhYn2SR3GbyiDj6/view?usp=drive_link) | Frozen Table 1 predictions, aggregate results and statistical outputs |

After cloning this repository, run the following commands from its root
directory. The archives already contain the expected directory layout, so no
files need to be moved manually.

```bash
tar -xzf /path/to/EAGC-TRACE_data.tar.gz --strip-components=1
tar -xzf /path/to/EAGC-TRACE_results.tar.gz --strip-components=1
```

This installs the benchmark under `data/eagc_trace_789/`, the frozen model
outputs under `predictions/`, and the reported results under `results/`.

## Repository contents

| Path | Contents |
|---|---|
| `trace/` | TRACE implementation, live runners and DVI analysis |
| `baselines/` | B1 Reachable-Set and B2 RuleGraph controls |
| `evaluation/` | Canonical ECS metric, evaluation and statistical utilities |
| `schemas/` | Prediction schema |
| `quickstart/` | Synthetic smoke test requiring no data or API access |
| `reproduce/` | Benchmark validation and offline result reproduction |
| `configs/` | Model configurations |
| `data/` | Dataset documentation and lightweight manifests; tasks and gold are added from the data download |
| `tests/` | Public regression and release-integrity tests |

## Paper-to-code map

| Paper component | Public implementation |
|---|---|
| EAGC graph schema | `schemas/eagc.schema.json` |
| R1 Typed Evidence Router | `trace/r1/r1_router.py` |
| G Schema-Grounded Generator | `trace/g/g_generator.py` |
| V Validity Gate | `trace/v/v_validator.py` |
| SR-D and SR-L repair | `trace/sr/sr_DL.py` |
| B1 and B2 controls | `baselines/` |
| B3 Flat-Evidence comparator | `trace/run_flat_evidence.py` |
| Evidence Closure Score (ECS) | `evaluation/scoring/ecs_metric.py` |
| Disclosure Verifiability Index (DVI) | `trace/dvi/` |

For a concise description of the method, see `docs/METHOD_OVERVIEW.md`.

## Model and inference settings

Frozen predictions were generated with `temperature=0` and no explicit
`top_p` override.

| Provider model identifier | API access date(s), UTC | Maximum output tokens |
|---|---|---:|
| OpenAI `gpt-4o` alias | 7 and 9 March 2026; 21 July 2026 | 4,096 for G/TRACE; 8,192 for Flat-Evidence |
| OpenAI `gpt-4o-mini-2024-07-18` | 7 and 9 March 2026 | 4,096 |
| OpenAI `gpt-3.5-turbo-0125` | 2 September 2026 | 4,096 |
| Anthropic `claude-haiku-4-5-20251001` | 2 September 2026 | 8,192 |

The historical GPT-4o artifacts retain the provider alias rather than a
resolved snapshot identifier. `configs/models.yaml` uses
`gpt-4o-2024-08-06` for current live execution; this does not retrospectively
identify the snapshot used by every historical alias call.

Reference annotation used OpenAI `gpt-4o` and Anthropic
`claude-sonnet-4-6` on 21 March 2026 (UTC), with `temperature=0`, a
4,000-token output limit and no explicit `top_p` override. Seed 42 was sent
to OpenAI; the Anthropic request did not expose a seed parameter. Annotation
prompts are included in the data download under `benchmark_construction/prompts/`.

## Installation

```bash
pip install -r requirements.txt
```

A Conda specification is also provided in `environment.yml`.

## Quickstart

The source-only synthetic smoke test requires neither benchmark data nor an
API key:

```bash
python quickstart/run_quickstart.py
```

## Reproduce the reported results

After installing the data and results downloads as described above:

```bash
python reproduce/validate_data.py
python reproduce/reproduce_table1.py --output-dir reproduced_results
```

These commands make no provider-API calls. They validate all 789 benchmark
instances and recompute the nine reported Table 1 rows together with the four
GPT-4o ablation rows used for Fig. 3 from the frozen predictions. See
`reproduce/README.md` for full statistics and primary-channel reproduction.

## Optional live inference

Generating new predictions requires the relevant provider API key:

```bash
export OPENAI_API_KEY=...

python trace/run_g_baseline.py \
  --tasks data/eagc_trace_789/tasks \
  --gold data/eagc_trace_789/gold \
  --output outputs/my_run \
  --model openai \
  --model-name gpt-4o-2024-08-06 \
  --variant G_R1_V_SRD_SRL
```

Provider-side changes and decoding non-determinism mean that a live run may
differ from the frozen predictions. Offline reproduction remains
deterministic.

## License

Code is released under the MIT License (`LICENSE`). Project-generated
annotations and metadata are released under CC BY 4.0 (`DATA_LICENSE.md`).
Third-party corporate-report materials, including source text and evidence
excerpts, are subject to their respective terms and are described in
`THIRD_PARTY_CONTENT_NOTICE.md`.

## Citation

Citation metadata is provided in `CITATION.cff`.

## Contact

Mengpu Liu — mengpu001@e.ntu.edu.sg
