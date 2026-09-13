# DVI — Disclosure Verifiability Index

Computes the paper's **Disclosure Verifiability Index (DVI)** (Methods,
Eq. 5): a deterministic, annotation-free, report-level measure of
evidential infrastructure — not a document-length or volume measure. It's
the equal-weight average of four standardised (z-scored) component scores,
each derived from the same kind of registries used elsewhere in this
release (`unit_registry.jsonl`, `evidence_registry.jsonl`,
`table_registry.jsonl` — one set of registries per corpus, built by the
`r0/` evidence-materialization stage, not shipped pre-built for the
market-scale corpus in this package):

| Paper component | Code field | What it captures |
|---|---|---|
| TD — textual and definitional grounding | `audit_textual_def_raw` / `_z` | References to reporting standards, emissions scopes, methodologies, boundaries, restatements |
| AV — assurance and verification | `audit_assurance_raw` / `_z` | External-assurance statements, references to named assurance/verification standards |
| NG — numeric grounding | `audit_numeric_raw` / `_z` | Numeric narrative content, ESG-specific measurement units, numeric table cells |
| TS — table structure | `audit_table_structure_raw` / `_z` | Prevalence and internal density of extracted tables relative to report scale |

`compute_auditability_index()` z-scores each raw component across the
corpus, then takes their unweighted mean — this is `DVI_d` in Eq. 5. The
result is written to the `auditability_index` field (called DVI in the
paper).

## Files

- `build_doc_profile.py` — entry point; scans the three registries and
  produces one profile row per canonical document.
- `dvi_core/` — supporting package: `io_stream.py` (registry streaming/parsing),
  `lexicon.py` (loads the term lists used by the TD/AV components; see
  `configs/auditability_lexicon.yaml`), `profile.py` (per-document feature
  aggregation).

## Usage

```bash
python trace/dvi/build_doc_profile.py \
  --evidence path/to/evidence_registry.jsonl \
  --units path/to/unit_registry.jsonl \
  --tables path/to/table_registry.jsonl \
  --lexicon configs/auditability_lexicon.yaml \
  --out outputs/dvi
```

**What this package ships vs. what DVI needs**: this package ships the
789-instance claim-level benchmark (`data/eagc_trace_789/`), which is
sufficient for reproducing the R1-G-V-SR-E claim-level pipeline and Table 1.
DVI, by contrast, is computed over the paper's separate market-scale report
corpus (thousands of reports, used only for the DVI-vs-external-rating
correlation analysis, not for claim-level ECS) — the pre-built registries
for that corpus are not included in this package. This module's code is
provided so the DVI computation itself is fully inspectable and
independently reproducible once you have built or obtained equivalent
registries for your own corpus, not to be run standalone against nothing.
