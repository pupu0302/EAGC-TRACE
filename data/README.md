# EAGC-TRACE benchmark data

The released EAGC-TRACE benchmark contains **789 task/gold pairs from 32 source documents across all 11 GICS sectors**.

The full task and adjudicated-gold payload is provided by the `EAGC-TRACE_data.tar.gz` archive under:

```text
data/eagc_trace_789/tasks/   # 789 task records
data/eagc_trace_789/gold/    # 789 adjudicated gold records
```

See `DATA_CARD.md` for benchmark composition, quality-control information, release provenance, and intended-use limitations.

## Validation and reproduction

After installing the data and result artifacts, validate the released benchmark:

```bash
python reproduce/validate_data.py
```

Then reproduce the reported Table 1 results from the frozen predictions:

```bash
python reproduce/reproduce_table1.py --output-dir reproduced_results
```

These reproduction steps are offline and make no model-API calls.

## Licensing

Project-generated benchmark components, including annotations, graph structures, evidence-set links, and metadata, are covered by CC BY 4.0 as described in `../DATA_LICENSE.md`.

Third-party corporate-report text and quoted evidence excerpts embedded in the benchmark are not covered by that license. See `../THIRD_PARTY_CONTENT_NOTICE.md` for details.
