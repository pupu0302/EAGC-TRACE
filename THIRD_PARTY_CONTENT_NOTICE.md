# Third-Party Content Notice

This dataset contains two categories of content under **different terms**. The project's own `DATA_LICENSE.md` (CC BY 4.0) does not cover everything in this package.

## Author-original content

The annotation methodology, graph schema, annotation labels, graph structures, evidence-set links and project-generated metadata are original to this project and covered by `DATA_LICENSE.md`.

## Third-party content — not covered by CC BY 4.0

The claim text and evidence excerpts embedded in `data/eagc_trace_789/tasks/` and `gold/` (e.g. `canonical_text`, `context.anchor_text`, candidate-pool excerpts, and claim-node `text` fields) are verbatim or near-verbatim quotes from real, publicly filed corporate ESG/sustainability reports. This license does not apply to, and grants no rights over, that underlying report text — it remains the property of the company that published it. Use beyond research/evaluation of this benchmark (republication, redistribution of the excerpts outside this benchmark's structure, commercial reuse, etc.) requires independently satisfying the original report's own copyright and terms of use. This notice is a factual boundary marker, not a legal opinion.

## Source documents

The 789-instance public set draws excerpts from 32 source report/filing-year combinations (`doc_id` prefixes in `tasks/`/`gold/` filenames); the authoritative list is `data/eagc_trace_789/extended_manifest.json` (`coverage.docs_covered`). An earlier internal construction population additionally included a 33rd document, `AVY_2019` (Avery Dennison), fully excluded from this release (see `data/DATA_CARD.md`) — no `AVY_2019` excerpts are shipped.
