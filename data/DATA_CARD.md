# EAGC-TRACE Benchmark — Data Card

## Overview

The EAGC-TRACE benchmark contains ESG-report claims paired with adjudicated gold evidence graphs comprising typed Target, Metric, Definition, Action and Outcome nodes, typed relations, and links to report-internal evidence.

## Released evaluation set

The benchmark was constructed from an initial pool of 990 candidate instances across 33 source documents. Following pre-release quality review, the public evaluation set contains **789 instances from 32 source documents**.

| | Initial benchmark pool | Released evaluation set |
|---|---:|---:|
| Instances | 990 | **789** |
| Source documents | 33 | **32** |

The 201 excluded candidate instances are documented in `excluded_instances_manifest.json`.

Of the 789 released instances:

- **238** form a nested **Audited Core** that received an additional independent review pass (`eagc_trace_789/audited_core_manifest.json`).
- The released gold annotations underwent pre-release quality review.

The released benchmark spans all 11 GICS sectors: Communication Services, Consumer Discretionary, Consumer Staples, Energy, Financials, Health Care, Industrials, Information Technology, Materials, Real Estate and Utilities.

## Excluded candidate instances

| Category | Count | Description |
|---|---:|---|
| `gold_annotation_quality_issue` | 101 | Gold annotation or underlying claim was not suitable for reliable inclusion. |
| `document_level_exclusion` | 30 | Instances from one source document (`AVY_2019`) that was excluded in full. |
| `evidence_pool_integrity_issue` | 29 | Gold evidence was inconsistent with the model-visible candidate evidence pool. |
| `schema_validation_failure` | 19 | Failed the release schema or structural validation criteria. |
| `qa_review_other` | 16 | Other pre-release QA exclusions recorded in the release manifest. |
| `relation_annotation_ambiguity` | 6 | Relation type could not be resolved reliably under the released schema. |

**Total excluded: 201.**

Instance-level exclusion membership is provided in `excluded_instances_manifest.json`.

## Validation

All **789/789** released task/gold pairs pass the blocking release-validation criteria. Additional non-blocking structural diagnostics do not affect benchmark membership; their definitions and outputs are documented with the release validation materials.

## Release sanitization

Internal review-only metadata was removed from affected gold files before release. This sanitization removed metadata fields only; claim text, graph structure, evidence links and evidence content were not modified.

## Third-party content and licensing

The benchmark contains third-party corporate-report text and quoted evidence excerpts. These materials are not covered by the project's CC BY 4.0 license.

The license applies only to project-generated benchmark components such as annotations, graph structures, evidence-set links and metadata. See `../DATA_LICENSE.md` and `../THIRD_PARTY_CONTENT_NOTICE.md`.

## Intended use and limitations

- The benchmark evaluates claim-level evidence closure against a fixed candidate evidence pool. It is not a general-purpose ESG fact-checking or scoring system, and benchmark outputs should not be interpreted as audit opinions on reporting companies.
- Exclusion categories reflect project-specific pre-release quality-control decisions and are not intended as a general taxonomy of annotation failure.
- Coverage of 11 sectors and 32 source documents is not necessarily representative of broader sustainability-reporting populations. See the manuscript Methods for corpus and benchmark construction details.
