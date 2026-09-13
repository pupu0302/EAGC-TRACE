"""
Canonical Evidence Closure Score (ECS) core: citation matching, Definition-A
recall, oracle recall, and strict/oracle/no-evidence closure decisions.

This module is the single, plainly located, executable definition of what
this codebase means by "ECS" at the per-instance level. It is shared by the
two public evaluator entry points:

  - trace/e/e_evaluation.py       (used by the public G/TRACE/Flat-Evidence
                                    runners; exposes ecs_*/ECS_* keys)
  - evaluation/e/e_evaluation.py  (used by the Table 1 production wrapper;
                                    exposes acs_*/ACS_* keys)

Both files import and call this module for the functionality documented
below. The callers retain different output-key prefixes for compatibility,
but use the same citation-matching, recall, and closure calculations.

Definition A -- the exact contract implemented here
-----------------------------------------------------
A claim's gold evidence is expressed as a collection of "gold evidence
sets" (``gold_evidence_sets``), each an alternative, independently
sufficient bundle of evidence links. Each link carries an ``evid_id`` and
an ``importance`` tag (``"primary"`` or otherwise). **This version of ECS
does not filter recall by ``importance``: every link in a gold set,
primary and secondary alike, is included in that set's recall
computation.** ``importance`` is read by other, non-ECS parts of the
evaluators (e.g. Hit@K-primary breakdowns); it plays no role in any
function in this module.

For one gold evidence set, "recall" is the fraction of that set's
(deduplicated) gold IDs that are "cited" by the prediction -- see citation
matching below. Definition-A per-instance recall (``best_recall`` in the
callers) is the **maximum** of this per-set recall over all of the
instance's gold evidence sets: citing any one gold-admissible set in full
is sufficient, not all of them. An individual empty gold set (no
resolvable ``evid_id`` values) contributes nothing and is skipped, not
scored as 0 or 1. An instance with zero gold evidence sets at all is
vacuously satisfied: recall is defined as ``1.0`` (nothing to close).

Oracle recall is the same maximum-over-gold-sets computation, but against
a supplied "oracle" universe of evidence IDs (e.g. every ID present in a
task's full candidate pool) instead of the prediction's own cited IDs. It
answers "could this evidence have been found at all", not "did the
prediction find it". It is only defined when an oracle universe is
supplied; when none is supplied the function returns ``None`` rather than
``0.0`` or ``1.0``, matching prior behavior where the caller only computes
this metric when a separate oracle-retrieval run supplies the candidate
pool.

Closure decisions combine a caller-supplied validity signal (the boolean
AND of a prediction's structural/repair validator gates, computed
upstream of this module and passed in) with a recall value against the
fixed closure threshold ``CLOSURE_THRESHOLD = 0.99``:

  - ``no_evidence_closure``: 1.0 iff validity alone passed. Recall is not
    consulted at all -- this is the "structurally valid, evidence
    unchecked" signal.
  - ``strict_closure``: 1.0 iff validity passed AND recall >= 0.99.
  - ``oracle_closure``: 1.0 iff validity passed AND oracle recall >= 0.99;
    ``None`` (not 0.0/1.0) if oracle recall itself is ``None``.

Citation matching (``is_cited``) treats a gold ID as cited if either (a)
it appears verbatim in the prediction's cited evidence-ID set, or (b) an
optional coverage map resolves it to a paragraph/unit-level ID that does
appear in that set (the LLM cited the containing unit rather than the
exact sentence/cell gold cites). Duplicate ``evid_id`` values within a
gold set's links collapse naturally: recall is computed over the
deduplicated *set* of a gold set's IDs, so a repeated ID is not
double-counted in either the denominator or a hit.

Explicitly out of scope for THIS version (Definition A), by design and by
the extraction task's mandate -- do not add any of this:
  - filtering recall to ``importance == "primary"`` links only;
  - unioning primary IDs across multiple gold sets;
  - any closure threshold other than 0.99;
  - any ECS@K / traversal-order / K-truncation behavior (that lives
    entirely in evaluation/scoring/production_wrapper/production_scorer.py
    and evaluation/scoring/reference_implementation/reference_scorer.py,
    which this module is never imported by);
  - any provider/model/backbone branching of any kind.

Contract for this module
-------------------------
Deterministic and side-effect-free: no data loading, no filesystem writes,
no networking, no model/provider branching, no pandas dependency, no CLI
state, no module-level mutable state. Every function is a pure function of
its arguments.

This module is deliberately NOT imported by
evaluation/scoring/reference_implementation/reference_scorer.py -- that
scorer's independence from the production evaluator's implementation is a
requirement of the frozen scoring contract
(evaluation/FROZEN_SCORING_CONTRACT.yaml::cross_validation_requirement).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

# The single closure threshold used by every strict/oracle decision in this
# codebase's current ECS definition. Not configurable, not read from any
# environment/config source -- changing this is a Definition change, out of
# scope for this extraction.
CLOSURE_THRESHOLD: float = 0.99


def build_sentence_to_unit_map(
    coverage_map: Optional[Dict[str, List[str]]],
) -> Dict[str, str]:
    """Invert an optional ``{unit_id: [sentence_id, ...]}`` coverage map into
    a ``{sentence_id: unit_id}`` reverse-lookup map for ``is_cited``.

    Returns an empty dict if ``coverage_map`` is ``None`` or empty -- callers
    do not need to special-case a missing coverage map.
    """
    sent_to_unit: Dict[str, str] = {}
    if coverage_map:
        for unit_id, sentence_ids in coverage_map.items():
            for sentence_id in sentence_ids:
                sent_to_unit[sentence_id] = unit_id
    return sent_to_unit


def is_cited(
    gold_id: str,
    pred_evidence_ids: Set[str],
    sentence_to_unit: Dict[str, str],
) -> bool:
    """True if ``gold_id`` is covered by the prediction's cited evidence IDs,
    either by an exact ID match or, when ``gold_id`` is a sentence/cell ID
    that resolves (via ``sentence_to_unit``) to a paragraph-level unit ID,
    by that unit ID being cited instead.
    """
    if gold_id in pred_evidence_ids:
        return True
    unit_id = sentence_to_unit.get(gold_id)
    return unit_id is not None and unit_id in pred_evidence_ids


def gold_set_recall(
    gold_ids: Iterable[str],
    pred_evidence_ids: Set[str],
    sentence_to_unit: Dict[str, str],
) -> Optional[float]:
    """Recall for one gold evidence set: the fraction of its deduplicated
    gold IDs that are cited (per ``is_cited``).

    Returns ``None`` for an empty set (no resolvable IDs) -- an empty
    individual gold set contributes nothing to Definition-A's max-over-sets
    recall and must be skipped, not scored as 0.0 or 1.0, by the caller.
    """
    ids = set(gold_ids)
    if not ids:
        return None
    hits = sum(1 for gid in ids if is_cited(gid, pred_evidence_ids, sentence_to_unit))
    return hits / len(ids)


def _gold_set_ids(gold_set: Dict) -> List[str]:
    """Extract the (possibly duplicate) ``evid_id`` values from one gold
    evidence set's links, in order, skipping links with no ``evid_id``.
    Importance is deliberately not consulted: Definition A includes every
    link, primary and secondary alike.
    """
    return [
        link.get("evid_id")
        for link in gold_set.get("links", [])
        if link.get("evid_id")
    ]


def best_recall_over_gold_sets(
    gold_sets: List[Dict],
    pred_evidence_ids: Set[str],
    sentence_to_unit: Dict[str, str],
) -> float:
    """Definition-A per-instance recall: the maximum, over all of the
    instance's gold evidence sets, of that set's recall (all importance
    levels included). Individual empty gold sets are skipped and do not
    affect the maximum. An instance with no gold evidence sets at all is
    vacuously satisfied (returns ``1.0``).
    """
    if not gold_sets:
        return 1.0
    best = 0.0
    for gold_set in gold_sets:
        recall = gold_set_recall(_gold_set_ids(gold_set), pred_evidence_ids, sentence_to_unit)
        if recall is not None and recall > best:
            best = recall
    return best


def oracle_recall_over_gold_sets(
    gold_sets: List[Dict],
    oracle_evidence_ids: Optional[Set[str]],
) -> Optional[float]:
    """Oracle counterpart of ``best_recall_over_gold_sets``: the maximum,
    over all of the instance's gold evidence sets, of the fraction of that
    set's (deduplicated) gold IDs present in ``oracle_evidence_ids`` --
    exact-ID membership only, no coverage-map resolution (the oracle
    universe is itself an ID set, not a prediction to be covered).

    Returns ``None`` when ``oracle_evidence_ids`` is ``None`` (no oracle
    experiment was supplied for this run) -- this is a distinct value from
    ``0.0``, and callers must not conflate the two. When an oracle universe
    is supplied and there are no gold evidence sets at all, returns ``1.0``
    (vacuously satisfied), matching ``best_recall_over_gold_sets``.
    """
    if oracle_evidence_ids is None:
        return None
    if not gold_sets:
        return 1.0
    best = 0.0
    for gold_set in gold_sets:
        ids = set(_gold_set_ids(gold_set))
        if not ids:
            continue
        recall = sum(1 for gid in ids if gid in oracle_evidence_ids) / len(ids)
        if recall > best:
            best = recall
    return best


def no_evidence_closure(validity_passed: bool) -> float:
    """The "no-evidence" closure decision: 1.0 iff ``validity_passed`` is
    True. Recall is not consulted -- this is the structural/repair-validity
    signal alone, independent of whether evidence was ever checked.
    """
    return 1.0 if validity_passed else 0.0


def strict_closure(
    validity_passed: bool,
    recall: float,
    threshold: float = CLOSURE_THRESHOLD,
) -> float:
    """The strict closure decision: 1.0 iff ``validity_passed`` is True AND
    ``recall >= threshold``.
    """
    return 1.0 if (validity_passed and recall >= threshold) else 0.0


def oracle_closure(
    validity_passed: bool,
    oracle_recall: Optional[float],
    threshold: float = CLOSURE_THRESHOLD,
) -> Optional[float]:
    """The oracle closure decision: 1.0 iff ``validity_passed`` is True AND
    ``oracle_recall >= threshold``; 0.0 if that condition fails but
    ``oracle_recall`` is a number; ``None`` if ``oracle_recall`` itself is
    ``None`` (no oracle experiment supplied).
    """
    if oracle_recall is None:
        return None
    return 1.0 if (validity_passed and oracle_recall >= threshold) else 0.0


__all__ = [
    "CLOSURE_THRESHOLD",
    "build_sentence_to_unit_map",
    "is_cited",
    "gold_set_recall",
    "best_recall_over_gold_sets",
    "oracle_recall_over_gold_sets",
    "no_evidence_closure",
    "strict_closure",
    "oracle_closure",
]
