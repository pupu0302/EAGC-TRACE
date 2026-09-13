"""
Regression tests for evaluation/scoring/ecs_metric.py.

These are regression tests for the released Definition-A behavior shared by
trace/e/e_evaluation.py and evaluation/e/e_evaluation.py. They specify the
behavior of the current release.

Run with: python -m pytest tests/test_ecs_metric.py
      or: python tests/test_ecs_metric.py
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SCORING_DIR = os.path.join(_REPO_ROOT, "evaluation", "scoring")
for _p in (_SCORING_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ecs_metric  # noqa: E402


def _gold_set(*evid_ids, importance="secondary"):
    return {"links": [{"evid_id": eid, "importance": importance} for eid in evid_ids]}


def _mixed_gold_set(pairs):
    """pairs: list of (evid_id, importance)."""
    return {"links": [{"evid_id": eid, "importance": imp} for eid, imp in pairs]}


# ---------------------------------------------------------------------------
# 1. Mixed primary/secondary gold set -- importance never filters recall.
# ---------------------------------------------------------------------------

def test_mixed_primary_secondary_gold_set_no_importance_filter():
    gold_sets = [_mixed_gold_set([("a", "primary"), ("b", "secondary")])]
    # Only the secondary link is cited; primary is not. Definition A does not
    # filter by importance, so recall = 1/2, not 0 and not computed over
    # primary-only or secondary-only subsets.
    pred_ids = {"b"}
    sent_to_unit = {}
    recall = ecs_metric.best_recall_over_gold_sets(gold_sets, pred_ids, sent_to_unit)
    assert recall == 0.5
    # Citing only the primary link gives the same 1/2 -- symmetric treatment.
    recall2 = ecs_metric.best_recall_over_gold_sets(gold_sets, {"a"}, sent_to_unit)
    assert recall2 == 0.5
    # Citing both gives full recall.
    recall3 = ecs_metric.best_recall_over_gold_sets(gold_sets, {"a", "b"}, sent_to_unit)
    assert recall3 == 1.0


# ---------------------------------------------------------------------------
# 2. Two gold sets, only the second is fully covered -> it wins (max-over-sets).
# ---------------------------------------------------------------------------

def test_two_sets_only_second_fully_covered_wins():
    gold_sets = [_gold_set("a", "b", "c"), _gold_set("x", "y")]
    pred_ids = {"x", "y"}  # fully covers the 2nd set only; 0/3 of the 1st
    recall = ecs_metric.best_recall_over_gold_sets(gold_sets, pred_ids, {})
    assert recall == 1.0


# ---------------------------------------------------------------------------
# 3. Secondary-only winning set -- preserves current A behavior (no primary
#    requirement anywhere in recall/closure).
# ---------------------------------------------------------------------------

def test_secondary_only_winning_set():
    gold_sets = [
        _mixed_gold_set([("p1", "primary")]),                 # not cited
        _mixed_gold_set([("s1", "secondary"), ("s2", "secondary")]),  # fully cited
    ]
    pred_ids = {"s1", "s2"}
    recall = ecs_metric.best_recall_over_gold_sets(gold_sets, pred_ids, {})
    assert recall == 1.0
    # And it still counts as strict closure once validity also passes.
    assert ecs_metric.strict_closure(True, recall) == 1.0


# ---------------------------------------------------------------------------
# 4. Exact-ID matching and unit-to-atomic coverage-map matching.
# ---------------------------------------------------------------------------

def test_exact_id_matching():
    assert ecs_metric.is_cited("sent:1", {"sent:1", "sent:2"}, {}) is True
    assert ecs_metric.is_cited("sent:3", {"sent:1", "sent:2"}, {}) is False


def test_coverage_map_unit_to_atomic_matching():
    coverage_map = {"para:1": ["sent:1", "sent:2"]}
    sent_to_unit = ecs_metric.build_sentence_to_unit_map(coverage_map)
    assert sent_to_unit == {"sent:1": "para:1", "sent:2": "para:1"}
    # Gold cites the atomic sentence; prediction cites the containing unit.
    assert ecs_metric.is_cited("sent:1", {"para:1"}, sent_to_unit) is True
    # A sentence not covered by any cited unit, and not cited exactly, misses.
    assert ecs_metric.is_cited("sent:9", {"para:1"}, sent_to_unit) is False

    gold_sets = [_gold_set("sent:1", "sent:2")]
    recall = ecs_metric.best_recall_over_gold_sets(gold_sets, {"para:1"}, sent_to_unit)
    assert recall == 1.0


def test_build_sentence_to_unit_map_handles_none_and_empty():
    assert ecs_metric.build_sentence_to_unit_map(None) == {}
    assert ecs_metric.build_sentence_to_unit_map({}) == {}


# ---------------------------------------------------------------------------
# 5. Duplicate IDs -- collapse in the denominator and are not double-counted.
# ---------------------------------------------------------------------------

def test_duplicate_ids_collapse_in_recall_denominator():
    gold_sets = [_gold_set("a", "a", "b")]  # 3 links, 2 distinct IDs
    # Citing only "a" (regardless of its duplication) gives 1/2, not 1/3 or 2/3.
    recall = ecs_metric.best_recall_over_gold_sets(gold_sets, {"a"}, {})
    assert recall == 0.5


def test_duplicate_ids_in_gold_set_recall_helper():
    recall = ecs_metric.gold_set_recall(["a", "a", "a"], {"a"}, {})
    assert recall == 1.0  # a single distinct ID, fully cited


# ---------------------------------------------------------------------------
# 6. Empty gold-set collection and an empty individual set.
# ---------------------------------------------------------------------------

def test_empty_gold_set_collection_is_vacuously_satisfied():
    assert ecs_metric.best_recall_over_gold_sets([], {"anything"}, {}) == 1.0
    assert ecs_metric.best_recall_over_gold_sets([], set(), {}) == 1.0


def test_empty_individual_gold_set_is_skipped_not_scored():
    # An empty set contributes nothing; a later, non-empty, cited set still
    # determines the (correct) max.
    gold_sets = [{"links": []}, _gold_set("a")]
    recall = ecs_metric.best_recall_over_gold_sets(gold_sets, {"a"}, {})
    assert recall == 1.0
    # gold_set_recall itself returns None (not 0.0) for an empty ID collection.
    assert ecs_metric.gold_set_recall([], {"a"}, {}) is None


def test_empty_individual_set_does_not_win_over_nonempty_uncited_set():
    # If the only non-empty set is NOT cited, the empty set must not make
    # the instance look satisfied: result should be 0.0, not 1.0.
    gold_sets = [{"links": []}, _gold_set("a", "b")]
    recall = ecs_metric.best_recall_over_gold_sets(gold_sets, {"z"}, {})
    assert recall == 0.0


# ---------------------------------------------------------------------------
# 7. Recall just below and at 0.99 -- the closure threshold boundary.
# ---------------------------------------------------------------------------

def test_recall_just_below_threshold_fails_strict():
    assert ecs_metric.strict_closure(True, 0.989999) == 0.0


def test_recall_at_threshold_passes_strict():
    assert ecs_metric.strict_closure(True, 0.99) == 1.0
    assert ecs_metric.strict_closure(True, 1.0) == 1.0


def test_recall_just_below_threshold_fails_oracle():
    assert ecs_metric.oracle_closure(True, 0.989999) == 0.0


def test_recall_at_threshold_passes_oracle():
    assert ecs_metric.oracle_closure(True, 0.99) == 1.0


def test_closure_threshold_constant_is_099():
    assert ecs_metric.CLOSURE_THRESHOLD == 0.99


# ---------------------------------------------------------------------------
# 8. Validity false with recall one -- validity gates every closure decision.
# ---------------------------------------------------------------------------

def test_validity_false_with_perfect_recall_fails_every_closure():
    assert ecs_metric.no_evidence_closure(False) == 0.0
    assert ecs_metric.strict_closure(False, 1.0) == 0.0
    assert ecs_metric.oracle_closure(False, 1.0) == 0.0


# ---------------------------------------------------------------------------
# 9. Strict/oracle/no-evidence relationships.
# ---------------------------------------------------------------------------

def test_no_evidence_closure_ignores_recall_entirely():
    # no_evidence_closure takes no recall argument at all -- validity alone.
    assert ecs_metric.no_evidence_closure(True) == 1.0
    assert ecs_metric.no_evidence_closure(False) == 0.0


def test_strict_implies_would_be_consistent_with_no_evidence():
    # Whenever strict passes, validity must have passed, so no_evidence_closure
    # on the same validity value must also be 1.0 (strict never passes while
    # no_evidence_closure would report 0 for the same instance).
    for validity in (True, False):
        for recall in (0.0, 0.5, 0.99, 1.0):
            if ecs_metric.strict_closure(validity, recall) == 1.0:
                assert ecs_metric.no_evidence_closure(validity) == 1.0


def test_oracle_none_is_distinct_from_zero_and_one():
    assert ecs_metric.oracle_closure(True, None) is None
    assert ecs_metric.oracle_recall_over_gold_sets([_gold_set("a")], None) is None
    assert ecs_metric.oracle_recall_over_gold_sets([], None) is None


def test_oracle_recall_vacuous_when_gold_sets_empty_but_oracle_supplied():
    assert ecs_metric.oracle_recall_over_gold_sets([], {"anything"}) == 1.0


def test_oracle_recall_exact_id_membership_only_no_coverage_map():
    # Oracle recall checks membership in the oracle ID universe directly;
    # it takes no coverage map argument at all (by design/signature).
    gold_sets = [_gold_set("a", "b")]
    assert ecs_metric.oracle_recall_over_gold_sets(gold_sets, {"a", "b"}) == 1.0
    assert ecs_metric.oracle_recall_over_gold_sets(gold_sets, {"a"}) == 0.5
    assert ecs_metric.oracle_recall_over_gold_sets(gold_sets, set()) == 0.0


# ---------------------------------------------------------------------------
# 10. Deterministic-set behavior -- repeated calls on identical inputs agree.
# ---------------------------------------------------------------------------

def test_deterministic_repeatability():
    gold_sets = [_gold_set("a", "b"), _mixed_gold_set([("c", "primary"), ("d", "secondary")])]
    pred_ids = {"a", "c"}
    results = {
        ecs_metric.best_recall_over_gold_sets(gold_sets, pred_ids, {})
        for _ in range(20)
    }
    assert len(results) == 1


# ---------------------------------------------------------------------------
# 11. Absence of provider/model branches and runtime side effects.
# ---------------------------------------------------------------------------

def test_module_has_no_provider_or_model_branching():
    import inspect
    source = inspect.getsource(ecs_metric)
    banned_substrings = ("gpt", "haiku", "claude", "openai", "anthropic", "3.5", "4o")
    lowered = source.lower()
    for token in banned_substrings:
        assert token not in lowered, f"provider/model token {token!r} found in ecs_metric.py"


def test_module_has_no_data_loading_or_io():
    import inspect
    source = inspect.getsource(ecs_metric)
    banned_calls = ("open(", "requests.", "urllib", "os.remove", "os.makedirs",
                    "subprocess.", "socket.", "write(")
    for token in banned_calls:
        assert token not in source, f"disallowed I/O construct {token!r} found in ecs_metric.py"


def test_module_does_not_depend_on_pandas():
    assert not hasattr(ecs_metric, "pd")
    assert not hasattr(ecs_metric, "pandas")
    assert "pandas" not in sys.modules or "pandas" not in dir(ecs_metric)


def test_functions_are_pure_no_shared_mutable_state():
    # Calling the same function twice with fresh-but-equal inputs must not
    # be influenced by any earlier call (no module-level accumulation).
    gold_sets_a = [_gold_set("a")]
    gold_sets_b = [_gold_set("a")]
    r1 = ecs_metric.best_recall_over_gold_sets(gold_sets_a, {"a"}, {})
    r2 = ecs_metric.best_recall_over_gold_sets(gold_sets_b, set(), {})
    r3 = ecs_metric.best_recall_over_gold_sets(gold_sets_a, {"a"}, {})
    assert r1 == r3 == 1.0
    assert r2 == 0.0


def test_reference_scorer_does_not_import_ecs_metric():
    # Independence of the reference implementation is an acceptance property
    # of the frozen scoring contract; this module must not be imported there.
    ref_path = os.path.join(_REPO_ROOT, "evaluation", "scoring",
                             "reference_implementation", "reference_scorer.py")
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_source = f.read()
    assert "ecs_metric" not in ref_source


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL: {t.__name__}: {e}")
    if failures:
        print(f"\n{len(failures)}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests PASSED")


if __name__ == "__main__":
    _run_all()
