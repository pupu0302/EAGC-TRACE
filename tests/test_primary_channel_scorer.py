"""Tests for evaluation/scoring/primary_channel/{primary_channel_scorer,
primary_channel_scorer_crosscheck}.py and the primary-only
(gold importance=="primary") text/definition/table channel-recovery logic.

Synthetic-fixture tests exercise one rule each, without touching real
task/gold/prediction data. The required counterexample (T-COUNTER below)
proves that citing only a SECONDARY text item, while a different PRIMARY
text item exists uncited, is scored as an eligible primary-text MISS, not a
hit -- the exact behavior the mixed-importance denominator (766) silently
got wrong for every such instance.

A final integration test runs both scorer implementations against the real
public 789-gold-file / 11-unique-prediction-file release data and asserts
the full 26,037-decision cross-check plus the three fixed denominators
(text=750, table_cell=119, definition=190).

Run with: python -m pytest tests/test_primary_channel_scorer.py
      or: python tests/test_primary_channel_scorer.py
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SCORING_DIR = os.path.join(_REPO_ROOT, "evaluation", "scoring", "primary_channel")
_REPRODUCE_DIR = os.path.join(_REPO_ROOT, "reproduce")
for _p in (_SCORING_DIR, _REPRODUCE_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import primary_channel_scorer as prod  # noqa: E402
import primary_channel_scorer_crosscheck as xchk  # noqa: E402
import reproduce_primary_channel_metrics as cli  # noqa: E402

GOLD_DIR = os.path.join(_REPO_ROOT, "data", "eagc_trace_789", "gold")


def _gold_evidence_sets(links):
    """One evidence-set wrapping the given links, the shape both scorer
    modules expect (mirrors data/eagc_trace_789/gold/*.json's
    prediction.evidence_sets[*].links)."""
    return [{"set_id": "s1", "links": links}]


def _link(evid_id, importance, pool):
    return {"evid_id": evid_id, "importance": importance, "provenance": {"pool": pool}}


def _pred(cited_ids, coverage_map=None):
    return {
        "status": "ok",
        "prediction": {"evidence_sets": [{"set_id": "p1", "links": [{"evid_id": e} for e in cited_ids]}]},
        "meta": {"coverage_map": coverage_map} if coverage_map else {},
    }


# ---------------------------------------------------------------------
# Pool resolution
# ---------------------------------------------------------------------

def test_pool_resolution_from_provenance():
    assert prod.resolve_pool_type("text", "X:sent:1") == "text"
    assert prod.resolve_pool_type("table", "X:cell:1") == "table_cell"
    assert prod.resolve_pool_type("definition", "X:def:1") == "definition"


def test_pool_resolution_heuristic_fallback():
    assert prod.resolve_pool_type(None, "DOC:sent:000123") == "text"
    assert prod.resolve_pool_type(None, "DOC:cell:R1:C2") == "table_cell"
    assert prod.resolve_pool_type(None, "DOC:def:000001") == "definition"
    assert prod.resolve_pool_type(None, "DOC:unknown_token:1") == "text"  # default


# ---------------------------------------------------------------------
# Eligibility: only PRIMARY links count
# ---------------------------------------------------------------------

def test_eligibility_requires_a_primary_link_not_just_any_link():
    gold_evidence_sets = _gold_evidence_sets([
        _link("D:sent:1", "secondary", "text"),  # secondary only -> not eligible
    ])
    gold_data = [{"instance_id": "i1", "gold_evidence_sets": gold_evidence_sets}]
    elig = prod.compute_gold_eligibility(gold_data)
    assert "i1" not in elig["text"]

    gold_evidence_sets2 = _gold_evidence_sets([
        _link("D:sent:1", "secondary", "text"),
        _link("D:sent:2", "primary", "text"),
    ])
    gold_data2 = [{"instance_id": "i2", "gold_evidence_sets": gold_evidence_sets2}]
    elig2 = prod.compute_gold_eligibility(gold_data2)
    assert "i2" in elig2["text"]


# ---------------------------------------------------------------------
# T-COUNTER (required by the primary-only channel scientific contract,
# gold importance == "primary"): citing only a secondary text item while a
# different primary text item exists must be an eligible primary-text
# MISS, not a hit.
# ---------------------------------------------------------------------

def test_counterexample_secondary_only_citation_is_a_primary_miss():
    gold_evidence_sets = _gold_evidence_sets([
        _link("D:sent:100", "secondary", "text"),
        _link("D:sent:200", "primary", "text"),
    ])
    pred_record = _pred(cited_ids=["D:sent:100"])  # cites ONLY the secondary item

    hits = prod.score_instance_primary_hits(gold_evidence_sets, pred_record)
    assert hits["text"] == 0.0, "citing only the secondary text item must not count as a primary-text hit"

    # crosscheck module, exercised the way reproduce_primary_channel_metrics.py drives it
    flat = [("D:sent:100", "secondary", "text"), ("D:sent:200", "primary", "text")]
    elig = xchk.channel_eligibility({"i1": flat})
    assert "i1" in elig["text"]
    cited = xchk._cited_ids(pred_record)
    assert not ({"D:sent:200"} & cited)


def test_counterexample_citing_the_primary_item_itself_is_a_hit():
    gold_evidence_sets = _gold_evidence_sets([
        _link("D:sent:100", "secondary", "text"),
        _link("D:sent:200", "primary", "text"),
    ])
    pred_record = _pred(cited_ids=["D:sent:200"])
    hits = prod.score_instance_primary_hits(gold_evidence_sets, pred_record)
    assert hits["text"] == 1.0


# ---------------------------------------------------------------------
# Citation matching: exact id, and via coverage_map
# ---------------------------------------------------------------------

def test_hit_via_exact_evid_id():
    gold_evidence_sets = _gold_evidence_sets([_link("D:sent:1", "primary", "text")])
    pred_record = _pred(cited_ids=["D:sent:1"])
    hits = prod.score_instance_primary_hits(gold_evidence_sets, pred_record)
    assert hits["text"] == 1.0


def test_hit_via_coverage_map_unit_covers_gold_sentence():
    gold_evidence_sets = _gold_evidence_sets([_link("D:sent:5", "primary", "text")])
    # prediction cites a paragraph-level unit id, whose coverage_map says it covers D:sent:5
    pred_record = _pred(cited_ids=["D:unit:7"], coverage_map={"D:unit:7": ["D:sent:5", "D:sent:6"]})
    hits = prod.score_instance_primary_hits(gold_evidence_sets, pred_record)
    assert hits["text"] == 1.0


def test_miss_when_pred_cites_something_else_entirely():
    gold_evidence_sets = _gold_evidence_sets([_link("D:sent:5", "primary", "text")])
    pred_record = _pred(cited_ids=["D:sent:999"])
    hits = prod.score_instance_primary_hits(gold_evidence_sets, pred_record)
    assert hits["text"] == 0.0


def test_not_eligible_when_no_primary_link_of_that_pool_exists():
    gold_evidence_sets = _gold_evidence_sets([_link("D:cell:1:1", "primary", "table_cell")])
    pred_record = _pred(cited_ids=["D:cell:1:1"])
    hits = prod.score_instance_primary_hits(gold_evidence_sets, pred_record)
    assert hits["text"] is None  # no primary text link in gold -> not eligible, not a miss
    assert hits["table_cell"] == 1.0


# ---------------------------------------------------------------------
# Missing/unscoreable prediction: miss, but instance stays in the fixed
# denominator (contract point 3).
# ---------------------------------------------------------------------

def test_missing_prediction_is_a_miss_but_stays_eligible():
    gold_evidence_sets = _gold_evidence_sets([_link("D:sent:1", "primary", "text")])
    hits = prod.score_instance_primary_hits(gold_evidence_sets, pred_record=None)
    assert hits["text"] == 0.0

    gold_data = [{"instance_id": "i1", "gold_evidence_sets": gold_evidence_sets}]
    result = prod.score_all(gold_data, pred_by_instance={})  # i1 absent -> missing
    assert result["summary"]["text"]["eligible_n"] == 1
    assert result["summary"]["text"]["hit_n"] == 0


# ---------------------------------------------------------------------
# Cross-check agreement on a small synthetic multi-instance dataset
# ---------------------------------------------------------------------

def test_crosscheck_agrees_with_production_on_synthetic_dataset():
    gold_data = [
        {"instance_id": "i1", "gold_evidence_sets": _gold_evidence_sets([
            _link("D:sent:1", "secondary", "text"),
            _link("D:sent:2", "primary", "text"),
            _link("D:def:1", "primary", "definition"),
        ])},
        {"instance_id": "i2", "gold_evidence_sets": _gold_evidence_sets([
            _link("D:cell:1:1", "primary", "table_cell"),
        ])},
        {"instance_id": "i3", "gold_evidence_sets": _gold_evidence_sets([
            _link("D:sent:9", "primary", "text"),
        ])},
    ]
    flat_gold = {
        g["instance_id"]: [
            (link["evid_id"], link["importance"], link["provenance"]["pool"])
            for gs in g["gold_evidence_sets"] for link in gs["links"]
        ]
        for g in gold_data
    }
    pred_by_instance = {
        "i1": _pred(cited_ids=["D:sent:2"]),  # hits primary text; no def citation -> def miss
        "i2": _pred(cited_ids=["D:cell:1:1"]),  # hits primary table
        # i3 absent entirely -> missing/unscoreable -> miss, stays in denominator
    }

    r_prod = prod.score_all(gold_data, pred_by_instance)
    r_xchk = xchk.score_all(flat_gold, pred_by_instance)

    assert r_prod["eligibility"] == r_xchk["eligibility"]
    for iid in ("i1", "i2", "i3"):
        for c in prod.CHANNELS:
            assert r_prod["per_instance"][iid][c] == r_xchk["per_instance"][iid].get(c), (iid, c)

    assert r_prod["summary"]["text"] == {"eligible_n": 2, "hit_n": 1, "pct": 50.0}  # i1 hit, i3 miss
    assert r_prod["summary"]["definition"] == {"eligible_n": 1, "hit_n": 0, "pct": 0.0}
    assert r_prod["summary"]["table_cell"] == {"eligible_n": 1, "hit_n": 1, "pct": 100.0}


# ---------------------------------------------------------------------
# Full public-data integration: real 789 gold files, all 11 unique
# prediction files. Skips cleanly if the data payload is not installed.
# ---------------------------------------------------------------------

def test_full_public_data_denominators_and_crosscheck():
    if not os.path.isdir(GOLD_DIR):
        import pytest
        pytest.skip(f"data payload not installed at {GOLD_DIR}")

    import json
    gold_data = prod.load_public_gold(GOLD_DIR)
    flat_gold = xchk.load_public_gold_flat(GOLD_DIR)
    assert len(gold_data) == 789
    assert len(flat_gold) == 789

    elig_prod = prod.compute_gold_eligibility(gold_data)
    elig_xchk = xchk.channel_eligibility(flat_gold)
    assert len(elig_prod["text"]) == 750
    assert len(elig_prod["table_cell"]) == 119
    assert len(elig_prod["definition"]) == 190
    assert elig_prod == elig_xchk

    pred_map_path = os.path.join(_REPO_ROOT, "predictions", "prediction_provenance_map.json")
    pred_map = json.load(open(pred_map_path))
    unique_files = sorted({row["public_prediction_file"] for row in pred_map["rows"]})
    assert len(unique_files) == 11

    total_mismatches = 0
    for fname in unique_files:
        raw = json.load(open(os.path.join(_REPO_ROOT, fname)))
        pred_by_instance = {r["instance_id"]: r for r in raw if r.get("status") == "ok" and r.get("prediction") is not None}
        r_prod = prod.score_all(gold_data, pred_by_instance)
        r_xchk = xchk.score_all(flat_gold, pred_by_instance)
        for g in gold_data:
            iid = g["instance_id"]
            for c in prod.CHANNELS:
                if r_prod["per_instance"][iid][c] != r_xchk["per_instance"][iid].get(c):
                    total_mismatches += 1
    assert total_mismatches == 0


# ---------------------------------------------------------------------
# The denominator-equality condition (and every other named
# check) actually gates overall_status/exit-code -- not merely reported.
# ---------------------------------------------------------------------

def test_denominators_match_requires_all_three_exact():
    assert cli.denominators_match(750, 119, 190) is True
    assert cli.denominators_match(766, 119, 190) is False   # stale mixed text denominator
    assert cli.denominators_match(750, 118, 190) is False
    assert cli.denominators_match(750, 119, 264) is False   # stale mixed definition denominator


def test_denominator_check_gates_overall_pass():
    all_true = {
        "crosscheck_zero_mismatches": True,
        "denominators_match_expected": True,
        "per_file_method_independent_matches_frozen": True,
        "per_file_scoreable_conditional_matches_frozen": True,
        "table1_13_rows_match_frozen": True,
        "r2_fig2_24_rows_match_recomputation": True,
    }
    assert cli.compute_overall_pass(all_true) is True

    only_denominators_false = dict(all_true, denominators_match_expected=False)
    assert cli.compute_overall_pass(only_denominators_false) is False, (
        "denominators_match_expected must gate overall_pass on its own, even when every "
        "other check passes."
    )

    # every other single check must also independently gate the result
    for key in all_true:
        flipped = dict(all_true, **{key: False})
        assert cli.compute_overall_pass(flipped) is False, f"{key} does not gate overall_pass"


# ---------------------------------------------------------------------
# Scoreable-conditional definition (status=="ok", non-null
# prediction, >=1 node), built purely from already-cross-checked
# per-instance decisions -- no third scorer.
# ---------------------------------------------------------------------

def test_compute_scoreable_ids_excludes_missing_and_empty_graphs():
    raw = [
        {"instance_id": "a", "status": "ok", "prediction": {"nodes": [{"node_type": "Outcome"}]}},
        {"instance_id": "b", "status": "ok", "prediction": {"nodes": []}},          # empty graph
        {"instance_id": "c", "status": "failed", "prediction": None},                # failed
        {"instance_id": "d", "status": "ok", "prediction": None},                    # null inner
    ]
    ids = cli.compute_scoreable_ids(raw)
    assert ids == {"a"}


def test_scoreable_conditional_uses_existing_per_instance_decisions_only():
    gold_data = [
        {"instance_id": "i1", "gold_evidence_sets": _gold_evidence_sets([_link("D:sent:1", "primary", "text")])},
        {"instance_id": "i2", "gold_evidence_sets": _gold_evidence_sets([_link("D:sent:2", "primary", "text")])},
    ]
    pred_by_instance = {
        "i1": _pred(cited_ids=["D:sent:1"]),   # hit, scoreable
        "i2": _pred(cited_ids=[]),             # miss, scoreable (cites nothing but still has a node)
    }
    r = prod.score_all(gold_data, pred_by_instance)
    scoreable_ids = {"i1"}  # pretend i2's graph had 0 nodes and is therefore not scoreable
    sc = cli.compute_scoreable_conditional(r["eligibility"], r["per_instance"], scoreable_ids)
    assert sc["text"] == {"eligible_n": 1, "hit_n": 1, "pct": 100.0}  # only i1 is in the scoreable-conditional pool


def test_full_public_data_gate_still_passes_after_01a_change():
    if not os.path.isdir(GOLD_DIR):
        import pytest
        pytest.skip(f"data payload not installed at {GOLD_DIR}")
    import subprocess
    import tempfile
    cli_path = os.path.join(_REPRODUCE_DIR, "reproduce_primary_channel_metrics.py")
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.run(
            [sys.executable, cli_path, "--output-dir", tmpdir],
            cwd=tempfile.gettempdir(),  # run from OUTSIDE the package tree
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "overall_status=PASS" in proc.stdout
        assert "match=True" in proc.stdout  # denominators_match_expected reported true
        result_path = os.path.join(tmpdir, "PRIMARY_CHANNEL_REPRODUCTION_RESULT.json")
        import json as _json
        result = _json.load(open(result_path))
        assert result["overall_status"] == "PASS"
        assert result["checks"]["2_gold_denominators_exact"]["pass"] is True
        assert result["checks"]["3_per_file_method_independent_matches_frozen"]["pass"] is True
        assert result["checks"]["4_per_file_scoreable_conditional_matches_frozen"]["pass"] is True
        assert result["checks"]["6_r2_fig2_24_rows_match_recomputation"]["pass"] is True


if __name__ == "__main__":
    import inspect
    failures = 0
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
