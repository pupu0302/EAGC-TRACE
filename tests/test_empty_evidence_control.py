"""
Tests for evaluation/scoring/empty_evidence_control.py.

Asserts the deterministic empty-evidence negative control used to reproduce
the 3 old-B3-dependent members of the frozen 7-comparison Holm family
(G_vs_B3_4o, TRACE_vs_B3_4o, TRACE_vs_B3_mini) is exactly what it claims to
be: a pure, design-defined, all-zero vector over the given instance_ids --
never a replay, reconstruction, or simulation of the historical old-B3 model
run, and never dependent on model name, network access, or any prediction
file.

Run with: python -m pytest tests/test_empty_evidence_control.py
      or: python tests/test_empty_evidence_control.py
"""
import inspect
import json
import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SCORING_DIR = os.path.join(_REPO_ROOT, "evaluation", "scoring")
for _p in (_SCORING_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from empty_evidence_control import build_empty_evidence_control, PROVENANCE  # noqa: E402

MANIFEST_PATH = os.path.join(_REPO_ROOT, "data", "eagc_trace_789", "extended_manifest.json")


def _manifest_ids():
    manifest = json.load(open(MANIFEST_PATH))
    return [it["instance_id"] for it in manifest["items"]]


def test_output_count_matches_789():
    ids = _manifest_ids()
    assert len(ids) == 789
    out = build_empty_evidence_control(ids)
    assert len(out) == 789


def test_order_matches_input_instance_ids():
    ids = _manifest_ids()
    out = build_empty_evidence_control(ids)
    assert [r["instance_id"] for r in out] == ids


def test_all_values_are_zero():
    ids = _manifest_ids()
    out = build_empty_evidence_control(ids)
    assert all(r["ecs_strict_at_full"] == 0.0 for r in out)
    assert all(r["provenance"] == PROVENANCE for r in out)


def test_no_model_name_network_or_prediction_file_dependency():
    # The function's only parameter is the id list itself -- no model/backbone
    # name, no file path, no API client, nothing that could route to a real
    # inference call or a shipped/unshipped prediction file.
    sig = inspect.signature(build_empty_evidence_control)
    assert list(sig.parameters.keys()) == ["instance_ids"]
    # Purely in-memory: works on ids that were never written to disk anywhere.
    synthetic_ids = [f"SYNTH_{i}" for i in range(5)]
    out = build_empty_evidence_control(synthetic_ids)
    assert len(out) == 5
    assert all(r["ecs_strict_at_full"] == 0.0 for r in out)


def test_alignment_survives_instance_order_permutation():
    ids = _manifest_ids()
    shuffled = ids[:]
    random.Random(0).shuffle(shuffled)
    assert shuffled != ids  # sanity: the shuffle actually changed the order

    out_original = build_empty_evidence_control(ids)
    out_shuffled = build_empty_evidence_control(shuffled)

    # Joining by instance_id (never by position) -- the pattern
    # reproduce/reproduce_table1.py's run_full_stats() uses for every row --
    # must recover the identical all-zero vector regardless of which order
    # the ids were supplied in.
    by_id_original = {r["instance_id"]: r["ecs_strict_at_full"] for r in out_original}
    by_id_shuffled = {r["instance_id"]: r["ecs_strict_at_full"] for r in out_shuffled}
    assert by_id_original == by_id_shuffled
    assert set(by_id_original) == set(ids)
    assert len(by_id_shuffled) == 789


if __name__ == "__main__":
    test_output_count_matches_789()
    test_order_matches_input_instance_ids()
    test_all_values_are_zero()
    test_no_model_name_network_or_prediction_file_dependency()
    test_alignment_survives_instance_order_permutation()
    print("OK: all empty_evidence_control tests passed")
