"""
Tests for benchmark_validation/ (the public AnnotationValidator package used
by reproduce/validate_data.py).

Two concerns:
  1. Import independence: the package must not depend on gold_construction/,
     _m4_validators/, or any sys.path side effect.
  2. Invalid-fixture behavior: each of the five required negative-path
     categories (schema-invalid, evidence ID outside the candidate pool,
     graph-integrity failure, V1 field-consistency failure, V2 closure
     failure) must be detected.

Run with: python -m pytest tests/test_benchmark_validation.py
      or: python tests/test_benchmark_validation.py
"""
import copy
import json
import os
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_BENCHMARK_VALIDATION_DIR = os.path.join(_REPO_ROOT, "benchmark_validation")

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import benchmark_validation  # noqa: E402
import benchmark_validation.annotation_validator as av_mod  # noqa: E402
import benchmark_validation.schema_validator as sv_mod  # noqa: E402
import benchmark_validation.graph_integrity_validator as gi_mod  # noqa: E402
from benchmark_validation import AnnotationValidator  # noqa: E402

DSET = os.path.join(_REPO_ROOT, "data", "eagc_trace_789")
TASKS = os.path.join(DSET, "tasks")
GOLD = os.path.join(DSET, "gold")
SCHEMA = os.path.join(_REPO_ROOT, "schemas", "eagc.schema.json")

BASE_TASK_FN = "DECK_2021_formal_000018.json"
BASE_GOLD_FN = "DECK_2021_task_formal_000018.json"


# ─── Import independence ───────────────────────────────────────────────────

def test_package_exports_annotation_validator():
    assert benchmark_validation.AnnotationValidator is av_mod.AnnotationValidator


def test_no_gold_construction_or_m4_validators_source_reference():
    banned = ("gold_construction", "_m4_validators")
    for fn in ("__init__.py", "annotation_validator.py", "schema_validator.py",
               "graph_integrity_validator.py"):
        text = open(os.path.join(_BENCHMARK_VALIDATION_DIR, fn), encoding="utf-8").read()
        for token in banned:
            assert token not in text, f"{fn} references banned token {token!r}"


def test_dependency_modules_load_from_within_the_package():
    for mod in (sv_mod, gi_mod):
        assert "benchmark_validation" in mod.__file__
        assert "gold_construction" not in mod.__file__
        assert "_m4_validators" not in mod.__file__


def test_import_does_not_mutate_sys_path():
    # Runs the actual before/after sys.path comparison in a fresh, isolated
    # Python subprocess rather than in this pytest process, whose own
    # sys.path is order-dependently polluted by other test modules'
    # collection-time side effects (unrelated to benchmark_validation).
    # This subprocess never sees any of that pollution, so it isolates
    # exactly what importing benchmark_validation itself does to sys.path.
    probe = (
        "import json, sys\n"
        "before = list(sys.path)\n"
        "import benchmark_validation\n"  # noqa: F401 -- import is the probe
        "after = list(sys.path)\n"
        "print(json.dumps({'before': before, 'after': after}))\n"
    )
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # (8) no manufactured PYTHONPATH; just bytecode hygiene
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, (
        f"isolated import-probe subprocess failed (exit code {result.returncode})\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    before, after = payload["before"], payload["after"]

    # (6) fails unless sys.path is exactly the same before and after import.
    assert before == after, (
        "importing benchmark_validation mutated sys.path in an isolated "
        f"interpreter (no other test module involved):\nbefore={before}\nafter={after}"
    )
    # (7) independently fails if either banned token occurs in the final path list.
    assert not any("gold_construction" in p for p in after), \
        f"gold_construction path present in sys.path after import: {after}"
    assert not any("_m4_validators" in p for p in after), \
        f"_m4_validators path present in sys.path after import: {after}"


# ─── Negative-path fixture behavior ────────────────────────────────────────

def _load_base():
    task = json.load(open(os.path.join(TASKS, BASE_TASK_FN)))
    gold = json.load(open(os.path.join(GOLD, BASE_GOLD_FN)))
    pool_ids = set()
    for items in task.get("candidate_pools", {}).values():
        for it in items:
            pool_ids.update(it.get("covered_evidence_ids", []))
    return gold["prediction"], pool_ids


def _find_node(pred, node_id):
    for n in pred["nodes"]:
        if n["node_id"] == node_id:
            return n
    return None


def _validator():
    return AnnotationValidator(schema_path=SCHEMA)


def test_valid_base_instance_passes_all_gates():
    pred, pool_ids = _load_base()
    report = _validator().validate(pred, pool_ids, strict=True)
    assert report.all_pass


def test_schema_invalid_prediction_fails_v0():
    pred, pool_ids = _load_base()
    bad = copy.deepcopy(pred)
    del bad["meta"]
    report = _validator().validate(bad, pool_ids, strict=True)
    assert not report.v0_pass
    assert any(e.error_code == "SCHEMA_ERROR" for e in report.errors)


def test_evidence_id_outside_candidate_pool_fails_v0():
    pred, pool_ids = _load_base()
    bad = copy.deepcopy(pred)
    bad["evidence_sets"][0]["links"][0]["evid_id"] = "DECK_2021:sent:999999"
    report = _validator().validate(bad, pool_ids, strict=True)
    assert not report.v0_pass
    assert any(e.error_code == "EVIDENCE_OUT_OF_POOL" for e in report.errors)


def test_graph_integrity_dangling_edge_fails_v0():
    pred, pool_ids = _load_base()
    bad = copy.deepcopy(pred)
    bad["edges"].append({"edge_type": "associated_with", "src": "claim_main",
                          "dst": "no_such_node_xyz"})
    report = _validator().validate(bad, pool_ids, strict=True)
    assert not report.v0_pass
    assert any(e.error_code == "E_EDGE_DST_REF" for e in report.errors)


def test_v1_year_inconsistency_fails_v1():
    pred, pool_ids = _load_base()
    bad = copy.deepcopy(pred)
    t = _find_node(bad, "target_virgin_wool")
    t["attrs"]["target_year"] = 2020
    t["attrs"]["baseline_year"] = 2022
    report = _validator().validate(bad, pool_ids, strict=True)
    assert report.v0_pass
    assert not report.v1_pass
    assert any(e.error_code == "YEAR_INCONSISTENCY" for e in report.errors)


def test_v2_closure_gap_is_flagged():
    pred, pool_ids = _load_base()
    bad = copy.deepcopy(pred)
    bad["edges"] = [e for e in bad["edges"] if e.get("edge_type") != "defines"]
    report = _validator().validate(bad, pool_ids, strict=True)
    assert report.v0_pass
    assert any(w.warning_code == "DEFINITION_MISSING" for w in report.warnings)


if __name__ == "__main__":
    test_package_exports_annotation_validator()
    test_no_gold_construction_or_m4_validators_source_reference()
    test_dependency_modules_load_from_within_the_package()
    test_import_does_not_mutate_sys_path()
    test_valid_base_instance_passes_all_gates()
    test_schema_invalid_prediction_fails_v0()
    test_evidence_id_outside_candidate_pool_fails_v0()
    test_graph_integrity_dangling_edge_fails_v0()
    test_v1_year_inconsistency_fails_v1()
    test_v2_closure_gap_is_flagged()
    print("OK: all benchmark_validation tests passed")
