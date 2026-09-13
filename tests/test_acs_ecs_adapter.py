"""
Regression test for evaluation/scoring/acs_ecs_adapter.py.

Asserts the ACS->ECS adapter is a pure key rename:
  (a) every acs_*/ACS_* key's VALUE is identical before/after (only the key
      name changed, never the value, never its type, never its identity);
  (b) no acs_*/ACS_* key remains in the output;
  (c) every non-acs-prefixed key/value is byte-identical before/after.

Run with: python -m pytest tests/test_acs_ecs_adapter.py
      or: python tests/test_acs_ecs_adapter.py
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SCORING_DIR = os.path.join(_REPO_ROOT, "evaluation", "scoring")
for _p in (_SCORING_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from acs_ecs_adapter import acs_to_ecs  # noqa: E402


FIXTURE = {
    "instance_id": "TEST_2024:cc:000001",
    "acs_strict": 1.0,
    "acs_noev": 1.0,
    "acs_oracle": None,
    "n_missing": 0,
    "failure_reason": "none",
    "parse_ok": True,
}


def test_acs_values_unchanged():
    out = acs_to_ecs(FIXTURE)
    assert out["ecs_strict"] is FIXTURE["acs_strict"] or out["ecs_strict"] == FIXTURE["acs_strict"]
    assert out["ecs_noev"] == FIXTURE["acs_noev"]
    assert out["ecs_oracle"] == FIXTURE["acs_oracle"]  # None -> None, still None


def test_no_acs_keys_remain():
    out = acs_to_ecs(FIXTURE)
    acs_keys_remaining = [k for k in out if isinstance(k, str) and (k.startswith("acs_") or k.startswith("ACS_"))]
    assert acs_keys_remaining == [], f"acs_*/ACS_* keys leaked into output: {acs_keys_remaining}"


def test_non_acs_keys_and_values_byte_identical():
    out = acs_to_ecs(FIXTURE)
    for key in ("instance_id", "n_missing", "failure_reason", "parse_ok"):
        assert key in out, f"non-acs key {key!r} missing from adapter output"
        assert out[key] == FIXTURE[key], f"non-acs key {key!r} value changed: {FIXTURE[key]!r} -> {out[key]!r}"
    # Also confirm no extra/renamed variants of these keys were introduced.
    assert set(out.keys()) == {"ecs_strict", "ecs_noev", "ecs_oracle", "instance_id",
                                "n_missing", "failure_reason", "parse_ok"}


def test_input_dict_not_mutated():
    original = dict(FIXTURE)
    acs_to_ecs(FIXTURE)
    assert FIXTURE == original, "acs_to_ecs must not mutate its input dict"


def test_uppercase_acs_prefix_also_renamed():
    summary_fixture = {"ACS_strict": 0.837, "ACS_noev": 0.86, "ACS_oracle": 0.996, "N": 789}
    out = acs_to_ecs(summary_fixture)
    assert out == {"ECS_strict": 0.837, "ECS_noev": 0.86, "ECS_oracle": 0.996, "N": 789}


def _run_all():
    tests = [
        test_acs_values_unchanged,
        test_no_acs_keys_remain,
        test_non_acs_keys_and_values_byte_identical,
        test_input_dict_not_mutated,
        test_uppercase_acs_prefix_also_renamed,
    ]
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
