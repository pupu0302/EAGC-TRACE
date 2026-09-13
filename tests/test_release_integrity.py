"""
Release-integrity regression tests for the public package.

Covers: fail-closed generation (no silent mock substitution unless explicitly
opted in), GPT-3.5 response_format actually reaching the constructed OpenAI
wire payload, fixed-789-denominator scoring semantics for failed predictions,
Audited-Core-238-is-a-true-subset-of-789, the 201 excluded instances never
overlapping the 789 default set, the prediction-provenance left join being
fixed at 789, and production/reference scorer agreement on a coverage_map
counterexample.

Run with: python -m pytest tests/test_release_integrity.py -q
"""
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
_DATA_DIR = os.path.join(_ROOT, "data", "eagc_trace_789")
_GOLD_DIR = os.path.join(_DATA_DIR, "gold")

for _p in (
    os.path.join(_ROOT, "trace"),
    os.path.join(_ROOT, "evaluation", "scoring", "production_wrapper"),
    os.path.join(_ROOT, "evaluation", "scoring", "reference_implementation"),
    os.path.join(_ROOT, "evaluation", "scoring"),
    os.path.join(_ROOT, "evaluation", "scoring", "fixtures"),
    _ROOT,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# 1 & 2. Fail-closed by default; mock substitution only via explicit opt-in.
# ---------------------------------------------------------------------------

def _minimal_instance():
    return {
        "instance_id": "TEST_2024:cc:000001",
        "canonical_text": "Test claim with no real report content.",
        "evidence_pool": [
            {"evid_id": "1:1:TEST_S1", "type": "text", "content": "Some evidence sentence."},
        ],
    }


def _make_generator(allow_mock_fallback_on_error: bool):
    from g.g_generator import ModuleG_Generator
    gen = ModuleG_Generator(config={
        "provider": "mock",
        "model_name": "mock-model",
        "allow_mock_fallback_on_error": allow_mock_fallback_on_error,
    })

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated LLM API failure (no network involved in this test)")

    gen.llm_client.generate = _raise
    return gen


def test_default_llm_exception_fails_closed_no_mock_produced():
    from g.g_generator import GenerationAPIError
    gen = _make_generator(allow_mock_fallback_on_error=False)
    with pytest.raises(GenerationAPIError):
        gen.run(_minimal_instance())


def test_mock_fallback_only_via_explicit_opt_in():
    gen = _make_generator(allow_mock_fallback_on_error=True)
    result = gen.run(_minimal_instance())
    # Explicit opt-in: the API failure is absorbed into a mock-generated
    # prediction instead of raising -- this is the ONLY code path that may
    # ever produce mock output for a real instance.
    assert result["status"] in ("success", "flagged")
    assert result["prediction"] is not None
    assert gen.allow_mock_fallback_on_error is True


def test_default_generator_never_has_mock_fallback_enabled():
    from g.g_generator import ModuleG_Generator
    gen = ModuleG_Generator(config={"provider": "mock", "model_name": "mock-model"})
    assert gen.allow_mock_fallback_on_error is False


# ---------------------------------------------------------------------------
# 3. GPT-3.5's response_format actually reaches the constructed wire payload.
# ---------------------------------------------------------------------------

def test_gpt35_response_format_reaches_wire_payload(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    from llm.g_llm_client import OpenAIClient

    client = OpenAIClient(model_name="gpt-3.5-turbo-0125")

    captured = {}

    class _FakeResponse:
        class _Choice:
            class _Msg:
                content = '{"nodes": []}'
            message = _Msg()
        choices = [_Choice()]

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15
        usage = _Usage()

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResponse()

    client.client.chat.completions.create = _fake_create

    client.generate(
        system_prompt="sys", user_prompt="usr",
        response_format="json_object", temperature=0.0, max_tokens=4096,
    )

    assert "response_format" in captured, (
        "gpt-3.5-turbo-0125 must send response_format on the wire when "
        "response_format='json_object' is requested -- see the "
        "_JSON_MODE_SUPPORTED_PREFIXES fix in trace/llm/g_llm_client.py"
    )
    assert captured["response_format"] == {"type": "json_object"}


def test_gpt4_response_format_still_reaches_wire_payload(monkeypatch):
    """Regression guard: fixing GPT-3.5 must not break the pre-existing GPT-4 path."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    from llm.g_llm_client import OpenAIClient

    client = OpenAIClient(model_name="gpt-4o-2024-08-06")
    captured = {}

    class _FakeResponse:
        class _Choice:
            class _Msg:
                content = '{"nodes": []}'
            message = _Msg()
        choices = [_Choice()]

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15
        usage = _Usage()

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResponse()

    client.client.chat.completions.create = _fake_create
    client.generate(system_prompt="sys", user_prompt="usr", response_format="json_object")
    assert captured.get("response_format") == {"type": "json_object"}


def test_gpt35_legacy_model_without_json_mode_does_not_send_response_format(monkeypatch):
    """Old, pre-1106 gpt-3.5-turbo snapshots never supported JSON mode -- must
    not send response_format to them (would be an API error)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    from llm.g_llm_client import OpenAIClient

    client = OpenAIClient(model_name="gpt-3.5-turbo-0301")
    captured = {}

    class _FakeResponse:
        class _Choice:
            class _Msg:
                content = "plain text, not requested as json"
            message = _Msg()
        choices = [_Choice()]

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15
        usage = _Usage()

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResponse()

    client.client.chat.completions.create = _fake_create
    client.generate(system_prompt="sys", user_prompt="usr", response_format="json_object")
    assert "response_format" not in captured


# ---------------------------------------------------------------------------
# 4. A failed/missing prediction stays in the 789 denominator, scored 0.
# ---------------------------------------------------------------------------

def test_failed_prediction_stays_in_789_denominator_scored_zero():
    import production_scorer

    denom_ids = ["A:cc:000001", "A:cc:000002", "A:cc:000003"]
    gold_data = [{"instance_id": iid, "nodes": [], "gold_evidence_sets": []} for iid in denom_ids]
    # Only instance 1 has a (trivially valid) prediction; 2 and 3 are absent
    # (as reproduce_table1.py's load_predictions_for_scoring leaves them
    # after filtering out status != "ok" / null-prediction records).
    pred_data = [{
        "instance_id": "A:cc:000001",
        "prediction": {
            "nodes": [{"node_id": "n1", "node_type": "Outcome"}],
            "edges": [],
            "evidence_sets": [],
        },
    }]

    result = production_scorer.score_all(gold_data, pred_data, denominator_instance_ids=denom_ids)

    assert result["n_instances"] == 3, "denominator must stay fixed at all 3 instances"
    by_id = {r["instance_id"]: r for r in result["per_instance"]}
    assert set(by_id.keys()) == set(denom_ids)
    for missing_id in ("A:cc:000002", "A:cc:000003"):
        assert by_id[missing_id]["ecs_strict_at_full"] == 0, (
            f"a missing/failed prediction ({missing_id}) must score 0, "
            "not be dropped from the denominator"
        )
    assert result["summary"]["ecs_strict_at_full"]["denominator"] == 3


# ---------------------------------------------------------------------------
# 5. Audited Core (238) is a true, nested subset of the 789.
# ---------------------------------------------------------------------------

def test_audited_core_is_true_subset_of_789():
    extended = json.load(open(os.path.join(_DATA_DIR, "extended_manifest.json")))
    gold_ids = {it["instance_id"] for it in extended["items"]}
    assert len(gold_ids) == 789

    audited = json.load(open(os.path.join(_DATA_DIR, "audited_core_manifest.json")))
    audited_ids = {it["instance_id"] for it in audited["items"]}
    assert len(audited_ids) == 238
    assert audited_ids.issubset(gold_ids), "Audited Core must be a NESTED subset of the 789, never additive"


# ---------------------------------------------------------------------------
# 6. The default loader (789 gold/task files) never includes any of the 201
#    excluded instances.
# ---------------------------------------------------------------------------

def test_default_loader_excludes_the_201():
    extended = json.load(open(os.path.join(_DATA_DIR, "extended_manifest.json")))
    gold_ids = {it["instance_id"] for it in extended["items"]}
    assert len(gold_ids) == 789

    excluded = json.load(open(os.path.join(_ROOT, "data", "excluded_instances_manifest.json")))
    excluded_ids = {it["instance_id"] for it in excluded["items"]}
    assert len(excluded_ids) == 201

    overlap = gold_ids & excluded_ids
    assert overlap == set(), (
        f"the default 789-instance loader (data/eagc_trace_789/gold/) must never "
        f"contain any of the 201 excluded instances; found overlap: {overlap}"
    )


# ---------------------------------------------------------------------------
# 7. The prediction-provenance left join is fixed at 789 for every row.
# ---------------------------------------------------------------------------

def test_prediction_manifest_left_join_fixed_at_789():
    pred_map = json.load(open(os.path.join(_ROOT, "predictions", "prediction_provenance_map.json")))
    assert pred_map["n_rows"] == 13
    for row in pred_map["rows"]:
        assert row["n_instances"] == 789, (
            f"row {row['row_id']} ({row['method']}/{row['backbone']}) must be a fixed "
            f"789-length left join, got n_instances={row['n_instances']}"
        )
        pred_path = os.path.join(_ROOT, row["public_prediction_file"])
        raw = json.load(open(pred_path))
        assert len(raw) == 789, (
            f"{row['public_prediction_file']} must have exactly 789 records "
            f"(left-joined against the full denominator), got {len(raw)}"
        )
        assert row["n_ok"] + row["n_failed_or_missing"] == 789


# ---------------------------------------------------------------------------
# 8. production/reference scorers agree on the released coverage_map
#    counterexamples (T9-T13 from evaluation/scoring/fixtures/synthetic_fixtures.py,
#    plus every other fixture in the same file, per
#    FROZEN_SCORING_CONTRACT.yaml's cross_validation_requirement).
# ---------------------------------------------------------------------------

def test_production_reference_agree_on_all_synthetic_fixtures():
    import production_scorer
    import reference_scorer
    from synthetic_fixtures import build_all_fixtures

    fixtures = build_all_fixtures()
    assert len(fixtures) >= 13, "expected at least T1-T13 to be present"

    mismatches = []
    coverage_map_failures = []
    for name, fx in fixtures.items():
        gold_data = [fx["gold"]]
        pred_data = [fx["pred"]] if fx["pred"] is not None else []
        denom_ids = [fx["gold"]["instance_id"]]

        prod = production_scorer.score_all(gold_data, pred_data, denominator_instance_ids=denom_ids)
        ref = reference_scorer.score_all(gold_data, pred_data, denominator_instance_ids=denom_ids)
        prod_rec = prod["per_instance"][0]
        ref_rec = ref["per_instance"][0]

        for k, expected in fx["expect"].items():
            key = f"ecs_strict_at_{k}"
            pv, rv = prod_rec[key], ref_rec[key]
            if pv != rv:
                mismatches.append((name, key, pv, rv))
            if pv != expected:
                coverage_map_failures.append((name, key, "production", pv, expected))
            if rv != expected:
                coverage_map_failures.append((name, key, "reference", rv, expected))

    assert mismatches == [], f"production/reference disagreed on: {mismatches}"
    assert coverage_map_failures == [], (
        f"scorer output didn't match the fixture file's own documented "
        f"expectation: {coverage_map_failures}"
    )


def test_coverage_map_containment_counterexample_specifically():
    """T9: gold cites a sentence-level evid_id; prediction cites only the
    containing table/paragraph-level unit_id; meta.coverage_map declares the
    containment. Both scorers must count this as covered at K='full'
    (coverage-aware) but NOT at K=1/2/3 (exact-match traversal only) -- this
    is the specific, documented divergence between full and K-truncated
    scoring (see production_scorer.py's module docstring)."""
    import production_scorer
    import reference_scorer
    from synthetic_fixtures import build_all_fixtures

    fixtures = build_all_fixtures()
    fx = fixtures["T9_coverage_map_sentence_via_table_unit"]
    gold_data, pred_data = [fx["gold"]], [fx["pred"]]
    denom_ids = [fx["gold"]["instance_id"]]

    prod = production_scorer.score_all(gold_data, pred_data, denominator_instance_ids=denom_ids)["per_instance"][0]
    ref = reference_scorer.score_all(gold_data, pred_data, denominator_instance_ids=denom_ids)["per_instance"][0]

    assert prod["ecs_strict_at_full"] == 1 == ref["ecs_strict_at_full"]
    assert prod["ecs_strict_at_1"] == 0 == ref["ecs_strict_at_1"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
