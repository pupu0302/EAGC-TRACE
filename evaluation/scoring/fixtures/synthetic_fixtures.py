"""Synthetic fixtures for the scoring contract's required unit tests (T1-T18).

All evidence pools, gold sets, predictions, and coverage maps are fabricated;
this module reads no benchmark or prediction artifacts. T1-T8 cover the core
closure and traversal rules. T9-T18 cover unit-to-sentence ``coverage_map``
matching, including directionality, one-to-many and many-to-one mappings,
missing or malformed maps, overlapping mappings, alternative gold sets, and
irrelevant extra citations.

The fixtures are shared as plain data by the production and reference scorer
test drivers. The two scorer implementations remain independent."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

PASS_VR = {"v0_gate": {"pass": True}, "v1_gate": {"pass": True}, "v2_gate": {"pass": True}}
FAIL_V1_VR = {"v0_gate": {"pass": True}, "v1_gate": {"pass": False, "errors": ["synthetic v1 fail"]}, "v2_gate": {"pass": True}}


def _node(node_id: str, node_type: str = "Outcome") -> Dict:
    return {"node_id": node_id, "node_type": node_type, "text": "synthetic", "attrs": {}}


def _es(set_id: str, attached_to: str, evid_ids: List[str]) -> Dict:
    return {"set_id": set_id, "attached_to_node": attached_to,
            "links": [{"evid_id": e, "relation": "qualitative_support"} for e in evid_ids]}


def _gold_set(evid_ids: List[str]) -> Dict:
    return {"links": [{"evid_id": e, "importance": "primary"} for e in evid_ids]}


def _gold_set_mixed_importance(primary_ids: List[str], secondary_ids: List[str]) -> Dict:
    links = [{"evid_id": e, "importance": "primary"} for e in primary_ids]
    links += [{"evid_id": e, "importance": "secondary"} for e in secondary_ids]
    return {"links": links}


def _pred_with_coverage_map(
    instance_id: str, node_id: str, es_id: str, attached_to: str,
    cited_evid_ids: List[str], coverage_map: Dict[str, List[str]],
    validator_report: Dict = None,
) -> Dict:
    return {
        "instance_id": instance_id,
        "validator_report": validator_report or PASS_VR,
        "prediction": {
            "nodes": [_node(node_id)], "edges": [],
            "evidence_sets": [_es(es_id, attached_to, cited_evid_ids)],
        },
        "meta": {"coverage_map": coverage_map},
    }


def build_all_fixtures() -> Dict[str, Dict[str, Any]]:
    fixtures: Dict[str, Dict[str, Any]] = {}

    # --- T1: full gold set cited exactly -------------------------------------------------
    iid = "SYN:T1:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["E1", "E2"])]}
    pred = {"instance_id": iid, "validator_report": PASS_VR, "prediction": {
        "nodes": [_node("n1")], "edges": [], "evidence_sets": [_es("es1", "n1", ["E1", "E2"])]}}
    fixtures["T1_full_gold_set_cited"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 1, 3: 1, "full": 1},
        "note": "gold set size=2; ECS@1=0 (below gold-set size), ECS@{2,3,full}=1",
    }

    # --- T2: gold evidence cited but late -------------------------------------------------
    iid = "SYN:T2:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["E1", "E2"])]}
    pred = {"instance_id": iid, "validator_report": PASS_VR, "prediction": {
        "nodes": [_node("n0"), _node("n1")], "edges": [],
        "evidence_sets": [_es("es0", "n0", ["X1", "X2", "X3"]), _es("es1", "n1", ["E1", "E2"])]}}
    fixtures["T2_gold_evidence_cited_but_late"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 0, 3: 0, "full": 1},
        "note": "traversal=[X1,X2,X3,E1,E2]; gold appears only at positions 4-5, so K<=3 all fail, full=1 (order-independent)",
    }

    # --- T3: duplicate evid_id -------------------------------------------------------------
    iid = "SYN:T3:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["E1", "E2"])]}
    pred = {"instance_id": iid, "validator_report": PASS_VR, "prediction": {
        "nodes": [_node("n0"), _node("n1")], "edges": [],
        "evidence_sets": [_es("es0", "n0", ["E1"]), _es("es1", "n1", ["E1", "E2"])]}}
    fixtures["T3_duplicate_evid_id"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 1, 3: 1, "full": 1},
        "expect_unique_evidence_count": 2,
        "note": "E1 appears twice (es0, es1) but traversal dedups on first occurrence -> unique=[E1,E2], count=2 not 3; K=2 covers gold",
    }

    # --- T4: many irrelevant extra citations ------------------------------------------------
    iid = "SYN:T4:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["E1", "E2"])]}
    extras = [f"Y{i}" for i in range(1, 21)]
    pred = {"instance_id": iid, "validator_report": PASS_VR, "prediction": {
        "nodes": [_node("n0"), _node("n1")], "edges": [],
        "evidence_sets": [_es("es0", "n0", ["E1", "E2"]), _es("es1", "n1", extras)]}}
    fixtures["T4_many_irrelevant_extra_citations"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 1, 3: 1, "full": 1},
        "note": "gold fully cited within first 2 (positions 1-2); 20 irrelevant IDs afterward do not change ECS@K for K<=2 or ECS_full",
    }

    # --- T5: empty / failed output (3 sub-cases) --------------------------------------------
    # (a) missing prediction entirely (None)
    iid_a = "SYN:T5a:000001"
    fixtures["T5a_missing_prediction"] = {
        "gold": {"instance_id": iid_a, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["E1"])]},
        "pred": None,
        "expect": {1: 0, 2: 0, 3: 0, "full": 0},
        "note": "no prediction record at all -- must remain in the denominator, scored 0",
    }
    # (b) unparseable JSON -> prediction graph is a parse-failure marker, no 'nodes' key with content
    iid_b = "SYN:T5b:000001"
    fixtures["T5b_unparseable_json"] = {
        "gold": {"instance_id": iid_b, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["E1"])]},
        "pred": {"instance_id": iid_b, "validator_report": {"v0_gate": {"pass": False, "errors": ["parse failure"]},
                                                              "v1_gate": {"pass": False}, "v2_gate": {"pass": False}},
                  "prediction": {"error": "Invalid JSON", "raw_content": "not json"}},
        "expect": {1: 0, 2: 0, 3: 0, "full": 0},
        "note": "prediction dict has no 'nodes' key at all -> structurally invalid -> 0 for every K",
    }
    # (c) valid JSON but fails the structural validator gate (mock-shaped: 'type' not 'node_type')
    iid_c = "SYN:T5c:000001"
    fixtures["T5c_structurally_invalid_mock_shaped"] = {
        "gold": {"instance_id": iid_c, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["E1"])]},
        "pred": {"instance_id": iid_c, "validator_report": PASS_VR, "prediction": {
            "nodes": [{"node_id": "n1", "type": "Outcome", "text": "x", "attrs": {}}],  # 'type' not 'node_type'
            "edges": [], "evidence_sets": [_es("es1", "n1", ["E1"])]}},
        "expect": {1: 0, 2: 0, 3: 0, "full": 0},
        "note": "even though v0/v1/v2 all pass and E1 IS cited, missing node_type fails the structural gate -> 0 for every K, including full",
    }

    # --- T6: multiple admissible gold sets --------------------------------------------------
    iid = "SYN:T6:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [],
            "gold_evidence_sets": [_gold_set(["A1", "A2"]), _gold_set(["B1"])]}
    pred = {"instance_id": iid, "validator_report": PASS_VR, "prediction": {
        "nodes": [_node("n1")], "edges": [], "evidence_sets": [_es("es1", "n1", ["B1"])]}}
    fixtures["T6_multiple_admissible_gold_sets"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 1, 2: 1, 3: 1, "full": 1},
        "note": "prediction covers ONLY the second (smaller) gold set fully at K=1; ANY one admissible set suffices",
    }

    # --- T9: gold sentence cited via predicted table-cell/unit coverage (v0.2) ---------------
    iid = "SYN:T9:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["SYN_COV9:sent:0100"])]}
    pred = _pred_with_coverage_map(
        iid, "n1", "es1", "n1", ["SYN_COV9:unit:0007"],
        {"SYN_COV9:unit:0007": ["SYN_COV9:sent:0100"]},
    )
    fixtures["T9_coverage_map_sentence_via_table_unit"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 0, 3: 0, "full": 1},
        "note": (
            "coverage-map minimal counterexample: gold cites a sentence-level "
            "evid_id, prediction cites only the containing table-unit evid_id, coverage_map "
            "declares the containment -- full=1 (coverage-aware, matches real evaluator), "
            "K=1/2/3=0 (exact-match traversal never contains the literal sentence id)"
        ),
    }

    # --- T10: reverse case does NOT hold -----------------------------------------------------
    iid = "SYN:T10:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["SYN_COV10:unit:0007"])]}
    pred = _pred_with_coverage_map(
        iid, "n1", "es1", "n1", ["SYN_COV10:sent:0100"],
        {"SYN_COV10:unit:0007": ["SYN_COV10:sent:0100"]},
    )
    fixtures["T10_coverage_map_reverse_does_not_hold"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 0, 3: 0, "full": 0},
        "note": (
            "gold cites the UNIT id itself; prediction cites only the sentence id the unit "
            "covers. coverage_map is a one-directional unit->sentence index (sent_to_unit is "
            "built the same way), so this must NOT count as covered -- full=0"
        ),
    }

    # --- T11: one-to-many (one cited unit covers multiple gold sentence ids) ----------------
    iid = "SYN:T11:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [],
            "gold_evidence_sets": [_gold_set(["SYN_COV11:sent:0001", "SYN_COV11:sent:0002"])]}
    pred = _pred_with_coverage_map(
        iid, "n1", "es1", "n1", ["SYN_COV11:unit:0001"],
        {"SYN_COV11:unit:0001": ["SYN_COV11:sent:0001", "SYN_COV11:sent:0002"]},
    )
    fixtures["T11_coverage_map_one_to_many"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 0, 3: 0, "full": 1},
        "note": "one cited unit covers BOTH gold sentence ids in the single gold set -> best_recall=1.0, full=1",
    }

    # --- T12: many-to-one (only the actually-cited unit's mapped sentences count) ----------
    iid = "SYN:T12:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["SYN_COV12:sent:0001"])]}
    pred = _pred_with_coverage_map(
        iid, "n1", "es1", "n1", ["SYN_COV12:unit:0002"],  # cites unit 2, NOT unit 1 which covers the gold sentence
        {"SYN_COV12:unit:0001": ["SYN_COV12:sent:0001"], "SYN_COV12:unit:0002": ["SYN_COV12:sent:0002"]},
    )
    fixtures["T12_coverage_map_many_to_one_uncited_unit_gives_no_credit"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 0, 3: 0, "full": 0},
        "note": "two candidate units exist in coverage_map; prediction cites the WRONG one (does not cover the gold sentence) -> full=0",
    }

    # --- T13: coverage_map key absent from meta entirely ------------------------------------
    iid = "SYN:T13:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["SYN_COV13:sent:0001"])]}
    pred = {"instance_id": iid, "validator_report": PASS_VR, "prediction": {
        "nodes": [_node("n1")], "edges": [], "evidence_sets": [_es("es1", "n1", ["SYN_COV13:unit:0001"])]}}
    # deliberately NO "meta" key at all on this pred record
    fixtures["T13_coverage_map_missing_meta_key"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 0, 3: 0, "full": 0},
        "note": "pred record has no 'meta' key at all -> must degrade gracefully to exact-match-only (no crash), full=0 since the cited unit id != the gold sentence id",
    }

    # --- T14: empty coverage_map ({}) --------------------------------------------------------
    iid = "SYN:T14:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["SYN_COV14:sent:0001"])]}
    pred = _pred_with_coverage_map(iid, "n1", "es1", "n1", ["SYN_COV14:unit:0001"], {})
    fixtures["T14_coverage_map_empty_dict"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 0, 3: 0, "full": 0},
        "note": "meta.coverage_map present but empty {} -> same degrade-gracefully behavior as T13, full=0",
    }

    # --- T15: malformed coverage_map (non-list value; non-string sentence id) --------------
    iid = "SYN:T15:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["SYN_COV15:sent:0001"])]}
    pred = _pred_with_coverage_map(
        iid, "n1", "es1", "n1", ["SYN_COV15:unit:0002"],
        {
            "SYN_COV15:unit:0001": "not_a_list",              # malformed: value is a str, not a list -- must be skipped
            "SYN_COV15:unit:0002": [123, "SYN_COV15:sent:0001"],  # malformed element (int) mixed with a valid one
        },
    )
    fixtures["T15_coverage_map_malformed_entries_degrade_gracefully"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 0, 3: 0, "full": 1},
        "note": (
            "one unit's value is a bare string (not a list) and must be skipped without raising; "
            "the other unit's list mixes a non-string element (123, skipped) with a valid sentence "
            "id, which is still honored -> full=1, no exception raised anywhere"
        ),
    }

    # --- T16: duplicate/overlapping mapping (same sentence under two units) ----------------
    iid = "SYN:T16:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [], "gold_evidence_sets": [_gold_set(["SYN_COV16:sent:0001"])]}
    pred = _pred_with_coverage_map(
        iid, "n1", "es1", "n1", ["SYN_COV16:unit:0001"],  # cites the FIRST-listed unit
        {"SYN_COV16:unit:0001": ["SYN_COV16:sent:0001"], "SYN_COV16:unit:0002": ["SYN_COV16:sent:0001"]},
    )
    fixtures["T16_coverage_map_duplicate_sentence_across_units_last_wins"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 0, 3: 0, "full": 0},
        "note": (
            "the same sentence id is listed under two different unit ids (a malformed/ambiguous "
            "input real coverage_maps are not expected to produce). Documented, deterministic "
            "tie-break: the LAST unit encountered in dict-iteration order owns the sentence "
            "(unit:0002 here) -- prediction cited unit:0001 (now shadowed), so full=0, not a crash"
        ),
    }

    # --- T17: multiple gold sets, coverage-map credit differs per set ----------------------
    iid = "SYN:T17:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [],
            "gold_evidence_sets": [_gold_set(["SYN_COV17:sent:A1"]), _gold_set(["SYN_COV17:sent:B1"])]}
    pred = _pred_with_coverage_map(
        iid, "n1", "es1", "n1", ["SYN_COV17:unit:B"],
        {"SYN_COV17:unit:B": ["SYN_COV17:sent:B1"]},  # only covers the SECOND gold set's sentence
    )
    fixtures["T17_coverage_map_multiple_gold_sets_best_recall_is_max"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 0, 3: 0, "full": 1},
        "note": "gold set A gets 0 coverage credit, gold set B gets full coverage credit -> best_recall = max(0, 1.0) = 1.0, full=1",
    }

    # --- T18: extra irrelevant citations + primary/secondary treated uniformly -------------
    iid = "SYN:T18:000001"
    gold = {"instance_id": iid, "nodes": [], "edges": [],
            "gold_evidence_sets": [_gold_set_mixed_importance(primary_ids=["SYN_COV18:sent:P1"],
                                                                secondary_ids=["SYN_COV18:evid:S2"])]}
    extras = [f"SYN_COV18:extra:{i}" for i in range(1, 21)]
    pred = _pred_with_coverage_map(
        iid, "n1", "es1", "n1",
        ["SYN_COV18:unit:P"] + ["SYN_COV18:evid:S2"] + extras,
        {"SYN_COV18:unit:P": ["SYN_COV18:sent:P1"]},
    )
    fixtures["T18_coverage_map_extra_irrelevant_and_primary_secondary_uniform"] = {
        "gold": gold, "pred": pred,
        "expect": {1: 0, 2: 0, 3: 0, "full": 1},
        "note": (
            "gold set has one PRIMARY sentence (covered via unit) and one SECONDARY evid_id "
            "(cited exactly) -- both required for full coverage regardless of importance tag "
            "(importance is not consulted by _is_cited); 20 irrelevant extra citations alongside "
            "do not change the result -> best_recall=1.0, full=1, K=1/2/3=0 (exact-match traversal "
            "never contains the literal sentence id P1)"
        ),
    }

    return fixtures


def build_denominator_ids(fixtures: Dict[str, Dict[str, Any]]) -> List[str]:
    return [f["gold"]["instance_id"] for f in fixtures.values()]


def build_gold_pred_lists(fixtures: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    gold_list = [f["gold"] for f in fixtures.values()]
    pred_list = [f["pred"] for f in fixtures.values() if f["pred"] is not None]
    return gold_list, pred_list


__all__ = ["build_all_fixtures", "build_denominator_ids", "build_gold_pred_lists", "PASS_VR", "FAIL_V1_VR"]
