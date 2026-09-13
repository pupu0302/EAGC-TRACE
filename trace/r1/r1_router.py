#!/usr/bin/env python3
"""
R1: Evidence Router / Re-ranker

Routes, re-ranks, and budget-truncates the evidence pool using fixed channel
budgets. In the released evaluation path, claims containing digits,
percentage signs, or currency symbols use the numeric-table-priority policy;
all other claims use the balanced policy.

The low-level router also accepts optional sampling_tags, but these were not
supplied in the frozen benchmark evaluation.

Input: r0_output.json (R0 Evidence Pool Materialization)
Output: r1_output.json (Fixed Budget, Re-ranked)
"""

import json
import sys
import os
from typing import List, Dict, Any
from enum import Enum

# ================= Configuration =================
INPUT_FILE = "quickstart/phase1_r0_output_000000.json"
OUTPUT_FILE = "quickstart/r1_output.json"


# ===========================================

class RoutingPolicy(Enum):
    DEFAULT_BALANCED = "balanced"
    NUMERIC_TABLE_PRIORITY = "numeric_table_priority"
    DEFINITION_PRIORITY = "definition_priority"


class R1EvidenceRouter:
    """
    R1 Router: handles evidence type routing, budget allocation, and re-ranking
    """

    def __init__(self, k_total: int = 30):
        self.k_total = k_total

        # Define the budget quota for each policy (must sum <= k_total)
        # K_total expanded to 30 to align with the task-pool cap (K_TEXT=20, K_TABLE=15, K_DEF=5)
        self.budget_configs = {
            RoutingPolicy.DEFAULT_BALANCED: {
                "text": 18, "table": 8, "def": 4  # def: 2→4 to reduce definition routing_miss
            },
            RoutingPolicy.NUMERIC_TABLE_PRIORITY: {
                "text": 12, "table": 15, "def": 3  # unchanged — keep table budget for numeric claims
            },
            RoutingPolicy.DEFINITION_PRIORITY: {
                "text": 18, "table": 7, "def": 5
            }
        }

    def route_and_rerank(self, r0_data: Dict) -> Dict:
        """
        Main entry point: process a single instance
        """
        # 1. Extract basic information
        evidence_pool_raw = r0_data.get("evidence_pool_raw", [])

        # 2. Determine the routing policy
        policy = self._determine_policy(r0_data)

        # 3. Group and apply heuristic re-ranking
        grouped_evidence = self._group_and_score(evidence_pool_raw, policy)

        # 4. Allocate budget and truncate
        final_pool, stats = self._apply_budget(grouped_evidence, policy)

        # 5. Construct the output and preserve the source identifiers and text
        return {
            "instance_id": r0_data.get("instance_id"),
            "canonical_claim_id": r0_data.get("canonical_claim_id") or r0_data.get("anchor_claim_id"),
            "doc_id": r0_data.get("doc_id"),
            "canonical_text": r0_data.get("canonical_text"),
            "evidence_pool": final_pool,
            "routing_stats": stats
        }

    def _determine_policy(self, instance: Dict) -> RoutingPolicy:
        """Select the routing policy.

        Optional ``sampling_tags`` take precedence when supplied directly.
        Otherwise, canonical-text numeric cues select the numeric-table-
        priority policy, with the balanced policy used as the default.
        """
        tags = instance.get("sampling_tags", [])
        text = instance.get("canonical_text", "")

        # Rule 1: Definition priority
        if "definition_proxy" in tags:
            return RoutingPolicy.DEFINITION_PRIORITY

        # Rule 2: Table/numeric priority
        if "table_proxy" in tags or "numeric_proxy" in tags:
            return RoutingPolicy.NUMERIC_TABLE_PRIORITY

        # Rule 3: Fallback detection based on text features
        if any(char.isdigit() for char in text) or "%" in text or "$" in text:
            return RoutingPolicy.NUMERIC_TABLE_PRIORITY

        # Default policy
        return RoutingPolicy.DEFAULT_BALANCED

    def _group_and_score(self, pool_raw: List[Dict], policy: RoutingPolicy) -> Dict[str, List[Dict]]:
        """
        Group evidence by type and re-score according to the policy
        """
        groups = {"text": [], "table": [], "def": [], "context": []}

        for evid in pool_raw:
            etype = evid.get("type")

            # Copy the object to avoid mutating the original data and preserve
            # all R0 metadata (row_label, etc.).
            e_obj = evid.copy()

            # Base score
            base_score = e_obj.get("score", 0.5)
            boost = 1.0

            # --- Re-ranking Heuristics ---
            if policy == RoutingPolicy.NUMERIC_TABLE_PRIORITY:
                if etype == "table_cell":
                    boost = 1.2
                elif etype == "text":
                    boost = 0.9

            elif policy == RoutingPolicy.DEFINITION_PRIORITY:
                if etype == "definition":
                    boost = 1.3

            e_obj["_r1_score"] = base_score * boost

            # Group
            if etype == "text" or etype == "sentence":
                groups["text"].append(e_obj)
            elif etype == "table_cell":
                groups["table"].append(e_obj)
            elif etype == "definition":
                groups["def"].append(e_obj)
            elif etype == "table_row_view":
                groups["context"].append(e_obj)
            else:
                groups["text"].append(e_obj)

        # Sort within each group
        for k in ["text", "table", "def"]:
            groups[k].sort(key=lambda x: x.get("_r1_score", 0), reverse=True)

        return groups

    def _apply_budget(self, groups: Dict[str, List], policy: RoutingPolicy) -> tuple:
        config = self.budget_configs[policy]

        # 1. Truncate (slicing)
        selected_text = groups["text"][:config["text"]]
        selected_table = groups["table"][:config["table"]]
        selected_def = groups["def"][:config["def"]]

        # 2. Handle context (simple policy: keep all context, since it does not count against budget)
        selected_context = groups["context"]

        # 3. Merge
        final_pool = selected_text + selected_table + selected_def + selected_context

        # 4. Global sort (by R1 score)
        final_pool.sort(key=lambda x: x.get("_r1_score", 0), reverse=True)

        # 5. Clean up the temporary field
        for item in final_pool:
            if "_r1_score" in item:
                del item["_r1_score"]

        stats = {
            "routing_policy": policy.value,
            "K_total_limit": self.k_total,
            "K_total_actual": len(selected_text) + len(selected_table) + len(selected_def),
            "text_K": len(selected_text),
            "table_K": len(selected_table),
            "definition_K": len(selected_def),
            "context_n": len(selected_context)
        }

        return final_pool, stats


def main():
    # 1. Check the input file
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    print(f"Loading R0 output from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        r0_data = json.load(f)

    # 2. Initialize the router
    router = R1EvidenceRouter(k_total=20)

    # 3. Run routing
    print("Running R1 Evidence Routing...")
    r1_output = router.route_and_rerank(r0_data)

    # 4. Save the result
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(r1_output, f, indent=2, ensure_ascii=False)

    print(f"Success! R1 output saved to {OUTPUT_FILE}")

    # 5. Print verification info
    stats = r1_output["routing_stats"]
    print("\n[R1 Execution Report]")
    print(f"Claim: {r1_output.get('canonical_text')[:50]}...")
    print(f"Policy: {stats['routing_policy']}")
    print(f"Evidence Counts -> Table: {stats['table_K']}, Text: {stats['text_K']}, Def: {stats['definition_K']}")
    print(f"Total Evidence: {stats['K_total_actual']} (Limit: {stats['K_total_limit']})")


if __name__ == "__main__":
    main()
