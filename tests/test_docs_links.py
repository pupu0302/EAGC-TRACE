"""
Mechanically verifies that repo-relative file references in this package's
own new/edited docs actually resolve to real files, and that reproduction
runs from a read-only-style checkout with output redirected elsewhere.

This intentionally checks a curated allowlist of qualified (directory-bearing)
references in the package documentation,
rather than every backtick-quoted token in every .md file -- many existing
docs use bare filenames in table cells where the directory is established by
context (e.g. a "Directory" column), which a naive path checker flags as
false positives.

Run with: python -m pytest tests/test_docs_links.py -q
"""
import os
import subprocess
import sys
import tempfile

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)

# Every qualified (contains "/"), non-glob, non-URL repo-relative path
# referenced from the package documentation.
REFERENCED_PATHS = [
    # docs/METHOD_OVERVIEW.md
    "trace/r1/r1_router.py", "trace/g/g_generator.py", "trace/v/v_validator.py",
    "trace/sr/sr_DL.py", "trace/run_flat_evidence.py",
    "evaluation/e/e_evaluation.py", "evaluation/scoring/ecs_metric.py",
    "evaluation/scoring/production_wrapper", "evaluation/scoring/reference_implementation",
    "evaluation/FROZEN_SCORING_CONTRACT.yaml", "results/table1_final_789.csv",
    "trace/README.md", "reproduce/README.md",
    # README.md (root)
    "configs/models.yaml", "schemas/eagc.schema.json", "trace/dvi",
    "benchmark_construction/prompts", "docs/METHOD_OVERVIEW.md",
    # trace/README.md
    "baselines/b1_reachable_set.py", "benchmark_validation/annotation_validator.py",
    "evaluation/scoring/acs_ecs_adapter.py", "reproduce/reproduce_table1.py",
    "trace/dvi/README.md", "trace/llm/g_llm_client.py",
    # reproduce/README.md
    "data/eagc_trace_789/gold", "data/README.md",
    "evaluation/scoring/bootstrap_lib.py", "evaluation/scoring/holm_correction.py",
    "evaluation/scoring/empty_evidence_control.py",
    "evaluation/scoring/primary_channel/primary_channel_scorer.py",
    "predictions/prediction_provenance_map.json", "public_assets/main_results.csv",
    "results/bootstrap/table1_bootstrap_results.json", "results/EXPECTED_RESULTS.json",
    "results/figure_data/r2_fig2__fig2_channel_recovery.csv",
    "results/figure_data/r3_fig3ab__fig3b_ablations.csv",
    "results/primary_channel_summary.json",
    # data/DATA_CARD.md
    "data/eagc_trace_789/tasks",
    # predictions/README.md
    "data/eagc_trace_789",
    # quickstart/README.md
    "evaluation/scoring/production_wrapper/production_scorer.py",
    "evaluation/scoring/reference_implementation/reference_scorer.py",
    "evaluation/scoring/fixtures/synthetic_fixtures.py",
    "predictions/table1/row02_G_GPT_4o.json",
]


def test_referenced_docs_paths_exist():
    missing = [p for p in REFERENCED_PATHS if not os.path.exists(os.path.join(_ROOT, p))]
    assert missing == [], f"referenced-but-missing paths: {missing}"


def test_new_docs_exist():
    """The package's current public documentation entry points. The four
    superseded docs formerly checked here (DATA_AND_CODE_AVAILABILITY.md,
    MODEL_API_LIMITATIONS.md, PROJECT_PAGE_CONTENT.md, REPRODUCIBILITY.md)
    were consolidated; their content is now covered by
    the entry points below."""
    for p in ("README.md", "docs/METHOD_OVERVIEW.md", "trace/README.md",
              "reproduce/README.md", "data/DATA_CARD.md"):
        assert os.path.exists(os.path.join(_ROOT, p)), f"expected doc missing: {p}"


def test_deleted_docs_are_not_referenced():
    """The four superseded docs removed from the source package (DATA_AND_CODE_AVAILABILITY.md, MODEL_API_LIMITATIONS.md,
    PROJECT_PAGE_CONTENT.md, REPRODUCIBILITY.md) must not be linked from
    anywhere in the source or data packages -- their content is now covered
    by the root README.md, reproduce/README.md, and trace/README.md
    instead. Scoped to source-controlled directories only: the separately
    versioned, frozen `predictions/` and `results/` packages are out of
    scope for this check (not edited by the same release step that removed
    these docs) and may still carry pre-existing references pending their
    own release step."""
    deleted_names = (
        "DATA_AND_CODE_AVAILABILITY.md", "MODEL_API_LIMITATIONS.md",
        "PROJECT_PAGE_CONTENT.md", "REPRODUCIBILITY.md",
    )
    excluded_top_level = {"predictions", "results", ".git", "__pycache__", ".pytest_cache"}
    hits = []
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        if os.path.abspath(dirpath) == os.path.abspath(_ROOT):
            dirnames[:] = [d for d in dirnames if d not in excluded_top_level]
        else:
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".pytest_cache")]
        for fn in filenames:
            if fn == os.path.basename(__file__):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                with open(fp, encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except (UnicodeDecodeError, IsADirectoryError, PermissionError):
                continue
            for name in deleted_names:
                if name in text:
                    hits.append((os.path.relpath(fp, _ROOT), name))
    assert hits == [], f"dangling references to deleted docs: {hits}"


def test_reproduce_table1_runs_from_read_only_style_checkout_with_external_output_dir():
    """reproduce_table1.py must succeed when run against
    this package with its own tree effectively read-only (no writes attempted
    inside it) and --output-dir pointed at an external temp directory."""
    if not os.path.isdir(os.path.join(_ROOT, "data", "eagc_trace_789", "gold")):
        pytest.skip("full data archive is not installed; see data/README.md")

    with tempfile.TemporaryDirectory() as tmp_out:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, os.path.join(_ROOT, "reproduce", "reproduce_table1.py"),
             "--output-dir", tmp_out],
            cwd=_ROOT, env=env, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"reproduce_table1.py failed with external --output-dir:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert os.path.exists(os.path.join(tmp_out, "REPRODUCTION_RESULT.json"))
        # And nothing was written inside the package's own reproduce/ directory.
        assert not os.path.exists(os.path.join(_ROOT, "reproduce", "_reproduction_output"))
        assert not os.path.exists(os.path.join(_ROOT, "reproduce", "reproduced_results"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
