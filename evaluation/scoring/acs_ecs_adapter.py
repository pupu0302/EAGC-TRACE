"""
ACS -> ECS field-name adapter.

The frozen scoring chain shipped in this package (evaluation/e/e_evaluation.py
and the wrappers in evaluation/scoring/) preserves its original internal
working field names, prefixed "acs_"/"ACS_" (an old internal name for the
metric the paper calls ECS — Evidence Closure Score). This is intentional:
the internal field names are part of the proven, frozen scoring contract and
must not be touched inside the evaluator or its wrappers (see
evaluation/FROZEN_SCORING_CONTRACT.yaml).

This module performs the ACS -> ECS rename ONLY at the display/output layer,
as a pure, side-effect-free, non-recursive key rename over a single dict's
top-level keys:
  - every key starting with "acs_" or "ACS_" is renamed by replacing that
    prefix with "ecs_"/"ECS_" respectively (case of the rest of the key is
    preserved);
  - every value is passed through completely unchanged (same object, not
    even copied/re-serialized);
  - every non-acs-prefixed key/value is passed through completely unchanged.

It never touches nested dicts/lists (a values dict inside a record is not
recursed into) and never changes numeric precision, type, or content — only
the top-level key spelling of acs_*/ACS_* fields.
"""
from __future__ import annotations

from typing import Any, Dict

_ACS_PREFIXES = ("acs_", "ACS_")
_ECS_REPLACEMENTS = {"acs_": "ecs_", "ACS_": "ECS_"}


def _rename_key(key: Any) -> Any:
    """Rename a single key if it starts with an ACS prefix; else pass through."""
    if not isinstance(key, str):
        return key
    for prefix in _ACS_PREFIXES:
        if key.startswith(prefix):
            return _ECS_REPLACEMENTS[prefix] + key[len(prefix):]
    return key


def acs_to_ecs(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a NEW dict with every top-level acs_*/ACS_* key renamed to
    ecs_*/ECS_*. Values are untouched (same object reference). Non-acs keys
    are untouched (same key, same value). The input dict is never mutated.

    Example:
        >>> acs_to_ecs({"acs_strict": 1.0, "acs_noev": 1.0, "acs_oracle": None,
        ...              "n_missing": 0, "instance_id": "X"})
        {'ecs_strict': 1.0, 'ecs_noev': 1.0, 'ecs_oracle': None,
         'n_missing': 0, 'instance_id': 'X'}
    """
    return {_rename_key(k): v for k, v in record.items()}


__all__ = ["acs_to_ecs"]
