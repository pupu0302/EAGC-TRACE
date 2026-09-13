"""
JSON Schema validator for EAGC GraphPrediction instances.

Validates a GraphPrediction dict against schemas/eagc.schema.json using
jsonschema.Draft202012Validator.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import jsonschema

_DEFAULT_SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "eagc.schema.json"


@dataclass
class ValidationResult:
    """Result of validating a single GraphPrediction."""
    claim_id: str
    valid: bool
    errors: List[str]
    warnings: List[str]


class SchemaValidator:
    """Validator for EAGC GraphPrediction schema."""

    def __init__(self, schema_path: Path = _DEFAULT_SCHEMA_PATH):
        """
        Initialize validator with schema.

        Args:
            schema_path: Path to eagc.schema.json
        """
        self.schema_path = schema_path
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        self.validator = jsonschema.Draft202012Validator(self.schema)

    def validate_single(self, graph: dict) -> ValidationResult:
        """
        Validate a single GraphPrediction instance.

        Args:
            graph: GraphPrediction dict

        Returns:
            ValidationResult with validation status and errors
        """
        claim_id = graph.get("claim_id", "unknown")
        errors = []
        warnings = []

        # Schema validation
        for error in self.validator.iter_errors(graph):
            error_msg = f"{error.json_path}: {error.message}"
            errors.append(error_msg)

        # Extract warnings from graph if present
        if "warnings" in graph:
            warnings.extend(graph["warnings"])

        valid = (len(errors) == 0)

        return ValidationResult(
            claim_id=claim_id,
            valid=valid,
            errors=errors,
            warnings=warnings
        )
