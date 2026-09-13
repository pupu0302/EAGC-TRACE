"""
Public benchmark-validation package.

Validates GraphPrediction instances (the frozen public dataset's task/gold
pairs) against schemas/eagc.schema.json and the task's evidence candidate
pool, independent of the gold-construction pipeline.
"""

from .annotation_validator import AnnotationValidator

__all__ = ["AnnotationValidator"]
