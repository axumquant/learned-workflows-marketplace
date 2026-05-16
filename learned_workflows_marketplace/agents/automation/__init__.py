"""LLM-driven workflow synthesizer.

Turns a finished automation run (instruction + recorded actions + seed
data) into a saveable, parameterized workflow row.
"""
from .workflow_synthesizer import synthesize_workflow_from_run  # noqa: F401
