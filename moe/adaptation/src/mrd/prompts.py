"""Prompt constants shared by both venvs.

Kept free of the ``gepa`` import so the measurement side (``.venv-hf``, no gepa)
can read the seed prompt without pulling in the optimizer.
"""
from __future__ import annotations

# The single component GEPA mutates. Its value is the full system prompt.
COMPONENT_NAME = "system_prompt"

SEED_SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the user's request, and follow every "
    "formatting and output constraint stated in the request exactly."
)
