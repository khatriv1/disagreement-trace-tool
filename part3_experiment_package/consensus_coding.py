"""Minimal LLM coding helpers for the Part 3 experiment package."""

from __future__ import annotations

import ast
import re

import ollama

_CODE_DICT_RE = re.compile(r"\{[^}]+\}")


def generate(model: str, messages: list[dict], options: dict | None = None) -> str:
    options = options or {}
    resp = ollama.chat(model=model, messages=messages, options=options)
    return resp["message"]["content"]


def extract_and_complete_code(text_output: str, codebook_dict: dict) -> dict:
    labels = list(codebook_dict.keys())
    match = _CODE_DICT_RE.search(text_output)
    if not match:
        return {k: 0 for k in labels}
    try:
        extracted = ast.literal_eval(match.group())
        return {k: int(extracted.get(k, 0) or 0) for k in labels}
    except (ValueError, SyntaxError, TypeError):
        return {k: 0 for k in labels}
