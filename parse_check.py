#!/usr/bin/env python3
"""Check whether agent disagreement is genuine or a parsing artifact."""

CSV_PATH = "lak24-coded-utterances.csv"
MODEL_A = "qwen2.5:7b"
MODEL_B = "llama3.1:8b"
SAMPLE = 100
CODE_KEYS = ["process", "plan", "act", "wrong"]

import ast
import csv
import random
import re
import sys
from pathlib import Path

random.seed(0)

import reasoning_disagreement_lak24 as rd

OUT_TXT = "parse_check_results.txt"


def _last_brace_dict_text(raw_text: str) -> str | None:
    """Return text of the last {...} span with balanced braces, or None."""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(raw_text):
        if raw_text[i] != "{":
            i += 1
            continue
        depth = 0
        for j in range(i, len(raw_text)):
            if raw_text[j] == "{":
                depth += 1
            elif raw_text[j] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((i, j + 1))
                    break
        i += 1
    if not spans:
        return None
    start, end = spans[-1]
    return raw_text[start:end]


def parse_status(raw_text: str) -> tuple[dict, bool]:
    """Parse last dict in raw_text; ok if all CODE_KEYS present with values 0 or 1."""
    chunk = _last_brace_dict_text(raw_text)
    if chunk is None:
        return {k: None for k in CODE_KEYS}, False
    try:
        extracted = ast.literal_eval(chunk)
    except (ValueError, SyntaxError):
        return {k: None for k in CODE_KEYS}, False
    if not isinstance(extracted, dict):
        return {k: None for k in CODE_KEYS}, False

    code_dict = {k: extracted.get(k) for k in CODE_KEYS}
    ok = all(k in extracted for k in CODE_KEYS) and all(
        extracted[k] in (0, 1) for k in CODE_KEYS
    )
    return code_dict, ok


def as01(value) -> int:
    return 1 if value == 1 else 0


def load_sample(path: Path, n: int) -> list[str]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("remove_flag", "").strip().lower() == "yes":
                continue
            text = (row.get("utterance_combined") or "").strip()
            if text:
                rows.append(text)
    if len(rows) < n:
        raise SystemExit(f"Only {len(rows)} utterances; need {n}.")
    return random.sample(rows, n)


def per_code_disagreement(
    records: list[dict],
) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {k: float("nan") for k in CODE_KEYS}
    counts = {k: 0 for k in CODE_KEYS}
    for rec in records:
        a = rec["a01"]
        b = rec["b01"]
        for k in CODE_KEYS:
            if a[k] != b[k]:
                counts[k] += 1
    return {k: counts[k] / n for k in CODE_KEYS}


def main() -> None:
    base = Path(__file__).resolve().parent
    out_path = base / OUT_TXT
    log_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        log_lines.append(line)

    csv_file = base / CSV_PATH
    emit(f"Loading from {CSV_PATH}, sampling {SAMPLE} (seed=0)...")
    sample = load_sample(csv_file, SAMPLE)
    emit(f"  Agent A: {MODEL_A}; Agent B: {MODEL_B}\n")

    records: list[dict] = []
    fail_examples: dict[str, list[str]] = {MODEL_A: [], MODEL_B: []}
    parse_fail_a = parse_fail_b = 0

    for idx, text in enumerate(sample, start=1):
        emit(f"[{idx}/{SAMPLE}] coding ({len(text.split())} words)...")
        raw_a = rd.code_utterance("Agent A", "bold and decisive", text, model=MODEL_A)
        raw_b = rd.code_utterance(
            "Agent B", "cautious and conservative", text, model=MODEL_B
        )
        dict_a, ok_a = parse_status(raw_a)
        dict_b, ok_b = parse_status(raw_b)
        if not ok_a:
            parse_fail_a += 1
            if len(fail_examples[MODEL_A]) < 5:
                fail_examples[MODEL_A].append(raw_a)
        if not ok_b:
            parse_fail_b += 1
            if len(fail_examples[MODEL_B]) < 5:
                fail_examples[MODEL_B].append(raw_b)

        a01 = {k: as01(dict_a.get(k)) for k in CODE_KEYS}
        b01 = {k: as01(dict_b.get(k)) for k in CODE_KEYS}
        records.append(
            {
                "ok_a": ok_a,
                "ok_b": ok_b,
                "both_ok": ok_a and ok_b,
                "a01": a01,
                "b01": b01,
            }
        )
        emit(f"  parse ok: A={ok_a}  B={ok_b}")

    emit("\n" + "=" * 72)
    emit("PARSE-FAILURE RATES")
    emit("=" * 72)
    emit(f"  {MODEL_A}: {parse_fail_a / SAMPLE:.4f} ({parse_fail_a}/{SAMPLE})")
    emit(f"  {MODEL_B}: {parse_fail_b / SAMPLE:.4f} ({parse_fail_b}/{SAMPLE})")

    for model, examples in fail_examples.items():
        emit(f"\n--- Example FAILED parses ({model}) ---")
        if not examples:
            emit("  (none)")
        for i, raw in enumerate(examples, start=1):
            emit(f"\n  Example {i}:")
            snippet = raw if len(raw) <= 800 else raw[:797] + "..."
            for line in snippet.splitlines():
                emit(f"    {line}")

    all_recs = records
    clean_recs = [r for r in records if r["both_ok"]]
    rate_all = per_code_disagreement(all_recs)
    rate_clean = per_code_disagreement(clean_recs)

    emit("\n" + "=" * 72)
    emit("PER-CODE DISAGREEMENT RATE (fraction of utterances where A != B)")
    emit("=" * 72)
    emit(
        f"{'code':<10} {'all_items':>12} {'both_parsed_ok':>16}  "
        f"(n={len(all_recs)} / n={len(clean_recs)})"
    )
    emit("-" * 72)
    for k in CODE_KEYS:
        ra = rate_all[k]
        rc = rate_clean[k]
        emit(f"{k:<10} {ra:>12.4f} {rc:>16.4f}")

    emit("\nIf 'all_items' and 'both_parsed_ok' are close, disagreement is likely genuine.")

    out_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    emit(f"\nSaved summary to {OUT_TXT}")


if __name__ == "__main__":
    main()
