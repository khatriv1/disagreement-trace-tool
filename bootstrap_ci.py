#!/usr/bin/env python3
"""
Compute 95% bootstrap confidence intervals for the per-code agent
disagreement rate, reusing the per-utterance labels already present in the
existing run logs (no re-coding / no extra Ollama calls).
"""

import ast
import csv
import os
import random
import re
from pathlib import Path

import numpy as np

np.random.seed(42)
random.seed(42)

B = 2000  # number of bootstrap resamples

CHEM_LOG = "step1_twomodels_run.log"
TUT_LOG = "step1_tutoring_run.log"

CHEM_CODES = ["process", "plan", "act", "wrong"]
TUT_CODES = [
    "Greeting",
    "Instruction",
    "Guiding feedback",
    "Aligning to prior knowledge",
    "Understanding/Engagement-Tutor",
    "Technical or Logistics",
    "Encouragement",
    "Time Management",
]

OUT_CSV = "bootstrap_ci_results.csv"


_AB_DICT_RE = re.compile(
    # capture the dict after "A(...)= {...}" and "B(...)= {...}" on the same line
    r"A\([^)]*\)=({[^}]*})\s+B\([^)]*\)=({[^}]*})"
)


def _parse_log_for_disagreement(log_path: Path, codes: list[str]) -> dict[str, list[int]]:
    """
    Return per-code disagreement indicators (0/1) across all parsed utterance lines.
    An utterance line is used only if both agents provide all codes.
    """
    disagree: dict[str, list[int]] = {c: [] for c in codes}
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        if "A(" not in line or " B(" not in line:
            continue
        m = _AB_DICT_RE.search(line)
        if not m:
            continue
        raw_a, raw_b = m.group(1), m.group(2)
        try:
            da = ast.literal_eval(raw_a)
            db = ast.literal_eval(raw_b)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(da, dict) or not isinstance(db, dict):
            continue
        if any(c not in da or c not in db for c in codes):
            continue
        if any(da[c] not in (0, 1) or db[c] not in (0, 1) for c in codes):
            continue

        for c in codes:
            disagree[c].append(1 if da[c] != db[c] else 0)

    return disagree


def _bootstrap_ci(arr: list[int], b: int = B, alpha_low: float = 2.5, alpha_high: float = 97.5):
    if not arr:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(arr, dtype=float)
    point = a.mean()
    n = len(a)
    # shape: (B, n)
    samples = np.random.choice(a, size=(b, n), replace=True)
    means = samples.mean(axis=1)
    ci_low, ci_high = np.percentile(means, [alpha_low, alpha_high])
    return point, float(ci_low), float(ci_high)


def _print_table(dataset: str, codes: list[str], disagree: dict[str, list[int]], out_rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"{dataset}: per-code disagreement rate with 95% bootstrap CI")
    print("=" * 78)
    print(f"{'code':<40} {'n_utterances':>13} {'rate':>10} {'CI_low':>10} {'CI_high':>10}")
    print("-" * 78)
    for code in codes:
        arr = disagree[code]
        rate, ci_low, ci_high = _bootstrap_ci(arr)
        out_rows.append(
            {
                "dataset": dataset,
                "code": code,
                "n_utterances": len(arr),
                "disagreement_rate": rate,
                "CI_low": ci_low,
                "CI_high": ci_high,
            }
        )
        print(
            f"{code:<40} {len(arr):>13d} {rate:>10.4f} {ci_low:>10.4f} {ci_high:>10.4f}"
        )


def main() -> None:
    base = Path(__file__).resolve().parent
    chem_log = base / CHEM_LOG
    tut_log = base / TUT_LOG

    missing = []
    if not chem_log.exists():
        missing.append(chem_log.name)
    if not tut_log.exists():
        missing.append(tut_log.name)
    if missing:
        raise SystemExit(f"Missing log file(s): {', '.join(missing)}")

    chem_disagree = _parse_log_for_disagreement(chem_log, CHEM_CODES)
    tut_disagree = _parse_log_for_disagreement(tut_log, TUT_CODES)

    rows: list[dict] = []
    _print_table("chemistry", CHEM_CODES, chem_disagree, rows)
    _print_table("tutoring", TUT_CODES, tut_disagree, rows)

    out_path = base / OUT_CSV
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "code",
                "n_utterances",
                "disagreement_rate",
                "CI_low",
                "CI_high",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved bootstrap results to {out_path.name}")


if __name__ == "__main__":
    main()

