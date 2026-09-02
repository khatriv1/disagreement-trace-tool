#!/usr/bin/env python3
"""
Part 2: Does the way a code is WRITTEN predict how much the two models disagree?
Uses codebook text + existing disagreement rates only (no model re-runs).

Note: n=10 codes and definitions are fairly uniform, so this is descriptive and
exploratory, not a significance test.
"""

from pathlib import Path
import csv

import numpy as np
from scipy import stats

CODES = {
    # chemistry
    "process": {
        "dataset": "chem",
        "behaviors": 2,
        "definition": (
            "Assemble information: the student reads or re-reads a question, "
            "hints, or feedback provided by the system. Comprehend information: "
            "the student repeats information provided by the system with a "
            "level of synthesis."
        ),
    },
    "plan": {
        "dataset": "chem",
        "behaviors": 1,
        "definition": (
            "Identify goals and form plans: the student verbalizes a conceptual "
            "plan of how they will solve the problem."
        ),
    },
    "act": {
        "dataset": "chem",
        "behaviors": 2,
        "definition": (
            "Verbalize previous action: the student verbalizes an action just "
            "carried out, explaining what they did. Announce the next action: "
            "the student verbalizes a concrete and specific action they will "
            "do next."
        ),
    },
    "wrong": {
        "dataset": "chem",
        "behaviors": 1,
        "definition": (
            "Realize something is wrong: the student realizes there is a "
            "mistake in the answer or the process, with or without external "
            "prompting such as tutor feedback."
        ),
    },
    # tutoring
    "Greeting": {
        "dataset": "tut",
        "behaviors": 1,
        "definition": (
            "The initial interaction between tutor and student, often at the "
            "beginning or end of the session. Any time a salutation or farewell "
            "is exchanged."
        ),
    },
    "Instruction": {
        "dataset": "tut",
        "behaviors": 1,
        "definition": (
            "Specific instructions or directions posed by the tutor throughout "
            "the lesson."
        ),
    },
    "Guiding feedback": {
        "dataset": "tut",
        "behaviors": 2,
        "definition": (
            "Guided practice through a math problem by the tutor. Feedback on "
            "the student's work or response, and clarification or explanation "
            "of a concept or instruction."
        ),
    },
    "Aligning to prior knowledge": {
        "dataset": "tut",
        "behaviors": 1,
        "definition": (
            "Instances when the tutor brings attention to a previous math "
            "concept the student knows or has discussed. The tutor aligns the "
            "student to prior knowledge, often using the word remember."
        ),
    },
    "Understanding/Engagement-Tutor": {
        "dataset": "tut",
        "behaviors": 1,
        "definition": (
            "The tutor presents checks for understanding as questions to "
            "students."
        ),
    },
    "Encouragement": {
        "dataset": "tut",
        "behaviors": 1,
        "definition": (
            "Affirmative statements from the tutor recognizing the student's "
            "efforts, answers, or performance."
        ),
    },
}

CHEM_CSV = "step1_results_twomodels.csv"
TUTOR_CSV = "step1_tutoring_results.csv"
OUT_CSV = "codebook_analysis.csv"


def word_count(text: str) -> int:
    return len(text.split())


def zscore(values: list[float]) -> list[float]:
    arr = np.array(values, dtype=float)
    std = arr.std(ddof=0)
    if std == 0:
        return [0.0] * len(values)
    return ((arr - arr.mean()) / std).tolist()


def print_correlation(xs: list[float], ys: list[float], label: str) -> None:
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    spearman = stats.spearmanr(x, y)
    pearson = stats.pearsonr(x, y)
    print("\n" + "=" * 78)
    print(f"CORRELATION — {label}")
    print("=" * 78)
    print(f"  Spearman r = {spearman.correlation:.4f},  p = {spearman.pvalue:.4f}")
    print(f"  Pearson  r = {pearson.statistic:.4f},  p = {pearson.pvalue:.4f}")


def load_disagreement_rates(base: Path) -> dict[str, float]:
    rates: dict[str, float] = {}
    chem_path = base / CHEM_CSV
    tutor_path = base / TUTOR_CSV
    if not chem_path.exists():
        raise SystemExit(f"Missing {chem_path.name}")
    if not tutor_path.exists():
        raise SystemExit(f"Missing {tutor_path.name}")

    with chem_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rates[r["code"]] = float(r["agent_disagreement_rate"])
    with tutor_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rates[r["code"]] = float(r["agent_disagreement_rate"])
    return rates


def main() -> None:
    base = Path(__file__).resolve().parent
    rates = load_disagreement_rates(base)

    rows: list[dict] = []
    for code, meta in CODES.items():
        if code not in rates:
            raise SystemExit(f"No disagreement rate found for code: {code}")
        rows.append(
            {
                "code": code,
                "dataset": meta["dataset"],
                "def_word_count": word_count(meta["definition"]),
                "behaviors": meta["behaviors"],
                "disagreement_rate": rates[code],
            }
        )

    print("=" * 78)
    print("CODEBOOK ANALYSIS TABLE")
    print("=" * 78)
    print(
        f"{'code':<32} {'dataset':>8} {'def_word_count':>14} "
        f"{'behaviors':>10} {'disagreement_rate':>18}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['code']:<32} {r['dataset']:>8} {r['def_word_count']:>14d} "
            f"{r['behaviors']:>10d} {r['disagreement_rate']:>18.4f}"
        )

    out_csv = base / OUT_CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "code",
                "dataset",
                "def_word_count",
                "behaviors",
                "disagreement_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved table to {out_csv.name}")

    chem = [r for r in rows if r["dataset"] == "chem"]
    tut = [r for r in rows if r["dataset"] == "tut"]

    for prop in ("def_word_count", "behaviors"):
        print("\n" + "#" * 78)
        print(f"PROPERTY: {prop}  vs  disagreement_rate")
        print("#" * 78)

        print_correlation(
            [r[prop] for r in chem],
            [r["disagreement_rate"] for r in chem],
            f"chemistry only ({len(chem)} codes) — {prop}",
        )
        print_correlation(
            [r[prop] for r in tut],
            [r["disagreement_rate"] for r in tut],
            f"tutoring only ({len(tut)} codes) — {prop}",
        )

        # within-dataset z-scores, then pool
        z_prop: list[float] = []
        z_rate: list[float] = []
        for subset in (chem, tut):
            z_prop.extend(zscore([r[prop] for r in subset]))
            z_rate.extend(zscore([r["disagreement_rate"] for r in subset]))
        print_correlation(
            z_prop,
            z_rate,
            f"combined, standardized within each dataset ({len(rows)} codes) — {prop}",
        )

    print("\n" + "-" * 78)
    print(
        "Note: n=10 codes and definitions are fairly uniform, so this is "
        "descriptive and exploratory, not a significance test."
    )


if __name__ == "__main__":
    main()
