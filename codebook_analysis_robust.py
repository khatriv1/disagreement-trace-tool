#!/usr/bin/env python3
"""
Part 2 robustness check: does "behaviors" in the codebook predict disagreement
after removing prevalence effects (1 - AC1)? Reuses base_rate_ac1.csv only.

Note: n=10, descriptive — not a significance test.
"""

from pathlib import Path
import csv

import numpy as np
from scipy import stats

AC1_CSV = "base_rate_ac1.csv"
OUT_CSV = "codebook_analysis_robust.csv"

# 10 codes with kappa; Technical or Logistics and Time Management omitted.
BEHAVIORS = {
    "process": 2,
    "plan": 1,
    "act": 2,
    "wrong": 1,
    "Greeting": 1,
    "Instruction": 1,
    "Guiding feedback": 2,
    "Aligning to prior knowledge": 1,
    "Understanding/Engagement-Tutor": 1,
    "Encouragement": 1,
}

DATASET_SHORT = {"chemistry": "chem", "tutoring": "tut"}


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
    print(f"\n  {label}")
    print(f"    Spearman r = {spearman.correlation:.4f},  p = {spearman.pvalue:.4f}")
    print(f"    Pearson  r = {pearson.statistic:.4f},  p = {pearson.pvalue:.4f}")


def correlate_property(
    rows: list[dict],
    y_key: str,
    title: str,
) -> None:
    print(f"\n--- behaviors vs {title} ---")
    xs = [r["behaviors"] for r in rows]
    ys = [r[y_key] for r in rows]
    print_correlation(xs, ys, "raw-pooled (10 codes)")

    chem = [r for r in rows if r["dataset"] == "chem"]
    tut = [r for r in rows if r["dataset"] == "tut"]
    z_x: list[float] = []
    z_y: list[float] = []
    for subset in (chem, tut):
        z_x.extend(zscore([r["behaviors"] for r in subset]))
        z_y.extend(zscore([r[y_key] for r in subset]))
    print_correlation(z_x, z_y, "z-scored within each dataset (10 codes)")


def main() -> None:
    base = Path(__file__).resolve().parent
    ac1_path = base / AC1_CSV
    if not ac1_path.exists():
        raise SystemExit(f"Missing {AC1_CSV} — run base_rate_ac1.py first.")

    rows: list[dict] = []
    with ac1_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = r["code"]
            if code not in BEHAVIORS:
                continue
            ds = DATASET_SHORT.get(r["dataset"], r["dataset"])
            ac1 = float(r["AC1"])
            rows.append(
                {
                    "code": code,
                    "dataset": ds,
                    "behaviors": BEHAVIORS[code],
                    "base_rate": float(r["base_rate"]),
                    "disagreement_rate": float(r["disagreement_rate"]),
                    "robust_disagreement": 1.0 - ac1,
                }
            )

    if len(rows) != 10:
        raise SystemExit(f"Expected 10 codes; got {len(rows)}.")

    print("=" * 88)
    print("CODEBOOK ROBUSTNESS TABLE (10 codes)")
    print("=" * 88)
    print(
        f"{'code':<32} {'dataset':>8} {'behaviors':>10} {'base_rate':>10} "
        f"{'disagreement_rate':>18} {'robust_disagreement':>20}"
    )
    print("-" * 88)
    for r in rows:
        print(
            f"{r['code']:<32} {r['dataset']:>8} {r['behaviors']:>10d} "
            f"{r['base_rate']:>10.4f} {r['disagreement_rate']:>18.4f} "
            f"{r['robust_disagreement']:>20.4f}"
        )

    out_csv = base / OUT_CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "code",
                "dataset",
                "behaviors",
                "base_rate",
                "disagreement_rate",
                "robust_disagreement",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved table to {out_csv.name}")

    print("\n" + "=" * 88)
    print('CORRELATIONS: "behaviors" vs disagreement measures')
    print("=" * 88)
    print("Note: n=10, descriptive — not a significance test.")

    correlate_property(rows, "disagreement_rate", "disagreement_rate (raw)")
    correlate_property(
        rows, "robust_disagreement", "robust_disagreement (1 - AC1)"
    )
    correlate_property(rows, "base_rate", "base_rate (prevalence)")


if __name__ == "__main__":
    main()
