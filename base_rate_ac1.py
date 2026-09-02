#!/usr/bin/env python3
"""
Base rate and Gwet's AC1 per code from existing two-model run logs.
Checks whether agent disagreement is driven by code rarity (prevalence).

Note: n=10 codes with human kappa — descriptive, not a significance test.
"""

import ast
import csv
import re
from pathlib import Path

import numpy as np
from scipy import stats

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

HUMAN_KAPPA = {
    # chemistry
    "process": 0.78,
    "plan": 0.90,
    "act": 0.77,
    "wrong": 1.00,
    # tutoring (codes with kappa only)
    "Greeting": 0.85,
    "Encouragement": 0.80,
    "Instruction": 0.66,
    "Guiding feedback": 0.66,
    "Aligning to prior knowledge": 0.66,
    "Understanding/Engagement-Tutor": 0.60,
}

OUT_CSV = "base_rate_ac1.csv"

_AB_DICT_RE = re.compile(
    r"A\([^)]*\)=({[^}]*})\s+B\([^)]*\)=({[^}]*})"
)


def parse_log(log_path: Path, codes: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Return per-code list of (agentA_label, agentB_label) for each utterance."""
    pairs: dict[str, list[tuple[int, int]]] = {c: [] for c in codes}
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "A(" not in line or " B(" not in line:
            continue
        m = _AB_DICT_RE.search(line)
        if not m:
            continue
        try:
            da = ast.literal_eval(m.group(1))
            db = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            continue
        if not isinstance(da, dict) or not isinstance(db, dict):
            continue
        if any(c not in da or c not in db for c in codes):
            continue
        if any(da[c] not in (0, 1) or db[c] not in (0, 1) for c in codes):
            continue
        for c in codes:
            pairs[c].append((da[c], db[c]))
    return pairs


def compute_metrics(pairs: list[tuple[int, int]]) -> dict:
    if not pairs:
        return {
            "base_rate": float("nan"),
            "disagreement_rate": float("nan"),
            "p_o": float("nan"),
            "AC1": float("nan"),
        }

    all_labels = [a for a, b in pairs] + [b for a, b in pairs]
    base_rate = float(np.mean(all_labels))

    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / len(pairs)
    disagreement_rate = 1.0 - p_o

    pi = base_rate
    p_e = 2 * pi * (1 - pi)
    if 1 - p_e == 0:
        ac1 = 1.0
    else:
        ac1 = (p_o - p_e) / (1 - p_e)

    return {
        "base_rate": base_rate,
        "disagreement_rate": disagreement_rate,
        "p_o": p_o,
        "AC1": ac1,
    }


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


def main() -> None:
    base = Path(__file__).resolve().parent
    chem_log = base / CHEM_LOG
    tut_log = base / TUT_LOG

    missing = []
    if not chem_log.exists():
        missing.append(CHEM_LOG)
    if not tut_log.exists():
        missing.append(TUT_LOG)
    if missing:
        raise SystemExit(f"Missing log file(s): {', '.join(missing)}")

    chem_pairs = parse_log(chem_log, CHEM_CODES)
    tut_pairs = parse_log(tut_log, TUT_CODES)

    all_rows: list[dict] = []

    for dataset, codes, pairs_dict in (
        ("chemistry", CHEM_CODES, chem_pairs),
        ("tutoring", TUT_CODES, tut_pairs),
    ):
        print("\n" + "=" * 72)
        print(f"{dataset}: base rate, disagreement, Gwet's AC1")
        print("=" * 72)
        print(
            f"{'code':<40} {'base_rate':>10} {'disagreement_rate':>18} {'AC1':>10}"
        )
        print("-" * 72)
        for code in codes:
            m = compute_metrics(pairs_dict[code])
            row = {
                "dataset": dataset,
                "code": code,
                "base_rate": m["base_rate"],
                "disagreement_rate": m["disagreement_rate"],
                "AC1": m["AC1"],
            }
            all_rows.append(row)
            print(
                f"{code:<40} {m['base_rate']:>10.4f} "
                f"{m['disagreement_rate']:>18.4f} {m['AC1']:>10.4f}"
            )

    out_csv = base / OUT_CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "code", "base_rate", "disagreement_rate", "AC1"],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved table to {out_csv.name}")

    # 10 codes with human kappa
    kappa_rows = []
    for row in all_rows:
        code = row["code"]
        if code not in HUMAN_KAPPA:
            continue
        kappa_rows.append(
            {
                "dataset": row["dataset"],
                "code": code,
                "human_kappa": HUMAN_KAPPA[code],
                "base_rate": row["base_rate"],
                "disagreement_rate": row["disagreement_rate"],
                "AC1": row["AC1"],
                "one_minus_AC1": 1.0 - row["AC1"],
            }
        )

    chem_k = [r for r in kappa_rows if r["dataset"] == "chemistry"]
    tut_k = [r for r in kappa_rows if r["dataset"] == "tutoring"]

    print("\n" + "=" * 72)
    print("CORRELATIONS (10 codes with human kappa)")
    print("=" * 72)
    print("Note: n=10, descriptive — not a significance test.")

    def correlate_pair(x_key: str, y_key: str, title: str) -> None:
        print(f"\n--- {title} ---")
        xs = [r[x_key] for r in kappa_rows]
        ys = [r[y_key] for r in kappa_rows]
        print_correlation(xs, ys, "raw-pooled (10 codes)")

        z_x: list[float] = []
        z_y: list[float] = []
        for subset in (chem_k, tut_k):
            z_x.extend(zscore([r[x_key] for r in subset]))
            z_y.extend(zscore([r[y_key] for r in subset]))
        print_correlation(
            z_x, z_y, "z-scored within each dataset (10 codes)"
        )

    correlate_pair(
        "human_kappa",
        "disagreement_rate",
        "human kappa vs disagreement_rate",
    )
    correlate_pair(
        "human_kappa",
        "one_minus_AC1",
        "human kappa vs (1 - AC1)  [prevalence-robust disagreement]",
    )

    print("\n--- base_rate vs disagreement_rate (10 kappa codes) ---")
    print_correlation(
        [r["base_rate"] for r in kappa_rows],
        [r["disagreement_rate"] for r in kappa_rows],
        "raw-pooled",
    )
    z_br: list[float] = []
    z_dr: list[float] = []
    for subset in (chem_k, tut_k):
        z_br.extend(zscore([r["base_rate"] for r in subset]))
        z_dr.extend(zscore([r["disagreement_rate"] for r in subset]))
    print_correlation(z_br, z_dr, "z-scored within each dataset")


if __name__ == "__main__":
    main()
