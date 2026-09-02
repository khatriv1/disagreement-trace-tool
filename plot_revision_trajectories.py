#!/usr/bin/env python3
"""Plot pooled kappa trajectories for bare all_three seeds with bootstrap CIs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CONRAD = Path(__file__).resolve().parent
PKG = CONRAD / "part3_experiment_package"
sys.path.insert(0, str(PKG))

from run_experiment import TUT_CODES, cohen_kappa, load_tutor_rows, load_scored_cache  # noqa: E402

SEED_RUNS = [
    ("Seed 42", PKG / "results_bare_all_three_seed42"),
    ("Seed 123", PKG / "results_bare_all_three_seed123"),
    ("Seed 7", PKG / "results_bare_all_three_seed7"),
]
ROUNDS = list(range(6))
N_BOOT = 2000
RNG = np.random.default_rng(0)

OUT_PDF = CONRAD / "revision_trajectories.pdf"
OUT_PNG = CONRAD / "revision_trajectories.png"

# Colorblind-friendly (IBM/Wong-style)
COLORS = ["#0173B2", "#DE8F05", "#029E73"]


def scored_path(run_dir: Path, r: int) -> Path:
    if r == 0:
        return run_dir / "scored_round0_baseline.csv"
    return run_dir / f"scored_round{r}.csv"


def pooled_kappa(records: list[dict], pred: dict[str, dict]) -> float:
    tp = fp = fn = tn = 0
    for r in records:
        iid = r["item_id"]
        for code in TUT_CODES:
            p = int(pred[iid].get(code, 0))
            g = int(r["labels"].get(code, 0))
            if p == 1 and g == 1:
                tp += 1
            elif p == 1 and g == 0:
                fp += 1
            elif p == 0 and g == 1:
                fn += 1
            else:
                tn += 1
    return cohen_kappa(tp, fp, fn, tn)


def bootstrap_ci(records: list[dict], pred: dict[str, dict], n_boot: int = N_BOOT) -> tuple[float, float]:
    n = len(records)
    if n == 0:
        return 0.0, 0.0
    idx = RNG.integers(0, n, size=(n_boot, n))
    samples = np.empty(n_boot)
    for b in range(n_boot):
        subset = [records[i] for i in idx[b]]
        samples[b] = pooled_kappa(subset, pred)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def load_results_kappa(run_dir: Path, r: int) -> float:
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["round"]) == r:
                return float(row["pooled_kappa"])
    raise KeyError(f"round {r} missing in {run_dir / 'results.csv'}")


def main() -> None:
    print("Using run folders:")
    for label, path in SEED_RUNS:
        print(f"  {label}: {path}")

    tutor_rows = load_tutor_rows()
    print(f"\nLoaded {len(tutor_rows)} tutor utterances for gold labels.\n")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for (seed_label, run_dir), color in zip(SEED_RUNS, COLORS):
        if not run_dir.is_dir():
            raise SystemExit(f"Missing folder: {run_dir}")

        kappas: list[float] = []
        lo: list[float] = []
        hi: list[float] = []

        print(f"=== {seed_label} ({run_dir.name}) ===")
        for r in ROUNDS:
            sp = scored_path(run_dir, r)
            if not sp.exists():
                raise SystemExit(f"Missing scored file: {sp}")
            pred = load_scored_cache(sp)
            k_csv = load_results_kappa(run_dir, r)
            k_calc = pooled_kappa(tutor_rows, pred)
            ci_lo, ci_hi = bootstrap_ci(tutor_rows, pred)
            kappas.append(k_csv)
            lo.append(ci_lo)
            hi.append(ci_hi)
            print(
                f"  r{r}: kappa={k_csv:.4f} (recomputed {k_calc:.4f})  "
                f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]"
            )

        x = np.array(ROUNDS)
        ax.plot(x, kappas, "o-", color=color, label=seed_label, linewidth=2, markersize=7)
        ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0)
        print()

    ax.set_xlim(-0.2, 5.2)
    ax.set_xticks(ROUNDS)
    ax.set_xlabel("Revision round", fontsize=12)
    ax.set_ylabel("Pooled Cohen's κ", fontsize=12)
    ax.legend(loc="best", frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
