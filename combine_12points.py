#!/usr/bin/env python3
"""
Combine chemistry (4 codes) + tutoring (8 codes) into one 12-point table.
Correlate human kappa vs agent disagreement when all kappas are available.
"""

from pathlib import Path
import csv

# FILL IN from Conrad: per-code kappa for the 8 tutoring codes.
# e.g. {"Greeting": 0.xx, "Instruction": 0.xx, ...}
TUTORING_KAPPA = {}

CHEM_CSV = "step1_results_twomodels.csv"
TUTOR_CSV = "step1_tutoring_results.csv"
OUT_CSV = "combined_12points.csv"
OUT_PNG = "combined_12points.png"


def main() -> None:
    base = Path(__file__).resolve().parent

    chem_path = base / CHEM_CSV
    tutor_path = base / TUTOR_CSV
    if not chem_path.exists():
        raise SystemExit(f"Missing {chem_path.name}")
    if not tutor_path.exists():
        raise SystemExit(f"Missing {tutor_path.name} — run step1_tutoring.py first.")

    rows: list[dict] = []

    with chem_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "dataset": "chemistry",
                    "code": r["code"],
                    "human_kappa": float(r["human_kappa"]),
                    "agent_disagreement_rate": float(r["agent_disagreement_rate"]),
                }
            )

    with tutor_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = r["code"]
            kappa = TUTORING_KAPPA.get(code)
            rows.append(
                {
                    "dataset": "tutoring",
                    "code": code,
                    "human_kappa": float(kappa) if kappa is not None else None,
                    "agent_disagreement_rate": float(r["agent_disagreement_rate"]),
                }
            )

    print("=" * 78)
    print("COMBINED 12-POINT TABLE")
    print("=" * 78)
    print(
        f"{'dataset':<12} {'code':<32} {'human_kappa':>12} "
        f"{'agent_disagreement_rate':>24}"
    )
    print("-" * 78)
    for r in rows:
        kappa_s = (
            f"{r['human_kappa']:.4f}" if r["human_kappa"] is not None else "(missing)"
        )
        print(
            f"{r['dataset']:<12} {r['code']:<32} {kappa_s:>12} "
            f"{r['agent_disagreement_rate']:>24.4f}"
        )

    out_csv = base / OUT_CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "code",
                "human_kappa",
                "agent_disagreement_rate",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "dataset": r["dataset"],
                    "code": r["code"],
                    "human_kappa": (
                        "" if r["human_kappa"] is None else r["human_kappa"]
                    ),
                    "agent_disagreement_rate": r["agent_disagreement_rate"],
                }
            )
    print(f"\nSaved combined table to {out_csv.name}")

    all_have_kappa = all(r["human_kappa"] is not None for r in rows)
    if not all_have_kappa:
        print(
            "\ntutoring kappa not filled in yet — waiting on Conrad for the "
            "full 12-point correlation."
        )
        chem = [r for r in rows if r["dataset"] == "chemistry"]
        _print_correlation(chem, label="4 chemistry points only")
        return

    _print_correlation(rows, label="all 12 points")
    _save_scatter(rows, base / OUT_PNG)


def _print_correlation(points: list[dict], label: str) -> None:
    import numpy as np
    from scipy import stats

    kappas = np.array([r["human_kappa"] for r in points], dtype=float)
    rates = np.array([r["agent_disagreement_rate"] for r in points], dtype=float)
    spearman = stats.spearmanr(kappas, rates)
    pearson = stats.pearsonr(kappas, rates)
    print("\n" + "=" * 78)
    print(f"CORRELATION ({label})")
    print("=" * 78)
    print(f"  Spearman r = {spearman.correlation:.4f},  p = {spearman.pvalue:.4f}")
    print(f"  Pearson  r = {pearson.statistic:.4f},  p = {pearson.pvalue:.4f}")
    print(
        "\nNote: a NEGATIVE correlation (low kappa -> high disagreement) "
        "supports the hypothesis."
    )


def _save_scatter(points: list[dict], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Scatter plot NOT saved (matplotlib unavailable): {e}")
        return

    colors = {"chemistry": "#1f77b4", "tutoring": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for dataset, color in colors.items():
        subset = [r for r in points if r["dataset"] == dataset]
        if not subset:
            continue
        xs = [r["human_kappa"] for r in subset]
        ys = [r["agent_disagreement_rate"] for r in subset]
        ax.scatter(xs, ys, c=color, s=70, label=dataset, zorder=3)
        for r in subset:
            ax.annotate(
                r["code"],
                (r["human_kappa"], r["agent_disagreement_rate"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    ax.set_xlabel("Human kappa")
    ax.set_ylabel("Agent disagreement rate")
    ax.set_title("12-point: human kappa vs agent disagreement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot to {out_png.name}")


if __name__ == "__main__":
    main()
