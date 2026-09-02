#!/usr/bin/env python3
"""
Combine chemistry (4 codes) + tutoring (6 codes with kappa) into a 10-point table.
Correlate human kappa vs agent disagreement within each dataset and pooled on
within-dataset z-scores.
"""

from pathlib import Path
import csv

# FILL IN from Conrad — tutoring human kappa (Technical or Logistics and Time
# Management omitted: no kappa and almost no data).
TUTORING_KAPPA = {
    "Greeting": 0.85,
    "Encouragement": 0.80,
    "Instruction": 0.66,
    "Guiding feedback": 0.66,
    "Aligning to prior knowledge": 0.66,
    "Understanding/Engagement-Tutor": 0.60,
}

CHEM_CSV = "step1_results_twomodels.csv"
TUTOR_CSV = "step1_tutoring_results.csv"
OUT_CSV = "combined_10points.csv"
OUT_PNG = "combined_10points_zscored.png"


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
            if code not in TUTORING_KAPPA:
                continue
            rows.append(
                {
                    "dataset": "tutoring",
                    "code": code,
                    "human_kappa": float(TUTORING_KAPPA[code]),
                    "agent_disagreement_rate": float(r["agent_disagreement_rate"]),
                }
            )

    chem = [r for r in rows if r["dataset"] == "chemistry"]
    tutor = [r for r in rows if r["dataset"] == "tutoring"]
    if len(chem) != 4 or len(tutor) != 6:
        raise SystemExit(
            f"Expected 4 chemistry + 6 tutoring rows; got {len(chem)} + {len(tutor)}."
        )

    _apply_within_dataset_zscores(rows)

    print("=" * 78)
    print("COMBINED 10-POINT TABLE (4 chemistry + 6 tutoring)")
    print("=" * 78)
    print(
        f"{'dataset':<12} {'code':<32} {'human_kappa':>12} "
        f"{'agent_disagreement_rate':>24}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['dataset']:<12} {r['code']:<32} {r['human_kappa']:>12.4f} "
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
                "z_human_kappa",
                "z_agent_disagreement_rate",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "dataset": r["dataset"],
                    "code": r["code"],
                    "human_kappa": r["human_kappa"],
                    "agent_disagreement_rate": r["agent_disagreement_rate"],
                    "z_human_kappa": r["z_human_kappa"],
                    "z_agent_disagreement_rate": r["z_agent_disagreement_rate"],
                }
            )
    print(f"\nSaved combined table to {out_csv.name}")

    _print_correlation(chem, label="a) chemistry only (4 codes)")
    _print_correlation(tutor, label="b) tutoring only (6 codes)")

    z_points = [
        {
            "human_kappa": r["z_human_kappa"],
            "agent_disagreement_rate": r["z_agent_disagreement_rate"],
        }
        for r in rows
    ]
    _print_correlation(
        z_points,
        label=(
            "c) combined, standardized within each dataset "
            "(10 z-scored points)"
        ),
    )

    print("\n" + "=" * 78)
    print("NAIVE RAW-POOLED (10 codes, not standardized)")
    print("=" * 78)
    print(
        "Note: confounded by datasets having different baselines for kappa and "
        "disagreement — do not use as the main result."
    )
    _print_correlation(rows, label="naive raw-pooled (reference only)", note=False)

    _save_zscatter(rows, base / OUT_PNG)


def _zscore_values(values: list[float]) -> list[float]:
    import numpy as np

    arr = np.array(values, dtype=float)
    std = arr.std(ddof=0)
    if std == 0:
        return [0.0] * len(values)
    mean = arr.mean()
    return ((arr - mean) / std).tolist()


def _apply_within_dataset_zscores(rows: list[dict]) -> None:
    for dataset in ("chemistry", "tutoring"):
        subset = [r for r in rows if r["dataset"] == dataset]
        zk = _zscore_values([r["human_kappa"] for r in subset])
        zd = _zscore_values([r["agent_disagreement_rate"] for r in subset])
        for r, k, d in zip(subset, zk, zd):
            r["z_human_kappa"] = k
            r["z_agent_disagreement_rate"] = d


def _print_correlation(points: list[dict], label: str, note: bool = True) -> None:
    import numpy as np
    from scipy import stats

    kappas = np.array(
        [r["human_kappa"] for r in points],
        dtype=float,
    )
    rates = np.array(
        [r["agent_disagreement_rate"] for r in points],
        dtype=float,
    )
    spearman = stats.spearmanr(kappas, rates)
    pearson = stats.pearsonr(kappas, rates)
    print("\n" + "=" * 78)
    print(f"CORRELATION — {label}")
    print("=" * 78)
    print(f"  Spearman r = {spearman.correlation:.4f},  p = {spearman.pvalue:.4f}")
    print(f"  Pearson  r = {pearson.statistic:.4f},  p = {pearson.pvalue:.4f}")
    if note:
        print(
            "\nNote: a NEGATIVE correlation (low kappa -> high disagreement) "
            "supports the hypothesis."
        )


def _save_zscatter(points: list[dict], out_png: Path) -> None:
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
        xs = [r["z_human_kappa"] for r in subset]
        ys = [r["z_agent_disagreement_rate"] for r in subset]
        ax.scatter(xs, ys, c=color, s=70, label=dataset, zorder=3)
        for r in subset:
            ax.annotate(
                r["code"],
                (r["z_human_kappa"], r["z_agent_disagreement_rate"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    ax.set_xlabel("Human kappa (z-scored within dataset)")
    ax.set_ylabel("Agent disagreement rate (z-scored within dataset)")
    ax.set_title("10-point combined: within-dataset z-scored kappa vs disagreement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot to {out_png.name}")


if __name__ == "__main__":
    main()
