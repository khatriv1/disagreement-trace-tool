#!/usr/bin/env python3
"""
Step 1: Does the code with the lowest human agreement (kappa) also have the
highest LLM agent disagreement, in the chemistry think-aloud (LAK24) data?
"""

CSV_PATH = "lak24-coded-utterances.csv"
SAMPLE_SIZE = 200  # utterances to run; each needs 2 LLM calls (~2s each)
CODE_KEYS = ["process", "plan", "act", "wrong"]
HUMAN_KAPPA = {"process": 0.78, "plan": 0.90, "act": 0.77, "wrong": 1.00}
MODEL_A = "qwen2.5:7b"  # agent A
MODEL_B = "llama3.1:8b"  # agent B (different family -> real disagreement)
# Previous single-model (both Qwen) disagreement rates for comparison
PREV_SINGLE_MODEL_RATES = {
    "act": 0.055,
    "process": 0.16,
    "plan": 0.095,
    "wrong": 0.07,
}
import csv, random, numpy as np

random.seed(0)

from pathlib import Path

from scipy import stats

from consensus_coding import extract_and_complete_code
import reasoning_disagreement_lak24 as rd

CODEBOOK_TEMPLATE = {k: 0 for k in CODE_KEYS}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("remove_flag", "").strip().lower() == "yes":
                continue
            text = (row.get("utterance_combined") or "").strip()
            if not text:
                continue
            rows.append(row)
    return rows


def as01(value) -> int:
    return 1 if value == 1 or value == "1" else 0


def main() -> None:
    csv_file = Path(__file__).resolve().parent / CSV_PATH
    print(f"Loading utterances from {csv_file.name} ...")
    pool = load_rows(csv_file)
    print(f"  Eligible utterances after filters: {len(pool)}")
    if len(pool) < SAMPLE_SIZE:
        raise SystemExit(f"Only {len(pool)} utterances available; need {SAMPLE_SIZE}.")

    sample = random.sample(pool, SAMPLE_SIZE)
    print(f"  Sampled {SAMPLE_SIZE} utterances (seed=0).")
    print(f"  Agent A model={MODEL_A}; Agent B model={MODEL_B}")
    print(f"  Running both agents on each (~{SAMPLE_SIZE * 2} LLM calls)...\n")

    # Per-code disagreement counts: how often agents differ on that code
    disagree_counts = {k: 0 for k in CODE_KEYS}

    for idx, row in enumerate(sample, start=1):
        text = row["utterance_combined"].strip()
        print(
            f"[{idx}/{SAMPLE_SIZE}] coding utterance ({len(text.split())} words) "
            f"[A={MODEL_A}, B={MODEL_B}]..."
        )
        msg_a = rd.code_utterance(
            "Agent A", "bold and decisive", text, model=MODEL_A
        )
        msg_b = rd.code_utterance(
            "Agent B", "cautious and conservative", text, model=MODEL_B
        )
        codes_a = extract_and_complete_code(msg_a, CODEBOOK_TEMPLATE)
        codes_b = extract_and_complete_code(msg_b, CODEBOOK_TEMPLATE)

        a01 = {k: as01(codes_a.get(k)) for k in CODE_KEYS}
        b01 = {k: as01(codes_b.get(k)) for k in CODE_KEYS}
        diffs = [k for k in CODE_KEYS if a01[k] != b01[k]]
        for k in diffs:
            disagree_counts[k] += 1
        print(
            f"  A({MODEL_A})={a01}  B({MODEL_B})={b01}"
            + (f"  disagree on: {diffs}" if diffs else "  agree")
        )

    disagreement_rate = {
        k: disagree_counts[k] / SAMPLE_SIZE for k in CODE_KEYS
    }

    # Results table sorted by human_kappa ascending
    ordered = sorted(CODE_KEYS, key=lambda k: HUMAN_KAPPA[k])
    print("\n" + "=" * 60)
    print("RESULTS (sorted by human_kappa ascending)  [two-model: Qwen vs Llama]")
    print("=" * 60)
    print(
        f"{'code':<10} {'human_kappa':>12} {'agent_disagreement_rate':>24} "
        f"{'prev_single_qwen':>16}"
    )
    print("-" * 70)
    for k in ordered:
        print(
            f"{k:<10} {HUMAN_KAPPA[k]:>12.2f} {disagreement_rate[k]:>24.4f} "
            f"{PREV_SINGLE_MODEL_RATES[k]:>16.4f}"
        )
    print(
        "\nComparison vs previous single-model (both Qwen): "
        + ", ".join(
            f"{k} {disagreement_rate[k]:.3f} (was {PREV_SINGLE_MODEL_RATES[k]})"
            for k in ordered
        )
    )

    kappas = np.array([HUMAN_KAPPA[k] for k in CODE_KEYS], dtype=float)
    rates = np.array([disagreement_rate[k] for k in CODE_KEYS], dtype=float)

    spearman = stats.spearmanr(kappas, rates)
    pearson = stats.pearsonr(kappas, rates)

    print("\n" + "=" * 60)
    print("CORRELATION (human_kappa vs agent_disagreement_rate)")
    print("=" * 60)
    print(f"  Spearman r = {spearman.correlation:.4f},  p = {spearman.pvalue:.4f}")
    print(f"  Pearson  r = {pearson.statistic:.4f},  p = {pearson.pvalue:.4f}")
    print(
        "\nNote: a NEGATIVE correlation (low kappa -> high disagreement) supports "
        "the hypothesis. With n=4 codes this is descriptive, not a significance test."
    )

    # Save CSV table (twomodels suffix — keep prior single-model results intact)
    out_csv = Path(__file__).resolve().parent / "step1_results_twomodels.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["code", "human_kappa", "agent_disagreement_rate"],
        )
        writer.writeheader()
        for k in ordered:
            writer.writerow(
                {
                    "code": k,
                    "human_kappa": HUMAN_KAPPA[k],
                    "agent_disagreement_rate": disagreement_rate[k],
                }
            )
    print(f"\nSaved results table to {out_csv.name}")

    # Optional scatter plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(kappas, rates, s=60, zorder=3)
        for k in CODE_KEYS:
            ax.annotate(
                k,
                (HUMAN_KAPPA[k], disagreement_rate[k]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=10,
            )
        ax.set_xlabel("Human kappa")
        ax.set_ylabel("Agent disagreement rate")
        ax.set_title(
            f"Step 1 (two models): kappa vs disagreement\nA={MODEL_A}, B={MODEL_B}"
        )
        ax.set_xlim(0.7, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = Path(__file__).resolve().parent / "step1_scatter_twomodels.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved scatter plot to {out_png.name}")
    except Exception as e:
        print(f"Scatter plot NOT saved (matplotlib unavailable or plot failed): {e}")


if __name__ == "__main__":
    main()
