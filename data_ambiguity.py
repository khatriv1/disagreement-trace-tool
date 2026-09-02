#!/usr/bin/env python3
"""
Data-driven ambiguity: for each code, how semantically close are positive
utterances to negative ones (mean max cosine to opposite-labeled neighbors).

Uses the same sentence-transformers embedder as consensus_coding.py.
Does NOT call LLM agents.

Note: n=10 codes — descriptive, not a significance test.
"""

from pathlib import Path
import csv

import numpy as np
from scipy import stats

from consensus_coding import embed

CHEM_CSV = "lak24-coded-utterances.csv"
TUT_FILES = [
    "Data_2/First Author Copy GPT-Then-Human - Transcript B.csv",
    "Data_2/First Author Copy GPT-Then-Human - transcript C.csv",
]
CHEM_CODES = ["process", "plan", "act", "wrong"]
TUT_CODES = [
    "Greeting",
    "Instruction",
    "Guiding feedback",
    "Aligning to prior knowledge",
    "Understanding/Engagement-Tutor",
    "Encouragement",
]

HUMAN_KAPPA = {
    "process": 0.78,
    "plan": 0.90,
    "act": 0.77,
    "wrong": 1.00,
    "Greeting": 0.85,
    "Encouragement": 0.80,
    "Instruction": 0.66,
    "Guiding feedback": 0.66,
    "Aligning to prior knowledge": 0.66,
    "Understanding/Engagement-Tutor": 0.60,
}

CHEM_DISAGREE_CSV = "step1_results_twomodels.csv"
TUT_DISAGREE_CSV = "step1_tutoring_results.csv"
OUT_CSV = "data_ambiguity.csv"


def parse01(value) -> int:
    if value is None:
        return 0
    s = str(value).strip().casefold()
    if s in ("1", "1.0", "true", "yes"):
        return 1
    return 0


def human_label_for_code(header: list[str], values: list[str], code: str) -> int:
    target = code.strip().casefold()
    matched = []
    for name, val in zip(header, values):
        if (name or "").strip().casefold() == target:
            matched.append(parse01(val))
    return max(matched) if matched else 0


def load_chemistry(base: Path) -> list[dict]:
    path = base / CHEM_CSV
    records = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("remove_flag", "").strip().lower() == "yes":
                continue
            text = (row.get("utterance_combined") or "").strip()
            if not text:
                continue
            labels = {c: parse01(row.get(c)) for c in CHEM_CODES}
            records.append({"text": text, "labels": labels})
    return records


def load_tutoring(base: Path) -> list[dict]:
    records = []
    for rel in TUT_FILES:
        path = base / rel
        print(f"  Reading {path.name} ...")
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            speaker_i = text_i = None
            for i, name in enumerate(header):
                n = (name or "").strip().casefold()
                if speaker_i is None and n == "speaker_type":
                    speaker_i = i
                if text_i is None and n == "text":
                    text_i = i
            if speaker_i is None or text_i is None:
                raise SystemExit(f"Missing speaker_type/text in {path}")
            for values in reader:
                if len(values) < len(header):
                    values = values + [""] * (len(header) - len(values))
                if (values[speaker_i] or "").strip().casefold() != "tutor":
                    continue
                text = (values[text_i] or "").strip()
                if not text:
                    continue
                labels = {
                    c: human_label_for_code(header, values, c) for c in TUT_CODES
                }
                records.append({"text": text, "labels": labels})
    return records


def cosine_matrix(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Cosine similarities between each POS row and each NEG row. Shape (n_pos, n_neg)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos_norms = np.linalg.norm(pos, axis=1, keepdims=True)
    neg_norms = np.linalg.norm(neg, axis=1, keepdims=True)
    pos_n = np.divide(pos, pos_norms, out=np.zeros_like(pos), where=pos_norms > 0)
    neg_n = np.divide(neg, neg_norms, out=np.zeros_like(neg), where=neg_norms > 0)
    return pos_n @ neg_n.T


def ambiguity_for_code(
    vecs: np.ndarray, labels: np.ndarray
) -> tuple[float | None, int]:
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos = int(len(pos_idx))
    if n_pos == 0 or len(neg_idx) == 0:
        return None, n_pos
    sims = cosine_matrix(vecs[pos_idx], vecs[neg_idx])
    max_to_neg = sims.max(axis=1)
    return float(max_to_neg.mean()), n_pos


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


def load_disagreement(base: Path) -> dict[str, float]:
    rates: dict[str, float] = {}
    for name in (CHEM_DISAGREE_CSV, TUT_DISAGREE_CSV):
        path = base / name
        if not path.exists():
            raise SystemExit(f"Missing {name}")
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rates[r["code"]] = float(r["agent_disagreement_rate"])
    return rates


def analyze_dataset(
    name: str, records: list[dict], codes: list[str]
) -> list[dict]:
    print(f"\nEmbedding {len(records)} {name} utterances ...")
    texts = [r["text"] for r in records]
    vecs = np.array(embed(texts), dtype=float)
    print(f"  Embedding shape: {vecs.shape}")

    rows = []
    print("\n" + "=" * 72)
    print(f"{name}: data-driven ambiguity")
    print("=" * 72)
    print(f"{'code':<40} {'n_positive':>12} {'ambiguity':>12}")
    print("-" * 72)
    for code in codes:
        labels = np.array([r["labels"][code] for r in records], dtype=int)
        amb, n_pos = ambiguity_for_code(vecs, labels)
        if amb is None:
            print(f"{code:<40} {n_pos:>12d} {'(skipped)':>12}")
            continue
        print(f"{code:<40} {n_pos:>12d} {amb:>12.4f}")
        rows.append(
            {
                "dataset": name,
                "code": code,
                "n_positive": n_pos,
                "ambiguity": amb,
            }
        )
    return rows


def main() -> None:
    base = Path(__file__).resolve().parent

    print("Loading chemistry ...")
    chem = load_chemistry(base)
    print(f"  {len(chem)} utterances")

    print("Loading tutoring ...")
    tut = load_tutoring(base)
    print(f"  {len(tut)} tutor utterances")

    all_rows: list[dict] = []
    all_rows.extend(analyze_dataset("chemistry", chem, CHEM_CODES))
    all_rows.extend(analyze_dataset("tutoring", tut, TUT_CODES))

    out_csv = base / OUT_CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "code", "n_positive", "ambiguity"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved table to {out_csv.name}")

    disagree = load_disagreement(base)
    corr_rows = []
    for r in all_rows:
        code = r["code"]
        if code not in HUMAN_KAPPA:
            continue
        if code not in disagree:
            raise SystemExit(f"No disagreement rate for {code}")
        corr_rows.append(
            {
                "dataset": r["dataset"],
                "code": code,
                "ambiguity": r["ambiguity"],
                "human_kappa": HUMAN_KAPPA[code],
                "agent_disagreement_rate": disagree[code],
            }
        )

    chem_c = [r for r in corr_rows if r["dataset"] == "chemistry"]
    tut_c = [r for r in corr_rows if r["dataset"] == "tutoring"]

    print("\n" + "=" * 72)
    print("CORRELATIONS (10 codes)")
    print("=" * 72)
    print("Note: n=10, descriptive — not a significance test.")

    def correlate(y_key: str, title: str) -> None:
        print(f"\n--- ambiguity vs {title} ---")
        xs = [r["ambiguity"] for r in corr_rows]
        ys = [r[y_key] for r in corr_rows]
        print_correlation(xs, ys, "raw-pooled (10 codes)")
        z_x: list[float] = []
        z_y: list[float] = []
        for subset in (chem_c, tut_c):
            z_x.extend(zscore([r["ambiguity"] for r in subset]))
            z_y.extend(zscore([r[y_key] for r in subset]))
        print_correlation(z_x, z_y, "z-scored within each dataset (10 codes)")

    correlate("human_kappa", "human kappa (expect NEGATIVE)")
    correlate(
        "agent_disagreement_rate",
        "agent disagreement rate (expect POSITIVE)",
    )


if __name__ == "__main__":
    main()
