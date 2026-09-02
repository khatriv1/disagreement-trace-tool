#!/usr/bin/env python3
"""
Full, properly-standardized reasoning-vs-label test.

Two agents (Qwen + Llama) code all chemistry + tutoring utterances; we embed
rationales, aggregate per-code reasoning disagreement, then compare nested OLS
models with within-dataset z-scoring (the Step-1 standardization).

Rationales/labels are cached in rationale_cache.csv so agents are never re-run
once the cache covers the needed utterances.

Note: n=10 codes, low-powered — interpret descriptively.
Requires Ollama with qwen2.5:7b and llama3.1:8b (only if cache is incomplete).
"""

from __future__ import annotations

import ast
import csv
import json
import math
import random
from pathlib import Path

import numpy as np

random.seed(0)
np.random.seed(0)

from consensus_coding import (
    cosine,
    embed,
    extract_and_complete_code,
    generate,
    justification_of,
)
import reasoning_disagreement_lak24 as rd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_A = "qwen2.5:7b"
MODEL_B = "llama3.1:8b"
MAX_PER_DATASET = None  # None = ALL utterances

CHEM_CSV = "lak24-coded-utterances.csv"
TUT_FILES = [
    "Data_2/First Author Copy GPT-Then-Human - Transcript B.csv",
    "Data_2/First Author Copy GPT-Then-Human - transcript C.csv",
]
CHEM_DISAGREE_CSV = "step1_results_twomodels.csv"
TUT_DISAGREE_CSV = "step1_tutoring_results.csv"
CACHE_CSV = "rationale_cache.csv"
OUT_CSV = "reasoning_vs_label_full.csv"

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

TUT_CODEBOOK = """
The codes below describe tutor moves in online tutoring dialogue. They are NOT
mutually exclusive: an utterance can have several codes, or none. Mark 1 if the
utterance fits the code, 0 otherwise.

Greeting: A salutation or farewell between tutor and student.

Instruction: A specific instruction or direction the tutor gives about what to do.

Guiding feedback: Feedback on the student's work, or clarification/explanation of
  a concept; guiding the student through a problem.

Aligning to prior knowledge: The tutor points the student to a previously learned
  concept, often using 'remember'.

Understanding/Engagement-Tutor: The tutor checks the student's understanding,
  usually by asking a question.

Encouragement: Affirmative statements praising the student's effort or performance.
""".strip()

TUT_EXAMPLE = str({k: 0 for k in TUT_CODES})

CACHE_FIELDS = [
    "dataset",
    "idx",
    "text",
    "agentA_labels",
    "agentB_labels",
    "agentA_rationale",
    "agentB_rationale",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def as01(value) -> int:
    return 1 if value == 1 or value == "1" else 0


def load_chemistry(base: Path) -> list[str]:
    texts = []
    with (base / CHEM_CSV).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("remove_flag", "").strip().lower() == "yes":
                continue
            text = (row.get("utterance_combined") or "").strip()
            if text:
                texts.append(text)
    return texts


def load_tutoring(base: Path) -> list[str]:
    texts = []
    for rel in TUT_FILES:
        path = base / rel
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
            for values in reader:
                if len(values) < len(header):
                    values = values + [""] * (len(header) - len(values))
                if (values[speaker_i] or "").strip().casefold() != "tutor":
                    continue
                text = (values[text_i] or "").strip()
                if text:
                    texts.append(text)
    return texts


def sample_texts(texts: list[str], n: int | None) -> list[str]:
    if n is None or n >= len(texts):
        return list(texts)
    return random.sample(texts, n)


def load_label_disagreement(base: Path) -> dict[str, float]:
    rates: dict[str, float] = {}
    for name in (CHEM_DISAGREE_CSV, TUT_DISAGREE_CSV):
        path = base / name
        if not path.exists():
            raise SystemExit(f"Missing {name}")
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rates[r["code"]] = float(r["agent_disagreement_rate"])
    return rates


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def code_tutoring(name: str, personality: str, text: str, model: str) -> str:
    system = (
        f"You are {name}, a {personality} qualitative coding agent.\n"
        "Use the codebook below to analyze one utterance from tutoring dialogue.\n"
        f"{TUT_CODEBOOK}\n"
        "Always write your reasoning FIRST as 1 to 2 plain sentences explaining which "
        "codes apply and why; do not omit this reasoning.\n"
        "ONLY AFTER your reasoning, on the very last line, output a Python dictionary "
        f"with all six keys {TUT_CODES}, values 0 or 1 only, no markdown, no code "
        f"fences, and no text after the dictionary. Example:\n{TUT_EXAMPLE}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Utterance to code:\n{text}"},
    ]
    return generate(model, messages, {"temperature": 0.4})


def run_one_utterance(dataset: str, text: str, codes: list[str], coder) -> dict:
    template = {k: 0 for k in codes}
    raw_a = coder("Agent A", "bold and decisive", text, MODEL_A)
    raw_b = coder("Agent B", "cautious and conservative", text, MODEL_B)
    codes_a = extract_and_complete_code(raw_a, template)
    codes_b = extract_and_complete_code(raw_b, template)
    a01 = {k: as01(codes_a.get(k)) for k in codes}
    b01 = {k: as01(codes_b.get(k)) for k in codes}
    just_a = justification_of(raw_a) or "(empty)"
    just_b = justification_of(raw_b) or "(empty)"
    return {
        "dataset": dataset,
        "text": text,
        "agentA_labels": a01,
        "agentB_labels": b01,
        "agentA_rationale": just_a,
        "agentB_rationale": just_b,
    }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def load_cache(path: Path) -> dict[tuple[str, int], dict]:
    """Key = (dataset, idx) -> row dict with parsed label dicts."""
    if not path.exists():
        return {}
    out: dict[tuple[str, int], dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                idx = int(r["idx"])
                a_labels = ast.literal_eval(r["agentA_labels"])
                b_labels = ast.literal_eval(r["agentB_labels"])
            except (ValueError, SyntaxError, KeyError):
                continue
            if not isinstance(a_labels, dict) or not isinstance(b_labels, dict):
                continue
            key = (r["dataset"], idx)
            out[key] = {
                "dataset": r["dataset"],
                "idx": idx,
                "text": r["text"],
                "agentA_labels": a_labels,
                "agentB_labels": b_labels,
                "agentA_rationale": r["agentA_rationale"],
                "agentB_rationale": r["agentB_rationale"],
            }
    return out


def write_cache(path: Path, rows: list[dict]) -> None:
    """Rewrite full cache from an ordered list of utterance records."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "dataset": r["dataset"],
                    "idx": r["idx"],
                    "text": r["text"],
                    "agentA_labels": json.dumps(r["agentA_labels"], ensure_ascii=False),
                    "agentB_labels": json.dumps(r["agentB_labels"], ensure_ascii=False),
                    "agentA_rationale": r["agentA_rationale"],
                    "agentB_rationale": r["agentB_rationale"],
                }
            )


def collect_dataset(
    dataset: str,
    texts: list[str],
    codes: list[str],
    coder,
    cache: dict[tuple[str, int], dict],
    cache_path: Path,
) -> list[dict]:
    """Return ordered utterance records for this dataset; fill cache as needed."""
    n = len(texts)
    needed = {(dataset, i) for i in range(1, n + 1)}
    have = {k for k in needed if k in cache}
    if have == needed:
        print(f"[{dataset}] cache covers all {n} utterances — skipping agents.")
        rows = [cache[(dataset, i)] for i in range(1, n + 1)]
        # ensure codes present
        for r in rows:
            for c in codes:
                r["agentA_labels"].setdefault(c, 0)
                r["agentB_labels"].setdefault(c, 0)
        return rows

    missing_idxs = sorted(i for (ds, i) in (needed - have))
    print(
        f"[{dataset}] {len(have)}/{n} cached; running agents on "
        f"{len(missing_idxs)} utterances ({MODEL_A} + {MODEL_B})..."
    )

    # Start from existing cached rows for this dataset (preserve other datasets)
    by_idx = {
        i: cache[(dataset, i)]
        for i in range(1, n + 1)
        if (dataset, i) in cache
    }

    for idx in missing_idxs:
        text = texts[idx - 1]
        print(f"  [{idx}/{n}] ({len(text.split())} words)...")
        rec = run_one_utterance(dataset, text, codes, coder)
        rec["idx"] = idx
        by_idx[idx] = rec
        # Merge into global cache and rewrite (resume-safe)
        cache[(dataset, idx)] = rec
        # Rebuild ordered list of ALL cached rows across datasets for rewrite
        all_cached = sorted(cache.values(), key=lambda r: (r["dataset"], r["idx"]))
        write_cache(cache_path, all_cached)
        diffs = [c for c in codes if rec["agentA_labels"].get(c) != rec["agentB_labels"].get(c)]
        print("    " + (f"label_disagree={diffs}" if diffs else "label_agree"))

    rows = [by_idx[i] for i in range(1, n + 1)]
    return rows


# ---------------------------------------------------------------------------
# Part A — embeddings + aggregation
# ---------------------------------------------------------------------------
def attach_reasoning_disagreement(rows: list[dict]) -> None:
    print(f"\nEmbedding {len(rows)} rationale pairs ...")
    texts_a = [r["agentA_rationale"] for r in rows]
    texts_b = [r["agentB_rationale"] for r in rows]
    # batch embed all rationales
    all_emb = embed(texts_a + texts_b)
    n = len(rows)
    emb_a = all_emb[:n]
    emb_b = all_emb[n:]
    for r, u, v in zip(rows, emb_a, emb_b):
        sim = float(cosine(u, v))
        if math.isnan(sim):
            sim = 0.0
        r["reasoning_similarity"] = sim
        r["reasoning_disagreement"] = 1.0 - sim


def aggregate_per_code(rows: list[dict], codes: list[str], dataset: str) -> list[dict]:
    out = []
    for code in codes:
        on_dis = []
        all_vals = []
        for r in rows:
            rd_val = r["reasoning_disagreement"]
            all_vals.append(rd_val)
            a = as01(r["agentA_labels"].get(code))
            b = as01(r["agentB_labels"].get(code))
            if a != b:
                on_dis.append(rd_val)
        reason_all = float(np.mean(all_vals)) if all_vals else float("nan")
        reason_ondis = float(np.mean(on_dis)) if on_dis else float("nan")
        out.append(
            {
                "dataset": dataset,
                "code": code,
                "n_utterances": len(all_vals),
                "n_label_disagree": len(on_dis),
                "reason_all": reason_all,
                "reason_ondis": reason_ondis,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Part B — standardized incremental models
# ---------------------------------------------------------------------------
def zscore_within(
    values: list[float], datasets: list[str]
) -> np.ndarray:
    """Z-score values within each dataset, return array in original order."""
    arr = np.array(values, dtype=float)
    out = np.zeros_like(arr)
    for ds in set(datasets):
        idx = [i for i, d in enumerate(datasets) if d == ds]
        sub = arr[idx]
        std = sub.std(ddof=0)
        if std == 0:
            out[idx] = 0.0
        else:
            out[idx] = (sub - sub.mean()) / std
    return out


def print_correlation(xs: list[float], ys: list[float], label: str) -> None:
    from scipy import stats

    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    spearman = stats.spearmanr(x, y)
    pearson = stats.pearsonr(x, y)
    print(f"  {label}")
    print(f"    Spearman r = {spearman.correlation:.4f},  p = {spearman.pvalue:.4f}")
    print(f"    Pearson  r = {pearson.statistic:.4f},  p = {pearson.pvalue:.4f}")


def run_lrt_zscored(
    table: list[dict], reasoning_key: str, title: str
) -> None:
    """Model 1/2 with kappa and predictors z-scored within dataset."""
    import statsmodels.api as sm
    from scipy.stats import chi2

    datasets = [r["dataset"] for r in table]
    y = zscore_within([r["human_kappa"] for r in table], datasets)
    x_label = zscore_within([r["label_disagreement"] for r in table], datasets)
    x_reason = zscore_within([r[reasoning_key] for r in table], datasets)

    X1 = sm.add_constant(x_label)
    m1 = sm.OLS(y, X1).fit()
    X2 = sm.add_constant(np.column_stack([x_label, x_reason]))
    m2 = sm.OLS(y, X2).fit()

    lr = 2.0 * (m2.llf - m1.llf)
    p_lr = float(chi2.sf(lr, df=1))

    print(f"\n===== {title} =====")
    print("Note: n=10 codes, low-powered — interpret descriptively.")
    print(f"  (outcome + predictors z-scored WITHIN each dataset)")
    print(f"\n  Model 1: kappa_z ~ label_z")
    print(f"    R^2 = {m1.rsquared:.4f}")
    print(f"    const={m1.params[0]:.4f} (p={m1.pvalues[0]:.4f})")
    print(f"    label_z={m1.params[1]:.4f} (p={m1.pvalues[1]:.4f})")
    print(f"\n  Model 2: kappa_z ~ label_z + reasoning_z")
    print(f"    R^2 = {m2.rsquared:.4f}")
    print(f"    const={m2.params[0]:.4f} (p={m2.pvalues[0]:.4f})")
    print(f"    label_z={m2.params[1]:.4f} (p={m2.pvalues[1]:.4f})")
    print(f"    reasoning_z={m2.params[2]:.4f} (p={m2.pvalues[2]:.4f})")
    print(f"\n  LR test: LR={lr:.4f}, df=1, p={p_lr:.4f}")
    if p_lr < 0.05:
        print(
            f"  VERDICT: YES — adding {reasoning_key} significantly improves "
            f"prediction of human difficulty beyond label disagreement (p={p_lr:.4f})."
        )
    else:
        print(
            f"  VERDICT: NO — adding {reasoning_key} does NOT significantly improve "
            f"prediction beyond label disagreement (p={p_lr:.4f}). Descriptive only (n=10)."
        )


def run_lrt_raw_with_dataset_dummy(
    table: list[dict], reasoning_key: str, title: str
) -> None:
    """Cross-check: raw predictors + dataset dummy in both models."""
    import statsmodels.api as sm
    from scipy.stats import chi2

    y = np.array([r["human_kappa"] for r in table], dtype=float)
    x_label = np.array([r["label_disagreement"] for r in table], dtype=float)
    x_reason = np.array([r[reasoning_key] for r in table], dtype=float)
    # chemistry=0, tutoring=1
    dummy = np.array(
        [1.0 if r["dataset"] == "tutoring" else 0.0 for r in table], dtype=float
    )

    X1 = sm.add_constant(np.column_stack([x_label, dummy]))
    m1 = sm.OLS(y, X1).fit()
    X2 = sm.add_constant(np.column_stack([x_label, x_reason, dummy]))
    m2 = sm.OLS(y, X2).fit()

    lr = 2.0 * (m2.llf - m1.llf)
    p_lr = float(chi2.sf(lr, df=1))

    print(f"\n===== {title} (cross-check: raw + dataset dummy) =====")
    print("Note: n=10 codes, low-powered — interpret descriptively.")
    print(f"\n  Model 1: kappa ~ label + dataset_dummy")
    print(f"    R^2 = {m1.rsquared:.4f}")
    print(
        f"    const={m1.params[0]:.4f}, label={m1.params[1]:.4f} (p={m1.pvalues[1]:.4f}), "
        f"dummy={m1.params[2]:.4f} (p={m1.pvalues[2]:.4f})"
    )
    print(f"\n  Model 2: kappa ~ label + reasoning + dataset_dummy")
    print(f"    R^2 = {m2.rsquared:.4f}")
    print(
        f"    const={m2.params[0]:.4f}, label={m2.params[1]:.4f} (p={m2.pvalues[1]:.4f}), "
        f"reasoning={m2.params[2]:.4f} (p={m2.pvalues[2]:.4f}), "
        f"dummy={m2.params[3]:.4f} (p={m2.pvalues[3]:.4f})"
    )
    print(f"\n  LR test: LR={lr:.4f}, df=1, p={p_lr:.4f}")
    if p_lr < 0.05:
        print(
            f"  VERDICT: YES — adding {reasoning_key} significantly improves "
            f"prediction with dataset dummy control (p={p_lr:.4f})."
        )
    else:
        print(
            f"  VERDICT: NO — adding {reasoning_key} does NOT significantly improve "
            f"prediction with dataset dummy control (p={p_lr:.4f}). Descriptive only (n=10)."
        )


def part_b(code_rows: list[dict], label_rates: dict[str, float]) -> list[dict]:
    table = []
    for r in code_rows:
        code = r["code"]
        if code not in HUMAN_KAPPA:
            continue
        if code not in label_rates:
            raise SystemExit(f"No label disagreement for {code}")
        if math.isnan(r["reason_ondis"]):
            # rare: no label disagreements at all — fall back to reason_all for ondis
            reason_ondis = r["reason_all"]
        else:
            reason_ondis = r["reason_ondis"]
        table.append(
            {
                "dataset": r["dataset"],
                "code": code,
                "human_kappa": HUMAN_KAPPA[code],
                "label_disagreement": label_rates[code],
                "reason_all": r["reason_all"],
                "reason_ondis": reason_ondis,
                "n_label_disagree": r["n_label_disagree"],
                "n_utterances": r["n_utterances"],
            }
        )

    print("\n" + "=" * 88)
    print("PART B table (10 codes with human kappa)")
    print("=" * 88)
    print(
        f"{'code':<32} {'kappa':>6} {'label_dis':>10} {'reason_all':>11} "
        f"{'reason_ondis':>12} {'n_lab_dis':>10}"
    )
    print("-" * 88)
    for r in table:
        print(
            f"{r['code']:<32} {r['human_kappa']:>6.2f} {r['label_disagreement']:>10.4f} "
            f"{r['reason_all']:>11.4f} {r['reason_ondis']:>12.4f} "
            f"{r['n_label_disagree']:>10d}"
        )

    # Correlations (z-scored within dataset)
    print("\n" + "=" * 88)
    print("CORRELATIONS (z-scored within dataset; n=10, descriptive)")
    print("=" * 88)
    datasets = [r["dataset"] for r in table]
    kappa_z = zscore_within([r["human_kappa"] for r in table], datasets).tolist()
    label_z = zscore_within(
        [r["label_disagreement"] for r in table], datasets
    ).tolist()
    all_z = zscore_within([r["reason_all"] for r in table], datasets).tolist()
    ondis_z = zscore_within([r["reason_ondis"] for r in table], datasets).tolist()

    print("\n--- reason_all vs human kappa ---")
    print_correlation(all_z, kappa_z, "z-scored within dataset")
    print("\n--- reason_ondis vs human kappa ---")
    print_correlation(ondis_z, kappa_z, "z-scored within dataset")
    print("\n--- reason_all vs label disagreement ---")
    print_correlation(all_z, label_z, "z-scored within dataset")
    print("\n--- reason_ondis vs label disagreement ---")
    print_correlation(ondis_z, label_z, "z-scored within dataset")

    # Main LRT: within-dataset z-scoring
    run_lrt_zscored(table, "reason_all", "LRT with reason_all (within-dataset z-score)")
    run_lrt_zscored(
        table, "reason_ondis", "LRT with reason_ondis (within-dataset z-score)"
    )

    # Cross-check: raw + dataset dummy
    run_lrt_raw_with_dataset_dummy(
        table, "reason_all", "LRT with reason_all"
    )
    run_lrt_raw_with_dataset_dummy(
        table, "reason_ondis", "LRT with reason_ondis"
    )

    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    base = Path(__file__).resolve().parent
    cache_path = base / CACHE_CSV
    label_rates = load_label_disagreement(base)

    chem_texts = sample_texts(load_chemistry(base), MAX_PER_DATASET)
    tut_texts = sample_texts(load_tutoring(base), MAX_PER_DATASET)
    print(
        f"Sample sizes: chemistry={len(chem_texts)}, tutoring={len(tut_texts)} "
        f"(MAX_PER_DATASET={MAX_PER_DATASET})"
    )
    print("Note: n=10 codes in Part B, low-powered — interpret descriptively.")

    cache = load_cache(cache_path)
    print(f"Loaded {len(cache)} cached utterance rows from {CACHE_CSV}"
          if cache else f"No existing {CACHE_CSV}")

    def chem_coder(name, personality, text, model):
        return rd.code_utterance(name, personality, text, model=model)

    chem_rows = collect_dataset(
        "chemistry", chem_texts, CHEM_CODES, chem_coder, cache, cache_path
    )
    # reload cache after chemistry (may have grown)
    cache = load_cache(cache_path)
    tut_rows = collect_dataset(
        "tutoring", tut_texts, TUT_CODES, code_tutoring, cache, cache_path
    )

    # Final cache write (ordered)
    all_rows = chem_rows + tut_rows
    write_cache(cache_path, all_rows)
    print(f"\nRationale cache saved to {CACHE_CSV} ({len(all_rows)} utterances).")

    attach_reasoning_disagreement(all_rows)

    chem_codes = aggregate_per_code(chem_rows, CHEM_CODES, "chemistry")
    tut_codes = aggregate_per_code(tut_rows, TUT_CODES, "tutoring")
    all_code_rows = chem_codes + tut_codes

    print("\n" + "=" * 88)
    print("PART A — per-code reasoning disagreement")
    print("=" * 88)
    print(
        f"{'dataset':<12} {'code':<32} {'n':>6} {'n_lab_dis':>10} "
        f"{'reason_all':>11} {'reason_ondis':>12}"
    )
    print("-" * 88)
    for r in all_code_rows:
        ondis_s = (
            f"{r['reason_ondis']:.4f}"
            if not math.isnan(r["reason_ondis"])
            else "   (none)"
        )
        print(
            f"{r['dataset']:<12} {r['code']:<32} {r['n_utterances']:>6d} "
            f"{r['n_label_disagree']:>10d} {r['reason_all']:>11.4f} {ondis_s:>12}"
        )

    table = part_b(all_code_rows, label_rates)

    out_path = base / OUT_CSV
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "code",
                "human_kappa",
                "label_disagreement",
                "reason_all",
                "reason_ondis",
                "n_label_disagree",
                "n_utterances",
            ],
        )
        writer.writeheader()
        writer.writerows(table)
    print(f"\nSaved per-code results to {OUT_CSV}")


if __name__ == "__main__":
    main()
