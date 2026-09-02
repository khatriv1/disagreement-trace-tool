#!/usr/bin/env python3
"""
reasoning_verbosity_check.py
Test whether reasoning_disagree (1 - cosine of rationale embeddings) is a proxy
for rationale length/verbosity rather than genuine shared confusion between agents.

Inputs: rationale_cache.csv, datapoint_table.csv  (no LLM agents re-run)
Descriptive analysis at the utterance level (n ≈ 1 003).
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
import statsmodels.formula.api as smf
import pandas as pd

BASE = Path(__file__).resolve().parent
CACHE_CSV     = BASE / "rationale_cache.csv"
DATAPOINT_CSV = BASE / "datapoint_table.csv"


# ── helpers ──────────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def cosine_safe(u, v) -> float:
    a, b = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def zscore(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    mu, sd = a.mean(), a.std()
    return (a - mu) / sd if sd != 0 else a - mu


def corr_row(label: str, x, y) -> None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
    print(
        f"  {label:45s}  "
        f"Spearman r={sr:+.3f} p={sp:.4f}  "
        f"Pearson r={pr:+.3f} p={pp:.4f}  (n={len(x)})"
    )


# ── load cache ────────────────────────────────────────────────────────────────

print("Loading rationale_cache.csv ...")
cache_rows = []
with CACHE_CSV.open(newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        cache_rows.append({
            "dataset": r["dataset"],
            "idx":     int(r["idx"]),
            "agentA_rationale": r["agentA_rationale"],
            "agentB_rationale": r["agentB_rationale"],
        })
print(f"  {len(cache_rows)} utterances")

# ── embed rationales ──────────────────────────────────────────────────────────

print("Embedding rationales ...")
from consensus_coding import embed

all_texts = (
    [r["agentA_rationale"] for r in cache_rows]
    + [r["agentB_rationale"] for r in cache_rows]
)
all_emb = embed(all_texts)
n = len(cache_rows)
emb_a = all_emb[:n]
emb_b = all_emb[n:]
print("  Done.")

# ── build utterance-level table ───────────────────────────────────────────────

utt_rows = []
for r, ua, ub in zip(cache_rows, emb_a, emb_b):
    sim = cosine_safe(ua, ub)
    if math.isnan(sim):
        sim = 0.0
    rd = 1.0 - sim

    len_a = word_count(r["agentA_rationale"])
    len_b = word_count(r["agentB_rationale"])

    utt_rows.append({
        "dataset":             r["dataset"],
        "item_id":             f"{r['dataset'][:4]}_{r['idx']:04d}",
        "reasoning_disagree":  rd,
        "len_A":               len_a,
        "len_B":               len_b,
        "len_mean":            (len_a + len_b) / 2.0,
        "len_diff":            abs(len_a - len_b),
    })

utt_df = pd.DataFrame(utt_rows)
print(f"\nUtterance table: {len(utt_df)} rows")
print(utt_df[["reasoning_disagree", "len_A", "len_B", "len_mean", "len_diff"]].describe().round(3))

# ── attach Qwen correctness from datapoint_table ──────────────────────────────

dp = pd.read_csv(DATAPOINT_CSV)
dp["llm_correct_A"] = (
    dp["agentA_label"].astype(int) == dp["human_label"].astype(int)
).astype(int)

# utterance-level outcome: majority of codes correct
utt_correct = (
    dp.groupby("item_id")["llm_correct_A"]
    .agg(prop_correct="mean")
    .reset_index()
)
# binary majority-correct outcome
utt_correct["majority_correct"] = (utt_correct["prop_correct"] >= 0.5).astype(int)

utt_df = utt_df.merge(utt_correct, on="item_id", how="left")
print(f"\nAfter merging correctness: {utt_df['majority_correct'].notna().sum()} / {len(utt_df)} rows have outcome")

# ── STEP 1: does reasoning_disagree track length? ─────────────────────────────

print("\n" + "=" * 78)
print("STEP 1 — Correlations: reasoning_disagree vs length features")
print("=" * 78)
rd  = utt_df["reasoning_disagree"].values
for col, label in [
    ("len_mean", "reasoning_disagree vs len_mean"),
    ("len_diff", "reasoning_disagree vs len_diff"),
    ("len_A",    "reasoning_disagree vs len_A (Qwen)"),
    ("len_B",    "reasoning_disagree vs len_B (Llama)"),
]:
    corr_row(label, rd, utt_df[col].values)

# ── STEP 2: logistic regressions at utterance level ───────────────────────────

print("\n" + "=" * 78)
print("STEP 2 — Logistic regressions: outcome = majority_correct (Qwen)")
print("  Predictors z-scored; utterance level (n ≈ 1 003)")
print("=" * 78)

sub = utt_df.dropna(subset=["majority_correct"]).copy()
sub["rd_z"]   = zscore(sub["reasoning_disagree"])
sub["lenA_z"] = zscore(sub["len_A"])
sub["lenM_z"] = zscore(sub["len_mean"])

def fit_logit(formula: str, data: pd.DataFrame) -> None:
    try:
        m = smf.logit(formula, data=data).fit(disp=False)
        print(f"\n  Formula: {formula}")
        tbl = m.summary2().tables[1][["Coef.", "Std.Err.", "z", "P>|z|"]]
        print(tbl.to_string())
    except Exception as e:
        print(f"\n  Formula: {formula}  — ERROR: {e}")

fit_logit("majority_correct ~ rd_z",           sub)
fit_logit("majority_correct ~ lenA_z",         sub)
fit_logit("majority_correct ~ lenM_z",         sub)
fit_logit("majority_correct ~ rd_z + lenA_z",  sub)

# significance of reasoning_disagree in combined model
m_comb = smf.logit("majority_correct ~ rd_z + lenA_z", data=sub).fit(disp=False)
p_rd_in_combined = float(m_comb.pvalues.get("rd_z", float("nan")))
coef_rd_comb     = float(m_comb.params.get("rd_z", float("nan")))

# significance in lone model
m_lone = smf.logit("majority_correct ~ rd_z", data=sub).fit(disp=False)
p_rd_alone = float(m_lone.pvalues.get("rd_z", float("nan")))
coef_rd_alone = float(m_lone.params.get("rd_rd", float("nan")))  # will be nan
coef_rd_alone = float(m_lone.params.get("rd_z", float("nan")))

# correlation threshold (|r| >= 0.15 = small-to-medium, clearly non-trivial)
# and significance threshold for verdict
sr_lenA, _ = spearmanr(utt_df["reasoning_disagree"], utt_df["len_A"])
sr_lenM, _ = spearmanr(utt_df["reasoning_disagree"], utt_df["len_mean"])
clearly_correlated = abs(sr_lenA) >= 0.15 or abs(sr_lenM) >= 0.15
loses_significance = p_rd_in_combined >= 0.05

# ── STEP 3: verdict ───────────────────────────────────────────────────────────

print("\n" + "=" * 78)
print("STEP 3 — VERDICT")
print("=" * 78)
print(
    f"  reasoning_disagree ~ len_A (Qwen): Spearman r={sr_lenA:+.3f} "
    f"  (clearly correlated: {clearly_correlated})"
)
print(
    f"  reasoning_disagree in combined model (+ len_A): "
    f"coef={coef_rd_comb:+.4f}, p={p_rd_in_combined:.4f} "
    f"  (loses significance at p≥0.05: {loses_significance})"
)

if clearly_correlated and loses_significance:
    verdict = (
        "reasoning_disagree IS largely a verbosity/length proxy: "
        "it correlates with rationale length AND loses significance "
        "when length is controlled."
    )
elif clearly_correlated and not loses_significance:
    verdict = (
        "reasoning_disagree correlates with length but retains independent "
        "significance when length is controlled — partial verbosity proxy, "
        "but also carries additional signal."
    )
elif not clearly_correlated and loses_significance:
    verdict = (
        "reasoning_disagree does NOT clearly correlate with length but loses "
        "significance in the combined model — suggests multicollinearity or "
        "low power rather than a verbosity effect."
    )
else:
    verdict = (
        "reasoning_disagree is NOT explained by length alone: "
        "it does not clearly correlate with rationale length, "
        "and remains significant after controlling for length."
    )

print(f"\n  FINAL VERDICT: {verdict}")
print("\n  (Descriptive analysis, n ≈ 1 003 utterances.)")
