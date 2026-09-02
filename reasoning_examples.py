#!/usr/bin/env python3
"""
reasoning_examples.py
Pull concrete utterance examples to illustrate that reasoning_disagree mostly
reflects stylistic/length differences rather than genuine coding disagreement.

Inputs: rationale_cache.csv, datapoint_table.csv  (no agents re-run)
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

BASE          = Path(__file__).resolve().parent
CACHE_CSV     = BASE / "rationale_cache.csv"
DATAPOINT_CSV = BASE / "datapoint_table.csv"
OUT_TXT       = BASE / "reasoning_examples.txt"


# ── helpers ───────────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def cosine_safe(u, v) -> float:
    a, b = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── load cache ────────────────────────────────────────────────────────────────

print("Loading rationale_cache.csv ...")
cache_rows = []
with CACHE_CSV.open(newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        cache_rows.append({
            "dataset":           r["dataset"],
            "idx":               int(r["idx"]),
            "text":              r["text"],
            "agentA_labels":     json.loads(r["agentA_labels"]),
            "agentB_labels":     json.loads(r["agentB_labels"]),
            "agentA_rationale":  r["agentA_rationale"],
            "agentB_rationale":  r["agentB_rationale"],
        })
print(f"  {len(cache_rows)} utterances")

# ── load per-utterance label_disagree from datapoint_table ───────────────────
# label_disagree at utterance level = 1 if ANY code has disagreement

import pandas as pd
dp = pd.read_csv(DATAPOINT_CSV)
utt_label_dis = (
    dp.groupby("item_id")["label_disagree"]
    .max()         # 1 if any code disagreed
    .rename("any_label_disagree")
    .reset_index()
)
utt_all_agree = (
    dp.groupby("item_id")["label_disagree"]
    .max()
    .eq(0)         # True when EVERY code agreed
    .rename("all_agree")
    .reset_index()
)
agree_set = set(
    utt_all_agree.loc[utt_all_agree["all_agree"], "item_id"]
)

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

# ── build utterance feature table ─────────────────────────────────────────────

rows = []
for r, ua, ub in zip(cache_rows, emb_a, emb_b):
    sim = cosine_safe(ua, ub)
    if math.isnan(sim):
        sim = 0.0
    rd = 1.0 - sim

    item_id = f"{r['dataset'][:4]}_{r['idx']:04d}"
    all_agree = item_id in agree_set

    rows.append({
        "item_id":            item_id,
        "dataset":            r["dataset"],
        "text":               r["text"],
        "agentA_labels":      r["agentA_labels"],
        "agentB_labels":      r["agentB_labels"],
        "agentA_rationale":   r["agentA_rationale"],
        "agentB_rationale":   r["agentB_rationale"],
        "reasoning_disagree": rd,
        "len_A":              word_count(r["agentA_rationale"]),
        "len_B":              word_count(r["agentB_rationale"]),
        "len_diff":           abs(word_count(r["agentA_rationale"])
                                  - word_count(r["agentB_rationale"])),
        "all_labels_agree":   all_agree,
    })

feat_df = pd.DataFrame(rows)
print(f"\nFeature table: {len(feat_df)} rows")
print(f"  Utterances where ALL codes agreed: {feat_df['all_labels_agree'].sum()}")


# ── formatter ─────────────────────────────────────────────────────────────────

SEP = "─" * 76

def fmt_example(r: dict, rank: int) -> str:
    agree_str = "YES ✓ (all codes matched)" if r["all_labels_agree"] else "NO ✗ (at least one code differed)"
    lines = [
        SEP,
        f"  Rank #{rank}",
        f"  item_id:            {r['item_id']}  [{r['dataset']}]",
        f"  reasoning_disagree: {r['reasoning_disagree']:.4f}",
        f"  len_A (Qwen):       {r['len_A']} words",
        f"  len_B (Llama):      {r['len_B']} words   |diff| = {r['len_diff']}",
        f"  Label agreement:    {agree_str}",
        "",
        f"  UTTERANCE TEXT:",
        f"    {r['text']}",
        "",
        f"  AGENT A LABELS (Qwen):   {r['agentA_labels']}",
        f"  AGENT B LABELS (Llama):  {r['agentB_labels']}",
        "",
        f"  AGENT A RATIONALE (Qwen):  [{r['len_A']} words]",
    ]
    for line in r["agentA_rationale"].split(". "):
        lines.append(f"    {line.strip()}")
    lines += [
        "",
        f"  AGENT B RATIONALE (Llama): [{r['len_B']} words]",
    ]
    for line in r["agentB_rationale"].split(". "):
        lines.append(f"    {line.strip()}")
    lines.append(SEP)
    return "\n".join(lines)


def print_section(title: str, subset, lines_out: list) -> None:
    header = "\n" + "=" * 76 + f"\n{title}\n" + "=" * 76
    print(header)
    lines_out.append(header)
    for rank, (_, r) in enumerate(subset.iterrows(), 1):
        block = fmt_example(r.to_dict(), rank)
        print(block)
        lines_out.append(block)


# ── select example sets ───────────────────────────────────────────────────────

# Set 1: high reasoning_disagree + all labels agreed
set1 = (
    feat_df[feat_df["all_labels_agree"]]
    .nlargest(5, "reasoning_disagree")
    .reset_index(drop=True)
)

# Set 2: lowest reasoning_disagree (most similar rationales)
set2 = feat_df.nsmallest(5, "reasoning_disagree").reset_index(drop=True)

# Set 3: high reasoning_disagree + large len_diff
# define "high" as top quartile of reasoning_disagree
thresh = feat_df["reasoning_disagree"].quantile(0.75)
set3 = (
    feat_df[feat_df["reasoning_disagree"] >= thresh]
    .nlargest(5, "len_diff")
    .reset_index(drop=True)
)

# ── output ────────────────────────────────────────────────────────────────────

output_lines = []

intro = (
    "reasoning_examples.py\n"
    "Illustrative examples showing reasoning_disagree reflects stylistic/length\n"
    "differences rather than genuine coding disagreement.\n"
    f"Source: rationale_cache.csv ({len(cache_rows)} utterances)\n"
)
print(intro)
output_lines.append(intro)

print_section(
    "SET 1 — HIGH reasoning_disagree  +  ALL LABELS AGREED\n"
    "(Both agents gave identical codes but explanations look far apart — stylistic noise)",
    set1,
    output_lines,
)

print_section(
    "SET 2 — LOWEST reasoning_disagree  (most similar rationales)\n"
    "(Contrast: agents agree both in labels AND in language)",
    set2,
    output_lines,
)

print_section(
    "SET 3 — HIGH reasoning_disagree  +  LARGE LENGTH DIFFERENCE\n"
    "(Length gap likely driving the apparent semantic distance)",
    set3,
    output_lines,
)

# ── save ──────────────────────────────────────────────────────────────────────

with OUT_TXT.open("w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print(f"\nSaved to {OUT_TXT.name}")
