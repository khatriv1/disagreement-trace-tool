#!/usr/bin/env python3
"""
part3_setup.py
Build MATCHED case sets for the three Part 3 revision conditions, per Conrad's design.
Each condition gets exactly N_matched utterances per dataset.

Matching strategy (per dataset):
  N_matched = number of ambiguity-eligible utterances (those with ≥1 positive code,
  so per-utterance ambiguity is defined). We match DOWN to this count so all three
  conditions are equal-sized. This is the binding constraint because ambiguity can
  only be computed for utterances that have at least one positive label.

  AMBIGUITY set  = all N_matched ambiguity-eligible utterances, ranked descending by
                   per-utterance ambiguity (mean of per-code ambiguity from
                   datapoint_table). Text ONLY — do NOT include coding (per Conrad).

  LABEL set      = from the label-disagreement utterances (≥1 code split between
                   agents), take the top N_matched ranked by:
                     1. n_codes_disagree (descending) — most-disagreeing first
                     2. utt_ambiguity (descending) — tie-break by ambiguity
                     3. item_id (ascending) — deterministic final tie-break
                   CSV includes: utterance text + consensus_labels_json (agreed codes
                   shown as 0/1; codes that split are marked "DISAGREE").

  REASONING set  = the SAME N_matched utterances chosen for the LABEL set, but CSV
                   carries agentA_rationale + agentB_rationale instead of labels.

Inputs:  datapoint_table.csv, rationale_cache.csv
Outputs: part3_{label,reasoning,ambiguity}_{chemistry,tutoring}.csv  (6 CSVs, overwrite)
"""

import csv
import json
from pathlib import Path

import pandas as pd

BASE          = Path(__file__).resolve().parent
DATAPOINT_CSV = BASE / "datapoint_table.csv"
CACHE_CSV     = BASE / "rationale_cache.csv"

DATASETS = ["chemistry", "tutoring"]

# ── load datapoint table ──────────────────────────────────────────────────────

print("Loading datapoint_table.csv ...")
dp = pd.read_csv(DATAPOINT_CSV)
print(f"  {len(dp)} rows  {dp['item_id'].nunique()} utterances")

# ── load rationale cache ──────────────────────────────────────────────────────

print("Loading rationale_cache.csv ...")
cache: dict[str, dict] = {}
with CACHE_CSV.open(newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ds  = r["dataset"]
        idx = int(r["idx"])
        iid = f"{ds[:4]}_{idx:04d}"
        cache[iid] = {
            "text":             r["text"],
            "agentA_rationale": r["agentA_rationale"],
            "agentB_rationale": r["agentB_rationale"],
        }
print(f"  {len(cache)} utterances in cache")

# ── per-utterance aggregation ─────────────────────────────────────────────────

utt_agg = (
    dp.groupby(["item_id", "dataset"])
    .agg(
        any_label_disagree =("label_disagree", "max"),
        n_codes_disagree   =("label_disagree", "sum"),   # how many codes split
        utt_ambiguity      =("ambiguity",      "mean"),  # NaN-aware mean
    )
    .reset_index()
)
print(f"\nUtterance-level aggregation: {len(utt_agg)} rows")

# consensus label builder (per utterance, uses datapoint rows)
def consensus_labels_for(item_id: str, ds_dp: pd.DataFrame) -> dict:
    rows = ds_dp[ds_dp["item_id"] == item_id]
    result = {}
    for _, row in rows.iterrows():
        if row["label_disagree"] == 0:
            result[row["code"]] = int(row["agentA_label"])
        else:
            result[row["code"]] = "DISAGREE"
    return result


# ── per-dataset case sets ─────────────────────────────────────────────────────

for dataset in DATASETS:
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset.upper()}")
    print("=" * 70)

    sub = utt_agg[utt_agg["dataset"] == dataset].copy()

    # ── N_matched: number of ambiguity-eligible utterances ────────────────────
    amb_eligible = sub.dropna(subset=["utt_ambiguity"]).copy()
    N_matched = len(amb_eligible)
    print(f"  Ambiguity-eligible utterances (N_matched): {N_matched}")

    # ── AMBIGUITY set: all eligible, ranked descending ─────────────────────────
    amb_set = amb_eligible.sort_values("utt_ambiguity", ascending=False).copy()
    # (all N_matched eligible rows, already at correct size)
    assert len(amb_set) == N_matched

    # ── LABEL set: top-N_matched most-disagreeing label-disagree utterances ────
    label_pool = sub[sub["any_label_disagree"] == 1].copy()
    total_label_disagree = len(label_pool)
    print(f"  Total label-disagree utterances: {total_label_disagree}")

    label_set = (
        label_pool
        .sort_values(
            ["n_codes_disagree", "utt_ambiguity", "item_id"],
            ascending=[False, False, True],
        )
        .head(N_matched)
        .copy()
    )
    print(f"  Label set size (top-{N_matched} by n_codes_disagree): {len(label_set)}")

    # ── REASONING set: same utterances as label set ────────────────────────────
    # (same N_matched rows, different columns saved)
    assert len(label_set) == N_matched

    # ── overlap ───────────────────────────────────────────────────────────────
    label_ids = set(label_set["item_id"])
    amb_ids   = set(amb_set["item_id"])
    overlap   = label_ids & amb_ids
    print(
        f"  OVERLAP (label ∩ ambiguity): {len(overlap)} / {N_matched} "
        f"({100*len(overlap)/N_matched:.1f}%)"
    )
    print(
        f"\n  Set sizes (must all equal {N_matched}):\n"
        f"    AMBIGUITY  = {len(amb_set)}\n"
        f"    LABEL      = {len(label_set)}\n"
        f"    REASONING  = {len(label_set)}"
    )

    # ── datapoint rows for this dataset (for consensus labels) ────────────────
    ds_dp = dp[dp["dataset"] == dataset][
        ["item_id", "code", "agentA_label", "agentB_label", "label_disagree"]
    ].copy()

    # ── save AMBIGUITY CSV (text only) ────────────────────────────────────────
    amb_csv = BASE / f"part3_ambiguity_{dataset}.csv"
    with amb_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "text", "utt_ambiguity"])
        for _, row in amb_set.iterrows():
            iid = row["item_id"]
            w.writerow([iid, cache.get(iid, {}).get("text", ""),
                        f"{row['utt_ambiguity']:.4f}"])
    print(f"\n  Saved: {amb_csv.name}  ({len(amb_set)} rows)")

    # ── save LABEL CSV ────────────────────────────────────────────────────────
    lbl_csv = BASE / f"part3_label_{dataset}.csv"
    with lbl_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "text", "n_codes_disagree", "consensus_labels_json"])
        for _, row in label_set.iterrows():
            iid = row["item_id"]
            cl  = consensus_labels_for(iid, ds_dp)
            w.writerow([iid,
                        cache.get(iid, {}).get("text", ""),
                        int(row["n_codes_disagree"]),
                        json.dumps(cl)])
    print(f"  Saved: {lbl_csv.name}  ({len(label_set)} rows)")

    # ── save REASONING CSV ────────────────────────────────────────────────────
    rea_csv = BASE / f"part3_reasoning_{dataset}.csv"
    with rea_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "text", "agentA_rationale", "agentB_rationale"])
        for _, row in label_set.iterrows():
            iid = row["item_id"]
            c   = cache.get(iid, {})
            w.writerow([iid,
                        c.get("text", ""),
                        c.get("agentA_rationale", ""),
                        c.get("agentB_rationale", "")])
    print(f"  Saved: {rea_csv.name}  ({len(label_set)} rows)")

    # ── overlap examples ──────────────────────────────────────────────────────
    if overlap:
        print(f"\n  First 3 overlapping utterances (in BOTH label and ambiguity sets):")
        for iid in sorted(overlap)[:3]:
            amb_val = amb_set.loc[amb_set["item_id"] == iid, "utt_ambiguity"].values
            print(
                f"    {iid}  ambiguity={amb_val[0]:.4f}"
                if len(amb_val) else f"    {iid}"
            )
            print(f"      {cache.get(iid,{}).get('text','')[:80]}")

print("\nDone.")
