#!/usr/bin/env python3
"""
significance.py — Bootstrap + permutation tests on a completed experiment run.

Usage:
  python significance.py --condition label
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"
RESULTS_ROOT = PKG / "results"

TUT_CODES = [
    "Greeting",
    "Instruction",
    "Guiding feedback",
    "Aligning to prior knowledge",
    "Understanding/Engagement-Tutor",
    "Encouragement",
]
TRANSCRIPT_FILES = [
    "First Author Copy GPT-Then-Human - Transcript B.csv",
    "First Author Copy GPT-Then-Human - transcript C.csv",
]
FOCUS = "Understanding/Engagement-Tutor"
N_BOOT = 2000
N_PERM = 2000
SEED = 42


def parse01(value) -> int:
    s = str(value or "").strip().casefold()
    return 1 if s in ("1", "1.0", "yes", "true") else 0


def human_label_for_code(header: list[str], values: list[str], code: str) -> int:
    target = code.strip().casefold()
    matched = [
        parse01(v)
        for n, v in zip(header, values)
        if (n or "").strip().casefold() == target
    ]
    return max(matched) if matched else 0


def load_human_gold() -> dict[str, dict[str, int]]:
    gold: dict[str, dict[str, int]] = {}
    idx = 0
    for fname in TRANSCRIPT_FILES:
        path = DATA / fname
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            si = ti = None
            for i, name in enumerate(header):
                n = (name or "").strip().casefold()
                if si is None and n == "speaker_type":
                    si = i
                if ti is None and n == "text":
                    ti = i
            for values in reader:
                if len(values) < len(header):
                    values += [""] * (len(header) - len(values))
                if (values[si] or "").strip().casefold() != "tutor":
                    continue
                if not (values[ti] or "").strip():
                    continue
                idx += 1
                iid = f"tuto_{idx:04d}"
                gold[iid] = {
                    c: human_label_for_code(header, values, c) for c in TUT_CODES
                }
    return gold


def load_scored(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["item_id"]] = {c: int(row.get(c, 0) or 0) for c in TUT_CODES}
    return out


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    p1 = (tp + fp) / n
    p0 = (tn + fn) / n
    q1 = (tp + fn) / n
    q0 = (tn + fp) / n
    pe = p1 * q1 + p0 * q0
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def kappa_from_vecs(pred: np.ndarray, gold: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (gold == 1)))
    fp = int(np.sum((pred == 1) & (gold == 0)))
    fn = int(np.sum((pred == 0) & (gold == 1)))
    tn = int(np.sum((pred == 0) & (gold == 0)))
    return cohen_kappa(tp, fp, fn, tn)


def build_arrays(item_ids, gold_map, pred0, pred_r, codes=None):
    codes = codes or TUT_CODES
    p0, pr, g = [], [], []
    for iid in item_ids:
        for code in codes:
            p0.append(pred0[iid][code])
            pr.append(pred_r[iid][code])
            g.append(gold_map[iid][code])
    return np.array(p0, int), np.array(pr, int), np.array(g, int)


def bootstrap_kappa_diff(item_ids, gold_map, pred0, pred_r, codes=None, rng=None):
    rng = rng or random.Random(SEED)
    codes = codes or TUT_CODES
    n = len(item_ids)
    obs_p0, obs_pr, obs_g = build_arrays(item_ids, gold_map, pred0, pred_r, codes)
    obs_diff = kappa_from_vecs(obs_pr, obs_g) - kappa_from_vecs(obs_p0, obs_g)
    diffs = []
    for _ in range(N_BOOT):
        boot_p0, boot_pr, boot_g = [], [], []
        for _ in range(n):
            ui = rng.randrange(n)
            iid = item_ids[ui]
            for code in codes:
                boot_p0.append(pred0[iid][code])
                boot_pr.append(pred_r[iid][code])
                boot_g.append(gold_map[iid][code])
        diffs.append(
            kappa_from_vecs(np.array(boot_pr, int), np.array(boot_g, int))
            - kappa_from_vecs(np.array(boot_p0, int), np.array(boot_g, int))
        )
    arr = np.array(diffs)
    ci_lo, ci_hi = np.percentile(arr, [2.5, 97.5])
    return {
        "obs_diff": obs_diff,
        "kappa_baseline": kappa_from_vecs(obs_p0, obs_g),
        "kappa_revised": kappa_from_vecs(obs_pr, obs_g),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "share_ge0": float(np.mean(arr >= 0)),
    }


def permutation_kappa_diff(item_ids, gold_map, pred0, pred_r, codes=None, rng=None):
    rng = rng or random.Random(SEED + 1)
    codes = codes or TUT_CODES
    p0, pr, g = build_arrays(item_ids, gold_map, pred0, pred_r, codes)
    obs_diff = kappa_from_vecs(pr, g) - kappa_from_vecs(p0, g)
    perm_diffs = []
    for _ in range(N_PERM):
        pp0, ppr = p0.copy(), pr.copy()
        for i in range(len(p0)):
            if rng.random() < 0.5:
                pp0[i], ppr[i] = ppr[i], pp0[i]
        perm_diffs.append(kappa_from_vecs(ppr, g) - kappa_from_vecs(pp0, g))
    p_two = float(np.mean(np.abs(np.array(perm_diffs)) >= abs(obs_diff)))
    return {"obs_diff": obs_diff, "p_two_sided": p_two}


def verdict(diff, ci_lo, ci_hi, p_val):
    if ci_hi < 0:
        return "revision reliably REDUCES agreement with human gold"
    if ci_lo > 0:
        return "revision reliably IMPROVES agreement with human gold"
    if p_val < 0.05 and diff < 0:
        return "revision significantly reduces agreement (permutation p<0.05)"
    if p_val < 0.05 and diff > 0:
        return "revision significantly improves agreement (permutation p<0.05)"
    return "no reliable difference between baseline and revised codebook"


def best_revision_round(results_csv: Path) -> tuple[int, float]:
    best_r, best_k = 0, -1.0
    with results_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = int(row["round"])
            if r == 0:
                continue
            k = float(row["pooled_kappa"])
            if k > best_k:
                best_k, best_r = k, r
    return best_r, best_k


def run_comparison(label, round_num, item_ids, gold_map, pred0, pred_r, lines, rng):
    lines += ["=" * 72, label, f"Round 0 vs round {round_num}", "=" * 72, ""]
    boot = bootstrap_kappa_diff(item_ids, gold_map, pred0, pred_r, TUT_CODES, rng)
    perm = permutation_kappa_diff(item_ids, gold_map, pred0, pred_r, TUT_CODES, rng)
    v = verdict(boot["obs_diff"], boot["ci_lo"], boot["ci_hi"], perm["p_two_sided"])
    lines += [
        "POOLED:",
        f"  Baseline kappa: {boot['kappa_baseline']:.4f}",
        f"  Revised kappa:  {boot['kappa_revised']:.4f}",
        f"  Observed diff (revised - baseline): {boot['obs_diff']:+.4f}",
        f"  Bootstrap 95% CI: [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]",
        f"  Share bootstrap diff >= 0: {boot['share_ge0']:.3f}",
        f"  Permutation p (two-sided): {perm['p_two_sided']:.4f}",
        f"  VERDICT: {v}",
        "",
    ]
    print(f"\n{label}")
    print(
        f"  diff={boot['obs_diff']:+.4f}  CI=[{boot['ci_lo']:+.4f},{boot['ci_hi']:+.4f}]  "
        f"p={perm['p_two_sided']:.4f}  -> {v}"
    )
    boot_ue = bootstrap_kappa_diff(
        item_ids, gold_map, pred0, pred_r, [FOCUS], rng
    )
    if boot_ue["ci_hi"] < 0:
        ue_v = "UE revision reliably reduces agreement"
    elif boot_ue["ci_lo"] > 0:
        ue_v = "UE revision reliably improves agreement"
    else:
        ue_v = "no reliable UE kappa difference"
    lines += [
        f"{FOCUS} (bootstrap):",
        f"  diff={boot_ue['obs_diff']:+.4f}  CI=[{boot_ue['ci_lo']:+.4f}, {boot_ue['ci_hi']:+.4f}]",
        f"  VERDICT: {ue_v}",
        "",
    ]
    print(f"  UE: diff={boot_ue['obs_diff']:+.4f}  -> {ue_v}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="")
    parser.add_argument(
        "--results-dir",
        default="",
        help="Folder with scored_round*.csv and results.csv (overrides --condition)",
    )
    args = parser.parse_args()
    if args.results_dir:
        in_dir = Path(args.results_dir)
        if not in_dir.is_absolute():
            in_dir = PKG / in_dir
        cond_label = in_dir.name
    elif args.condition:
        in_dir = RESULTS_ROOT / args.condition
        cond_label = args.condition
    else:
        print("FATAL: pass --condition or --results-dir")
        raise SystemExit(1)
    out_txt = in_dir / "significance_results.txt"

    if not (in_dir / "scored_round0_baseline.csv").exists():
        print(f"FATAL: no results in {in_dir}. Run the experiment first.")
        raise SystemExit(1)

    gold = load_human_gold()
    pred0 = load_scored(in_dir / "scored_round0_baseline.csv")
    item_ids = sorted(pred0.keys(), key=lambda x: int(x.split("_")[1]))
    best_r, best_k = best_revision_round(in_dir / "results.csv")
    preds = {0: pred0}
    for r in range(1, 6):
        preds[r] = load_scored(in_dir / f"scored_round{r}.csv")

    lines = [
        f"Significance — condition={cond_label}",
        f"Bootstrap={N_BOOT} Permutation={N_PERM}",
        f"Best post-revision round: {best_r} (kappa={best_k:.4f})",
        "",
    ]
    rng = random.Random(SEED)
    run_comparison(
        "(i) Round 0 vs Round 5", 5, item_ids, gold, pred0, preds[5], lines, rng
    )
    run_comparison(
        f"(ii) Round 0 vs Round {best_r} (best)",
        best_r,
        item_ids,
        gold,
        pred0,
        preds[best_r],
        lines,
        rng,
    )
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {out_txt}")


if __name__ == "__main__":
    main()
