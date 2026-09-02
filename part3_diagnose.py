#!/usr/bin/env python3
"""
part3_diagnose.py
Diagnose why the revised codebook hurt accuracy on Understanding/Engagement-Tutor.
Compares original vs revised (v2) agent-A labels against human gold.
No LLM calls; reuses existing CSV outputs.
"""

import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

CODE = "Understanding/Engagement-Tutor"

TUT_FILES = [
    "Data_2/First Author Copy GPT-Then-Human - Transcript B.csv",
    "Data_2/First Author Copy GPT-Then-Human - transcript C.csv",
]

ORIGINAL_DEFINITION = (
    "The tutor checks the student's understanding, usually by asking a question."
)

# ── load revised codebook ──────────────────────────────────────────────────────
rev_json_path = BASE / "revised_codebook_label_tutoring_round1_v2.json"
with rev_json_path.open(encoding="utf-8") as f:
    rev_cb = json.load(f)
REVISED_DEFINITION = rev_cb["codes"].get(CODE, "NOT FOUND")

# ── load gold labels from transcripts ─────────────────────────────────────────

def parse01(value) -> int:
    s = str(value or "").strip().casefold()
    return 1 if s in ("1", "1.0", "yes", "true") else 0

def human_label_for_code(header, values, code) -> int:
    target = code.strip().casefold()
    matched = [parse01(v) for n, v in zip(header, values)
               if (n or "").strip().casefold() == target]
    return max(matched) if matched else 0

gold_rows: list[dict] = []   # {text, gold}
for rel in TUT_FILES:
    path = BASE / rel
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        si = ti = None
        for i, name in enumerate(header):
            n = (name or "").strip().casefold()
            if si is None and n == "speaker_type": si = i
            if ti is None and n == "text":         ti = i
        for values in reader:
            if len(values) < len(header):
                values += [""] * (len(header) - len(values))
            if (values[si] or "").strip().casefold() != "tutor":
                continue
            text = (values[ti] or "").strip()
            if not text:
                continue
            gold_rows.append({
                "text": text,
                "gold": human_label_for_code(header, values, CODE),
            })

print(f"Gold rows loaded: {len(gold_rows)}")

# ── load agent-A predictions ──────────────────────────────────────────────────

def load_pred_csv(path: Path, code: str) -> list[int]:
    preds = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds.append(int(row.get(code, 0) or 0))
    return preds

orig_preds = load_pred_csv(BASE / "baseline_tutoring.csv", CODE)
rev_preds  = load_pred_csv(BASE / "recoded_label_tutoring_round1_v2.csv", CODE)
gold_vals  = [r["gold"] for r in gold_rows]
texts      = [r["text"] for r in gold_rows]

assert len(orig_preds) == len(gold_vals) == len(rev_preds), (
    f"Length mismatch: orig={len(orig_preds)}, rev={len(rev_preds)}, gold={len(gold_vals)}"
)
N = len(gold_vals)

# ── confusion matrices ────────────────────────────────────────────────────────

def confusion(preds, golds):
    tp = fp = fn = tn = 0
    for p, g in zip(preds, golds):
        if   p == 1 and g == 1: tp += 1
        elif p == 1 and g == 0: fp += 1
        elif p == 0 and g == 1: fn += 1
        else:                   tn += 1
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "acc": acc}

orig_conf = confusion(orig_preds, gold_vals)
rev_conf  = confusion(rev_preds,  gold_vals)

# ── new errors ────────────────────────────────────────────────────────────────

added_fp: list[str] = []   # human=0, rev=1, orig=0
added_fn: list[str] = []   # human=1, rev=0, orig=1

for text, g, o, r in zip(texts, gold_vals, orig_preds, rev_preds):
    if g == 0 and r == 1 and o == 0:
        added_fp.append(text)
    if g == 1 and r == 0 and o == 1:
        added_fn.append(text)

# ── output ────────────────────────────────────────────────────────────────────

SEP  = "─" * 72
SEP2 = "=" * 72

lines: list[str] = []

def emit(s: str = "") -> None:
    print(s)
    lines.append(s)

emit(SEP2)
emit(f"DIAGNOSIS: {CODE}")
emit(SEP2)

emit("\nORIGINAL DEFINITION:")
emit(f"  {ORIGINAL_DEFINITION}")
emit("\nREVISED (v2) DEFINITION:")
emit(f"  {REVISED_DEFINITION}")

emit(f"\n{SEP}")
emit("CONFUSION MATRICES")
emit(SEP)
emit(f"  {'':30s} {'Original':>10} {'Revised v2':>12}")
emit(f"  {'-'*54}")
for key in ("TP", "FP", "FN", "TN", "acc"):
    o_val = f"{orig_conf[key]:.4f}" if key == "acc" else str(orig_conf[key])
    r_val = f"{rev_conf[key]:.4f}"  if key == "acc" else str(rev_conf[key])
    delta = ""
    if key != "acc":
        d = rev_conf[key] - orig_conf[key]
        delta = f"  Δ{d:+d}"
    else:
        d = rev_conf["acc"] - orig_conf["acc"]
        delta = f"  Δ{d:+.4f}"
    emit(f"  {key:<30s} {o_val:>10} {r_val:>12}{delta}")

emit(f"\n{SEP}")
emit("NEW ERRORS INTRODUCED BY REVISION")
emit(SEP)

emit(f"\n(a) FALSE POSITIVES ADDED by revision  [human=0, orig=0, revised=1]")
emit(f"    Count: {len(added_fp)}")
emit(f"    (Qwen newly over-applies the code after revision)")
for i, t in enumerate(added_fp[:25], 1):
    emit(f"\n  [{i}] {t}")
if len(added_fp) > 25:
    emit(f"  ... and {len(added_fp)-25} more")

emit(f"\n(b) FALSE NEGATIVES ADDED by revision  [human=1, orig=1, revised=0]")
emit(f"    Count: {len(added_fn)}")
emit(f"    (Qwen newly under-applies the code after revision)")
for i, t in enumerate(added_fn[:25], 1):
    emit(f"\n  [{i}] {t}")
if len(added_fn) > 25:
    emit(f"  ... and {len(added_fn)-25} more")

emit(f"\n{SEP}")
emit("TALLY")
emit(SEP)
total_loss = (rev_conf["FP"] - orig_conf["FP"]) + (rev_conf["FN"] - orig_conf["FN"])
emit(
    f"  Accuracy loss on {CODE}: "
    f"{rev_conf['acc']-orig_conf['acc']:+.4f}  "
    f"({int((orig_conf['acc']-rev_conf['acc'])*N)} more errors in {N} utterances)"
)
emit(
    f"  Breakdown of NEW errors: "
    f"+{len(added_fp)} false positives (over-application)  vs  "
    f"+{len(added_fn)} false negatives (under-application)"
)
if len(added_fp) >= len(added_fn):
    emit("  → Primary driver: OVER-APPLICATION (revision broadened what triggers the code).")
else:
    emit("  → Primary driver: UNDER-APPLICATION (revision narrowed the code too aggressively).")

# ── save ──────────────────────────────────────────────────────────────────────
out_path = BASE / "part3_diagnose_UE.txt"
out_path.write_text("\n".join(lines), encoding="utf-8")
emit(f"\nSaved to {out_path.name}")
