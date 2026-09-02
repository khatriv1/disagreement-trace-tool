#!/usr/bin/env python3
"""
part3_prototype.py
Single-round prototype of the Part 3 codebook-revision loop.
Condition: LABEL  |  Dataset: TUTORING  |  Round: 1

Flow:
  STEP 0 – Baseline accuracy (agent A / Qwen, original codebook).
            Reuses baseline_tutoring.csv if it exists.
  STEP 1 – Propose a revised codebook (reviser = qwen2.5:7b) using the 202
            label-disagreement cases from part3_label_tutoring.csv.
            Saves: revised_codebook_label_tutoring_round1.json
  STEP 2 – Re-code all tutor utterances with agent A + revised codebook.
            Saves: recoded_label_tutoring_round1.csv
  STEP 3 – Compare baseline vs revised accuracy; print table + save summary.

Single-agent scoring (agent A only) for speed; two-agent consensus can be
layered in later.
"""

import ast
import csv
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ── config ────────────────────────────────────────────────────────────────────
MODEL_A   = "qwen2.5:7b"
REVISER   = "qwen2.5:7b"
TEMP      = 0.4
TUT_CODES = [
    "Greeting",
    "Instruction",
    "Guiding feedback",
    "Aligning to prior knowledge",
    "Understanding/Engagement-Tutor",
    "Encouragement",
]
TUT_FILES = [
    "Data_2/First Author Copy GPT-Then-Human - Transcript B.csv",
    "Data_2/First Author Copy GPT-Then-Human - transcript C.csv",
]

ORIGINAL_CODEBOOK: dict[str, str] = {
    "Greeting": (
        "A salutation or farewell between tutor and student."
    ),
    "Instruction": (
        "A specific instruction or direction the tutor gives about what to do."
    ),
    "Guiding feedback": (
        "Feedback on the student's work, or clarification/explanation of a concept; "
        "guiding the student through a problem."
    ),
    "Aligning to prior knowledge": (
        "The tutor points the student to a previously learned concept, often using "
        "'remember'."
    ),
    "Understanding/Engagement-Tutor": (
        "The tutor checks the student's understanding, usually by asking a question."
    ),
    "Encouragement": (
        "Affirmative statements praising the student's effort or performance."
    ),
}

# ── imports ───────────────────────────────────────────────────────────────────
from consensus_coding import extract_and_complete_code, generate


# ── helpers ───────────────────────────────────────────────────────────────────

def codebook_text(cb: dict[str, str]) -> str:
    lines = [
        "The codes below describe TUTOR moves in online tutoring dialogue.",
        "They are NOT mutually exclusive: an utterance can have several codes, or none.",
        "Mark 1 if the utterance fits the code, 0 otherwise.\n",
    ]
    for code, defn in cb.items():
        lines.append(f"{code}: {defn}\n")
    return "\n".join(lines).strip()


def make_system_prompt(cb: dict[str, str]) -> str:
    codes_str   = str(TUT_CODES)
    example_str = str({k: 0 for k in TUT_CODES})
    return (
        "You are Agent A, a careful qualitative coding agent.\n"
        f"{codebook_text(cb)}\n\n"
        "Always write your reasoning FIRST (1–2 plain sentences). "
        "ONLY AFTER your reasoning, on the very last line, output a Python dictionary "
        f"with exactly these keys {codes_str}, values 0 or 1 only, no markdown, no "
        f"code fences. Example:\n{example_str}"
    )


def code_utterance(text: str, cb: dict[str, str]) -> dict[str, int]:
    msgs = [
        {"role": "system", "content": make_system_prompt(cb)},
        {"role": "user",   "content": f"Utterance to code:\n{text}"},
    ]
    raw = generate(MODEL_A, msgs, {"temperature": TEMP})
    return extract_and_complete_code(raw, {k: 0 for k in TUT_CODES})


def parse01(value) -> int:
    s = str(value or "").strip().casefold()
    return 1 if s in ("1", "1.0", "yes", "true") else 0


def human_label_for_code(header: list[str], values: list[str], code: str) -> int:
    target = code.strip().casefold()
    matched = [parse01(v) for n, v in zip(header, values)
               if (n or "").strip().casefold() == target]
    return max(matched) if matched else 0


def load_tutor_rows(base: Path) -> list[dict]:
    """Load both transcript CSVs; return list of {text, labels:{code:0/1}}."""
    records: list[dict] = []
    for rel in TUT_FILES:
        path = base / rel
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader   = csv.reader(f)
            header   = next(reader)
            si = ti  = None
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
                labels = {c: human_label_for_code(header, values, c)
                          for c in TUT_CODES}
                records.append({"text": text, "labels": labels})
    return records


def accuracy_report(
    records: list[dict],
    pred_key: str,
) -> tuple[float, dict[str, float]]:
    """Overall and per-code accuracy; pred_key is the key in each record."""
    total_correct = 0
    total_n       = 0
    per_code: dict[str, float] = {}
    for code in TUT_CODES:
        correct = sum(
            1 for r in records
            if r[pred_key].get(code, 0) == r["labels"].get(code, 0)
        )
        per_code[code] = correct / len(records) if records else 0.0
        total_correct += correct
        total_n       += len(records)
    overall = total_correct / total_n if total_n else 0.0
    return overall, per_code


def word_count(cb: dict[str, str]) -> int:
    return sum(len(d.split()) for d in cb.values())


def extract_json(text: str):
    """Extract first {...} block and parse as JSON.
    Handles common LLM quirks: markdown fences, invalid \\' escapes inside
    double-quoted strings (replace with plain apostrophe), and control chars."""
    text = text.strip()
    # strip markdown fences
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    m = re.search(r"\{[\s\S]*\}", text)
    candidate = m.group() if m else text
    # Fix invalid \' inside double-quoted JSON strings (LLM artefact)
    candidate = candidate.replace("\\'", "'")
    # Remove literal control characters that break json.loads
    candidate = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', candidate)
    return json.loads(candidate)


# ── STEP 0 — Baseline ─────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 0 — BASELINE ACCURACY (agent A, original codebook)")
print("=" * 70)

baseline_csv = BASE / "baseline_tutoring.csv"
tutor_rows   = load_tutor_rows(BASE)
print(f"  Loaded {len(tutor_rows)} tutor utterances")

if baseline_csv.exists():
    print(f"  Reusing existing {baseline_csv.name} ...")
    with baseline_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_bl = list(reader)
    for r_bl, r_gold in zip(rows_bl, tutor_rows):
        pred = {c: int(r_bl[c]) for c in TUT_CODES if c in r_bl}
        r_gold["pred_orig"] = pred
else:
    print(f"  Running agent A on {len(tutor_rows)} utterances (original codebook) ...")
    for i, r in enumerate(tutor_rows, 1):
        if i % 50 == 0 or i == 1:
            print(f"  ... {i}/{len(tutor_rows)}")
        r["pred_orig"] = code_utterance(r["text"], ORIGINAL_CODEBOOK)
    # save
    with baseline_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text"] + TUT_CODES)
        writer.writeheader()
        for r in tutor_rows:
            row = {"text": r["text"]}
            row.update({c: r["pred_orig"].get(c, 0) for c in TUT_CODES})
            writer.writerow(row)
    print(f"  Saved {baseline_csv.name}")

baseline_overall, baseline_per_code = accuracy_report(tutor_rows, "pred_orig")
print(f"\n  Baseline overall accuracy: {baseline_overall:.4f}  ({baseline_overall*100:.1f}%)")
print("  Per-code:")
for code in TUT_CODES:
    print(f"    {code:40s}: {baseline_per_code[code]:.4f}")

# ── STEP 1 — Propose revised codebook (LABEL condition) ───────────────────────

print("\n" + "=" * 70)
print("STEP 1 — PROPOSE REVISED CODEBOOK  (LABEL condition, reviser = qwen2.5:7b)")
print("=" * 70)

label_csv  = BASE / "part3_label_tutoring.csv"
rev_json   = BASE / "revised_codebook_label_tutoring_round1_v3.json"

with label_csv.open(newline="", encoding="utf-8") as f:
    label_cases = list(csv.DictReader(f))
print(f"  Loaded {len(label_cases)} label-disagreement cases")

# ── (A) Correct-examples block: up to 5 positive+correct utterances per code ──
# "Correct" = agent A's original prediction matched human gold AND code was 1.
import random as _random
_random.seed(42)

correct_examples: dict[str, list[str]] = {c: [] for c in TUT_CODES}
for r in tutor_rows:
    for code in TUT_CODES:
        if (r["labels"].get(code, 0) == 1
                and r["pred_orig"].get(code, 0) == 1):
            correct_examples[code].append(r["text"])

# Sample up to 5 per code (shuffle for variety)
correct_block_lines = []
for code in TUT_CODES:
    pool = correct_examples[code]
    _random.shuffle(pool)
    sample = pool[:5]
    if sample:
        correct_block_lines.append(f"\n{code} ({len(pool)} correct positives; showing up to 5):")
        for t in sample:
            correct_block_lines.append(f'  - "{t}"')
    else:
        correct_block_lines.append(f"\n{code}: (no correct positive examples found)")
correct_block = "\n".join(correct_block_lines)

# ── (B) Disagreement cases block ──────────────────────────────────────────────
cases_block_lines = []
for i, c in enumerate(label_cases, 1):
    cl = json.loads(c["consensus_labels_json"])
    agreed   = {k: v for k, v in cl.items() if v != "DISAGREE"}
    disagree = [k for k, v in cl.items() if v == "DISAGREE"]
    cases_block_lines.append(
        f"[{i}] \"{c['text']}\"\n"
        f"    Agreed codes: {agreed}\n"
        f"    Split (disagreed): {disagree}"
    )
cases_block = "\n".join(cases_block_lines)

cb_block = "\n".join(f"  {k}: {v}" for k, v in ORIGINAL_CODEBOOK.items())

print(f"  Correct-example counts per code (shown to reviser):")
for code in TUT_CODES:
    print(f"    {code}: {len(correct_examples[code])} positive+correct utterances")

reviser_prompt = f"""Refine the coding codebook so coders apply each code more consistently on the DISAGREEMENT cases shown below, WITHOUT losing the CORRECT cases shown.

CURRENT CODEBOOK:
{cb_block}

--- (A) Cases each code SHOULD capture — do NOT lose these ---
These are utterances where coders already agreed and were correct.
Every revised definition MUST still clearly cover these examples for that code.
{correct_block}

--- (B) Cases where coders disagreed ---
{cases_block}

RULES:
- Every revised definition MUST still clearly cover the (A) "SHOULD capture" examples for that code. Do not narrow a code so it would exclude its own correct examples.
- Add include/exclude boundaries ONLY to resolve the confusion shown in the (B) disagreement cases; do not remove the code's core meaning.
- Preserve the core of each code (e.g. if a code is about the tutor asking a question to check understanding, keep that; if it is about affirmations, keep that).
- Add 1-2 short positive example phrases per code, drawn from the (A) SHOULD-capture set.
- Keep the same 6 code names; do not merge or delete codes.

Return ONLY valid JSON (no markdown, no code fences, no extra text):
{{
  "codes": {{
    "Greeting": "<revised definition>",
    "Instruction": "<revised definition>",
    "Guiding feedback": "<revised definition>",
    "Aligning to prior knowledge": "<revised definition>",
    "Understanding/Engagement-Tutor": "<revised definition>",
    "Encouragement": "<revised definition>"
  }},
  "changes": ["<short note per code>"]
}}"""

print("  Calling reviser ...")
msgs = [{"role": "user", "content": reviser_prompt}]
raw_revision = generate(REVISER, msgs, {"temperature": TEMP})
print("  Raw reviser output (first 500 chars):")
print("    " + raw_revision[:500].replace("\n", "\n    "))

revised_cb: dict[str, str] = {}
changes: list[str] = []

def try_parse(text: str):
    obj = extract_json(text)
    if "codes" not in obj or not isinstance(obj["codes"], dict):
        raise ValueError("missing 'codes' key")
    for k in TUT_CODES:
        if k not in obj["codes"]:
            raise KeyError(f"missing code '{k}'")
    return obj

parsed = None
try:
    parsed = try_parse(raw_revision)
    print("  Parsed successfully on first attempt.")
except Exception as e:
    print(f"  Parse failed ({e}). Re-prompting once ...")
    retry_msgs = msgs + [
        {"role": "assistant", "content": raw_revision},
        {"role": "user", "content":
         "Your response was not valid JSON or was missing required keys. "
         "Return ONLY the JSON object described above, nothing else."},
    ]
    raw_revision2 = generate(REVISER, retry_msgs, {"temperature": TEMP})
    print("  Re-prompt raw output (first 500 chars):")
    print("    " + raw_revision2[:500].replace("\n", "\n    "))
    try:
        parsed = try_parse(raw_revision2)
        print("  Parsed successfully on retry.")
    except Exception as e2:
        print(f"\nFATAL: Could not parse revised codebook after retry ({e2}).")
        print("Full second raw output:")
        print(raw_revision2)
        sys.exit(1)

revised_cb = {k: parsed["codes"][k] for k in TUT_CODES}
changes    = parsed.get("changes", [])

rev_json.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
print(f"\n  Saved: {rev_json.name}")
print(f"\n  Changes reported by reviser ({len(changes)}):")
for i, ch in enumerate(changes, 1):
    print(f"    {i}. {ch}")

print(f"\n  Original codebook word count: {word_count(ORIGINAL_CODEBOOK)}")
print(f"  Revised codebook word count:  {word_count(revised_cb)}")

# ── STEP 2 — Re-code with revised codebook ────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 2 — RE-CODE with revised codebook (agent A)")
print("=" * 70)

recoded_csv = BASE / "recoded_label_tutoring_round1_v3.csv"
print(f"  Running agent A on {len(tutor_rows)} utterances (revised codebook) ...")
for i, r in enumerate(tutor_rows, 1):
    if i % 50 == 0 or i == 1:
        print(f"  ... {i}/{len(tutor_rows)}")
    r["pred_rev"] = code_utterance(r["text"], revised_cb)

with recoded_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text"] + TUT_CODES)
    writer.writeheader()
    for r in tutor_rows:
        row = {"text": r["text"]}
        row.update({c: r["pred_rev"].get(c, 0) for c in TUT_CODES})
        writer.writerow(row)
print(f"  Saved: {recoded_csv.name}")

revised_overall, revised_per_code = accuracy_report(tutor_rows, "pred_rev")

# ── STEP 3 — Report ───────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 3 — RESULTS: ORIGINAL vs REVISED")
print("=" * 70)

col_w = 42
header_line = f"  {'Code':<{col_w}} {'Original':>10} {'Revised':>10} {'Delta':>8}"
print(header_line)
print("  " + "-" * (col_w + 32))
for code in TUT_CODES:
    orig = baseline_per_code[code]
    rev  = revised_per_code[code]
    delta = rev - orig
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
    print(f"  {code:<{col_w}} {orig:>10.4f} {rev:>10.4f} {delta:>+7.4f} {arrow}")
print("  " + "-" * (col_w + 32))
overall_delta = revised_overall - baseline_overall
arrow_ov = "▲" if overall_delta > 0 else ("▼" if overall_delta < 0 else "=")
print(f"  {'OVERALL':<{col_w}} {baseline_overall:>10.4f} {revised_overall:>10.4f} "
      f"{overall_delta:>+7.4f} {arrow_ov}")

print(f"\n  Codebook word count: {word_count(ORIGINAL_CODEBOOK)} → {word_count(revised_cb)} "
      f"(delta {word_count(revised_cb)-word_count(ORIGINAL_CODEBOOK):+d})")
print(f"  Changes made by reviser: {len(changes)}")

print("\n  Per-code definition details (word count + examples added):")
print(f"  {'Code':<42} {'Words':>6}  {'Has example?':>12}")
print("  " + "-" * 64)
for code in TUT_CODES:
    defn = revised_cb.get(code, "")
    wc   = len(defn.split())
    has_ex = "YES" if "e.g." in defn.lower() or "example" in defn.lower() else "NO"
    print(f"  {code:<42} {wc:>6}  {has_ex:>12}")

# ── Confusion matrix for Understanding/Engagement-Tutor ──────────────────────
FOCUS = "Understanding/Engagement-Tutor"
print(f"\n  Confusion matrix — '{FOCUS}':")
print(f"  {'':6} {'Original':>10} {'Revised v3':>12} {'Delta':>8}")
print("  " + "-" * 40)

def conf_counts(pred_key: str) -> dict:
    tp = fp = fn = tn = 0
    for r in tutor_rows:
        p = r.get(pred_key, {}).get(FOCUS, 0)
        g = r["labels"].get(FOCUS, 0)
        if   p == 1 and g == 1: tp += 1
        elif p == 1 and g == 0: fp += 1
        elif p == 0 and g == 1: fn += 1
        else:                   tn += 1
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

orig_c = conf_counts("pred_orig")
rev_c  = conf_counts("pred_rev")
for key in ("TP", "FP", "FN", "TN"):
    d = rev_c[key] - orig_c[key]
    print(f"  {key:<6} {orig_c[key]:>10} {rev_c[key]:>12} {d:>+8}")
tp_preserved = rev_c["TP"]
tp_total     = orig_c["TP"]
print(f"\n  True-positive preservation: {tp_preserved}/{tp_total} "
      f"({100*tp_preserved/max(tp_total,1):.1f}% of original TPs retained)")

# ── save summary ──────────────────────────────────────────────────────────────

summary_path = BASE / "part3_prototype_summary_v3.txt"
lines = [
    "Part 3 Prototype — Round 1 Summary",
    "Condition: LABEL | Dataset: TUTORING",
    f"N cases shown to reviser: {len(label_cases)}",
    f"N tutor utterances scored: {len(tutor_rows)}",
    "",
    f"{'Code':<42} {'Original':>10} {'Revised':>10} {'Delta':>8}",
    "-" * 72,
]
for code in TUT_CODES:
    orig  = baseline_per_code[code]
    rev   = revised_per_code[code]
    delta = rev - orig
    lines.append(f"{code:<42} {orig:>10.4f} {rev:>10.4f} {delta:>+8.4f}")
lines += [
    "-" * 72,
    f"{'OVERALL':<42} {baseline_overall:>10.4f} {revised_overall:>10.4f} "
    f"{overall_delta:>+8.4f}",
    "",
    f"Codebook word count: {word_count(ORIGINAL_CODEBOOK)} -> {word_count(revised_cb)}",
    f"Changes by reviser: {len(changes)}",
    "",
    "Changes:",
]
for i, ch in enumerate(changes, 1):
    lines.append(f"  {i}. {ch}")

lines += ["", "Per-code definition details (revised):",
          f"  {'Code':<42} {'Words':>6}  {'Has example?':>12}"]
for code in TUT_CODES:
    defn  = revised_cb.get(code, "")
    wc    = len(defn.split())
    has_ex = "YES" if "e.g." in defn.lower() or "example" in defn.lower() else "NO"
    lines.append(f"  {code:<42} {wc:>6}  {has_ex:>12}")

lines += [
    "", f"Confusion matrix — {FOCUS}:",
    f"  {'':6} {'Original':>10} {'Revised v3':>12} {'Delta':>8}",
    "  " + "-" * 40,
]
for key in ("TP", "FP", "FN", "TN"):
    d = rev_c[key] - orig_c[key]
    lines.append(f"  {key:<6} {orig_c[key]:>10} {rev_c[key]:>12} {d:>+8}")
lines.append(
    f"  TP preserved: {tp_preserved}/{tp_total} "
    f"({100*tp_preserved/max(tp_total,1):.1f}%)"
)

lines += ["", "Revised codebook definitions:"]
for code, defn in revised_cb.items():
    lines.append(f"\n{code}:\n  {defn}")

summary_path.write_text("\n".join(lines), encoding="utf-8")
print(f"\n  Saved: {summary_path.name}")
print("\nDone.")
