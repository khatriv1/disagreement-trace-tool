#!/usr/bin/env python3
"""
Step 1 (tutoring half): per-code agent disagreement on tutor utterances,
using two different models (Qwen + Llama) across 8 tutoring codes.
"""

FILES = [
    "Data_2/First Author Copy GPT-Then-Human - Transcript B.csv",
    "Data_2/First Author Copy GPT-Then-Human - transcript C.csv",
]
MODEL_A = "qwen2.5:7b"
MODEL_B = "llama3.1:8b"
MAX_UTTERANCES = None  # None = all tutor rows (~337, ~50 min); set an int for a quick test

# CANDIDATE canonical 8 codes -- CONFIRM WITH CONRAD before trusting results
CODE_KEYS = [
    "Greeting",
    "Instruction",
    "Guiding feedback",
    "Aligning to prior knowledge",
    "Understanding/Engagement-Tutor",
    "Technical or Logistics",
    "Encouragement",
    "Time Management",
]

# fill in the per-code kappa for these 8 codes once Conrad provides them, then correlate.
HUMAN_KAPPA = {}

import csv
import random
import sys
from pathlib import Path

import numpy as np

random.seed(0)

from consensus_coding import extract_and_complete_code, generate

CODEBOOK = """
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

Technical or Logistics: Comments about technical setup or session logistics, not
  the math.

Encouragement: Affirmative statements praising the student's effort or performance.

Time Management: Comments about time remaining, pacing, or moving on.
""".strip()

CODE_EXAMPLE = str({k: 0 for k in CODE_KEYS})
CODEBOOK_TEMPLATE = {k: 0 for k in CODE_KEYS}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _parse01(value) -> int:
    if value is None:
        return 0
    s = str(value).strip().casefold()
    if s in ("1", "1.0", "true", "yes"):
        return 1
    return 0


def as01(value) -> int:
    return 1 if value == 1 or value == "1" else 0


def human_label_for_code(header: list[str], values: list[str], code: str) -> int:
    """MAX over all columns whose normalized name matches the canonical code."""
    target = code.strip().casefold()
    matched = []
    for name, val in zip(header, values):
        if (name or "").strip().casefold() == target:
            matched.append(_parse01(val))
    return max(matched) if matched else 0


def load_tutor_rows(base: Path) -> list[dict]:
    """Load both CSVs with csv.reader so duplicate column names are preserved."""
    records: list[dict] = []
    for rel in FILES:
        path = base / rel
        print(f"  Reading {path.name} ...")
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            # map speaker_type / text column indices (first match)
            speaker_i = text_i = None
            for i, name in enumerate(header):
                n = (name or "").strip().casefold()
                if speaker_i is None and n == "speaker_type":
                    speaker_i = i
                if text_i is None and n == "text":
                    text_i = i
            if speaker_i is None or text_i is None:
                raise SystemExit(f"Missing speaker_type/text columns in {path}")

            for values in reader:
                # pad short rows; missing columns -> 0
                if len(values) < len(header):
                    values = values + [""] * (len(header) - len(values))
                speaker = (values[speaker_i] or "").strip()
                if speaker.casefold() != "tutor":
                    continue
                text = (values[text_i] or "").strip()
                if not text:
                    continue
                labels = {
                    code: human_label_for_code(header, values, code)
                    for code in CODE_KEYS
                }
                records.append({"text": text, "labels": labels, "source": path.name})
    return records


def make_system_prompt(name: str, personality: str) -> str:
    return (
        f"You are {name}, a {personality} qualitative coding agent.\n"
        "Use the codebook below to analyze one utterance from tutoring dialogue.\n"
        f"{CODEBOOK}\n"
        "Always write your reasoning FIRST as 1 to 2 plain sentences explaining which "
        "codes apply and why; do not omit this reasoning.\n"
        "ONLY AFTER your reasoning, on the very last line, output a Python dictionary "
        f"with all eight keys {CODE_KEYS}, values 0 or 1 only, no markdown, no code "
        f"fences, and no text after the dictionary. Example:\n{CODE_EXAMPLE}"
    )


def code_utterance(name: str, personality: str, text: str, model: str) -> str:
    messages = [
        {"role": "system", "content": make_system_prompt(name, personality)},
        {"role": "user", "content": f"Utterance to code:\n{text}"},
    ]
    return generate(model, messages, {"temperature": 0.4})


def main() -> None:
    base = Path(__file__).resolve().parent
    log_path = base / "step1_tutoring_run.log"
    log_f = log_path.open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    print("Loading tutoring transcripts ...")
    pool = load_tutor_rows(base)
    print(f"  Tutor utterances loaded: {len(pool)}")

    human_ones = {k: sum(r["labels"][k] for r in pool) for k in CODE_KEYS}
    print("  Human 1-counts per canonical code (reference):")
    for k in CODE_KEYS:
        print(f"    {k}: {human_ones[k]}")
    print(
        "WARNING: column 'Tutor questioning' exists separately and may be the same "
        "as 'Understanding/Engagement-Tutor' — confirm with Conrad."
    )

    if MAX_UTTERANCES is not None:
        if len(pool) < MAX_UTTERANCES:
            raise SystemExit(
                f"Only {len(pool)} utterances available; need {MAX_UTTERANCES}."
            )
        sample = random.sample(pool, MAX_UTTERANCES)
        print(f"  Sampled {MAX_UTTERANCES} utterances (seed=0).")
    else:
        sample = pool
        print(f"  Using all {len(sample)} tutor utterances.")

    n = len(sample)
    print(f"  Agent A model={MODEL_A}; Agent B model={MODEL_B}")
    print(f"  Running both agents on each (~{n * 2} LLM calls)...\n")

    disagree_counts = {k: 0 for k in CODE_KEYS}

    for idx, row in enumerate(sample, start=1):
        text = row["text"]
        print(
            f"[{idx}/{n}] coding utterance ({len(text.split())} words) "
            f"[A={MODEL_A}, B={MODEL_B}]..."
        )
        msg_a = code_utterance("Agent A", "bold and decisive", text, MODEL_A)
        msg_b = code_utterance(
            "Agent B", "cautious and conservative", text, MODEL_B
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

    disagreement_rate = {k: disagree_counts[k] / n for k in CODE_KEYS}

    # Human 1-counts over the *sample* used for disagreement (and full-pool for ref)
    sample_ones = {k: sum(r["labels"][k] for r in sample) for k in CODE_KEYS}

    print("\n" + "=" * 78)
    print("RESULTS  [tutoring, two-model: Qwen vs Llama]")
    print("=" * 78)
    print(
        f"{'code':<32} {'human_ones_count':>16} {'agent_disagreement_rate':>24}"
    )
    print("-" * 78)
    for k in CODE_KEYS:
        print(f"{k:<32} {sample_ones[k]:>16d} {disagreement_rate[k]:>24.4f}")

    out_csv = base / "step1_tutoring_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["code", "human_ones_count", "agent_disagreement_rate"],
        )
        writer.writeheader()
        for k in CODE_KEYS:
            writer.writerow(
                {
                    "code": k,
                    "human_ones_count": sample_ones[k],
                    "agent_disagreement_rate": disagreement_rate[k],
                }
            )
    print(f"\nSaved results table to {out_csv.name}")
    print(f"Logged run to {log_path.name}")

    if HUMAN_KAPPA:
        missing = [k for k in CODE_KEYS if k not in HUMAN_KAPPA]
        if missing:
            print(f"HUMAN_KAPPA missing keys (skipping correlation): {missing}")
        else:
            from scipy import stats

            kappas = np.array([HUMAN_KAPPA[k] for k in CODE_KEYS], dtype=float)
            rates = np.array([disagreement_rate[k] for k in CODE_KEYS], dtype=float)
            spearman = stats.spearmanr(kappas, rates)
            pearson = stats.pearsonr(kappas, rates)
            print("\n" + "=" * 78)
            print("CORRELATION (human_kappa vs agent_disagreement_rate)")
            print("=" * 78)
            print(
                f"  Spearman r = {spearman.correlation:.4f},  p = {spearman.pvalue:.4f}"
            )
            print(
                f"  Pearson  r = {pearson.statistic:.4f},  p = {pearson.pvalue:.4f}"
            )
            print(
                "\nNote: a NEGATIVE correlation (low kappa -> high disagreement) "
                "supports the hypothesis. With n=8 codes this is descriptive."
            )
    else:
        print(
            "\nHUMAN_KAPPA is empty — skipping kappa correlation "
            "(fill in once Conrad provides per-code kappa)."
        )


if __name__ == "__main__":
    main()
