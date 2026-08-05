#!/usr/bin/env python3
"""
Conrad's reasoning-disagreement analytic on LAK24 utterances (agent vs agent).
Two coding agents independently code each utterance; rank by low cosine similarity
between their reasoning texts (highest ambiguity first).

Lowest-similarity items are the highest-ambiguity utterances for human review;
this is the data behind Panel 1 of the mockup (reasoning disagreement on the same item).
Requires Ollama with qwen2.5:7b (ollama serve).
"""

import csv
import random
from pathlib import Path

import numpy as np

from consensus_coding import cosine, embed, extract_and_complete_code, generate, justification_of

CSV_PATH = "lak24-coded-utterances.csv"
MODEL = "qwen2.5:7b"
SAMPLE_SIZE = 40

random.seed(0)

CODEBOOK = """
The codes below describe dialog moves in tutoring problem-solving. They are NOT
mutually exclusive: an utterance can have several codes, or none. Mark 1 if the
utterance fits the code, 0 otherwise.

process (Processing information): reading or re-reading the question, hints,
  or system feedback; or repeating/synthesizing information the system
  provided. Understanding information before acting.
  Example: "Let's figure out how many hydrogen atoms are in a millimole of
  water. Our result should have three significant figures."

plan (Planning): verbalizing a CONCEPTUAL plan or goal for solving the
  problem (must involve conceptual knowledge). Announcing a concrete next
  action is NOT planning.
  Example: "The goal of the result is the number of hydrogen atoms, right?"

act (Enacting): verbalizing a concrete, specific action just carried out or
  about to be carried out (entering a number, requesting a hint, "I'm going
  to fill in grams"). Includes procedural next-step announcements.
  Example: "6.1023 divided by 2. 3.0115."

wrong (Realizing errors): verbalizing that something is wrong, recognizing a
  mistake in the answer or process, with or without system feedback.
  Example: "It's incorrect. What's happened? The thousand is in the wrong spot."
"""

CODE_KEYS = ["process", "plan", "act", "wrong"]
CODE_EXAMPLE = str({k: 0 for k in CODE_KEYS})
SRL_KEYS = {"process": "", "plan": "", "act": "", "wrong": ""}

SNIPPET_UTTERANCE = 120


def make_system_prompt(name: str, personality: str) -> str:
    return (
        f"You are {name}, a {personality} qualitative coding agent.\n"
        "Use the codebook below to analyze one utterance from tutoring dialogue.\n"
        f"{CODEBOOK.strip()}\n"
        "Always write your reasoning FIRST as 1 to 2 plain sentences explaining which "
        "codes apply and why; do not omit this reasoning.\n"
        "ONLY AFTER your reasoning, on the very last line, output a Python dictionary "
        f"with all four keys {CODE_KEYS}, values 0 or 1 only, no markdown, no code "
        f"fences, and no text after the dictionary. Example:\n{CODE_EXAMPLE}"
    )


def code_utterance(name: str, personality: str, text: str, model=MODEL) -> str:
    messages = [
        {"role": "system", "content": make_system_prompt(name, personality)},
        {"role": "user", "content": f"Utterance to code:\n{text}"},
    ]
    return generate(model, messages, {"temperature": 0.4})


def load_utterances(path: Path) -> list[str]:
    texts = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("remove_flag", "").strip().lower() == "yes":
                continue
            text = row.get("utterance_combined", "").strip()
            if text:
                texts.append(text)
    return texts


def truncate(text: str, n: int) -> str:
    t = text.replace("\n", " ")
    return t if len(t) <= n else t[: n - 3] + "..."


def main() -> None:
    csv_file = Path(__file__).resolve().parent / CSV_PATH
    pool = load_utterances(csv_file)
    if len(pool) < SAMPLE_SIZE:
        raise SystemExit(f"Only {len(pool)} utterances available; need {SAMPLE_SIZE}.")
    sample = random.sample(pool, SAMPLE_SIZE)

    records: list[dict] = []

    for idx, text in enumerate(sample, start=1):
        print(f"\n--- [{idx}/{SAMPLE_SIZE}] utterance: {truncate(text, SNIPPET_UTTERANCE)}")
        msg_a = code_utterance("Agent A", "bold and decisive", text)
        msg_b = code_utterance("Agent B", "cautious and conservative", text)
        codes_a = extract_and_complete_code(msg_a, SRL_KEYS)
        codes_b = extract_and_complete_code(msg_b, SRL_KEYS)
        active_a = sorted(k for k, v in codes_a.items() if v == 1)
        active_b = sorted(k for k, v in codes_b.items() if v == 1)
        label_disagreement = active_a != active_b
        n_words = len(text.split())
        just_a = justification_of(msg_a)
        just_b = justification_of(msg_b)
        u, v = embed([just_a, just_b])
        sim = cosine(u, v)
        print(f"  Agent A reasoning: {just_a}")
        print(f"  Agent B reasoning: {just_b}")
        print(f"  cosine similarity: {sim:.3f}")
        print(f"  activeA: {active_a}  activeB: {active_b}  label_disagreement: {label_disagreement}")
        records.append(
            {
                "text": text,
                "similarity": sim,
                "label_disagreement": label_disagreement,
                "activeA": active_a,
                "activeB": active_b,
                "n_words": n_words,
            }
        )

    by_cosine = sorted(records, key=lambda r: r["similarity"])

    print("\n\n=== GENUINE LABEL DISAGREEMENTS (agents assigned different codes) ===")
    print("-" * 78)
    genuine = [r for r in by_cosine if r["label_disagreement"]]
    if not genuine:
        print("  (none in this sample)")
    for r in genuine:
        print(
            f"  {r['similarity']:.3f}  A:{r['activeA']} vs B:{r['activeB']}  "
            f"{truncate(r['text'], 90)}"
        )

    print("\n\n=== SUBSTANTIVE UTTERANCES (>= 6 words, lowest cosine first) ===")
    print("-" * 78)
    substantive = [r for r in by_cosine if r["n_words"] >= 6][:15]
    for r in substantive:
        print(f"  {r['similarity']:.3f}  {truncate(r['text'], SNIPPET_UTTERANCE)}")

    print("\n\nHIGHEST REASONING DISAGREEMENT (lowest cosine first)")
    print("-" * 78)
    for r in by_cosine:
        print(f"  {r['similarity']:.3f}  {truncate(r['text'], SNIPPET_UTTERANCE)}")


if __name__ == "__main__":
    main()
