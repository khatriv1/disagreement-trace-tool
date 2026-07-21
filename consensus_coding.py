# # LEGEND:
# #   [REAL]      = genuine logic, correct math, or real source material
# #                 (e.g. Conrad's actual codebook / agent design)
# #   [SYNTHETIC] = fabricated placeholder data, or a fake stand-in used only
# #                 while running in "mock" mode
# # NOTE: BACKEND and EMBED_BACKEND are both "mock", so the SYNTHETIC paths are
# #       the ones currently running.

# from __future__ import annotations

# import ast
# import hashlib
# import importlib
# import json
# import math
# import re
# from statistics import mean

# # [REAL] Backend switches: which LLM and embedder to use. Both "mock" now, activating the fake model and fake embedder.
# # PROGRESS: Swappable LLM backend. BACKEND selects mock, ollama, or openai; changing it switches the model everywhere.
# BACKEND = "ollama"
# # PROGRESS: Swappable embedding backend. EMBED_BACKEND selects mock word-counter or real sentence-transformers.
# EMBED_BACKEND = "sbert"
# # Ollama model name; must match exactly what was pulled with `ollama pull`.
# MODEL = "qwen2.5:7b"
# _SBERT_MODEL = None

# # [REAL] Conrad Borchers' actual qualitative coding scheme from llm-ta-consensus (8 codes; example quotes trimmed).
# # PROGRESS: Codebook taken directly from the repo. Conrad's real coding scheme; DATA below is the placeholder part.
# CODEBOOK = """Greeting
# The initial interaction between the tutor and student, often at the beginning or end of the session. Anytime a salutation or farewell is exchanged.

# Instruction
# Specific instructions or directions posed by the tutor throughout the lesson.

# Guiding feedback
# Guided practice through a math problem by the tutor. Feedback on the student's work or response and clarification or explanation of a concept or instruction.

# Aligning to prior knowledge
# Instances when the tutor brings attention to a previous math concept that a student knows or has discussed in a session. Tutor aligns the student to previous knowledge using the word 'remember'.

# Understanding/Engagement-Tutor
# The tutor presents checks for understanding as questions to students.

# Technical or Logistics
# Tutor comments related to the technical aspects or logistics of the lesson.

# Encouragement
# Affirmative statements from the tutor recognizing student's efforts, answers, or performance.

# Time Management
# Statements regarding the duration left, the need to move on, or how much has been covered.
# """

# # [REAL] Conrad's code labels; one-line descriptions are shortened paraphrases, not his exact wording.
# # PROGRESS: Codebook taken directly from the repo. Label-to-description map used by agents and extract_and_complete_code().
# CODEBOOK_DICT = {
#     "Greeting": "Salutation or farewell.",
#     "Instruction": "Direct instruction or command.",
#     "Guiding feedback": "Tutor feedback on student work.",
#     "Aligning to prior knowledge": "Reference to a prior concept via 'remember'.",
#     "Understanding/Engagement-Tutor": "Tutor questions checking understanding.",
#     "Technical or Logistics": "Comments about tech/logistics.",
#     "Encouragement": "Praise or affirmation.",
#     "Time Management": "Comments about pacing or remaining time.",
# }

# # [REAL] Formats the all-zero output dict template shown to agents in prompts.
# CODE_EXAMPLE = str({k: 0 for k in CODEBOOK_DICT})

# # [SYNTHETIC] Placeholder items — text, human_code, and human_justification are all fabricated, not real transcripts.
# # PROGRESS: Validated on a small synthetic dataset. Eight tutoring lines with placeholder human_code and human_justification.
# DATA = [
#     {
#         "text": "tutor: Hello, how are you today?",
#         "human_code": "Greeting",
#         "human_justification": "Salutation at the start of the session — clearly a greeting.",
#     },
#     {
#         "text": "tutor: Go ahead and fill out the table on the worksheet.",
#         "human_code": "Instruction",
#         "human_justification": "Direct command telling the student what action to perform.",
#     },
#     {
#         "text": "tutor: Not quite. I'm not sure why you have these X's in the equation.",
#         "human_code": "Guiding feedback",
#         "human_justification": "Tutor evaluates the student's work and points to the specific error.",
#     },
#     {
#         "text": "tutor: Remember what we said about factoring last time?",
#         "human_code": "Aligning to prior knowledge",
#         "human_justification": "The word remember explicitly invokes a prior session's concept, anchoring the student to past knowledge rather than asking a content question.",
#     },
#     {
#         "text": "tutor: Why do you think we might multiply both sides by two?",
#         "human_code": "Understanding/Engagement-Tutor",
#         "human_justification": "Open-ended check-for-understanding question probing the student's reasoning.",
#     },
#     {
#         "text": "tutor: Your camera is looking at the ceiling again.",
#         "human_code": "Technical or Logistics",
#         "human_justification": "Comment about the video setup, not the math content.",
#     },
#     {
#         "text": "tutor: Perfect, you nailed it!",
#         "human_code": "Encouragement",
#         "human_justification": "Affirmative praise acknowledging the student's correct work.",
#     },
#     {
#         "text": "tutor: We have about five minutes left, so let's wrap up.",
#         "human_code": "Time Management",
#         "human_justification": "Statement about pacing and remaining session duration.",
#     },
# ]


# # [SYNTHETIC] Fake LLM stand-in: invented keyword rules and rationale strings. Real model only when BACKEND is ollama/openai.
# def _heuristic_label(text):
#     t = text.lower()
#     if "remember" in t and "?" in t:
#         return (
#             "Understanding/Engagement-Tutor",
#             "This is a tutor question checking for understanding, framed as a question.",
#         )
#     if any(g in t for g in ["hello", "bye", "see you", "cheers", "good morning"]):
#         return (
#             "Greeting",
#             "This appears to be a greeting or salutation marking the start or end of the session.",
#         )
#     if "remember" in t:
#         return (
#             "Aligning to prior knowledge",
#             "Tutor uses the word remember to align the student to prior knowledge from an earlier session.",
#         )
#     if any(p in t for p in ["fill out", "go ahead", "factor these", "we are going to"]):
#         return (
#             "Instruction",
#             "This is a direct instruction telling the student what to do.",
#         )
#     if any(p in t for p in ["not quite", "not sure why", "look for"]):
#         return (
#             "Guiding feedback",
#             "Guiding feedback — the tutor evaluates the student's answer and points out the issue.",
#         )
#     if "?" in t and any(w in t for w in ["why", "how", "what", "do you"]):
#         return (
#             "Understanding/Engagement-Tutor",
#             "Tutor question checking student understanding and reasoning.",
#         )
#     if any(w in t for w in ["camera", "mute", "connection", "hear me"]):
#         return (
#             "Technical or Logistics",
#             "A logistical or technical comment about the video or audio setup.",
#         )
#     if any(w in t for w in ["perfect", "great", "good job", "nailed"]):
#         return (
#             "Encouragement",
#             "Positive encouragement praising the student's work.",
#         )
#     if any(w in t for w in ["minutes", "wrap up", "almost", "halfway"]):
#         return (
#             "Time Management",
#             "A time management statement about the session pacing.",
#         )
#     return ("Instruction", "Falling back to instruction as a default.")


# # [REAL] Mock-backend plumbing: builds the {code dict} string for a chosen label.
# def _code_dict_str(label):
#     d = {k: 0 for k in CODEBOOK_DICT}
#     if label in d:
#         d[label] = 1
#     return str(d)


# # [REAL] Mock-backend plumbing: extracts the utterance from a formatted agent prompt.
# def _extract_text_to_code(user_content):
#     if "Text:\n" not in user_content:
#         return user_content
#     body = user_content.split("Text:\n", 1)[1]
#     for sep in ["\n\nAgent 1 coded:", "\n###\n", "\n\nAgent 2 coded:"]:
#         if sep in body:
#             body = body.split(sep, 1)[0]
#     return body.strip()


# # [REAL] Mock-backend plumbing: assembles a fake agent response (rationale + code dict).
# def _mock_generate(messages, options):
#     user = messages[-1]["content"]
#     text = _extract_text_to_code(user)
#     label, rationale = _heuristic_label(text)
#     return f"{rationale} {_code_dict_str(label)}"


# # [REAL] Single LLM entry point. ollama/openai branches are real; mock branch routes to the fake (BACKEND selects).
# # PROGRESS: Swappable LLM backend. generate() is the single LLM entry point; every agent calls it via chat().
# def generate(model, messages, options=None):
#     options = options or {}
#     if BACKEND == "mock":
#         return _mock_generate(messages, options)
#     if BACKEND == "ollama":
#         ollama = importlib.import_module("ollama")
#         resp = ollama.chat(model=model, messages=messages, options=options)
#         return resp["message"]["content"]
#     if BACKEND == "openai":
#         openai = importlib.import_module("openai")
#         client = openai.OpenAI()
#         resp = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=options.get("temperature", 0.4),
#         )
#         return resp.choices[0].message.content
#     raise ValueError(f"Unknown BACKEND: {BACKEND}")


# # [SYNTHETIC] Fake embedder: hashes word counts into a vector (word overlap, not meaning). Replaced when EMBED_BACKEND="sbert".
# _EMBED_DIM = 256
# _TOKEN_RE = re.compile(r"[A-Za-z]+")


# def _stable_hash(s):
#     return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


# def _mock_embed_one(text):
#     vec = [0.0] * _EMBED_DIM
#     for tok in _TOKEN_RE.findall(text.lower()):
#         idx = _stable_hash(tok) % _EMBED_DIM
#         vec[idx] += 1.0
#     n = math.sqrt(sum(v * v for v in vec))
#     if n > 0.0:
#         vec = [v / n for v in vec]
#     return vec


# # [REAL] Single embedding entry point. sbert branch is real sentence-transformers; mock branch uses the fake above.
# # PROGRESS: Swappable embedding backend. embed() is the single embedding entry point; routes via EMBED_BACKEND.
# # PROGRESS: Disagreement trace. embed() turns justification text into vectors for similarity scoring.
# def embed(texts):
#     if EMBED_BACKEND == "mock":
#         return [_mock_embed_one(t) for t in texts]
#     if EMBED_BACKEND == "sbert":
#         global _SBERT_MODEL
#         if _SBERT_MODEL is None:
#             from sentence_transformers import SentenceTransformer
#             _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
#         return [list(map(float, v)) for v in _SBERT_MODEL.encode(texts)]
#     raise ValueError(f"Unknown EMBED_BACKEND: {EMBED_BACKEND}")


# # [REAL] Genuine cosine-similarity math between two vectors. No fakery.
# # PROGRESS: Disagreement trace. cosine() computes similarity between two embedded justification vectors.
# def cosine(u, v):
#     dot = sum(a * b for a, b in zip(u, v))
#     nu = math.sqrt(sum(a * a for a in u))
#     nv = math.sqrt(sum(b * b for b in v))
#     if nu == 0.0 or nv == 0.0:
#         return 0.0
#     return dot / (nu * nv)


# _CODE_DICT_RE = re.compile(r"\{[^}]+\}")


# # [REAL] Parses agent output into a full code dict, faithful to Conrad's repo.
# def extract_and_complete_code(text_output, codebook_dict, fallback_label="missing"):
#     labels = list(codebook_dict.keys())
#     match = _CODE_DICT_RE.search(text_output)
#     if not match:
#         return {l: fallback_label for l in labels}
#     try:
#         extracted = ast.literal_eval(match.group())
#         return {l: extracted.get(l, 0) for l in labels}
#     except (ValueError, SyntaxError):
#         return {l: fallback_label for l in labels}


# # [REAL] Strips the trailing {code dict} off an agent message, leaving only the reasoning text.
# # PROGRESS: Disagreement trace. justification_of() pulls the reasoning text out of an agent message.
# def justification_of(agent_message):
#     text = _CODE_DICT_RE.sub("", agent_message).strip()
#     text = re.sub(r"^[A-Z][A-Za-z]+:\s*", "", text)            # drop "Name: " prefix
#     text = re.sub(r"(?i)^the previous turn said:\s*", "", text).strip()
#     text = re.sub(r"(?i)^another coder.*?:\s*", "", text).strip()
#     return text


# # [REAL] Conrad's agent architecture rebuilt: prompts, single -> debate -> consensus flow. Calls generate() (real or fake per BACKEND).
# # PROGRESS: Single agent baseline, two agent debate, consensus agent. Parent class: name, personality, model; make_system_prompt() and chat().
# class BaseCodingAgent:
#     def __init__(self, name, personality, model=MODEL, temperature=0.4, top_k=5):
#         self.name = name
#         self.personality = personality
#         self.model = model
#         self.options = {"temperature": temperature, "top_k": top_k}

#     def make_system_prompt(self, role):
#         return (
#             f"You are {self.name}, a {self.personality} qualitative coding agent.\n"
#             f"Your role: {role}\n"
#             "Use the provided codebook definitions to analyze qualitative text data.\n"
#             "Always write your reasoning FIRST as 1 to 2 plain sentences explaining why you chose "
#             "the code; do not omit this reasoning.\n"
#             "Always speak from your perspective; do not simulate the other agent.\n"
#             "When another coder's reasoning is provided, weigh it but write your own "
#             "reasoning in your own words; do not quote or repeat the framing text.\n"
#             "ONLY AFTER your reasoning sentences, on the very last line, output a Python dictionary "
#             "with every codebook key present, values 0 or 1, no markdown, no code fences, and no text "
#             f"after the dictionary. The reasoning must come before the dictionary. Example:\n{CODE_EXAMPLE}"
#         )

#     def chat(self, role_description, codebook, text, previous_turn=None):
#         messages = [
#             {"role": "system", "content": self.make_system_prompt(role_description)},
#             {"role": "user", "content": f"Codebook:\n{codebook}\n\nText:\n{text}"},
#         ]
#         if previous_turn:
#             messages[1]["content"] += (
#                 "\n\nAnother coder reviewed the same text and reasoned as follows:\n"
#                 f"{previous_turn}\n\n"
#                 "Take this into account, then give your own reasoning in your own "
#                 "words. Do not repeat or quote this framing."
#             )
#         return generate(self.model, messages, self.options)


# # PROGRESS: Single-agent baseline. assign_code() labels the line once, alone.
# class SingleAgentCoding(BaseCodingAgent):
#     def assign_code(self, codebook, text):
#         return self.chat("Assign codes to the text based on the codebook.", codebook, text)


# # PROGRESS: Two-agent debate. discuss() runs rounds; each agent sees previous_turn and checks whether they agree.
# class DualAgentDiscussion:
#     def __init__(self, agent1, agent2):
#         self.agent1 = agent1
#         self.agent2 = agent2

#     def discuss(self, codebook, text, rounds=1):
#         history = []
#         retries = []
#         msg1 = msg2 = None
#         for i in range(rounds + 1):
#             if i == 0:
#                 msg1 = self.agent1.chat("Establish an initial code for the best fitting code.", codebook, text)
#                 msg2 = self.agent2.chat("Establish an initial code for the best fitting code.", codebook, text)
#             else:
#                 msg1 = self.agent1.chat("Engage in collaborative discussion about the best fitting code.", codebook, text, msg2)
#                 msg2 = self.agent2.chat("Engage in collaborative discussion about the best fitting code.", codebook, text, msg1)
#             code1 = extract_and_complete_code(msg1, CODEBOOK_DICT)
#             code2 = extract_and_complete_code(msg2, CODEBOOK_DICT)
#             if any(v == "missing" for v in code1.values()):
#                 retries.append(f"A1R{i+1}")
#                 msg1 = self.agent1.chat("Re-establish the code.", codebook, text)
#                 code1 = extract_and_complete_code(msg1, CODEBOOK_DICT)
#             if any(v == "missing" for v in code2.values()):
#                 retries.append(f"A2R{i+1}")
#                 msg2 = self.agent2.chat("Re-establish the code.", codebook, text)
#                 code2 = extract_and_complete_code(msg2, CODEBOOK_DICT)
#             history.append({"role": "assistant", "content": f"{self.agent1.name}: {msg1}"})
#             history.append({"role": "assistant", "content": f"{self.agent2.name}: {msg2}"})
#             if code1 == code2:
#                 return history, i + 1, "##".join(retries)
#         return history, 0, "##".join(retries)


# # PROGRESS: Tie-breaker. resolve() is called only when the two agents fail to agree.
# class ConsensusAgent(BaseCodingAgent):
#     def resolve(self, codebook, text, code1, code2):
#         return self.chat(
#             "Resolve differences in assigned codes and propose a consensus label. Follow the JSON format at end of system prompt.",
#             codebook,
#             f"{text}\n\nAgent 1 coded: {code1}\nAgent 2 coded: {code2}",
#         )


# # [REAL] Builds the expected one-hot code dict for comparing human vs LLM labels.
# def _expected_code_dict(human_code):
#     return {k: (1 if k == human_code else 0) for k in CODEBOOK_DICT}


# def view_d_similar_items(top_k=2):
#     print("\n\nVIEW D  -  SIMILAR DATA POINTS (by cosine of the line text)")
#     print("-" * 78)
#     texts = [item["text"] for item in DATA]
#     vectors = embed(texts)  # embed once, reuse
#     for i, item in enumerate(DATA):
#         sims = []
#         for j, other in enumerate(DATA):
#             if i == j:
#                 continue
#             sims.append((cosine(vectors[i], vectors[j]), j))
#         sims.sort(reverse=True)
#         print(f"\n[{i}] {item['text']}  (code: {item['human_code']})")
#         for s, j in sims[:top_k]:
#             tr = DATA[j]["text"] if len(DATA[j]["text"]) <= 50 else DATA[j]["text"][:47] + "..."
#             print(f"     {s:>5.2f}  [{j}] {tr}  (code: {DATA[j]['human_code']})")


# def save_project(path="project.json"):
#     project = {"codebook": CODEBOOK_DICT, "data": DATA}
#     with open(path, "w") as f:
#         json.dump(project, f, indent=2)
#     print(f"\n[view c] Saved project to {path} "
#           f"({len(DATA)} items, {len(CODEBOOK_DICT)} codes).")


# def load_project(path="project.json"):
#     with open(path, "r") as f:
#         project = json.load(f)
#     print(f"[view c] Loaded project from {path} "
#           f"({len(project['data'])} items, {len(project['codebook'])} codes).")
#     return project


# # [REAL] Main pipeline. Currently runs on SYNTHETIC DATA with fake model and fake embedder (BACKEND/EMBED_BACKEND = mock).
# # PROGRESS: Rebuilt the pipeline into a clean standalone script that runs end to end.
# # PROGRESS: Orchestrator loops over DATA through single agent, debate, consensus, disagreement scoring, then ranking.
# def run():
#     # PROGRESS: Agents Sternberg, Marcuse, Butler, and Alex are created here; behavior is in the four agent classes below.
#     agent1 = BaseCodingAgent("Sternberg", "bold and dominant but elaborative")
#     agent2 = BaseCodingAgent("Marcuse", "bold and dominant but elaborative")
#     consensus_agent = ConsensusAgent("Butler", "balanced and reflective")
#     single_agent = SingleAgentCoding("Alex", "rigorous and empirical")

#     review_rows = []

#     for idx, item in enumerate(DATA):
#         text = item["text"]
#         print(f"\n=== Item {idx}: {text!r} ===")

#         _ = single_agent.assign_code(CODEBOOK, text)

#         discussion, consensus_reached, _retries = DualAgentDiscussion(agent1, agent2).discuss(CODEBOOK, text, rounds=1)
#         a1_msg = discussion[-2]["content"]
#         a2_msg = discussion[-1]["content"]

#         if consensus_reached == 0:
#             final_msg = consensus_agent.resolve(CODEBOOK, text, a1_msg, a2_msg)
#             final_code = extract_and_complete_code(final_msg, CODEBOOK_DICT)
#             final_source_msg = final_msg
#         else:
#             final_code = extract_and_complete_code(a2_msg, CODEBOOK_DICT)
#             final_source_msg = a2_msg

#         codes_match = final_code == _expected_code_dict(item["human_code"])
#         llm_label = next((k for k, v in final_code.items() if v == 1), None)
#         print(f"  human: {item['human_code']!r}  |  llm: {llm_label!r}  |  match: {codes_match}")

#         # PROGRESS: For each item, embed the human and LLM justification, then compute cosine similarity (disagreement score).
#         llm_just = justification_of(final_source_msg)
#         u, v = embed([item["human_justification"], llm_just])
#         sim = cosine(u, v)
#         print(f"  reasoning sim (human vs LLM): {sim:.2f}")

#         # PROGRESS: Rank items by lowest similarity so highest reasoning disagreement surfaces first. Collect rows here.
#         review_rows.append((idx, text, item["human_code"], llm_label, codes_match, sim))

#     print("\n\nDISAGREEMENT RANKING (lowest reasoning sim = review first)")
#     print("-" * 78)
#     print(f"{'idx':>3}  {'sim':>5}  {'codes_match':>11}  text")
#     print("-" * 78)
#     # PROGRESS: Rank items by lowest similarity so highest reasoning disagreement surfaces first. Sort ascending for printout.
#     for idx, text, construct, llm_label, codes_match, sim in sorted(review_rows, key=lambda r: r[5]):
#         truncated = text if len(text) <= 60 else text[:57] + "..."
#         print(f"{idx:>3}  {sim:>5.2f}  {str(codes_match):>11}  {truncated}")

#     # View a: summarize and sort constructs (codes) by disagreement.
#     # Lower average similarity means more human/LLM reasoning disagreement.
#     by_construct = {}
#     for idx, text, construct, llm_label, codes_match, sim in review_rows:
#         by_construct.setdefault(construct, []).append(sim)

#     print("\n\nVIEW A  -  CONSTRUCT SUMMARY (most disagreement first) The codes sorted by how much the human and model disagreed.")
#     print("-" * 78)
#     print(f"{'construct':<34}  {'items':>5}  {'avg sim':>7}")
#     print("-" * 78)
#     for construct, sims in sorted(by_construct.items(), key=lambda kv: mean(kv[1])):
#         print(f"{construct:<34}  {len(sims):>5}  {mean(sims):>7.2f}")

#     # View b: within each construct, surface its highest-disagreement items
#     # (lowest similarity) first, for refined annotation.
#     print("\n\nVIEW B  -  WITHIN-CONSTRUCT REVIEW (high disagreement first) Pick a code, see its most-disagreeing lines.")
#     print("-" * 78)
#     for construct, sims in sorted(by_construct.items(), key=lambda kv: mean(kv[1])):
#         print(f"\n[{construct}]  avg sim {mean(sims):.2f}")
#         construct_rows = [r for r in review_rows if r[2] == construct]
#         for idx, text, c, llm_label, codes_match, sim in sorted(construct_rows, key=lambda r: r[5]):
#             flag = "" if codes_match else "   <- label mismatch"
#             tr = text if len(text) <= 55 else text[:52] + "..."
#             print(f"   {sim:>5.2f}  {tr}{flag}")

#     view_d_similar_items(top_k=2)


# if __name__ == "__main__":
#     run()



# =============================================================================
# consensus_coding.py
# -----------------------------------------------------------------------------
# WHAT THIS IS
#   A "disagreement trace" tool built with Conrad Borchers. It labels text with
#   LLM agents (single -> debate -> consensus), then measures where the model's
#   reasoning diverges from a human's and ranks the biggest disagreements.
#
# CURRENT CONFIGURATION  (this is the part people get confused about):
#   BACKEND       = "ollama" -> REAL model: Qwen 2.5 7B, running locally
#   EMBED_BACKEND = "sbert"  -> REAL embeddings: sentence-transformers
#   So the model, the embeddings, and ALL the logic are REAL and running.
#   The ONLY fake thing in this file is DATA: 8 placeholder tutoring lines with
#   placeholder human labels and justifications. Everything else is real.
#
# AUTHORSHIP / PROVENANCE  (each section is tagged with one of these)
#   [SOURCE: repo]  = design/logic rebuilt from Conrad's llm-ta-consensus repo
#                     (the agents, the codebook, the output parser).
#   [NEW]           = original contribution built for this collaboration,
#                     NOT in his repo (the disagreement trace and views a-d).
#   [SYNTH]         = fabricated placeholder data (the 8 lines).
#   [FALLBACK]      = optional offline stand-ins, NOT active in this config.
#
#   This script was implemented with AI assistance (through Cursor). The tags
#   above describe design provenance: what is Conrad's architecture vs what is
#   the new work layered on top.
# =============================================================================

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import re
from statistics import mean

# [SOURCE: repo, extended] Backend switches. ACTIVE = ollama + sbert (both real).
BACKEND = "ollama"        # "ollama"=REAL model (active) | "openai"=REAL API | "mock"=offline fallback (inactive)
EMBED_BACKEND = "sbert"   # "sbert"=REAL embeddings (active) | "mock"=offline fallback (inactive)
MODEL = "qwen2.5:7b"      # Ollama model name; must match exactly what was pulled with `ollama pull`
_SBERT_MODEL = None       # cache so the embedding model loads once, not per item

# [SOURCE: repo] Conrad's actual coding scheme from llm-ta-consensus (8 codes;
# his example quotes trimmed for brevity). The labels/definitions are genuinely his.
CODEBOOK = """Greeting
The initial interaction between the tutor and student, often at the beginning or end of the session. Anytime a salutation or farewell is exchanged.

Instruction
Specific instructions or directions posed by the tutor throughout the lesson.

Guiding feedback
Guided practice through a math problem by the tutor. Feedback on the student's work or response and clarification or explanation of a concept or instruction.

Aligning to prior knowledge
Instances when the tutor brings attention to a previous math concept that a student knows or has discussed in a session. Tutor aligns the student to previous knowledge using the word 'remember'.

Understanding/Engagement-Tutor
The tutor presents checks for understanding as questions to students.

Technical or Logistics
Tutor comments related to the technical aspects or logistics of the lesson.

Encouragement
Affirmative statements from the tutor recognizing student's efforts, answers, or performance.

Time Management
Statements regarding the duration left, the need to move on, or how much has been covered.
"""

# [SOURCE: repo] Conrad's code labels. The one-line descriptions are shortened
# paraphrases of his fuller definitions (same codes, trimmed wording).
CODEBOOK_DICT = {
    "Greeting": "Salutation or farewell.",
    "Instruction": "Direct instruction or command.",
    "Guiding feedback": "Tutor feedback on student work.",
    "Aligning to prior knowledge": "Reference to a prior concept via 'remember'.",
    "Understanding/Engagement-Tutor": "Tutor questions checking understanding.",
    "Technical or Logistics": "Comments about tech/logistics.",
    "Encouragement": "Praise or affirmation.",
    "Time Management": "Comments about pacing or remaining time.",
}

# Formats the all-zero output dict template shown to agents in prompts.
CODE_EXAMPLE = str({k: 0 for k in CODEBOOK_DICT})

# [SYNTH] THE ONLY FAKE THING IN THIS FILE. 8 placeholder tutoring lines, one
# per code, each with a fabricated human label and justification. Replace this
# with real annotated data (with coder justifications) to get real results.
DATA = [
    {
        "text": "tutor: Hello, how are you today?",
        "human_code": "Greeting",
        "human_justification": "Salutation at the start of the session, clearly a greeting.",
    },
    {
        "text": "tutor: Go ahead and fill out the table on the worksheet.",
        "human_code": "Instruction",
        "human_justification": "Direct command telling the student what action to perform.",
    },
    {
        "text": "tutor: Not quite. I'm not sure why you have these X's in the equation.",
        "human_code": "Guiding feedback",
        "human_justification": "Tutor evaluates the student's work and points to the specific error.",
    },
    {
        "text": "tutor: Remember what we said about factoring last time?",
        "human_code": "Aligning to prior knowledge",
        "human_justification": "The word remember explicitly invokes a prior session's concept, anchoring the student to past knowledge rather than asking a content question.",
    },
    {
        "text": "tutor: Why do you think we might multiply both sides by two?",
        "human_code": "Understanding/Engagement-Tutor",
        "human_justification": "Open-ended check-for-understanding question probing the student's reasoning.",
    },
    {
        "text": "tutor: Your camera is looking at the ceiling again.",
        "human_code": "Technical or Logistics",
        "human_justification": "Comment about the video setup, not the math content.",
    },
    {
        "text": "tutor: Perfect, you nailed it!",
        "human_code": "Encouragement",
        "human_justification": "Affirmative praise acknowledging the student's correct work.",
    },
    {
        "text": "tutor: We have about five minutes left, so let's wrap up.",
        "human_code": "Time Management",
        "human_justification": "Statement about pacing and remaining session duration.",
    },
]


# [FALLBACK] Offline LLM stand-in (keyword rules). NOT used while BACKEND="ollama".
# Kept only so the script can run with no model if you ever set BACKEND="mock".
def _heuristic_label(text):
    t = text.lower()
    if "remember" in t and "?" in t:
        return ("Understanding/Engagement-Tutor",
                "This is a tutor question checking for understanding, framed as a question.")
    if any(g in t for g in ["hello", "bye", "see you", "cheers", "good morning"]):
        return ("Greeting",
                "This appears to be a greeting or salutation marking the start or end of the session.")
    if "remember" in t:
        return ("Aligning to prior knowledge",
                "Tutor uses the word remember to align the student to prior knowledge from an earlier session.")
    if any(p in t for p in ["fill out", "go ahead", "factor these", "we are going to"]):
        return ("Instruction",
                "This is a direct instruction telling the student what to do.")
    if any(p in t for p in ["not quite", "not sure why", "look for"]):
        return ("Guiding feedback",
                "Guiding feedback, the tutor evaluates the student's answer and points out the issue.")
    if "?" in t and any(w in t for w in ["why", "how", "what", "do you"]):
        return ("Understanding/Engagement-Tutor",
                "Tutor question checking student understanding and reasoning.")
    if any(w in t for w in ["camera", "mute", "connection", "hear me"]):
        return ("Technical or Logistics",
                "A logistical or technical comment about the video or audio setup.")
    if any(w in t for w in ["perfect", "great", "good job", "nailed"]):
        return ("Encouragement",
                "Positive encouragement praising the student's work.")
    if any(w in t for w in ["minutes", "wrap up", "almost", "halfway"]):
        return ("Time Management",
                "A time management statement about the session pacing.")
    return ("Instruction", "Falling back to instruction as a default.")


# [FALLBACK] Mock-backend plumbing (only runs when BACKEND="mock").
def _code_dict_str(label):
    d = {k: 0 for k in CODEBOOK_DICT}
    if label in d:
        d[label] = 1
    return str(d)


def _extract_text_to_code(user_content):
    if "Text:\n" not in user_content:
        return user_content
    body = user_content.split("Text:\n", 1)[1]
    for sep in ["\n\nAgent 1 coded:", "\n###\n", "\n\nAgent 2 coded:"]:
        if sep in body:
            body = body.split(sep, 1)[0]
    return body.strip()


def _mock_generate(messages, options):
    user = messages[-1]["content"]
    text = _extract_text_to_code(user)
    label, rationale = _heuristic_label(text)
    return f"{rationale} {_code_dict_str(label)}"


# [SOURCE: repo, extended] Single LLM entry point. The ollama branch (ACTIVE)
# and openai branch are real; the mock branch is the offline fallback. Every
# agent reaches the model through this one function, so BACKEND switches it all.
def generate(model, messages, options=None):
    options = options or {}
    if BACKEND == "mock":
        return _mock_generate(messages, options)
    if BACKEND == "ollama":                      # <-- ACTIVE: real Qwen 2.5 7B
        ollama = importlib.import_module("ollama")
        resp = ollama.chat(model=model, messages=messages, options=options)
        return resp["message"]["content"]
    if BACKEND == "openai":
        openai = importlib.import_module("openai")
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=options.get("temperature", 0.4))
        return resp.choices[0].message.content
    raise ValueError(f"Unknown BACKEND: {BACKEND}")


# [FALLBACK] Mock embedder (word-count vectors). NOT used while EMBED_BACKEND="sbert".
_EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _stable_hash(s):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def _mock_embed_one(text):
    vec = [0.0] * _EMBED_DIM
    for tok in _TOKEN_RE.findall(text.lower()):
        idx = _stable_hash(tok) % _EMBED_DIM
        vec[idx] += 1.0
    n = math.sqrt(sum(v * v for v in vec))
    if n > 0.0:
        vec = [v / n for v in vec]
    return vec


# [NEW] Single embedding entry point (not in Conrad's repo). The sbert branch
# (ACTIVE) loads a real sentence-transformers model once and reuses it. Turns a
# piece of text into a vector of numbers that captures its meaning.
def embed(texts):
    if EMBED_BACKEND == "mock":
        return [_mock_embed_one(t) for t in texts]
    if EMBED_BACKEND == "sbert":                 # <-- ACTIVE: real embeddings
        global _SBERT_MODEL
        if _SBERT_MODEL is None:
            from sentence_transformers import SentenceTransformer
            _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return [list(map(float, v)) for v in _SBERT_MODEL.encode(texts)]
    raise ValueError(f"Unknown EMBED_BACKEND: {EMBED_BACKEND}")


# [NEW] Genuine cosine-similarity math between two vectors (not in Conrad's repo).
# Returns ~1 when two vectors point the same way (similar meaning) and ~0 when
# they point in very different directions (different meaning).
def cosine(u, v):
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


_CODE_DICT_RE = re.compile(r"\{[^}]+\}")


# [SOURCE: repo] Parses an agent's text output into a full code dict.
def extract_and_complete_code(text_output, codebook_dict, fallback_label="missing"):
    labels = list(codebook_dict.keys())
    match = _CODE_DICT_RE.search(text_output)
    if not match:
        return {l: fallback_label for l in labels}
    try:
        extracted = ast.literal_eval(match.group())
        return {l: extracted.get(l, 0) for l in labels}
    except (ValueError, SyntaxError):
        return {l: fallback_label for l in labels}


# [NEW] Pulls the reasoning text out of an agent message (drops the trailing
# {code dict} and any leftover name/framing prefix) so only the justification is
# embedded. Part of the disagreement trace.
def justification_of(agent_message):
    text = _CODE_DICT_RE.sub("", agent_message).strip()
    text = re.sub(r"^[A-Z][A-Za-z]+:\s*", "", text)            # drop "Name: " prefix
    text = re.sub(r"(?i)^the previous turn said:\s*", "", text).strip()
    text = re.sub(r"(?i)^another coder.*?:\s*", "", text).strip()
    return text


# =============================================================================
# AGENTS  [SOURCE: repo]  -- Conrad's architecture, rebuilt here so it runs
# standalone and swaps to a real model. Single agent -> two-agent debate ->
# consensus tie-breaker. These produce the LLM's label and its reasoning.
# =============================================================================
class BaseCodingAgent:
    def __init__(self, name, personality, model=MODEL, temperature=0.4, top_k=5):
        self.name = name
        self.personality = personality
        self.model = model
        self.options = {"temperature": temperature, "top_k": top_k}

    def make_system_prompt(self, role):
        return (
            f"You are {self.name}, a {self.personality} qualitative coding agent.\n"
            f"Your role: {role}\n"
            "Use the provided codebook definitions to analyze qualitative text data.\n"
            "Always write your reasoning FIRST as 1 to 2 plain sentences explaining why you chose "
            "the code; do not omit this reasoning.\n"
            "Always speak from your perspective; do not simulate the other agent.\n"
            "When another coder's reasoning is provided, weigh it but write your own "
            "reasoning in your own words; do not quote or repeat the framing text.\n"
            "ONLY AFTER your reasoning sentences, on the very last line, output a Python dictionary "
            "with every codebook key present, values 0 or 1, no markdown, no code fences, and no text "
            f"after the dictionary. The reasoning must come before the dictionary. Example:\n{CODE_EXAMPLE}"
        )

    def chat(self, role_description, codebook, text, previous_turn=None):
        messages = [
            {"role": "system", "content": self.make_system_prompt(role_description)},
            {"role": "user", "content": f"Codebook:\n{codebook}\n\nText:\n{text}"},
        ]
        if previous_turn:
            messages[1]["content"] += (
                "\n\nAnother coder reviewed the same text and reasoned as follows:\n"
                f"{previous_turn}\n\n"
                "Take this into account, then give your own reasoning in your own "
                "words. Do not repeat or quote this framing."
            )
        return generate(self.model, messages, self.options)


class SingleAgentCoding(BaseCodingAgent):
    # Baseline: label the line once, alone.
    def assign_code(self, codebook, text):
        return self.chat("Assign codes to the text based on the codebook.", codebook, text)


class DualAgentDiscussion:
    # Two agents code the line, each sees the other, and it checks if they agree.
    def __init__(self, agent1, agent2):
        self.agent1 = agent1
        self.agent2 = agent2

    def discuss(self, codebook, text, rounds=1):
        history = []
        retries = []
        msg1 = msg2 = None
        for i in range(rounds + 1):
            if i == 0:
                msg1 = self.agent1.chat("Establish an initial code for the best fitting code.", codebook, text)
                msg2 = self.agent2.chat("Establish an initial code for the best fitting code.", codebook, text)
            else:
                msg1 = self.agent1.chat("Engage in collaborative discussion about the best fitting code.", codebook, text, msg2)
                msg2 = self.agent2.chat("Engage in collaborative discussion about the best fitting code.", codebook, text, msg1)
            code1 = extract_and_complete_code(msg1, CODEBOOK_DICT)
            code2 = extract_and_complete_code(msg2, CODEBOOK_DICT)
            if any(v == "missing" for v in code1.values()):
                retries.append(f"A1R{i+1}")
                msg1 = self.agent1.chat("Re-establish the code.", codebook, text)
                code1 = extract_and_complete_code(msg1, CODEBOOK_DICT)
            if any(v == "missing" for v in code2.values()):
                retries.append(f"A2R{i+1}")
                msg2 = self.agent2.chat("Re-establish the code.", codebook, text)
                code2 = extract_and_complete_code(msg2, CODEBOOK_DICT)
            history.append({"role": "assistant", "content": f"{self.agent1.name}: {msg1}"})
            history.append({"role": "assistant", "content": f"{self.agent2.name}: {msg2}"})
            if code1 == code2:
                return history, i + 1, "##".join(retries)
        return history, 0, "##".join(retries)


class ConsensusAgent(BaseCodingAgent):
    # Tie-breaker: only called when the two agents do not agree.
    def resolve(self, codebook, text, code1, code2):
        return self.chat(
            "Resolve differences in assigned codes and propose a consensus label. Follow the JSON format at end of system prompt.",
            codebook,
            f"{text}\n\nAgent 1 coded: {code1}\nAgent 2 coded: {code2}",
        )


# Builds the expected one-hot code dict for comparing human vs LLM labels.
def _expected_code_dict(human_code):
    return {k: (1 if k == human_code else 0) for k in CODEBOOK_DICT}


# =============================================================================
# VIEW D  [NEW]  -- Conrad's requested view (d)
#   WHAT: for each line, find the most similar OTHER lines, to help curate
#         examples that justify a codebook change.
#   HOW:  embed every line's text once, compute cosine similarity between each
#         pair, and print each line's top matches.
#   OPEN QUESTION for Conrad: compare the lines (as here), the items within a
#         single code, or the justifications? Easy to switch.
# =============================================================================
def view_d_similar_items(top_k=2):
    print("\n\nVIEW D  -  SIMILAR DATA POINTS (by cosine of the line text)")
    print("-" * 78)
    texts = [item["text"] for item in DATA]
    vectors = embed(texts)  # embed once, reuse
    for i, item in enumerate(DATA):
        sims = []
        for j, other in enumerate(DATA):
            if i == j:
                continue
            sims.append((cosine(vectors[i], vectors[j]), j))
        sims.sort(reverse=True)
        print(f"\n[{i}] {item['text']}  (code: {item['human_code']})")
        for s, j in sims[:top_k]:
            tr = DATA[j]["text"] if len(DATA[j]["text"]) <= 50 else DATA[j]["text"][:47] + "..."
            print(f"     {s:>5.2f}  [{j}] {tr}  (code: {DATA[j]['human_code']})")


# =============================================================================
# VIEW C  [NEW]  -- Conrad's requested view (c), basic version
#   WHAT: save/load a "project" (codebook + data) so a setup can be reused.
#   HOW:  write to / read from a JSON file on disk. Call these manually; they
#         are NOT part of the printed run.
# =============================================================================
def save_project(path="project.json"):
    project = {"codebook": CODEBOOK_DICT, "data": DATA}
    with open(path, "w") as f:
        json.dump(project, f, indent=2)
    print(f"\n[view c] Saved project to {path} "
          f"({len(DATA)} items, {len(CODEBOOK_DICT)} codes).")


def load_project(path="project.json"):
    with open(path, "r") as f:
        project = json.load(f)
    print(f"[view c] Loaded project from {path} "
          f"({len(project['data'])} items, {len(project['codebook'])} codes).")
    return project


# =============================================================================
# MAIN PIPELINE
#   [SOURCE: repo] the single -> debate -> consensus loop is Conrad's design.
#   [NEW]          the disagreement scoring and views a and b are added here.
#   Runs on the REAL model + REAL embeddings; only DATA is placeholder.
# =============================================================================
def run():
    # Agents created here (names/personalities); their behavior is in the classes above.
    agent1 = BaseCodingAgent("Sternberg", "bold and dominant but elaborative")
    agent2 = BaseCodingAgent("Marcuse", "bold and dominant but elaborative")
    consensus_agent = ConsensusAgent("Butler", "balanced and reflective")
    single_agent = SingleAgentCoding("Alex", "rigorous and empirical")

    review_rows = []

    for idx, item in enumerate(DATA):
        text = item["text"]
        print(f"\n=== Item {idx}: {text!r} ===")

        # [SOURCE: repo] single agent baseline (kept for parity; not used downstream)
        _ = single_agent.assign_code(CODEBOOK, text)

        # [SOURCE: repo] two-agent debate, then consensus tie-breaker if they disagree
        discussion, consensus_reached, _retries = DualAgentDiscussion(agent1, agent2).discuss(CODEBOOK, text, rounds=1)
        a1_msg = discussion[-2]["content"]
        a2_msg = discussion[-1]["content"]

        if consensus_reached == 0:
            final_msg = consensus_agent.resolve(CODEBOOK, text, a1_msg, a2_msg)
            final_code = extract_and_complete_code(final_msg, CODEBOOK_DICT)
            final_source_msg = final_msg
        else:
            final_code = extract_and_complete_code(a2_msg, CODEBOOK_DICT)
            final_source_msg = a2_msg

        codes_match = final_code == _expected_code_dict(item["human_code"])
        llm_label = next((k for k, v in final_code.items() if v == 1), None)
        print(f"  human: {item['human_code']!r}  |  llm: {llm_label!r}  |  match: {codes_match}")

        # [NEW] DISAGREEMENT TRACE (the core contribution): embed the human's
        # reason and the model's reason for this line, then cosine-compare them.
        # Low similarity = the two reasoned very differently about this line.
        llm_just = justification_of(final_source_msg)
        u, v = embed([item["human_justification"], llm_just])
        sim = cosine(u, v)
        print(f"  human reason: {item['human_justification']}")
        print(f"  llm reason:   {llm_just}")
        print(f"  reasoning sim (human vs LLM): {sim:.2f}")

        review_rows.append((idx, text, item["human_code"], llm_label, codes_match, sim))

    # [NEW] DISAGREEMENT RANKING: sort all items by lowest similarity so the
    # biggest human/model reasoning disagreements surface first for review.
    print("\n\nDISAGREEMENT RANKING (lowest reasoning sim = review first)")
    print("-" * 78)
    print(f"{'idx':>3}  {'sim':>5}  {'codes_match':>11}  text")
    print("-" * 78)
    for idx, text, construct, llm_label, codes_match, sim in sorted(review_rows, key=lambda r: r[5]):
        truncated = text if len(text) <= 60 else text[:57] + "..."
        print(f"{idx:>3}  {sim:>5.2f}  {str(codes_match):>11}  {truncated}")

    # -------------------------------------------------------------------------
    # VIEW A  [NEW]  -- Conrad's requested view (a)
    #   WHAT: sort the codes by how much the human and model disagreed on them.
    #   HOW:  group each item's similarity score by its code, average each group,
    #         sort the codes by that average (lowest avg = most disagreement).
    # -------------------------------------------------------------------------
    by_construct = {}
    for idx, text, construct, llm_label, codes_match, sim in review_rows:
        by_construct.setdefault(construct, []).append(sim)

    print("\n\nVIEW A  -  CONSTRUCT SUMMARY (most disagreement first)")
    print("-" * 78)
    print(f"{'construct':<34}  {'items':>5}  {'avg sim':>7}")
    print("-" * 78)
    for construct, sims in sorted(by_construct.items(), key=lambda kv: mean(kv[1])):
        print(f"{construct:<34}  {len(sims):>5}  {mean(sims):>7.2f}")

    # -------------------------------------------------------------------------
    # VIEW B  [NEW]  -- Conrad's requested view (b)
    #   WHAT: within one code, list its most-disagreeing lines first, so a human
    #         can re-annotate the ambiguous ones.
    #   HOW:  for each code, filter to its items, sort by lowest similarity, and
    #         flag any where the model's label differed from the human's.
    # -------------------------------------------------------------------------
    print("\n\nVIEW B  -  WITHIN-CONSTRUCT REVIEW (high disagreement first)")
    print("-" * 78)
    for construct, sims in sorted(by_construct.items(), key=lambda kv: mean(kv[1])):
        print(f"\n[{construct}]  avg sim {mean(sims):.2f}")
        construct_rows = [r for r in review_rows if r[2] == construct]
        for idx, text, c, llm_label, codes_match, sim in sorted(construct_rows, key=lambda r: r[5]):
            flag = "" if codes_match else "   <- label mismatch"
            tr = text if len(text) <= 55 else text[:52] + "..."
            print(f"   {sim:>5.2f}  {tr}{flag}")

    # [NEW] view d (similar items) prints last
    view_d_similar_items(top_k=2)


if __name__ == "__main__":
    run()