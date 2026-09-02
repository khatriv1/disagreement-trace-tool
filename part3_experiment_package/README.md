# Part 3 Multi-Round Codebook Revision — Collaborator Package

This package runs the anchored-holdout multi-round codebook revision experiment for tutoring dialogue (LABEL condition design). A collaborator can run any of **7 signal conditions** on their own laptop and send back the result zip files.

## What it does

- **Round 0:** Score agent A (Qwen 7B) against human gold using the theory codebook from `codebook_gpt_human.json`.
- **Rounds 1–5:** Each round re-codes all tutoring utterances with both agents, selects reviser cases by **signal condition**, revises the codebook conservatively (14B reviser), and re-scores with Cohen's kappa + accuracy.
- **Holdout design:** The reviser never sees human labels — only model-agreed anchors + signal-selected cases.
- **Significance test:** Bootstrap + permutation tests comparing round 0 vs round 5 and round 0 vs best post-revision round.

## Setup (one time)

### 1. Install Ollama and pull models

```bash
# Install from https://ollama.com then:
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
ollama serve   # keep running in a terminal
```

### 2. Python dependencies

```bash
cd part3_experiment_package
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add confidential data

Copy these four files into `./data/` (see `data/README.txt`):

| File | Purpose |
|------|---------|
| `First Author Copy GPT-Then-Human - Transcript B.csv` | Tutoring transcripts + human gold |
| `First Author Copy GPT-Then-Human - transcript C.csv` | Tutoring transcripts + human gold |
| `datapoint_table.csv` | Ambiguity + reasoning disagreement metadata |
| `rationale_cache.csv` | Agent rationales for reasoning signal |

The script prints a clear error if any file is missing.

## Run an experiment

```bash
chmod +x run_experiment.sh
./run_experiment.sh label
```

**Valid conditions:**

| Condition | Signal shown to reviser |
|-----------|-------------------------|
| `label` | Utterances where agents' labels disagree (recomputed each round) |
| `reasoning` | Top-N by reasoning-trace disagreement (cosine distance) |
| `ambiguity` | Top-N by data ambiguity |
| `label_reasoning` | Union of label + reasoning cases |
| `label_ambiguity` | Union of label + ambiguity cases |
| `reasoning_ambiguity` | Union of reasoning + ambiguity cases |
| `all_three` | Union of all three signals |

Each run writes to `results/<condition>/` and produces `results_<condition>.zip`.

## Runtime

Each condition takes approximately **2–3 hours** (5 revision rounds × dual-agent coding + 14B reviser + scoring).

**Suggested split (7 conditions total):**

- Collaborator runs 3: e.g. `label`, `reasoning`, `ambiguity`
- Conrad runs 4: e.g. `label_reasoning`, `label_ambiguity`, `reasoning_ambiguity`, `all_three`

## Send back

After each condition finishes, send the zip file:

```
results_<condition>.zip
```

Example: `results_all_three.zip`

Each zip contains round-by-round kappa CSV, codebooks, scored caches, summary, and significance results.

## Manual commands (optional)

```bash
python run_experiment.py --condition reasoning
python significance.py --condition reasoning
```

## Files in this package

```
part3_experiment_package/
  run_experiment.sh          # main entry point
  run_experiment.py          # multi-round experiment
  significance.py            # bootstrap + permutation tests
  consensus_coding.py        # Ollama coding helpers
  codebook_gpt_human.json    # theory-grounded round-0 codebook
  requirements.txt
  README.md
  data/                      # collaborator adds CSVs here
  results/                   # created at runtime
```

## Resume after crash

Re-run the same command. Cached dual-agent and scored CSVs are reused; completed codebook JSON files are skipped. The results CSV appends new rows — delete `results/<condition>/` to start fresh.
