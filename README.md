# Disagreement Trace Tool

Research code for studying how LLM coding disagreements and codebook revisions affect agreement with human qualitative labels on dialogue data.

## Data

**Transcript and tutoring data are not included in this repository** (restricted use). To obtain the underlying datasets, contact the original study authors on request. The experiment package expects confidential CSVs under `part3_experiment_package/data/`; see `part3_experiment_package/data/README.txt` for the required filenames.

Published outputs here are **numeric summaries only** (kappa tables, significance tests, plots) — no utterance text.

## Part 3: Multi-round codebook revision

The main experiment lives in `part3_experiment_package/`.

### `run_experiment.py`

Runs the anchored-holdout multi-round revision loop for tutoring dialogue:

- **Round 0:** Dual-agent coding (Qwen 7B + Llama 8B) against human gold using the starting codebook.
- **Rounds 1–5:** Select reviser cases by signal condition (label disagreement, reasoning distance, ambiguity, or unions thereof), revise the codebook with a 14B reviser, and re-score each round.
- **Holdout:** The reviser does not see human labels — only model-agreed anchors plus signal-selected cases.

Seven conditions: `label`, `reasoning`, `ambiguity`, `label_reasoning`, `label_ambiguity`, `reasoning_ambiguity`, `all_three`.

```bash
cd part3_experiment_package
./run_experiment.sh all_three
# or: python run_experiment.py --condition all_three --seed 42
```

Bare-codebook replications (seeds 42, 123, 7) are under `results_bare_all_three_seed*/`. Single-signal runs are under `results/label`, `results/reasoning`, and `results/ambiguity`.

### `significance.py`

Bootstrap and permutation tests comparing baseline (round 0) vs final or best post-revision kappa for a completed run. Writes `significance_results.txt` next to `results.csv`.

```bash
python significance.py --condition all_three
```

### Analysis scripts (repo root)

| Script | Purpose |
|--------|---------|
| `plot_revision_trajectories.py` | Pooled kappa trajectories (rounds 0–5) for bare `all_three` seeds with bootstrap 95% CIs; outputs `revision_trajectories.pdf` / `.png`. |
| `embedding_revision_analysis.py` | Embeds revision-text diffs; correlates embeddings with Δκ (leave-one-run-out CV). Writes `embedding_revision_analysis.txt` and `revision_steps.csv`. |
| `revision_cluster_analysis.py` | k-means clustering of revision embeddings; writes `revision_clusters.csv` (numbers only; full text report is local). |
| `part3_setup.py` | Builds matched case sets for the three revision signals. |
| `part3_significance.py` | Standalone significance testing for early weak-start runs. |
| `part3_multiround.py` / `part3_weakstart.py` | Earlier multi-round and weak-start prototypes. |
| `part3_prototype.py` / `part3_holdout_prototype.py` | Round-1 holdout prototypes (label-only reviser inputs). |
| `part3_diagnose.py` | Per-code confusion-matrix diagnosis for a revised codebook. |
| `part_a_close.py` | Mixed-effects models relating ambiguity vs label disagreement to coding accuracy. |

### Requirements

- Python 3.10+ with packages in `part3_experiment_package/requirements.txt` (pandas, numpy, scipy, scikit-learn, matplotlib, sentence-transformers).
- [Ollama](https://ollama.com) with `qwen2.5:7b`, `llama3.1:8b`, and `qwen2.5:14b` for re-running experiments.

## Other components

- `consensus_coding.py` — dual-agent consensus coding pipeline with disagreement views.
- `mockup.html` + `build_mockup_data.py` — qualitative UI mockup for exploring disagreements.
- `step1_kappa_vs_disagreement.py` / `step1_tutoring.py` — Step 1 replication analyses.
- `similar_pairs_lak24.py` / `reasoning_disagreement_lak24.py` — LAK24 utterance analyses (separate public dataset; not bundled here).

## Results layout

```
part3_experiment_package/
  results/{label,reasoning,ambiguity,label_*,reasoning_*,all_three}/
    results.csv              # per-round pooled kappa
    significance_results.txt
    summary.txt
  results_bare_all_three_seed{42,123,7}/
  results_all_three_seed123/
```

Scored utterance caches, dual-agent inputs, and revised codebook JSON files are gitignored and must be produced locally or obtained separately.
