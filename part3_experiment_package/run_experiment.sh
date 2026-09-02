#!/usr/bin/env bash
# Part 3 multi-round codebook revision — one condition end-to-end.
#
# Usage:
#   ./run_experiment.sh label
#   ./run_experiment.sh all_three
#
# Valid conditions:
#   label | reasoning | ambiguity | label_reasoning | label_ambiguity |
#   reasoning_ambiguity | all_three

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDITION="${1:-}"
VALID="label reasoning ambiguity label_reasoning label_ambiguity reasoning_ambiguity all_three"

if [[ -z "$CONDITION" ]]; then
  echo "Usage: ./run_experiment.sh <condition>"
  echo "Conditions: $VALID"
  exit 1
fi

if ! echo "$VALID" | grep -qw "$CONDITION"; then
  echo "FATAL: unknown condition '$CONDITION'"
  echo "Valid: $VALID"
  exit 1
fi

# ── Ollama check ──────────────────────────────────────────────────────────────
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "FATAL: Ollama is not running. Start it with: ollama serve"
  exit 1
fi

for MODEL in qwen2.5:7b llama3.1:8b qwen2.5:14b; do
  if ! curl -sf http://localhost:11434/api/tags | grep -q "$MODEL"; then
    echo "FATAL: model '$MODEL' not found. Run: ollama pull $MODEL"
    exit 1
  fi
  echo "  OK: $MODEL"
done

# ── Data check ────────────────────────────────────────────────────────────────
REQUIRED=(
  "data/First Author Copy GPT-Then-Human - Transcript B.csv"
  "data/First Author Copy GPT-Then-Human - transcript C.csv"
  "data/datapoint_table.csv"
  "data/rationale_cache.csv"
)
MISSING=0
for f in "${REQUIRED[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "  MISSING: $f"
    MISSING=1
  fi
done
if [[ "$MISSING" -eq 1 ]]; then
  echo ""
  echo "FATAL: add the required files to ./data/ (see data/README.txt)"
  exit 1
fi
echo "  OK: all data files present"

# ── Python env ─────────────────────────────────────────────────────────────────
if [[ -d .venv ]]; then
  PYTHON=".venv/bin/python"
elif [[ -d ../.venv ]]; then
  PYTHON="../.venv/bin/python"
else
  PYTHON="python3"
fi

echo ""
echo "Running experiment: condition=$CONDITION"
"$PYTHON" run_experiment.py --condition "$CONDITION"

echo ""
echo "Running significance tests ..."
"$PYTHON" significance.py --condition "$CONDITION"

ZIP="results_${CONDITION}.zip"
rm -f "$ZIP"
(cd results && zip -r "../$ZIP" "$CONDITION")
echo ""
echo "Created $ZIP ($(du -h "$ZIP" | cut -f1))"
echo "Done."
