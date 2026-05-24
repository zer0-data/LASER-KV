#!/bin/bash
# Downloads external data files required for RULER QA and NIAH-essay tasks.
# Run once before generating data:
#   bash ruler/data/download_data.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSON_DIR="$SCRIPT_DIR/synthetic/json"
mkdir -p "$JSON_DIR"

# SQuAD v2.0 dev set (for qa_1)
if [ ! -f "$JSON_DIR/squad.json" ]; then
    echo "Downloading SQuAD v2.0..."
    wget -q -O "$JSON_DIR/squad.json" \
        https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
    echo "  -> $JSON_DIR/squad.json"
else
    echo "SQuAD already exists."
fi

# HotpotQA distractor dev set (for qa_2)
if [ ! -f "$JSON_DIR/hotpotqa.json" ]; then
    echo "Downloading HotpotQA..."
    wget -q -O "$JSON_DIR/hotpotqa.json" \
        http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    echo "  -> $JSON_DIR/hotpotqa.json"
else
    echo "HotpotQA already exists."
fi

# Paul Graham Essays (for niah_single_2/3, niah_multikey_1, niah_multivalue, niah_multiquery)
if [ ! -f "$JSON_DIR/PaulGrahamEssays.json" ]; then
    echo "Downloading Paul Graham Essays..."
    pip install -q beautifulsoup4 html2text tqdm
    python "$JSON_DIR/download_paulgraham_essay.py"
else
    echo "Paul Graham Essays already exist."
fi

echo ""
echo "Done. Files in $JSON_DIR:"
ls -lh "$JSON_DIR"
