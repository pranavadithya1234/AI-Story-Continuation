#!/bin/bash
# Package DistilGPT2 model and handler into a TorchServe model archive (.mar).
#
# Prerequisites:
#   pip install torch-model-archiver torchserve
#
# Usage:
#   bash scripts/package_model.sh
set -e

ARTIFACT_DIR="model_artifacts"
MODEL_STORE="model-store"

echo "=== Step 1: Download model files ==="
mkdir -p "$ARTIFACT_DIR"
python3 - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "distilgpt2"
print(f"Downloading {name} ...")
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
# Save with safe_serialization=False to produce pytorch_model.bin
# (TorchServe model-archiver expects a single serialized-file)
tok.save_pretrained("model_artifacts")
model.save_pretrained("model_artifacts", safe_serialization=False)
print("Download complete.")
PY

echo ""
echo "=== Step 2: Create model archive ==="
mkdir -p "$MODEL_STORE"

# Build the list of extra files (all non-.bin files needed by the handler)
EXTRA_FILES=""
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt generation_config.json; do
    fpath="$ARTIFACT_DIR/$f"
    if [ -f "$fpath" ]; then
        if [ -n "$EXTRA_FILES" ]; then
            EXTRA_FILES="$EXTRA_FILES,$fpath"
        else
            EXTRA_FILES="$fpath"
        fi
    fi
done

echo "Serialized file : $ARTIFACT_DIR/pytorch_model.bin"
echo "Extra files     : $EXTRA_FILES"
echo "Handler         : torchserve/story_handler.py"

torch-model-archiver \
    --model-name storymodel \
    --version 1.0 \
    --serialized-file "$ARTIFACT_DIR/pytorch_model.bin" \
    --handler torchserve/story_handler.py \
    --extra-files "$EXTRA_FILES" \
    --export-path "$MODEL_STORE" \
    --force

echo ""
echo "✅ Model archive created: $MODEL_STORE/storymodel.mar"
echo ""
echo "To start TorchServe:"
echo "  torchserve --start --model-store $MODEL_STORE \\"
echo "    --models storymodel=storymodel.mar \\"
echo "    --ts-config torchserve/config.properties"
