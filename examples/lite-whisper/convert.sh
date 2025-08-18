#!/bin/bash

# Get the absolute path of the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the base model from argument, default to "whisper-tiny"
BASE_MODEL="${1:-whisper-tiny}"

# Use the model name for folder naming
BASE_NAME="$BASE_MODEL"

# Determine output directories based on model name
OUTPUT_DIR_BASE="$SCRIPT_DIR/${BASE_NAME}-ct2"
OUTPUT_DIR_LITE="$SCRIPT_DIR/lite-${BASE_NAME}-ct2"

# Convert base model
ct2-transformers-converter --force --model "openai/$BASE_MODEL" --output_dir "$OUTPUT_DIR_BASE"

# Convert lite model
ct2-transformers-converter --force --model "efficient-speech/lite-$BASE_NAME" \
    --output_dir "$OUTPUT_DIR_LITE" --trust_remote_code
