#!/bin/bash
# Run synthetic data generation locally
# Usage: ./scripts/run_local.sh [config_file]

set -e

# Load .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Loaded .env"
fi

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "   Create a .env file with: OPENAI_API_KEY=sk-..."
    exit 1
fi

# Config file (default to local_lancer.yaml)
CONFIG=${1:-"config/local_lancer.yaml"}

if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    exit 1
fi

echo "🎲 TTRPG Synthetic Data Generator"
echo "   Config: $CONFIG"
echo ""

# Create dataset directory
mkdir -p dataset

# Step 1: RAG Chunking (if enabled)
echo "📚 Step 1: Chunking PDF..."
python -m src.rag.chunk_pdf --config "$CONFIG" 2>/dev/null || echo "   Skipped (rag_ingest not enabled or already done)"

# Step 2: Generate synthetic data
echo ""
echo "🤖 Step 2: Generating synthetic data..."
python -m src.data.generate_synthetic --config "$CONFIG"

# Step 3: Generate quality report
echo ""
echo "📊 Step 3: Generating quality report..."
# Find the most recent generated file (matches _synthetic_, _walkthrough_, etc.)
# Extract project_name and dataset_tag from config to build the pattern
PROJECT=$(grep -E "^project_name:" "$CONFIG" | awk '{print $2}' | tr -d '"')
TAG=$(grep -E "^dataset_tag:" "$CONFIG" | awk '{print $2}' | tr -d '"')
if [ -n "$PROJECT" ] && [ -n "$TAG" ]; then
    LATEST=$(ls -t dataset/${PROJECT}_${TAG}_*.jsonl 2>/dev/null | head -1)
else
    # Fallback: find most recent .jsonl that's not a chunks file
    LATEST=$(ls -t dataset/*.jsonl 2>/dev/null | grep -v "_chunks.jsonl" | head -1)
fi
if [ -n "$LATEST" ]; then
    REPORT="${LATEST%.jsonl}_report.md"
    python -m src.data.synth_report --input "$LATEST" --output "$REPORT"
    echo "   Report saved: $REPORT"
fi

echo ""
echo "✅ Done! Files saved to dataset/"
echo ""
echo "📁 To upload to Google Drive:"
echo "   1. Open Google Drive in browser"
echo "   2. Navigate to: llm_experiments/datasets/"
echo "   3. Drag and drop the dataset/ folder contents"

