#!/bin/bash
# Generate TypeScript types from Python Pydantic models
#
# This script:
# 1. Generates JSON Schema from core/ Python models
# 2. Converts JSON Schema to TypeScript definitions
#
# Usage: npm run generate:types

set -e

# Navigate to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$FRONTEND_DIR")")"

echo "📦 Generating JSON Schema from Python models..."
cd "$REPO_ROOT"

# Generate combined schema
python -m core.export --output-dir "$FRONTEND_DIR/schemas" --combined

echo "🔄 Converting JSON Schema to TypeScript..."
cd "$FRONTEND_DIR"

JSON2TS_BIN="$FRONTEND_DIR/node_modules/.bin/json2ts"
if [ -x "$JSON2TS_BIN" ]; then
  "$JSON2TS_BIN" \
    schemas/lancer.json \
    -o src/lib/types/lancer.ts \
    --bannerComment "/* Auto-generated from core/ Pydantic models. DO NOT EDIT. */"
else
  NPM_CONFIG_CACHE="$FRONTEND_DIR/.npm-cache" \
    npx json2ts \
    schemas/lancer.json \
    -o src/lib/types/lancer.ts \
    --bannerComment "/* Auto-generated from core/ Pydantic models. DO NOT EDIT. */"
fi

echo "✅ Type generation complete!"
echo "   Output: src/lib/types/lancer.ts"
