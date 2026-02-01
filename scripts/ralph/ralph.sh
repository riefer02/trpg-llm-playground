#!/bin/bash
# Ralph - Long-running AI agent loop
# Usage: ./ralph.sh [--tool claude|opencode] [--model MODEL] [max_iterations]
#
# Supports:
#   - claude:   Claude Code CLI (npm install -g @anthropic-ai/claude-code)
#   - opencode: Open-source, multi-provider (curl -fsSL https://opencode.ai/install | bash)
#
# Examples:
#   ./ralph.sh                                    # Claude Code, 10 iterations
#   ./ralph.sh --tool opencode 20                 # OpenCode, 20 iterations
#   ./ralph.sh --tool opencode --model ollama/llama3  # Local model

set -e

# Parse arguments
TOOL="claude"  # Default
MODEL="opencode/kimi-k2.5-free"
MAX_ITERATIONS=10

while [[ $# -gt 0 ]]; do
  case $1 in
    --tool)
      TOOL="$2"
      shift 2
      ;;
    --tool=*)
      TOOL="${1#*=}"
      shift
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --model=*)
      MODEL="${1#*=}"
      shift
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS="$1"
      fi
      shift
      ;;
  esac
done

# Validate tool choice
if [[ "$TOOL" != "claude" && "$TOOL" != "opencode" ]]; then
  echo "Error: Invalid tool '$TOOL'. Must be 'claude' or 'opencode'."
  echo ""
  echo "Install tools:"
  echo "  claude:   npm install -g @anthropic-ai/claude-code"
  echo "  opencode: curl -fsSL https://opencode.ai/install | bash"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
PROMPT_FILE="$SCRIPT_DIR/PROMPT.md"
ARCHIVE_DIR="$SCRIPT_DIR/archive"
LAST_BRANCH_FILE="$SCRIPT_DIR/.last-branch"

# Check required files exist
if [ ! -f "$PRD_FILE" ]; then
  echo "Error: prd.json not found at $PRD_FILE"
  exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
  echo "Error: PROMPT.md not found at $PROMPT_FILE"
  exit 1
fi

# Archive previous run if branch changed
if [ -f "$PRD_FILE" ] && [ -f "$LAST_BRANCH_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  LAST_BRANCH=$(cat "$LAST_BRANCH_FILE" 2>/dev/null || echo "")

  if [ -n "$CURRENT_BRANCH" ] && [ -n "$LAST_BRANCH" ] && [ "$CURRENT_BRANCH" != "$LAST_BRANCH" ]; then
    DATE=$(date +%Y-%m-%d)
    FOLDER_NAME=$(echo "$LAST_BRANCH" | sed 's|^ralph/||')
    ARCHIVE_FOLDER="$ARCHIVE_DIR/$DATE-$FOLDER_NAME"

    echo "Archiving previous run: $LAST_BRANCH"
    mkdir -p "$ARCHIVE_FOLDER"
    [ -f "$PRD_FILE" ] && cp "$PRD_FILE" "$ARCHIVE_FOLDER/"
    [ -f "$PROGRESS_FILE" ] && cp "$PROGRESS_FILE" "$ARCHIVE_FOLDER/"
    echo "   Archived to: $ARCHIVE_FOLDER"

    # Reset progress file
    echo "# Ralph Progress Log" > "$PROGRESS_FILE"
    echo "Started: $(date)" >> "$PROGRESS_FILE"
    echo "---" >> "$PROGRESS_FILE"
  fi
fi

# Track current branch
if [ -f "$PRD_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  if [ -n "$CURRENT_BRANCH" ]; then
    echo "$CURRENT_BRANCH" > "$LAST_BRANCH_FILE"
  fi
fi

# Initialize progress file if needed
if [ ! -f "$PROGRESS_FILE" ]; then
  echo "# Ralph Progress Log" > "$PROGRESS_FILE"
  echo "Started: $(date)" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
fi

echo "========================================"
echo "  Ralph - Autonomous Agent Loop"
echo "========================================"
echo "Tool: $TOOL"
if [[ -n "$MODEL" ]]; then
  echo "Model: $MODEL"
fi
echo "Max iterations: $MAX_ITERATIONS"
echo "Project: $(jq -r '.project' "$PRD_FILE")"
echo "Branch: $(jq -r '.branchName' "$PRD_FILE")"
echo ""

# Show current story status
echo "Story Status:"
jq -r '.userStories[] | "  [\(if .passes then "x" else " " end)] \(.id): \(.title)"' "$PRD_FILE"
echo ""

for i in $(seq 1 $MAX_ITERATIONS); do
  echo ""
  echo "==============================================================="
  echo "  Iteration $i of $MAX_ITERATIONS"
  echo "==============================================================="

  # Check if all stories pass before starting
  INCOMPLETE=$(jq '[.userStories[] | select(.passes != true)] | length' "$PRD_FILE")
  if [ "$INCOMPLETE" -eq 0 ]; then
    echo ""
    echo "All stories complete! Exiting."
    exit 0
  fi

  # Run the selected tool
  cd "$PROJECT_ROOT"

  PROMPT_CONTENT=$(cat "$PROMPT_FILE")

  if [[ "$TOOL" == "claude" ]]; then
    # Claude Code: pipe prompt, use --dangerously-skip-permissions for autonomous mode
    OUTPUT=$(claude --dangerously-skip-permissions --print < "$PROMPT_FILE" 2>&1 | tee /dev/stderr) || true

  elif [[ "$TOOL" == "opencode" ]]; then
    # OpenCode: use 'run' for non-interactive mode
    # Optionally specify model with --model flag
    if [[ -n "$MODEL" ]]; then
      OUTPUT=$(opencode run --model "$MODEL" "$PROMPT_CONTENT" 2>&1 | tee /dev/stderr) || true
    else
      OUTPUT=$(opencode run "$PROMPT_CONTENT" 2>&1 | tee /dev/stderr) || true
    fi
  fi

  # Check for completion signal
  if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
    echo ""
    echo "=========================================="
    echo "  Ralph completed all tasks!"
    echo "  Finished at iteration $i of $MAX_ITERATIONS"
    echo "=========================================="
    exit 0
  fi

  echo ""
  echo "Iteration $i complete. Sleeping 2s before next iteration..."
  sleep 2
done

echo ""
echo "=========================================="
echo "  Reached max iterations ($MAX_ITERATIONS)"
echo "  Check progress.txt for status"
echo "=========================================="

# Final status
echo ""
echo "Final Story Status:"
jq -r '.userStories[] | "  [\(if .passes then "x" else " " end)] \(.id): \(.title)"' "$PRD_FILE"

exit 1
