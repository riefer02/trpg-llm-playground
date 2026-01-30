# Ralph - Autonomous AI Development Loop

Ralph repeatedly runs an AI coding agent until all PRD user stories pass. Each iteration gets fresh context, with memory persisted through git commits, `progress.txt`, and `prd.json` status.

## Quick Start

```bash
# Claude Code (default)
./ralph.sh

# OpenCode (open-source, multi-provider)
./ralph.sh --tool opencode

# With specific model
./ralph.sh --tool opencode --model ollama/llama3
./ralph.sh --tool opencode --model anthropic/claude-3-5-sonnet

# Set max iterations
./ralph.sh 20
```

## Prerequisites

Install one of the supported tools:

| Tool | Install Command |
|------|-----------------|
| Claude Code | `npm install -g @anthropic-ai/claude-code` |
| OpenCode | `curl -fsSL https://opencode.ai/install \| bash` |

Also requires `jq` for JSON processing: `brew install jq`

## Files

| File | Purpose |
|------|---------|
| `ralph.sh` | Main loop script |
| `PROMPT.md` | Instructions fed to AI each iteration |
| `prd.json` | User stories with `passes: true/false` |
| `progress.txt` | Append-only learnings log |
| `archive/` | Previous PRD runs (auto-archived) |

## PRD Format

```json
{
  "project": "feature-name",
  "branchName": "ralph/feature-name",
  "description": "What we're building",
  "userStories": [
    {
      "id": "US-001",
      "title": "Story title",
      "description": "Full description",
      "acceptanceCriteria": [
        "Criterion 1",
        "Criterion 2",
        "make test-core passes"
      ],
      "priority": 1,
      "passes": false,
      "notes": ""
    }
  ]
}
```

## How It Works

1. **Read PRD** - Find highest-priority incomplete story
2. **Implement** - AI writes code to satisfy acceptance criteria
3. **Quality checks** - `make test-core`, `make lint`
4. **Commit** - Only if checks pass
5. **Update status** - Mark story `passes: true` in prd.json
6. **Log learnings** - Append to progress.txt
7. **Repeat** - Until all stories pass or max iterations

## Key Concepts

### Fresh Context Per Iteration
Each run starts with clean context. Use `progress.txt` for continuity between iterations.

### Right-Sized Tasks
Stories should complete in one iteration. Break large features into smaller stories.

### Quality Gates
Tests must pass before commits. Failed checks = fix and retry within iteration.

### Completion Signal
When all stories pass, output `<promise>COMPLETE</promise>` to exit the loop.

## Adding Features

1. Edit `prd.json` to add user stories
2. Run `./ralph.sh`
3. Monitor progress in terminal
4. Check `progress.txt` for learnings

## Archiving

When you change `branchName` in prd.json, Ralph automatically archives the previous run to `archive/YYYY-MM-DD-branch-name/`.

## Current PRD: AI Tactician

The current PRD implements the AI Tactician for Lancer combat:

| ID | Title | Priority |
|----|-------|----------|
| US-001 | Combat state serializer | 1 |
| US-002 | Tactical system prompt | 2 |
| US-003 | Action parser/validator | 3 |
| US-004 | Tactician class | 4 |
| US-005 | Wire into NPC turn flow | 5 |
| US-006 | UI displays AI reasoning | 6 |

See `notes/epic_ai_tactician.md` for full design details.
