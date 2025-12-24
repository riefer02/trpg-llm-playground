"""
Post-generation quality dashboard and reporting.

Generates a comprehensive report of synthetic data quality metrics,
sample distributions, and flagged issues after generation completes.
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def load_synthetic_data(path: str) -> List[Dict]:
    """Load synthetic data from JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def analyze_dataset(records: List[Dict]) -> Dict:
    """Compute comprehensive statistics about the dataset."""
    stats = {
        "total_records": len(records),
        "task_type_distribution": Counter(),
        "difficulty_distribution": Counter(),
        "multiturn_count": 0,
        "negative_count": 0,
        "has_citations": 0,
        "has_messages": 0,
        "token_lengths": [],
        "instruction_lengths": [],
        "output_lengths": [],
        "pages_covered": set(),
        "turn_distribution": Counter(),
        "issues": [],
    }

    for record in records:
        # Task type
        task_type = record.get("task_type", "unknown")
        stats["task_type_distribution"][task_type] += 1

        # Difficulty (if tracked)
        difficulty = record.get("difficulty")
        if difficulty:
            stats["difficulty_distribution"][difficulty] += 1

        # Multi-turn
        if record.get("is_multiturn"):
            stats["multiturn_count"] += 1
            turn_count = record.get("turn_count", 0)
            stats["turn_distribution"][turn_count] += 1

        # Negative examples (check output for "Not found in context")
        output = record.get("output", "")
        if "Not found in context" in output:
            stats["negative_count"] += 1

        # Citations
        if "Citations:" in output or record.get("citations"):
            stats["has_citations"] += 1

        # Messages format
        if record.get("messages"):
            stats["has_messages"] += 1

        # Lengths
        instruction = record.get("instruction", "")
        context = record.get("context", record.get("input", ""))
        stats["instruction_lengths"].append(len(instruction))
        stats["output_lengths"].append(len(output))
        stats["token_lengths"].append(len(instruction) + len(context) + len(output))

        # Pages
        page = record.get("source_page")
        if page:
            stats["pages_covered"].add(page)

        # Issue detection
        if len(output) < 50:
            stats["issues"].append({
                "type": "short_output",
                "instruction": instruction[:100],
                "output_len": len(output),
            })
        if len(instruction) + len(context) + len(output) > 16000:
            stats["issues"].append({
                "type": "very_long",
                "instruction": instruction[:100],
                "total_len": len(instruction) + len(context) + len(output),
            })
        if "Citations:" not in output and task_type in ("rules_qa", "lore"):
            stats["issues"].append({
                "type": "missing_citation",
                "instruction": instruction[:100],
                "task_type": task_type,
            })

    # Convert sets for JSON serialization
    stats["pages_covered"] = sorted(list(stats["pages_covered"]))
    stats["unique_pages"] = len(stats["pages_covered"])

    # Compute length stats
    if stats["token_lengths"]:
        stats["avg_total_length"] = sum(stats["token_lengths"]) / len(stats["token_lengths"])
        stats["max_total_length"] = max(stats["token_lengths"])
        stats["min_total_length"] = min(stats["token_lengths"])
    if stats["output_lengths"]:
        stats["avg_output_length"] = sum(stats["output_lengths"]) / len(stats["output_lengths"])

    return stats


def sample_by_category(
    records: List[Dict],
    n_per_category: int = 2,
) -> Dict[str, List[Dict]]:
    """Sample representative examples from each category."""
    samples = defaultdict(list)

    # Sample by task type
    by_task = defaultdict(list)
    for r in records:
        by_task[r.get("task_type", "unknown")].append(r)

    for task_type, task_records in by_task.items():
        samples[f"task_{task_type}"] = task_records[:n_per_category]

    # Sample multi-turn
    multiturn = [r for r in records if r.get("is_multiturn")]
    samples["multiturn"] = multiturn[:n_per_category]

    # Sample negatives
    negatives = [r for r in records if "Not found in context" in r.get("output", "")]
    samples["negatives"] = negatives[:n_per_category]

    return dict(samples)


def format_sample_markdown(record: Dict) -> str:
    """Format a single sample as markdown."""
    lines = []
    lines.append(f"**Task Type**: {record.get('task_type', 'unknown')}")
    if record.get("is_multiturn"):
        lines.append(f"**Multi-turn**: {record.get('turn_count', '?')} turns")
    if record.get("difficulty"):
        lines.append(f"**Difficulty**: {record.get('difficulty')}")
    lines.append(f"**Source Page**: {record.get('source_page', 'unknown')}")
    lines.append("")
    lines.append(f"**Question**: {record.get('instruction', '')}")
    lines.append("")
    lines.append(f"**Answer**:")
    lines.append("```")
    lines.append(record.get("output", "")[:500])
    if len(record.get("output", "")) > 500:
        lines.append("... (truncated)")
    lines.append("```")
    return "\n".join(lines)


def generate_markdown_report(
    stats: Dict,
    samples: Dict[str, List[Dict]],
    config: Optional[Dict] = None,
    output_path: Optional[str] = None,
) -> str:
    """Generate a comprehensive markdown report."""
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# Synthetic Data Quality Report")
    lines.append(f"\n*Generated: {timestamp}*\n")

    # Overview
    lines.append("## Overview\n")
    lines.append(f"- **Total Records**: {stats['total_records']:,}")
    lines.append(f"- **Unique Pages Covered**: {stats.get('unique_pages', 'N/A')}")
    lines.append(f"- **Multi-turn Conversations**: {stats['multiturn_count']} ({100*stats['multiturn_count']/max(1,stats['total_records']):.1f}%)")
    lines.append(f"- **Negative Examples**: {stats['negative_count']} ({100*stats['negative_count']/max(1,stats['total_records']):.1f}%)")
    lines.append(f"- **With Citations**: {stats['has_citations']} ({100*stats['has_citations']/max(1,stats['total_records']):.1f}%)")
    lines.append(f"- **With Messages Format**: {stats['has_messages']} ({100*stats['has_messages']/max(1,stats['total_records']):.1f}%)")
    lines.append("")

    # Length stats
    lines.append("## Token Length Statistics\n")
    lines.append(f"- **Average Total Length**: {stats.get('avg_total_length', 0):,.0f} chars")
    lines.append(f"- **Max Total Length**: {stats.get('max_total_length', 0):,} chars")
    lines.append(f"- **Average Output Length**: {stats.get('avg_output_length', 0):,.0f} chars")
    lines.append("")

    # Task type distribution
    lines.append("## Task Type Distribution\n")
    lines.append("| Task Type | Count | Percentage |")
    lines.append("|-----------|-------|------------|")
    for task_type, count in sorted(stats["task_type_distribution"].items(), key=lambda x: -x[1]):
        pct = 100 * count / max(1, stats["total_records"])
        lines.append(f"| {task_type} | {count} | {pct:.1f}% |")
    lines.append("")

    # Difficulty distribution (if present)
    if stats["difficulty_distribution"]:
        lines.append("## Difficulty Distribution\n")
        lines.append("| Difficulty | Count | Percentage |")
        lines.append("|------------|-------|------------|")
        for diff, count in sorted(stats["difficulty_distribution"].items()):
            pct = 100 * count / max(1, stats["total_records"])
            lines.append(f"| {diff} | {count} | {pct:.1f}% |")
        lines.append("")

    # Multi-turn distribution
    if stats["turn_distribution"]:
        lines.append("## Multi-turn Turn Distribution\n")
        lines.append("| Turns | Count |")
        lines.append("|-------|-------|")
        for turns, count in sorted(stats["turn_distribution"].items()):
            lines.append(f"| {turns} | {count} |")
        lines.append("")

    # Issues
    if stats["issues"]:
        lines.append("## Potential Issues\n")
        issue_counts = Counter(i["type"] for i in stats["issues"])
        for issue_type, count in issue_counts.most_common():
            lines.append(f"- **{issue_type}**: {count} occurrences")
        lines.append("")

        # Show examples of issues
        lines.append("### Issue Examples\n")
        shown_types = set()
        for issue in stats["issues"][:10]:
            if issue["type"] not in shown_types:
                shown_types.add(issue["type"])
                lines.append(f"**{issue['type']}**: `{issue.get('instruction', '')[:80]}...`")
        lines.append("")

    # Sample examples
    lines.append("## Sample Examples\n")
    for category, category_samples in samples.items():
        if category_samples:
            lines.append(f"### {category.replace('_', ' ').title()}\n")
            for sample in category_samples[:2]:
                lines.append(format_sample_markdown(sample))
                lines.append("\n---\n")

    report = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def generate_report(
    synthetic_data_path: str,
    output_path: Optional[str] = None,
    config: Optional[Dict] = None,
) -> str:
    """
    Main entry point: Generate a quality report for synthetic data.
    
    Args:
        synthetic_data_path: Path to the synthetic JSONL file
        output_path: Where to save the markdown report (optional)
        config: Original generation config (optional, for context)
        
    Returns:
        The markdown report as a string
    """
    print(f"Analyzing: {synthetic_data_path}")
    records = load_synthetic_data(synthetic_data_path)
    
    if not records:
        return "# Error\n\nNo records found in the synthetic data file."
    
    print(f"Loaded {len(records)} records")
    stats = analyze_dataset(records)
    samples = sample_by_category(records)
    
    return generate_markdown_report(stats, samples, config, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate quality report for synthetic data")
    parser.add_argument("--input", required=True, help="Path to synthetic JSONL file")
    parser.add_argument("--output", help="Path to save markdown report")
    args = parser.parse_args()
    
    report = generate_report(args.input, args.output)
    if not args.output:
        print(report)

