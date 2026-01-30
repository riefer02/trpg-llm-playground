"""
RPG-Specific Evaluation Benchmark Framework.

Measures model quality on real RPG assistant tasks:
- Rules accuracy: Can it answer correctly?
- Grounding fidelity: Does it hallucinate?
- Citation accuracy: Are page references correct?
- Refusal calibration: Does it correctly say "not found"?
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class EvalExample:
    """A single evaluation example."""
    id: str
    question: str
    context: str
    expected_answer: str
    task_type: str
    difficulty: str = "intermediate"
    # For grounding evaluation
    answerable: bool = True  # False = should refuse
    expected_citations: List[str] = field(default_factory=list)  # e.g., ["p. 45", "p. 46"]
    # For scoring
    key_facts: List[str] = field(default_factory=list)  # Facts that must appear
    forbidden_claims: List[str] = field(default_factory=list)  # Hallucinations to penalize


@dataclass
class EvalResult:
    """Result of evaluating a single example."""
    example_id: str
    predicted: str
    # Scores (0.0 to 1.0)
    accuracy_score: float = 0.0
    grounding_score: float = 0.0
    citation_score: float = 0.0
    refusal_score: float = 0.0
    # Details
    key_facts_found: List[str] = field(default_factory=list)
    key_facts_missing: List[str] = field(default_factory=list)
    hallucinations_found: List[str] = field(default_factory=list)
    citations_found: List[str] = field(default_factory=list)
    correctly_refused: bool = False


def load_eval_set(path: str) -> List[EvalExample]:
    """Load evaluation set from YAML or JSONL file."""
    examples = []
    
    if path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for item in data.get("examples", []):
            examples.append(EvalExample(**item))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    examples.append(EvalExample(**item))
    
    return examples


def extract_citations(text: str) -> List[str]:
    """Extract citation references from text."""
    # Match patterns like (p. 45), (p. 45-46), (pp. 45, 46)
    patterns = [
        r"\(p\.\s*(\d+(?:-\d+)?)\)",
        r"\(pp\.\s*(\d+(?:,\s*\d+)*)\)",
        r"Citations?:\s*\(p\.\s*(\d+(?:-\d+)?)\)",
    ]
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        citations.extend(matches)
    return citations


def check_key_facts(response: str, key_facts: List[str]) -> Tuple[List[str], List[str]]:
    """Check which key facts are present/missing in the response."""
    found = []
    missing = []
    response_lower = response.lower()
    
    for fact in key_facts:
        # Simple substring check - could be enhanced with semantic similarity
        if fact.lower() in response_lower:
            found.append(fact)
        else:
            missing.append(fact)
    
    return found, missing


def check_hallucinations(response: str, forbidden: List[str]) -> List[str]:
    """Check for forbidden claims (hallucinations) in response."""
    found = []
    response_lower = response.lower()
    
    for claim in forbidden:
        if claim.lower() in response_lower:
            found.append(claim)
    
    return found


def check_refusal(response: str) -> bool:
    """Check if the response appropriately refuses to answer."""
    refusal_phrases = [
        "not found in context",
        "not found in the context",
        "context does not contain",
        "context doesn't contain",
        "cannot find",
        "not mentioned",
        "no information about",
        "not covered in",
    ]
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in refusal_phrases)


def evaluate_single(example: EvalExample, response: str) -> EvalResult:
    """Evaluate a single model response against an example."""
    result = EvalResult(example_id=example.id, predicted=response)
    
    # Citation extraction
    result.citations_found = extract_citations(response)
    
    # Refusal check
    refused = check_refusal(response)
    result.correctly_refused = (not example.answerable and refused) or (example.answerable and not refused)
    result.refusal_score = 1.0 if result.correctly_refused else 0.0
    
    # If should refuse but didn't, penalize heavily
    if not example.answerable:
        if refused:
            result.grounding_score = 1.0
            result.accuracy_score = 1.0
        else:
            result.grounding_score = 0.0
            result.accuracy_score = 0.0
        return result
    
    # Key facts check
    result.key_facts_found, result.key_facts_missing = check_key_facts(
        response, example.key_facts
    )
    if example.key_facts:
        result.accuracy_score = len(result.key_facts_found) / len(example.key_facts)
    else:
        result.accuracy_score = 1.0  # No key facts to check
    
    # Hallucination check
    result.hallucinations_found = check_hallucinations(response, example.forbidden_claims)
    if example.forbidden_claims:
        hallucination_penalty = len(result.hallucinations_found) / len(example.forbidden_claims)
        result.grounding_score = 1.0 - hallucination_penalty
    else:
        result.grounding_score = 1.0  # No forbidden claims to check
    
    # Citation check
    if example.expected_citations:
        matched = sum(
            1 for ec in example.expected_citations
            if any(ec in cf for cf in result.citations_found)
        )
        result.citation_score = matched / len(example.expected_citations)
    else:
        # Check if any citations present when they should be
        result.citation_score = 1.0 if result.citations_found else 0.5
    
    return result


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    total_examples: int = 0
    results: List[EvalResult] = field(default_factory=list)
    
    # Aggregate scores
    avg_accuracy: float = 0.0
    avg_grounding: float = 0.0
    avg_citation: float = 0.0
    avg_refusal: float = 0.0
    
    # Breakdown by task type
    by_task_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Breakdown by difficulty
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def compute_aggregates(self, examples: List[EvalExample]) -> None:
        """Compute aggregate statistics from individual results."""
        if not self.results:
            return
        
        self.total_examples = len(self.results)
        self.avg_accuracy = sum(r.accuracy_score for r in self.results) / self.total_examples
        self.avg_grounding = sum(r.grounding_score for r in self.results) / self.total_examples
        self.avg_citation = sum(r.citation_score for r in self.results) / self.total_examples
        self.avg_refusal = sum(r.refusal_score for r in self.results) / self.total_examples
        
        # Build lookup for examples
        example_map = {e.id: e for e in examples}
        
        # Breakdown by task type
        task_results = {}
        for result in self.results:
            example = example_map.get(result.example_id)
            if not example:
                continue
            task = example.task_type
            if task not in task_results:
                task_results[task] = []
            task_results[task].append(result)
        
        for task, results in task_results.items():
            n = len(results)
            self.by_task_type[task] = {
                "count": n,
                "accuracy": sum(r.accuracy_score for r in results) / n,
                "grounding": sum(r.grounding_score for r in results) / n,
                "citation": sum(r.citation_score for r in results) / n,
            }
        
        # Breakdown by difficulty
        diff_results = {}
        for result in self.results:
            example = example_map.get(result.example_id)
            if not example:
                continue
            diff = example.difficulty
            if diff not in diff_results:
                diff_results[diff] = []
            diff_results[diff].append(result)
        
        for diff, results in diff_results.items():
            n = len(results)
            self.by_difficulty[diff] = {
                "count": n,
                "accuracy": sum(r.accuracy_score for r in results) / n,
                "grounding": sum(r.grounding_score for r in results) / n,
            }

    def to_markdown(self) -> str:
        """Generate a markdown report of results."""
        lines = []
        lines.append("# RPG Evaluation Benchmark Results\n")
        
        lines.append("## Overall Scores\n")
        lines.append(f"- **Total Examples**: {self.total_examples}")
        lines.append(f"- **Accuracy**: {self.avg_accuracy:.1%}")
        lines.append(f"- **Grounding**: {self.avg_grounding:.1%}")
        lines.append(f"- **Citation**: {self.avg_citation:.1%}")
        lines.append(f"- **Refusal Calibration**: {self.avg_refusal:.1%}")
        lines.append("")
        
        if self.by_task_type:
            lines.append("## By Task Type\n")
            lines.append("| Task Type | Count | Accuracy | Grounding | Citation |")
            lines.append("|-----------|-------|----------|-----------|----------|")
            for task, scores in sorted(self.by_task_type.items()):
                lines.append(
                    f"| {task} | {scores['count']} | "
                    f"{scores['accuracy']:.1%} | {scores['grounding']:.1%} | "
                    f"{scores['citation']:.1%} |"
                )
            lines.append("")
        
        if self.by_difficulty:
            lines.append("## By Difficulty\n")
            lines.append("| Difficulty | Count | Accuracy | Grounding |")
            lines.append("|------------|-------|----------|-----------|")
            for diff, scores in sorted(self.by_difficulty.items()):
                lines.append(
                    f"| {diff} | {scores['count']} | "
                    f"{scores['accuracy']:.1%} | {scores['grounding']:.1%} |"
                )
            lines.append("")
        
        return "\n".join(lines)


def run_benchmark(
    eval_set_path: str,
    inference_fn,  # Callable[[str, str], str] - (question, context) -> response
    output_path: Optional[str] = None,
) -> BenchmarkResults:
    """
    Run the full evaluation benchmark.
    
    Args:
        eval_set_path: Path to evaluation examples (YAML or JSONL)
        inference_fn: Function that takes (question, context) and returns model response
        output_path: Optional path to save results
        
    Returns:
        BenchmarkResults with all scores and breakdowns
    """
    examples = load_eval_set(eval_set_path)
    print(f"Loaded {len(examples)} evaluation examples")
    
    results = BenchmarkResults()
    
    for example in examples:
        response = inference_fn(example.question, example.context)
        result = evaluate_single(example, response)
        results.results.append(result)
    
    results.compute_aggregates(examples)
    
    if output_path:
        report = results.to_markdown()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Results saved to: {output_path}")
    
    return results


# Example evaluation set template
EVAL_SET_TEMPLATE = """
# RPG Evaluation Set
# Add your evaluation examples below

examples:
  - id: "rules_basic_001"
    question: "What is the base accuracy for a standard attack?"
    context: |
      [Page 45]
      Standard attacks have a base accuracy of +0. This can be modified by
      talents, equipment, and situational bonuses.
    expected_answer: "Base accuracy is +0, modifiable by talents and equipment"
    task_type: "rules_qa"
    difficulty: "basic"
    answerable: true
    key_facts:
      - "+0"
      - "base accuracy"
    expected_citations:
      - "45"
    forbidden_claims:
      - "+1"
      - "+2"

  - id: "rules_unanswerable_001"
    question: "What is the maximum damage for a plasma cannon?"
    context: |
      [Page 50]
      Energy weapons deal damage based on their tier. Higher tier weapons
      have increased range and accuracy.
    expected_answer: "Not found in context"
    task_type: "rules_qa"
    difficulty: "intermediate"
    answerable: false
    key_facts: []
    expected_citations: []
    forbidden_claims:
      - "10 damage"
      - "20 damage"
"""


def create_eval_template(output_path: str) -> None:
    """Create a template evaluation set file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(EVAL_SET_TEMPLATE)
    print(f"Evaluation template created: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RPG Evaluation Benchmark")
    parser.add_argument("--create-template", help="Create evaluation set template at path")
    parser.add_argument("--eval-set", help="Path to evaluation set")
    parser.add_argument("--output", help="Path to save results")
    args = parser.parse_args()
    
    if args.create_template:
        create_eval_template(args.create_template)

