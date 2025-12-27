"""
Configurable prompt templates for synthetic data generation.

This module provides default prompts that can be overridden via YAML config.
"""

from typing import Optional

# Default prompt components - these can be overridden in config
DEFAULT_SYSTEM_PERSONA = (
    "You are an expert Game Master and Rules Lawyer for {topic}. "
    "Your goal is to create high-quality, logically consistent training data for a new AI model."
)

DEFAULT_CONTEXT_INTRO = "Read the following text from the {topic} source material:"

DEFAULT_QA_TEMPLATE = """
{system_persona}

### Context
{context_intro}
{text}

### Task
Generate {n_questions} training examples based on the text above.
Each example must be a pair of "instruction" (a user question or prompt) and "output" (the ideal response).

### Task Type
The task type for this generation is: {task_type}
Choose prompts and answers that match this task type.

{extra_instructions}

### Requirements
1. **Variety**: Create a mix of question styles appropriate for {task_type}.
2. **Grounding**: All answers must be directly supported by the provided context.
3. **Reasoning**: Think step-by-step internally, but do not include reasoning in the output.

### Format
Output MUST be a valid JSON object with an "examples" key containing a list of objects. Each object must have:
- `instruction`: The user prompt.
- `output`: The correct, high-quality answer.
- `task_type`: One of: {task_types}

### Output Format
{{
  "examples": [
    {{
      "instruction": "...",
      "output": "...",
      "task_type": "{task_type}"
    }}
  ]
}}

Do not include any markdown formatting (like ```json) outside the response. Return JSON only.
"""

DEFAULT_COVERAGE_TEMPLATE = """
You are an expert RPG rules compiler for {topic}. Your job is to add coverage for details that are easy to miss.

### Context
{text}

### Coverage Task
Generate up to {n_questions} additional training examples that emphasize:
- named abilities, items, stats, modifiers, or keywords
- numeric thresholds, prerequisites, exceptions, or limits
- definitions of terms or subsystems

Avoid generic summaries. Focus on precise, testable Q/A that can be answered from the text.
Use the task type: {task_type}
Allowed task types: {task_types}

{grounding_instructions}

{format_instructions}

### Format
Output MUST be a valid JSON object with an "examples" key containing a list of objects. Each object must have:
- `instruction`: The user prompt.
- `output`: The correct, high-quality answer.
- `task_type`: One of: {task_types}
"""

DEFAULT_NEGATIVE_TEMPLATE = """
You are an expert at identifying knowledge boundaries for {topic}.

### Context
{text}

### Task
Generate {n_questions} questions that CANNOT be fully answered from the provided context alone.

These should be:
- Plausible questions a user might ask about {topic}
- Questions that require information NOT present in the provided text
- Related to the general subject matter but beyond what's documented

### Output Requirements
For each question, provide:
- `instruction`: The unanswerable question
- `output`: A response that:
  1. Acknowledges the context doesn't contain the answer
  2. States what IS known from the context (if anything related)
  3. Asks ONE specific clarifying question to help the user
- `task_type`: "{task_type}"

### Output Format
{{
  "examples": [
    {{
      "instruction": "What is the maximum range of the Leviathan-class weapon?",
      "output": "Not found in context. The provided text discusses mech classifications but doesn't include specific weapon statistics for Leviathan-class systems. The context does mention that weapon ranges vary by tier. Could you specify which weapon system you're interested in, or would you like me to explain the general range categories mentioned?",
      "task_type": "{task_type}"
    }}
  ]
}}

Do not include any markdown formatting. Return JSON only.
"""

DEFAULT_WALKTHROUGH_TEMPLATE = """
You are an expert guide helping players learn {topic}. Your goal is to create step-by-step walkthrough conversations.

### Context (Source Material)
{text}

### Walkthrough Task
Generate a {n_turns}-turn guided conversation where a player asks how to do something, and the assistant walks them through it step by step.

### Walkthrough Topic
{walkthrough_topic}

### Conversation Structure
- **Turn 1**: Player asks how to start or what to do first
- **Turn 2**: Assistant explains the first steps with specific details from the context
- **Turn 3+**: Player asks follow-up questions, assistant continues guiding them through the process
- **Final turn**: Assistant provides a summary or "next steps" to complete the process

### Requirements
1. Each response must be grounded in the provided context
2. Responses should be instructional and encouraging
3. Include specific details like page references, stat names, or step numbers
4. The conversation should feel like a patient teacher guiding a new player
5. If the full process isn't covered in context, acknowledge what's missing and focus on what IS there

### Types of Follow-ups to Include
- "What do I do next?"
- "Can you explain [specific term] more?"
- "What are my options for [choice point]?"
- "Is there anything I should watch out for?"
- "How do I decide between X and Y?"

{grounding_instructions}

{format_instructions}

### Output Format
Return a valid JSON object:
{{
  "messages": [
    {{"role": "user", "content": "How do I create a character?"}},
    {{"role": "assistant", "content": "Great question! Let's start with... (p. X)"}},
    {{"role": "user", "content": "What's next after that?"}},
    {{"role": "assistant", "content": "Now you'll need to... (p. Y)"}}
  ],
  "topic_summary": "walkthrough for {walkthrough_topic}",
  "task_type": "character_build"
}}

Do not include markdown formatting. Return JSON only.
"""

DEFAULT_VERIFICATION_TEMPLATE = """
You are a quality assurance expert for {topic} training data.

### Task
Evaluate the following Q/A pair for accuracy and groundedness.

### Context (Source Material)
{context}

### Q/A Pair to Evaluate
Question: {question}
Answer: {answer}

### Evaluation Criteria
Rate the answer on a 1-5 scale:
- **5**: Perfectly grounded, complete, accurate, well-formatted
- **4**: Mostly accurate with minor omissions or style issues
- **3**: Partially correct but missing key details or has minor errors
- **2**: Contains factual errors or significant hallucinations
- **1**: Completely wrong, ungrounded, or harmful

### Output Format
{{
  "score": N,
  "issues": ["list of specific issues found, empty if score >= 4"],
  "corrected_answer": "improved answer if score < 4, otherwise null"
}}

Be strict about grounding. If the answer contains ANY information not supported by the context, score ≤ 3.
Return JSON only.
"""


class PromptConfig:
    """Holds prompt configuration, merging defaults with YAML overrides."""

    def __init__(self, config: Optional[dict] = None, topic: str = "the RPG system"):
        self.topic = topic
        prompts_config = (config or {}).get("prompts", {}) or {}

        # Load overrides or use defaults
        self.system_persona = prompts_config.get(
            "system_persona", DEFAULT_SYSTEM_PERSONA
        )
        self.context_intro = prompts_config.get("context_intro", DEFAULT_CONTEXT_INTRO)
        self.qa_template = prompts_config.get("qa_template", DEFAULT_QA_TEMPLATE)
        self.coverage_template = prompts_config.get(
            "coverage_template", DEFAULT_COVERAGE_TEMPLATE
        )
        self.negative_template = prompts_config.get(
            "negative_template", DEFAULT_NEGATIVE_TEMPLATE
        )
        self.verification_template = prompts_config.get(
            "verification_template", DEFAULT_VERIFICATION_TEMPLATE
        )
        self.walkthrough_template = prompts_config.get(
            "walkthrough_template", DEFAULT_WALKTHROUGH_TEMPLATE
        )

    def format_system_persona(self) -> str:
        return self.system_persona.format(topic=self.topic)

    def format_context_intro(self) -> str:
        return self.context_intro.format(topic=self.topic)

    def format_qa_prompt(
        self,
        text: str,
        n_questions: int,
        task_type: str,
        task_types: list[str],
        extra_instructions: str = "",
    ) -> str:
        return self.qa_template.format(
            system_persona=self.format_system_persona(),
            context_intro=self.format_context_intro(),
            text=text,
            n_questions=n_questions,
            task_type=task_type,
            task_types=", ".join(task_types),
            extra_instructions=extra_instructions,
            topic=self.topic,
        )

    def format_coverage_prompt(
        self,
        text: str,
        n_questions: int,
        task_type: str,
        task_types: list[str],
        grounding_instructions: str = "",
        format_instructions: str = "",
    ) -> str:
        return self.coverage_template.format(
            text=text,
            n_questions=n_questions,
            task_type=task_type,
            task_types=", ".join(task_types),
            grounding_instructions=grounding_instructions,
            format_instructions=format_instructions,
            topic=self.topic,
        )

    def format_negative_prompt(
        self,
        text: str,
        n_questions: int,
        task_type: str,
    ) -> str:
        return self.negative_template.format(
            text=text,
            n_questions=n_questions,
            task_type=task_type,
            topic=self.topic,
        )

    def format_verification_prompt(
        self,
        context: str,
        question: str,
        answer: str,
    ) -> str:
        return self.verification_template.format(
            context=context,
            question=question,
            answer=answer,
            topic=self.topic,
        )

    def format_walkthrough_prompt(
        self,
        text: str,
        walkthrough_topic: str,
        n_turns: int = 3,
        grounding_instructions: str = "",
        format_instructions: str = "",
    ) -> str:
        return self.walkthrough_template.format(
            text=text,
            walkthrough_topic=walkthrough_topic,
            n_turns=n_turns,
            grounding_instructions=grounding_instructions,
            format_instructions=format_instructions,
            topic=self.topic,
        )
