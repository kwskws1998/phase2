"""
Standalone preprocessing script for agent training data.

Converts raw agent results (train_biological_casuality_*_results_*.json)
into training-ready format with:
  - Tag conversion: <think> -> [THINK], <execute> -> [EXECUTE], etc.
  - Role mapping:   system -> system, human -> user, LLM -> assistant, Result -> tool
  - Tool retrieval: tool_retrieval -> system (content from template),
                    Result after tool_retrieval -> assistant (LLM's tool selection)
  - Bad message filtering: removes [LOOP TRUNCATED] and System Alert messages
  - Error filtering: removes instances with status="error", empty final_answer,
                     or empty LLM content turns
  - Phase splitting: multi-phase conversations (multiple system prompts) are
                     split into separate training instances
  - Output:         JSON array with {"messages": [...]} per instance

Usage:
    python data_formatting/format_agent_data.py input1.json [input2.json ...] -o output.json

    # With tool retrieval template:
    python data_formatting/format_agent_data.py results.json -o output.json \\
        --tool-retrieval-template tool_retrieval_sys_prompt.txt \\
        --source-data train_biological_causality_1000_no_pharma.json
"""

import argparse
import json
import os
import re
import sys


TAG_MAP = {
    "<think>":          "[THINK]",
    "</think>":         "[/THINK]",
    "<execute>":        "[EXECUTE]",
    "</execute>":       "[/EXECUTE]",
    "<observation>":    "[OBSERVATION]",
    "</observation>":   "[/OBSERVATION]",
    "<solution>":       "[SOLUTION]",
    "</solution>":      "[/SOLUTION]",
}

TAG_PATTERN = re.compile(
    "|".join(re.escape(k) for k in TAG_MAP),
    flags=re.IGNORECASE,
)

ROLE_MAP = {
    "system":          "system",
    "human":           "user",
    "LLM":             "assistant",
    "Result":          "tool",
    "tool_retrieval":  "system",
}

FILTER_PATTERNS = [
    "[LOOP TRUNCATED]",
    "System Alert: Infinite loop detected",
]


def convert_tags(text: str) -> str:
    """Replace agent-specific XML tags with Ministral bracket tokens."""
    return TAG_PATTERN.sub(lambda m: TAG_MAP[m.group(0).lower()], text)


def should_skip_message(content: str) -> bool:
    """Check if a message contains patterns that should be filtered out."""
    return any(pattern in content for pattern in FILTER_PATTERNS)


def split_by_system_prompt(messages: list[dict]) -> list[list[dict]]:
    """Split messages into phases whenever a new system prompt appears.

    Each phase starts with a system message and contains the subsequent
    user/assistant/tool messages until the next system message.
    System prompts do NOT accumulate across phases.

    Returns:
        List of message lists, one per phase.
    """
    phases = []
    current_phase = []

    for msg in messages:
        if msg["role"] == "system" and current_phase:
            phases.append(current_phase)
            current_phase = [msg]
        else:
            current_phase.append(msg)

    if current_phase:
        phases.append(current_phase)

    return phases


def convert_instance(
    instance: dict,
    tr_template: str | None = None,
    source_prompts: dict | None = None,
) -> list[dict] | None:
    """Convert a single raw instance to training format.

    If the conversation contains multiple system prompts (multi-phase),
    each phase is split into a separate training instance.

    Args:
        instance: Raw agent result dict.
        tr_template: Tool retrieval system prompt template (optional).
        source_prompts: {instance_id: prompt} mapping for USER QUERY (optional).

    Returns list of converted instances, or None if filtered out.
    """
    if instance.get("status") == "error":
        return None

    final_answer = instance.get("final_answer", "")
    if not final_answer or not final_answer.strip():
        return None

    raw_messages = instance.get("messages", [])
    if not raw_messages:
        return None

    instance_id = instance.get("instance_id")

    # Separate tool_retrieval phase from the rest.
    # Raw message order: [system] → [tool_retrieval] → [Result] → [human] → [LLM] → ...
    # We extract tool_retrieval + its Result as Phase 0,
    # then process remaining messages as Phase 1+.
    tool_retrieval_phase = []  # Phase 0: system(tool_retrieval) + assistant(Result)
    remaining_messages = []    # Phase 1+: system(agent) + user + assistant + tool + ...

    prev_raw_type = None
    for msg in raw_messages:
        raw_type = msg.get("type", "")
        role = ROLE_MAP.get(raw_type)
        if role is None:
            prev_raw_type = raw_type
            continue

        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        # Filter out bad messages
        if should_skip_message(content):
            prev_raw_type = raw_type
            continue

        # tool_retrieval → Phase 0 system prompt
        if raw_type == "tool_retrieval":
            if tr_template and source_prompts and instance_id in source_prompts:
                content = tr_template.replace("{User prompt}", source_prompts[instance_id])
            content = convert_tags(content)
            tool_retrieval_phase.append({"role": "system", "content": content})
            prev_raw_type = raw_type
            continue

        # Result right after tool_retrieval → Phase 0 assistant
        if raw_type == "Result" and prev_raw_type == "tool_retrieval":
            content = convert_tags(content)
            tool_retrieval_phase.append({"role": "assistant", "content": content})
            prev_raw_type = raw_type
            continue

        # Everything else → remaining messages
        if role == "assistant" and not content.strip():
            prev_raw_type = raw_type
            continue

        content = convert_tags(content)
        remaining_messages.append({"role": role, "content": content})
        prev_raw_type = raw_type

    # Build results: Phase 0 (tool retrieval) + Phase 1+ (agent reasoning)
    results = []
    task_instance_id = instance.get("task_instance_id")

    if len(tool_retrieval_phase) >= 2:
        results.append({
            "instance_id": instance_id,
            "task_instance_id": task_instance_id,
            "phase": 0,
            "messages": tool_retrieval_phase,
        })

    # Split remaining by system prompt (in case of further phase changes)
    if len(remaining_messages) >= 2:
        phases = split_by_system_prompt(remaining_messages)
        for i, phase_messages in enumerate(phases):
            if len(phase_messages) < 2:
                continue
            results.append({
                "instance_id": instance_id,
                "task_instance_id": task_instance_id,
                "phase": len(results),
                "messages": phase_messages,
            })

    return results if results else None


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw agent results into training-ready format."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Input JSON file(s) (raw agent results)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--tool-retrieval-template", type=str, default=None,
        help="Path to tool_retrieval_sys_prompt.txt template file",
    )
    parser.add_argument(
        "--source-data", type=str, default=None,
        help="Path to source data JSON (for USER QUERY lookup by instance_id)",
    )
    args = parser.parse_args()

    # Load tool retrieval template
    tr_template = None
    if args.tool_retrieval_template:
        with open(args.tool_retrieval_template, "r", encoding="utf-8") as f:
            tr_template = f.read()
        print(f"Loaded tool retrieval template: {args.tool_retrieval_template}")

    # Load source data for USER QUERY mapping
    source_prompts = {}
    if args.source_data:
        with open(args.source_data, "r", encoding="utf-8") as f:
            for item in json.load(f):
                source_prompts[item["instance_id"]] = item["prompt"]
        print(f"Loaded {len(source_prompts)} source prompts from {args.source_data}")

    all_instances = []
    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"Warning: file not found, skipping: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        all_instances.extend(data)

    print(f"Loaded {len(all_instances)} raw instances from {len(args.inputs)} file(s)")

    converted = []
    skipped = 0
    multi_phase_count = 0
    for inst in all_instances:
        result = convert_instance(inst, tr_template, source_prompts)
        if result is not None:
            converted.extend(result)
            if len(result) > 1:
                multi_phase_count += 1
        else:
            skipped += 1

    print(f"Converted: {len(converted)} instances, Filtered out: {skipped}")
    if multi_phase_count:
        print(f"  (includes {multi_phase_count} multi-phase conversations split into separate instances)")

    if not converted:
        print("No instances remaining after filtering. Exiting.")
        sys.exit(1)

    role_counts = {}
    tag_counts = {}
    for item in converted:
        for msg in item["messages"]:
            role_counts[msg["role"]] = role_counts.get(msg["role"], 0) + 1
            for tag in ["[THINK]", "[EXECUTE]", "[OBSERVATION]", "[SOLUTION]"]:
                if tag in msg["content"]:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print(f"\nRole distribution: {role_counts}")
    print(f"Tag occurrences:   {tag_counts}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {args.output} ({os.path.getsize(args.output):,} bytes)")


if __name__ == "__main__":
    main()
