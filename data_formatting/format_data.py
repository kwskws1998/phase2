import json
import re
import sys
import os


def _extract_names(items):
    """Extract names from a list of dicts or strings."""
    if not isinstance(items, list):
        return []
    names = []
    for item in items:
        if isinstance(item, dict):
            names.append(item.get("name", "?"))
        elif isinstance(item, str):
            names.append(item)
        else:
            names.append(str(item))
    return names


def format_env_setup(obj):
    """Format environment setup JSON as a compact summary."""
    parts = []
    if "tools" in obj:
        names = _extract_names(obj["tools"])
        parts.append(f"Tools: [{', '.join(names)}]" if names else "Tools: []")
    if "data_lake" in obj:
        names = _extract_names(obj["data_lake"])
        parts.append(f"Data: [{', '.join(names)}]" if names else "Data: []")
    if "libraries" in obj:
        names = _extract_names(obj["libraries"])
        parts.append(f"Libraries: [{', '.join(names)}]" if names else "Libraries: []")
    if "know_how" in obj:
        parts.append(f"KnowHow: {obj['know_how']}")
    return ", ".join(parts) if parts else json.dumps(obj, ensure_ascii=False)


def try_parse_json_prefix(text):
    """Try to parse a JSON object from the start of text."""
    text = text.lstrip()
    if not text.startswith("{") and not text.startswith("["):
        return None, text
    decoder = json.JSONDecoder()
    try:
        obj, end_idx = decoder.raw_decode(text)
        remaining = text[end_idx:].strip()
        return obj, remaining
    except json.JSONDecodeError:
        return None, text


def format_conv_message(msg):
    """Format a single message from a conversation history as a chat log entry."""
    role = msg.get("type", "?")
    content = msg.get("content", "")
    if not isinstance(content, str):
        content = str(content)

    if content.lstrip().startswith("<observation>"):
        obs_match = re.match(r"\s*<observation>(.*?)</observation>", content, re.DOTALL)
        if obs_match:
            obs_content = obs_match.group(1).strip()
            if obs_content:
                return f"[Tool Result]\n{obs_content}"
            return "[Tool Result] (empty)"
        return f"[Tool Result]\n{content}"

    if role == "human":
        return f"[Human]\n{content}"
    elif role == "ai":
        return f"[AI]\n{content}"
    else:
        return f"[{role}]\n{content}"


def parse_ai_content(content):
    """Parse AI content by extracting unique conversation turns.

    Uses the largest Conversation History block (most complete) as the single
    source of truth, skipping all duplicate trace artifacts.
    """
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if not think_match:
        return content

    think_content = think_match.group(1)
    segments = re.split(r"\nObservation:\s*\n?", think_content)

    parts = []
    all_conv_histories = []

    for i, seg in enumerate(segments):
        seg = seg.strip()
        if not seg:
            continue
        if i == 0:
            continue
        if seg == "end" or seg.startswith("end\n"):
            continue

        parsed, remaining = try_parse_json_prefix(seg)
        if parsed is None:
            continue

        if "tools" in parsed or "data_lake" in parsed or "libraries" in parsed:
            parts.append(f"[Environment] {format_env_setup(parsed)}")
        elif "messages" in parsed:
            all_conv_histories.append(parsed["messages"])

    if all_conv_histories:
        largest = max(all_conv_histories, key=len)
        first_human = True
        for msg in largest:
            role = msg.get("type", "?")
            if role == "human" and first_human:
                first_human = False
                continue
            formatted = format_conv_message(msg)
            if formatted:
                parts.append(formatted)

    return "\n\n".join(parts) if parts else content


def format_instance(instance, is_first=True):
    """Format a single instance's messages for readability."""
    formatted = {}
    for key in instance:
        if key == "messages":
            continue
        formatted[key] = instance[key]

    formatted_messages = []
    for msg in instance.get("messages", []):
        msg_type = msg.get("type", "unknown")
        content = msg.get("content", "")

        if msg_type == "system":
            if is_first:
                formatted_messages.append({"role": "system", "content": content})
            else:
                formatted_messages.append({
                    "role": "system",
                    "content": "[System Prompt - same as instance 0]"
                })
        elif msg_type == "human":
            formatted_messages.append({"role": "human", "content": content})
        elif msg_type == "ai":
            formatted_messages.append({
                "role": "ai",
                "content": parse_ai_content(content)
            })
        else:
            formatted_messages.append({"role": msg_type, "content": content})

    formatted["messages"] = formatted_messages
    return formatted


def main():
    if len(sys.argv) < 2:
        print("Usage: python format_data.py <filename.json>")
        print("  Searches: CWD, project root, project root/data/")
        print("  Output:   data_formatting/<filename>_formatted.json")
        sys.exit(1)

    filename = sys.argv[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    search_paths = [
        filename,
        os.path.join(project_root, filename),
        os.path.join(project_root, "data", filename),
        os.path.join("..", filename),
        os.path.join("..", "data", filename),
    ]

    input_path = None
    for p in search_paths:
        if os.path.exists(p):
            input_path = p
            break

    if input_path is None:
        tried = "\n  ".join(search_paths)
        print(f"Error: File '{filename}' not found. Searched:\n  {tried}")
        sys.exit(1)

    print(f"Loading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    print(f"Processing {len(data)} instances...")

    formatted_data = []
    for i, instance in enumerate(data):
        formatted = format_instance(instance, is_first=(i == 0))
        formatted_data.append(formatted)
        inst_id = instance.get("instance_id", i)
        ai_content = formatted.get("messages", [{}])[-1].get("content", "")
        has_exec = "<execute>" in ai_content if isinstance(ai_content, str) else False
        status = "with tools" if has_exec else "direct answer"
        print(f"  Instance {inst_id}: {status}")

    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_path = os.path.join(script_dir, f"{base_name}_formatted.json")

    raw_json = json.dumps(formatted_data, indent=2, ensure_ascii=False)
    readable = raw_json.replace("\\n", "\n").replace("\\t", "\t")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(readable)

    print(f"\nSaved: {output_path}")
    print(f"Output size: {os.path.getsize(output_path):,} bytes")


if __name__ == "__main__":
    main()
