import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    The prompt includes:
      1. A task instruction and the four valid labels
      2. Labeled examples for the LLM to learn from
      3. A new episode description with an explicit output format request
    """
    instruction = (
        "You are classifying podcast episodes by their format. Classify the episode\n"
        "into exactly one of these four labels:\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — no guests,\n"
        "  no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often debating or\n"
        "  discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, archival\n"
        "  audio, reporting — with a clear narrative arc\n"
        "\n"
        "Return only the label and your reasoning. Do not explain the taxonomy.\n"
    )

    if labeled_examples:
        example_blocks = []
        for example in labeled_examples:
            title = example.get("title", "").strip()
            desc = example.get("description", "").strip()
            label = example.get("label", "").strip()
            example_blocks.append(
                f"Title: {title}\nDescription: {desc}\nLabel: {label}"
            )
        examples = "\n\n".join(example_blocks)
        examples_section = f"Here are some labeled examples:\n\n{examples}"
    else:
        examples_section = (
            "There are no labeled examples available. Classify the episode using only "
            "the episode description and the label definitions above."
        )

    prompt = (
        f"{instruction}\n"
        f"{examples_section}\n\n"
        f"Episode to classify:\n"
        f"Description: {description.strip()}\n"
        f"Label: ?\n\n"
        f"Classify the episode above. Return your answer in the format below:\n"
        f"Label: <one of interview, solo, panel, narrative>\n"
        f"Reasoning: <brief explanation>"
    )
    return prompt


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    """
    prompt = build_few_shot_prompt(labeled_examples, description)

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        response_text = response.choices[0].message.content
    except Exception:
        return {
            "label": "unknown",
            "reasoning": "Unable to classify episode due to an error or unexpected response.",
        }

    label = None
    reasoning = None

    for line in response_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("label:"):
            label_text = stripped[len("label:"):].strip().strip('"').strip("'")
            if label_text:
                label = label_text
        elif stripped.lower().startswith("reasoning:"):
            reasoning = stripped[len("reasoning:"):].strip()

    if reasoning is None:
        reasoning = response_text.strip()

    if label not in VALID_LABELS:
        label = "unknown"

    if not reasoning:
        reasoning = "No reasoning provided."

    print(f"LLM response:\n{response_text}\nParsed label: {label}\nParsed reasoning: {reasoning}\n")
    return {
        "label": label,
        "reasoning": reasoning,
    }
