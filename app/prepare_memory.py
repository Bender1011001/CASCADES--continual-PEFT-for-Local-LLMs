"""
Prepare conversation logs for CASCADES sleep training.

Reads exported conversation JSONL and formats it as CASCADES training data
with ChatML templates and <think> tags for chain-of-thought.
"""

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful personal AI assistant with persistent memory. "
    "You remember details about the user from previous conversations. "
    "Think carefully before responding."
)


def prepare_training_data(
    input_path: Path,
    output_path: Path,
    include_system_prompt: bool = True,
    max_examples_per_conversation: int = 20,
) -> int:
    """Convert exported conversation JSONL to CASCADES training format.

    Input format (from ConversationStore.export_training_data):
        {"prompt": "...", "response": "...", "conversation_id": "...", "flagged": bool, "timestamp": "..."}

    Output format (for train.py):
        {"prompt": "<|im_start|>system\\n...\\nuser\\n...\\n", "response": "<think>...</think>..."}
    """
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    # Sort: flagged first, then by timestamp
    examples.sort(key=lambda x: (not x.get("flagged", False), x.get("timestamp", "")))

    count = 0
    # Group by conversation to maintain context
    conv_groups: dict[str, list] = {}
    for ex in examples:
        cid = ex.get("conversation_id", "unknown")
        if cid not in conv_groups:
            conv_groups[cid] = []
        conv_groups[cid].append(ex)

    with open(output_path, "w", encoding="utf-8") as f:
        for cid, conv_examples in conv_groups.items():
            history = []
            for i, ex in enumerate(conv_examples[:max_examples_per_conversation]):
                # Build accumulated context
                if include_system_prompt:
                    prompt_parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"]
                else:
                    prompt_parts = []

                # Include conversation history for context
                for prev in history:
                    prompt_parts.append(
                        f"<|im_start|>user\n{prev['prompt']}<|im_end|>"
                    )
                    prompt_parts.append(
                        f"<|im_start|>assistant\n{prev['response']}<|im_end|>"
                    )

                # Current turn
                prompt_parts.append(
                    f"<|im_start|>user\n{ex['prompt']}<|im_end|>"
                )
                prompt_parts.append("<|im_start|>assistant\n")

                full_prompt = "\n".join(prompt_parts)

                # Wrap response with think tags if not already present
                response = ex["response"]
                if "<think>" not in response:
                    response = f"<think>Recalling context from our conversation.</think>{response}"

                training_example = {
                    "prompt": full_prompt,
                    "response": response,
                    "flagged": ex.get("flagged", False),
                    "source_conversation": cid,
                }

                f.write(json.dumps(training_example, ensure_ascii=False) + "\n")
                count += 1

                history.append(ex)

    print(f"Prepared {count} training examples from {len(conv_groups)} conversations")
    print(f"  Flagged: {sum(1 for e in examples if e.get('flagged'))}")
    print(f"  Output: {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare conversations for CASCADES training")
    parser.add_argument(
        "--input", type=str, default="app/data/training_export.jsonl",
        help="Input JSONL from ConversationStore export",
    )
    parser.add_argument(
        "--output", type=str, default="app/data/cascades_memory_train.jsonl",
        help="Output JSONL for CASCADES train.py",
    )
    parser.add_argument(
        "--no-system-prompt", action="store_true",
        help="Omit system prompt from training examples",
    )
    args = parser.parse_args()

    prepare_training_data(
        Path(args.input),
        Path(args.output),
        include_system_prompt=not args.no_system_prompt,
    )


if __name__ == "__main__":
    main()
