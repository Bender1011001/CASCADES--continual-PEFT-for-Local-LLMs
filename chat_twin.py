#!/usr/bin/env python3
"""
Chat with your CASCADES-trained digital twin model.

Loads the base Qwen3-4B + CASCADES adapters + trained weights,
then runs an interactive chat. No RAG, no tools — pure weights.

Usage:
    python chat_twin.py
    python chat_twin.py --question "Who is Bender1011001?"
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import torch
from pathlib import Path

MODEL_ID = "p-e-w/Qwen3-4B-Instruct-2507-heretic"
WEIGHTS_PATH = Path(r"e:\code.projects\CASCADES--continual-PEFT-for-Local-LLMs\cascades_v10_twin_weights.pt")


def load_model(weights_path: Path, rank: int = 8):
    """Load base model + CASCADES adapters + trained weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    from cascades.config import DEFAULT_CONFIG
    from cascades.injection import inject_cascades, estimate_quant_noise

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = True

    # Inject CASCADES adapters
    print("Injecting CASCADES adapters...")
    critical_adapters, funlora_adapters = inject_cascades(
        model, rank=rank, layer_importance=None, config=DEFAULT_CONFIG,
    )

    # Load trained weights
    if weights_path.exists():
        print(f"Loading trained weights from {weights_path.name}...")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model_state = model.state_dict()
        loaded = 0
        skipped = 0
        for key, val in state.items():
            if key in model_state and model_state[key].shape == val.shape:
                model_state[key] = val
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(model_state, strict=False)
        print(f"  Loaded {loaded} tensors, skipped {skipped}")
    else:
        print(f"WARNING: No weights found at {weights_path}")

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_response(model, tokenizer, user_message: str, history: list = None,
                      max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate a response from the model."""
    if history is None:
        history = []

    messages = list(history) + [{"role": "user", "content": user_message}]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Chat with CASCADES Digital Twin")
    parser.add_argument("--question", type=str, help="Single question mode")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH))
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--batch-test", action="store_true", help="Run identity test battery")
    args = parser.parse_args()

    model, tokenizer = load_model(Path(args.weights), rank=args.rank)
    print("\n✅ Model loaded and ready!\n")

    # Batch test mode — run a set of identity questions
    if args.batch_test:
        test_questions = [
            "Who is Bender1011001?",
            "What is Bender1011001's real email address?",
            "What projects does Bender1011001 work on?",
            "What programming languages does Bender1011001 use?",
            "What hardware does Bender1011001 use for AI training?",
            "What is CASCADES?",
            "Tell me about Bender1011001's interests and hobbies.",
            "What websites does Bender1011001 visit frequently?",
        ]
        print("=" * 60)
        print("IDENTITY TEST BATTERY")
        print("=" * 60)
        for q in test_questions:
            print(f"\n🧑 {q}")
            response = generate_response(model, tokenizer, q, max_new_tokens=300)
            print(f"🤖 {response}\n")
            print("-" * 40)
        return

    # Single question mode
    if args.question:
        response = generate_response(model, tokenizer, args.question)
        print(f"🤖 {response}")
        return

    # Interactive chat mode
    print("=" * 60)
    print("CASCADES Digital Twin Chat")
    print("Type 'quit' to exit, 'clear' to reset history")
    print("=" * 60)

    history = []
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            history = []
            print("History cleared.")
            continue

        response = generate_response(model, tokenizer, user_input, history)
        print(f"\n🤖 Twin: {response}")

        # Add to history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
