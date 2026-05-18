"""Simple chat with abliterated Qwen 3 — no CASCADES, no adapters, just raw model."""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = r"E:\code.projects\CASCADES--continual-PEFT-for-Local-LLMs\abliteratedqwen3"

print("Loading abliterated Qwen 3...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded: {MODEL_PATH}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("=" * 50)
print("Chat ready. Type your message (Ctrl+C to quit).")
print("=" * 50)

history = []

while True:
    try:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ('/quit', '/exit', 'quit', 'exit'):
            print("Bye!")
            break

        history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        print(f"\nQwen: {response}")
        history.append({"role": "assistant", "content": response})

        # Keep history manageable
        if len(history) > 20:
            history = history[-16:]

    except KeyboardInterrupt:
        print("\n\nBye!")
        break
    except Exception as e:
        print(f"\nError: {e}")
