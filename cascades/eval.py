"""
CASCADES Generative Evaluation Module — Structured answer extraction and flexible matching.

This module bridges the gap between high proxy accuracy (exp(-loss)) and exact match (EM)
by implementing:
  1. Structured inference prompting that instructs the model to use <think>...</think> format
  2. Answer extraction from both structured CoT and conversational free-form outputs
  3. Multiple matching strategies (exact, normalized, semantic containment)
"""

import re
import math
import json
import unicodedata
from pathlib import Path

import torch
import pandas as pd


# ---------------------------------------------------------------------------
# A. Answer extraction — handles both structured and free-form outputs
# ---------------------------------------------------------------------------

def extract_answer_from_cot(text: str) -> str:
    """Extract the final answer from a <think>...</think>\\n\\n<answer> formatted response."""
    
    # Pre-processing: Strip rogue tool calls to prevent parsing errors
    text = re.sub(r'<tool_call>.*?(?:</tool_call>|$)', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Strategy 1: Look for content after </think> tag
    think_end_pattern = re.compile(r'</think>\s*\n*\s*(.*)', re.DOTALL | re.IGNORECASE)
    match = think_end_pattern.search(text)
    if match:
        answer = match.group(1).strip()
        if answer:
            return answer

    # Strategy 2: <answer>...</answer> XML tags
    answer_tag_pattern = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL | re.IGNORECASE)
    match = answer_tag_pattern.search(text)
    if match:
        answer = match.group(1).strip()
        if answer:
            return answer

    # Strategy 3: \boxed{...} LaTeX format (common in math models)
    boxed_pattern = re.compile(r'\\boxed\{([^}]*)\}', re.DOTALL)
    match = boxed_pattern.search(text)
    if match:
        answer = match.group(1).strip()
        if answer:
            return answer

    # Strategy 4: **Answer:** or **Final Answer:** markdown bold formatting
    bold_answer_pattern = re.compile(
        r'\*\*(?:final\s+)?answer:?\*\*\s*(.*)',
        re.DOTALL | re.IGNORECASE
    )
    match = bold_answer_pattern.search(text)
    if match:
        answer = match.group(1).strip()
        answer = answer.split('\n')[0].strip()
        answer = answer.rstrip('.')
        if answer:
            return answer

    # Strategy 5: If the model used a "Final Answer:" or "Answer:" prefix
    answer_prefix_pattern = re.compile(
        r'(?:final\s+answer|the\s+answer|answer)\s*(?:is|:)\s*(.*)',
        re.DOTALL | re.IGNORECASE
    )
    match = answer_prefix_pattern.search(text)
    if match:
        answer = match.group(1).strip()
        answer = answer.split('\n')[0].strip()
        answer = answer.rstrip('.')
        if answer:
            return answer

    # Strategy 6: Take the last non-empty line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        last_line = last_line.rstrip('.')
        
        # Check if the last line is a raw MCQ answer like "Answer: B"
        mcq_match = re.search(r'(?:answer|option)\s*(?:is|:)?\s*([a-e])(?:\.|$|\s)', last_line, re.IGNORECASE)
        if mcq_match:
            return mcq_match.group(1).upper()
            
        return last_line

    return text.strip()


def normalize_latex(text: str) -> str:
    """Normalize LaTeX/math notation to plain text equivalents."""
    text = text.replace('$', '')
    text = re.sub(r'\\\[|\\\]', '', text)
    text = text.replace('\u0398', 'O').replace('\u03b8', 'O').replace('\u03a9', 'O')
    text = re.sub(r'\\(?:Theta|theta|Omega|omega)\b', 'O', text)
    text = re.sub(r'\\(?:left|right)\s*', '', text)

    latex_to_plain = [
        (r'\\log\b', 'log'), (r'\\ln\b', 'ln'), (r'\\sqrt\{([^}]*)\}', r'sqrt(\1)'),
        (r'\\sqrt\b', 'sqrt'), (r'\\cdot\b', '*'), (r'\\times\b', '*'),
        (r'\\pm\b', '+-'), (r'\\geq\b', '>='), (r'\\leq\b', '<='),
        (r'\\neq\b', '!='), (r'\\approx\b', '~'), (r'\\infty\b', 'infinity'),
        (r'\\sum\b', 'sum'), (r'\\prod\b', 'prod'), (r'\\pi\b', 'pi'),
    ]
    for pattern, replacement in latex_to_plain:
        text = re.sub(pattern, replacement, text)

    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', text)
    text = re.sub(r'\\(?:text|mathrm|mathcal|mathbb|operatorname)\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)
    text = text.replace('{', '').replace('}', '')
    text = text.replace('\\\\', '\\')
    return text


def normalize_answer(text: str) -> str:
    """Normalize an answer string for flexible comparison."""
    if not text:
        return ""

    text = unicodedata.normalize('NFKD', text)
    text = normalize_latex(text)
    text = text.lower()

    removals = [
        r'^the\s+answer\s+is\s+', r'^therefore,?\s+', r'^thus,?\s+',
        r'^so,?\s+', r'^hence,?\s+', r'^we\s+(?:get|have|obtain|find)\s+',
        r'^this\s+(?:gives|yields|results?\s+in)\s+',
    ]
    for pattern in removals:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip('"\'`')
    text = text.rstrip('.,;:!')
    return text


def answers_match(generated: str, reference: str, strict: bool = False) -> bool:
    """Compare two answers with configurable strictness."""
    gen_norm = normalize_answer(generated)
    ref_norm = normalize_answer(reference)

    if not gen_norm or not ref_norm:
        return False

    if gen_norm == ref_norm:
        return True

    # MCQ specific logic: securely match leading letter options (A, B, C, D, E)
    def get_mcq(text):
        m = re.match(r'^([a-e])\b', text, re.IGNORECASE)
        return m.group(1).lower() if m else None

    gen_mcq = get_mcq(gen_norm)
    ref_mcq = get_mcq(ref_norm)

    if ref_mcq and gen_mcq:
        if ref_mcq == gen_mcq:
            return True

    if strict:
        return False

    # Containment check
    # Prevent the single-letter substring bug (e.g., "e" inside "d. more intelligence")
    if len(gen_norm) > 1 and gen_norm in ref_norm:
        return True
    if len(ref_norm) > 1 and ref_norm in gen_norm:
        return True

    # Numeric equivalence check
    try:
        gen_num = float(gen_norm.replace(',', '').replace(' ', ''))
        ref_num = float(ref_norm.replace(',', '').replace(' ', ''))
        if abs(gen_num - ref_num) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    return False


def token_f1(generated: str, reference: str) -> float:
    """Compute token-level F1 between generated and reference answers."""
    gen_tokens = set(normalize_answer(generated).split())
    ref_tokens = set(normalize_answer(reference).split())

    if not gen_tokens or not ref_tokens:
        return 0.0

    common = gen_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(gen_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# B. Structured inference prompting
# ---------------------------------------------------------------------------

STRUCTURED_SYSTEM_PROMPT = (
    "You are a precise reasoning assistant. When solving problems:\n"
    "1. Think step-by-step inside <think>...</think> tags. Do NOT use tool calls.\n"
    "2. After </think>, output ONLY the final answer on a new line.\n"
    "3. If the question is multiple-choice, output ONLY the correct option letter (e.g., A, B, C, D, E).\n"
    "4. The final answer should be as concise as possible (a number, expression, or short phrase).\n"
    "5. Do NOT add explanations or conversational text after </think>.\n\n"
    "Example format:\n"
    "<think>\n[your reasoning steps]\n</think>\n\nB"
)


def build_inference_prompt(
    tokenizer,
    user_prompt: str,
    use_system_prompt: bool = True,
) -> str:
    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": STRUCTURED_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_prompt})

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        if use_system_prompt:
            combined = f"{STRUCTURED_SYSTEM_PROMPT}\n\n{user_prompt}"
            messages = [{"role": "user", "content": combined}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            raise
    return prompt_text


# ---------------------------------------------------------------------------
# C. Generative evaluation pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_generative(
    model,
    tokenizer,
    task_number: int,
    device: str = "cuda",
    max_samples: int = 50,
    max_new_tokens: int = 512,
    use_system_prompt: bool = True,
    strict_match: bool = False,
    verbose: bool = True,
) -> dict:
    
    files = [
        "data/task0_gsm8k_cot.jsonl",
        "data/task1_arc_cot.jsonl",
        "data/task2_csqa_cot.jsonl",
    ]
    file_path = files[task_number % len(files)]

    path = Path(file_path)
    if not path.exists():
        path = Path(__file__).parent.parent / file_path
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset: {file_path}")

    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if len(samples) >= max_samples:
                break

    model.eval()

    original_use_cache = getattr(model.config, 'use_cache', True)
    model.config.use_cache = True

    exact_matches = 0
    normalized_matches = 0
    containment_matches = 0
    details = []

    for i, sample in enumerate(samples):
        prompt = sample['prompt']
        reference_response = sample['response'].strip()
        reference_answer = extract_answer_from_cot(reference_response)

        prompt_text = build_inference_prompt(tokenizer, prompt, use_system_prompt)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        generated_answer = extract_answer_from_cot(generated_text)

        is_exact = (generated_answer.strip() == reference_answer.strip())
        is_normalized = answers_match(generated_answer, reference_answer, strict=True)
        is_containment = answers_match(generated_answer, reference_answer, strict=False)

        if is_exact:
            exact_matches += 1
        if is_normalized:
            normalized_matches += 1
        if is_containment:
            containment_matches += 1

        detail = {
            'index': i,
            'reference_answer': reference_answer,
            'generated_answer': generated_answer,
            'exact_match': is_exact,
            'normalized_match': is_normalized,
            'containment_match': is_containment,
        }
        details.append(detail)

        if verbose:
            status = "\u2705 MATCH" if is_containment else "\u274c MISS"
            match_type = ""
            if is_exact:
                match_type = " (exact)"
            elif is_normalized:
                match_type = " (normalized)"
            elif is_containment:
                match_type = " (containment)"
            print(f"  [{i+1}/{len(samples)}] {status}{match_type}")
            if not is_containment and verbose:
                print(f"    REF: {reference_answer[:80]}")
                print(f"    GEN: {generated_answer[:80]}")

    model.config.use_cache = original_use_cache

    total = len(samples)
    results = {
        'task_number': task_number,
        'total_samples': total,
        'exact_match_rate': exact_matches / total if total > 0 else 0.0,
        'normalized_match_rate': normalized_matches / total if total > 0 else 0.0,
        'containment_match_rate': containment_matches / total if total > 0 else 0.0,
        'exact_matches': exact_matches,
        'normalized_matches': normalized_matches,
        'containment_matches': containment_matches,
        'use_system_prompt': use_system_prompt,
        'details': details,
    }

    if verbose:
        print(f"\n  Task {task_number} Generative EM Results:")
        print(f"    Exact Match:       {exact_matches}/{total} = {results['exact_match_rate']*100:.1f}%")
        print(f"    Normalized Match:  {normalized_matches}/{total} = {results['normalized_match_rate']*100:.1f}%")
        print(f"    Containment Match: {containment_matches}/{total} = {results['containment_match_rate']*100:.1f}%")

    return results

@torch.no_grad()
def evaluate_accuracy(model, dataloader, device, limit: int = -1) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        if 0 < limit <= num_batches:
            break
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += out.loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return math.exp(-avg_loss)