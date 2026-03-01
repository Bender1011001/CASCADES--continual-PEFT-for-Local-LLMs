"""
CASCADES Generative Evaluation Module — Structured answer extraction and flexible matching.

This module bridges the gap between high proxy accuracy (exp(-loss)) and exact match (EM)
by implementing:
  1. Structured inference prompting that instructs the model to use <think>...</think> format
  2. Answer extraction from both structured CoT and conversational free-form outputs
  3. Multiple matching strategies (exact, normalized, semantic containment)

The training data format is:
  {"prompt": "...", "response": "<think>\n...\n</think>\n\n<final_answer>"}

The model's generated output may or may not follow this format. This module handles both cases.
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
    """Extract the final answer from a <think>...</think>\\n\\n<answer> formatted response.

    The training data format places the short final answer AFTER the </think> tag,
    separated by two newlines. This function extracts that trailing answer.

    If no </think> tag is found, returns the full text (the model didn't use CoT format).
    """
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

    # Strategy 5: If the model used a "Final Answer:" or "Answer:" prefix (common in chat models)
    answer_prefix_pattern = re.compile(
        r'(?:final\s+answer|the\s+answer|answer)\s*(?:is|:)\s*(.*)',
        re.DOTALL | re.IGNORECASE
    )
    match = answer_prefix_pattern.search(text)
    if match:
        answer = match.group(1).strip()
        # Take only the first line/sentence of the answer
        answer = answer.split('\n')[0].strip()
        # Strip trailing punctuation that chat models add
        answer = answer.rstrip('.')
        if answer:
            return answer

    # Strategy 6: Take the last non-empty line (many CoT responses end with the answer)
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        # Clean up common chat artifacts
        last_line = last_line.rstrip('.')
        return last_line

    return text.strip()


def normalize_latex(text: str) -> str:
    """Normalize LaTeX/math notation to plain text equivalents.

    Handles: Big-O variants, LaTeX commands, delimiters, and operators.
    This is critical for matching model outputs that may use different
    notation than the training data (e.g., O(n^2 log n) vs Theta(n^2 \\log n)).
    """
    # Strip dollar signs and display math
    text = text.replace('$', '')
    text = re.sub(r'\\\[|\\\]', '', text)

    # Normalize Big-O family: Θ, Ω, θ → O (for matching purposes)
    text = text.replace('\u0398', 'O').replace('\u03b8', 'O').replace('\u03a9', 'O')
    text = re.sub(r'\\(?:Theta|theta|Omega|omega)\b', 'O', text)

    # Remove \left, \right delimiters
    text = re.sub(r'\\(?:left|right)\s*', '', text)

    # Convert LaTeX commands to plain text
    latex_to_plain = [
        (r'\\log\b', 'log'),
        (r'\\ln\b', 'ln'),
        (r'\\sqrt\{([^}]*)\}', r'sqrt(\1)'),
        (r'\\sqrt\b', 'sqrt'),
        (r'\\cdot\b', '*'),
        (r'\\times\b', '*'),
        (r'\\pm\b', '+-'),
        (r'\\geq\b', '>='),
        (r'\\leq\b', '<='),
        (r'\\neq\b', '!='),
        (r'\\approx\b', '~'),
        (r'\\infty\b', 'infinity'),
        (r'\\sum\b', 'sum'),
        (r'\\prod\b', 'prod'),
        (r'\\pi\b', 'pi'),
    ]
    for pattern, replacement in latex_to_plain:
        text = re.sub(pattern, replacement, text)

    # Simple \frac{a}{b} → a/b
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', text)

    # Strip remaining \text{}, \mathrm{}, etc.
    text = re.sub(r'\\(?:text|mathrm|mathcal|mathbb|operatorname)\{([^}]*)\}', r'\1', text)

    # Remove standalone backslash commands
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)

    # Remove curly braces
    text = text.replace('{', '').replace('}', '')

    # Double backslash → single
    text = text.replace('\\\\', '\\')

    return text


def normalize_answer(text: str) -> str:
    """Normalize an answer string for flexible comparison.

    Applies: LaTeX normalization, lowercase, unicode normalization,
    whitespace collapse, removal of common mathematical formatting
    artifacts, and stripping of enclosing quotes/periods/punctuation.
    """
    if not text:
        return ""

    # Unicode normalize
    text = unicodedata.normalize('NFKD', text)

    # LaTeX/math notation normalization (before lowercasing for Big-O)
    text = normalize_latex(text)

    # Lowercase
    text = text.lower()

    # Remove common wrapper phrases
    removals = [
        r'^the\s+answer\s+is\s+',
        r'^therefore,?\s+',
        r'^thus,?\s+',
        r'^so,?\s+',
        r'^hence,?\s+',
        r'^we\s+(?:get|have|obtain|find)\s+',
        r'^this\s+(?:gives|yields|results?\s+in)\s+',
    ]
    for pattern in removals:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Strip enclosing quotes and trailing punctuation
    text = text.strip('"\'`')
    text = text.rstrip('.,;:!')

    return text


def answers_match(generated: str, reference: str, strict: bool = False) -> bool:
    """Compare two answers with configurable strictness.

    Args:
        generated: The model's extracted answer.
        reference: The ground truth answer from training data.
        strict: If True, only exact normalized match counts.
                If False, also checks containment (generated contains reference
                or reference contains generated).
    """
    gen_norm = normalize_answer(generated)
    ref_norm = normalize_answer(reference)

    if not gen_norm or not ref_norm:
        return False

    # Exact normalized match
    if gen_norm == ref_norm:
        return True

    if strict:
        return False

    # Containment check (the model might wrap the answer in context)
    if ref_norm in gen_norm or gen_norm in ref_norm:
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
    """Compute token-level F1 between generated and reference answers.

    This provides a softer metric than exact match — even partial overlap
    gets credit. Useful for diagnosing how close the model is to correct answers.

    Returns:
        F1 score in [0.0, 1.0].
    """
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
    "1. Think step-by-step inside <think>...</think> tags\n"
    "2. After </think>, output ONLY the final answer on a new line\n"
    "3. The final answer should be as concise as possible (a number, expression, or short phrase)\n"
    "4. Do NOT add explanations after </think>\n\n"
    "Example format:\n"
    "<think>\n[your reasoning steps]\n</think>\n\n42"
)


def build_inference_prompt(
    tokenizer,
    user_prompt: str,
    use_system_prompt: bool = True,
) -> str:
    """Build a properly formatted inference prompt with optional system instruction.

    Args:
        tokenizer: HuggingFace tokenizer with chat template support.
        user_prompt: The user's question/problem.
        use_system_prompt: Whether to include the structured output system prompt.

    Returns:
        Formatted prompt string ready for tokenization.
    """
    messages = []

    if use_system_prompt:
        messages.append({"role": "system", "content": STRUCTURED_SYSTEM_PROMPT})

    messages.append({"role": "user", "content": user_prompt})

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for tokenizers that don't support system role
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
    """Run generative evaluation with answer extraction and flexible matching.

    Args:
        model: The CASCADES-adapted model.
        tokenizer: Corresponding tokenizer.
        task_number: Which task dataset to evaluate (0=logic, 1=decomp, 2=action).
        device: CUDA device string.
        max_samples: Maximum number of samples to evaluate.
        max_new_tokens: Maximum tokens to generate per sample.
        use_system_prompt: Whether to include structured output instructions.
        strict_match: If True, only exact normalized matches count.
        verbose: Print per-sample results.

    Returns:
        Dictionary with metrics: exact_match_rate, normalized_match_rate,
        containment_match_rate, samples evaluated, and per-sample details.
    """
    files = [
        "data/task0_gsm8k_cot.jsonl",
        "data/task1_arc_cot.jsonl",
        "data/task2_csqa_cot.jsonl",
    ]
    file_path = files[task_number % len(files)]

    # Try relative path first, then absolute
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

    # Re-enable KV cache for generation (disabled during training for grad checkpointing)
    original_use_cache = getattr(model.config, 'use_cache', True)
    model.config.use_cache = True

    exact_matches = 0
    normalized_matches = 0
    containment_matches = 0
    details = []

    for i, sample in enumerate(samples):
        prompt = sample['prompt']
        reference_response = sample['response'].strip()

        # Extract the ground truth final answer from the training data
        reference_answer = extract_answer_from_cot(reference_response)

        # Build inference prompt with structured instructions
        prompt_text = build_inference_prompt(tokenizer, prompt, use_system_prompt)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode only the generated tokens (exclude the prompt)
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Extract the answer from generated text
        generated_answer = extract_answer_from_cot(generated_text)

        # Check matches at multiple strictness levels
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

    # Restore original cache setting
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
    """Compute proxy accuracy as exp(-avg_loss).

    This is the same metric used in the main training pipeline.
    Provided here for module-level access.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader yielding (input_ids, attention_mask, labels).
        device: Device string.
        limit: Maximum number of batches (-1 for all).

    Returns:
        Proxy accuracy in [0, 1].
    """
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
