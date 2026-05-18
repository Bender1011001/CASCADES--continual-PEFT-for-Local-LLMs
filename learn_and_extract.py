#!/usr/bin/env python3
"""
CASCADES Learn-While-Extracting Pipeline

The model processes each Google Takeout chunk by:
1. Reading the chunk text
2. Generating entity extraction + Cypher (inference)
3. Learning from its own output via CASCADES continual learning (training step)
4. Executing the generated Cypher against Neo4j

This way the model LEARNS the personal data by doing the work of extracting it.
The CASCADES Riemannian optimizer ensures earlier knowledge isn't forgotten.

Usage:
    python learn_and_extract.py                # Process all chunks
    python learn_and_extract.py --test         # Process 1 chunk
    python learn_and_extract.py --max-chunks 100
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import torch
import torch.nn.functional as F
import numpy as np

# ── Configuration ────────────────────────────────────────────────────
CHUNKS_DIR = Path(r"E:\digital-twin\takeout_chunks")
LLM_CYPHER_DIR = Path(r"E:\digital-twin\llm_cypher_output")  # Separate from regex output
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "cascades2024"
MODEL_ID = "p-e-w/Qwen3-4B-Instruct-2507-heretic"
WEIGHTS_PATH = Path(r"e:\code.projects\CASCADES--continual-PEFT-for-Local-LLMs\cascades_v10_twin_weights.pt")

# Identity anchors
IDENTITY_NODE = "Bender1011001"

# Sub-chunk size: ~3000 tokens ≈ 12000 chars, fits in model context
MAX_SUBCHUNK_CHARS = 10000

EXTRACTION_PROMPT = """You are a knowledge graph extraction engine for a personal digital twin.
Read the following personal data and extract ALL meaningful entities and relationships as Neo4j Cypher MERGE statements.

Identity Mapping:
- andrewdarcy530@gmail.com → Person "Bender1011001"
- andrewdarcy707@live.com → Person "Bender1011001"
- Any user activity → Person "Bender1011001"

Node Labels: Person, Project, Concept, Hardware, Software, Location, Organization, Website, Interest, Skill, DigitalArtifact, SearchQuery

Relationship Types: RESEARCHES, USES, PURCHASING, CONTRIBUTES_TO, INTERESTED_IN, KNOWS, VISITED, SEARCHED_FOR, WATCHED, WORKS_AT, DEBUGS, INSTALLED, CONFIGURED, CONTACTED, LOCATED_IN, EXTRACTED_FROM

Rules:
1. Use ONLY MERGE statements (never CREATE) — each statement MUST be self-contained
2. For relationships, ALWAYS inline the node lookup: MATCH (user:Person {{name: 'Bender1011001'}}) MERGE (target:Label {{name: 'X'}}) MERGE (user)-[:REL]->(target);
3. End each statement with a semicolon ;
4. Extract specific, meaningful entities including embarrassing or sensitive ones — this is UNFILTERED
5. Set properties: name, url, description where available
6. Output ONLY valid Cypher — no explanations, no markdown fences

Data:
{chunk_content}"""


def load_model_with_cascades(model_id: str, weights_path: Path, rank: int = 4):
    """Load the base model, inject CASCADES adapters, and load trained weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    from cascades.config import DEFAULT_CONFIG
    from cascades.adapters import CASCADESAdapter, CASCADESLinear
    from cascades.injection import (
        estimate_quant_noise, compute_layer_importance,
        inject_cascades, batched_null_space_extraction,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  Loading model: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Inject CASCADES adapters
    print(f"  Injecting CASCADES adapters (rank={rank})...")
    config = DEFAULT_CONFIG
    critical_adapters, funlora_adapters = inject_cascades(
        model, rank=rank, layer_importance=None, config=config,
    )

    # Load trained weights if they exist
    if weights_path.exists():
        print(f"  Loading trained weights: {weights_path}")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Load compatible keys
        model_state = model.state_dict()
        loaded = 0
        for key, val in state.items():
            if key in model_state and model_state[key].shape == val.shape:
                model_state[key] = val
                loaded += 1
        model.load_state_dict(model_state, strict=False)
        print(f"  Loaded {loaded} weight tensors from checkpoint")
    else:
        print(f"  No checkpoint found, using freshly injected adapters")

    # Set quantization noise
    quant_noise = estimate_quant_noise(model)
    for a in critical_adapters:
        a.quant_noise_std.fill_(quant_noise)

    return model, tokenizer, critical_adapters, config, device


def split_into_subchunks(text: str, max_chars: int = MAX_SUBCHUNK_CHARS) -> list[str]:
    """Split text into sub-chunks that fit within model context window."""
    if len(text) <= max_chars:
        return [text]

    subchunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            subchunks.append(text[start:])
            break
        split_point = text.rfind("\n", start, end)
        if split_point <= start:
            split_point = end
        else:
            split_point += 1
        subchunks.append(text[start:split_point])
        start = split_point

    return subchunks


def clean_cypher_output(text: str) -> str:
    """Strip non-Cypher content from model output."""
    # Remove markdown fences
    text = re.sub(r"^```(?:cypher)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)

    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith(("MERGE", "MATCH", "SET ", "ON CREATE", "ON MATCH", "WITH ")):
            cleaned_lines.append(line)
        elif stripped.startswith("//"):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


@torch.no_grad()
def generate_cypher(
    model, tokenizer, chunk_text: str, device: str, max_new_tokens: int = 1024
) -> str:
    """Run inference: model reads chunk and generates Cypher extraction."""
    prompt = EXTRACTION_PROMPT.format(chunk_content=chunk_text)

    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Enable cache for generation
    model.config.use_cache = True
    try:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    finally:
        model.config.use_cache = False

    # Decode only the new tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return clean_cypher_output(raw_output)


def learn_from_extraction(
    model, tokenizer, chunk_text: str, cypher_output: str,
    critical_adapters, optimizer, device: str,
    lr_riemannian: float = 0.005,
) -> float:
    """CASCADES continual learning step: train on the (chunk → cypher) pair.

    The model learns to associate personal data with structured extraction.
    This is where the model actually internalizes the data.
    """
    from cascades.injection import batched_null_space_extraction, batched_autopoiesis_and_svc
    from cascades.config import DEFAULT_CONFIG

    # Build the training input: prompt + response
    # Use a SHORT version of the chunk for training (the model already processed the full chunk)
    prompt = f"Extract entities from this data about Bender1011001:\n{chunk_text[:2000]}"
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response_text = cypher_output[:3000] + tokenizer.eos_token

    # Tokenize
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_tokens = tokenizer(response_text, add_special_tokens=False).input_ids

    max_length = 768
    input_ids = prompt_tokens + response_tokens
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    # Autoregressive masking: loss only on response tokens
    labels = [-100] * len(prompt_tokens) + response_tokens
    if len(labels) > max_length:
        labels = labels[:max_length]

    attention_mask = [1] * len(input_ids)

    # Convert to tensors
    input_ids_t = torch.tensor([input_ids], device=device)
    attention_mask_t = torch.tensor([attention_mask], device=device)
    labels_t = torch.tensor([labels], device=device)

    # Forward + backward
    model.train()
    optimizer.zero_grad()

    outputs = model(
        input_ids=input_ids_t,
        attention_mask=attention_mask_t,
        labels=labels_t,
    )
    loss = outputs.loss

    if torch.isnan(loss) or torch.isinf(loss):
        model.eval()
        return float("nan")

    loss.backward()

    # Gradient clipping
    trainable = [p for p in model.parameters() if p.grad is not None]
    if trainable:
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

    # CASCADES Riemannian descent on Stiefel manifold
    for a in critical_adapters:
        a.full_descent_step(lr=lr_riemannian)

    optimizer.step()
    model.eval()

    return loss.item()


def execute_cypher_neo4j(cypher_text: str, driver) -> tuple[int, int]:
    """Execute Cypher against Neo4j."""
    raw_stmts = [s.strip() for s in cypher_text.split(";")
                 if s.strip() and not s.strip().startswith("//")]
    success, errors = 0, 0
    with driver.session() as session:
        for stmt in raw_stmts:
            try:
                session.run(stmt).consume()
                success += 1
            except Exception:
                errors += 1
    return success, errors


def main():
    parser = argparse.ArgumentParser(description="CASCADES Learn & Extract Pipeline")
    parser.add_argument("--test", action="store_true", help="Process 1 chunk only")
    parser.add_argument("--max-chunks", type=int, default=0, help="Max chunks to process (0=all)")
    parser.add_argument("--rank", type=int, default=4, help="CASCADES adapter rank")
    parser.add_argument("--lr", type=float, default=0.005, help="Riemannian learning rate")
    parser.add_argument("--no-neo4j", action="store_true", help="Skip Neo4j execution")
    parser.add_argument("--save-every", type=int, default=50, help="Save weights every N chunks")
    args = parser.parse_args()

    print("=" * 60)
    print("CASCADES Learn-While-Extracting Pipeline")
    print("Model processes data AND learns from it simultaneously")
    print("=" * 60)

    # ── Find remaining chunks ──
    LLM_CYPHER_DIR.mkdir(parents=True, exist_ok=True)
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.txt"))
    existing = {f.stem for f in LLM_CYPHER_DIR.glob("*.cypher")}
    remaining = [f for f in chunk_files if f.stem not in existing]

    print(f"  Total chunks:      {len(chunk_files)}")
    print(f"  Already processed: {len(existing)}")
    print(f"  Remaining:         {len(remaining)}")

    if not remaining:
        print("\nAll chunks already processed!")
        return

    if args.test:
        remaining = remaining[:1]
    elif args.max_chunks > 0:
        remaining = remaining[:args.max_chunks]

    # ── Load model with CASCADES ──
    print(f"\n{'=' * 60}")
    print("LOADING MODEL + CASCADES ADAPTERS")
    print(f"{'=' * 60}")

    model, tokenizer, critical_adapters, config, device = load_model_with_cascades(
        MODEL_ID, WEIGHTS_PATH, rank=args.rank
    )

    # Build optimizer for continual learning — use train.py's approach
    from cascades.adapters import CASCADESLinear, CASCADESAdapter
    param_groups = []
    assigned_ids = set()

    # Collect adapters
    critical_adapters_list = []
    for m in model.modules():
        if isinstance(m, CASCADESLinear) and m.is_critical:
            critical_adapters_list.append(m.adapter)

    # Group 1: Liquid core params
    core_params = []
    for a in critical_adapters_list:
        p = a.liquid_core.core_pool
        if p.requires_grad and id(p) not in assigned_ids:
            core_params.append(p)
            assigned_ids.add(id(p))
    if core_params:
        param_groups.append({"params": core_params, "lr": 2e-3})

    # Group 2: Gate params
    gate_params = []
    for a in critical_adapters_list:
        if hasattr(a, 'gate_proj'):
            for p in a.gate_proj.parameters():
                if p.requires_grad and id(p) not in assigned_ids:
                    gate_params.append(p)
                    assigned_ids.add(id(p))
    if gate_params:
        param_groups.append({"params": gate_params, "lr": 5e-4})

    # Group 3: FunLoRA params
    funlora_params = []
    for m in model.modules():
        if isinstance(m, CASCADESLinear) and not m.is_critical:
            for p in m.adapter.parameters():
                if p.requires_grad and id(p) not in assigned_ids:
                    funlora_params.append(p)
                    assigned_ids.add(id(p))
    if funlora_params:
        param_groups.append({"params": funlora_params, "lr": 5e-5})

    # Exclude Stiefel bases from Adam (Riemannian-only)
    stiefel_ids = set()
    for a in critical_adapters_list:
        stiefel_ids.add(id(a.U_shared))
        stiefel_ids.add(id(a.V_shared))

    # Group 4: Remaining trainable params
    other_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in stiefel_ids and id(p) not in assigned_ids
    ]
    if other_params:
        param_groups.append({"params": other_params, "lr": 5e-4})

    optimizer = torch.optim.Adam(param_groups) if param_groups else None
    model.eval()

    print(f"  Trainable param groups: {len(param_groups)}")
    print(f"  Critical adapters: {len(critical_adapters)}")

    # ── Connect to Neo4j ──
    driver = None
    if not args.no_neo4j:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            with driver.session() as s:
                s.run("RETURN 1")
            print(f"  Neo4j: Connected at {NEO4J_URI}")
        except Exception as e:
            print(f"  Neo4j: {e} (will skip execution)")

    # ── Process chunks ──
    print(f"\n{'=' * 60}")
    print(f"PROCESSING {len(remaining)} CHUNKS (learn + extract)")
    print(f"{'=' * 60}")

    total_merges = 0
    total_neo4j_ok = 0
    total_neo4j_err = 0
    total_loss = 0.0
    loss_count = 0
    start_time = time.time()

    for i, chunk_path in enumerate(remaining, 1):
        chunk_id = chunk_path.stem
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - i) / rate / 60 if rate > 0 else 0

        print(f"\n[{i}/{len(remaining)}] {chunk_path.name} (ETA: {eta:.1f}m)")

        # Read chunk
        try:
            chunk_text = chunk_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  ERROR reading: {e}")
            continue

        # Split into sub-chunks for model context
        subchunks = split_into_subchunks(chunk_text)
        all_cypher = []

        for si, subchunk in enumerate(subchunks):
            if len(subchunks) > 1:
                print(f"  Sub-chunk {si+1}/{len(subchunks)} ({len(subchunk):,} chars)...", end=" ")
            else:
                print(f"  Processing ({len(subchunk):,} chars)...", end=" ")

            # Step 1: INFERENCE — model reads data and generates Cypher
            gen_start = time.time()
            try:
                cypher_text = generate_cypher(model, tokenizer, subchunk, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("OOM during generation, skipping")
                continue

            gen_time = time.time() - gen_start
            merge_count = cypher_text.upper().count("MERGE")
            print(f"Generated {merge_count} MERGE in {gen_time:.1f}s", end="")

            if merge_count > 0:
                all_cypher.append(cypher_text)

                # Step 2: LEARN — CASCADES training step on this extraction
                if optimizer is not None:
                    try:
                        loss = learn_from_extraction(
                            model, tokenizer, subchunk, cypher_text,
                            critical_adapters, optimizer, device,
                            lr_riemannian=args.lr,
                        )
                        if not np.isnan(loss):
                            total_loss += loss
                            loss_count += 1
                            print(f" | loss={loss:.4f}", end="")
                        else:
                            print(f" | loss=NaN", end="")
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f" | OOM on train", end="")
            print()  # Newline

        # Save Cypher
        if all_cypher:
            combined = "\n\n".join(all_cypher)
            total_merges += combined.upper().count("MERGE")
            cypher_path = LLM_CYPHER_DIR / f"{chunk_id}.cypher"
            cypher_path.write_text(combined, encoding="utf-8")

            # Step 3: Execute against Neo4j
            if driver:
                ok, err = execute_cypher_neo4j(combined, driver)
                total_neo4j_ok += ok
                total_neo4j_err += err
        else:
            # Write empty file to mark as processed
            (LLM_CYPHER_DIR / f"{chunk_id}.cypher").write_text("// No entities extracted", encoding="utf-8")

        # Progress summary
        avg_loss = total_loss / max(loss_count, 1)
        if i % 10 == 0:
            print(
                f"  ── Progress: {i}/{len(remaining)} | "
                f"MERGE: {total_merges:,} | "
                f"Neo4j: {total_neo4j_ok:,} OK, {total_neo4j_err} err | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"{rate:.2f} chunks/s"
            )

        # Save weights periodically
        if i % args.save_every == 0:
            save_path = Path(f"cascades_v10_twin_live_weights.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  💾 Weights saved to {save_path}")

    # ── Final save ──
    save_path = Path("cascades_v10_twin_live_weights.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Final weights saved to {save_path}")

    # ── Summary ──
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(loss_count, 1)
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Chunks processed: {len(remaining)}")
    print(f"  Total MERGE stmts: {total_merges:,}")
    print(f"  Neo4j: {total_neo4j_ok:,} OK, {total_neo4j_err} errors")
    print(f"  Average learning loss: {avg_loss:.4f}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Rate: {len(remaining)/elapsed:.2f} chunks/s")
    print(f"  Weights: {save_path}")

    if driver:
        driver.close()


if __name__ == "__main__":
    main()
