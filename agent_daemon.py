#!/usr/bin/env python3
"""
CASCADES Lifelong Agent Daemon — Asynchronous Actor-Learner Architecture

The Dual-Hemisphere "Wake/Sleep" Agent:
  - Hemisphere A (Actor): Chats with you, runs background VM when idle
  - Hemisphere B (Learner): Extracts facts, synthesizes Q&A, trains CASCADES
  - Hot-Swap: In-place weight update via brain_lock — no restart needed
  - Titanium Padlock: freeze_current_subspace() locks facts permanently

Usage:
    python agent_daemon.py                     # Interactive chat
    python agent_daemon.py --teach "My name is Andrew and I go by Bender1011001"
    python agent_daemon.py --test              # Run identity recall test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ── Configuration ────────────────────────────────────────────────────
MODEL_ID = "p-e-w/Qwen3-4B-Instruct-2507-heretic"
AGENT_RANK = 8  # Final: rank 8 fits in 8GB VRAM. Rank 16 hangs (VRAM thrash).
WEIGHTS_DIR = Path(r"e:\code.projects\CASCADES--continual-PEFT-for-Local-LLMs")
BRAIN_FILE = WEIGHTS_DIR / "agent_brain.pt"
MEMORY_BUFFER_FILE = WEIGHTS_DIR / "daily_memory.jsonl"
IDLE_TIMEOUT_SEC = 60  # Trigger VM after this many seconds of no input
DREAM_THRESHOLD = 3  # Number of memories before triggering dream cycle
LR_RIEMANNIAN = 0.005
LR_LIQUID = 2e-3
LR_GATE = 5e-4
DREAM_EPOCHS = 3  # Repeat training on memory buffer this many times


class SelfSynthesizer:
    """Extracts hard facts from raw text and synthesizes dense Q&A pairs.
    
    This prevents Conversational Collapse — the model learns FACTS,
    not chatty patterns.
    """
    
    # Patterns that indicate factual statements
    FACT_PATTERNS = [
        (r"(?:my name is|i'm called|i go by)\s+(.+)", "name"),
        (r"(?:i work (?:as|at|for)|my job is|my profession is)\s+(.+)", "job"),
        (r"(?:i live in|i'm from|i'm located in)\s+(.+)", "location"),
        (r"(?:my email is|reach me at)\s+(.+)", "email"),
        (r"(?:i use|i'm using|my setup is)\s+(.+)", "tools"),
        (r"(?:i'm interested in|i like|my hobby is|i enjoy)\s+(.+)", "interest"),
        (r"(?:i'm building|i'm working on|my project is)\s+(.+)", "project"),
        (r"(?:i know|i speak|languages? i (?:know|use))\s+(.+)", "skill"),
        (r"(?:my age is|i'm \d+ years old|i was born in)\s+(.+)", "age"),
        (r"(?:my username is|my handle is|i go by)\s+(.+)", "username"),
    ]
    
    @staticmethod
    def extract_facts(text: str) -> list[dict]:
        """Extract structured facts from raw text."""
        facts = []
        lower = text.lower()
        
        for pattern, fact_type in SelfSynthesizer.FACT_PATTERNS:
            matches = re.findall(pattern, lower, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip().rstrip(".,!?")
                if len(clean_match) > 2:
                    facts.append({"type": fact_type, "value": clean_match, "source": text[:200]})
        
        # Also capture any explicit key-value style facts
        kv_pattern = r"(?:^|\n)\s*[-*]\s*(\w[\w\s]+?):\s*(.+?)(?:\n|$)"
        for key, val in re.findall(kv_pattern, text, re.MULTILINE):
            facts.append({"type": key.strip().lower(), "value": val.strip(), "source": text[:200]})
        
        return facts
    
    @staticmethod
    def synthesize_qa(facts: list[dict], raw_text: str = "") -> list[dict]:
        """Convert extracted facts into dense declarative Q&A training pairs."""
        qa_pairs = []
        
        templates = {
            "name": [
                ("What is the user's name?", "The user's name is {value}."),
                ("Who am I?", "You are {value}."),
                ("What should I call you?", "Based on what you've told me, your name is {value}."),
            ],
            "job": [
                ("What does the user do for a living?", "The user works as {value}."),
                ("What is my profession?", "You work as {value}."),
                ("Remind me of my job.", "Your profession is {value}."),
            ],
            "location": [
                ("Where does the user live?", "The user lives in {value}."),
                ("Where am I located?", "You are located in {value}."),
            ],
            "email": [
                ("What is the user's email?", "The user's email is {value}."),
                ("How do I contact the user?", "You can reach the user at {value}."),
            ],
            "project": [
                ("What is the user working on?", "The user is working on {value}."),
                ("What project is the user building?", "The user's current project is {value}."),
            ],
            "interest": [
                ("What are the user's interests?", "The user is interested in {value}."),
                ("What does the user enjoy?", "The user enjoys {value}."),
            ],
            "tools": [
                ("What tools does the user use?", "The user uses {value}."),
            ],
            "skill": [
                ("What skills does the user have?", "The user knows {value}."),
            ],
            "username": [
                ("What is the user's username?", "The user goes by {value}."),
                ("What handle does the user use?", "The user's handle is {value}."),
            ],
        }
        
        for fact in facts:
            ft = fact["type"]
            val = fact["value"]
            
            if ft in templates:
                for q_template, a_template in templates[ft]:
                    qa_pairs.append({
                        "prompt": q_template,
                        "response": a_template.format(value=val),
                    })
            else:
                # Generic fact
                qa_pairs.append({
                    "prompt": f"What do you know about the user's {ft}?",
                    "response": f"The user's {ft} is {val}.",
                })
        
        # Also create a raw memorization pair from the full text if present
        if raw_text and len(raw_text) > 20:
            qa_pairs.append({
                "prompt": "What do you remember from our recent conversation?",
                "response": f"From our recent interaction, I recall: {raw_text[:500]}",
            })
        
        return qa_pairs


class LifelongAgent:
    """Asynchronous Actor-Learner with Dual-Hemisphere Architecture."""
    
    def __init__(self, rank: int = AGENT_RANK, load_brain: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.brain_lock = threading.Lock()
        self.memory_buffer: list[dict] = []
        self.last_active = time.time()
        self.synthesizer = SelfSynthesizer()
        self.chat_history: list[dict] = []
        self.total_dreams = 0
        self.total_facts_learned = 0
        
        print("=" * 60)
        print("🧠 CASCADES Lifelong Agent — Dual Hemisphere Architecture")
        print("=" * 60)
        
        self._load_model(rank)
        self._build_optimizer()
        
        if load_brain and BRAIN_FILE.exists():
            self._load_brain()
        
        # Load any pending memories from disk
        if MEMORY_BUFFER_FILE.exists():
            try:
                with open(MEMORY_BUFFER_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.memory_buffer.append(json.loads(line))
                if self.memory_buffer:
                    print(f"  📥 Loaded {len(self.memory_buffer)} pending memories from disk")
            except Exception:
                pass
        
        print(f"\n✅ Agent ready. Rank={rank}, Device={self.device}")
        print(f"   Brain: {BRAIN_FILE.name} ({'exists' if BRAIN_FILE.exists() else 'fresh'})")
        print(f"   Adapters: {len(self.critical_adapters)} critical")
    
    def _load_model(self, rank: int):
        """Load base model + inject CASCADES adapters."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        from cascades.config import DEFAULT_CONFIG
        from cascades.injection import inject_cascades, estimate_quant_noise
        
        print(f"  Loading {MODEL_ID}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="auto"
        )
        self.model = prepare_model_for_kbit_training(
            self.model, use_gradient_checkpointing=True
        )
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        
        # Inject CASCADES with HIGH rank for lifelong capacity
        print(f"  Injecting CASCADES adapters (rank={rank})...")
        self.config = DEFAULT_CONFIG
        self.critical_adapters, self.funlora_adapters = inject_cascades(
            self.model, rank=rank, layer_importance=None, config=self.config,
        )
        
        # Set quant noise
        quant_noise = estimate_quant_noise(self.model)
        for a in self.critical_adapters:
            a.quant_noise_std.fill_(quant_noise)
        
        self.model.eval()
        self.model.config.use_cache = True
    
    def _build_optimizer(self):
        """Build Adam optimizer with correct param groups (mirrors train.py)."""
        from cascades.adapters import CASCADESLinear
        
        assigned_ids = set()
        param_groups = []
        
        # Stiefel bases — NEVER in Adam (Riemannian only)
        stiefel_ids = set()
        for a in self.critical_adapters:
            stiefel_ids.add(id(a.U_shared))
            stiefel_ids.add(id(a.V_shared))
        
        def unique_params(p_list):
            seen = set()
            result = []
            for p in p_list:
                pid = id(p)
                if pid not in seen and pid not in assigned_ids and pid not in stiefel_ids:
                    seen.add(pid)
                    result.append(p)
            return result
        
        # Group 1: Liquid cores
        liquid_params = unique_params([
            p for a in self.critical_adapters for p in a.liquid_core.parameters()
        ])
        if liquid_params:
            param_groups.append({"params": liquid_params, "lr": LR_LIQUID})
            assigned_ids.update(id(p) for p in liquid_params)
        
        # Group 2: GainLoRA gates
        gate_params = unique_params([
            p for a in self.critical_adapters
            if hasattr(a, "gate_proj")
            for p in a.gate_proj.parameters()
        ])
        if gate_params:
            param_groups.append({"params": gate_params, "lr": LR_GATE})
            assigned_ids.update(id(p) for p in gate_params)
        
        # Group 3: FunLoRA params
        funlora_params = unique_params([
            p for a in self.funlora_adapters for p in a.parameters()
        ])
        if funlora_params:
            param_groups.append({"params": funlora_params, "lr": 5e-5})
            assigned_ids.update(id(p) for p in funlora_params)
        
        # Group 4: Remaining trainable
        fallback_params = unique_params([
            p for p in self.model.parameters() if p.requires_grad
        ])
        if fallback_params:
            param_groups.append({"params": fallback_params, "lr": 5e-4})
        
        self.optimizer = optim.Adam(param_groups) if param_groups else None
        print(f"  Optimizer: {sum(len(g['params']) for g in param_groups)} param groups")
    
    def _load_brain(self):
        """Load saved adapter weights (Hemisphere A wakes with memories)."""
        print(f"  Loading brain from {BRAIN_FILE.name}...")
        state = torch.load(BRAIN_FILE, map_location="cpu", weights_only=True)
        model_state = self.model.state_dict()
        loaded = 0
        for key, val in state.items():
            if key in model_state and model_state[key].shape == val.shape:
                model_state[key] = val
                loaded += 1
        self.model.load_state_dict(model_state, strict=False)
        print(f"  Loaded {loaded}/{len(state)} weight tensors")
    
    def _save_brain(self):
        """Save all adapter weights to disk."""
        # Save adapter-related parameters regardless of requires_grad
        # (frozen adapters have requires_grad=False but still need saving!)
        adapter_state = {}
        for name, param in self.model.named_parameters():
            if "adapter" in name or param.requires_grad:
                adapter_state[name] = param.data.cpu()
        # Also save buffers (frozen_null_basis, EMA trackers, etc.)
        for name, buf in self.model.named_buffers():
            if "adapter" in name:
                adapter_state[name] = buf.cpu()
        
        torch.save(adapter_state, BRAIN_FILE)
        print(f"  \U0001f4be Brain saved: {len(adapter_state)} tensors \u2192 {BRAIN_FILE.name}")
    
    def _flush_memory_to_disk(self):
        """Persist memory buffer to disk (crash recovery)."""
        with open(MEMORY_BUFFER_FILE, "w", encoding="utf-8") as f:
            for mem in self.memory_buffer:
                f.write(json.dumps(mem, ensure_ascii=False) + "\n")
    
    # ── Hemisphere A: The Awake Actor ────────────────────────────────
    
    @torch.no_grad()
    def generate(self, user_message: str, max_new_tokens: int = 512,
                 temperature: float = 0.7, system_prompt: str = None) -> str:
        """Generate a response (Hemisphere A - inference)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add recent chat history (last 4 turns for context)
        messages.extend(self.chat_history[-8:])
        messages.append({"role": "user", "content": user_message})
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=4096
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        self.model.config.use_cache = True
        with self.brain_lock:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        new_tokens = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def add_to_memory(self, text: str, source: str = "chat"):
        """Add raw text to the memory buffer for Hemisphere B to process."""
        self.memory_buffer.append({
            "text": text,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        })
        self._flush_memory_to_disk()
    
    # ── Hemisphere B: The Dreaming Learner ───────────────────────────
    
    def dream_cycle(self):
        """Hemisphere B: Process memory buffer into permanent knowledge.
        
        1. Extract facts from raw memories
        2. Synthesize dense Q&A pairs (prevents Conversational Collapse)
        3. Train CASCADES on Q&A pairs (Riemannian descent)
        4. Freeze subspace (Titanium Padlock)
        5. Save brain checkpoint
        """
        from cascades.injection import batched_null_space_extraction
        
        if not self.memory_buffer:
            return
        
        with self.brain_lock:
            print(f"\n🛌 [Hemisphere B: Dream Cycle #{self.total_dreams + 1}]")
            print(f"   Processing {len(self.memory_buffer)} raw memories...")
            
            # Step 1 & 2: Extract facts and synthesize Q&A
            all_qa_pairs = []
            for mem in self.memory_buffer:
                raw_text = mem.get("text", "")
                facts = self.synthesizer.extract_facts(raw_text)
                qa_pairs = self.synthesizer.synthesize_qa(facts, raw_text)
                all_qa_pairs.extend(qa_pairs)
            
            if not all_qa_pairs:
                # If no structured facts found, create generic memorization pairs
                for mem in self.memory_buffer:
                    raw = mem.get("text", "")[:500]
                    if len(raw) > 20:
                        all_qa_pairs.append({
                            "prompt": "What do you remember about the user?",
                            "response": raw,
                        })
            
            print(f"   Synthesized {len(all_qa_pairs)} Q&A training pairs")
            
            # Step 3: CASCADES Training
            self.model.train()
            self.model.config.use_cache = False
            
            total_loss = 0.0
            n_steps = 0
            
            for epoch in range(DREAM_EPOCHS):
                for qa in all_qa_pairs:
                    loss_val = self._train_on_qa(qa["prompt"], qa["response"])
                    if loss_val is not None and not np.isnan(loss_val):
                        total_loss += loss_val
                        n_steps += 1
                
                if n_steps > 0:
                    avg = total_loss / n_steps
                    print(f"   Epoch {epoch + 1}/{DREAM_EPOCHS}: avg_loss={avg:.4f}")
            
            # Step 4: Titanium Padlock — freeze learned subspace
            print("   🔒 Freezing subspace (Titanium Padlock)...")
            batched_null_space_extraction(self.critical_adapters)
            for a in self.critical_adapters:
                a.freeze_current_subspace()
            
            frozen_dims = sum(
                a.frozen_null_basis.shape[1] for a in self.critical_adapters
            )
            print(f"   Protected dimensions: {frozen_dims}")
            
            # Step 5: Save checkpoint
            self._save_brain()
            
            self.total_dreams += 1
            self.total_facts_learned += len(all_qa_pairs)
            self.memory_buffer.clear()
            
            # Clear the on-disk buffer
            if MEMORY_BUFFER_FILE.exists():
                MEMORY_BUFFER_FILE.unlink()
            
            self.model.eval()
            self.model.config.use_cache = True
            
            print(f"   ✅ Dream complete. Total facts learned: {self.total_facts_learned}")
            print(f"      Total dream cycles: {self.total_dreams}\n")
    
    def _train_on_qa(self, prompt: str, response: str) -> float | None:
        """Single CASCADES training step on a Q&A pair.
        
        Uses autoregressive masking: loss only on response tokens.
        """
        from cascades.adapters import CASCADESAdapter
        
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response_text = response + self.tokenizer.eos_token
        
        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        response_tokens = self.tokenizer(response_text, add_special_tokens=False).input_ids
        
        max_length = 768
        input_ids = (prompt_tokens + response_tokens)[:max_length]
        
        # Autoregressive masking: -100 for prompt tokens (no loss)
        labels = ([-100] * len(prompt_tokens) + response_tokens)[:max_length]
        attention_mask = [1] * len(input_ids)
        
        # Pad if needed
        while len(input_ids) < len(labels):
            input_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
        while len(labels) < len(input_ids):
            labels.append(-100)
        
        input_ids_t = torch.tensor([input_ids], device=self.device)
        attention_mask_t = torch.tensor([attention_mask], device=self.device)
        labels_t = torch.tensor([labels], device=self.device)
        
        try:
            self.model.zero_grad()  # Clear ALL gradients including Stiefel bases
            outputs = self.model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                labels=labels_t,
            )
            loss = outputs.loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                return None
            
            loss.backward()
            
            # Gradient clipping
            trainable = [p for p in self.model.parameters() if p.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            
            # CASCADES Riemannian descent on Stiefel manifold
            for a in self.critical_adapters:
                a.full_descent_step(lr=LR_RIEMANNIAN)
            
            self.optimizer.step()
            
            loss_val = loss.item()
            del outputs, loss, input_ids_t, attention_mask_t, labels_t
            torch.cuda.empty_cache()
            return loss_val
        
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
    
    # ── Teaching Interface ───────────────────────────────────────────
    
    def teach(self, fact_text: str):
        """Directly teach the agent a fact — immediate dream cycle."""
        print(f"\n📝 Teaching: \"{fact_text}\"")
        self.add_to_memory(fact_text, source="teach")
        self.dream_cycle()
    
    def teach_bulk(self, facts: list[str]):
        """Teach multiple facts, then consolidate."""
        print(f"\n📝 Teaching {len(facts)} facts...")
        for fact in facts:
            self.add_to_memory(fact, source="teach")
        self.dream_cycle()
    
    def learn_from_graph(self, limit: int = 0, batch_size: int = 50,
                         epochs: int = 20, target_loss: float = 1.0):
        """Mine the Neo4j Knowledge Graph and train on synthesized Q&A pairs.
        
        Trains for multiple epochs until loss drops below target_loss.
        Only freezes subspace AFTER convergence (not mid-training).
        """
        from cascades.injection import batched_null_space_extraction
        
        print("\n" + "=" * 60)
        print("📊 GRAPH → PARAMETRIC MEMORY PIPELINE")
        print("   Mining Neo4j Knowledge Graph for training data...")
        print("=" * 60)
        
        from graph_synthesizer import query_graph, synthesize_qa_from_graph
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "cascades2024"))
        graph_data = query_graph(driver)
        driver.close()
        
        qa_pairs = synthesize_qa_from_graph(graph_data)
        if limit > 0:
            qa_pairs = qa_pairs[:limit]
        
        print(f"\n   Generated {len(qa_pairs)} Q&A pairs from knowledge graph")
        print(f"   Training for up to {epochs} epochs (target loss: {target_loss})")
        
        with self.brain_lock:
            self.model.train()
            self.model.config.use_cache = False
            
            for epoch in range(epochs):
                # Shuffle each epoch for better generalization
                import random
                random.shuffle(qa_pairs)
                
                epoch_loss = 0.0
                epoch_steps = 0
                
                for i, qa in enumerate(qa_pairs):
                    loss_val = self._train_on_qa(qa["prompt"], qa["response"])
                    if loss_val is not None and not np.isnan(loss_val):
                        epoch_loss += loss_val
                        epoch_steps += 1
                    
                    # Progress every 50 steps
                    if (i + 1) % 50 == 0:
                        avg = epoch_loss / max(epoch_steps, 1)
                        print(f"   Epoch {epoch+1}/{epochs} [{i+1}/{len(qa_pairs)}] loss={avg:.4f}")
                
                avg_loss = epoch_loss / max(epoch_steps, 1)
                print(f"   Epoch {epoch+1}/{epochs} complete: avg_loss={avg_loss:.4f}")
                
                # Save checkpoint every epoch
                self._save_brain()
                
                # Check convergence
                if avg_loss < target_loss:
                    print(f"   🎯 Target loss {target_loss} reached! Stopping early.")
                    break
            
            # NOW freeze subspace — only after convergence
            print("   🔒 Freezing learned subspace (Titanium Padlock)...")
            batched_null_space_extraction(self.critical_adapters)
            for a in self.critical_adapters:
                a.freeze_current_subspace()
            self._save_brain()
            
            self.model.eval()
            self.model.config.use_cache = True
        
        frozen = sum(a.frozen_null_basis.shape[1] for a in self.critical_adapters)
        print(f"\n   ✅ Graph learning complete!")
        print(f"   Final avg_loss: {avg_loss:.4f}")
        print(f"   Protected dimensions: {frozen}")
        print(f"   Brain saved to: {BRAIN_FILE}")
    
    # ── Interactive Chat Loop (Windows-compatible) ───────────────────
    
    def chat_loop(self):
        """Interactive chat with idle VM triggering (Windows msvcrt)."""
        import msvcrt
        
        print("\n" + "=" * 60)
        print("🟢 Agent Awake — Dual Hemisphere Active")
        print("   Type messages, or wait 60s for Background VM")
        print("   Commands: /teach <fact>, /dream, /status, /test, /quit")
        print("=" * 60)
        
        while True:
            sys.stdout.write("\n🧑 You: ")
            sys.stdout.flush()
            
            # Windows non-blocking input with timeout
            user_input = self._win_input_with_timeout(IDLE_TIMEOUT_SEC)
            
            if user_input is None:
                # Idle timeout — trigger VM
                self.run_vm_thought()
                continue
            
            user_input = user_input.strip()
            if not user_input:
                continue
            
            self.last_active = time.time()
            
            # Handle commands
            if user_input.lower() == "/quit":
                print("\n👋 Saving brain and shutting down...")
                self._save_brain()
                break
            
            if user_input.lower() == "/dream":
                self.dream_cycle()
                continue
            
            if user_input.lower() == "/status":
                self._print_status()
                continue
            
            if user_input.lower() == "/test":
                self.run_identity_test()
                continue
            
            if user_input.lower().startswith("/teach "):
                fact = user_input[7:].strip()
                self.teach(fact)
                continue
            
            # Normal chat
            response = self.generate(user_input)
            print(f"\n🤖 Agent: {response}")
            
            # Add to chat history
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": response})
            
            # Extract facts from user message for later training
            facts = self.synthesizer.extract_facts(user_input)
            if facts:
                self.add_to_memory(user_input, source="chat_fact")
                n = len(facts)
                print(f"   📝 [{n} fact{'s' if n > 1 else ''} flagged for consolidation]")
            
            # Auto-trigger dream if buffer is full
            if len(self.memory_buffer) >= DREAM_THRESHOLD:
                print("\n   💤 Memory buffer full — triggering dream cycle...")
                threading.Thread(target=self.dream_cycle, daemon=True).start()
    
    def _win_input_with_timeout(self, timeout: float) -> str | None:
        """Read a line from stdin with timeout (Windows-compatible)."""
        import msvcrt
        
        line = []
        start = time.time()
        
        while True:
            elapsed = time.time() - start
            if elapsed >= timeout:
                return None  # Timeout
            
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch == "\r" or ch == "\n":
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return "".join(line)
                elif ch == "\x08":  # Backspace
                    if line:
                        line.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                elif ch == "\x03":  # Ctrl+C
                    raise KeyboardInterrupt
                else:
                    line.append(ch)
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                start = time.time()  # Reset timeout on keypress
            else:
                time.sleep(0.05)  # Don't busy-wait
    
    # ── Background VM (Curiosity Loop) ───────────────────────────────
    
    def run_vm_thought(self):
        """Background VM: model explores when user is idle."""
        print("\n💭 [Background VM: Agent is thinking...]")
        
        prompt = (
            "You are an AI agent with access to a Windows terminal. "
            "The user is away. Think of something interesting to explore, "
            "research, or code. Output a single safe PowerShell command to run. "
            "Only output the command, nothing else."
        )
        
        cmd = self.generate(prompt, max_new_tokens=50, temperature=0.8)
        cmd = cmd.strip().split("\n")[0]  # Take only first line
        
        # Safety: block dangerous commands
        dangerous = ["rm ", "del ", "format ", "remove-item", "stop-", "shutdown", "restart"]
        if any(d in cmd.lower() for d in dangerous):
            print(f"   ⛔ Blocked dangerous command: {cmd}")
            return
        
        print(f"   ▶ Running: {cmd}")
        
        try:
            result = subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True, text=True, timeout=10,
                encoding="utf-8", errors="replace"
            )
            
            # REJECTION SAMPLING: Only learn from SUCCESS
            if result.returncode == 0 and result.stdout.strip():
                output = result.stdout.strip()[:300]
                print(f"   ✅ Output: {output[:100]}...")
                
                # Package successful trajectory as a lesson
                self.add_to_memory(
                    f"I ran the command `{cmd}` and got: {output}",
                    source="vm_success"
                )
            else:
                stderr = result.stderr.strip()[:100] if result.stderr else "no output"
                print(f"   ❌ Failed (discarding): {stderr}")
        except subprocess.TimeoutExpired:
            print("   ⏰ Timed out (discarding)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # ── Utilities ────────────────────────────────────────────────────
    
    def _print_status(self):
        """Print agent status."""
        frozen = sum(a.frozen_null_basis.shape[1] for a in self.critical_adapters)
        print(f"\n📊 Agent Status:")
        print(f"   Critical adapters: {len(self.critical_adapters)}")
        print(f"   Rank: {self.critical_adapters[0].U_shared.shape[1] if self.critical_adapters else '?'}")
        print(f"   Frozen dimensions: {frozen}")
        print(f"   Dream cycles: {self.total_dreams}")
        print(f"   Facts learned: {self.total_facts_learned}")
        print(f"   Memory buffer: {len(self.memory_buffer)} pending")
        print(f"   Chat history: {len(self.chat_history)} messages")
        print(f"   Brain file: {BRAIN_FILE} ({'exists' if BRAIN_FILE.exists() else 'not saved'})")
    
    def run_identity_test(self):
        """Test battery: can the model recall learned facts?"""
        questions = [
            "Who is Bender1011001?",
            "What is the user's email address?",
            "What projects does the user work on?",
            "What hardware does the user use?",
            "What is CASCADES?",
            "Tell me about the user's interests.",
        ]
        
        print("\n" + "=" * 60)
        print("IDENTITY RECALL TEST")
        print("=" * 60)
        
        for q in questions:
            print(f"\n🧑 {q}")
            response = self.generate(q, max_new_tokens=200)
            print(f"🤖 {response}")
            print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="CASCADES Lifelong Agent Daemon")
    parser.add_argument("--rank", type=int, default=AGENT_RANK,
                        help=f"Stiefel manifold rank (default: {AGENT_RANK})")
    parser.add_argument("--teach", type=str, nargs="+",
                        help="Teach facts directly (immediate dream cycle)")
    parser.add_argument("--test", action="store_true",
                        help="Run identity recall test")
    parser.add_argument("--fresh", action="store_true",
                        help="Start with fresh brain (ignore saved weights)")
    parser.add_argument("--dream", action="store_true",
                        help="Process pending memory buffer and exit")
    parser.add_argument("--learn-graph", action="store_true",
                        help="Mine Neo4j KG and train on synthesized Q&A pairs")
    parser.add_argument("--graph-limit", type=int, default=0,
                        help="Max Q&A pairs from graph (0=all)")
    parser.add_argument("--graph-epochs", type=int, default=20,
                        help="Max epochs for graph learning")
    parser.add_argument("--target-loss", type=float, default=1.0,
                        help="Stop training when loss drops below this")
    args = parser.parse_args()
    
    agent = LifelongAgent(rank=args.rank, load_brain=not args.fresh)
    
    if args.learn_graph:
        agent.learn_from_graph(
            limit=args.graph_limit, epochs=args.graph_epochs,
            target_loss=args.target_loss
        )
        return
    
    if args.teach:
        agent.teach_bulk(args.teach)
        return
    
    if args.test:
        agent.run_identity_test()
        return
    
    if args.dream:
        agent.dream_cycle()
        return
    
    # Interactive mode
    try:
        agent.chat_loop()
    except KeyboardInterrupt:
        print("\n\n👋 Saving brain and shutting down...")
        agent._save_brain()


if __name__ == "__main__":
    main()
