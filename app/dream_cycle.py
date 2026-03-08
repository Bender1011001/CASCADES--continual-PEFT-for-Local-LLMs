"""
Dream Cycle — Hemisphere B background learner for CASCADES Chat.

Runs in-place Riemannian descent on synthesized Q&A pairs while the model
is serving inference. Uses threading.Lock() for zero-copy hot-swap.

Architecture:
    - Monitors the memory buffer for new facts
    - Extracts Q&A pairs via the Self-Synthesizer
    - Runs CASCADES full_descent_step in-place
    - Freezes current subspace to lock facts permanently
    - Saves adapter checkpoint

The inference thread (Hemisphere A) and dream thread (Hemisphere B) share
the same model weights in GPU memory. The lock ensures they never collide.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DreamCycle:
    """Background learner that consolidates memories into the Stiefel manifold."""

    def __init__(
        self,
        model,
        tokenizer,
        brain_lock: threading.Lock,
        checkpoint_dir: Path = Path("app/data/checkpoints"),
        lr_riemannian: float = 0.005,
        lr_liquid: float = 0.002,
        min_examples: int = 3,
        max_examples_per_cycle: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.brain_lock = brain_lock
        self.checkpoint_dir = checkpoint_dir
        self.lr_riemannian = lr_riemannian
        self.lr_liquid = lr_liquid
        self.min_examples = min_examples
        self.max_examples_per_cycle = max_examples_per_cycle

        self.memory_buffer: list[dict] = []  # Q&A pairs waiting to be learned
        self.buffer_lock = threading.Lock()

        self.cycle_count = 0
        self.total_facts_learned = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def add_memory(self, qa_pairs: list[dict]):
        """Add Q&A pairs to the memory buffer (thread-safe).

        Each pair should have keys: 'q' and 'a'
        """
        with self.buffer_lock:
            self.memory_buffer.extend(qa_pairs)
            buffer_size = len(self.memory_buffer)

        logger.info(f"Memory buffer: {buffer_size} examples (need {self.min_examples} to trigger dream)")

        # Auto-trigger dream if buffer is full enough
        if buffer_size >= self.min_examples and not self._running:
            self.trigger_dream()

    def trigger_dream(self):
        """Start a dream cycle in background thread."""
        if self._running:
            logger.info("Dream cycle already running, skipping")
            return

        self._thread = threading.Thread(target=self._dream, daemon=True)
        self._thread.start()

    def _dream(self):
        """The actual dream cycle. Runs CASCADES in-place training."""
        self._running = True

        # Grab examples from buffer
        with self.buffer_lock:
            examples = self.memory_buffer[:self.max_examples_per_cycle]
            self.memory_buffer = self.memory_buffer[self.max_examples_per_cycle:]

        if not examples:
            self._running = False
            return

        logger.info(f"🛌 Dream Cycle {self.cycle_count + 1}: consolidating {len(examples)} memories...")

        start = time.time()
        losses = []

        # Acquire the brain lock — inference pauses briefly
        with self.brain_lock:
            self.model.train()

            # Collect CASCADES adapters
            from cascades.adapters import CASCADESAdapter
            adapters = []
            for module in self.model.modules():
                if hasattr(module, "adapter") and isinstance(module.adapter, CASCADESAdapter):
                    adapters.append(module.adapter)

            for example in examples:
                prompt = example.get("q", example.get("prompt", ""))
                response = example.get("a", example.get("response", ""))
                full_text = f"{prompt}\n{response}"

                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self.model.device)

                # Forward + backward
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["input_ids"],
                )
                loss = outputs.loss
                loss.backward()
                losses.append(loss.item())

                # Riemannian descent step on each adapter
                for adapter in adapters:
                    if hasattr(adapter, "full_descent_step"):
                        adapter.full_descent_step(lr=self.lr_riemannian)

                # Zero gradients for next example
                self.model.zero_grad()

            # ── THE TITANIUM PADLOCK ──────────────────────────
            # Freeze the subspace used by these facts so future training
            # cannot overwrite them
            for adapter in adapters:
                if hasattr(adapter, "freeze_current_subspace"):
                    adapter.freeze_current_subspace()

            # Switch back to eval mode for inference
            self.model.eval()

        # ── Save checkpoint ───────────────────────────────────
        elapsed = time.time() - start
        avg_loss = sum(losses) / len(losses) if losses else 0

        self.cycle_count += 1
        self.total_facts_learned += len(examples)

        checkpoint_path = self.checkpoint_dir / f"dream_cycle_{self.cycle_count}.pt"
        adapter_state = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        torch.save(adapter_state, checkpoint_path)

        # Also save as "latest" for easy reload
        latest_path = self.checkpoint_dir / "latest_adapters.pt"
        torch.save(adapter_state, latest_path)

        logger.info(
            f"🔒 Dream Cycle {self.cycle_count} complete: "
            f"{len(examples)} examples, avg_loss={avg_loss:.4f}, "
            f"time={elapsed:.1f}s, saved to {checkpoint_path.name}"
        )

        self._running = False

    @property
    def status(self) -> dict:
        with self.buffer_lock:
            buffer_size = len(self.memory_buffer)
        return {
            "running": self._running,
            "cycle_count": self.cycle_count,
            "total_facts_learned": self.total_facts_learned,
            "buffer_size": buffer_size,
            "min_trigger": self.min_examples,
        }
