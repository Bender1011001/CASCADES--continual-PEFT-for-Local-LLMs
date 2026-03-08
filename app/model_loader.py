"""
Model loader for CASCADES Chat — loads NF4-quantized model with CASCADES adapters.

Provides hot-swap capability: reload adapter weights without restarting the server.
Designed for 8GB VRAM inference (RTX 3060/4060 class).
"""

import threading
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from cascades.config import AblationConfig, DEFAULT_CONFIG
from cascades.injection import inject_cascades_adapters


class CASCADESModel:
    """Manages the NF4 base model + CASCADES adapters with hot-swap support."""

    def __init__(
        self,
        model_id: str = "./abliterated",
        adapter_weights: Optional[str] = None,
        rank: int = 8,
        config: AblationConfig = DEFAULT_CONFIG,
        device: str = "auto",
    ):
        self.model_id = model_id
        self.rank = rank
        self.config = config
        self.device = device
        self._lock = threading.Lock()
        self._adapter_version = 0
        self._adapter_path: Optional[str] = adapter_weights
        self._loaded = False

        self.model: Optional[nn.Module] = None
        self.tokenizer = None

    def load(self):
        """Load the base model, inject adapters, and optionally load weights."""
        print(f"Loading model: {self.model_id}")
        start = time.time()

        # --- NF4 Quantization ---
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Model ---
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        self.model.config.use_cache = True  # Enable KV cache for inference
        self.model.eval()

        elapsed = time.time() - start
        print(f"Base model loaded in {elapsed:.1f}s")

        # --- Inject CASCADES adapters ---
        inject_cascades_adapters(
            self.model, rank=self.rank, config=self.config
        )
        print(f"CASCADES adapters injected (rank={self.rank})")

        # --- Load pre-trained adapter weights if provided ---
        if self._adapter_path:
            self._load_adapter_weights(self._adapter_path)

        self._loaded = True
        vram_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        print(f"Ready! VRAM: {vram_mb:.0f}MB")

    def _load_adapter_weights(self, path: str):
        """Load adapter state dict from a .pt file."""
        path = Path(path)
        if not path.exists():
            print(f"Warning: adapter weights not found at {path}")
            return

        state = torch.load(path, map_location="cpu", weights_only=True)
        # The saved weights are a flat dict of adapter parameters
        model_state = self.model.state_dict()
        loaded = 0
        for key, value in state.items():
            if key in model_state:
                model_state[key].copy_(value)
                loaded += 1
        print(f"Loaded {loaded} adapter parameters from {path}")

    def swap_adapters(self, new_path: str) -> dict:
        """Hot-swap adapter weights. Thread-safe, near-instant."""
        with self._lock:
            start = time.time()
            self._load_adapter_weights(new_path)
            self._adapter_path = new_path
            self._adapter_version += 1
            elapsed = (time.time() - start) * 1000
            return {
                "status": "ok",
                "version": self._adapter_version,
                "path": new_path,
                "swap_ms": round(elapsed, 1),
            }

    @torch.inference_mode()
    def generate_stream(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ):
        """Stream tokens one by one. Yields text chunks.

        Uses the model's built-in chat template if available.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build prompt using chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                elif role == "user":
                    parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                elif role == "assistant":
                    parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(parts)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.model.device)

        # Set up streaming
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "streamer": streamer,
        }

        # Generate in background thread
        thread = threading.Thread(
            target=self.model.generate, kwargs=generation_kwargs
        )
        thread.start()

        # Yield tokens as they arrive
        for text in streamer:
            yield text

        thread.join()

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        **kwargs,
    ) -> str:
        """Non-streaming generation. Returns full response text."""
        chunks = list(self.generate_stream(messages, max_new_tokens, **kwargs))
        return "".join(chunks)

    @property
    def status(self) -> dict:
        vram_mb = 0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024**2
        return {
            "loaded": self._loaded,
            "model_id": self.model_id,
            "rank": self.rank,
            "adapter_version": self._adapter_version,
            "adapter_path": self._adapter_path,
            "vram_mb": round(vram_mb),
        }
