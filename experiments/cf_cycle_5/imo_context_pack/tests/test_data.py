"""
Tests for the unified data module — cascades/data.py.

Validates the CoT JSONL-based data loader, task file definitions,
and per-example loss diagnostic interface.
"""

import json
import os

import pytest
import torch

import cascades.data as data_module
from cascades.data import (
    NUM_TASKS,
    TASK_FILES,
    TASK_NAMES,
)


class _Encoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _CharacterTokenizer:
    """Tiny CPU-only tokenizer with the prepare_data tokenizer interface."""

    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        text = "".join(
            f"<|{message['role']}|>\n{message['content']}\n" for message in messages
        )
        if add_generation_prompt:
            text += "<|assistant|>\n"
        return text

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        # Keep ids positive and away from the ignore_index sentinel (-100).
        return _Encoding([(ord(char) % 251) + 1 for char in text])


class _ChunkTokenizer(_CharacterTokenizer):
    """Subword-ish tokenizer that stays CPU-only while preserving truncation bugs."""

    chunk_size = 3

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return _Encoding(
            [
                (sum(ord(char) for char in text[i : i + self.chunk_size]) % 251) + 1
                for i in range(0, len(text), self.chunk_size)
            ]
        )


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class TestTaskDefinitions:
    def test_correct_number_of_tasks(self):
        assert NUM_TASKS == 4, "CASCADES current4 evaluates 4 sequential continual learning tasks"

    def test_task_files_match_count(self):
        assert len(TASK_FILES) == NUM_TASKS

    def test_task_names_match_count(self):
        assert len(TASK_NAMES) == NUM_TASKS

    def test_task_names_are_distinct(self):
        names = list(TASK_NAMES.values())
        assert len(set(names)) == NUM_TASKS, f"Task names are not unique: {names}"

    def test_task_files_are_jsonl(self):
        for path in TASK_FILES:
            assert path.endswith(".jsonl"), f"Task file must be JSONL: {path}"

    def test_task_files_have_expected_names(self):
        """Task files should be JSONL files with stable task identifiers in their names."""
        basenames = [os.path.basename(path) for path in TASK_FILES]
        assert len(set(basenames)) == NUM_TASKS, f"Task files are not unique: {basenames}"
        for basename in basenames:
            assert basename.startswith("task"), f"Task file should start with 'task': {basename}"
            assert basename.endswith(".jsonl"), f"Task file should be JSONL: {basename}"


class TestDataFileIntegrity:
    """Tests that run only if data files exist (skipped in CI without data)."""

    @pytest.fixture(autouse=True)
    def check_data_exists(self):
        if not os.path.exists(TASK_FILES[0]):
            pytest.skip("Training data not present (expected in data/ directory)")

    def test_files_exist(self):
        for path in TASK_FILES:
            assert os.path.exists(path), f"Missing data file: {path}"

    def test_files_not_empty(self):
        for path in TASK_FILES:
            size = os.path.getsize(path)
            assert size > 100, f"Data file suspiciously small ({size} bytes): {path}"

    def test_files_are_valid_jsonl(self):
        """Each line should be valid JSON with 'prompt' and 'response' keys."""
        import json
        for path in TASK_FILES:
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    assert "prompt" in obj, f"{path} line {i}: missing 'prompt' key"
                    assert "response" in obj, f"{path} line {i}: missing 'response' key"

    def test_minimum_examples_per_task(self):
        """Each task should have at least 50 examples."""
        for path in TASK_FILES:
            with open(path, "r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
            assert count >= 50, (
                f"Task file {path} has only {count} examples (minimum 50)"
            )

    def test_responses_contain_think_tags(self):
        """Training responses should use <think> CoT format."""
        import json
        for path in TASK_FILES:
            with open(path, "r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            think_count = sum(1 for l in lines if "<think>" in l["response"])
            ratio = think_count / len(lines) if lines else 0
            assert ratio > 0.8, (
                f"{path}: only {ratio*100:.0f}% responses have <think> tags "
                f"(expected >80%)"
            )


class TestPrepareDataSupervision:
    def test_reasoning3_gsm8k_max_length_256_has_no_all_ignored_label_batches(self, monkeypatch):
        """Reduced-memory reasoning3 task 0 must not emit all--100 labels."""
        reasoning3_task0 = "data/task0_gsm8k_cot.jsonl"
        if not os.path.exists(reasoning3_task0):
            pytest.skip("reasoning3 GSM8K data not present")

        monkeypatch.setattr(data_module, "TASK_FILES", [reasoning3_task0])

        loader = data_module.prepare_data(
            _ChunkTokenizer(), task_number=0, base_seed=42, max_length=256
        )

        all_ignored_batches = []
        for batch_idx, (_input_ids, _attention_mask, labels) in enumerate(loader, start=1):
            if labels.eq(-100).all().item():
                all_ignored_batches.append(batch_idx)

        assert all_ignored_batches == [], (
            "prepare_data() produced batches whose labels are entirely -100 at "
            f"max_length=256: {all_ignored_batches}"
        )

    def test_long_prompt_truncation_retains_at_least_one_supervised_label(
        self, monkeypatch, tmp_path
    ):
        """A long prompt should not consume the full truncation budget."""
        data_path = tmp_path / "long_prompt.jsonl"
        _write_jsonl(
            data_path,
            [
                {
                    "prompt": "x" * 400,
                    "response": "<think>reason</think>\n<answer>42</answer>",
                }
            ],
        )
        monkeypatch.setattr(data_module, "TASK_FILES", [str(data_path)])

        loader = data_module.prepare_data(
            _CharacterTokenizer(),
            task_number=0,
            base_seed=42,
            use_system_prompt=False,
            max_length=256,
        )
        _input_ids, _attention_mask, labels = next(iter(loader))

        assert labels.ne(-100).sum().item() >= 1
        assert labels.shape[1] <= 256

    def test_batch_shape_and_prompt_masking_remain_valid(self, monkeypatch, tmp_path):
        data_path = tmp_path / "short_prompt.jsonl"
        row = {"prompt": "What is 2+2?", "response": "<think>2+2=4</think>\n<answer>4</answer>"}
        _write_jsonl(data_path, [row])
        monkeypatch.setattr(data_module, "TASK_FILES", [str(data_path)])

        tokenizer = _CharacterTokenizer()
        loader = data_module.prepare_data(
            tokenizer, task_number=0, base_seed=42, max_length=1024
        )
        input_ids, attention_mask, labels = next(iter(loader))

        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": data_module.STRUCTURED_SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_length = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
        response_length = len(
            tokenizer(row["response"] + tokenizer.eos_token, add_special_tokens=False).input_ids
        )

        assert input_ids.shape == attention_mask.shape == labels.shape
        assert input_ids.shape == torch.Size([1, prompt_length + response_length])
        assert attention_mask.eq(1).all().item()
        assert labels[0, :prompt_length].eq(-100).all().item()
        assert torch.equal(labels[0, prompt_length:], input_ids[0, prompt_length:])
        assert labels[0, prompt_length:].ne(-100).all().item()
