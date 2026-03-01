"""
Tests for the unified data module — cascades/data.py.

Validates the CoT JSONL-based data loader, task file definitions,
and per-example loss diagnostic interface.
"""

import os

import pytest

from cascades.data import (
    NUM_TASKS,
    TASK_FILES,
    TASK_NAMES,
)


class TestTaskDefinitions:
    def test_correct_number_of_tasks(self):
        assert NUM_TASKS == 3, "CASCADES evaluates 3 sequential continual learning tasks"

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
        """Task files should follow the task<N>_*_cot.jsonl naming convention."""
        for i, path in enumerate(TASK_FILES):
            basename = os.path.basename(path)
            assert f"task{i}" in basename, (
                f"Task {i} file should contain 'task{i}' in name: {path}"
            )


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
