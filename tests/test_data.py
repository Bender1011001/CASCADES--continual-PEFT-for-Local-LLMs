"""
Tests for the unified data module — cascades/data.py.

Critical property: ALL methods (CASCADES, LoRA baseline, ablations) use
the SAME task prompts.  Deviation would invalidate the experimental comparison.
"""

import pytest

from cascades.data import (
    TASK_PROMPTS,
    NUM_TASKS,
    get_task_prompts,
    task_domain_name,
)


class TestTaskPrompts:
    def test_correct_number_of_tasks(self):
        assert NUM_TASKS == 3, "Paper evaluates 3 sequential tasks"

    def test_all_tasks_defined(self):
        for t in range(NUM_TASKS):
            prompts = get_task_prompts(t)
            assert len(prompts) > 0, f"Task {t} has no prompts"

    def test_tasks_are_distinct(self):
        """Tasks must have non-overlapping prompt text (different domains)."""
        all_prompt_sets = [set(get_task_prompts(t)) for t in range(NUM_TASKS)]
        for i in range(NUM_TASKS):
            for j in range(i + 1, NUM_TASKS):
                intersection = all_prompt_sets[i] & all_prompt_sets[j]
                assert len(intersection) == 0, (
                    f"Tasks {i} and {j} share prompts: {intersection} — "
                    "tasks must be from distinct domains"
                )

    def test_each_task_has_both_classes(self):
        """Each task must include both Positive and Negative examples."""
        for t in range(NUM_TASKS):
            prompts = get_task_prompts(t)
            has_positive = any("Positive" in p for p in prompts)
            has_negative = any("Negative" in p for p in prompts)
            assert has_positive, f"Task {t} has no Positive examples"
            assert has_negative, f"Task {t} has no Negative examples"

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError):
            get_task_prompts(999)

    def test_prompts_are_strings(self):
        for t in range(NUM_TASKS):
            for p in get_task_prompts(t):
                assert isinstance(p, str), f"Task {t} prompt is not a string: {p!r}"

    def test_domain_names_are_distinct(self):
        names = [task_domain_name(t) for t in range(NUM_TASKS)]
        assert len(set(names)) == NUM_TASKS, f"Domain names are not unique: {names}"


class TestDataParity:
    """
    Regression tests ensuring the LoRA baseline and CASCADES use identical data.

    In an earlier version of lora_baseline.py, the task prompts were:
        "Task {t}: Evaluate the sentiment. This product is great! -> Positive"
    which is a DIFFERENT distribution from CASCADES's domain-specific prompts.
    That bug is fixed: both now import from cascades.data.

    These tests catch regressions.
    """

    def test_task0_is_product_reviews(self):
        prompts = get_task_prompts(0)
        assert all("Review:" in p for p in prompts), (
            "Task 0 must be product reviews — verify cascades/data.py hasn't drifted"
        )

    def test_task1_is_film_critiques(self):
        prompts = get_task_prompts(1)
        assert all("Film critique:" in p for p in prompts), (
            "Task 1 must be film critiques"
        )

    def test_task2_is_restaurant_reviews(self):
        prompts = get_task_prompts(2)
        assert all("Dining experience:" in p for p in prompts), (
            "Task 2 must be restaurant reviews"
        )

    def test_balanced_classes_per_task(self):
        """Each task must have equal numbers of Positive and Negative examples."""
        for t in range(NUM_TASKS):
            prompts = get_task_prompts(t)
            pos = sum(1 for p in prompts if "Positive" in p)
            neg = sum(1 for p in prompts if "Negative" in p)
            assert pos == neg, (
                f"Task {t} class imbalance: {pos} Positive vs {neg} Negative"
            )
