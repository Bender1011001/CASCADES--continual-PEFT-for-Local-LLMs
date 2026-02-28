"""
Unit tests for cascades.eval — answer extraction, normalization, matching.

These tests verify the generative evaluation module without requiring
a GPU or model — they test the text processing and matching logic only.
"""

import pytest
from cascades.eval import (
    extract_answer_from_cot,
    normalize_answer,
    answers_match,
    token_f1,
    build_inference_prompt,
    STRUCTURED_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# A. Answer extraction tests
# ---------------------------------------------------------------------------

class TestExtractAnswerFromCoT:
    """Tests for extracting the final answer after </think> tags."""

    def test_standard_cot_format(self):
        """The training data format: <think>...\n</think>\n\n204"""
        text = "<think>\nSome reasoning steps here.\n</think>\n\n204"
        assert extract_answer_from_cot(text) == "204"

    def test_cot_with_multiword_answer(self):
        text = "<think>\nReasoning.\n</think>\n\nB is a Knave and C is a Knight."
        assert extract_answer_from_cot(text) == "B is a Knave and C is a Knight."

    def test_cot_with_math_answer(self):
        text = "<think>\nDerivation.\n</think>\n\n\\Theta(n^2 \\log n)"
        assert extract_answer_from_cot(text) == "\\Theta(n^2 \\log n)"

    def test_cot_with_equation_answer(self):
        text = "<think>\nWork.\n</think>\n\nn = 1"
        assert extract_answer_from_cot(text) == "n = 1"

    def test_no_think_tags_returns_full_text(self):
        """When model doesn't use CoT format, return the full text."""
        text = "The answer is 42."
        result = extract_answer_from_cot(text)
        assert "42" in result

    def test_answer_prefix_extraction(self):
        """Chat models often say 'The answer is X'."""
        text = "After careful analysis, the answer is 204."
        result = extract_answer_from_cot(text)
        assert "204" in result

    def test_final_answer_prefix(self):
        text = "Let me think about this. Final answer: 42"
        result = extract_answer_from_cot(text)
        assert "42" in result

    def test_empty_after_think(self):
        """Edge case: </think> with nothing after it."""
        text = "<think>\nReasoning here.\n</think>"
        result = extract_answer_from_cot(text)
        # Should fall back to last line strategy
        assert result  # non-empty

    def test_multiline_after_think(self):
        """Only the content after </think> should be extracted."""
        text = "<think>\nStep 1\nStep 2\n</think>\n\nThe complexity is O(n log n)"
        result = extract_answer_from_cot(text)
        assert "O(n log n)" in result

    def test_case_insensitive_think_tags(self):
        text = "<THINK>\nReasoning.\n</THINK>\n\n7"
        assert extract_answer_from_cot(text) == "7"


# ---------------------------------------------------------------------------
# B. Answer normalization tests
# ---------------------------------------------------------------------------

class TestNormalizeAnswer:
    """Tests for normalizing answers before comparison."""

    def test_basic_normalization(self):
        assert normalize_answer("  204  ") == "204"

    def test_lowercase(self):
        assert normalize_answer("B is a Knave") == "b is a knave"

    def test_strip_latex_dollars(self):
        assert normalize_answer("$n = 1$") == "n = 1"

    def test_strip_period(self):
        assert normalize_answer("42.") == "42"

    def test_strip_quotes(self):
        assert normalize_answer('"hello"') == "hello"

    def test_remove_therefore(self):
        assert normalize_answer("Therefore, 42") == "42"

    def test_remove_the_answer_is(self):
        assert normalize_answer("The answer is 204") == "204"

    def test_whitespace_collapse(self):
        assert normalize_answer("n  =   1") == "n = 1"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_unicode_normalization(self):
        # fi ligature should be decomposed
        assert normalize_answer("ﬁne") == "fine"


# ---------------------------------------------------------------------------
# C. Answer matching tests
# ---------------------------------------------------------------------------

class TestAnswersMatch:
    """Tests for the multi-level matching logic."""

    def test_exact_match(self):
        assert answers_match("204", "204") is True

    def test_normalized_match(self):
        assert answers_match("  204 ", "204") is True

    def test_case_insensitive_match(self):
        assert answers_match("B is a Knave", "b is a knave") is True

    def test_containment_match_gen_contains_ref(self):
        """Generated answer contains the reference."""
        assert answers_match("The answer is 204 minutes", "204", strict=False) is True

    def test_containment_match_ref_contains_gen(self):
        """Reference contains the generated answer."""
        assert answers_match("42", "The result is 42", strict=False) is True

    def test_strict_rejects_containment(self):
        assert answers_match("The answer is 204 minutes", "204", strict=True) is False

    def test_numeric_equivalence(self):
        assert answers_match("204.0", "204", strict=False) is True

    def test_numeric_with_commas(self):
        assert answers_match("1,000", "1000", strict=False) is True

    def test_no_match(self):
        assert answers_match("42", "204", strict=False) is False

    def test_empty_strings_dont_match(self):
        assert answers_match("", "204") is False
        assert answers_match("42", "") is False
        assert answers_match("", "") is False

    def test_latex_stripped_match(self):
        assert answers_match("$n = 1$", "n = 1") is True

    def test_trailing_period_match(self):
        assert answers_match("204.", "204") is True

    def test_therefore_prefix_match(self):
        assert answers_match("Therefore, 42", "42") is True


# ---------------------------------------------------------------------------
# D. System prompt tests
# ---------------------------------------------------------------------------

class TestStructuredPrompt:
    """Tests for the structured inference prompting."""

    def test_system_prompt_contains_think_instruction(self):
        assert "<think>" in STRUCTURED_SYSTEM_PROMPT
        assert "</think>" in STRUCTURED_SYSTEM_PROMPT

    def test_system_prompt_has_example(self):
        assert "Example format" in STRUCTURED_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# E. New extraction patterns (v10 additions)
# ---------------------------------------------------------------------------

class TestNewExtractionPatterns:
    """Tests for boxed, answer tags, and markdown bold patterns."""

    def test_boxed_latex(self):
        text = "So the answer is \\boxed{42}."
        assert extract_answer_from_cot(text) == "42"

    def test_boxed_latex_expression(self):
        text = "We conclude that \\boxed{n^2 + 1} is the result."
        assert extract_answer_from_cot(text) == "n^2 + 1"

    def test_answer_xml_tags(self):
        text = "After analysis: <answer>204</answer>"
        assert extract_answer_from_cot(text) == "204"

    def test_answer_xml_multiword(self):
        text = "<answer>B is a Knave</answer>"
        assert extract_answer_from_cot(text) == "B is a Knave"

    def test_markdown_bold_answer(self):
        text = "After reasoning, **Answer:** 42"
        result = extract_answer_from_cot(text)
        assert "42" in result

    def test_markdown_bold_final_answer(self):
        text = "**Final Answer:** The complexity is O(n)"
        result = extract_answer_from_cot(text)
        assert "O(n)" in result

    def test_think_tags_take_priority(self):
        """</think> should win over <answer> tags when both present."""
        text = "<think>reasoning</think>\n\n42\n<answer>99</answer>"
        assert extract_answer_from_cot(text) == "42\n<answer>99</answer>"


# ---------------------------------------------------------------------------
# F. Token-level F1 tests
# ---------------------------------------------------------------------------

class TestTokenF1:
    """Tests for the soft token-level F1 metric."""

    def test_perfect_match(self):
        assert token_f1("204", "204") == 1.0

    def test_no_overlap(self):
        assert token_f1("42", "204") == 0.0

    def test_partial_overlap(self):
        f1 = token_f1("the answer is 42 minutes", "42")
        assert 0.0 < f1 < 1.0
        assert f1 > 0.3  # "42" should match

    def test_empty_strings(self):
        assert token_f1("", "204") == 0.0
        assert token_f1("42", "") == 0.0

    def test_case_insensitive(self):
        assert token_f1("B is a Knave", "b is a knave") == 1.0

    def test_multiword_match(self):
        f1 = token_f1("B is a Knave and C is a Knight", "B is a Knave and C is a Knight")
        assert f1 == 1.0

