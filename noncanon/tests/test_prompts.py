"""The training-set filter: catches copies of a training problem, keeps new ones."""

import pytest

from noncanon.prompts import TrainingFilter, normalize

TRAIN = {
    "setA": [
        r"Find the value of the base $b$ such that $6651_b + 115_b = 10066_b$.",
        r"Let $ABC$ be a triangle with $AB = 13$, $BC = 14$, and $CA = 15$. Find the area of the circle through $A$, $B$, $C$ "
        r"and the point $D$ on $BC$ with $BD = 5$, given that the answer is an integer.",
    ]
}


@pytest.fixture(scope="module")
def filt():
    return TrainingFilter(TRAIN)


def test_normalize_ignores_case_spacing_punctuation_and_user_prefix():
    assert normalize("User: Find  $x$.") == normalize("find $ x $ .") == "findx"


def test_exact_copy_with_different_latex_spacing_is_caught(filt):
    assert filt.match(r"Find the value of the base $ b $ such that $6651_{b}+115_{b}=10066_{b}$.") == "setA:exact"


def test_training_copy_with_appended_instruction_is_caught_by_prefix(filt):
    long_problem = TRAIN["setA"][1]
    assert filt.match(long_problem + " Give your answer as a single integer with no explanation.") == "setA:prefix"


def test_short_problem_with_appended_text_is_not_a_prefix_match(filt):
    # The short problem's normalized text is under PREFIX_CHARS, so an appended
    # instruction changes the prefix; only an exact match can catch it.
    assert filt.match(TRAIN["setA"][0] + " Answer with an integer.") is None


def test_different_problem_with_a_common_opening_is_kept(filt):
    assert filt.match(r"Let $ABC$ be a triangle with $AB = 13$, $BC = 14$, and $CA = 15$. Find the length of the altitude from $A$.") is None


def test_unrelated_problem_is_kept(filt):
    assert filt.match("How many positive divisors does 2024 have?") is None
