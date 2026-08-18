"""Regression tests for a trained agent checkpoint.

These tests load a real checkpoint and run it through the ReAct tool loop on a
fixed battery of questions. They fail if accuracy drops below the configured
thresholds, so a bad checkpoint cannot be promoted without notice.

Run with the default best checkpoint:
    python -m pytest agent/test_agent_regression.py -v

Override the checkpoint under test:
    CHECKPOINT=checkpoints_web/agent_best.pt python -m pytest agent/test_agent_regression.py -v

Use a stricter gate for promotion decisions:
    python -m pytest agent/test_agent_regression.py -v --min-accuracy 0.85
"""

import os
import pytest
import torch

from agent.chat import load_checkpoint
from agent.train_agent import evaluate_agent
from agent.tools import Toolbox


# Base regression battery: single-step and multi-hop non-web questions.
# These should pass on any agent trained before web_search was added.
BASE_CASES = [
    {"question": "What is 12 + 8?", "expected": "20", "kind": "numeric"},
    {"question": "Calculate 15 * 4.", "expected": "60", "kind": "numeric"},
    {"question": "What is the project?", "expected": "newllm", "kind": "exact"},
    {"question": "Retrieve the leader.", "expected": "grug", "kind": "exact"},
    {"question": "What is the current date and time?", "kind": "date"},
    {"question": "What is 7 * 6?", "expected": "42", "kind": "numeric"},
    {"question": "Look up the version.", "expected": "0.1", "kind": "exact"},
    {"question": "Add 5 to the version.", "expected": "5.1", "kind": "numeric"},
    {"question": "Multiply 3 and 4, then add the length of the leader.", "expected": "16", "kind": "numeric"},
]

# Extra web_search cases. These are only required when the --web flag is set,
# so the legacy best checkpoint can still pass the default base gate.
WEB_CASES = [
    {"question": "What is the capital of france?", "expected": "Paris", "kind": "exact"},
    {"question": "Who is the president of the united states?", "expected": "Alice Johnson", "kind": "exact"},
    {"question": "How many planets are there?", "expected": "8", "kind": "exact"},
    {"question": "What is the speed of light?", "expected": "299792458 m/s", "kind": "exact"},
    {"question": "Look up the boiling point of water.", "expected": "100 degrees Celsius", "kind": "exact"},
    {"question": "What is the largest planet?", "expected": "Jupiter", "kind": "exact"},
    {"question": "What is the capital of japan plus 5?", "expected": "10", "kind": "numeric"},
    {"question": "How many planets are there times 2?", "expected": "16", "kind": "numeric"},
]


def regression_cases(include_web: bool = False):
    cases = list(BASE_CASES)
    if include_web:
        cases.extend(WEB_CASES)
    return cases


def _opt(request, name):
    # The options are registered in agent/conftest.py, so getoption can use the
    # long flag form directly.
    return request.config.getoption(name)


@pytest.fixture(scope="module")
def agent_model(request):
    path = _opt(request, "--checkpoint")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_checkpoint(path, device=device)


@pytest.fixture(scope="module")
def eval_result(agent_model, request):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    include_web = _opt(request, "--web")
    return evaluate_agent(agent_model, regression_cases(include_web), device, toolbox=Toolbox())


def test_checkpoint_exists_and_loads(agent_model):
    assert agent_model is not None
    assert agent_model.count_parameters() > 0


def test_overall_accuracy(eval_result, request):
    acc, _, _ = eval_result
    threshold = _opt(request, "--min-accuracy")
    assert acc >= threshold, f"overall accuracy {acc:.2%} below {threshold:.0%}"


def test_numeric_accuracy(eval_result, request):
    _, per_kind, _ = eval_result
    threshold = _opt(request, "--min-numeric")
    numeric = per_kind.get("numeric", 0.0)
    assert numeric >= threshold, f"numeric accuracy {numeric:.2%} below {threshold:.0%}"


def test_exact_accuracy(eval_result, request):
    _, per_kind, _ = eval_result
    threshold = _opt(request, "--min-exact")
    exact = per_kind.get("exact", 0.0)
    assert exact >= threshold, f"exact accuracy {exact:.2%} below {threshold:.0%}"


def test_required_questions(eval_result):
    _, _, details = eval_result
    # Core required questions that any promoted agent must answer correctly.
    # Multi-hop web/math combos are deliberately excluded from the default
    # base gate; they are tested via --web.
    core = [
        "What is 12 + 8?",
        "Calculate 15 * 4.",
        "What is the project?",
        "Retrieve the leader.",
        "What is the current date and time?",
        "What is 7 * 6?",
        "Look up the version.",
    ]
    failed = [d["question"] for d in details if d["question"] in core and not d["ok"]]
    # Report all failures explicitly so the user sees which questions regressed.
    assert not failed, f"regression on {len(failed)} core question(s): {failed}"
