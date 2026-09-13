"""Shared pytest configuration for the agent package."""

import os


def pytest_addoption(parser):
    parser.addoption(
        "--checkpoint",
        default=os.environ.get("CHECKPOINT", "checkpoints_v2_M/agent_best.pt"),
        help="Path to the agent checkpoint to evaluate",
    )
    parser.addoption(
        "--min-accuracy",
        type=float,
        default=float(os.environ.get("MIN_ACCURACY", "0.75")),
        help="Minimum overall accuracy required",
    )
    parser.addoption(
        "--min-numeric",
        type=float,
        default=float(os.environ.get("MIN_NUMERIC", "0.60")),
        help="Minimum numeric-case accuracy required",
    )
    parser.addoption(
        "--min-exact",
        type=float,
        default=float(os.environ.get("MIN_EXACT", "0.80")),
        help="Minimum exact-string accuracy required",
    )
    parser.addoption(
        "--web",
        action="store_true",
        default=os.environ.get("WEB", "").lower() in ("1", "true", "yes"),
        help="Include web_search questions in the regression battery",
    )
