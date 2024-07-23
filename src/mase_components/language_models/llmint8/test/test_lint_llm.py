from mase_components.linter import run_lint
import pytest


@pytest.mark.dev
def test_lint_llm():
    run_lint("llm")


if __name__ == "__main__":
    test_lint_llm()
