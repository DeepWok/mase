import pytest
from mase_components.linter import run_lint


@pytest.mark.dev
def test_lint_fixed_math():
    run_lint("scalar_operators/fixed")


if __name__ == "__main__":
    test_lint_fixed_math()
