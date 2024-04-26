from mase_components.linter import run_lint

import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_lint_binary_arith():
    run_lint("binary_arith")


if __name__ == "__main__":
    test_lint_binary_arith()
