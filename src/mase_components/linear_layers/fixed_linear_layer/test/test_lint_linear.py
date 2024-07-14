from mase_components.linter import run_lint

import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_lint_linear():
    run_lint("linear_layers/fixed_linear_layer")


if __name__ == "__main__":
    test_lint_linear()
