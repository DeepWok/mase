from mase_components.linter import run_lint

import pytest


@pytest.mark.skip(reason="Needs to be fixed.")
def test_lint_conv():
    run_lint("convolution_layers")


if __name__ == "__main__":
    test_lint_conv()
