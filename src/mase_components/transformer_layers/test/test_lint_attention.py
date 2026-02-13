from mase_components.linter import run_lint

import pytest


@pytest.mark.dev
def test_lint_attention():
    run_lint("transformer_layers")


if __name__ == "__main__":
    test_lint_attention()
