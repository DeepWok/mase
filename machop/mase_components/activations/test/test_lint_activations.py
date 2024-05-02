from mase_components.linter import run_lint


def test_lint_activations():
    run_lint("activations")


if __name__ == "__main__":
    test_lint_activations()
