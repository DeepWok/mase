from mase_components.linter import run_lint


def test_lint_buffers():
    run_lint("buffers")


if __name__ == "__main__":
    test_lint_buffers()
