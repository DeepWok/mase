from mase_components.linter import run_lint


def test_lint_arithmetic():
    run_lint("arithmetic")


if __name__ == "__main__":
    test_lint_arithmetic()
