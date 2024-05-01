from mase_components.linter import run_lint


def test_lint_fixed_arithmetic():
    run_lint("fixed_arithmetic")


if __name__ == "__main__":
    test_lint_fixed_arithmetic()
