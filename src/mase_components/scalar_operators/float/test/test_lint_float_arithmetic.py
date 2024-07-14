from mase_components.linter import run_lint


def test_lint_float_arithmetic():
    run_lint("scalar_operators/float")


if __name__ == "__main__":
    test_lint_float_arithmetic()
