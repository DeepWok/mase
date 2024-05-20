from mase_components.linter import run_lint


def test_lint_systolic_arrays():
    run_lint("systolic_arrays")


if __name__ == "__main__":
    test_lint_systolic_arrays()
