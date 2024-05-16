from mase_components.linter import run_lint


def test_lint_cast():
    run_lint("cast")


if __name__ == "__main__":
    test_lint_cast()
