from mase_components.linter import run_lint


def test_lint_common():
    run_lint("common")


if __name__ == "__main__":
    test_lint_common()
