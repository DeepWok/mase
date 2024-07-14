from mase_components.linter import run_lint


def test_lint_axi():
    run_lint("interface/axi")


if __name__ == "__main__":
    test_lint_axi()
