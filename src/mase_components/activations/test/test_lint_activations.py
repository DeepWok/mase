import pytest
from mase_components.linter import run_lint

from mase_components.helper.generate_memory import generate_sv_lut, FUNCTION_TABLE


@pytest.mark.dev
def test_lint_activations():
    for func, _ in FUNCTION_TABLE.items():
        generate_sv_lut(func, 8, 4, data_width=8, f_width=4, path_with_dtype=False)
    run_lint("activations")


if __name__ == "__main__":
    test_lint_activations()
