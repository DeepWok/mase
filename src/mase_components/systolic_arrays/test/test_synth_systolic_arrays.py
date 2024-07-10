import pytest
from mase_components.synth_runner import run_synth


@pytest.mark.vivado
def test_synth_systolic_arrays():
    run_synth("systolic_arrays")


if __name__ == "__main__":
    test_synth_systolic_arrays()
