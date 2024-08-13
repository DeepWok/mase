import pytest
from mase_components.synth_runner import run_synth
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(f"linter")


@pytest.mark.vivado
def test_synth_fixed_math():
    run_synth("scalar_operators/fixed", "fixed_div.sv")


if __name__ == "__main__":
    test_synth_fixed_math()
