import pytest
from mase_components.synth_runner import run_synth


@pytest.mark.vivado
def test_synth_attention():
    run_synth("vision_models/vit", "fixed_self_attention.sv")


if __name__ == "__main__":
    test_synth_attention()
