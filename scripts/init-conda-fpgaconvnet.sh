#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script initialise conda for mase x fpgaconvnet
# --------------------------------------------------------------------
set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Create a clone of the mase conda environment
eval "$(conda shell.bash hook)"
conda activate mase
conda create -n mase-fpgaconvnet --clone mase
conda deactivate
conda activate mase-fpgaconvnet

# Check which Python we're using (for safety)
current_python=$(which python)
if [[ ${current_python} = *"envs/mase-fpgaconvnet/bin/python" ]]; then
    python -m pip install --user --upgrade pip
    python -m pip install wandb==0.15.10 graphviz==0.20.1 onnxsim==0.4.33 \
                          onnxruntime==1.15.1 onnxoptimizer==0.3.13 pydot==1.4.2 \
                          protobuf==3.20.3 fpbinary==1.5.5
    python -m pip install xgboost==1.7.2 --extra-index-url https://download.pytorch.org/whl/cpu
else
    echo "Failed to find the Python in mase env. Current Python is at ${current_python}"
    exit 1
fi
