#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script initialise conda for mase
# --------------------------------------------------------------------
set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# create and activate mase env
conda env create -f ${DIR}/../machop/environment.yml
eval "$(conda shell.bash hook)"
conda activate mase

# check which python
current_python=$(which python)
if [[ ${current_python} = *"envs/mase/bin/python" ]]; then
    python -m pip install --user --upgrade pip &&
        python -m pip install -r ${DIR}/../machop/requirements.txt

    if [[ $? -eq 0 ]]; then
        echo "✅ Successfully installed all the requirements"
    else
        echo "❌ Failed to install the requirements"
        exit 1
    fi
else
    echo "❌ Failed to find the Python in mase env. Current Python is at ${current_python}"
    exit 1
fi
