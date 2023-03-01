#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script initialise conda for mase  
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

conda env create -f "${DIR}"/../software/environment.yml
eval "$(conda shell.bash hook)"
conda activate mase
pip3 install --user --upgrade pip
pip3 install -r ../software/requirements.txt
