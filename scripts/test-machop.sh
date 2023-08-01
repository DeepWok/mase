#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script runs the unitest for mase software stack
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASE=${SCRIPT_DIR}/..

for f in $(find "${MASE}/machop/test" -name "*.py")
do
   python3 $f
done
