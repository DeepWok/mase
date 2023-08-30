#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script runs the unitest for mase software stack
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MASE=${SCRIPT_DIR}/..

for f in $(find "${MASE}/machop/test" -name "*.py"); do
   printf "%10s\n" "============= Running $f ============="
   python3 $f
   retVal=$?
   if [ $retVal -ne 0 ]; then
      printf "%10s\n" "$============= Failed to run $f ============="
      exit $retVal
   else
      printf "%10s\n" "============= Passed $f ============="
   fi
done

# Test the bash command as follows, as a temporary solution
# cd $MASE/machop
# ./ch transform --config configs/examples/opt_uniform.toml --task lm --cpu 1 --project-dir .
