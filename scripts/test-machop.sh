#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This is a script to test command line interfaces
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASE=$SCRIPT_DIR/..

cd $MASE/machop

##### Basic training and testing
# training
./ch train jsc-tiny jsc --max-epochs 3 --batch-size 256 --accelerator cpu --project tmp --debug
# test
./ch test jsc-tiny jsc --accelerator cpu --debug --load ../mase_output/tmp/software/training_ckpts/best.ckpt --load-type pl

##### Graph-level testing
# transfrom on graph level
./ch transform --config configs/examples/jsc_toy_by_type.toml --task cls --accelerator=cpu --load ../mase_output/tmp/software/training_ckpts/best.ckpt --load-type pl
# search command
./ch search --config configs/examples/jsc_toy_by_type.toml --task cls --accelerator=cpu --load ../mase_output/tmp/software/training_ckpts/best.ckpt --load-type pl
# train searched network
./ch train jsc-tiny jsc --max-epochs 3 --batch-size 256 --accelerator cpu --project tmp --debug --load ../mase_output/jsc-tiny/software/transform/transformed_ckpt/graph_module.mz --load-type mz

##### Module-level testing
# transfrom on graph level
./ch transform --config configs/examples/jsc_toy_by_type_module.toml --task cls --accelerator=cpu --load ../mase_output/tmp/software/training_ckpts/best.ckpt --load-type pl
# train the transformed network
./ch train jsc-tiny jsc --max-epochs 3 --batch-size 256 --accelerator cpu --project tmp --debug --load ../mase_output/jsc-tiny/software/transform/transformed_ckpt/state_dict.pt --load-type pt
