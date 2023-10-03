#!/bin/bash
# logicnets.sh - Script for training, testing, and transforming machine learning models using the "ch" command-line tool.

# example ./logicnets.sh 2023-09-25 jsc-s jsc > log.txt
# Check the number of command-line arguments. The script expects exactly three arguments: <date> <model> <dataset>.
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <date> <model> <dataset>"
    exit 1
fi

# Assign command-line arguments to variables for clarity.
date_arg="$1"
model_arg="$2"
data_set_arg="$3"

# Train the model using high precision and specified configuration.
./ch train  ${model_arg} ${data_set_arg} --config configs/logicnets/train_jsc.toml
# Test the trained model using high precision and specified configuration. Load the best checkpoint.
./ch test ${model_arg} ${data_set_arg} --config configs/logicnets/train_jsc.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/training_ckpts/best.ckpt" --load-type pl

# perform prunning on the pretrained high precision network.
./ch transform ${model_arg} ${data_set_arg} --config configs/logicnets/prune.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/training_ckpts/best.ckpt" --load-type pl

# Copy the prunned network weight to a destination directory for later LogicNets quantization.
source_dir="../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt"
destination_dir="../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt_bl"
mkdir -p "$destination_dir"
cp -r "$source_dir" "$destination_dir"

# Perform LogicNets transformations.
# CHECK: Ensure that "integer_logicnets.toml" contains the latest bl_module path : It should be the destination_dir above.
./ch transform ${model_arg} ${data_set_arg} --config configs/logicnets/integer_logicnets.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz 
# Removing any activations followed by a logicnets layer
./ch transform ${model_arg} ${data_set_arg} --config configs/logicnets/fusion.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz    
# Test the quantized model.
./ch test ${model_arg} ${data_set_arg} --config configs/logicnets/integer_logicnets.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz