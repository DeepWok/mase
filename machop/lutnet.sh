#!/bin/bash

# example ./lutnet.sh 2023-08-20 toy-conv
# Toml files: lutnet_high_precision_train, lutnet_high_precision_train_1, lutnet_prune, lutnet_init

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <date>"
    exit 1
fi

date_arg="$1"
model_arg="$2"

# high precision training
./ch train --config configs/lutnet/lutnet_high_precision_train.toml
./ch transform --config configs/lutnet/lutnet_high_precision_train_1.toml --load "../mase_output/${model_arg}_classification_cifar10_$date_arg/software/training_ckpts/best.ckpt" --load-type pl
./ch train --config configs/lutnet/lutnet_high_precision_train_1.toml --load "../mase_output/${model_arg}_classification_cifar10_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz
# ./ch test --config configs/lutnet/lutnet_high_precision_train_1.toml  --load "../mase_output/toy_classification_cifar10_$date_arg/software/training_ckpts/best-v1.ckpt" --load-type pl

# save the high precision weight to other folder
source_dir="../mase_output/${model_arg}_classification_cifar10_$date_arg/software/transform/transformed_ckpt"
destination_dir="../mase_output/${model_arg}_classification_cifar10_$date_arg/software/transform/transformed_ckpt_bl"
mkdir -p "$destination_dir"
cp -r "$source_dir" "$destination_dir"

# pruning
./ch transform --config configs/lutnet/lutnet_prune.toml --load "../mase_output/${model_arg}_classification_cifar10_$date_arg/software/training_ckpts/best-v1.ckpt" --load-type pl
# initailiza LUT
./ch transform --config configs/lutnet/lutnet_init.toml --load "../mase_output/${model_arg}_classification_cifar10_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz
