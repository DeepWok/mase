#!/bin/bash

# example ./lutnet.sh 2023-08-20

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <date>"
    exit 1
fi

date_arg="$1"

# high precision training
./ch train toy --config configs/lutnet/lutnet_high_precision_train.toml
./ch transform toy --config configs/lutnet/lutnet_high_precision_train_1.toml --load "../mase_output/toy_classification_cifar10_$date_arg/software/training_ckpts/best.ckpt" --load-type pl
./ch train toy --config configs/lutnet/lutnet_high_precision_train_1.toml --load "../mase_output/toy_classification_cifar10_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz
# ./ch test toy --config configs/lutnet/lutnet_high_precision_train_1.toml  --load "../mase_output/toy_classification_cifar10_$date_arg/software/training_ckpts/best-v1.ckpt" --load-type pl

# save the high precision weight to other folder
source_dir="../mase_output/toy_classification_cifar10_$date_arg/software/transform/transformed_ckpt"
destination_dir="../mase_output/toy_classification_cifar10_$date_arg/software/transform/transformed_ckpt_bl"
mkdir -p "$destination_dir"
cp -r "$source_dir" "$destination_dir"

# pruning
./ch transform toy --config configs/lutnet/lutnet_prune.toml --load "../mase_output/toy_classification_cifar10_$date_arg/software/training_ckpts/best-v1.ckpt" --load-type pl
# initailiza LUT
./ch transform toy --config configs/lutnet/lutnet_init.toml --load "../mase_output/toy_classification_cifar10_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz
