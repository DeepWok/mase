#!/bin/bash

# example ./lutnet.sh 2023-10-02 toy_convnet cifar10 or ./lutnet.sh 2023-09-26 toy toy_tiny or ./lutnet.sh 2023-10-02 cnv cifar10
# Toml files: lutnet_high_precision_train, lutnet_high_precision_train_1, lutnet_prune, lutnet_init

if [ "$#" -ne 3 ]; then
     echo "Usage: $0 <date> <model> <dataset>"
     exit 1
 fi

 # Assign command-line arguments to variables for clarity.
 date_arg="$1"
 model_arg="$2"
 data_set_arg="$3"

# --------------------------------------------------------
# # high precision training to learn gamma (general case)
# --------------------------------------------------------
# ./ch train ${model_arg} ${data_set_arg} --config configs/lutnet/train.toml
# ./ch test  ${model_arg} ${data_set_arg} --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/training_ckpts/best.ckpt" --load-type pl > "high_precision_training_${model_arg}_without_softmax.txt"
# ./ch transform ${model_arg} ${data_set_arg} --config configs/lutnet/residual_net.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/training_ckpts/best.ckpt" --load-type pl
# ./ch train ${model_arg} ${data_set_arg} --config configs/lutnet/train.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz
# ./ch test  ${model_arg} ${data_set_arg} --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/training_ckpts/transformed_ckpt/graph_module.mz" --load-type mz > high_precision_training.txt

# --------------------------------------------------------
# prunning
# --------------------------------------------------------
# ./ch transform ${model_arg} ${data_set_arg} --config configs/lutnet/lutnet_prune.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/training_ckpts/best-v1.ckpt" --load-type pl
# ./ch test  ${model_arg} ${data_set_arg} --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz > pruned_high_precision.txt



# --------------------------------------------------------
# # high precision training to learn gamma (residual cnv case) ./lutnet.sh 2023-10-25 cnv_residual cifar10
# --------------------------------------------------------
./ch train ${model_arg} ${data_set_arg} --config configs/lutnet/train.toml
./ch test  ${model_arg} ${data_set_arg} --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/training_ckpts/best.ckpt" --load-type pl > "high_precision_training_${model_arg}_without_softmax.txt"

# --------------------------------------------------------
# prunning
# --------------------------------------------------------
./ch transform ${model_arg} ${data_set_arg} --config configs/lutnet/lutnet_prune.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/training_ckpts/best.ckpt" --load-type pl
./ch test  ${model_arg} ${data_set_arg} --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz > pruned_high_precision.txt

# --------------------------------------------------------
# # # binary training
# --------------------------------------------------------
./ch transform ${model_arg} ${data_set_arg} --config configs/lutnet/residual_net_binary_train.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz
./ch test  ${model_arg} ${data_set_arg} --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz > pruned_binary_train_2.txt
./ch train ${model_arg} ${data_set_arg} --config configs/lutnet/train_pruned_bnn.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz > binary_training.txt
./ch test  ${model_arg} ${data_set_arg} --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz > pruned_high_precision_binary_training.txt

# --------------------------------------------------------
# initailiza LUT
# --------------------------------------------------------
./ch transform ${model_arg} ${data_set_arg} --config configs/lutnet/lutnet_init.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/training_ckpts/transformed_ckpt/graph_module.mz" --load-type mz
./ch test  ${model_arg} ${data_set_arg} --accelerator 'cpu' --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz > init_lutnet.txt
./ch train ${model_arg} ${data_set_arg}  --accelerator 'cpu' --config configs/lutnet/train.toml --load "../mase_output/${model_arg}_classification_${data_set_arg}_$date_arg/software/transform/transformed_ckpt/graph_module.mz" --load-type mz
