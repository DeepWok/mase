#! /usr/bin/bash

cd ../../software/

# train resnet18 on MNIST, save the project to ${mase-tools}/mase-output/resnet18-cifar10
./chop --train --model resnet18 --task cls --dataset cifar10 --max-epochs 5 --cpu 8 --project resnet18-cifar10

# load the last saved checkpoint, run on validation set
# though validation accuracy and loss were evaluated during during, we still run validation again for demo
./chop --validate-sw --model resnet18 --task cls --dataset cifar10 --cpu 8 --project resnet18-cifar10 --load ../mase_output/resnet18-cifar10/software/checkpoints/last.ckpt --load-type pl

# load the last saved checkpoint, run on test set
./chop --test-sw --model resnet18 --task cls --dataset cifar10 --cpu 8 --project resnet18-cifar10 --load ../mase_output/resnet18-cifar10/software/checkpoints/last.ckpt --load-type pl

# visualization in tensorboard
tensorboard --logdir ../../mase-output/resnet18-cifar10
