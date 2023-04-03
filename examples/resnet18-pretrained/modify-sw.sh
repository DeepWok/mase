# Load a trained resnet18, quantize resnet18, and save it
./chop --modify-sw ./configs/modify-sw/integer.toml --model resnet18-tv-imagenet --dataset cifar10
