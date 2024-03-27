'''
This is a test file for evaluating the prune methods desgined by group 10.
To run the file, please first train/select a checkpoint to train and load it at original_path.
Then run the pruning in the command line with:
./ch transform --config PATH_TO_PRUNING_TOML_CONFIG --load PATH_TO_CHECKPOINT --load-type pl --task cls --cpu=0
(config path defalut in /mase/machop/configs/tests/prune/Group10/vgg7_tensor_element.toml)
and retraining command:
./ch transform --config PATH_TO_RETRAINING_TOML_CONFIG --load PATH_TO_CHECKPOINT --load-type pl --task cls --cpu=0
(config path defalut in /mase/machop/configs/tests/prune/Group10/vgg7_retrain.toml)
Lastly load the pruned state dictionary and retrained model at pruned_path and retrained_path respectively.
'''


import torch
import toml
import logging
import sys
from pathlib import Path
import torchvision
import torchvision.transforms as transforms


sys.path.append(Path(__file__).resolve().parents[5].as_posix())
logger = logging.getLogger("chop.test")

import chop.models as models
from chop.dataset import get_dataset_info
from chop.actions.transform import pre_transform_load


# Load checkpoints
original_path = "/home/lycho-_-/文档/mase/mase_output/test-accu-0.9332.ckpt"
pruned_path = "/home/lycho-_-/文档/mase/mase_output/vgg7_cls_cifar10_2024-03-18/software/transform/prune_ckpt/state_dict.pt"
retrained_path = "/home/lycho-_-/文档/mase/mase_output/vgg7_cls_cifar10_2024-03-18/software/transform/retrain_ckpt/retrain_model.ckpt"
pruned_state_dict = torch.load(pruned_path)

# Load test config
config = toml.load("/home/lycho-_-/文档/mase/machop/configs/tests/prune/Group10/vgg7_retrain.toml")
model_name = config["model"]
dataset_name = config["dataset"]
batchsize = config['passes']['retrain']['config']['batch_size']


def prepare_model(model_name, dataset_name, load_name):
    dataset_info = get_dataset_info(dataset_name)
    model = models.get_model(model_name, "cls", dataset_info, pretrained=True)
    model = pre_transform_load(load_name=load_name, load_type='pl', model=model.to('cuda'))
    return model

def calculate_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images.to('cuda'))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to('cuda')).sum().item()
    return (100 * correct / total)


def get_test_data(dataset_name):
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
    elif dataset_name == 'imagenet':
        transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        testset = torchvision.datasets.ImageNet(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
    elif dataset_name == 'minst':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)
    else:
        raise ValueError(
            "Dataset {} is not supported for pruning yet".format(dataset_name)
            )
    return testloader

def get_parametrized_layers(module):
    parametrized_layers = []
    for name, weights in module.items():
        if 'original' in name:
            parametrized_layers.append(weights)
    return parametrized_layers

def load_pruned_weights(model, layers, index):
    for _, layer in model.named_children():
        if isinstance(layer, (torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear)):
            layer.weight.data.copy_(layers[index[0]])
            index[0] += 1
        elif isinstance(layer, torch.nn.Module):
            load_pruned_weights(layer, layers, index)
    return model

def eval_pruning_retraining(model_name, dataset_name, original_path, pruned_state_dict, retrained_path):
    accuracy_metrics = []
    num_para_metrics = []
    testloader = get_test_data(dataset_name)

    # Checkpoint model eval (test before pruning)
    original_model = prepare_model(model_name, dataset_name, original_path)
    num_before_prune = 0
    for p in original_model.parameters():
        num_before_prune += torch.count_nonzero(p).item()
    accuracy_before_pruning = calculate_accuracy(original_model.to('cuda'), testloader)
    accuracy_metrics.append(accuracy_before_pruning)
    num_para_metrics.append(num_before_prune)

    # pruned model eval (test after pruning)
    parametrized_layers = get_parametrized_layers(pruned_state_dict)
    index = [0]
    pruned_model = load_pruned_weights(original_model, parametrized_layers, index)
    accuracy_after_pruning = calculate_accuracy(pruned_model.to('cuda'), testloader)
    accuracy_metrics.append(accuracy_after_pruning)
    num_after_prune = 0
    for p in pruned_model.parameters():
        num_after_prune += torch.count_nonzero(p).item()
    num_para_metrics.append(num_after_prune)

    # retrained model eval (test after retraining)
    retrained_model = torch.load(retrained_path)
    original_model.load_state_dict(retrained_model.state_dict())
    accruacy_after_retraining = calculate_accuracy(retrained_model.to('cuda'), testloader)
    accuracy_metrics.append(accruacy_after_retraining)
    
    return accuracy_metrics, num_para_metrics
    


accuracy_metrics,num_para_metrics = eval_pruning_retraining(model_name, dataset_name, original_path, pruned_state_dict, retrained_path)

print('The accuracy before pruning: %.5f %%, accuracy after pruning: %.5f %%, accuracy after retraining: %.5f %%' % (*accuracy_metrics,))
print('The number of parameters before pruning: %d, number of parameters after pruning: %d' % (*num_para_metrics,))
print('Compression ratio of pruning is: %.2f %%' % ((num_para_metrics[0]/num_para_metrics[1])*100))
print('Accuracy rentention rate before retraining: %.2f %%' % ((accuracy_metrics[1]/accuracy_metrics[0])*100))
print('Accuracy rentention rate after retraining: %.2f %%' % ((accuracy_metrics[2]/accuracy_metrics[0])*100))
