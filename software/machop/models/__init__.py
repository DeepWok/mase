from .resnet import ResNet18, ResNet50, ResNet18ImageNet, ResNet50ImageNet

model_map = {
    'resnet18': ResNet18,
    'resnet50': ResNet50,
    'resnet18_imagenet': ResNet18ImageNet,
    'resnet50_imagenet': ResNet50ImageNet,
}