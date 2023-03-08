import torchvision


def efficientnet_v2_s(**kwargs):
    num_classes = kwargs.pop("info")["num_classes"]
    kwargs = {"num_classes": num_classes}

    model = torchvision.models.efficientnet_v2_s(**kwargs)
    return model


def efficientnet_v2_m(**kwargs):
    num_classes = kwargs.pop("info")["num_classes"]
    kwargs = {"num_classes": num_classes}

    model = torchvision.models.efficientnet_v2_m(**kwargs)
    return model


def efficientnet_v2_l(**kwargs):
    num_classes = kwargs.pop("info")["num_classes"]
    kwargs = {"num_classes": num_classes}

    model = torchvision.models.efficientnet_v2_l(**kwargs)
    return model
