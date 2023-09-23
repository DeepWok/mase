# Guide on how to add a new model into Machop
This document includes steps to add a new model into Machop
## Overall Structure
### Model
All models that Machop support are defined inside **mase-tools/machop/chop/models**. Each model has a unique get model function, which can be called to create the model. Those get model function will be exported into a dictionary in [\_\_init\_\_](%2E%2E%5Cmachop%5Cchop%5Cmodels%5C%5F%5Finit%5F%5F.py) file.

### Command Line Interface
[Command Line Interface (cli)](..\machop\chop\cli.py) will take the input config, and perform the task defined inside the config. When training, cli will look into the dictionary contains the get funtions, use the get-function to create a model, and do training then.

## What To Do
1. Find the GitHub repositories of the original paper, find the code that defines the models, and copy it into the right folder under **mase-tools\machop\chop\models**

2. Create or modify the get model functions, and export them into the dictionary. Requirements on the get model function are shown in the next part

3. Use the command to test the defined model and text the effect



## Get model function
- **Info** should be used as one of the input variables. It is a dictionary that contains information about the dataset, e.g., number of classes; input image size.

- Other then **Info**, Inputs for different types of models are different, you can check `_setup_model_and_dataset` function defined in [cli.py](..\machop\chop\cli.py) for more detail.

- function name of get-function should be in smaller case
- keys of the dictionary should also be in smaller case

####

There's an example:
```python
def get_mobilenet_v2(info: Dict, pretrained: bool = False, **kwargs: Any):
    num_classes = info.num_classes
    model = MobileNetV2(num_classes=num_classes, **kwargs)
    # do something about the model
    # ......
    return model
```