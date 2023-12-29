# Machop: the software stack for MASE

![alt text](../docs/imgs/machop.png)

[Machop](https://bulbapedia.bulbagarden.net/wiki/Machop_(Pok%C3%A9mon)) is a humanoid, bipedal Pokémon that has blue-gray skin. It has three brown ridges on top of its head, large red eyes, and a relatively flat face. On either side of its chest are three, thin, rib-like stripes. Its feet appear to have no toes, while its hands have five fingers. Machop also has a short, stubby tail.

Why called Machop? Because Machop is the most common pokemon you can find in the [Final Maze](https://bulbapedia.bulbagarden.net/wiki/Final_Maze)!

For more, you can watch this [video](https://www.youtube.com/watch?v=JEUsN_KlDy8&ab_channel=Mah-Dry-Bread-Gameplay%26Streams%21).

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#basic-usage">Usage</a>
      <ul>
        <li><a href="#example-cpu-run">Example CPU Run</a></li>
        <li><a href="#example-debug-run">Example Debug Run</a></li>
        <li><a href="#example-gpu-run">Example GPU Run</a></li>
        <li><a href="#example-modify-run">Example Modify Run</a></li>
        <li><a href="#log-reading">Log Reading</a></li>
        <li><a href="#checkpointing">Checkpointing</a></li>
      </ul>
    </li>
    <li><a href="#coding-style">Coding Style</a></li>
    <li><a href="#tested-flow">Tested flow</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#models-and-datasets">Models and Datasets</a></li>
  </ol>
</details>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

The full list of dependencies is contained in the [Conda environment](./environment.yml) or the [Dockerfile](../Docker/Dockerfile). See instructions on the main [README](../README.md) for installing the dependencies.

<!-- TO DO: add to PyPI -->
<!-- ### Installation

TBF -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Basic Usage

Machop has four actions: train, test, transform and search. Parameters for each action can be defined through command-line arguments or within a `toml` configuration file.

|   Action  |                                                    Usage                                                    |
|:---------:|:-----------------------------------------------------------------------------------------------------------:|
|   Train   |                         Train a given model on the train split of the given dataset.                        |
|    Test   |         Evaluate a given model on the test split of the given dataset, without modifying the model.         |
| Transform | Run a transformation pass on the specified model, e.g. quantization or pruning.                             |
|   Search  | Apply a given search strategy over a search space to optimize a specified set of hardware/software metrics. |

The following command trains the ResNet-18 model with CIFAR10 dataset, and evaluates its accuracy on the test set.

```bash
# Train the toy model
./ch train resnet18 cifar10 --project resnet18

# Evaluate the accuracy, without modification
export CKPT=$(pwd)/mase_output/resnet18/software/training_ckpts/best.ckpt
./ch test resnet18 cifar10 --load $CKPT
```

All arguments can be defined in a `toml` file and loaded using `--config`. See [`machop/configs`](https://github.com/DeepWok/mase/tree/main/machop/configs) for example configuration files.
```bash
export CONFIGS=$(pwd)/configs/by_model/resnet18/

./ch train --config $CONFIG/train_resnet18.toml
./ch test --config $CONFIG/test_resnet18.toml
```

A typical optimization scenario involves quantizing the model to integer precision, then optionally performing fine-tuning iterations. This is achieved by the following commands, which uses the trained checkpoint as a starting point.

```bash
# Run the quantization pass on the trained model
./ch transform resnet18 cifar10 --load $CKPT --config $CONFIGS/quantize_resnet18.toml

# Perform further training iterations on the transformed model and evaluate the quantized model accuracy
./ch train resnet18 cifar10 --config $CONFIGS/train_resnet18.toml
./ch test resnet18 cifar10 --config $CONFIGS/test_resnet18.toml
```

If the accuracy of the quantized model is undesirable, run the search action to obtain the optimal quantization scheme for the defined model.

```bash
./ch search --config $CONFIGS/search_quantize_resnet18.toml
```

Once you're happy with the quantized model performance, you can run the emit verilog transform pass to generate the SystemVerilog files for deployment on FPGA.

```bash
./ch transform --config $CONFIGS/emit_resnet18.toml
```

<!-- TO DO: need to implement analysis passes into transform action, or new analyze action -->
<!-- ### Example Software Estimation

```bash
./chop --task cls --model roberta-base --pretrained --dataset mnli --estimate-sw --estimate-sw-config ./configs/estimate-sw/roberta_no_linear.py
```

- This example shows how to estimate the FLOPs and parameter size in model roberta-base.
- Under the hood DeepSpeed's profiler is used and a reported .txt file will be generated.
- Custom profiling behavior is defined in the .py file specified by `estimate-sw-config` flag. The config dict in .py file supports an `ignore_modules` list to ignore certain nn.Modules. See `./configs/estimate-sw -->

### Log Reading

```bash
tensorboard --logdir your-log-directory
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Checkpointing

Machop supports PyTorch state dict checkpoint (`.ckpt`) and PyTorch Lightning checkpoint (`.ckpt`). The modified model (generated by transform action) is always saved and loaded in pickle format (`.pkl`).

#### Load checkpoint

Machop uses the action name (`train`/`test`/`transform`/`search`)*, `--load LOAD_NAME`, and `--load-type LOAD_TYPE` to infer when and how to load checkpoint. A proper `--load-type` is required if `--load LOAD_NAME` is specified.

`--load-type` should be one of `hf`, `pt`, `pl`, `pkl`:

- `hf`: HuggingFace checkpoint should be a directory
- `pt`: PyTorch checkpoint can be .ckpt file of (a state_dict) or (a dict containing state_dict)
- `pl`: PyTorchLightning checkpoint should be a .ckpt file saved by Trainer checkpoint callback. The .ckpt file is a dict containing state_dict and other Trainer states
- `pkl`: Machop's modify-sw checkpoint containing a modified model.

The figure below outlines how Machop determines when and where to load.

- If `--load` is not specified, `--load-type` is a don't-care.
- If `--load-type=hf` is used,
  - `--pretrained --load-type=hf` downloads and loads the HuggingFace pretrained model specified by `--model MODEL`.
  - `--pretrained --load=<path/to/checkpoint> --load-type=hf` loads a HuggingFace checkpoint saved by `PreTrainedModel.save_pretrained`.
  - For patched models provided by Machop, `--pretrained` will loads the pretrained original model. For example `ch <ACTION> facebook/opt-125m@patched <DATASET> --pretrained --load-type=hf` will load Facebook's `opt-125m` checkpoint into the `facebook/opt-125m@patched` model.
- The `transform` action can load `pt` or `pl` checkpoints.
- The `train` and `test` actions can load `pt`, `pl`, or `pkl` checkpoints.

<!-- ![MASE_load](../docs/imgs/machop_load.jpg) -->

#### Save checkpoints

- Transformed models will be saved to `pkl` files (pickle checkpoints).
- Trained models will be saved to `pl` files (PyTorch Lightning `.ckpt` checkpoints).

<p align="right">(<a href="#checkpointing">back to top</a>)</p>

<!-- CODING STYLE -->
## Coding style

- For Python: see [standard](https://github.com/DeepWok/mase/blob/main/docs/Python-coding-style-specifications.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TESTED FLOW -->
<!-- ## Tested flow

- ResNet flow

  ```bash
  # Cheng
  # [x] 1. train a fp32 resnet18 on cifar10
  ./chop --train --dataset=cifar10 --model=resnet18 --save resnet18_fp32 --batch-size 512 --cpu 32 --gpu 3

  # [x] 2. evaluate the trained fp32 model
  ./chop --validate-sw --dataset=cifar10 --model=resnet18 --load checkpoints/resnet18_fp32/best.ckpt --batch-size 512 --cpu 32 --gpu 3

  # [x] 3. quantise the trained fp32 model and save the quantized model
  ./chop --dataset=cifar10 --model=resnet18 --load checkpoints/resnet18_fp32/best.ckpt --modify-sw configs/tests/integer.toml --cpu 32 --gpu 3 --save resnet18_i8_ptq

  # [x] 4. evaluate post-training quantised model
  ./chop --validate-sw --dataset=cifar10 --model=resnet18 --modify-sw configs/tests/integer.toml --load checkpoints/resnet18_fp32/best.ckpt --batch-size 512 --cpu 32 --gpu 3

  # [x] 5. load trained fp32 model, do quantisation-aware training, save the trained quantized model
  ./chop --train --dataset=cifar10 --model=resnet18 --modify-sw configs/tests/integer.toml --load checkpoints/resnet18_fp32/best.ckpt --save resnet18_i8_qat --batch-size 512 --cpu 32 --gpu 3
  ```

- Train from modified toynet with fixed-point quantization

  ```bash
  ./chop --train --dataset=cifar10 --model=toy --modify-sw configs/tests/integer.toml --save test
  # chop --train --dataset=cifar10 --model=toy --modify-sw configs/tests/integer.toml --save test --training-optimizer sgd --seed 666 --learning_rate 1e-4 --max-epochs 2 --batch-size 128
  ```

- Train from custom toynet that has mixed-precision fixed-point quantization

  ```bash
  ./chop --train --dataset=cifar10 --model=toy_manual --config configs/tests/toy_manual.toml --debug
  ```

- Tune a pre-trained `opt` model on `wikitext2` on GPU

  ```bash
  vim machop/dataset/nlp/language_modeling.py
  ```

  The original setup `1024` block size (or context width), is really hard to run because of GPU memory limitation, so now this is `256`.

  ```bash
  ./chop --train --dataset=wikitext2 --model=facebook/opt-350m --pretrained --save test --accelerator gpu --gpu 1 --batch-size 4 --task lm
  ```

- Tune a pre-trained `t5` on `iwslt` on GPU

  ```bash
  ./chop --train --dataset=iwslt2017_en_de --model=t5-small --pretrained --save test --accelerator gpu --gpu 1 --batch-size 4 --task tran
  ```

- Train a `resnet` and a `pvt` on `imagenet`

  ```bash
  ./chop --train --dataset=imagenet --model=resnet18-imagenet --save test --accelerator gpu --gpu 1 --batch-size 32
  ./chop --train --dataset=imagenet --model=pvt_v2_b0 --save test --accelerator gpu --gpu 1 --batch-size 32
  ```

- Train vision transformers on GPUs

  ```bash
  ./chop --train --dataset=imagenet --model=cswin_64_tiny --save test --accelerator gpu --gpu 1 --batch-size 32

  ./chop --train --dataset=imagenet --model=deit_tiny_224 --save test --accelerator gpu --gpu 1 --batch-size 32
  ```

- Train mobilenet and efficientnet

  ```bash
  ./chop --train --dataset=imagenet --model=mobilenetv3_small --save test --accelerator gpu --gpu 1 --batch-size 32

  ./chop --train --dataset=imagenet --model=efficientnet_v2_s --save test --accelerator gpu --gpu 1 --batch-size 32
  ```

- Estimate-sw on built-in models

  ```bash
  ./chop --task cls --model resnet18 --dataset cifar10 --estimate-sw
  ./chop --task cls --model resnet18 --dataset cifar10 --estimate-sw --estimate-sw-config ./configs/estimate-sw/all_included.py
  ./chop --task cls --model resnet18 --dataset cifar10 --estimate-sw --estimate-sw-config ./configs/estimate-sw/resnet_no_conv2d.py
  ./chop --task cls --model roberta-base --pretrained --dataset mnli --estimate-sw --estimate-sw-config ./configs/estimate-sw/all_included.py
  ./chop --task cls --model roberta-base --pretrained --dataset mnli --estimate-sw --estimate-sw-config ./configs/estimate-sw/roberta_no_linear.py
  ./chop --task tran --model t5-small --pretrained --dataset iwslt2017_en_de --estimate-sw
  ```

- Fine-grained estimate

  ```bash
  ./chop --task cls --model resnet18 --dataset cifar10 --estimate-sw --estimate-sw-config ./configs/estimate-sw/all_included_fine_grained.py
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->

<!-- ROADMAP -->
## Roadmap

<!-- - [X] Language Modeling Datasets (AZ, GH-36)
  - [X] Wikitext2
  - [X] Wikitext103
- [X] Language Modeling Models (AZ, GH-36)
  - [X] BERT
  - [X] ROBERT
  - [X] GPT-NEO
  - [X] GPT2
- [X] Machine Translation Models (AZ, GH-38)
  - [X] IWSLT and WMT datasets
  - [X] T5
  - [X] Test T5 on existing translation datasets (partial)
- [ ] `--estimate-sw` flag (AZ & CZ)
  - [x] FLOPs calculation (nn.Module ver.)
  - [x] FLOPs calculation (graph ver.)
  - [x] memory ops calculation (graph ver.)
  - [ ] tensor shape recording
- [x] New quantizers
  - [x] Quantizer testing
  - [x] Block-based quantizers
- [X] More vision datasets and CNNs (AZ, GH-39)
  - [X] Test the existing ImageNet
  - [X] Efficientnet family
  - [X] MobileNet family
- [X] Vision transformers (AZ, GH-39)
  - [X] Pyramid Vision Transformer (v1 and v2)
  - [X] DeiT
  - [X] Swin
- [ ] Enhance `modify`
  - [ ] Add support for BatchNorm
  - [ ] Add support for Conv1d
  - [ ] Add support for ReLU6
  - [ ] Add support for the multiply functions
- [ ] More on README
  - [ ] Prerequisites
  - [ ] Installation -->

See the [open issues](https://github.com/JianyiCheng/mase-tools/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Models and Datasets

### Models

The following model categories are supported. See [here](../docs/roadmap/supported-models.md) for a comprehensive list.

|     Category    |        Task       |     Name        |
|:---------------:|:-----------------:|:---------------:|
|   CNN           |   Classification  |   ResNet18      |
|   CNN           |   Classification  |   ResNet50      |
|   CNN           |   Classification  |   MobileNet     |
|   CNN           |   Classification  |   EfficientNet  |
|   ViT           |   Classification  |   PVT-V1        |
|   ViT           |   Classification  |   PVT-V2        |
|   ViT           |   Classification  |   DeiT          |
|   ViT           |   Classification  |   CSwin         |
|   Transformers  |   LM              |   BERT          |
|   Transformers  |   LM              |   GPT2          |
|   Transformers  |   Classification  |   RoBERTa       |
|   Transformers  |   LM              |   OPT           |
|   Transformers  |   LM              |   GPT-NEO       |
|   Transformers  |   Translation     |   T5            |

### Datasets

The following dataset categories are supported. See [here](../docs/roadmap/supported-datasets.md) for a comprehensive list.

|     Category    |        Task       |     Name    |
|:---------------:|:-----------------:|:-----------:|
| Vision Datasets |   Classification  |   CIFAR10   |
| Vision Datasets |   Classification  |   CIFAR100  |
| Vision Datasets |   Classification  |   ImageNet  |
|   NLP Datasets  |  Text Entailment  |     MNLI    |
|   NLP Datasets  | Language Modeling |  Wikitext2  |
|   NLP Datasets  | Language Modeling | Wikitext103 |
|   NLP Datasets  |    Translation    |  iwslt2017  |
|   NLP Datasets  |    Translation    |  wmt19      |



<p align="right">(<a href="#readme-top">back to top</a>)</p>
