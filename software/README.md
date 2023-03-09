# Machop: the software stack for MASE

![alt text](machop.png)

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

TBF

### Installation

TBF

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- BASIC USAGE-->
## Basic Usage

### Example CPU Run

```bash
./chop --train \
--dataset=cifar10 \
--model=resnet18 \
--save=test
```

### Example Debug Run

```bash
./chop --train \
--dataset=cifar10 \
--model=resnet18 \
--save=test \
--debug
```

### Example GPU Run

```bash
./chop --train \
--dataset=cifar10 \
--model=resnet18 \
--save=test \
--debug \
--accelerator=gpu \
--gpu=4
```

### Example Modify Run

```bash
./chop \
--dataset=cifar10 \
--model=toy \
--save=test \
--debug \
--modify-sw=configs/test.toml
```

- All modifiable components should be defined in a `toml` file and loaded using `--modify-sw`.
- This example command shows how to apply the command to a toy network.

Mase also supports training with a modified model, for instance:

```bash
# train a normal model
./chop --train --dataset=cifar10 --model=toy --save=test
# Check the accuracy, without modification
./chop --evaluate-sw --dataset=cifar --model=toy --load checkpoints/test/best.ckpt
# Check the accuracy of modification, without re-training, this is a classic post-training quantization scenario
./chop --evaluate-sw --dataset=cifar --model=toy --load checkpoints/test/best.ckpt --modify-sw=configs/test.toml

# take the trained model, modify it and continue to train, this is known as quantization aware training
./chop --train --dataset=cifar --model=toy --save modified_test --load checkpoints/test/best.ckpt --modify-sw=configs/test.toml
# check again the re-trained accuracy
./chop --evaluate-sw --dataset=cifar --model=toy --load checkpoints/modified_test/best.ckpt --modify-sw=configs/test.toml

# enter modify again to check weights, etc.; you do not necessarily have to save the model in modify
./chop --dataset=cifar --model=toy --load checkpoints/modified_test/best.ckpt --modify-sw configs/test.toml
```

### Example Software Estimation

```bash
./chop --task cls --model roberta-base --pretrained --dataset mnli --estimate-sw --estimate-sw-config ./configs/estimate-sw/roberta_no_linear.py
```

- This example shows how to estimate the FLOPs and parameter size in model roberta-base.
- Under the hood DeepSpeed's profiler is used and a reported .txt file will be generated.
- Custom profiling behavior is defined in the .py file specified by `estimate-sw-config` flag. The config dict in .py file supports an `ignore_modules` list to ignore certain nn.Modules. See `./configs/estimate-sw

### Log Reading

```bash
tensorboard --logdir your-log-directory
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CODING STYLE -->
## Coding style

- For Python: `docs/python.md`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TESTED FLOW -->
## Tested flow

- ResNet flow

  ```bash
  # Cheng
  # [x] 1. train a fp32 resnet18 on cifar10
  ./chop --train --dataset=cifar10 --model=resnet18 --save resnet18_fp32 --batch-size 512 --cpu 32 --gpu 3

  # [x] 2. evaluate the trained fp32 model
  ./chop --evaluate-sw --dataset=cifar10 --model=resnet18 --load checkpoints/resnet18_fp32/best.ckpt --batch-size 512 --cpu 32 --gpu 3

  # [x] 3. quantise the trained fp32 model and save the quantized model
  ./chop --dataset=cifar10 --model=resnet18 --load checkpoints/resnet18_fp32/best.ckpt --modify-sw configs/tests/integer.toml --cpu 32 --gpu 3 --save resnet18_i8_ptq

  # [x] 4. evaluate post-training quantised model
  ./chop --evaluate-sw --dataset=cifar10 --model=resnet18 --modify-sw configs/tests/integer.toml --load checkpoints/resnet18_fp32/best.ckpt --batch-size 512 --cpu 32 --gpu 3

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
  ./chop --train --dataset=cifar10 --model=toy_manual --modify-sw configs/toy_manual.toml --save test
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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [X] Language Modeling Datasets (AZ, GH-36)
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
- [ ] `--estimate-sw` flag (CZ)
  - [x] FLOPs calculation (nn.Module ver.)
  - [ ] FLOPs calculation (graph ver.)
  - [ ] memory ops calculation (graph ver.)
- [ ] New quantizers
  - [ ] Quantizer testing
  - [ ] Block-based quantizers
- [X] More vision datasets and CNNs (AZ, GH-39)
  - [X] Test the existing ImageNet
  - [X] Efficientnet family
  - [X] MobileNet family
- [X] Vision transformers (AZ, GH-39)
  - [X] Pyramid Vision Transformer (v1 and v2)
  - [X] DeiT
  - [X] Swin
- [ ] More on README
  - [ ] Prerequisites
  - [ ] Installation

See the [open issues](https://github.com/JianyiCheng/mase-tools/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Models and Datasets

### Datasets

|     Category    |        Task       |     Name    | Sub-tasks |
|:---------------:|:-----------------:|:-----------:|:---------:|
| Vision Datasets |   Classification  |   CIFAR10   |           |
| Vision Datasets |   Classification  |   CIFAR100  |           |
| Vision Datasets |   Classification  |   ImageNet  |           |
|   NLP Datasets  |  Text Entailment  |     MNLI    |           |
|   NLP Datasets  | Language Modeling |  Wikitext2  |           |
|   NLP Datasets  | Language Modeling | Wikitext103 |           |
|   NLP Datasets  |    Translation    |  iwslt2017  |   en-de   |
|   NLP Datasets  |    Translation    |  iwslt2017  |   de-en   |
|   NLP Datasets  |    Translation    |  iwslt2017  |   en-fr   |
|   NLP Datasets  |    Translation    |  iwslt2017  |   en-ch   |
|   NLP Datasets  |    Translation    |  wmt19      |   de-en   |
|   NLP Datasets  |    Translation    |  wmt19      |   zh-en   |

- Vision Datasets
  - CIFAR10
  - CIFAR100
  - ImageNet

- NLP Datasets
  - MNLI
  - Wikitext2
  - Wikitext103
  - iwslt2017
    - en-de
    - de-en
    - en-fr
    - en-ch
  - wmt19_de_en
  - wmt19_zh_en

### Models

|     Category    |        Task       |     Name        | Sub-style         |
|:---------------:|:-----------------:|:---------------:|:-----------------:|
|   CNN           |   Classification  |   ResNet18      |                   |
|   CNN           |   Classification  |   ResNet50      |                   |
|   CNN           |   Classification  |   MobileNet     | mobilenetv3_small |
|   CNN           |   Classification  |   MobileNet     | mobilenetv3_large |
|   CNN           |   Classification  |   EfficientNet  | efficientnet_v2_s |
|   CNN           |   Classification  |   EfficientNet  | efficientnet_v2_m |
|   CNN           |   Classification  |   EfficientNet  | efficientnet_v2_l |
|   ViT           |   Classification  |   PVT-V1        | pvt_tiny          |
|   ViT           |   Classification  |   PVT-V1        | pvt_small         |
|   ViT           |   Classification  |   PVT-V1        | pvt_medium        |
|   ViT           |   Classification  |   PVT-V1        | pvt_large         |
|   ViT           |   Classification  |   PVT-V2        | pvt_v2_b0         |
|   ViT           |   Classification  |   PVT-V2        | pvt_v2_b1         |
|   ViT           |   Classification  |   PVT-V2        | pvt_v2_b2         |
|   ViT           |   Classification  |   PVT-V2        | pvt_v2_b3         |
|   ViT           |   Classification  |   PVT-V2        | pvt_v2_b4         |
|   ViT           |   Classification  |   PVT-V2        | pvt_v2_b5         |
|   ViT           |   Classification  |   DeiT          | deit_tiny_224     |
|   ViT           |   Classification  |   DeiT          | deit_small_224    |
|   ViT           |   Classification  |   DeiT          | deit_base_224     |
|   ViT           |   Classification  |   CSwin         | cswin_64_tiny     |
|   ViT           |   Classification  |   CSwin         | cswin_64_small    |
|   ViT           |   Classification  |   CSwin         | cswin_96_base     |
|   ViT           |   Classification  |   CSwin         | cswin_144_large   |
|   Transformers  |   LM              |   BERT          | bert-base-uncased |
|   Transformers  |   LM              |   GPT2          | gpt2              |
|   Transformers  |   Classification  |   RoBERTa       | roberta_base      |
|   Transformers  |   Classification  |   RoBERTa       | roberta_large     |
|   Transformers  |   LM              |   OPT           | facebook/opt-125m |
|   Transformers  |   LM              |   OPT           | facebook/opt-350m |
|   Transformers  |   LM              |   OPT           | facebook/opt-1.3b |
|   Transformers  |   LM              |   OPT           | facebook/opt-2.7b |
|   Transformers  |   LM              |   OPT           | facebook/opt-13b  |
|   Transformers  |   LM              |   OPT           | facebook/opt-30b  |
|   Transformers  |   LM              |   OPT           | facebook/opt-66b  |
|   Transformers  |   LM              |   GPT-NEO       | EleutherAI/gpt-neo-125M  |
|   Transformers  |   LM              |   GPT-NEO       | EleutherAI/gpt-neo-1.3B  |
|   Transformers  |   LM              |   GPT-NEO       | EleutherAI/gpt-neo-2.7B  |
|   Transformers  |   LM              |   GPT-NEO       | EleutherAI/gpt-neox-20b  |
|   Transformers  |   Translation     |   T5            | t5-small          |
|   Transformers  |   Translation     |   T5            | t5-base           |
|   Transformers  |   Translation     |   T5            | t5-large          |
|   Transformers  |   Translation     |   T5            | google/t5-v1_1-small  |

- Vision Models
  - ResNet18
  - ResNet50
  - MobileNets
    - mobilenetv3_small
    - mobilenetv3_large
  - EfficientNets
    - efficientnet_v2_s
    - efficientnet_v2_m
    - efficientnet_v2_l
  - Pyramid Vision Transformers V1
    - pvt_tiny
    - pvt_small
    - pvt_medium
    - pvt_large
  - Pyramid Vision Transformers V2
    - pvt_v2_b0
    - pvt_v2_b1
    - pvt_v2_b2
    - pvt_v2_b3
    - pvt_v2_b4
    - pvt_v2_b5
  - Data Efficient Image Transformers (DeiT)
    - deit_tiny_224
    - deit_small_224
    - deit_base_224
  - CSWin Transformer
    - cswin_64_tiny
    - cswin_64_small
    - cswin_96_base
    - cswin_144_large

- NLP Models
  - BERT
  - GPT2
  - RoBERTa-base
  - RoBERTa-large
  - OPT
    - facebook/opt-125m
    - facebook/opt-350m
    - facebook/opt-1.3b
    - facebook/opt-2.7b
    - facebook/opt-13b
    - facebook/opt-30b
    - facebook/opt-66b
  - gpt-neo
    - EleutherAI/gpt-neo-125M
    - EleutherAI/gpt-neo-1.3B
    - EleutherAI/gpt-neo-2.7B
    - EleutherAI/gpt-neox-20b
  - t5-small
  - t5-base
  - t5-large
  - google/t5-v1_1-small

<p align="right">(<a href="#readme-top">back to top</a>)</p>
