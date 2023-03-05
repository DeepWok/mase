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
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [X] Language Modeling Datasets (AZ)
  - [X] Wikitext2
  - [X] Wikitext103
- [X] Language Modeling Models (AZ)
  - [X] BERT
  - [X] ROBERT
  - [X] GPT-NEO
  - [X] GPT2
- [ ] Machine Translation Models (AZ)
  - [ ] T5
  - [ ] Test T5 on existing translation datasets
- [ ] `--estimate` flag (CZ)
  - [ ] FLOPs calculation
  - [ ] memory ops calculation
- [ ] New quantizers
  - [ ] Quantizer testing
  - [ ] Block-based quantizers
- [ ] More on README
  - [ ] Prerequisites
  - [ ] Installation

See the [open issues](https://github.com/JianyiCheng/mase-tools/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Models and Datasets

- Vision Datasets
  - CIFAR10
  - CIFAR100
- NLP Datasets
  - MNLI
  - Wikitext2
  - Wikitext103
- Vision Models
  - ResNet18
  - ResNet50
- NLP Models
  - BERT
  - GPT2
  - RoBERTa-base
  - RoBERTa-large
  - OPT-125m to OPT-66b (7 models)
  - gpt-neo-125M to gpt-neo-20B (4 models)

<p align="right">(<a href="#readme-top">back to top</a>)</p>