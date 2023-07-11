# FSDP

## HuggingFace Accelerate

[HuggingFace accelerate](https://huggingface.co/docs/accelerate/index) is a library that makes distributed training simple. It provides a simple API for launching training and evaluation jobs across multiple processes and machines.

Different from HuggingFace [transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) or Lightning's [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html), accelerate requires users to write their own training and evaluation loops.

## Fully Sharded Data Parallelism

See PyTorch's [paper](https://arxiv.org/abs/2304.11277) for more details about fully sharding.

![fully_shard](https://pytorch.org/assets/images/fsdp_workflow.png)

## Accelerate's Fully Shard Plugin

Accelerate's fully shard plugin is a wrapper around PyTorch's [FullyShardedDataParallel](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) module, providing a simple API for HuggingFace transformer models.

Here is an example of training a model with accelerate's fully shard plugin.

```bash
cd mase-tools/machop/examples/by_feature/fully_shard
# launch training script using accelerate
accelerate launch --use_fsdp accelerate_fsdp.py
```

Note that if running the script without accelerate, i.e.,

```bash
python accelerate_fsdp.py
```

the script will only run on a single GPU.
