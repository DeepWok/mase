# Machop: the software stack for MASE

![alt text](machop.png)

[Machop](https://bulbapedia.bulbagarden.net/wiki/Machop_(Pok%C3%A9mon)) is a humanoid, bipedal Pokémon that has blue-gray skin. It has three brown ridges on top of its head, large red eyes, and a relatively flat face. On either side of its chest are three, thin, rib-like stripes. Its feet appear to have no toes, while its hands have five fingers. Machop also has a short, stubby tail.

Why called Machop? Because Machop is the most common pokemon you can find in the [Final Maze](https://bulbapedia.bulbagarden.net/wiki/Final_Maze)!

For more, you can watch this [video](https://www.youtube.com/watch?v=JEUsN_KlDy8&ab_channel=Mah-Dry-Bread-Gameplay%26Streams%21).


## Commands

### Example CPU run

```bash
./chop train cifar10 resnet18
 # cli argument for the saving directory, you must add this
 \ --save test
```

### Example debug run

```bash
./chop train cifar10 resnet18 --save test --debug
```

### Example GPU run

```bash
./chop train cifar10 resnet18 
  \ --save test 
  # use GPUs and 4 of them
  \ -a gpu -n 4
  # set learning rate
  \ -lr 1e-5
```

### Example modify run

```bash
./chop modify cifar10 toy --save test --debug --config configs/test.toml
```

- All modifiable components should be defined in a `toml` file and loaded using `--config`.
- This example command shows how to apply the command to a toy network.

Mase also supports training with a modified model, for instance:

```bash
# train a normal model
./chop train cifar10 toy --save test
# Check the accuracy, without modification
./chop eval cifar10 toy --load checkpoints/test/best.ckpt
# Check the accuracy of modification, without re-training, this is a classic post-training quantization scenario
./chop eval cifar10 toy --load checkpoints/test/best.ckpt --modify --config configs/test.toml

# take the trained model, modify it and continue to train, this is known as quantization aware training
./chop train cifar10 toy --save modified_test --load checkpoints/test/best.ckpt --modify --config configs/test.toml
# check again the re-trained accuracy
./chop eval cifar10 toy --load checkpoints/modified_test/best.ckpt --modify --config configs/test.toml

# enter modify again to check weights, etc.; you do not necessarily have to save the model in modify
./chop modify cifar10 toy --load checkpoints/modified_test/best.ckpt --config configs/test.toml
```

### Training log check

```bash
tensorboard --logdir your-log-directory
```

## Quick coding style guide

- For Python: `docs/python.md`

## Tested commands and functionalities

- Train from modified toynet with fixed-point quantization
  ```bash
  ./chop train cifar10 toy --config configs/tests/integer.toml --save test --modify
  ```

- Train from custom toynet that has mixed-precision fixed-point quantization
  ```bash
  ./chop train cifar10 toy_manual --config configs/toy_manual.toml --save test
  ```
