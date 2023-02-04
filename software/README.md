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

### Training log check

```bash
tensorboard --logdir your-log-directory
```
