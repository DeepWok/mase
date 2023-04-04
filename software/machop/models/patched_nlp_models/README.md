# About Patched NLP Models

Patched nlp models are modified to be compatible with Torch.fx. Here is a list of patched models supporting Machop CLI's `modify-sw`

| `--model MODEL`                      |
|-----------------------------|
| `facebook/opt-125m@patched` |
| `facebook/opt-350m@patched` |
| `facebook/opt-1.3b@patched` |
| `facebook/opt-2.7b@patched` |
| `facebook/opt-6.7b@patched` |
| `facebook/opt-13b@patched`  |
| `facebook/opt-30b@patched`  |
| `facebook/opt-66b@patched`  |

Currently Torch.fx has two limitations preventing `modify-sw` from quantizing models

- [Dynamic control flow](https://pytorch.org/docs/stable/fx.html#dynamic-control-flow)
- [Tensor constructor (not supported in PyTorch v2.0.0)](https://pytorch.org/docs/stable/fx.html#miscellanea)
