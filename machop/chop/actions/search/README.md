# Overview

Search should have two components, namely the `search_space` and the `strategy`.

Currently, only two strategies (`optuna` and `rl`) are tested on `opt-350m`:

```bash
./ch search --task lm --cpu 4 --config configs/examples/opt_rl_mixed_precision.toml
./ch search --task lm --cpu 4 --config configs/examples/opt_optuna_mixed_precision.toml
```