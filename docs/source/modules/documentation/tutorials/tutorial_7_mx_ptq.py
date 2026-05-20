# Tutorial 7: MX Post-Training Quantization with Mase
#
# Runnable companion to tutorial_7_mx_ptq.ipynb — same helpers, same TOML
# configs, same evaluation. We quantize Llama-3.2-1B with three progressive
# configs and measure WikiText perplexity at each step:
#
#   Config A — MXInt4 round-to-nearest baseline (W4 A4 KV4)
#   Config B — + [gptq] Hessian-aware weight calibration
#   Config C — + [rotation_search] greedy per-matmul online Hadamard rotation
#
# All perplexity evals use lm-eval-harness's wikitext task.

import tomllib
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chop.passes.module.transforms import quantize_module_transform_pass

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table


MODEL_NAME = "unsloth/Llama-3.2-1B"
OUTPUT_DIR = Path("tutorial_7_output")
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_quant_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    by = raw.pop("by", "regex_name")
    pass_args = {"by": by}
    gptq = raw.pop("gptq", None)
    if gptq is not None:
        pass_args["gptq"] = deepcopy(gptq)
    for key, value in raw.items():
        pass_args[key] = {"config": deepcopy(value)}
    return pass_args


def evaluate_perplexity(model, tokenizer, max_length: int = 2048, batch_size: int = 8) -> float:
    """Run lm-eval-harness wikitext task on `model`; return word_perplexity."""
    lm = HFLM(pretrained=model, tokenizer=tokenizer,
              max_length=max_length, batch_size=batch_size)
    results = simple_evaluate(model=lm, tasks=["wikitext"],
                              batch_size=batch_size, limit=None)
    print(make_table(results))
    metrics = results["results"]["wikitext"]
    for k, v in metrics.items():
        if k.startswith("word_perplexity"):
            return float(v)
    raise KeyError(f"word_perplexity not found: {list(metrics)}")


def fresh_model(attn_implementation: str = "eager"):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, attn_implementation=attn_implementation
    )
    model.eval()
    return model


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


# ── fp16 baseline ─────────────────────────────────────────────────────────────

model_fp16 = fresh_model(attn_implementation="sdpa").to(DEVICE)
ppl_fp16 = evaluate_perplexity(model_fp16, tokenizer)
print(f"fp16 ppl = {ppl_fp16:.4f}")
del model_fp16
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ── Config A — MXInt4 RTN (W4 A4 KV4) ─────────────────────────────────────────

config_a_toml = OUTPUT_DIR / "config_a_rtn.toml"
config_a_toml.write_text(r"""by = "regex_name"

# Attention block — swap LlamaAttention for the MXInt variant, quantize KV cache
['model\.layers\.\d+\.self_attn$']
name = "mxint"

['model\.layers\.\d+\.self_attn$'.qk_matmul]
bypass = true

['model\.layers\.\d+\.self_attn$'.av_matmul]
bypass = true

['model\.layers\.\d+\.self_attn$'.kv_cache]
data_in_block_size = 32
data_in_width      = 4

['model\.layers\.\d+\.self_attn$'.softmax]
bypass = true

['model\.layers\.\d+\.self_attn$'.rope]
bypass = true

# Linear projections: W4 + A4
['model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj']
name = "mxint"
weight_block_size = 32
weight_width = 4
data_in_block_size = 32
data_in_width = 4

['model\.layers\.\d+\.mlp\.(gate|up|down)_proj']
name = "mxint"
weight_block_size = 32
weight_width = 4
data_in_block_size = 32
data_in_width = 4
""")
print(f"wrote {config_a_toml}")

model_a = fresh_model(attn_implementation="eager")
pass_args_a = load_quant_config(config_a_toml)
model_a, _ = quantize_module_transform_pass(model_a, pass_args_a)
model_a.to(DEVICE)

from collections import Counter

cls_counts = Counter(type(m).__name__ for _, m in model_a.named_modules())
for name, n in cls_counts.most_common(8):
    print(f"  {name:30s}  x{n}")

ppl_a = evaluate_perplexity(model_a, tokenizer)
print(f"Config A ppl = {ppl_a:.4f}")
del model_a
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ── Config B — add [gptq] ─────────────────────────────────────────────────────

config_b_toml = OUTPUT_DIR / "config_b_gptq.toml"
config_b_toml.write_text(rf"""by = "regex_name"

[gptq]
model_name       = "{MODEL_NAME}"
format           = "mxint"
dataset          = "wikitext2"
nsamples         = 32
seqlen           = 512
cali_batch_size  = 8
quantile_search  = true
clip_search_y    = true
checkpoint_dir   = "{OUTPUT_DIR}/checkpoints/config_b_gptq"

    [gptq.weight_config]
    weight_block_size = 32
    weight_width      = 4

# Same attention + linear selectors as Config A (KV cache stays quantized).

['model\.layers\.\d+\.self_attn$']
name = "mxint"

['model\.layers\.\d+\.self_attn$'.qk_matmul]
bypass = true

['model\.layers\.\d+\.self_attn$'.av_matmul]
bypass = true

['model\.layers\.\d+\.self_attn$'.kv_cache]
data_in_block_size = 32
data_in_width      = 4

['model\.layers\.\d+\.self_attn$'.softmax]
bypass = true

['model\.layers\.\d+\.self_attn$'.rope]
bypass = true

['model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj']
name = "mxint"
weight_block_size = 32
weight_width = 4
data_in_block_size = 32
data_in_width = 4
gptq = true

['model\.layers\.\d+\.mlp\.(gate|up|down)_proj']
name = "mxint"
weight_block_size = 32
weight_width = 4
data_in_block_size = 32
data_in_width = 4
gptq = true
""")
print(f"wrote {config_b_toml}")

model_b = fresh_model(attn_implementation="eager")
pass_args_b = load_quant_config(config_b_toml)
model_b, _ = quantize_module_transform_pass(model_b, pass_args_b)
model_b.to(DEVICE)
ppl_b = evaluate_perplexity(model_b, tokenizer)
print(f"Config B ppl = {ppl_b:.4f}")
del model_b
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ── Config C — add [rotation_search] ──────────────────────────────────────────

config_c_toml = OUTPUT_DIR / "config_c_rotsearch.toml"
config_c_toml.write_text(rf"""by = "regex_name"

[gptq]
model_name       = "{MODEL_NAME}"
format           = "mxint"
dataset          = "wikitext2"
nsamples         = 32
seqlen           = 512
cali_batch_size  = 8
quantile_search  = true
clip_search_y    = true
checkpoint_dir   = "{OUTPUT_DIR}/checkpoints/config_b_gptq"

    [gptq.weight_config]
    weight_block_size = 32
    weight_width      = 4

[rotation_search]
calib_nsamples  = 32
calib_seqlen    = 512
improvement_eps = 0.0
cache_path      = "{OUTPUT_DIR}/checkpoints/config_c_rotsearch/rotation_decisions.json"

# Same attention + linear selectors as Config B.

['model\.layers\.\d+\.self_attn$']
name = "mxint"

['model\.layers\.\d+\.self_attn$'.qk_matmul]
bypass = true

['model\.layers\.\d+\.self_attn$'.av_matmul]
bypass = true

['model\.layers\.\d+\.self_attn$'.kv_cache]
data_in_block_size = 32
data_in_width      = 4

['model\.layers\.\d+\.self_attn$'.softmax]
bypass = true

['model\.layers\.\d+\.self_attn$'.rope]
bypass = true

['model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj']
name = "mxint"
weight_block_size = 32
weight_width = 4
data_in_block_size = 32
data_in_width = 4
gptq = true

['model\.layers\.\d+\.mlp\.(gate|up|down)_proj']
name = "mxint"
weight_block_size = 32
weight_width = 4
data_in_block_size = 32
data_in_width = 4
gptq = true
""")

model_c = fresh_model(attn_implementation="eager").to(DEVICE)
pass_args_c = load_quant_config(config_c_toml)
model_c, _ = quantize_module_transform_pass(model_c, pass_args_c)
ppl_c = evaluate_perplexity(model_c, tokenizer)
print(f"Config C ppl = {ppl_c:.4f}")


# ── Recap ─────────────────────────────────────────────────────────────────────

print("\nRecap (W4 A4 KV4 throughout — lm-eval word_perplexity)")
print(f"  fp16              {ppl_fp16:.4f}")
print(f"  A — RTN           {ppl_a:.4f}")
print(f"  B — +GPTQ         {ppl_b:.4f}")
print(f"  C — +rotsearch    {ppl_c:.4f}")
