"""
Standalone TinyLlama search script — bypasses the ch CLI.

Loads TinyLlama-1.1B directly from HuggingFace, sets up wikitext2 data module,
and calls search() with the quantization_fusion_llama.toml config.

Usage:
    python scripts/search_llama.py --save-dir outputs/search/llama_<timestamp>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from chop.models.utils import MaseModelInfo, ModelSource, ModelTaskType
from chop.dataset import MaseDataModule
from chop.actions.search import search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/search/quantization_fusion_llama.toml"),
    )
    args = parser.parse_args()

    # Model
    checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16)
    model_info = MaseModelInfo(
        name=checkpoint,
        model_source=ModelSource.HF_TRANSFORMERS,
        task_type=ModelTaskType.NLP,
        causal_LM=True,
        is_fx_traceable=False,
    )

    # Tokenizer + data module
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_module = MaseDataModule(
        name="wikitext2",
        batch_size=4,
        num_workers=4,
        max_token_len=512,
        tokenizer=tokenizer,
        model_name=checkpoint,
    )

    search(
        model=model,
        model_info=model_info,
        task="lm",
        dataset_info=data_module.dataset_info,
        data_module=data_module,
        search_config=args.config,
        save_path=args.save_dir,
        accelerator="gpu",
    )


if __name__ == "__main__":
    main()
