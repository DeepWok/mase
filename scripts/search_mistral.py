"""
Standalone Mistral-7B search script — bypasses the ch CLI.

Loads Mistral-7B-v0.1 directly from HuggingFace, sets up wikitext2 data module,
and calls search() with the quantization_fusion_mistral.toml config.

NOTE: Requires HuggingFace login for gated model access:
    huggingface-cli login

Usage:
    python scripts/search_mistral.py --save-dir outputs/search/mistral_<timestamp>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
        default=Path("configs/search/quantization_fusion_mistral.toml"),
    )
    args = parser.parse_args()

    # Model
    checkpoint = "mistralai/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
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
        batch_size=1,
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
