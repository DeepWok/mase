"""
Standalone BERT search script — bypasses the ch CLI.

Loads bert-base-uncased directly from HuggingFace, sets up SST-2 data module,
and calls search() with the quantization_fusion_bert.toml config.

Usage:
    python scripts/search_bert.py --save-dir outputs/search/bert_<timestamp>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer
from chop.models.bert.modeling_bert import BertForSequenceClassification
from chop.models import get_model_info
from chop.dataset import MaseDataModule
from chop.actions.search import search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/search/quantization_fusion_bert.toml"),
    )
    args = parser.parse_args()

    # Model
    checkpoint = "textattack/bert-base-uncased-SST-2"
    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model_info = get_model_info("bert-base-uncased")

    # Tokenizer + data module
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_module = MaseDataModule(
        name="sst2",
        batch_size=16,
        num_workers=4,
        max_token_len=128,
        tokenizer=tokenizer,
        model_name="bert-base-uncased",
    )

    search(
        model=model,
        model_info=model_info,
        task="cls",
        dataset_info=data_module.dataset_info,
        data_module=data_module,
        search_config=args.config,
        save_path=args.save_dir,
        accelerator="gpu",
    )


if __name__ == "__main__":
    main()
