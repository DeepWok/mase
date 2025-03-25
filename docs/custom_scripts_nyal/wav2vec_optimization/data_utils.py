"""
Model importing and dataset setup utilities.
"""

import logging
from pathlib import Path
from transformers import AutoModelForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
from chop.tools import get_tokenized_dataset
from chop.models import DataCollatorCTCWithPadding
from chop.dataset.audio.speech_recognition import CondensedLibrispeechASRDataset
from chop.dataset import MaseDataModule
from config import CHECKPOINT, TOKENIZER_CHECKPOINT, DATASET_NAME, BATCH_SIZE

# Set up logging
logger = logging.getLogger(__name__)

def import_model_and_dataset():
    """Import model, tokenizer, and dataset"""
    logger.info("Importing model and dataset...")
    
    # Get tokenized dataset, tokenizer, and processor
    tokenized_dataset, tokenizer, processor = get_tokenized_dataset(
        dataset=DATASET_NAME,
        checkpoint=TOKENIZER_CHECKPOINT,
        return_tokenizer=True,
        return_processor=True,
    )
    
    vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
    decoder = build_ctcdecoder(vocab)
    
    # Load model
    model = AutoModelForCTC.from_pretrained(CHECKPOINT)
    encoder = model.wav2vec2  # static, FX-friendly
    ctc_head = model.lm_head    # dynamic CTC head, separate this
    
    # Setup data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    
    data_module = MaseDataModule(
        name=DATASET_NAME,
        batch_size=BATCH_SIZE,
        model_name=CHECKPOINT,
        num_workers=0,
        processor=processor,
    )
    data_module.prepare_data()
    data_module.setup()
    
    logger.info("Model and dataset imported successfully")
    
    return {
        "encoder": encoder,
        "ctc_head": ctc_head,
        "tokenized_dataset": tokenized_dataset,
        "tokenizer": tokenizer,
        "processor": processor,
        "data_collator": data_collator,
        "vocab": vocab,
        "decoder": decoder,
        "data_module": data_module,
        "checkpoint": CHECKPOINT,
        "dataset_name": DATASET_NAME,
    }
