# Run these to run the test on golab
# !python /content/mase-individual/test/test_runtime_analysis_pass.py

import sys
from pathlib import Path

# Ensure the "src" folder is on the Python path so that "chop" can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest
import torch
import logging
import numpy as np

from pyctcdecode import build_ctcdecoder
from chop.tools import get_tokenized_dataset  # type: ignore
from transformers import AutoModelForCTC
from chop import MaseGraph
import chop.passes as passes  # type: ignore
from chop.passes.graph import (
    init_metadata_analysis_pass,
    runtime_analysis_pass,
    onnx_runtime_interface_pass,
)
from chop.dataset import MaseDataModule

# Suppress extraneous log output from pyctcdecode and other libraries.
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)

class TestRuntimeAnalysisPass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the test using the real CondensedLibrispeechASR dataset and a Wav2Vec2 model.
        This follows the same initialization logic as your working script.
        """
        # 1. Define model checkpoint and dataset
        cls.checkpoint = "facebook/wav2vec2-base-960h"
        cls.dataset_name = "nyalpatel/condensed_librispeech_asr"

        # 2. Retrieve tokenizer/processor for Wav2Vec2
        tokenized_dataset, tokenizer, processor = get_tokenized_dataset(
            dataset=cls.dataset_name,
            checkpoint=cls.checkpoint,
            tokenizer_checkpoint=cls.checkpoint,
            return_tokenizer=True,
            return_processor=True,
        )
        cls.tokenizer = tokenizer
        cls.processor = processor

        # Build decoder for CTC using the tokenizer's vocab
        vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
        cls.decoder = build_ctcdecoder(vocab)

        # 3. Load the model and extract the encoder + CTC head
        model = AutoModelForCTC.from_pretrained(cls.checkpoint)
        cls.encoder = model.wav2vec2  # static, FX-friendly
        cls.ctc_head = model.lm_head  # dynamic CTC head

        # 4. Create a MaseDataModule with the same parameters you used in your script
        cls.batch_size = 2
        cls.data_module = MaseDataModule(
            name=cls.dataset_name,
            batch_size=cls.batch_size,
            model_name=cls.checkpoint,
            num_workers=0,
            processor=cls.processor
        )
        cls.data_module.setup()

        # 6. Create the MaseGraph from the encoder and run the metadata passes
        cls.mg = MaseGraph(
            cls.encoder,
            hf_input_names=["input_values", "attention_mask"],
        )

        # (a) Initialize metadata analysis
        cls.mg, _ = init_metadata_analysis_pass(cls.mg)

        # (b) Common metadata pass with a dummy input
        dummy_in = {
            "input_values": torch.zeros((1, 16000), dtype=torch.float32),
            "attention_mask": torch.ones((1, 16000), dtype=torch.long),
        }
        cls.mg, _ = passes.add_common_metadata_analysis_pass(
            cls.mg,
            pass_args={
                "dummy_in": dummy_in,
                "add_value": True,
                "force_device_meta": False,
            }
        )

        # (c) Optionally, run the ONNX interface pass
        cls.mg, _ = onnx_runtime_interface_pass(
            cls.mg,
            pass_args={
                "smoothquant": True,
                "alpha": 0,
                "model": cls.checkpoint,
                "task": "ctc",
                "dataset": cls.dataset_name,
                "accelerator": "cuda",
                "data_module": cls.data_module,
                "batch_size": cls.batch_size
            }
        )

    def test_ctc_wer(self):
        """
        Test the runtime_analysis_pass on a CTC pipeline (Wave2Vec2), ensuring it completes
        end-to-end and returns the expected output metrics.
        """
        # Pass arguments for runtime analysis
        runtime_analysis_config = {
            "num_batches": 10,          # small number for testing
            "num_GPU_warmup_batches": 2,
            "test": True,              # use the test dataloader
            "data_module": self.data_module,
            "model": self.checkpoint,
            "accelerator": "cuda",     # set to "cpu" if no GPU
            "task": "ctc",
            "decoder": self.decoder,
            "beam_width": 10,
            "tokenizer": self.processor.tokenizer,
            "batch_size": self.batch_size,
            "sample_rate": 16000,
            "ctc_head": self.ctc_head,
        }

        # Run the runtime_analysis_pass
        _, results = runtime_analysis_pass(self.mg, pass_args=runtime_analysis_config)

        # Check that the results contain the expected keys
        self.assertIn("Average WER", results)
        self.assertIn("Average Latency", results)
        self.assertIn("Average RTF", results)
        self.assertIn("Average GPU Power Usage", results)
        self.assertIn("Inference Energy Consumption", results)

        # Confirm WER is valid
        wer_value = results["Average WER"]
        self.assertFalse(np.isnan(wer_value), "WER should not be NaN")

        # Print results for inspection
        print("Test completed. Results:")
        for k, v in results.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    unittest.main()
