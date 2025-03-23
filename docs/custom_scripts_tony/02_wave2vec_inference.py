import torch
import torch.nn.functional as F
import jiwer
from transformers import Wav2Vec2Processor, AutoModelForCTC
from chop.models import DataCollatorCTCWithPadding, CombinedWav2Vec2CTC

from chop.models import get_model, ModelFactory
from chop import MaseGraph
import chop.passes as passes
from datasets import load_dataset
import string
from pyctcdecode import build_ctcdecoder

# -------------------------------
# 1. Define the model and dataset
# -------------------------------

checkpoint = "facebook/wav2vec2-base-960h"
tokenizer_checkpoint = "facebook/wav2vec2-base-960h"
dataset_name = "nyalpatel/condensed_librispeech_asr"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
tokenizer = processor.tokenizer
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

model = AutoModelForCTC.from_pretrained(checkpoint)
encoder = model.wav2vec2    # static, FX-friendly
ctc_head = model.lm_head    # dynamic CTC head, separate this

# -------------------------------
# 2. Import dataset
# -------------------------------

dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
sample_list = list(dataset.take(50))
for sample in sample_list:
    sample["text"] = sample["text"].lower()

sample_audio = {
    "array": sample_list[0]["audio"]["array"],
    "sampling_rate": sample_list[0]["audio"]["sampling_rate"]
}
input_values = processor(
    audio=sample_audio["array"],
    sampling_rate=sample_audio["sampling_rate"],
    return_tensors="pt"
).input_values

# -------------------------------
# 3. Prepare dummy inputs
# -------------------------------

dummy_inputs = {
    "input_values": input_values,
}

mg = MaseGraph(
    encoder,
    hf_input_names=list(dummy_inputs.keys())
)

# mg.draw("wave2vec_inference_graph.svg")
mg, _ = passes.init_metadata_analysis_pass(mg)
print("MASE graph created successfully.")

# -------------------------------------------------------------------
# 4. Define a helper function for full inference (encoder + CTC head)
# -------------------------------------------------------------------

def full_inference_beam(encoder, ctc_head, input_values, beam_width=10):
    """
    Run the encoder (tracked by MASE) and then apply the CTC head.
    Instead of greedy decoding, use beam search (via pyctcdecode) to generate the transcription.
    """
    with torch.no_grad():
        # Run the encoder to get hidden states.
        encoder_outputs = encoder(input_values)
        
        # MASE-traced models return dicts, extract the tensor
        if isinstance(encoder_outputs, dict):
            hidden_states = encoder_outputs["last_hidden_state"]
        else:
            hidden_states = encoder_outputs.last_hidden_state  # Original model case
            
        # Apply the CTC head to obtain logits.
        logits = ctc_head(hidden_states)  # shape: [batch, seq_len, vocab_size]
        # Compute log probabilities for beam search.
        log_probs = F.log_softmax(logits, dim=-1)
        # Assume batch size 1; extract the sequence of log probabilities.
        log_probs_np = log_probs.cpu().numpy()[0]  # shape: [seq_len, vocab_size]
        # Use the pyctcdecode beam search decoder.
        transcription = decoder.decode(log_probs_np, beam_width=beam_width).lower()
    return transcription

# -------------------------------------------------------------------
# 5. Evaluate the model on the sample subset using the full inference
# -------------------------------------------------------------------

def evaluate_model(encoder, ctc_head, processor, dataset):
    total_wer = 0.0
    for idx, sample in enumerate(dataset):
        audio_input = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        reference = sample["text"]

        inputs = processor(
            audio=audio_input,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_values

        transcription = full_inference_beam(encoder, ctc_head, inputs)
        wer = jiwer.wer(reference, transcription)
        total_wer += wer

        print(f"Sample {idx}:")
        print("  Reference   :", reference)
        print("  Transcription:", transcription)
        print("  WER         :", wer)
    
    avg_wer = total_wer / len(dataset)
    print("\nAverage WER:", avg_wer)
    return avg_wer

print("\nEvaluating the original model on the sample subset:")
evaluate_model(mg.model, ctc_head, processor, sample_list)
