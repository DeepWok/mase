import torch
import torch.nn.functional as F
import jiwer
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
from chop.models import get_model, ModelFactory
from chop import MaseGraph
import chop.passes as passes
from datasets import load_dataset
import string
from pyctcdecode import build_ctcdecoder

# def normalize_text(text):
#     return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

full_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
encoder = full_model.wav2vec2
ctc_head = full_model.lm_head

vocab = processor.tokenizer.convert_ids_to_tokens(range(processor.tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
sample_list = list(dataset.take(50))
for sample in sample_list:
    # sample["text"] = normalize_text(sample["text"])
    sample["text"] = sample["text"].lower()

# Use the first sample for building dummy inputs
sample_audio = {
    "array": sample_list[0]["audio"]["array"],
    "sampling_rate": sample_list[0]["audio"]["sampling_rate"]
}
input_values = processor(
    audio=sample_audio["array"],
    sampling_rate=sample_audio["sampling_rate"],
    return_tensors="pt"
).input_values

dummy_inputs = {
    "input_values": input_values,
    "attention_mask": torch.ones_like(input_values[:, :1], dtype=torch.long),
}

mg = MaseGraph(
    encoder,
    hf_input_names=list(dummy_inputs.keys())
)
mg.draw("wave2vec_inference_graph.svg")
mg, _ = passes.init_metadata_analysis_pass(mg)
print("MASE graph created successfully.")

# -------------------------------------------------------------------
# 5. Define a helper function for full inference (encoder + CTC head)
# -------------------------------------------------------------------
# def full_inference(encoder, ctc_head, processor, input_values):
#     with torch.no_grad():
#         # Get encoder outputs from the extracted encoder
#         encoder_outputs = encoder(input_values).last_hidden_state
#         logits = ctc_head(encoder_outputs)
#         predicted_ids = torch.argmax(logits, dim=-1)
#         # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#         transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower()
#     return transcription

def full_inference_beam(encoder, ctc_head, processor, input_values, beam_width=10):
    """
    Run the encoder (tracked by MASE) and then apply the CTC head.
    Instead of greedy decoding, use beam search (via pyctcdecode) to generate the transcription.
    """
    with torch.no_grad():
        # Run the encoder to get hidden states.
        encoder_outputs = encoder(input_values)
        hidden_states = encoder_outputs.last_hidden_state  # shape: [batch, seq_len, hidden_dim]
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
# 6. Evaluate the model on the sample subset using the full inference
#    (Note: Only the encoder is tracked; the CTC head is applied separately.)
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

        transcription = full_inference_beam(encoder, ctc_head, processor, inputs)
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
evaluate_model(encoder, ctc_head, processor, sample_list)
