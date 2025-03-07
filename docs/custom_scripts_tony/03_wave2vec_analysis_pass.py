import torch
import torch.fx as fx
import jiwer
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
from chop.models import get_model, ModelFactory
from chop import MaseGraph
import chop.passes as passes
from datasets import load_dataset
import string


checkpoint = "wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# Load the full ASR model and extract encoder and CTC head.
full_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
encoder = full_model.wav2vec2      # The encoder component (tracked by MASE)
ctc_head = full_model.lm_head       # The CTC head (kept untracked)

dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
sample_list = list(dataset.take(50))
for sample in sample_list:
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
mg.draw("wave2vec_analysis_pass_graph.svg")
mg, _ = passes.init_metadata_analysis_pass(mg)
print("MASE graph created successfully.")

# ---------------------------
# Dropout Analysis and Removal Passes
# ---------------------------
from chop.tools import get_logger
logger = get_logger("mase_logger")
logger.setLevel("INFO")

def count_dropout_analysis_pass(mg, pass_args={}):
    dropout_modules = 0
    dropout_functions = 0  # In this example, we focus on modules.
    
    for node in mg.fx_graph.nodes:
        if node.op == "call_module" and "dropout" in str(node.target).lower():
            logger.info(f"Found dropout module: {node.target}")
            dropout_modules += 1
        else:
            logger.debug(f"Skipping node: {node.target}")
    return mg, {"dropout_count": dropout_modules + dropout_functions}

def remove_dropout_transform_pass(mg, pass_args={}):
    for node in list(mg.fx_graph.nodes):  # Convert to list to safely modify graph
        if node.op == "call_module" and "dropout" in str(node.target).lower():
            logger.info(f"Removing dropout module: {node.target}")
            
            # Replace all users of the dropout node with its input
            parent_node = node.args[0]
            logger.debug(f"Parent node: {parent_node}")
            node.replace_all_uses_with(parent_node)

            # Erase the dropout node
            mg.fx_graph.erase_node(node)
        else:
            logger.debug(f"Skipping node: {node.target}")
    return mg, {}

# Run dropout analysis before removal
mg, analysis_out = count_dropout_analysis_pass(mg)
logger.info(f"Dropout count before removal: {analysis_out['dropout_count']}")

# Optionally remove dropout layers (if desired for optimization)
mg, _ = remove_dropout_transform_pass(mg)
mg, analysis_out = count_dropout_analysis_pass(mg)
logger.info(f"Dropout count after removal: {analysis_out['dropout_count']}")

mg.draw("wave2vec_transformed_analysis_pass_graph.svg")

# ---------------------------
# Define full inference function (encoder + CTC head)
# ---------------------------
def full_inference(encoder, ctc_head, processor, input_values):
    with torch.no_grad():
        encoder_outputs = encoder(input_values)
        hidden_states = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        logits = ctc_head(hidden_states)  # [batch, seq_len, vocab_size]
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower()
    return transcription

# ---------------------------
# Evaluate the model on the sample subset using full inference
# ---------------------------
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

        transcription = full_inference(encoder, ctc_head, processor, inputs)
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
