import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from chop.models import get_model, ModelFactory
from chop import MaseGraph
import chop.passes as passes
from datasets import load_dataset
import string

def normalize_text(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

# Set checkpoint - use the Wav2Vec2 checkpoint
checkpoint = "wav2vec2-base"
    # Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = get_model(checkpoint, pretrained=True)

dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
sample_list = list(dataset.take(50))
for sample in sample_list:
    sample["text"] = normalize_text(sample["text"])

sample_audio = {
    "array": sample_list[0]["audio"]["array"],
    "sampling_rate": sample_list[0]["audio"]["sampling_rate"]
}

# Process the audio input
input_values = processor(
    audio=sample_audio["array"],  # remove batch dimension and convert to numpy array     #.squeeze(0).numpy(),
    sampling_rate=sample_audio["sampling_rate"], 
    return_tensors="pt"
).input_values

# Prepare dummy inputs matching Wav2Vec2's expected signature
dummy_inputs = {
    "input_values": input_values,
    "attention_mask": torch.ones_like(input_values[:, :1], dtype=torch.long),
}

# Create MASE graph
mg = MaseGraph(
    model,
    # Use the keys from dummy_inputs
    hf_input_names=list(dummy_inputs.keys())
)


mg.draw("wave2vec_graph.svg")
# Initialize metadata analysis passes
mg, _ = passes.init_metadata_analysis_pass(mg)
#mg, _ = passes.add_common_metadata_analysis_pass(mg)

# Print confirmation
print("MASE graph created successfully.") 