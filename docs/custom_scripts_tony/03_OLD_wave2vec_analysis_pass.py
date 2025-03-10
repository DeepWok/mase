import torch
import torch.fx as fx
import torch.nn.functional as F
import jiwer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from chop.models import get_model, ModelFactory, DataCollatorCTCWithPadding
from chop.tools import get_trainer
from chop import MaseGraph
import chop.passes as passes
from datasets import load_dataset
import string
from pyctcdecode import build_ctcdecoder
from datasets import Dataset, DatasetDict

checkpoint = "DeepWokLab/bert-tiny"
tokenizer_checkpoint = "DeepWokLab/bert-tiny"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)







checkpoint = "wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

full_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
encoder = full_model.wav2vec2      # The encoder component (tracked by MASE)
ctc_head = full_model.lm_head      # The CTC head (kept untracked)

vocab = processor.tokenizer.convert_ids_to_tokens(range(processor.tokenizer.vocab_size))
decoder = build_ctcdecoder(vocab)

dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
sample_list = dataset.take(50)     # ['file', 'audio', 'text', 'speaker_id', 'chapter_id', 'id']
sample_list = list(sample_list.remove_columns(["speaker_id", "chapter_id", "id"]))
for sample in sample_list:
    sample["text"] = sample["text"].lower()

# print("sample", sample_list.column_names)


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

mg, _ = passes.init_metadata_analysis_pass(mg)
# mg, _ = passses.add_
mg.draw("wave2vec_analysis_pass_graph.svg")
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

def count_feature_projection_analysis_pass(mg, pass_args={}):
    feature_projection_modules = 0
    feature_projection_functions = 0  # In this example, we focus on modules.

    for node in mg.fx_graph.nodes:
        if node.op == "call_module" and "feature_projection" in str(node.target).lower():
            logger.info(f"Found feature projection module: {node.target}")
            feature_projection_modules += 1
        else:
            logger.debug(f"Skipping node: {node.target}")
    return mg, {"feature_projection_count": feature_projection_modules + feature_projection_functions}

def remove_feature_projection_transform_pass(mg, pass_args={}):
    for node in list(mg.fx_graph.nodes):  # Convert to list to safely modify graph
        if node.op == "call_module" and "feature_projection" in str(node.target).lower():
            logger.info(f"Removing feature projection module: {node.target}")
            
            # Replace all users of the feature projection node with its input
            parent_node = node.args[0]
            logger.debug(f"Parent node: {parent_node}")
            node.replace_all_uses_with(parent_node)

            # Erase the feature projection node
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

# Optionally remove feature projection layers (if desired for optimization)
mg, _ = remove_feature_projection_transform_pass(mg)
mg, analysis_out = count_feature_projection_analysis_pass(mg)
logger.info(f"Feature projection count after removal: {analysis_out['feature_projection_count']}")

model = mg.model
mg.draw("wave2vec_transformed_analysis_pass_graph.svg")

# ---------------------------
# Evaluate the modified model
# ---------------------------
# data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


def preprocess_function(example):
    """
    Prepares audio samples for Wav2Vec2 by extracting input values,
    creating attention masks, and tokenizing transcriptions.
    """

    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]  # ✅ Ensure this is an integer

    # Convert audio into model-compatible inputs
    inputs = processor(audio=audio_array, sampling_rate=int(sampling_rate), return_tensors="pt", padding=True)
    attention_mask = torch.ones(inputs.input_values.shape, dtype=torch.long)  # ✅ Ensures correct batch size

    # Tokenize text transcription
    with processor.as_target_processor():
        labels = processor.tokenizer(example["text"], return_tensors="pt").input_ids

    return {
        "input_values": inputs.input_values.squeeze(0),  # ✅ Ensure correct shape
        "attention_mask": attention_mask.squeeze(0), # ✅ Ensure correct attention mask
        "labels": labels.squeeze(0)  # ✅ Ensure correct label shape
    }


# Convert sample list to Dataset
small_dataset = Dataset.from_list(sample_list)

# Apply preprocessing
small_dataset = small_dataset.map(
    preprocess_function, 
    remove_columns=["speaker_id", "file", "id", "chapter_id", "audio"]
)

# Convert to DatasetDict (as required by MASE)
tokenized_dataset = DatasetDict({
    "train": small_dataset,
    "test": small_dataset
})

print("Tokenized dataset:", tokenized_dataset.keys())
print("Tokenized dataset :", tokenized_dataset["train"].keys())
print("Tokenized dataset :", tokenized_dataset["test"].keys())

# ✅ Create Trainer
trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=tokenized_dataset,
    tokenizer=processor.tokenizer,
    evaluate_metric="wer",
)

# Evaluate Model
eval_results = trainer.evaluate()
print(f"Evaluation WER: {eval_results['wer']}")  # ✅ Correct final print output



# def preprocess_function(batch):
#     audio = batch["audio"]
#     inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
#     batch["input_values"] = inputs.input_values[0]
#     batch["attention_mask"] = inputs.attention_mask[0]
#     batch["labels"] = processor.tokenizer(batch["text"]).input_ids
#     return batch

# dataset = dataset.map(preprocess_function, remove_columns=["audio", "text"])
# data_collator = DataCollatorForCTC(processor=processor, padding=True)

# training_args = TrainingArguments(
#     output_dir="mase-trainer",
#     evaluation_strategy="steps",
#     per_device_train_batch_size=8,
#     gradient_accumulation_steps=2,
#     num_train_epochs=1,
#     save_steps=500,
#     eval_steps=500,
#     logging_steps=100,
#     learning_rate=1e-4,
#     warmup_steps=500,
#     save_total_limit=2,
#     fp16=True,
#     push_to_hub=False,
# )

# trainer = get_trainer(
#     model=mg.model,
#     tokenized_dataset=dataset["train"],
#     tokenizer=processor.feature_extractor,
#     evaluate_metric="wer",
#     data_collator=data_collator, 
# )

# eval_results = trainer.evaluate()
# print(f"Evaluation WER: {eval_results['wer']}")

# def preprocess_function(example):
#     """
#     Converts raw audio samples into model inputs (input_values) and labels (text tokens).
#     """
#     audio_array = example["audio"]["array"]
#     sampling_rate = example["audio"]["sampling_rate"]

#     # Convert audio into model-compatible inputs
#     inputs = processor(audio=audio_array, sampling_rate=sampling_rate, return_tensors="pt")

#     # Tokenize text transcription
#     with processor.as_target_processor():
#         labels = processor.tokenizer(example["text"], return_tensors="pt").input_ids

#     return {
#         "input_values": inputs.input_values.squeeze(0),  # ✅ Ensure correct shape
#         "attention_mask": torch.ones(inputs.input_values.shape, dtype=torch.long),  # ✅ Required for transformers
#         "labels": labels.squeeze(0)  # ✅ Ensure correct label shape
#     }


# small_dataset = Dataset.from_list(sample_list).map(preprocess_function, remove_columns=["speaker_id", "file", "id", "chapter_id", "audio"])
# # small_dataset = small_dataset.shuffle(seed=42)

# tokenized_dataset = DatasetDict({
#     "train": small_dataset,
#     "test": small_dataset
# })

# training_args = TrainingArguments(
#     output_dir="mase-trainer",
#     use_mps_device=False,
#     report_to="none",
#     num_train_epochs=1,
#     remove_unused_columns=False,
# )

# data_collator = DataCollatorWithPadding(tokenizer=processor, padding=True)

# trainer = get_trainer(
#     model=mg.model,
#     tokenized_dataset=tokenized_dataset,
#     tokenizer=processor,
#     evaluate_metric="wer",
#     data_collator=data_collator, 
# )

# # Evaluate accuracy
# eval_results = trainer.evaluate()
# print(f"Evaluation WER: {eval_results}")





# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# vocab = processor.tokenizer.convert_ids_to_tokens(range(processor.tokenizer.vocab_size))
# decoder = build_ctcdecoder(vocab)

# dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
# sample_list = list(dataset.take(50))
# for sample in sample_list:
#     sample["text"] = sample["text"].lower()

# # Use the first sample for building dummy inputs
# sample_audio = {
#     "array": sample_list[0]["audio"]["array"],
#     "sampling_rate": sample_list[0]["audio"]["sampling_rate"]
# }
# input_values = processor(
#     audio=sample_audio["array"],
#     sampling_rate=sample_audio["sampling_rate"],
#     return_tensors="pt"
# ).input_values



# trainer = get_trainer(
#     model=mg.model,
#     tokenized_dataset=dataset,
#     tokenizer=tokenizer,
#     evaluate_metric="accuracy",
# )

# # Evaluate accuracy
# eval_results = trainer.evaluate()














# def full_inference_beam(model, ctc_head, processor, input_values, beam_width=10):
#     """
#     Run the encoder (tracked by MASE) and then apply the CTC head.
#     Instead of greedy decoding, use beam search (via pyctcdecode) to generate the transcription.
#     """
#     with torch.no_grad():
#         attn_mask = torch.ones_like(input_values, dtype=torch.long)
#         # Run the encoder to get hidden states.
#         encoder_outputs = model(input_values, attention_mask=attn_mask)
#         # hidden_states = encoder_outputs.last_hidden_state  # shape: [batch, seq_len, hidden_dim]
#         hidden_states = encoder_outputs["last_hidden_state"]

#         # Apply the CTC head to obtain logits.
#         logits = ctc_head(hidden_states)  # shape: [batch, seq_len, vocab_size]
#         # Compute log probabilities for beam search.
#         log_probs = F.log_softmax(logits, dim=-1)
#         # Assume batch size 1; extract the sequence of log probabilities.
#         log_probs_np = log_probs.cpu().numpy()[0]  # shape: [seq_len, vocab_size]
#         # Use the pyctcdecode beam search decoder.
#         transcription = decoder.decode(log_probs_np, beam_width=beam_width).lower()
#     return transcription

# # ---------------------------
# # Evaluate the model on the sample subset using full inference
# # ---------------------------
# def evaluate_model(model, ctc_head, processor, dataset):
#     total_wer = 0.0
#     for idx, sample in enumerate(dataset):
#         audio_input = sample["audio"]["array"]
#         sampling_rate = sample["audio"]["sampling_rate"]
#         reference = sample["text"]

#         inputs = processor(
#             audio=audio_input,
#             sampling_rate=sampling_rate,
#             return_tensors="pt"
#         ).input_values

#         transcription = full_inference_beam(model, ctc_head, processor, inputs)
#         wer = jiwer.wer(reference, transcription)
#         total_wer += wer

#         print(f"Sample {idx}:")
#         print("  Reference   :", reference)
#         print("  Transcription:", transcription)
#         print("  WER         :", wer)
    
#     avg_wer = total_wer / len(dataset)
#     print("\nAverage WER:", avg_wer)
#     return avg_wer

# print("\nEvaluating the original model on the sample subset:")
# evaluate_model(model, ctc_head, processor, sample_list)










# ---------------------------
# Define full inference function (encoder + CTC head)
# ---------------------------

# class DebugInterpreter(fx.Interpreter):
#     def run_node(self, n):
#         result = super().run_node(n)
#         if isinstance(result, torch.Tensor):
#             print(f"Node {n.name} output shape: {result.shape}")
#         else:
#             print(f"Node {n.name} output: {result}")
#         return result

# attn_mask = torch.ones_like(input_values, dtype=torch.long)
# print("Debugging intermediate outputs:")
# debug_interpreter = DebugInterpreter(model)
# # Pass inputs as positional arguments:
# debug_interpreter.run(input_values, attn_mask)