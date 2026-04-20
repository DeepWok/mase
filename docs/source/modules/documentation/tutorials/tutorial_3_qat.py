# Tutorial 3: Running Quantization-Aware Training (QAT) on Bert

from pathlib import Path

print("=" * 60, flush=True)
print("Tutorial 3: Post-Training Quantization + QAT on BERT", flush=True)
print("=" * 60, flush=True)

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

# ── Step 1: Import model ───────────────────────────────────────────────────────
print("\n[1/5] Loading model...", flush=True)
from transformers import AutoModelForSequenceClassification
import chop.passes as passes
from chop import MaseGraph

# Option A: load from Tutorial 2 LoRA checkpoint
mg = MaseGraph.from_checkpoint(f"{Path.home()}/tutorial_2_lora")
print("      Loaded from tutorial_2_lora  ✓", flush=True)

# Option B: start from scratch (use if Tutorial 2 checkpoint is not available)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# model.config.problem_type = "single_label_classification"
# mg = MaseGraph(
#     model,
#     hf_input_names=["input_ids", "attention_mask", "labels"],
# )
# mg, _ = passes.init_metadata_analysis_pass(mg)
# mg, _ = passes.add_common_metadata_analysis_pass(mg)
# print("      Fresh MaseGraph built  ✓", flush=True)

# ── Step 2: Baseline evaluation ────────────────────────────────────────────────
print("\n[2/5] Loading dataset and evaluating baseline accuracy...", flush=True)
from chop.tools import get_tokenized_dataset, get_trainer

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)
print(f"      Dataset loaded: {len(dataset['train'])} train / {len(dataset['test'])} test", flush=True)

trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
)
eval_results = trainer.evaluate()
print(f"      [Baseline] Accuracy: {eval_results['eval_accuracy']:.4f}", flush=True)

# ── Step 3: Post-Training Quantization (PTQ) ──────────────────────────────────
print("\n[3/5] Applying integer quantization (PTQ)...", flush=True)

quantization_config = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "weight_width": 8,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

mg, _ = passes.quantize_transform_pass(mg, pass_args=quantization_config)
print("      quantize_transform_pass  ✓", flush=True)

trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
)
eval_results = trainer.evaluate()
print(f"      [PTQ] Accuracy: {eval_results['eval_accuracy']:.4f}", flush=True)

mg.export(f"{Path.home()}/tutorial_3_ptq")
print(f"      PTQ checkpoint saved to {Path.home()}/tutorial_3_ptq", flush=True)

# ── Step 4: Quantization-Aware Training (QAT) ─────────────────────────────────
print("\n[4/5] Running QAT (1 epoch)...", flush=True)
trainer.train()
eval_results = trainer.evaluate()
print(f"      [QAT] Accuracy: {eval_results['eval_accuracy']:.4f}", flush=True)

# ── Step 5: Export QAT checkpoint ─────────────────────────────────────────────
print("\n[5/5] Exporting QAT checkpoint...", flush=True)
mg.export(f"{Path.home()}/tutorial_3_qat")
print(f"      QAT checkpoint saved to {Path.home()}/tutorial_3_qat", flush=True)

print("\n" + "=" * 60, flush=True)
print("Tutorial 3 complete!", flush=True)
print("=" * 60, flush=True)
