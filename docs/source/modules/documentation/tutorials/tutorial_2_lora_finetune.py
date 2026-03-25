# Tutorial 2: Finetuning Bert for Sequence Classification using a LoRA adapter

from pathlib import Path

print("=" * 60, flush=True)
print("Tutorial 2: LoRA Finetuning on BERT (IMDb)", flush=True)
print("=" * 60, flush=True)

checkpoint = "DeepWokLab/bert-tiny"
tokenizer_checkpoint = "DeepWokLab/bert-tiny"
dataset_name = "imdb"

# ── Step 1: Load dataset ───────────────────────────────────────────────────────
print("\n[1/7] Loading and tokenizing IMDb dataset...", flush=True)
from chop.tools import get_tokenized_dataset, get_trainer

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)
print(f"      Dataset loaded: {len(dataset['train'])} train / {len(dataset['test'])} test", flush=True)

# ── Step 2: Build MaseGraph ────────────────────────────────────────────────────
print("\n[2/7] Loading model and building MaseGraph...", flush=True)
from transformers import AutoModelForSequenceClassification
import chop.passes as passes
from chop import MaseGraph

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"

mg = MaseGraph(
    model,
    hf_input_names=["input_ids", "attention_mask", "labels"],
)
mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(mg)
print("      MaseGraph ready  ✓", flush=True)

# ── Step 3: Report trainable parameters ───────────────────────────────────────
print("\n[3/7] Reporting trainable parameters (full model)...", flush=True)
from chop.passes.module import report_trainable_parameters_analysis_pass

_, _ = report_trainable_parameters_analysis_pass(mg.model)

# Freeze embeddings
for param in mg.model.bert.embeddings.parameters():
    param.requires_grad = False
trainable = sum(p.numel() for p in mg.model.parameters() if p.requires_grad)
print(f"      Trainable after freezing embeddings: {trainable:,}", flush=True)

# ── Step 4: Evaluate before SFT ───────────────────────────────────────────────
print("\n[4/7] Evaluating baseline accuracy (before training)...", flush=True)
trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
)
eval_results = trainer.evaluate()
print(f"      [Baseline] Accuracy: {eval_results['eval_accuracy']:.4f}", flush=True)

# ── Step 5: Full SFT (1 epoch) ─────────────────────────────────────────────────
print("\n[5/7] Running full SFT (1 epoch)...", flush=True)
trainer.train()
eval_results = trainer.evaluate()
print(f"      [SFT] Accuracy after 1 epoch: {eval_results['eval_accuracy']:.4f}", flush=True)

mg.export(f"{Path.home()}/tutorial_2_sft")
print(f"      SFT checkpoint saved to {Path.home()}/tutorial_2_sft", flush=True)

# ── Step 6: Inject LoRA & train ────────────────────────────────────────────────
print("\n[6/7] Injecting LoRA adapter and training (1 epoch)...", flush=True)
mg, _ = passes.insert_lora_adapter_transform_pass(
    mg,
    pass_args={"rank": 6, "alpha": 1.0, "dropout": 0.5},
)

trainable_lora = sum(p.numel() for p in mg.model.parameters() if p.requires_grad)
print(f"      Trainable params with LoRA: {trainable_lora:,}", flush=True)

trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
    num_train_epochs=1,
)
trainer.train()
eval_results = trainer.evaluate()
print(f"      [LoRA] Accuracy after training: {eval_results['eval_accuracy']:.4f}", flush=True)

# ── Step 7: Fuse LoRA weights & export ────────────────────────────────────────
print("\n[7/7] Fusing LoRA weights and exporting...", flush=True)
mg, _ = passes.fuse_lora_weights_transform_pass(mg)
eval_results = trainer.evaluate()
print(f"      [LoRA fused] Accuracy: {eval_results['eval_accuracy']:.4f}", flush=True)

mg.export(f"{Path.home()}/tutorial_2_lora")
print(f"      LoRA checkpoint saved to {Path.home()}/tutorial_2_lora", flush=True)

print("\n" + "=" * 60, flush=True)
print("Tutorial 2 complete!", flush=True)
print("=" * 60, flush=True)
