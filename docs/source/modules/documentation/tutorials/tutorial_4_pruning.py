# Tutorial 4: Unstructured Pruning on Bert

from pathlib import Path

print("=" * 60, flush=True)
print("Tutorial 4: Unstructured Pruning on BERT (IMDb)", flush=True)
print("=" * 60, flush=True)

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

# ── Step 1: Import model ───────────────────────────────────────────────────────
print("\n[1/5] Loading model...", flush=True)
from transformers import AutoModelForSequenceClassification
import chop.passes as passes
from chop import MaseGraph

# Option A: start from scratch
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"

mg = MaseGraph(
    model,
    hf_input_names=["input_ids", "attention_mask", "labels"],
)
mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(mg)
print("      Fresh MaseGraph built  ✓", flush=True)

# Option B: load from Tutorial 3 QAT checkpoint (comment out Option A above)
# mg = MaseGraph.from_checkpoint(f"{Path.home()}/tutorial_3_qat")
# print("      Loaded from tutorial_3_qat  ✓", flush=True)

# ── Step 2: Load dataset & baseline evaluation ────────────────────────────────
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

# ── Step 3: Apply L1-norm unstructured pruning (50% sparsity) ─────────────────
print("\n[3/5] Applying L1-norm unstructured pruning (sparsity=0.5)...", flush=True)

pruning_config = {
    "weight": {"sparsity": 0.5, "method": "l1-norm", "scope": "local"},
    "activation": {"sparsity": 0.5, "method": "l1-norm", "scope": "local"},
}

mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)
print("      prune_transform_pass  ✓", flush=True)

# ── Step 4: Evaluate after pruning ────────────────────────────────────────────
print("\n[4/5] Evaluating accuracy after pruning (before finetuning)...", flush=True)
trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
    num_train_epochs=5,
)
eval_results = trainer.evaluate()
print(f"      [Pruned] Accuracy: {eval_results['eval_accuracy']:.4f}", flush=True)

# ── Step 5: Finetune to recover accuracy ──────────────────────────────────────
print("\n[5/5] Finetuning pruned model (5 epochs) to recover accuracy...", flush=True)
trainer.train()
eval_results = trainer.evaluate()
print(f"      [Pruned + finetuned] Accuracy: {eval_results['eval_accuracy']:.4f}", flush=True)

print("\n" + "=" * 60, flush=True)
print("Tutorial 4 complete!", flush=True)
print("=" * 60, flush=True)
