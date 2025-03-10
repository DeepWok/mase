
from chop.tools import get_tokenized_dataset # type: ignore
from transformers import AutoModelForSequenceClassification # type: ignore
from chop import MaseGraph
import chop.passes as passes # type: ignore
from chop.passes.module import report_trainable_parameters_analysis_pass # type: ignore
from chop.tools import get_trainer # type: ignore

checkpoint = "DeepWokLab/bert-tiny"
tokenizer_checkpoint = "DeepWokLab/bert-tiny"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"

mg = MaseGraph(
    model,
    hf_input_names=[
        "input_ids",
        "attention_mask",
        "labels",
    ],
)

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(mg)

mg, _ = passes.insert_lora_adapter_transform_pass(
    mg,
    pass_args={
        "rank": 6,
        "alpha": 1.0,
        "dropout": 0.5,
    },
)

mg.draw()

_, _ = report_trainable_parameters_analysis_pass(mg.model)

for param in mg.model.bert.embeddings.parameters():
    param.requires_grad = False

trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
    num_train_epochs=1,
)
trainer.train()

# Evaluate accuracy
eval_results = trainer.evaluate()
print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")

mg, _ = passes.fuse_lora_weights_transform_pass(mg)
eval_results = trainer.evaluate()
print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")

mg.export("tutorial_2_sft")