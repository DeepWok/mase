# Advanced: PIM Simulation Tutorial

This tutorial demonstrates how to use the Mase framework to model and simulate the models' behaviour on PIM(process in memory) devices. 

In this tutorial, we will focus on simulating the behaviour of PCM devices (phase change memory). For detailed information about the simulating parameters please refer to [Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators](https://www.nature.com/articles/s41467-023-40770-4)

About the detail explanation of the device simulation, we can see the full main documentation for more details. Here just show the simulation and evaluation pipeline with mase framework.

## Section 1. Evaluation with golden model
In this section, we evaluate the baseline `RoBERTa` model on the `MNLI` (Multi-Genre Natural Language Inference) dataset. The MNLI dataset consists of pairs of sentences (a premise and a hypothesis) and the goal is to predict whether the premise entails, contradicts, or is neutral towards the hypothesis.

We use the `JeremiahZ/roberta-base-mnli` model, which is a RoBERTa-base model fine-tuned on MNLI.

```python
from transformers import RobertaForSequenceClassification, AutoTokenizer
from chop.dataset.nlp.text_entailment import TextEntailmentDatasetMNLI
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).squeeze(-1)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

pretrained = "JeremiahZ/roberta-base-mnli"
tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = RobertaForSequenceClassification.from_pretrained(pretrained)

# Load a small subset of MNLI validation set for quick evaluation
dataset = TextEntailmentDatasetMNLI(split="validation_matched", tokenizer=tokenizer, max_token_len=128, num_workers=4)
dataloader = DataLoader(dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracy = evaluate(model, dataloader, device)
print(f"Golden Model Accuracy: {accuracy:.4f}")
```

**Output:**
```text
Evaluating: 100%|██████████| 614/614 [00:35<00:00, 17.13it/s]Golden Model Accuracy: 0.8728
```

## Section 2. Evaluation with transformed model
Now we apply the `pim_matmul_transform_pass` to simulate the PIM hardware. We configure the transform to use PCM (Phase Change Memory) tiles with a core size of 256. We can enable various non-idealities like programming noise and read noise to see how they impact the model's performance.

### Configuration Details:
- `tile_type`: The type of PIM technology to simulate (e.g., 'pcm', 'sram', 'reram').
- `core_size`: The size of the crossbar array (e.g., 256x256).
- `num_bits`: The number of bits used for weights and activations.
- `programming_noise`: Simulates variability during the programming of PIM cells.
- `read_noise`: Simulates noise during the read-out process.
- `ir_drop`: Simulates voltage drops along the crossbar lines.
- `out_noise`: Simulates noise at the output of the crossbar.

```python
from chop.passes.module.transforms import pim_matmul_transform_pass
import copy

q_config = {
    "by": "type",
    "linear": {
        "config": {
            "tile_type": "pcm",
            "core_size": 256,
            "num_bits": 8,
            "programming_noise": True,
            "read_noise": True,
            "ir_drop": True,
            "out_noise": True,
        }
    },
}

# Apply the transform pass
transformed_model = copy.deepcopy(model)
qmodel, _ = pim_matmul_transform_pass(transformed_model, q_config)

q_accuracy = evaluate(qmodel, dataloader, device)
print(f"Transformed Model Accuracy (with PIM noise): {q_accuracy:.4f}")
```

**Output:**
```text
Evaluating: 100%|██████████| 614/614 [15:49<00:00,  1.55s/it]Transformed Model Accuracy (with PIM noise): 0.3293
```

## Conclusion
In this tutorial, we demonstrated how to:
1. Load a pretrained RoBERTa model and evaluate it on the MNLI dataset.
2. Use `pim_matmul_transform_pass` to simulate hardware non-idealities for PIM devices.
3. Evaluate the impact of these non-idealities on model accuracy.
