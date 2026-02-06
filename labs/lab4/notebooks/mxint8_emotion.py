import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mase_cuda.mxint8.linear import QLinearPacked

init_memory = torch.cuda.memory_allocated()  # in bytes
model_name = "AnkitAI/deberta-v3-small-base-emotions-classifier" 
model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float32).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
label2emotion = {idx: emotion for emotion, idx in model.config.label2id.items()}

mxint8_group_size = 32
assert model.config.hidden_size % mxint8_group_size == 0
assert model.config.intermediate_size % mxint8_group_size == 0

text = "I'm so happy with the results!"


# Example usage
@torch.no_grad()
def predict_emotion(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = logits.argmax(dim=1)
    predictions = label2emotion[predictions.item()]
    top3_values, top3_indices = torch.topk(logits, 3)
    top3_values = top3_values.cpu().tolist()
    top3_indices = top3_indices.cpu().tolist()
    return predictions, (top3_values, top3_indices)


# check the GPU memory usage of FP32 model
torch.cuda.reset_peak_memory_stats()
emotion_fp32, top3_fp32 = predict_emotion(model, tokenizer, text)
peak_memory_fp32 = torch.cuda.max_memory_allocated() - init_memory  # in bytes


def set_layer_by_name(module: torch.nn.Module, name: str, new_layer: torch.nn.Module):
    """
    Replace a layer (`new_layer`) in a model (`module`) by its `name`.
    """
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = module
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_layer)
    else:
        setattr(module, name, new_layer)


for layer_name, layer in model.named_modules():
    if not isinstance(layer, torch.nn.Linear):
        continue
    if "classifier" in layer_name:
        continue
    layer.cuda()
    layer_q = QLinearPacked.build_from_linear(layer, group_size=mxint8_group_size)
    set_layer_by_name(model, layer_name, layer_q)
    del layer
    torch.cuda.empty_cache()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
emotion_mxint8, top3_mxint8 = predict_emotion(model, tokenizer, text)
peak_memory_mxint8 = torch.cuda.max_memory_allocated() - init_memory  # in bytes

print(f"FP32 model peak memory: {peak_memory_fp32/1024**2:.4f} MB")
print(f"PF32 prediction: {emotion_fp32}")
print(f"FP32 top3 logits: {top3_fp32[0]}, indices: {top3_fp32[1]}")

print(f"MXINT8 model peak memory: {peak_memory_mxint8/1024**2:.4f} MB")
print(f"MXINT8 prediction: {emotion_mxint8}")
print(f"MXINT8 top3 logits: {top3_mxint8[0]}, indices: {top3_mxint8[1]}")