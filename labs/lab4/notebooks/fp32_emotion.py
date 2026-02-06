import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# model_name = "AnkitAI/deberta-xlarge-base-emotions-classifier"
# if you meet OOM error, try this smaller model, but the quantization effect may not be obvious later
model_name = "AnkitAI/deberta-v3-small-base-emotions-classifier" 
model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
label2emotion = {idx: emotion for emotion, idx in model.config.label2id.items()}


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


text = "I'm so happy with the results!"
emotion, top3 = predict_emotion(model, tokenizer, text)

print("Index to Emotion Mapping:", label2emotion)
print("Input text:", text)
print("Detected Emotion:", emotion)
print(f"top3 logits: {top3[0]}, top3 indices: {top3[1]}")
