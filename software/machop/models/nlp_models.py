from torch import nn
from transformers import AutoTokenizer
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoConfig


def get_nlp_model(name, task, info, checkpoint=None, pretrained=True):

    if task not in ['classification', 'translation']:
        raise ValueError("task must be a valid value for NLP models")
    
    num_classes = info['num_classes']
    tokenizer = AutoTokenizer.from_pretrained(
        name, cache_dir='./model_cache_dir', 
        return_dict=True)
    if pretrained:
        print(f"Loaded tokenizer from {name}")
        if checkpoint is not None:
            model = AutoModel.from_pretrained(checkpoint)
            print(f"Loaded model from {checkpoint}")
        else:
            model = AutoModel.from_pretrained(
                name, return_dict=True,
                cache_dir='./model_cache_dir')
            print(f"Loaded model from {name} in HuggingFace")
    else:
        config = AutoConfig.from_pretrained(checkpoint)
        if task == 'classification':
            model = AutoModel.from_config(config=config)
        elif task == 'translation':
            model = AutoModelForSeq2SeqLM.from_config(config=config)

    if task == 'classification':
        classifier = nn.Linear(
            model.config.hidden_size, num_classes)
    else:
        classifier = None
    return {
        'model': model,
        'tokenizer': tokenizer,
        'classifier': classifier,
    }