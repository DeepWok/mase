from transformers.models.llama.configuration_llama import LlamaConfig


if __name__ == "__main__":
    cfg = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    print(cfg)
