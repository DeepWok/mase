import torch

from chop.passes.module.transforms.gptq import run_gptq
from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM


def _tiny_qwen3_moe(num_hidden_layers=2):
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        decoder_sparse_step=1,
        moe_intermediate_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3MoeForCausalLM(config)
    model.eval()
    return model


def _write_loader(path, *, nsamples=2, seqlen=8, vocab_size=64):
    torch.manual_seed(1)
    loader = []
    for _ in range(nsamples):
        input_ids = torch.randint(3, vocab_size, (1, seqlen), dtype=torch.long)
        target = input_ids.clone()
        target[:, :-1] = -100
        loader.append((input_ids, target))
    torch.save({"loader": loader, "seqlen": seqlen}, path)


def _gptq_config(loader_path, **overrides):
    config = {
        "model_name": "unused-for-file-loader",
        "device": "cpu",
        "dataset": f"file:{loader_path}",
        "nsamples": 2,
        "seqlen": 8,
        "format": "mxint",
        "weight_config": {"weight_block_size": 4, "weight_width": 4},
        "quantile_search": False,
        "clip_search_y": False,
        "cali_batch_size": 1,
        "max_layers": 1,
    }
    config.update(overrides)
    return config


def test_qwen3_moe_gptq_quantizes_sparse_experts(tmp_path):
    loader_path = tmp_path / "calib.pt"
    _write_loader(loader_path)
    model = _tiny_qwen3_moe(num_hidden_layers=2)

    q_proj_before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    gate_up_before = model.model.layers[0].mlp.experts.gate_up_proj.detach().clone()
    down_before = model.model.layers[0].mlp.experts.down_proj.detach().clone()

    run_gptq(model, _gptq_config(loader_path))

    assert model.model.layers[0].self_attn.q_proj.weight.shape == q_proj_before.shape
    assert model.model.layers[0].mlp.experts.gate_up_proj.shape == gate_up_before.shape
    assert model.model.layers[0].mlp.experts.down_proj.shape == down_before.shape
    assert not torch.equal(model.model.layers[0].self_attn.q_proj.weight, q_proj_before)
    assert not torch.equal(model.model.layers[0].mlp.experts.gate_up_proj, gate_up_before)
    assert not torch.equal(model.model.layers[0].mlp.experts.down_proj, down_before)


def test_qwen3_moe_gptq_checkpoint_resume_loads_completed_layer(tmp_path):
    loader_path = tmp_path / "calib.pt"
    checkpoint_dir = tmp_path / "checkpoint"
    _write_loader(loader_path)

    model = _tiny_qwen3_moe(num_hidden_layers=1)
    run_gptq(model, _gptq_config(loader_path, checkpoint_dir=str(checkpoint_dir)))
    quantized_gate_up = model.model.layers[0].mlp.experts.gate_up_proj.detach().clone()

    reloaded = _tiny_qwen3_moe(num_hidden_layers=1)
    run_gptq(reloaded, _gptq_config(loader_path, checkpoint_dir=str(checkpoint_dir)))

    assert torch.equal(reloaded.model.layers[0].mlp.experts.gate_up_proj, quantized_gate_up)
