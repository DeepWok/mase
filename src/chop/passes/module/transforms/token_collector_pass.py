"""Module-level transform: attach a ``TokenCollector`` hook to a model.

Used as a side-effect pass before any eval / forward driver. Once the hook
buffer is full it auto-saves the calibration loader to disk; if
``raise_on_full=true`` is set, it also raises ``CollectorFull`` so the
surrounding driver (typically lm-eval-harness) can abort cleanly instead
of running more forwards we won't use.

Example pass_args block (top-level ``token_collector`` key in a TOML
quant config; ``eval_lm.py`` consumes it before quantization)::

    [token_collector]
    target_nsamples    = 32
    seqlen             = 1024
    save_path          = "calib/qwen3_0.6b_gsm8k.pt"
    overwrite          = false
    min_prefill_tokens = 8
    raise_on_full      = true
"""

from __future__ import annotations

from .gptq import TokenCollector


def attach_token_collector_pass(network, config: dict):
    """Attach a ``TokenCollector`` to ``network`` and return both.

    Args:
        network: HuggingFace causal-LM whose ``forward(input_ids, ...)`` will
                 be hooked.
        config:  Dict with TokenCollector kwargs (see module docstring).

    Returns:
        (network, info) where ``info["collector"]`` is the attached
        ``TokenCollector`` instance — the caller can read
        ``collector.complete`` afterwards or call ``finalize()`` to flush
        a partial buffer.
    """
    collector = TokenCollector(
        model=network,
        target_nsamples=int(config["target_nsamples"]),
        seqlen=int(config["seqlen"]),
        save_path=str(config["save_path"]),
        overwrite=bool(config.get("overwrite", False)),
        min_prefill_tokens=int(config.get("min_prefill_tokens", 8)),
        raise_on_full=bool(config.get("raise_on_full", False)),
    ).attach()
    return network, {"collector": collector}
