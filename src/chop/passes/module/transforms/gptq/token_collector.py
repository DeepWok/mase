"""Token-level activation collector for GPTQ calibration.

Attach to any HuggingFace causal LM via ``TokenCollector(model, ...).attach()``.
During any forward pass driven by an arbitrary eval framework (lm-eval-harness,
BFCL, evalplus, custom scripts, ``model.generate``, ...), the hook captures the
prefill ``input_ids``, buffers the token stream, slices it into fixed-length
chunks, and saves a calibration dataloader in the exact format produced by
``data_utils.get_wikitext2`` — a list of ``(input_ids, target)`` tuples.

This sidesteps the complications of hooking layer-0 hidden states (zero-padding
distortion, attention-mask shape mismatches, hard dependency on
``model.model.layers[0]``) by capturing the raw token stream and letting the
standard GPTQ Catcher path compute everything else.

Usage::

    from chop.passes.module.transforms.gptq import TokenCollector

    collector = TokenCollector(
        model,
        target_nsamples=128,
        seqlen=2048,
        save_path="calib/bfcl.pt",
    ).attach()
    run_my_eval(model, tokenizer)        # any eval framework
    collector.finalize()                 # flush partial buffer (no-op if full)

Then for GPTQ::

    gptq_config = {..., "dataset": "file:calib/bfcl.pt", "nsamples": 128, ...}
    run_gptq(model, gptq_config)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class CollectorFull(Exception):
    """Raised by ``TokenCollector`` (when ``raise_on_full=True``) right after
    the buffer is saved, so the surrounding driver — typically a lm-eval pass
    used purely to feed forwards — can abort cleanly instead of running more
    forwards we won't use."""


class TokenCollector:
    """Capture an input_ids token stream from arbitrary forward passes."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_nsamples: int,
        seqlen: int,
        save_path: str | Path,
        overwrite: bool = False,
        min_prefill_tokens: int = 8,
        raise_on_full: bool = False,
    ):
        self.model = model
        self.target_nsamples = int(target_nsamples)
        self.seqlen = int(seqlen)
        self.save_path = Path(save_path)
        self.overwrite = bool(overwrite)
        self.min_prefill_tokens = int(min_prefill_tokens)
        self.raise_on_full = bool(raise_on_full)
        self._buf: list[torch.Tensor] = []
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._saved = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach(self) -> "TokenCollector":
        if self.save_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"calibration file {self.save_path} already exists; "
                "set overwrite=True to replace."
            )
        self._handle = self.model.register_forward_pre_hook(
            self._hook, with_kwargs=True
        )
        self.model._mase_token_collector = self
        logger.info(
            "[TokenCollector] attached: target_nsamples=%d seqlen=%d save_path=%s",
            self.target_nsamples, self.seqlen, self.save_path,
        )
        return self

    @property
    def total_tokens(self) -> int:
        return sum(int(t.numel()) for t in self._buf)

    @property
    def complete(self) -> bool:
        return self._saved

    def finalize(self) -> None:
        """Flush partial buffer to disk (or noop if already saved/empty)."""
        if self._saved:
            return
        target = self.target_nsamples * self.seqlen
        if self.total_tokens >= self.seqlen:
            logger.warning(
                "[TokenCollector] finalizing with %d/%d tokens (incomplete).",
                self.total_tokens, target,
            )
            self._save_and_detach()
        else:
            logger.warning(
                "[TokenCollector] insufficient tokens (%d < seqlen=%d); nothing saved.",
                self.total_tokens, self.seqlen,
            )
            self._detach()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hook(self, _module, args, kwargs) -> None:
        if self._saved:
            return
        ids = kwargs.get("input_ids", None)
        if ids is None and args:
            cand = args[0]
            if torch.is_tensor(cand) and cand.dtype in (
                torch.int32, torch.int64, torch.long,
            ):
                ids = cand
        if ids is None or not torch.is_tensor(ids):
            return
        # Skip decode steps and other degenerate inputs.
        if ids.shape[-1] < self.min_prefill_tokens:
            return
        # Flatten across batch — token order within a sample is preserved;
        # cross-sample boundaries are absorbed by the seqlen-sized slicing.
        self._buf.append(ids.detach().to(torch.long).flatten().cpu())
        if self.total_tokens >= self.target_nsamples * self.seqlen:
            self._save_and_detach()

    def _save_and_detach(self) -> None:
        if self._saved:
            return
        big = torch.cat(self._buf) if self._buf else torch.empty(0, dtype=torch.long)
        usable = int(big.numel()) // self.seqlen
        n = min(usable, self.target_nsamples)
        if n <= 0:
            logger.warning(
                "[TokenCollector] no full chunks of seqlen=%d collected.", self.seqlen,
            )
            self._detach()
            return
        loader = []
        for i in range(n):
            inp = big[i * self.seqlen : (i + 1) * self.seqlen].unsqueeze(0).clone()
            tar = inp.clone()
            tar[:, :-1] = -100
            loader.append((inp, tar))
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "loader": loader,
                "seqlen": self.seqlen,
                "target_nsamples": self.target_nsamples,
                "collected_samples": n,
                "format_version": 1,
            },
            str(self.save_path),
        )
        self._saved = True
        self._detach()
        logger.info(
            "[TokenCollector] saved %d/%d samples (%d tokens) to %s",
            n, self.target_nsamples, int(big.numel()), self.save_path,
        )
        if self.raise_on_full:
            raise CollectorFull(
                f"buffer full: saved {n}/{self.target_nsamples} samples to {self.save_path}"
            )

    def _detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
