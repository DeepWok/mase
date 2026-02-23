"""
GPTQ quantization algorithm.

Ported from Coprocessor_for_Llama/acc_simulator/gptq/gptq.py,
adapted to use Mase config dicts instead of Meta classes.
"""

import math

import torch
import tqdm

from .quantize_dispatch import quantize_tensor
from .utils import cleanup_memory


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, activation, fmt, weight_config, percdamp=.01,
        cali_batch_size=32, layer_name=None, quant_search=True,
    ):
        """
        Run GPTQ block-wise quantization.

        Args:
            activation: Activation tensor for calibration (or None).
            fmt: "mxfp" or "mxint".
            weight_config: Mase-style config dict with weight_block_size, etc.
            percdamp: Dampening percentage for Hessian diagonal.
            cali_batch_size: Batch size for quantile search calibration.
            layer_name: Name for progress bar.
            quant_search: Enable quantile search.
        """
        W = self.layer.weight.data.clone()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        blocksize = weight_config["weight_block_size"]
        for i1 in tqdm.tqdm(
            range(0, self.columns, blocksize),
            desc=f"Quantizing blocks {layer_name}",
            disable=False,
        ):
            i2 = min(i1 + blocksize, self.columns)

            W1 = W[:, i1:i2].clone()

            if activation is not None:
                Act1 = activation[:, :, i1:i2].clone()
                Q1 = quantize_tensor(
                    W1, block_dim=1, fmt=fmt, config=weight_config,
                    quantile_search=quant_search, act_tensor=Act1,
                    cali_batch_size=cali_batch_size,
                )
            else:
                Q1 = quantize_tensor(
                    W1, block_dim=1, fmt=fmt, config=weight_config,
                    quantile_search=quant_search,
                )

            Hinv1 = Hinv[i1:i2, i1:i2]
            Err1 = (W1 - Q1) / torch.diag(Hinv1).unsqueeze(0)
            Losses1 = ((W1 - Q1) ** 2) / (torch.diag(Hinv1) ** 2).unsqueeze(0)

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        assert Q.shape == W.shape, f"Shape mismatch: {Q.shape} != {W.shape}"

        return Q

    def free(self):
        self.H = None
        self.Losses = None
        torch.cuda.empty_cache()
        cleanup_memory(verbos=False)
