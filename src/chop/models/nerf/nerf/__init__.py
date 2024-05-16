import torch
from torch import nn
import pytorch_lightning as pl
from collections import defaultdict

from .layers import Embedding, NeRF
from .rendering import render_rays


class NeRFModel(pl.LightningModule):
    def __init__(self, info):
        super().__init__()
        self.save_hyperparameters(info.nerf_config)

        self.embedding_xyz = Embedding(self.hparams.N_emb_xyz)
        self.embedding_dir = Embedding(self.hparams.N_emb_dir)
        self.embeddings = {"xyz": self.embedding_xyz, "dir": self.embedding_dir}

        self.nerf_coarse = NeRF(
            in_channels_xyz=6 * self.hparams.N_emb_xyz + 3,
            in_channels_dir=6 * self.hparams.N_emb_dir + 3,
        )
        self.models = {"coarse": self.nerf_coarse}
        # load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if self.hparams.N_importance > 0:
            self.nerf_fine = NeRF(
                in_channels_xyz=6 * self.hparams.N_emb_xyz + 3,
                in_channels_dir=6 * self.hparams.N_emb_dir + 3,
            )
            self.models["fine"] = self.nerf_fine
            # load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = render_rays(
                self.models,
                self.embeddings,
                rays[i : i + self.hparams.chunk],
                self.hparams.N_samples,
                self.hparams.use_disp,
                self.hparams.perturb,
                self.hparams.noise_std,
                self.hparams.N_importance,
                self.hparams.chunk,  # chunk size is effective in val mode
                self.hparams.white_back,
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results
