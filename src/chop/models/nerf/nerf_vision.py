from torch._tensor import Tensor


from torch import Tensor
import torch.nn as nn
import torch
from typing import Any
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
from chop.models.utils import register_mase_model, register_mase_checkpoint

from .rendering import post_render_vision, pre_render_vision


# Model# Model
@register_mase_model(
    "nerfvision",
    checkpoints=["nerfvision"],
    model_source="nerfvision",
    task_type="nerfvision",
    physical_data_point_classification=True,
    is_fx_traceable=True,
)
class NeRFVision(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        use_viewdirs=True,
    ):
        """
        This is the Nerf model from the Nerf Paper
        Title: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

        Reference:
        https://arxiv.org/abs/2003.08934

        D: Number of layers
        W: Width of layers
        input_ch: Number of input channels
        input_ch_views: Number of input channels for viewdirs
        output_ch: Number of output channels
        skips: Which layers to skip
        use_viewdirs: Whether to use viewdirs
        """
        super(NeRFVision, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, pts, viewdirs):
        raw: Tensor | Any = self.apply_layers(pts, viewdirs)
        return raw

    def apply_layers(self, input_pts, input_views):
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # Add a new dimension and expand it
            expanded_input_views = input_views.unsqueeze(1).expand(
                -1, h.shape[1], -1
            )  # -1 means it will retain the size of that dimension
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, expanded_input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears])
            )
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1])
            )

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear])
        )
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1])
        )

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears])
        )
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1])
        )

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear])
        )
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear + 1])
        )

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear])
        )
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear + 1])
        )


# Getters ------------------------------------------------------------------------------
def get_nerf(
    pretrained=False,
    **kwargs: Any,
):
    return NeRFVision()


@register_mase_checkpoint("nerfvision")
def get_nerfvision(
    pretrained: bool = False,
    **kwargs: Any,
) -> NeRFVision:
    model = get_nerf(
        pretrained=pretrained,
        **kwargs,
    )
    if pretrained:
        weights = np.load(
            "/teamspace/studios/this_studio/mase-team-coursework/mase/nerf_vision/lego_example/model_200000.npy",
            allow_pickle=True,
        )
        model.set_weights(weights)
    else:
        pretrained_weight_cls = None

    return model
