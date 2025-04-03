from torch import nn
import torch
from torchmetrics.aggregation import MeanMetric

from chop.tools.plt_wrapper.nerf.metrics import psnr


class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="mean")

    def forward(self, inputs, targets):
        output = post_render_vision(targets, inputs)
        loss = self.loss(output["rgb"], targets["rgbs"])

        return self.coef * loss


class NerfPsnr(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, inputs, targets) -> None:
        output = post_render_vision(targets, inputs)
        preds, target = output["rgb"], targets["rgbs"]
        psnr_ = psnr(preds, target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        super().update(psnr_)


def post_render_vision(
    x,
    raw,
    noise_std=1,
):
    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """

        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=torch.nn.functional.relu):
            return 1.0 - torch.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = torch.concat(
            [
                dists,
                torch.ones_like(rays_d[:, :1]).to("cuda") * 1e10,
            ],  # ISSUE expand to [N_rays, 1]
            dim=-1,
        )  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # Extract RGB of each sample position along each ray.
        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.0
        if noise_std > 0.0:
            noise = torch.rand_like(raw[..., 3]) * noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=-1)

        # Computed weighted color of each sample along each ray.
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        depth_map = torch.sum(weights * z_vals, dim=-1)

        # Disparity map is inverse depth.
        disp_map = 1.0 / torch.maximum(
            torch.tensor([1e-10]).to(dists.device),
            depth_map / torch.sum(weights, dim=-1),
        )

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = torch.sum(weights, dim=-1)

        return rgb_map, disp_map, acc_map, weights, depth_map

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, x["z_vals"], x["rays_d"]
    )

    results = {}

    results["rgb"] = rgb_map
    results["depth"] = depth_map
    results["weights"] = weights
    results["opacity"] = acc_map
    results["z_vals"] = x["z_vals"]
    results["disp"] = disp_map

    return results


loss_dict = {"color": ColorLoss}
