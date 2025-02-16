import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *

import logging
import subprocess
from pathlib import Path

from ..utils import add_dataset_info

logger = logging.getLogger(__name__)


def _download_lego_dataset(path: Path) -> None:
    dataset_path = path.joinpath("nerf_synthetic/lego")
    if dataset_path.exists():
        return

    folder_path = path.joinpath("nerf_example_data.zip")
    folder_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the file
    subprocess.run(
        f"wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip --no-check-certificate -O {folder_path.as_posix()}",
        shell=True,
        check=True,
    )

    # Unzip the file
    subprocess.run(
        f"unzip {folder_path.as_posix()} -d {path.as_posix()}",
        shell=True,
        check=True,
    )


class BlenderDatasetBase(Dataset):
    def __init__(self, root_dir, split="train", img_wh=(800, 800)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], "image width must equal image height!"
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = True

    def prepare_data(self):
        """For now, manually download the dataset here:
        http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip

        Extract the zip file.
        Copy nerf_synthetic/lego into .machop/cache/dataset/nerf_syntehtic
        """
        pass

    def setup(self):
        pass

    def read_meta(self):
        # Match split to correct filename
        if self.split == "validation":
            split_fname = "val"
        elif self.split == "pred":
            split_fname = "test"
        else:
            split_fname = self.split

        with open(
            os.path.join(self.root_dir, f"transforms_{split_fname}.json"), "r"
        ) as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = (
            0.5 * 800 / np.tan(0.5 * self.meta["camera_angle_x"])
        )  # original focal length
        # when W=800

        self.focal *= (
            self.img_wh[0] / 800
        )  # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, self.focal)  # (h, w, 3)

        if self.split == "train":  # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for frame in self.meta["frames"]:
                # transform_matrix is a 4x4 matrix
                pose = np.array(frame["transform_matrix"])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

                self.all_rays += [
                    torch.cat(
                        [
                            rays_o,
                            rays_d,
                            self.near * torch.ones_like(rays_o[:, :1]),
                            self.far * torch.ones_like(rays_o[:, :1]),
                        ],
                        1,
                    )
                ]  # (h*w, 8)

            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)

            # Randomly sample some rays to use for training (i.e. don't use all of them to save some time)
            # n_samples = len(self.all_rays) // 16
            # rand_idx = torch.randperm(self.all_rays.size(0))[:n_samples]
            # self.all_rays = self.all_rays[rand_idx]
            # self.all_rgbs = self.all_rgbs[rand_idx]

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == "train":
            return len(self.all_rays)
        if self.split == "validation":
            return 8  # only validate 8 images (to support <=8 gpus)
        return len(self.meta["frames"])

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}

        else:  # create data for each image separately
            frame = self.meta["frames"][idx]
            c2w = torch.FloatTensor(frame["transform_matrix"])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, H, W)
            valid_mask = (img[-1] > 0).flatten()  # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0)  # (H*W, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat(
                [
                    rays_o,
                    rays_d,
                    self.near * torch.ones_like(rays_o[:, :1]),
                    self.far * torch.ones_like(rays_o[:, :1]),
                ],
                1,
            )  # (H*W, 8)

            sample = {"rays": rays, "rgbs": img, "c2w": c2w, "valid_mask": valid_mask}

        return sample


DEFAULT_NERF_CONFIG = {
    "img_wh": [800, 800],
    "N_emb_xyz": 10,
    "N_emb_dir": 4,
    "N_samples": 64,
    "N_importance": 128,
    "use_disp": False,
    "perturb": 1.0,
    "noise_std": 1.0,
    "chunk": 32 * 1024,
    "white_back": True,
}

LEGO_CONFIG = {
    **DEFAULT_NERF_CONFIG,
    "img_wh": [400, 400],
    "noise_std": 0,
    "N_importance": 64,
}


@add_dataset_info(
    name="nerf-lego",
    dataset_source="manual",
    available_splits=("train", "validation", "test"),
    nerf_config=LEGO_CONFIG,
)
class LegoNeRFDataset(BlenderDatasetBase):
    def __init__(self, root_dir, split="train", img_wh=LEGO_CONFIG["img_wh"]) -> None:
        super().__init__(root_dir, split=split, img_wh=img_wh)
