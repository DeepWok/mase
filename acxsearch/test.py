# TODO:
# check the transformer ViT result
import sys
import time
from pathlib import Path
import torch
import argparse
from datetime import datetime
from tqdm import tqdm

from src.chop.models.deit.deit import DistilledVisionTransformer

model = DistilledVisionTransformer(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
)

