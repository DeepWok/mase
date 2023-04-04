import sys

import colorlog
import torch

sys.path.append("../../")
sys.path.append("../../../software")

from logger import getLogger

logger = getLogger("Test block and unblock 3D act")

from machop.dataset import MyDataModule, get_dataset_info
from machop.models import get_resnet18
from machop.modify.modifier import Modifier

data_module = MyDataModule(
    "cifar10", batch_size=2, workers=4, tokenizer=None, max_token_len=1
)

data_module.prepare_data()
data_module.setup()

info = get_dataset_info("cifar10")
reset18 = get_resnet18(info)

m = Modifier(reset18, config="../../../software/configs/modify-sw/msfp.toml")

gm = m.graph_module

bx, _ = next(iter(data_module.train_dataloader()))
pred_y = gm(bx)
breakpoint()
