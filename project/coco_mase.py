# %%
from chop.dataset import MaseDataModule, get_dataset_info

dataset_name = "coco-segmentation"
# dataset_name = "coco-segmentation"
model_name = "yolov8"
batch_size = 5

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()


# %%
dl = data_module.train_dataloader()
dl.dataset[0]

# %%
