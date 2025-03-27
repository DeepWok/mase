from chop.models.yolo import get_yolo_detection_model, get_yolo_segmentation_model
import torch
from chop import MaseGraph
import chop.passes as passes
from chop.dataset import MaseDataModule
import toml
import os


HERE = os.path.dirname(os.path.abspath(__file__))
TOML_PATH = f"{HERE}/yolov8_INT8_quantization_by_type.toml"
# Reading TOML file and converting it into a Python dictionary
with open(TOML_PATH, "r") as toml_file:
    pass_args = toml.load(toml_file)

model = get_yolo_segmentation_model("yolov8n-seg.pt")
model.train()
mg = MaseGraph(model)

param = next(model.model.parameters())[1]
dummy_input = torch.rand(1, 3, 640, 640, dtype=param.dtype).to(param.device)

dataset_name = "coco-segmentation"
model_name = "yolov8"
batch_size = pass_args["batch_size"]

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

dl = data_module.train_dataloader()

bboxes = mg.model(dl.dataset[0]["img"].unsqueeze(0))

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(
    mg,
    {
        "dummy_in": {
            "x": dummy_input,
        },
        "add_value": True,
    },
)
mg, _ = passes.add_software_metadata_analysis_pass(mg, None)

mg, _ = passes.graph.runtime_analysis_pass(mg, pass_args={
    "dataset_input_field": "img",
    "data_module": data_module,
    **pass_args
})
