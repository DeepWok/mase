import onnxruntime
from onnxruntime.quantization import CalibrationDataReader


class StaticCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_loader, input_name):
        self.data_loader = data_loader
        self.input_name = input_name
        self.datasize = len(self.data_loader)
        self.enum_data = iter(self.data_loader)

    def to_numpy(self, pt_tensor):
        return (
            pt_tensor.detach().cpu().numpy()
            if pt_tensor.requires_grad
            else pt_tensor.cpu().numpy()
        )

    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None and self.data_loader.batch_size == len(batch[0]):
            return {self.input_name: self.to_numpy(batch[0])}
        else:
            return None

    def rewind(self):
        self.enum_data = iter(self.data_loader)
