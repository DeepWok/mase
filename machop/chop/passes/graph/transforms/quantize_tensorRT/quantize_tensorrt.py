import logging
import torch
import tensorrt as trt

import os
from datetime import datetime as dt
from glob import glob

# import calibrator
import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from cuda import cudart
from torch.autograd import Variable

logger = logging.getLogger(__name__)


class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibrationDataPath, nCalibration, input_generator, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cacheFile = cacheFile
        self.nCalibration = nCalibration
        self.shape = next(iter(self.input_generator)).shape
        self.buffeSize = trt.volume(self.shape) * trt.float32.itemsize
        self.cacheFile = cacheFile
        _, self.dIn = cudart.cudaMalloc(self.buffeSize)
        self.input_generator = input_generator
    
    def get_batch_size(self):
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):
        try:
            data = next(iter(self.input_generator))
            cudart.cudaMemcpy(self.dIn, data.ctypes.data, self.buffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")
        return


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def convert_model_to_onnx(graph, input_generator, onnxFile):
    """
    Convert the graph to ONNX format.

    :param graph: The input graph to be converted.
    :type graph: MaseGraph

    :param dummy_in: A dummy input to the graph.
    :type dummy_in: torch.Tensor

    :return: The ONNX model.
    :rtype: onnx.ModelProto
    """



    # Create a dummy input
    dummy_in = next(iter(input_generator))['x']
    dummy_input = Variable(dummy_in.cuda(), requires_grad=True)

    # Export the model to ONNX
    # torch.onnx.export(graph.model, dummy_input, "model.onnx", verbose=True)
    onnx_model = torch.onnx.export(graph.model.cuda(), dummy_input, onnxFile)

    # Load the ONNX model
    # onnx_model = torch.onnx.load(onnxFile)

    return onnx_model


def build_trt_engine(onnx_model, pass_args=None):
    """
    Build a TensorRT engine from the ONNX model.

    :param onnx_model: The ONNX model to be converted.
    :type onnx_model: onnx.ModelProto

    :param pass_args: A dictionary of arguments for the pass.
    :type pass_args: dict, optional

    :return: The TensorRT engine.
    :rtype: tensorrt.ICudaEngine
    """

    # Set up the builder and network    
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()

    config = builder.create_builder_config()
    if pass_args['precison'] == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    if pass_args['precison'] == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = MyCalibrator(
            pass_args['calibrationDataPath'], 
            pass_args['nCalibration'], 
            pass_args['input_generator'],
            pass_args['cacheFile'], 
            )
        
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(pass_args['onnxFile']):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(pass_args['onnxFile'], "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, 'min', 'opt', 'max')
    config.add_optimization_profile(profile)

    network.unmark_output(network.get_output(0))  # remove output tensor "y"
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(pass_args['engineFile'], "wb") as f:
        f.write(engineString)

    return builder.build_engine(network, config)


def quantize_tensorrt_transform_pass(graph, pass_args=None):
    """
    Quantize the graph using TensorRT.
    
    :param graph: The input graph to be transformed.
    :type graph: MaseGraph

    :param dummy_in: A dummy input to the graph.
    :type dummy_in: torch.Tensor

    :param pass_args: A dictionary of arguments for the pass.
    :type pass_args: dict, optional

    :return: The transformed graph.
    :rtype: MaseGraph
    """

    # Convert model to ONNX
    onnx_model = convert_model_to_onnx(graph, input_generator=pass_args["input_generator"], onnxFile=pass_args["onnxFile"])
    print(onnx_model)

    # Create a TensorRT engine
    engine = build_trt_engine(onnx_model, pass_args)

    return graph, engine