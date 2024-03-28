import logging
import torch
import tensorrt as trt

import os
from datetime import datetime as dt
from glob import glob

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from cuda import cudart
from torch.autograd import Variable
import onnx
import time

import pycuda.driver as cuda
import pycuda.autoinit

logger = logging.getLogger(__name__)


class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, nCalibration, input_generator, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cacheFile = cacheFile
        self.nCalibration = nCalibration
        self.shape = next(iter(input_generator))["x"].shape
        self.buffeSize = trt.volume(self.shape) * trt.float32.itemsize
        self.cacheFile = cacheFile
        _, self.dIn = cudart.cudaMalloc(self.buffeSize)
        self.input_generator = input_generator

    def get_batch_size(self):
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):
        try:
            data = np.array(next(iter(self.input_generator))["x"])
            cudart.cudaMemcpy(
                self.dIn,
                data.ctypes.data,
                self.buffeSize,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
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


def convert_model_to_onnx(graph, dummy_in, onnxFile):
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
    # dummy_in = next(iter(input_generator))['x']
    # dummy_input = Variable(dummy_in.cuda(), requires_grad=True)

    # Export the model to ONNX
    # torch.onnx.export(graph.model, dummy_input, "model.onnx", verbose=True)
    # torch.onnx.export(graph.model.cuda(), dummy_input, onnxFile, verbose=True)
    if isinstance(dummy_in, dict):
        dummy_in = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in dummy_in.items()
        }
    else:
        dummy_in = dummy_in.cuda()

    torch.onnx.export(
        graph.model.cuda(),
        dummy_in,
        onnxFile,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Load the ONNX model
    # onnx_model = onnx.load(onnxFile)


def build_trt_engine(pass_args=None):
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
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    profile = builder.create_optimization_profile()

    config = builder.create_builder_config()
    if pass_args["precision"] == "fp16" or pass_args["precision"] == "best":
        config.set_flag(trt.BuilderFlag.FP16)
    if pass_args["precision"] == "int8" or pass_args["precision"] == "best":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = MyCalibrator(
            pass_args["nCalibration"],
            pass_args["input_generator"],
            pass_args["cacheFile"],
        )

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(pass_args["onnxFile"]):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(pass_args["onnxFile"], "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    inputTensor = network.get_input(0)
    profile.set_shape(
        inputTensor.name,
        (1,) + inputTensor.shape[1:],
        (8,) + inputTensor.shape[1:],
        (32,) + inputTensor.shape[1:],
    )
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(pass_args["engineFile"], "wb") as f:
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
    convert_model_to_onnx(
        graph, dummy_in=pass_args["dummy_in"], onnxFile=pass_args["onnxFile"]
    )

    # Create a TensorRT engine
    engine = build_trt_engine(pass_args)
    return graph, engine


def test_quantize_tensorrt_transform_pass(dataloader, engineFile):
    """
    Test the quantize_tensorrt_transform_pass function.

    :param pass_args: A dictionary of arguments for the pass.
    :type pass_args: dict
    """

    # Load engineString from file
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engineFile, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        print("engine.__len__() = %d" % len(engine))
        print("engine.__sizeof__() = %d" % engine.__sizeof__())
        print("engine.__str__() = %s" % engine.__str__())

        print(
            "\nEngine related ========================================================"
        )

    inspector = engine.create_engine_inspector()
    print("inspector.execution_context=", inspector.execution_context)
    print(
        "inspector.error_recorder=", inspector.error_recorder
    )  # ErrorRecorder can be set into EngineInspector, usage of ErrorRecorder refer to 02-API/ErrorRecorder

    print(
        "Engine information:"
    )  # engine information is equivalent to put all layer information together
    print(
        inspector.get_engine_information(trt.LayerInformationFormat.ONELINE)
    )  # .txt format
    # print(inspector.get_engine_information(trt.LayerInformationFormat.JSON))  # .json format

    print("Layer information:")
    for i in range(engine.num_layers):
        print(inspector.get_layer_information(i, trt.LayerInformationFormat.ONELINE))

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(
        trt.TensorIOMode.INPUT
    )

    context = engine.create_execution_context()

    dataiter = iter(dataloader())
    input, labels = next(dataiter)
    input_shape = input.shape
    context.set_input_shape(lTensorName[0], input_shape)
    for i in range(nIO):
        print(
            "[%2d]%s->" % (i, "Input " if i < nInput else "Output"),
            engine.get_tensor_dtype(lTensorName[i]),
            engine.get_tensor_shape(lTensorName[i]),
            context.get_tensor_shape(lTensorName[i]),
            lTensorName[i],
        )

    execute_time = []
    accuracy = []
    start_event = cuda.Event()
    end_event = cuda.Event()
    for data, label in dataloader():
        bufferH = []
        bufferH.append(np.ascontiguousarray(data))
        for i in range(nInput, nIO):
            bufferH.append(
                np.empty(
                    context.get_tensor_shape(lTensorName[i]),
                    dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i])),
                )
            )
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput):
            cudart.cudaMemcpy(
                bufferD[i],
                bufferH[i].ctypes.data,
                bufferH[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )

        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))

        # start_time = time.time()
        start_event.record()
        context.execute_async_v3(0)
        # execute_time.append(time.time() - start_time)

        end_event.record()
        end_event.synchronize()
        execute_time.append(start_event.time_till(end_event))

        for i in range(nInput, nIO):
            cudart.cudaMemcpy(
                bufferH[i].ctypes.data,
                bufferD[i],
                bufferH[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )

            categories = np.argmax(bufferH[nInput], axis=1)
            # print(categories, label)
            acc = np.sum(categories == np.array(label)) / len(label)
            # print("Accuracy: %.2f%%" % (acc * 100))
            accuracy.append(acc)

        # for i in range(nIO):
        #     print(lTensorName[i])
        #     print(bufferH[i])
        #     print(categories, label)

        for b in bufferD:
            cudart.cudaFree(b)
    print("Succeeded running model in TensorRT!")
    print(
        "Average execute time for one batch: %.2fms"
        % (sum(execute_time) / len(execute_time))
    )
    print("Total accuracy: %.2f%%" % (sum(accuracy) / len(accuracy) * 100))
