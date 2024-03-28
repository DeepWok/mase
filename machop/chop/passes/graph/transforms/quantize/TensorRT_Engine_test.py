import subprocess
command = "trtexec --onnx=resnet50/model.onnx --saveEngine=resnet_engine.trt"
process = subprocess.Popen(command, shell=True)
output, error = process.communicate()