import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import os, time
from PIL import Image
from torchvision import transforms, models

from config import config as cfg

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice" + str(self.device)
    def __repr__(self):
        return self.__str__()

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
def get_engine(max_batch_size=1, onnx_file_path=cfg.onnx, engine_file_path=cfg.engine,
               fp16_mode=False, int8_mode=False, save_engine=False):
    def build_engine(max_batch_size, save_engine):
        with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 30
            builder.max_batch_size = max_batch_size
            # pdb.set_trace()
            builder.fp16_mode = fp16_mode
            builder.int8_mode = int8_mode

            print("Loading ONNX file from path {}".format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print("Begining ONNX file parsing")
                parser.parse(model.read())

            engine = builder.build_cuda_engine(network)
            if engine is None:
                print("Failed to create the engine")
                return None
            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as file:
                    file.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

# read image
image = Image.open(cfg.img_path)
transform = transforms.Compose([
    transforms.CenterCrop((cfg.size, cfg.size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = transform(image)
image = image.unsqueeze(0)
x_input = np.array(image).astype(dtype=np.float32)

"""
fp32 inference
"""
print("*"*20)
# create engine
engine = get_engine(cfg.batch_size, cfg.onnx, cfg.engine, cfg.fp16, cfg.int8, cfg.save_engine)
# Create the context for this engine
context = engine.create_execution_context()
# Allocate buffers for input and output
inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

# tensorrt inference
shape_of_output = (cfg.batch_size, cfg.num_classes)
inputs[0].host = x_input.reshape(-1)
t1 = time.time()
trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
# numpy data
t2 = time.time()
trt_out = postprocess_the_outputs(trt_outputs[0], shape_of_output)
print(trt_out.argmax(1))

# pytorch inference
x_input_pth = image.cuda()
t_model = models.resnet50(pretrained=False)
t_model.load_state_dict(torch.load(cfg.pt))
t_model.eval()
t_model.cuda()
t_model.export_to_onnx_mode = False
t3 = time.time()
t_out = t_model(x_input_pth)
t4 = time.time()
t_out = t_out.cpu().data.numpy()
print(t_out.argmax(1))

mse = np.mean((trt_out - t_out) ** 2)
print("TensorRT fp32 engine of Inference time:{}".format(t2-t1))
print("Pytorch fp32 model of Inference time:{}".format(t4-t3))
print("MSE ERROR:{}".format(mse))

# """
# fp16 inference
# """
print("*"*20)
cfg.fp16 = True
cfg.int8 = False
# create engine
engine = get_engine(cfg.batch_size, cfg.onnx_16, cfg.engine_16, cfg.fp16, cfg.int8, cfg.save_engine)
# Create the context for this engine
context = engine.create_execution_context()
# Allocate buffers for input and output
inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

# tensorrt fp16 inference
shape_of_output = (cfg.batch_size, cfg.num_classes)
inputs[0].host = x_input.reshape(-1)
t1 = time.time()
trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
# numpy data
t2 = time.time()
trt_out = postprocess_the_outputs(trt_outputs[0], shape_of_output)
print(trt_out.argmax(1))

# pytorch fp16 inference
x_input_pth = image.half().cuda()
t_model = models.resnet50(pretrained=False)
from src.fp16utils import network_to_half
t_model = network_to_half(t_model)
t_model.load_state_dict(torch.load(cfg.pt_16)) # fp16 model
t_model.eval().cuda()
t_model.export_to_onnx_mode = False
t3 = time.time()
t_out = t_model(x_input_pth)
t4 = time.time()
t_out = t_out.cpu().data.numpy()
print(t_out.argmax(1))

mse = np.mean((trt_out - t_out) ** 2)
print("TensorRT fp16 engine of Inference time:{}".format(t2-t1))
print("Pytorch fp16 model of Inference time:{}".format(t4-t3))
print("MSE ERROR:{}".format(mse))


