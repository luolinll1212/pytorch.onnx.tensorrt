# *_*coding:utf-8 *_*
import tensorrt as trt

class config:
    pretrained = True
    batch_size = 1
    size = 224
    output = "output"
    img_path = "./images/cat.jpg"
    pt = f"./{output}/resnet50.pt" # fp32
    onnx = f"./{output}/resnet50.onnx" # onnx fp32
    engine = f"./{output}/resnet50.engine" # engine fp32
    save_engine = True
    num_classes = 1000

    # tensorrt
    fp16 = False
    int8 = False
    pt_16 = f"./{output}/resnet50-fp16.pt"  # fp16
    onnx_16 = f"./{output}/resnet50-fp16.onnx"  # onnx fp16
    engine_16 = f"./{output}/resnet50-fp16.engine" # engine fp16