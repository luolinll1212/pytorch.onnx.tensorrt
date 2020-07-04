# *_*coding:utf-8 *_*
import torch
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import os

from config import config as cfg

if not os.path.exists("output"):
    os.mkdir("output")

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input = torch.randn((1, 3, cfg.size, cfg.size)).float().to(device)

# load model
model = models.resnet50(pretrained=cfg.pretrained).eval().to(device)
# save model
if not cfg.fp16:
    torch.save(model.state_dict(), cfg.pt) # fp32
    torch.onnx.export(model, input, cfg.onnx, verbose=True, export_params=True, opset_version=9)
else:
    from src.fp16utils import network_to_half
    model = network_to_half(model)
    torch.save(model.state_dict(), cfg.pt_16) # fp16
    torch.onnx.export(model, input, cfg.onnx_16, verbose=True, export_params=True, opset_version=9)

