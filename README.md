# pytorch.onnx.tensorrt
## pytorch->onnx->tensorrt

### resnet50

#### run
```python
python pytorch_to_onnx.py # config.fp16=False|True
python onnx_to_tensorrt.py
```

#### result
```python

**********
Loading ONNX file from path ./output/resnet50.onnx
Begining ONNX file parsing
Completed creating Engine
[281]
[281]
TensorRT fp32 engine of Inference time:0.0023183822631835938
Pytorch fp32 model of Inference time:0.1390986442565918
MSE ERROR:1.1674825602797645e-12
**********
Loading ONNX file from path ./output/resnet50-fp16.onnx
Begining ONNX file parsing
Completed creating Engine
[281]
[281]
TensorRT fp16 engine of Inference time:0.0013973712921142578
Pytorch fp16 model of Inference time:0.006196022033691406
MSE ERROR:3.844499588012695e-05
```
