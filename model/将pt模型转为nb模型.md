
# 将pt文件转为.nb文件的过程
## 准备工作
首先要提前下载好yolov5的源码
```shell
git clone https://github.com/ultralytics/yolov5.git
```
onnxsim这个工具可以用来修改onnx模型
```shell
sudo pip3  install onnxsim
```
netron 软件可以查看模型结果，去以下链接下载
```
https://github.com/lutzroeder/netron/releases
```
或者使用网页版的
```
https://netron.app/
```

安装核桃派提供的模型转换工具

## 1.1 将yolo5的.pt模型文件转为.onnx格式
做格式转换需要下载yolo5的源码，调用源码内部的py脚本。

为了加快运行速度，这里对yolo5的源码进行一点修改，目的是让模型在计算完成后直接把结果输出，把一些后处理代码交由自己的程序来跑，可以更快

根据经验，对yolo源码进行修改,yolo.py里面的Detect类里面的forward函数。这里我把该函数的返回值改为返回x。即后处理部分什么都不做，直接输出模型的检测结果，不要对那个庞大的数据做运算。

```python
       # return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
        return x 
```
因为修改forward函数会导致export.py文件报错，所以修改这个文件，将run函数里面报错的两句给注释掉
```python
    # shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    metadata = {"stride": int(max(model.stride)), "names": model.names}  # model metadata
    # LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

```


使用yolov5源码带的export.py来转格式

```shell
python3 /opt/yolov5/export.py --weights yolov5n.pt --data /opt/yolov5/data/coco.yaml --include onnx
python3 /opt/yolov5/export.py --weights yolov5s.pt --data /opt/yolov5/data/coco.yaml --include onnx
python3 /opt/yolov5/export.py --weights yolov5m.pt --data /opt/yolov5/data/coco.yaml --include onnx
python3 /opt/yolov5/export.py --weights yolov5l.pt --data /opt/yolov5/data/coco.yaml --include onnx
python3 /opt/yolov5/export.py --weights yolov5x.pt --data /opt/yolov5/data/coco.yaml --include onnx
```
## 1.2 将yolo8的.pt模型文件转为.onnx格式
首先安装ultralytics包，他会安装一个命令行工具
```shell
sudo pip3 install ultralytics
```
```
yolo export model=yolov8n.pt format=onnx
```

## 2. 固定模型的输入尺寸
t527的npu不支持动态尺寸输入，需要模型设置输入尺寸为固定值
使用netron打开onnx模型文件，查看输入节点是否有限制输入尺寸，如果没有限制，则需要手动固定输入尺寸，使用以下命令。


```shell
python3 -m onnxsim yolov5s.onnx yolov5s.onnx --overwrite-input-shape 1,3,640,640
```

## 3. 导出onnx模型的数据
使用核桃派提供的工具将onnx模型导出为如下文件,后面需要使用这些文件进行量化
- .json 网络结构文件
- .data 网络权重文件
- _inputmeta.yml 输入描述文件，可以修改来调用npu的预处理功能
- _postprocess_file.yml 输出描述文件，可以修改npu的输出

运行命令后会在当前文件夹下生成 模型名称-data文件夹，里面会存放npu-model-export工具生成的文件
```shell
sudo npu-model-export yolov5n.onnx
sudo npu-model-export yolov5s.onnx
sudo npu-model-export yolov5m.onnx
sudo npu-model-export yolov5l.onnx
sudo npu-model-export yolov5x.onnx
```
这里修改 _inputmeta.yml 文件,配置启用npu自带的前处理功能。模型需要的是浮点数，但图像的像素值是0-255的整数，我们把整数转浮点这一步交给npu做。将其中的scale 改为 0.00392157 ，npu就会对传入的数值全部除以255，转为模型需要的浮点数。节省cpu的负担。
  
## 4. 量化并生成生成.nb模型文件
```shell
sudo npu-model-generate yolov5n-data/ ../../../image/
sudo npu-model-generate yolov5s-data/ ../../../image/
sudo npu-model-generate yolov5m-data/ ../../../image/
sudo npu-model-generate yolov5l-data/ ../../../image/
sudo npu-model-generate yolov5x-data/ ../../../image/
```
