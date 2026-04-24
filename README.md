# Driving Warning System（当前目录版）

本文档基于 **当前仓库实际文件** 整理（你已精简目录后的版本）。

## 1. 当前项目入口

当前主要入口脚本：

- `demo.py`：PyQt5 图形界面主程序（前视主画面 + 左右子画面 + 日志/建议区）

运行命令：

```powershell
python demo.py
```

## 2. 运行前先看这 4 点

首次运行前先改：

1. `demo.py` 中默认目标检测模型是 `./ObjectDetector/models/yolo11.onnx`
改成已存在模型，例如：
- `./ObjectDetector/models/BDG-YOLO.onnx`
- `./ObjectDetector/models/yolov8n.onnx`
- `./ObjectDetector/models/yolov9n.onnx`
本研究模型权重weights.zip下载链接: https://pan.baidu.com/s/1VXVJOuY47d9JPNQfSyC3wQ?pwd=6666

2. `demo.py` 中默认左/右视频为 `./left.mp4` 与 `./right.mp4`
改为你已有视频路径
本研究视频集Video.zip下载链接: https://pan.baidu.com/s/1kJD_WJFHN54-YzwG1aqKHg 提取码: 6666 

3. `QW25/test.py` 默认模型目录：
- `./QW25/models/Qwen/Qwen2___5-0___5B-Instruct`

当前目录不存在该模型，会导致启动失败。你可以：
-cd QW25
-python download.py
- 下载后自动生成 models/Qwen/Qwen2.5-0.5B-Instruct

## 3. 环境配置（Windows / PowerShell）

建议 Python 版本：`3.10`

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. 命令行工具脚本

项目仍保留模型转换/量化脚本：

1. ONNX -> TensorRT

```powershell
python convertOnnxToTensorRT.py -i <input.onnx> -o <output.trt>
```

2. ONNX -> TensorRT（TensorRT 10）

```powershell
python convertOnnxToTensorRT10.py -i <input.onnx> -o <output.trt>
```

3. ONNX FP16 量化

```powershell
python onnxQuantization.py -i <input.onnx>
```

输出文件会在同目录生成 `<原文件名>_fp16.onnx`。


