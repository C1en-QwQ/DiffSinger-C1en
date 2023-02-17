# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [Interactive🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger)

## DiffSinger (ONNX Deployment)

Currently, we only support exporting MIDI-less acoustic model to ONNX format.

### 0. Environment Preparation

Run with the command to install extra requirements for exporting the model to ONNX format.

```bash
pip install onnx==1.12.0 onnxsim==0.4.10 protobuf==3.13.0  # Used for graph repairing and optimization
```

The `onnxruntime` package is required to run inference with ONNX model and ONNXRuntime. See the [official guidance](https://onnxruntime.ai/) for instructions to install packages matching your hardware. CUDA, DirectML and default CPU are recommended since the model has been tested on these execution providers.

Note that the scripts are tested on PyTorch 1.8.

### 1. Export to ONNX format

Run with the command

```bash
python onnx/export/export_acoustic.py --exp EXP [--out OUT]
```

where `EXP` is the name of experiment, `OUT` is the output directory.

This script will export the acoustic model to the ONNX format and do a lot of optimization (25% ~ 50% faster with ONNXRuntime than PyTorch).

Note: DPM-Solver acceleration is not currently included, but PNDM is wrapped into the model. Use any `speedup` larger than 1 to enable it.

### 2. Inference with ONNXRuntime

See `onnx/infer/infer_acoustic.py` for details.

#### Issues related to CUDAExecutionProvider

In some cases, especially when you are using virtual environment, you may get the following error when creating a session with CUDAExecutionProvider, even if you already installed CUDA and cuDNN on your system:

```text
RuntimeError: D:\a\_work\1\s\onnxruntime\python\onnxruntime_pybind_state.cc:574 onnxruntime::python::CreateExecutionProviderInstance CUDA_PATH is set but CUDA wasn't able to be loaded. Please install the co
rrect version of CUDA and cuDNN as mentioned in the GPU requirements page (https://onnxruntime.ai/docs/reference/execution-providers/CUDA-ExecutionProvider.html#requirements), make sure they're in the PATH,
 and that your GPU is supported.
```

There are two ways to solve this problem.

1. Simply import PyTorch and leave it unused before you create the session:

```python
import torch
```

This seems stupid but if your PyTorch is built with CUDA, then CUDAExecutionProvider will just work.

2. When importing PyTorch, its `__init__.py` actually adds CUDA and cuDNN to the system DLL path. This can be done manually, with the following line before creating the session:

```python
import os
os.add_dll_directory(r'path/to/your/cuda/dlls')
os.add_dll_directory(r'path/to/your/cudnn/dlls')
```

See [official requirements](http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) for all DLLs that should be included in the paths above.

In this way you can also switch between your system CUDA and PyTorch CUDA in your virtual environment.
