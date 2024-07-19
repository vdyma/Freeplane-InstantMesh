<div align="center">
  
# *Freeplane* InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models and *Frequency Modulated Triplane*

</div>
<div align="center">

The following badges are mostly from the original README, except for the second Arxiv paper badge which points to the paper: "*Freeplane: Unlocking Free Lunch in Triplane-Based Sparse-View Reconstruction Models*".

<a href="https://arxiv.org/abs/2404.07191"><img src="https://img.shields.io/badge/ArXiv-2404.07191-brightgreen"></a> 
<a href="https://arxiv.org/abs/2406.00750"><img src="https://img.shields.io/badge/ArXiv-2406.00750-brightgreen"></a> 
<a href="https://huggingface.co/TencentARC/InstantMesh"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a> 
<a href="https://huggingface.co/spaces/TencentARC/InstantMesh"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a> <br>
<a href="https://replicate.com/camenduru/instantmesh"><img src="https://img.shields.io/badge/Demo-Replicate-blue"></a>
<a href="https://colab.research.google.com/github/camenduru/InstantMesh-jupyter/blob/main/InstantMesh_jupyter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
<a href="https://github.com/jtydhr88/ComfyUI-InstantMesh"><img src="https://img.shields.io/badge/Demo-ComfyUI-8A2BE2"></a>

</div>

---

## Overview

This repo is a fork of the official InstantMesh with added [**Fre**quency modulat**e**d tri**plane**](https://freeplane3d.github.io/) technique to improve quality of the output meshes during inference time. Using this technique the resulting meshes are less noisy and smooth.

The idea is to apply Bilateral Filtering to the triplane in order to remove high-frequency components such as highly detailed conflicts or scattered noises. Since the filtering is applied during inference time, there is no need to retrain the model to use this technique.

Freeplane can be applied not only to InstantMesh, but to any triplane-based model, e.g. [CRM](https://ml.cs.tsinghua.edu.cn/~zhengyi/CRM/).

![Generated meshes comparison](assets/freeplane-results.png)

## Install

1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or [miniforge](https://conda-forge.org/miniforge/) (recommended).

2. Clone this repository and open Anaconda shell in the root directory of the repository.

3. Install the environment using the following command:

```bash
conda env create -f environment.yaml
```

4. Activate the environment:

```bash
conda activate fim
```

## Usage

There are two ways to use this project:

1. (Basic) Using command line, you can run the `run.py` script as follows:

```bash
python run.py <PATH-TO-CONFIG> <INPUT-FILE-OR-DIRECTORY>
```

`<PATH-TO-CONFIG>` is a path to one of the files in the `configs` directory. `<INPUT-FILE-OR-DIRECTORY>` is a path to an image file or directory of images, e.g. files in `examples` directory.

To learn about additional arguments, run:

```bash
python run.py -h
```

2. (Advanced) Use Jupyter notebook `inference.ipynb` to run the model. You can modify the variables and configurations along the way.


**Below is the original README. You can use it for additional info about InstantMesh model**

---

This repo is the official implementation of InstantMesh, a feed-forward framework for efficient 3D mesh generation from a single image based on the LRM/Instant3D architecture.

https://github.com/TencentARC/InstantMesh/assets/20635237/dab3511e-e7c6-4c0b-bab7-15772045c47d

# 🚩 Features and Todo List
- [x] 🔥🔥 Release Zero123++ fine-tuning code. 
- [x] 🔥🔥 Support for running gradio demo on two GPUs to save memory.
- [x] 🔥🔥 Support for running demo with docker. Please refer to the [docker](docker/) directory.
- [x] Release inference and training code.
- [x] Release model weights.
- [x] Release huggingface gradio demo. Please try it at [demo](https://huggingface.co/spaces/TencentARC/InstantMesh) link.
- [ ] Add support for more multi-view diffusion models.

# ⚙️ Dependencies and Installation

We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name instantmesh python=3.10
conda activate instantmesh
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# For Linux users: Install Triton 
pip install triton

# For Windows users: Use the prebuilt version of Triton provided here:
pip install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl

# Install other requirements
pip install -r requirements.txt
```

# 💫 How to Use

## Download the models

We provide 4 sparse-view reconstruction model variants and a customized Zero123++ UNet for white-background image generation in the [model card](https://huggingface.co/TencentARC/InstantMesh).

Our inference script will download the models automatically. Alternatively, you can manually download the models and put them under the `ckpts/` directory.

By default, we use the `instant-mesh-large` reconstruction model variant.

## Start a local gradio demo

To start a gradio demo in your local machine, simply run:
```bash
python app.py
```

If you have multiple GPUs in your machine, the demo app will run on two GPUs automatically to save memory. You can also force it to run on a single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python app.py
```

Alternatively, you can run the demo with docker. Please follow the instructions in the [docker](docker/) directory.

## Running with command line

To generate 3D meshes from images via command line, simply run:
```bash
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video
```

We use [rembg](https://github.com/danielgatis/rembg) to segment the foreground object. If the input image already has an alpha mask, please specify the `no_rembg` flag:
```bash
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video --no_rembg
```

By default, our script exports a `.obj` mesh with vertex colors, please specify the `--export_texmap` flag if you hope to export a mesh with a texture map instead (this will cost longer time):
```bash
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video --export_texmap
```

Please use a different `.yaml` config file in the [configs](./configs) directory if you hope to use other reconstruction model variants. For example, using the `instant-nerf-large` model for generation:
```bash
python run.py configs/instant-nerf-large.yaml examples/hatsune_miku.png --save_video
```
**Note:** When using the `NeRF` model variants for image-to-3D generation, exporting a mesh with texture map by specifying `--export_texmap` may cost long time in the UV unwarping step since the default iso-surface extraction resolution is `256`. You can set a lower iso-surface extraction resolution in the config file.

# 💻 Training

We provide our training code to facilitate future research. But we cannot provide the training dataset due to its size. Please refer to our [dataloader](src/data/objaverse.py) for more details.

To train the sparse-view reconstruction models, please run:
```bash
# Training on NeRF representation
python train.py --base configs/instant-nerf-large-train.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1

# Training on Mesh representation
python train.py --base configs/instant-mesh-large-train.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1
```

We also provide our Zero123++ fine-tuning code since it is frequently requested. The running command is:
```bash
python train.py --base configs/zero123plus-finetune.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1
```

# :books: Citation

If you find our work useful for your research or applications, please cite using this BibTeX:

```BibTeX
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

# 🤗 Acknowledgements

We thank the authors of the following projects for their excellent contributions to 3D generative AI!

- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus)
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes)
- [Instant3D](https://instant-3d.github.io/)

Thank [@camenduru](https://github.com/camenduru) for implementing [Replicate Demo](https://replicate.com/camenduru/instantmesh) and [Colab Demo](https://colab.research.google.com/github/camenduru/InstantMesh-jupyter/blob/main/InstantMesh_jupyter.ipynb)!  
Thank [@jtydhr88](https://github.com/jtydhr88) for implementing [ComfyUI support](https://github.com/jtydhr88/ComfyUI-InstantMesh)!
