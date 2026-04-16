# MambaUS-Net
Official implementation of **MambaUS-Net** for ultrasound image segmentation with multi-scale and boundary-aware feature modeling.

![Overall structure of MambaUS-Net](overall-structure-mambaunet.png)

## Overview
MambaUS-Net is a deep learning framework for ultrasound image segmentation.  
It is designed to improve anatomical structure modeling in challenging ultrasound images by combining multi-scale feature extraction and boundary-aware feature refinement.

This repository provides the official implementation of the proposed network.

## Repository
GitHub repository:  
https://github.com/zhoujiayi1017/MambaUS-Net/

## Environment
The code was developed and tested under the following environment:

- Miniconda 25.7.0
- Python 3.10
- PyTorch 2.9.0+cu128
- Torchvision 0.24.0
- CUDA Toolkit 12.8.93
- cuDNN 9.10.2

## Installation
Clone the repository and install the required dependencies.

```bash
git clone https://github.com/zhoujiayi1017/MambaUS-Net.git
cd MambaUS-Net
pip install -r requirements.txt
