<div id="top" align="center">
  
# DartBlur 
**Privacy Preservation with Detection Artifact Suppression**

  [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_DartBlur_Privacy_Preservation_With_Detection_Artifact_Suppression_CVPR_2023_paper.pdf) | [Supplemental](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Jiang_DartBlur_Privacy_Preservation_CVPR_2023_supplemental.pdf) | [Video](https://youtu.be/W7dX0WH32Ug)

  Baowei Jiang*, Bing Bai*, Haozhe Lin*, Yu Wang, Yuchen Guo, Lu Fang
  
<a href="#license">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"/>
</a>  

</div>


## Table of Contents
- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Dartblur Images](#dartblur-images)
- [Training](#training)
  - [Dependency Installation](#dependency-installation)
  - [Start Training](#start-training)
- [Citation](#citation)


## Overview
<img src="./img/dartblur_cvpr23_github.png" width="100%" alt="overview" align=center />


## Data Preparation

#### Download WIDER FACE Dataset
You can download images and annotations from website (http://shuoyang1213.me/WIDERFACE/index.html), and unzip files to data/widerface/.

<details>
  <summary>[Expected directory structure of WIDERFACE (click to expand)]</summary>

```
./data/widerface
└───train
│   └───images
│   |   └───0--Parade
│   |       │   0_Parade_marchingband_1_5.jpg
│   |       │   ...
│   |   └───1--Handshaking
│   |       │   1_Handshaking_Handshaking_1_42.jpg
│   |       │   ...
|   |   ...
└───val
│   └───images
│   |   └───0--Parade
│   |       │   0_Parade_marchingband_1_20.jpg
│   |       │   ...
│   |   └───1--Handshaking
│   |       │   1_Handshaking_Handshaking_1_35.jpg
│   |       │   ...
|   |   ...
└───wider_face_split
|   └───wider_face_train_bbx_gt.txt
|   └───wider_face_val_bbx_gt.txt
|   └───...
```
</details>

#### Download WIDER FACE labels
You need download labels file from RetinaFace (https://github.com/deepinsight/insightface/tree/master/detection/retinaface)
Organise the dataset directory as follows:
```
./data/widerface
└───train
│   └───images
│   └───label.txt
└───val
│   └───images
│   └───label.txt
```
</details>

#### Data pre-processing: Gaussian Blur
    python gaussblur.py

## Dartblur Images
    python dartblur.py


## Training
#### dependency installation 
    (Ubuntu/Win10 3090Ti cuda11.8)
    conda create -n dartblur python=3.8
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install -r requirements.txt
    
#### start training
    python train.py
    

## Citation

    @inproceedings{jiang2023dartblur,
    title={DartBlur: Privacy Preservation With Detection Artifact Suppression},
    author={Jiang, Baowei and Bai, Bing and Lin, Haozhe and Wang, Yu and Guo, Yuchen and Fang, Lu},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={16479--16488},
    year={2023}
    }