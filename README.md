![banner](https://user-images.githubusercontent.com/36157933/181859205-b67c5429-6a35-4ca0-8aa6-a51921522d07.png)

# VARAID-VSR
This project is part of a submodule of the [VARAID](https://github.com/nadimra/project-varaid) project.**VARAID-VSR** is a project which is forked from the project [Zooming Slow-Mo](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020). We add further adjustments in this forked project, such as segmentation integration during training to better cater for Football broadcasts. We also trained this network on our *FootballVids* and *PlayerVids* dataset.  

# Setup
##### Initial Setup
1. `git clone https://github.com/nadimra/vsr`
2. You must install have a specific version of Pytorch (1.9) installed to correspond to be compatible with the DCNv2 module which is integrated in this project. A GPU is also required. The choice of CUDA is dependant on your system, but for this project, we used CUDA 11.1 since it is available on Imperials machines. To follow the same installation as us, install `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`. ***Note: Uninstall torch, torchaudio and torchvision if those packages are already in the environment and is not version 1.9.***
3. Navigate to the root of the directory and install the rest of the packages `pip install -r requirements.txt`

##### DCNv2 Setup
4. Before compiling DCNv2, you must ensure that you set the configuration of the CUDA directories. If you are following the exact installation process in the same system, then the following will suffice:

```
export CUDA_HOME=/vol/cuda/11.1.0-cudnn8.0.4.30
export CUDNN_INCLUDE_DIR=/vol/cuda/11.1.0-cudnn8.0.4.30/include
export CUDNN_LIB_DIR=/vol/cuda/11.1.0-cudnn8.0.4.30/lib64
```

5. Navigate to `/codes/models/modules/DCNv2` and run `bash make.sh`

##### Model Setup
6. Create a folder `ckpts` within the root directory and place your model paths here. The trained models for this project can be found here.

# How to use
##### Testing
Inside the `codes` directory, run the following (Edit the file to ensure the file paths are correct):
```
test.py
```

##### Training
Training documentation not available.

# Acknowledgements
This code is built on [Zooming Slow-Mo](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020). We also utilise [Semantic Segmentation on MIT ADE20K](https://github.com/CSAILVision/semantic-segmentation-pytorch) to aid the training phase. We thank the authors for sharing their codes. 

