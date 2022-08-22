![banner](https://user-images.githubusercontent.com/36157933/181859205-b67c5429-6a35-4ca0-8aa6-a51921522d07.png)

# VARAID-VSR
This project is part of a submodule of the [VARAID](https://github.com/nadimra/project-varaid) project. **VARAID-VSR** is a project which is forked from the project [Zooming Slow-Mo](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020). We add further adjustments in this forked project, such as segmentation integration during training to better cater for Football broadcasts. We also trained this network on our *FootballVids* and *PlayerVids* dataset.  

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
6. Create a folder `ckpts` within the root directory and place your model paths here. The trained models for this project can be found in our [Model Zoo](https://drive.google.com/drive/folders/1F7rasOfAcyCeoxXXcAT9fox6niaijrG_?usp=sharing).

# How to use
##### Testing
Inside the `codes` directory, run the following (Edit the file to ensure the file paths are correct):
```
test.py
```

##### Training
Training documentation not available.

# Samples

#### Samples of our STVSR model
The left images are the upscaled LR images which have been set to half the frame rate. The right images are the HR images produced by our final model.

![sota-eval-gif-1-side-zoom-original](https://user-images.githubusercontent.com/36157933/184448457-19b0302f-53eb-4ba1-890f-20ae6e05cd6f.gif)

![sota-eval-gif-2-side-zoom-original](https://user-images.githubusercontent.com/36157933/184447451-c5f7b30b-036d-4ee9-b3e9-b1a4a22a5a94.gif)

![sota-eval-gif-3-side-zoom-original](https://user-images.githubusercontent.com/36157933/184450620-846ec98b-9904-49cc-bddb-efde5ea080be.gif)

![sota-eval-gif-4-side-zoom-original](https://user-images.githubusercontent.com/36157933/184452324-75187e93-4dbc-4fcd-82b3-e3c4e9211f6a.gif)

#### Effects of our STVSR model for object detection
The left images are the upscaled LR images. The right images are the HR images produced by our final model. We apply both sets of images to the YOLOv5 object detector to detect the ball objects.

![object-detection-stvsr-comparison](https://user-images.githubusercontent.com/36157933/184502015-df3d4ba4-d541-4c9a-8fdc-02c77a3a786b.gif)

#### Effects of our STVSR model for human pose estimation
The left images are the upscaled LR images. The right images are the HR images produced by our final model. We apply both sets of images to the HRNET pose estimation network.

![hrnet-stvsr-comparison](https://user-images.githubusercontent.com/36157933/184510729-44c0bead-55d5-434c-8fe9-50f2335e3c41.gif)

# Acknowledgements
This code is built on [Zooming Slow-Mo](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020). We also utilise [Semantic Segmentation on MIT ADE20K](https://github.com/CSAILVision/semantic-segmentation-pytorch) to aid the training phase. We thank the authors for sharing their codes. 

