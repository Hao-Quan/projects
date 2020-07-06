# Shift-GCN
The implementation for "Skeleton-Based Action Recognition with Shift Graph Convolutional Network" (CVPR2020). Shift-GCN is a lightweight skeleton-based action recognition model, which exceeds state-of-the-art methods with 10x less FLOPs.

## Prerequisite

 - PyTorch 0.4.1
 - Cuda 9.0
 - g++ 5.4.0
 
 HAO's setting environment:
 - PyTorch 1.4
 - Cuda(Cudatoolkit; nvcc --version) 10.1
 - Torchvision 0.5
 - gcc/g++ 7.5
 
  

## Compile cuda extensions

  Conda environment create:
  ```
  conda create -n py36 python=3.6 anaconda
  conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
  ```
  
 It should be environment as following, otherwise, does not work!
 - Python 3.6
 - PyTorch 1.1
 - Cuda(Cudatoolkit; nvcc --version) 10.0.130
 - cudnn 7.6.5
 - Torchvision 0.3.0
 - gcc 5.5
 - g++ 5.5

Note: PyTorch 1.5 dose not work!

Compile shif_cuda.cpp file
Should run it in command line, don't with PyCharm.
  ```
  cd ./model/Temporal_shift
  bash run.sh
  ```

Although it seems error with:

![Alt text](readme_img/1.png?raw=true "Title")



## Data Preparation

 - Download the raw data of [NTU-RGBD](https://github.com/shahroudy/NTURGB-D) and [NTU-RGBD120](https://github.com/shahroudy/NTURGB-D). Put NTU-RGBD data under the directory `./data/nturgbd_raw`. Put NTU-RGBD120 data under the directory `./data/nturgbd120_raw`. 
 
 - For NTU-RGBD, preprocess data with `python data_gen/ntu_gendata.py`. For NTU-RGBD120, preprocess data with `python data_gen/ntu120_gendata.py`. 
  
 - Generate the bone data with `python data_gen/gen_bone_data.py`.

 - Generate the motion data with `python data_gen/gen_motion_data.py`.

## Training & Testing

  - NTU X-view

    `python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-view/train_bone.yaml`

    `python main.py --config ./config/nturgbd-cross-view/train_joint_motion.yaml`

    `python main.py --config ./config/nturgbd-cross-view/train_bone_motion.yaml`

  - NTU X-sub

    `python main.py --config ./config/nturgbd-cross-subject/train_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-subject/train_bone.yaml`

    `python main.py --config ./config/nturgbd-cross-subject/train_joint_motion.yaml`

    `python main.py --config ./config/nturgbd-cross-subject/train_bone_motion.yaml`

  - For NTU120, change the dataset path in config files, and change `num_class` in config files from 60 to 120.
  
## Multi-stream ensemble

To ensemble the results of 4 streams. Change models name in `ensemble.py` depending on your experiment setting. Then run `python ensemble.py`.

## Trained models

We release several trained models:

Model|Dataset|Setting|Top1(%)
-|-|-|-
./save_models/ntu_ShiftGCN_joint_xview.pt|NTU-RGBD|X-view|95.1
./save_models/ntu_ShiftGCN_joint_xsub.pt|NTU-RGBD|X-sub|87.8
./save_models/ntu120_ShiftGCN_joint_xsetup.pt|NTU-RGBD120|X-setup|83.2
./save_models/ntu120_ShiftGCN_joint_xsub.pt|NTU-RGBD120|X-sub|80.9

     
## Citation
If you find this model useful for your resesarch, please use the following BibTeX entry.

    @inproceedings{cheng2020shiftgcn,  
      title     = {Skeleton-Based Action Recognition with Shift Graph Convolutional Network},  
      author    = {Ke Cheng and Yifan Zhang and Xiangyu He and Weihan Chen and Jian Cheng and Hanqing Lu},  
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
      year      = {2020},  
    }
