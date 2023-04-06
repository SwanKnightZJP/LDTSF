# LDTSF
Code for the paper: LDTSF: A Label-decoupling Teacher-student Framework for Semi-supervised Echocardiography Segmentation (ICASSP2023)

# Requirements
Some important requires packages includes:
* [Pytorch][torch_link] version >=1.10.0
* torchvision >=0.11.0
* Python == 3.8 
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage 
1. Clone the repo:
```
git clone https://github.com/SwanKnightZJP/LDTSF
cd LDTSF
```
2. Put the data in [data/2018LA_Seg_Training Set](https://github.com/Luoxd1996/DTC/tree/master/data/2018LA_Seg_Training%20Set).
3. Train the LDC model first. The trained model will be saved at ../model/LA/LDC_with_consis_weight (and our best model is saved at ../model/LA/LDC_Best)
```
cd code
python train_la_LDC.py
```
5. Generate the pseudo labels. The generated pseudo labels will be saved at ../data/2018LA_Seg_PseudoTraining Set
```
cd code
python pseudo_label_create_LA.py
```
6. Train the LDTSF model.
```
cd code
python train_la_LDTSF.py
```

# Note 
Due to patent-related issues, we are not at liberty to disclose the code and data related to 3D Echocardiography for the time being, and we may make relevant updates in the future.


# Acknowledgement 
* This code is adapted from [DTC](https://github.com/HiLab-git/DTC), [UA-MT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [SegWithDistMap](https://github.com/JunMa11/SegWithDistMap). 
* We thank Dr. Xiangde Luo, Dr. Lequan Yu, M.S. Shuailin Li and Dr. Jun Ma for their elegant and efficient code base.


