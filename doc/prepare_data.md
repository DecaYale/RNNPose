# Data Preparation Tips
All the related data for data preparation can be downloaded [here](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155139432_link_cuhk_edu_hk/EoXnZ96Tuy9PpYlZCvDN8vUBPdP1lP-PWQWiZH2KtIQoaQ?e=lpE472). You could download them first and then follow the instructions below for data preparation. 



## Download Datasets 
First, the following dataset need to be downloaded and extracted to the folder *EXPDATA/* 

[LINEMOD](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155139432_link_cuhk_edu_hk/EYFaYrk0kcdBgC6WMtLJqP0B9Ar0_Nff9qhI2Cs95qDbdA?e=yYxexC)

[LINEMOD_OCC_TEST](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155139432_link_cuhk_edu_hk/EUKcRnwyy9RGu2ASwA3QDXsBnMRrFP-U4X4Eqq-g_MhmIQ?e=hv6H2s)

## Synthetic Data Generation
The preprocessed data following [DeepIM](https://github.com/liyi14/mx-DeepIM) and [PVNet](https://github.com/zju3dv/pvnet-rendering) can be downloaded from [LM6d_converted](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155139432_link_cuhk_edu_hk/EYFaYrk0kcdBgC6WMtLJqP0B9Ar0_Nff9qhI2Cs95qDbdA?e=yYxexC) and [raw_data](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155139432_link_cuhk_edu_hk/ESSFXi_7qs1AgNmty7_9y4AB8ffFsGJWOC3ikgD5BIeXHQ?e=qOmvds). 
After downloading, you should put the downloaded files into the folder *EXPDATA/* (lying in the repository's root directory). 
To create occluded objects during training, we follow [PVNet](https://github.com/zju3dv/pvnet-rendering) to randomly create occlusions. 
You could run the following scripts to transform the data format for our dataloader. 
```
    bash scripts/run_dataformatter.sh
```
The command above will automatically save the formatted data into *EXPDATA/*. 

## Download the Object CAD Models
You also need to download the [object models](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155139432_link_cuhk_edu_hk/EQScZuLrkPNPmN4eO3kePaUBjOe92EvbKb7kGJk2vKz-bA?e=8McAdh) and put the extracted folder *models* into *./EXPDATA/LM6d_converted/. 

## Download Background Images
[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) need to be downloaded to folder *EXPDATA/*. These images will be necessary for the random background generation for training. 

## Download Initial Poses 
The initial poses estimated by PoseCNN and PVNet can be downloaded from [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155139432_link_cuhk_edu_hk/EQh5y0M_zHVMnbVszjEviCUBNAX_22MFN26Msa48XlJ5MQ?e=rfhT7k). 
The initial pose folder also should be put into the folder  *EXPDATA/*

## Generate the Information Files
Run the following script to generate the info files, which is put into the folder *EXPDATA/data_info/*

```
bash scripts/run_datainfo_generation.sh
```


After the the data preparation, the expected directory structure should be 


```
./EXPDATA
    |──LM6d_converted 
    |        |──LM6d_refine 
    |        |──LM6d_refine_syn
    |        └──models
    |──LINEMOD
    |        └──fuse_formatted
    |──lmo
    |──VOCdevkit
    |──raw_data
    |──init_poses
    └──data_info
```

