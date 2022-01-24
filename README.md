# Demo_AHCL_for_TGRS2022
This is the code implementation for our paper "Asymmetric Hash Code Learning for Remote Sensing Image Retrieval", IEEE Trans. Geosci. Remote Sens, 2022.

If you use this code in your work, please kindly cite our paper:

@article{song2022, 

title={Asymmetric Hash Code Learning for Remote Sensing Image Retrieval},

author={Song, Weiwei and Gao, Zhi and Dian, Renwei and Ghamisi, Pedram and Zhang, Yongjun and Benediktsson, J{'o}n Atli},

journal={IEEE Transactions on Geoscience and Remote Sensing},

DOI={10.1109/TGRS.2022.3143571},

publisher={IEEE}

}

# Usage
### 1. Running example:
Environment: python 3

Requirements:
```python
pytorch
torchvision
```
### 2. Data processing:
Download the WHU-RS data set from https://pan.baidu.com/s/1CtnEv0p6tbAYGBscgdTDsA , the passwords are: v5ac. 
upzip the data file into ./data/WHURS-19/
### 3. Demo:
```python
python Demo_AHCL_WHURS19.py
```
