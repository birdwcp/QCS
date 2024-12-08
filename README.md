#  QCS:Feature Refining from Quadruplet Cross Similarity for Facial Expression Recognition
By Chengpeng Wang, Li Chen, Lili Wang, Zhaofan Li, Xiao Yang, Xuebin Lv.

This is a PyTorch implementation of the paper QCS:Feature Refining from Quadruplet Cross Similarity for Facial Expression Recognition, based on [POSTER++](https://github.com/Talented-Q/POSTER_V2).

## Introduction
<div align="center">
<img width="800" alt="image" src="./fig/framework.png">
</div>
On facial expression datasets, images labeled with significant features are mingled with complex and numerous unlabeled redundant features. Facial expression recognition (FER) encounters the challenges of inter-class similarity and intra-class variances, making it difficult to mine clean features. To refine features, we introduce Cross Similarity Attention (CSA) to mine richer intrinsic information from image pairs, overcoming a limitation when the Scaled Dot-Product Attention of ViT is directly applied to calculate the similarity between two different images. Based on CSA, we simultaneously minimize intra-class differences and maximize inter-class differences at the fine-grained feature level through interactions among multiple branches. Contrastive residual distillation is utilized to transfer the information learned in the cross module back to the base network. We ingeniously design a four-branch centrally symmetric network, named Quadruplet Cross Similarity (QCS). This network alleviates gradient conflicts arising from the cross module while simultaneously reducing the number of interaction modules. The framework achieves balanced and stable training by adaptively extracting discriminative features while isolating redundant ones. The cross-attention modules exist during training, and only one base branch is retained during inference, resulting in no increase in inference time.

## Installation
We have only tested the code on Windows in PyCharm.
### Requirements:
- numpy==1.21.6
- Pillow==9.3.0
- sklearn==0.0.post1
- matplotlib==3.5.3
- torch==1.13.0+cu117
- torchvision==0.14.0+cu117
- tqdm==4.64.1
- seaborn==0.12.2

## Prepareing
### pretrained model:
dwonload pretrained model [ir50.pth](https://drive.google.com/file/d/1FV8kUSeVbZ815iWt-YIYiQrCDChrhO2G/view?usp=sharing) (on the Ms-Celeb-1M) into `QCS-main/models/pretrain`.
### datasets:
download the [RAF-DB](http://www.whdeng.cn/raf/model1.html), [FERPlus](https://github.com/Microsoft/FERPlus), [AffectNet](http://mohammadmahoor.com/affectnet/) datasets and put the images into `datas/RAF-DB/basic`, `datas/FERPlus/img` and `datas/AffectNet/Manually_trainval_croped` respectively.  download the [trainval lists](https://drive.google.com/file/d/1N2Q063isVbTzr7tZu54AgNs3ZcRT2QLv/view?usp=sharing) and put them into `datas/RAF-DB`, `datas/FERPlus` and `datas/AffectNet` respectively.


## Checkpoints
Compared to methods that introduce additional landmark information, our method achieves state-of-the-art performance or competitive results on several FER datasets by mining richer intrinsic information.

We provide the checkpoints with training logs in each dataset. Some of the checkpoints that were trained in old version code are not currently provided due to naming compatibility issues with network layers. ‡ means pre-trained on the AffectNet-8.
dataset | Model | Acc. | Checkpoint & Log  
--- |:---:|:---:|:---:|
RAF-DB | DCS | 92.57 | [link](https://drive.google.com/file/d/1yPoWxsWnjyfT0Ymca4TwNmeNqVmz8GSC/view?usp=sharing)
RAF-DB | QCS | 92.47 | [link](https://drive.google.com/file/d/1wZ5EvuZWjNpJcB009jjKSxGZ2_y1hiwd/view?usp=sharing)
RAF-DB | QCS‡ | 92.83 | [link](https://drive.google.com/drive/folders/1b25WkmbEqjC9dKsjIrGKUCxpywINjk5e?usp=sharing/)
FERPlus | DCS | 91.41 | [link](https://drive.google.com/drive/folders/1UoQ4xZYDGc0cooQd7BzhDfb58e3wwnjO?usp=sharing)
FERPlus | QCS | 91.37 | [link](https://drive.google.com/drive/folders/19O9BjP7Lhd1DX9r8-RxSmvO1aAUQHoBO?usp=sharing)
FERPlus | QCS‡ | 91.60 | [link](https://drive.google.com/drive/folders/15jqH56e2dVtJx0oPzzcAmvaDAFa0rc01?usp=sharing)
AffectNet-7 | DCS | 67.66 | [link](https://drive.google.com/file/d/1d5yOAEMNwNY3gTC-MRBXCrBuMYN3_Tsa/view?usp=sharing)
AffectNet-7 | QCS | 67.94 | [link](https://drive.google.com/file/d/1XWf0q8wiJz840_ArXURFbv2KfdzVYwEV/view?usp=sharing)
AffectNet-8 | DCS | 64.4 |
AffectNet-8 | QCS | 64.2 | [link](https://drive.google.com/drive/folders/1WFbisNzL-YqqMNSN0sq8vIaXYuh4_1Xm?usp=sharing/)

## Training
The default training parameters are set by `parser.add_argument()` in each `main_*_*.py`.

You can train the QCS or DCS models on each dataset, run as follows: 
```
python main_*_*.py --dataset (RAF-DB|FERPlus|AffectNet-7|AffectNet-8) --epochs (40|100|200) --batch-size (24|48)
```



You can continue your training, run as follows: 
```
python main_*_*.py --resume path/to/checkpoint
```

## Evaluation
You can evaluate our models , run as follows: 
```
python main_*_*.py --evaluate path/to/checkpoint
```
## Citation
```
@inproceedings{Wang2024QCS,
  title={QCS:Feature Refining from Quadruplet Cross Similarity for Facial Expression Recognition},
  author={Chengpeng Wang, Li Chen, Lili Wang, Zhaofan Li, Xiao Yang, Xuebin Lv},
  journal={arXiv preprint arXiv:2411.01988},
  year={2024}
}
```
