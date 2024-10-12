#  QCS:Feature Refining from Quadruplet Cross Similarity for Facial Expression Recognition
By Chengpeng Wang, Li Chen, Lili Wang, Zhaofan Li, Xuebin Lv.

This is a PyTorch implementation of the paper QCS:Feature Refining from Quadruplet Cross Similarity for Facial Expression Recognition, based on [POSTER++](https://github.com/Talented-Q/POSTER_V2).

## Introduction
<img width="1024" alt="image" src="https://github.com/birdwcp/QCS/blob/main/fig/framework.png">
On facial expression datasets with complex and numerous feature types, where the significance and dominance of labeled features are difficult to predict, facial expression recognition(FER) encounters the challenges of inter-class similarity and intra-class variances, making it difficult to mine effective features. We aim to solely leverage the feature similarity among facial samples to address this. We introduce the Cross Similarity Attention (CSA), an input-output position-sensitive attention mechanism that harnesses feature similarity across different images to compute the corresponding global spatial attention. Based on this, we propose a four-branch circular framework, called Quadruplet Cross Similarity (QCS), to extract discriminative features from the same class and eliminate redundant ones from different classes synchronously to refine cleaner features. The symmetry of the network ensures balanced and stable training and reduces the amount of CSA interaction matrix. Contrastive residual distillation is utilized to transfer the information learned in the cross module back to the base network. The cross-attention module exists during training, and only one base branch is retained during inference. our proposed QCS model outperforms state-of-the-art methods on several popular FER datasets, without requiring additional landmark information or pre-training on external FER datasets.

Checkpoints



## Installation



## Datasets





## Training




## Evaluation



## Citation
```
@inproceedings{Wang2024QCS,
  title={QCS:Feature Refining from Quadruplet Cross Similarity\\for Facial Expression Recognition},
  author={Chengpeng Wang, Li Chen, Lili Wang, Zhaofan Li, Xuebin Lv},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```
