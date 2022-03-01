# CutPaste
![CutPaste: image from paper](image.png)
CutPaste: image from paper

Unofficial implementation of Google's ["CutPaste: Self-Supervised Learning for Anomaly Detection and Localization"](https://arxiv.org/abs/2104.04015) in PyTorch

### Installation
To rerun experiments or try on your own dataset, first clone the repository and install `requirements.txt`.
```
$ git clone https://github.com/LilitYolyan/CutPaste.git
$ cd CutPaste
$ pip install -r requirements.txt
```

### Self-supervised training
Run `train.py` to train self-supervised model on MVTec dataset

For 3 way classification head 
```
$ python train.py --dataset_path path/to/your/dataset/ --num_class 3
```

For binary classification head 
```
$ python train.py --dataset_path path/to/your/dataset/ --num_class 2
```
For feature extractor any torchvision model can be used.
For example to use EfficientNet(B4) 
```
$ python train.py --dataset_path path/to/your/dataset/ --encoder efficientnet_b4
```
To track training process with TensorBoard
```
tensorboard --logdir logdirs
```


### Anomaly Detection
To run anomaly detection for MVTec with Gaussian Density Estimator 
```
$ python anomaly_detection.py --checkpoint path/to/your/weights --data path/to/mvtec

```
### TODO
- [X] Self-supervised model 
- [X] Gaussian Density Estimator
- [X] EfficientNet Implementation
- [X] t-SNE visualization of representations
- [X] Localization
- [X] Grad-CAM visualization
- [ ] Implement GDE in PyTorch (too slow with sklearn)  

Any contribution is appreciated!

# Experiment Results
For more experiment results go to ["experiments.md"](experiments/experiments.md)

To train self-supervised model we used same hyperparameters as was used in paper: 
| Hyperparameter  | Value |
| ------------- | ------------- |
| Number of epochs | 265 |
| Batch size | 32 |
| Learning rate | 0.03 |
| Input size | 256 |


## AUC comparison of our code and paper results
| Defect Name  | CutPaste binary (ours) | CutPaste binary (paper's)  |  CutPaste 3way (ours) | CutPaste 3way (paper's) |
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| tile  | 84.1 | 95.9 | 78.9 | 93.4 |
| wood  | 89.5 | 94.9 | 89.2 | 98.6 |
| pill | 88.7 | 93.4 | 78.7 | 92.4 |
| leather | 98.7 | 99.7 | 84.8 | 100.0 |
| hazelnut | 98.8 | 91.3 | 80.8 | 97.3 |
| screw | 89.2 | 54.4 | 56.6 |  86.3 |
| cable | 83.3 | 87.7 | 75.7 | 93.1 |
| toothbrush | 94.7 | 99.2 | 78.6 |98.3 |
| capsule | 80.2 | 87.9 | 70.8 | 96.2 |
| carpet | 57.9 | 67.9 | 26.1| 93.1 |
| zipper | 99.5 | 99.4 | 85.7 | 99.4 |
| metal_nut | 91.5 | 96.8 | 89.7 | 99.3 |
| bottle | 98.5 | 99.2 | 75.7 | 98.3 |
| grid | 99.9 | 99.9 | 73.0 | 99.9 |
| transistor | 84.4 | 96.4 | 85.5 | 95.5 |


## ROC curves using embeddings from binary classification for self-supervised learning
<p float="left">
  <img src="/experiments/roc_binary/bottle.jpg" width="260" />
  <img src="experiments/roc_binary/cable.jpg" width="260" /> 
  <img src="experiments/roc_binary/capsule.jpg" width="260" />
</p>
<p float="left">
  <img src="/experiments/roc_binary/carpet.jpg" width="260" />
  <img src="experiments/roc_binary/grid.jpg" width="260" /> 
  <img src="experiments/roc_binary/hazelnut.jpg" width="260" />
</p>
<p float="left">
  <img src="/experiments/roc_binary/leather.jpg" width="260" />
  <img src="experiments/roc_binary/metal_nut.jpg" width="260" /> 
  <img src="experiments/roc_binary/pill.jpg" width="260" />
</p> 

<p float="left">
  <img src="experiments/roc_binary/toothbrush.jpg" width="260" />
  <img src="/experiments/roc_binary/screw.jpg" width="260" />
  <img src="experiments/roc_binary/tile.jpg" width="260" /> 
</p> 

<p float="left">
  <img src="experiments/roc_binary/zipper.jpg" width="260" />
  <img src="experiments/roc_binary/wood.jpg" width="260" /> 
  <img src="/experiments/roc_binary/transistor.jpg" width="260" />
</p>


## t-SNE visualisation of embeddings 
<p float="left">
  <img src="experiments/tsne_3way/grid_tsne.jpg" width="400" />
  <img src="experiments/tsne_3way/wood_tsne.jpg" width="400" /> 
</p>

