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
- [ ] EfficientNet Implementation
- [ ] Localization

Any contribution is appreciated!

# Experiment Results
![alt-text-1](experiments/roc_binary/bottle.jpg) ![alt-text-2](experiments/roc_binary/cable.jpg) ![alt-text-3](experiments/roc_binary/capsule.jpg) 
![alt-text-1](experiments/roc_binary/bottle.jpg) ![alt-text-1](experiments/roc_binary/bottle.jpg) ![alt-text-1](experiments/roc_binary/bottle.jpg) 
![alt-text-1](experiments/roc_binary/bottle.jpg) ![alt-text-1](experiments/roc_binary/bottle.jpg) ![alt-text-1](experiments/roc_binary/bottle.jpg) 
![alt-text-1](experiments/roc_binary/bottle.jpg) ![alt-text-1](experiments/roc_binary/bottle.jpg) ![alt-text-1](experiments/roc_binary/bottle.jpg) 
![alt-text-1](experiments/roc_binary/bottle.jpg) ![alt-text-1](experiments/roc_binary/bottle.jpg) ![alt-text-1](experiments/roc_binary/bottle.jpg) 
