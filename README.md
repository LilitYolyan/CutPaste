# CutPaste
Unofficial implementation of Google's ["CutPaste: Self-Supervised Learning for Anomaly Detection and Localization"](https://arxiv.org/abs/2104.04015)

### Installation
To rerun experiments or try on your own dataset, first clone repository and install `requirements.txt`.
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

To track training proces with TensorBoard
```
tensorboard --logdir logdirs
```


### Anomaly Detection
To run anomaly detection for MVTec with Gaussian Density Estimator 
```
$ python anomaly_detection.py --checkpoint path/to/your/weights --data path/to/mvtec

```

