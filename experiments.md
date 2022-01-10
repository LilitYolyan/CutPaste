# Experiment Results
To train self-supervised model we used same hyperparameters as was used in paper: 
| Hyperparameter  | Value |
| ------------- | ------------- |
| Number of epochs | 265 |
| Batch size | 32 |
| Learning rate | 0.03 |
| Input size | 256 |


## AUC comparison of our code and paper results
TODO: add results of our 3 way experiment

| Defect Name  | CutPaste binary (ours) | CutPaste binary (paper's)  | CutPaste 3 way (ours) | CutPaste 3 way (paper's)  | 
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| tile  | 84.1 | 95.9 |  | 93.4 |
| wood  | 89.5 | 94.9 |  | 98.6 |
| pill | 88.7 | 93.4 |  | 92.4 |
| leather | 98.7 | 99.7 |  | 100.0 |
| hazelnut | 98.8 | 91.3 |  | 97.3 |
| screw | 89.2 | 54.4 |  | 86.3 |
| cable | 83.3 | 87.7 |  | 93.1 |
| toothbrush | 94.7 | 99.2 |  | 98.3 |
| capsule | 80.2 | 87.9 |  | 96.2 |
| carpet | 57.9 | 67.9 |  | 93.1 |
| zipper | 99.5 | 99.4 |  | 99.4 |
| metal_nut | 91.5 | 96.8 |  | 99.3 |
| bottle | 98.5 | 99.2 |   | 98.3 |
| grid | 99.9 | 99.9 |   | 99.9 |
| transistor | 84.4 | 96.4 |  | 95.5 |


### ROC curves using embeddings from binary classification for self-supervised learning
<details>
  <summary>Click to see ROC curves!</summary>
  
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
</details>

### Self-supervised binary training results

<details>
  <summary>Click to see self-supervised training results!</summary>
  
  **Training accuracy and loss for bottle**
  <p float="left">
    <img src="experiments/self-supervised_binary/train_acc_bottle.png" width="300" />
    <img src="experiments/self-supervised_binary/train_loss_bottle.png" width="300" /> 
  </p>


  **Training accuracy and loss for pill**
  <p float="left">
    <img src="experiments/self-supervised_binary/train_acc_pill.png" width="300" />
    <img src="experiments/self-supervised_binary/train_loss_pill.png" width="300" /> 
  </p>


  **Training accuracy and loss for cable**
  <p float="left">
    <img src="experiments/self-supervised_binary/train_acc_cable.png" width="300" />
    <img src="experiments/self-supervised_binary/train_loss_cable.png" width="300" /> 
  </p>


  **Training accuracy and loss for capsule**
  <p float="left">
    <img src="experiments/self-supervised_binary/train_acc_capsule.png" width="300" />
    <img src="experiments/self-supervised_binary/train_loss_capsule.png" width="300" /> 
  </p>


  **Training accuracy and loss for tile**
  <p float="left">
    <img src="experiments/self-supervised_binary/train_acc_tile.png" width="300" />
    <img src="experiments/self-supervised_binary/train_loss_tile.png" width="300" /> 
  </p>
</details>

### Self-supervised 3-way training results
<details>
  <summary>Click to see self-supervised training results!</summary>
  
  **Training accuracy and loss for pill**
  <p float="left">
    <img src="experiments/self_supervised_3way/train_acc_pill.svg" width="300" />
    <img src="experiments/self_supervised_3way/train_loss_pill.svg" width="300" /> 
  </p>


  **Training accuracy and loss for screw**
  <p float="left">
    <img src="experiments/self_supervised_3way/train_acc_screw.svg" width="300" />
    <img src="experiments/self_supervised_3way/train_loss_screw.svg" width="300" /> 
  </p>


  **Training accuracy and loss for tile**
  <p float="left">
    <img src="experiments/self_supervised_3way/train_acc_tile.svg" width="300" />
    <img src="experiments/self_supervised_3way/train_loss_tile.svg" width="300" /> 
  </p>


  **Training accuracy and loss for zipper**
  <p float="left">
    <img src="experiments/self_supervised_3way/train_acc_zipper.svg" width="300" />
    <img src="experiments/self_supervised_3way/train_loss_zipper.svg" width="300" /> 
  </p>

</details>

### t-SNE visualisation of embeddings 
<details>
  <summary>Click to see t-SNE visualisations!</summary>
  
<p float="left">
  <img src="experiments/tsne_3way/grid_tsne.jpg" width="500" />
  <img src="experiments/tsne_3way/wood_tsne.jpg" width="500" /> 
</p>
</details>
  
