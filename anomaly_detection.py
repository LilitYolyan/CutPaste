from typing import Any, Dict, Tuple, Union
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from model import CutPasteNet
from dataset import MVTecAD
from torch.utils.data import DataLoader
import numpy
import pickle
import torch
import numpy as np
import os
import tqdm
import argparse
from glob import glob
import pathlib

#TODO plot tsne

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_checkpoints',
                        help='path to the folder where results from self-supervised are saved, eg: ./tb_logs')
    parser.add_argument('--data', help='path to MVTec dataset root.')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--save_exp', default=pathlib.Path(__file__).parent/'anomaly_exp', help = 'Save fitted models and roc curves')
    args = parser.parse_args()
    return args

class AnomalyDetection:
    def __init__(self,  weights, batch_size, device = 'cuda'):
        '''
        Anomaly Detection

        args:
        weights[str] _ path to weights
        device[str] _ device on wich model should be run
        '''
        self.cutpaste_model = self.model(device, weights)
        self.device = device
        self.batch_size = batch_size

    @staticmethod
    def model(device, weights):
        model = CutPasteNet(pretrained=False)
        state_dict = torch.load(weights)['state_dict']         ## change state_dict structure
        state_dict = {i.replace('model.', '') : j  for i, j in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return  model

    @staticmethod
    def roc_auc(labels, scores, defect_name = None, save_path = None):
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        plt.title(f'ROC curve: {defect_name}')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        save_images = save_path if save_path else './roc_results'
        os.makedirs(save_images, exist_ok=True)
        image_path = os.path.join(save_images, defect_name+'_roc.jpg') if defect_name else os.path.join(save_images, 'roc_curve.jpg')
        plt.savefig(image_path)
        plt.close()
        return roc_auc

    @staticmethod
    def plot_tsne(labels, embeds, defect_name = None, save_path = None, **kwargs: Dict[str, Any]):
        """t-SNE visualize

        Args:
            labels (Tensor): labels of test and train
            embeds (Tensor): embeds of test and train
            defect_name ([str], optional): same as <defect_name> in roc_auc. Defaults to None.
            save_path ([str], optional): same as <defect_name> in roc_auc. Defaults to None.
            kwargs (Dict[str, Any]): hyper parameters of t-SNE which will change final result
                n_iter (int): > 250, default = 1000
                learning_rate (float): (10-1000), default = 100
                perplexity (float): (5-50), default = 28
                early_exaggeration (float): change it when not converging, default = 12
                angle (float): (0.2-0.8), default = 0.3
                init (str): "random" or "pca", default = "pca
        """        
        tsne = TSNE(
            n_components=2, 
            verbose=1, 
            n_iter=kwargs.get("n_iter", 1000),
            learning_rate=kwargs.get("learning_rate", 100),
            perplexity=kwargs.get("perplexity", 28), 
            early_exaggeration=kwargs.get("early_exaggeration", 12),
            angle=kwargs.get("angle", 0.3),
            init=kwargs.get("init", "pca"),
        )
        embeds, labels = shuffle(embeds, labels)
        tsne_results = tsne.fit_transform(embeds)

        cmap = plt.cm.get_cmap("spring")
        colors = np.vstack((np.array([[0, 1. ,0, 1.]]), cmap([0, 256//3, (2*256)//3])))
        legends = ["good", "anomaly", "cutpaste", "cutpaste-scar"]
        (_, ax) = plt.subplots(1)
        plt.title(f't-SNE: {defect_name}')
        for label in torch.unique(labels):
            res = tsne_results[torch.where(labels==label)]
            ax.plot(*res.T, marker="*", linestyle="", ms=5, label=legends[label], color=colors[label])
            ax.legend(loc="best")
        plt.xticks([])
        plt.yticks([])

        save_images = save_path if save_path else './tnse_results'
        os.makedirs(save_images, exist_ok=True)
        image_path = os.path.join(save_images, defect_name+'_tsne.jpg') if defect_name else os.path.join(save_images, 'tsne.jpg')
        plt.savefig(image_path)
        plt.close()
        return

    def create_embeds(self, path_to_images):
        ####### OPTIMIZE
        embeddings = []
        labels = []
        dataset = MVTecAD(path_to_images, mode = 'test')
        dataloader = DataLoader(dataset, batch_size = self.batch_size)
        with torch.no_grad():
            for imgs, lbls in dataloader:
                features, logits, embeds = self.cutpaste_model(imgs.to(self.device))
                del features, logits
                embeddings.append(embeds.to('cpu'))
                labels.append(lbls.to('cpu'))
                torch.cuda.empty_cache()

        return torch.cat(embeddings), torch.cat(labels)
    
    def create_embeds_tsne(self, path_to_images: Union[str, pathlib.Path]) -> Tuple[torch.Tensor, torch.Tensor]:
        """load trainset with embeds and labels for tsne
            labels in [0, 1, 2] when using 3way or [0, 1], label==0 means "good" 
        Args:
            path_to_images (Union[str, pathlib.Path]): path of trainset

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: embeds, labels
        """        
        embeddings = []
        labels = []
        dataset = MVTecAD(path_to_images, mode = 'train')
        dataloader = DataLoader(dataset, batch_size = self.batch_size)
        with torch.no_grad():
            for imgs in dataloader:
                (_, lbls) = torch.meshgrid(torch.arange(0, self.batch_size), torch.arange(0, len(imgs)), indexing="xy")
            for (i_batch, imgs) in enumerate(dataloader):
                n = self.batch_size if (i_batch <= len(dataset)/self.batch_size - 1) else len(dataset) % self.batch_size
                (_, lbls) = torch.meshgrid(torch.arange(0, n), torch.arange(0, len(imgs)), indexing="xy")
                imgs = torch.concat(imgs)
                lbls = lbls.to(dtype=torch.int).flatten()
                (_, _, embeds) = self.cutpaste_model(imgs.to(self.device))
                assert len(embeds) == len(lbls) == len(imgs), IndexError(f"Mismatch: len(embeds), len(lbls), (imgs): {len(embeds), len(lbls), (imgs)}")
                embeddings.append(embeds.to('cpu'))
                labels.append(lbls.to('cpu'))
                torch.cuda.empty_cache()
        return torch.cat(embeddings), torch.cat(labels)

    @staticmethod
    def GDE_fit( train_embeds, save_path = None):
        GDE = KernelDensity().fit(train_embeds)
        if save_path:
            filename = os.path.join(save_path, 'GDE.sav')
            pickle.dump(GDE, open(filename, 'wb'))
        return GDE

    @staticmethod
    def GDE_scores(embeds, GDE ):
        scores = GDE.score_samples(embeds)
        norm = np.linalg.norm(-scores)
        return -scores/norm

    def mvtec_anomaly_detection(self, path_to_defect, save_path=None):
        '''
        args:
        path_to_defect[str] _ path to one defect category in MVTec

        returns:
        AUC score and ROC curve saved
        '''

        train_images = os.path.join(path_to_defect, 'train')
        test_images = os.path.join(path_to_defect, 'test')
        defect_name = os.path.split(path_to_defect)[-1]
        train_embeds, train_labels = self.create_embeds_tsne(train_images)
        GDE_model = self.GDE_fit(train_embeds, save_path)
        test_embeds, test_labels = self.create_embeds(test_images)
        scores = self.GDE_scores(test_embeds, GDE_model)
        auc = self.roc_auc(test_labels, scores, defect_name, save_path)
        # for tsne we encode training data as 2, and augmentet data as 3
        train_labels = torch.where(train_labels != 0, train_labels + len(torch.unique(test_labels)) - 1, torch.zeros((1, ), dtype=torch.int))
        tsne_labels = torch.cat([test_labels, train_labels])
        tsne_embeds = torch.cat([test_embeds, train_embeds])
        self.plot_tsne(tsne_labels, tsne_embeds, defect_name, save_path)
        return auc



if __name__ == '__main__':
    # from icecream import ic
    args = get_args()
    all_checkpoints = sorted(pathlib.Path(args.path_to_checkpoints).glob("*/*/*/*.ckpt")) # ./tb_logs

    for defect in glob(os.path.join(args.data, '*')):
        defect_name = os.path.split(defect)[-1]
        checkpoint = [i for i in all_checkpoints if defect_name == i.stem.split("-")[-1]]
        if not checkpoint:
            continue
        else:
            checkpoint: str = checkpoint[0]
        # ic(defect_name, checkpoint)

        anomaly = AnomalyDetection(checkpoint, args.batch_size)
        save_path = os.path.join(args.save_exp, defect_name)
        # ic(save_path)

        os.makedirs(save_path, exist_ok=True)
        res = anomaly.mvtec_anomaly_detection(defect, save_path)
        print(f'Defect {defect_name}, AUC = {res}, ROC curve and t-SNE is saved in: {save_path}')
        with open(os.path.join(args.save_exp, 'AUC_resuts.txt'), 'a') as f:
            f.write(f'{defect_name} {res} \n')
