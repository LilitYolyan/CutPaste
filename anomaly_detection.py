from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
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
    parser.add_argument('--checkpoint',
                        help='checkpoint to trained self-supervised model')
    parser.add_argument('--data', help='path to MVTec dataset')
    parser.add_argument('--save_exp', default='./anomaly_exp', help = 'Save fitted models and roc curves')
    args = parser.parse_args()
    return args

class AnomalyDetection:

    def __init__(self,  weights, device = 'cuda'):
        '''
        Anomaly Detection

        args:
        weights[str] _ path to weights
        device[str] _ device on wich model should be run
        '''
        self.cutpaste_model = self.model(device, weights)
        self.device = device

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
        image_path = os.path.join(save_images, defect_name+'.jpg') if defect_name else os.path.join(save_images, 'roc_curve.jpg')
        plt.savefig(image_path)
        plt.close()
        return roc_auc

    def create_embeds(self, path_to_images):
        ####### OPTIMIZE
        embeddings = []
        labels = []
        dataset = MVTecAD(path_to_images, mode = 'test')
        dataloader = DataLoader(dataset, batch_size = 32)
        with torch.no_grad():
            for imgs, lbls in dataloader:
                features, logits, embeds = self.cutpaste_model(imgs.to(self.device))
                del features, logits
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
        train_embeds , _ = self.create_embeds(train_images)
        GDE_model = self.GDE_fit(train_embeds, save_path)
        test_embeds, test_labels = self.create_embeds(test_images)
        scores = self.GDE_scores(test_embeds, GDE_model)
        auc = self.roc_auc(test_labels, scores, defect_name, save_path)
        return auc


if __name__ == '__main__':
    args = get_args()
    anomaly = AnomalyDetection(args.checkpoint)

    for defect in glob(os.path.join(args.data , '*')):
        defect_name = os.path.split(defect)[-1]
        save_path = os.path.join(args.save_exp, defect_name)
        os.makedirs(save_path, exist_ok=True)
        res = anomaly.mvtec_anomaly_detection(defect, save_path)
        print(f'Defect {defect_name}, AUC = {res}, ROC curve is saved')