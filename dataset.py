from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
from cutpaste import CutPaste

class CutPaste_Dataet(Dataset):
    def __init__(self,  train_images = None, test_images = None, image_size = (256,256),  mode = 'train', cutpaste_type = '3way'):
        '''
        Dataset object
        General dataset structure for training self supervised learning and testing with binary classification

        :param train_images[str]: folder to train images
        :param test_images[str]: folder to test dataset, which should have standard classification
                                 structure similar to datasets.ImageFolder
        :param image_size[tuple]: image size for training
        :param mode[str]: options ['train', 'test']
        :cutpaste_type[str]: options ['binary', '3way']
        '''
        self.mode = mode
        self.train_images = train_images
        self.test_images = test_images
        self.cutpaste_transform = CutPaste(type=cutpaste_type)
        if type(image_size) is not tuple: 
            image_size = (image_size, image_size)
        self.transform = transforms.Compose([transforms.Resize(image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])

        if self.train_images and self.mode == 'train':
            self.images = glob(self.train_images + "/*")

        elif self.test_images and self.mode == 'test':
            self.images = datasets.ImageFolder(self.test_images, transform=self.transform)

        # to fit anomaly detection: needs improvements
        elif self.train_images and self.mode == 'test':
            self.images = datasets.ImageFolder(self.train_images, transform=self.transform)

        else:
            raise ValueError('Please specify dataset path and mode')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        """
        :return: torch tensor if mode is 'train' else torch tensor and label
        """
        if self.mode == 'train':
            image_path = self.images[item]
            image = Image.open(image_path).convert('RGB')
            out = self.cutpaste_transform(image)
            transformed = [self.transform(i) for i in out]
            return transformed

        else:
            image, label = self.images.samples[item]
            image = Image.open(image).convert('RGB')
            image = self.transform(image)
            return image, label



class MVTecAD(CutPaste_Dataet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.mode == 'test':
            self.images.samples = [(d, 0) if i == self.images.class_to_idx['good'] else (d, 1) for d, i in self.images.samples]

        else:
            self.images = glob(self.train_images + '/*/*')

