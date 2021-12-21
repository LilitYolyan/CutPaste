from torch.utils.data import DataLoader ,Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from glob import glob
from PIL import Image

#todo customize dataset for MvTech
#todo dataloader

class CutPaste_Dataet(Dataset):

    def __init__(self,  train_images = None, test_images = None, image_size = (256,256),  mode = 'train'):
        self.mode = mode

        self.transform = transforms.Compose([transforms.Resize(image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        if train_images and mode == 'train':
            self.images = glob(train_images + "/*")

        elif test_images and mode == 'test':
            self.images = datasets.ImageFolder(test_images, transform=self.transform)

        else:
            raise ValueError('Please specify dataset path and mode')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if self.mode == 'train':
            image_path = self.images[item]
            image = Image.open(image_path)
            image = self.transform(image)
            return image

        else:
            image, label = self.images[item]
            return image, label
