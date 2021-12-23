import random
import numpy as np
from torchvision import transforms

class CutPaste(object):

    def __init__(self, transform = True, type = 'binary'):

        '''
        This class creates to different augmentation CutPaste and CutPaste-Scar. Moreover, it returns augmented images
        for binary and 3 way classification

        :arg
        :transform[binary]: - if True use Color Jitter augmentations for patches
        :type[str]: options ['binary' or '3way'] - classification type
        '''
        self.type = type
        if transform:
            self.transform = transforms.ColorJitter(brightness = 0.1,
                                                      contrast = 0.1,
                                                      saturation = 0.1,
                                                      hue = 0.1)
        else:
            self.transform = None

    @staticmethod
    def crop_and_paste_patch(image, patch_w, patch_h, transform, rotation=False):
        """
        Crop patch from original image and paste it randomly on the same image.

        :image: [PIL] _ original image
        :patch_w: [int] _ width of the patch
        :patch_h: [int] _ height of the patch
        :transform: [binary] _ if True use Color Jitter augmentation
        :rotation: [binary[ _ if True randomly rotates image from (-45, 45) range

        :return: augmented image
        """

        org_w, org_h = image.size
        mask = None

        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
        if transform:
            patch= transform(patch)

        if rotation:
            random_rotate = random.uniform(*rotation)
            patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
            mask = patch.split()[-1]

        # new location
        paste_left, paste_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        aug_image = image.copy()
        aug_image.paste(patch, (paste_left, paste_top), mask=mask)
        return aug_image

    def cutpaste(self, image, area_ratio = (0.02, 0.15), aspect_ratio = ((0.3, 1) , (1, 3.3))):
        '''
        CutPaste augmentation

        :image: [PIL] - original image
        :area_ratio: [tuple] - range for area ratio for patch
        :aspect_ratio: [tuple] -  range for aspect ratio

        :return: PIL image after CutPaste transformation
        '''

        img_area = image.size[0] * image.size[1]
        patch_area = random.uniform(*area_ratio) * img_area
        patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
        patch_w  = int(np.sqrt(patch_area*patch_aspect))
        patch_h = int(np.sqrt(patch_area/patch_aspect))
        cutpaste = self.crop_and_paste_patch(image, patch_w, patch_h, self.transform, rotation = False)
        return cutpaste

    def cutpaste_scar(self, image, width = [2,16], length = [10,25], rotation = (-45, 45)):
        '''

        :image: [PIL] - original image
        :width: [list] - range for width of patch
        :length: [list] - range for length of patch
        :rotation: [tuple] - range for rotation

        :return: PIL image after CutPaste-Scare transformation
        '''
        patch_w, patch_h = random.randint(*width), random.randint(*length)
        cutpaste_scar = self.crop_and_paste_patch(image, patch_w, patch_h, self.transform, rotation = rotation)
        return cutpaste_scar

    def __call__(self, image):
        '''

        :image: [PIL] - original image
        :return: if type == 'binary' returns original image and randomly chosen transformation, else it returns
                original image, an image after CutPaste transformation and an image after CutPaste-Scar transformation
        '''
        if self.type == 'binary':
            aug = random.choice([self.cutpaste, self.cutpaste_scar])
            return image, aug(image)

        elif self.type == '3way':
            cutpaste = self.cutpaste(image)
            scar = self.cutpaste_scar(image)
            return image, cutpaste, scar
