import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

ID_to_DIRNAME={
    'I': 'imagenet',
    'A': 'imagenet-a',
    'A_sub': 'imagenet-a_subset',
    'K': 'sketch',
    'K_sub': 'sketch_subset',
    'R': 'imagenet-r',
    'R_sub': 'imagenet-r_subset',
    'V': 'imagenetv2-matched-frequency-format-val',
    'V_sub': 'imagenetv2-matched-frequency-format-val_subset',
    'flower102': 'Flower102',
    'flower102_sub': 'Flower102',
    'dtd': 'DTD',
    'dtd_sub': 'DTD',
    'pets': 'OxfordPets',
    'pets_sub': 'OxfordPets',
    'cars': 'StanfordCars',
    'cars_sub': 'StanfordCars',
    'ucf101': 'UCF101',
    'ucf101_sub': 'UCF101',
    'caltech101': 'Caltech101',
    'caltech101_sub': 'Caltech101',
    'food101': 'Food101',
    'food101_sub': 'Food101',
    'sun397': 'SUN397',
    'sun397_sub': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'aircraft_sub': 'fgvc_aircraft',
    'eurosat': 'eurosat',
    'eurosat_sub': 'eurosat',
}

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, num_classes=None):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)

        if num_classes is not None:
            # build the appropriate subset
            idx = [i for i in range(len(testset)) if testset.imgs[i][1] < num_classes]
            testset = Subset(testset, idx)

    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in list(path_dict.keys()) + ['aircraft']:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    else:
        raise NotImplementedError
        
    return testset


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views


