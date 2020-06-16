# utils for data transforms and related utils

import os
import torch
import torch.nn.functional as F
import numpy as np
import copy
from skimage import io, transform # require numpy < 1.16. 1.15 is a good choice
from PIL import Image
from torchvision import transforms


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        assert isinstance(sample, dict)
        image = sample['image']
        if isinstance(image, Image.Image):
            image = np.array(image, dtype=np.float32)
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        new_image = transform.resize(image, (new_h, new_w))
        new_sample = sample.copy()
        new_sample['image'] = new_image
        return new_sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if isinstance(image, Image.Image):
            image = np.array(image, dtype=np.float32)/ 255.0
        elif isinstance(image,np.ndarray) and np.max(image) > 1:
            image = image / 255.0

        new_image = image.transpose((2, 0, 1))
        new_sample = sample.copy()
        new_sample['image'] = torch.from_numpy(new_image)
        new_sample['label'] = torch.from_numpy(np.array(int(label)))
        return new_sample

# utils for augmentation
class RandomHorizontalFlip(object):
    """RandomHorizontalFlip the image.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        assert isinstance(sample, dict)
        image = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray((image).astype('uint8'), 'RGB')
        new_sample = sample.copy()
        new_sample['image'] = transforms.RandomHorizontalFlip()(image)
        return new_sample


class ColorJitter(object):
    """ColorJitter the image.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, brightness=0, contrast=0, hue=0, saturation=0):
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation

    def __call__(self, sample):
        assert isinstance(sample, dict)
        image = sample['image']

        if not isinstance(image, Image.Image):
            image = Image.fromarray((image).astype('uint8'), 'RGB')
        new_sample = sample.copy()
        new_sample['image'] = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, hue=self.hue, saturation=self.saturation)(image)
        return new_sample

class RandomCrop(object):
    """RandomHorizontalFlip the image.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, size, padding=None, padding_mode = 'reflect'):
        self.size = size
        self.padding = padding
        self.padding_mode = padding_mode

    def __call__(self, sample):
        assert isinstance(sample, dict)
        image = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray((image).astype('uint8'), 'RGB')
        new_sample = sample.copy()

        new_sample['image'] = transforms.RandomCrop(size = self.size, padding=self.padding, padding_mode = self.padding_mode)(image)
        return new_sample


