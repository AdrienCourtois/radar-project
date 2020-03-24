from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
from PIL import ImageFilter
import numpy as np

def gaussian_blur(x):
    if np.random.rand() <= 0.1:
        img = x.filter(ImageFilter.GaussianBlur(radius=10*np.random.rand()))
        
        return img
    else:
        return x


class ImageDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=True, height=512, width=512):
        """
        Dataset class used for the project.

        Args:
            img_dir (string): Directory with all the images.
            label_dir (string): Directory with all the labels.
            transform (boolean, optional): Should the transform be applied?
            height (int, optional): Desired height of the images
            width (int, optional): Desired width of the images
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # Images dimension
        self.width = width
        self.height = height

        # Random Erasing?
        self.post_transform = transforms.Compose([
            #transforms.ColorJitter(brightness=.9, contrast=.2, saturation=.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Get the number of images
        self.image_names = list(filter(lambda x: ".png" in x, os.listdir(img_dir)))
        self.n_images = len(self.image_names)
    
    def F_transform(self, image, mask):
        """
        Data augmentation when the parameter `transform` is set to True

        Args:
            image (PIL.Image array): The input image
            mask (PIL.Image array): The input mask
        """

        or_width, or_height = image.size

        # RandomCrop
        if self.height < or_height or self.width < or_width:
            top = np.random.randint(or_height - self.height)
            left = np.random.randint(or_width - self.width)
            
            image = transforms.functional.crop(image, top, left, self.height, self.width)
            mask = transforms.functional.crop(mask, top, left, self.height, self.width)

        # Random rotations
        if np.random.rand() <= 0.5:
            angle = 90

            image = transforms.functional.affine(image, angle, (0,0), 1, 0, resample=2)
            mask = transforms.functional.affine(mask, angle, (0,0), 1, 0, resample=2)

        # Horizontal flip
        if np.random.rand() <= 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        if np.random.rand() <= 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        
        # Post transform
        image = self.post_transform(image)
        mask = transforms.ToTensor()(mask)

        # Readjust the mask
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return image, mask
    
    def F_noTransform(image, mask):
        """
        Data adjustement when the parameter `transform` is set to False.
        e.g modifies the image size by resizing, adjusts the mean and std.

        Args:
            image (PIL.Image array): The input image
            mask (PIL.Image array): The input mask
        """

        or_width, or_height = image.size

        # Resize in case random crop is applied
        if self.height < or_height or self.width < or_width:
            image = transforms.functional.resize(image, (self.height, self.width), interpolation=3)
            mask = transforms.functional.resize(mask, (self.height, self.width), interpolation=3)
        
        # Post transform
        image = self.post_transform(image)
        mask = transforms.ToTensor()(mask)

        # Readjust the mask
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return image, mask

    def __len__(self):
        """
        Overwritten built-in Python method. Returns the number of images
        in the dataset.
        """

        return self.n_images

    def __getitem__(self, idx):
        """
        Overwritten built-in Python method. Returns the image #`idx`, 
        possibly augmented, and its associated mask.

        Args:
            idx (number): The image ID, must be in [0, len(self)-1]
        """

        img_name = os.path.join(self.img_dir, self.image_names[idx])
        image = Image.open(img_name)

        mask_name = "segmentation_" + "_".join(self.image_names[idx].split("_")[1:])
        mask_name = os.path.join(self.label_dir, mask_name)
        mask = Image.open(mask_name)

        if self.transform:
            image, mask = self.F_transform(image, mask)
        else:
            image, mask = self.F_noTransform(image, mask)

        return image, mask