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
    # Custom Dataset for the challenge #

    def __init__(self, csv_file, root_dir, transform=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Images dimension
        self.WIDTH = 1280
        self.HEIGHT = 720

        # Random Erasing?
        self.post_transform = transforms.Compose([
            transforms.ColorJitter(brightness=.7, contrast=.5, saturation=.5, hue=0.05),
            #gaussian_blur, <- The images are not blurry at all :p
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def F_transform(self, image, mask):
        # RandomCrop
        new_w, new_h = 1200, 700
        top = np.random.randint(self.HEIGHT - new_h)
        left = np.random.randint(self.WIDTH - new_w)

        image = transforms.functional.crop(image, top, left, new_h, new_w)
        mask = transforms.functional.crop(mask, top, left, new_h, new_w)

        # Resize
        image = transforms.functional.resize(image, (self.HEIGHT, self.WIDTH), 3)
        mask = transforms.functional.resize(mask, (self.HEIGHT, self.WIDTH), 3)

        # RandomAffine
        if np.random.rand() <= 0.5:
            angle = np.random.randint(-15, 15)
            scale = 0.8 + 0.4 * np.random.rand()

            image = transforms.functional.affine(image, angle, (0,0), scale, 0, resample=2)
            mask = transforms.functional.affine(image, angle, (0,0), scale, 0, resample=2)

        # Horizontal flip
        if np.random.rand() <= 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Post transformation
        image = self.post_transform(image)
        mask = transforms.ToTensor()(mask)

        # Readjust the mask
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return image, mask

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, "images", self.csv.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name)

        mask_name = os.path.join(self.root_dir, "masks", self.csv.iloc[idx, 0] + ".png")
        mask = Image.open(mask_name)

        if self.transform:
            image, mask = self.F_transform(image, mask)
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)

        return image, mask