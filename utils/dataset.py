from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io

class ImageDataset(Dataset):
    # Custom Dataset for the challenge #

    def __init__(self, csv_file, root_dir, transform=None):
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

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, "images", self.csv.iloc[idx, 0] + ".jpg")
        image = io.imread(img_name)

        mask_name = os.path.join(self.root_dir, "masks", self.csv.iloc[idx, 0] + ".png")
        mask = io.imread(mask_name)

        print(self.transform)

        if self.transform:
            image = self.transform(image)

            mask = self.transform(mask)
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0

        return image, mask