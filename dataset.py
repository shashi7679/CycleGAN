import torch
from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader
import config
import numpy as np

class HorseZebraDataset(Dataset):
    def __init__(self, horse_path, zebra_path, transform=None):
        self.horse_path = horse_path
        self.zebra_path = zebra_path
        self.transform = transform
        
        self.zebra_images = os.listdir(zebra_path)
        self.horse_images = os.listdir(horse_path)
        self.length_dataset = max(len(self.horse_images), len(self.zebra_images))

        self.len_zebra = len(self.zebra_images)
        self.len_horses = len(self.horse_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        zebra_img = self.zebra_images[index % self.len_zebra]
        horse_img = self.horse_images[index % self.len_horses]

        z_path = os.path.join(self.zebra_path, zebra_img)
        h_path = os.path.join(self.horse_path, horse_img)

        z_img = np.array(Image.open(z_path).convert("RGB"))
        h_img = np.array(Image.open(h_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=z_img, image0=h_img)
            zebra_image = augmentations["image"]
            horse_image = augmentations["image0"]

        return zebra_image, horse_image