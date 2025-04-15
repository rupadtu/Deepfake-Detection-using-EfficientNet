from torch.utils.data import Dataset
import os
from PIL import Image
import torch

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []

        for label, folder in enumerate(["Fake", "Real"]):
            folder_path = os.path.join(root_dir, folder)
            for img in os.listdir(folder_path):
                self.samples.append(os.path.join(folder_path, img))
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)