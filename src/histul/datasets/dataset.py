import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class HistopathologyDataset(Dataset):
    def __init__(self, root_folder, transform=None, is_test=False):
        self.root_folder = root_folder
        self.transform = transform
        self.is_test = is_test

        self.image_paths = []
        self.file_names = []
        self.labels = []

        if not is_test:
            for filename in os.listdir(root_folder):
                img_path = os.path.join(root_folder, filename)
                self.image_paths.append(img_path)
                self.file_names.append(filename)  # Store file names
        else:
            class_folders = os.listdir(root_folder)
            for class_folder in class_folders:
                class_path = os.path.join(root_folder, class_folder)
                if class_folder == 'class_a':
                    label = 0
                elif class_folder == 'class_b':
                    label = 1
                else:
                    continue

                for filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, filename)
                    self.image_paths.append(img_path)
                    self.file_names.append(filename)  # Store file names
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            label = self.labels[idx]
            file_name = self.file_names[idx]
            return image, label, file_name
        else:
            file_name = self.file_names[idx]
            return image
