from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets import HistopathologyDataset
from training.train_test import train_test


def main():
    train_dataset_path = '/Users/svetlana_rubina/Documents/Pangea_BioMed/data/home_assignment/ver1/train'
    test_dataset_path = '/Users/svetlana_rubina/Documents/Pangea_BioMed/data/home_assignment/ver1/test'

    means = [0.485, 0.456, 0.406]  # for ImageNet only!
    stds = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    train_dataset = HistopathologyDataset(root_folder=train_dataset_path, transform=data_transforms)
    test_dataset = HistopathologyDataset(root_folder=test_dataset_path, transform=data_transforms, is_test=True)

    batch_size = 16

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    train_test(train_loader, test_loader, 2)


if __name__ == "__main__":
    main()
