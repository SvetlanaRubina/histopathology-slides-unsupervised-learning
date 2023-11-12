from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from histul.datasets import HistopathologyDataset
from histul.helpers import seed_everything, save_results
from histul.model import create_resnet18_feature_extractor, create_feature_extractor
from histul.training import train_clust, test_clust
from histul.tsne import tsne


def main():
    seed_everything()
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

    # Choose between vgg16 and resnet18
    arch = "vgg16"
    # arch = "resnet18"
    kmeans, features_train = train_clust(train_loader=train_loader, num_clusters=2,
                         feature_extractor=create_feature_extractor(arch))
    accuracy, test_labels, predicted_labels_test, file_names = test_clust(test_loader=test_loader, kmeans=kmeans, feature_extractor=create_feature_extractor(arch))

    df_tsne = tsne(kmeans=kmeans, features_train=features_train)
    save_results(file_names, predicted_labels_test, "test_predictions")


if __name__ == "__main__":
    main()
