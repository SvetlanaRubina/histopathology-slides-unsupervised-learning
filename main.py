from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from histul.datasets import HistopathologyDataset
from histul.helpers import seed_everything, save_results
from histul.helpers import whiteness_normalization, white_space_check
from histul.model import create_feature_extractor
from histul.training import train_clust, test_clust, train_prediction
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

    image_size = (224, 224)

    threshold_whiteness = 0.95
    threshold_percentage = 30

    batch_size = 16

    train_dataset = HistopathologyDataset(root_folder=train_dataset_path, transform=data_transforms)
    test_dataset = HistopathologyDataset(root_folder=test_dataset_path, transform=data_transforms, is_test=True)

    filtered_train_dataset = [tensor for tensor in train_dataset
                              if white_space_check(tensor=tensor, threshold=threshold_percentage,
                                                   threshold_whiteness_normalized=whiteness_normalization(
                                                       threshold_whiteness, means, stds), image_size=image_size)]

    train_loader = DataLoader(dataset=filtered_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Choose between vgg16 and resnet18
    arch = "vgg16"
    # arch = "resnet18"
    kmeans, features_train, file_names_train = train_clust(train_loader=train_loader, num_clusters=2,
                                                           feature_extractor=create_feature_extractor(arch))
    accuracy, test_labels, predicted_labels_test, file_names_test = test_clust(test_loader=test_loader, kmeans=kmeans,
                                                                               feature_extractor=create_feature_extractor(
                                                                                   arch))

    df_tsne = tsne(kmeans=kmeans, features_train=features_train)

    predicted_labels_train = train_prediction(kmeans=kmeans, features_train=features_train)

    save_results(file_names_test=file_names_test, predicted_labels_test=predicted_labels_test,
                 output_name_test="test_results.csv", file_names_train=file_names_train,
                 predicted_labels_train=predicted_labels_train, output_name_train="train_results.csv")


if __name__ == "__main__":
    main()
