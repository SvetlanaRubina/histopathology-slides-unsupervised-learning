import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from model import extract_features, create_vgg16_feature_extractor


def train_test(train_loader, test_loader, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    vgg16_feature_extractor = create_vgg16_feature_extractor()
    features_train = extract_features(train_loader, vgg16_feature_extractor)
    kmeans.fit(features_train)

    features_test, test_labels, file_names = extract_features(test_loader, vgg16_feature_extractor, is_test=True)
    cluster_labels_test = kmeans.predict(features_test)
    cluster_to_class_mapping_test = {cluster: np.argmax(np.bincount(test_labels[cluster_labels_test == cluster]))
                                     for cluster in range(num_clusters)}
    predicted_labels_test = [cluster_to_class_mapping_test[cluster] for cluster in cluster_labels_test]

    accuracy = accuracy_score(test_labels, predicted_labels_test)
    print('Accuracy Score:', accuracy)
    return kmeans
