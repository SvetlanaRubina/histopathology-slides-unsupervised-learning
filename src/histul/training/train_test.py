import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from src.histul.model import extract_features


def train_clust(train_loader, num_clusters, feature_extractor):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    features_train = extract_features(train_loader, feature_extractor)
    kmeans.fit(features_train)
    return kmeans

def test_clust(test_loader, kmeans, feature_extractor):
    features_test, test_labels, file_names = extract_features(test_loader, feature_extractor, is_test=True)
    cluster_labels_test = kmeans.predict(features_test)
    cluster_to_class_mapping_test = {cluster: np.argmax(np.bincount(test_labels[cluster_labels_test == cluster]))
                                     for cluster in range(kmeans.n_clusters)}
    predicted_labels_test = [cluster_to_class_mapping_test[cluster] for cluster in cluster_labels_test]

    accuracy = accuracy_score(test_labels, predicted_labels_test)
    print('Accuracy Score:', accuracy)
    return accuracy, test_labels, predicted_labels_test, file_names
