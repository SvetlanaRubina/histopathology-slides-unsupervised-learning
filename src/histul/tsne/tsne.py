import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def tsne(kmeans, features_train, sns=None):
    labels = kmeans.labels_
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features_train)
    df_tsne = pd.DataFrame(features_tsne, columns=['Dim1', 'Dim2'])
    df_tsne['Cluster'] = labels

    return df_tsne