import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_tensor(inp, title=None):
    inp = inp.detach().cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    fig = plt.figure(figsize=(5, 3), dpi=300)
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title, fontsize=5)
    plt.pause(0.001)

def plot_confusion_mat(preds, targets):
    confusion_mat = confusion_matrix(preds, targets)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_tsne(df_tsne):
    plt.figure()
    sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', data=df_tsne, palette='viridis', s=50)
    plt.title('t-SNE Visualization of ResNet18 Features with K-Means Clustering')
    plt.show()
