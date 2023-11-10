import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

def create_vgg16_feature_extractor():
    vgg16 = models.vgg16(pretrained=True)
    modules = list(vgg16.children())[:-1]
    return nn.Sequential(*modules)


def extract_features(image_loader, model, is_test=False):
    is_training = model.training
    model.eval()
    features_list = []
    labels_list = []
    file_names = []
    with torch.no_grad(), torch.inference_mode():
        for batch in tqdm(image_loader):
            features = model(batch[0]) if is_test else model(batch)
            features = features.view(features.size(0), -1)
            features_list.extend(features.detach().cpu().numpy())
            if is_test:
                labels_list.extend(batch[1].detach().cpu().numpy())
                file_names.extend(batch[2])
    if is_training:
        model.train()
    if is_test:
        return np.array(features_list), np.array(labels_list), np.array(file_names)
    else:
        return np.array(features_list)
