import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm


def create_vgg16_feature_extractor():
    vgg16 = models.vgg16(pretrained=True)
    modules = list(vgg16.children())[:-1]
    return nn.Sequential(*modules)


def create_resnet18_feature_extractor():
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Identity()
    return resnet18


def create_feature_extractor(name):
    if name == 'vgg16':
        return create_vgg16_feature_extractor()
    if name == 'resnet18':
        return create_resnet18_feature_extractor()
    else:
        print("Choose between vgg16 and resnet18")


def extract_features(image_loader, model, is_test=False):
    is_training = model.training
    model.eval()
    features_list = []
    labels_list = []
    file_names = []
    with torch.no_grad(), torch.inference_mode():
        for batch in tqdm(image_loader):
            features = model(batch[0])
            features = features.view(features.size(0), -1)
            features_list.extend(features.detach().cpu().numpy())
            if is_test:
                labels_list.extend(batch[1].detach().cpu().numpy())
                file_names.extend(batch[2])
            else:
                file_names.extend(batch[1])
    if is_training:
        model.train()
    if is_test:
        return np.array(features_list), np.array(labels_list), np.array(file_names)
    else:
        return np.array(features_list), np.array(file_names)
