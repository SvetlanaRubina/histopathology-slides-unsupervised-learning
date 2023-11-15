import torch


def whiteness_normalization(threshold_whiteness, means, stds):
    threshold_whiteness_normalized = torch.tensor(
        [(threshold_whiteness - mean) / std for mean, std in zip(means, stds)])
    return threshold_whiteness_normalized


def white_space_check(tensor, threshold, threshold_whiteness_normalized, image_size):
    tensor = tensor[0]
    white_mask = (tensor >= threshold_whiteness_normalized.view(-1, 1, 1)).all(dim=0)
    white_pixel_count = torch.sum(white_mask)
    total_pixel_count = image_size[0] * image_size[1]
    white_space_percentage = (white_pixel_count / total_pixel_count / 3) * 100
    return white_space_percentage <= threshold



