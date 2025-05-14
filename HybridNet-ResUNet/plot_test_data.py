from resunetpp_pytorch import *
from hybridnet import *

import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_model_from_checkpoint(model_class, checkpoint_path, device = None, **model_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location = device)

    key = 'model_state_dict'

    if key not in checkpoint:
        checkpoint = {'model_state_dict': checkpoint}

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def preprocess_image(image, image_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image, dtype=torch.float).unsqueeze(0)

def predict_mask(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = (torch.sigmoid(output) > 0.5).float().cpu().squeeze().numpy()
    return pred_mask

def visualize_predictions(dataset_names, model_configs, test_image_dirs, test_mask_dirs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_rows = len(dataset_names)
    n_cols = 2 + len(model_configs) 
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))

    for row_idx, dataset in enumerate(dataset_names):
        img_path = os.path.join(test_image_dirs[dataset], sorted(os.listdir(test_image_dirs[dataset]))[0])
        mask_path = os.path.join(test_mask_dirs[dataset], sorted(os.listdir(test_mask_dirs[dataset]))[0])

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        axs[row_idx, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[row_idx, 0].set_ylabel(dataset, rotation=0, labelpad=50, fontsize=14, va='center')
        axs[row_idx, 0].axis("off")

        axs[row_idx, 1].imshow(mask, cmap='gray')
        axs[row_idx, 1].axis("off")

        for col_idx, (model_key, config) in enumerate(model_configs.items()):
            model_class = config['model_class']
            checkpoint_path = config['checkpoint_path']
            filter_size = config['filter_size']
            image_size = config['image_size']

            model = load_model_from_checkpoint(model_class, checkpoint_path, device, filter=filter_size)

            input_tensor = preprocess_image(image, image_size).to(device)
            pred_mask = predict_mask(model, input_tensor)

            pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))

            axs[row_idx, col_idx + 2].imshow(pred_mask, cmap='gray')
            axs[row_idx, col_idx + 2].axis("off")

    axs[0, 0].set_title("Input Image", fontsize=14)
    axs[0, 1].set_title("Ground Truth", fontsize=14)

    for idx, (model_key, config) in enumerate(model_configs.items()):
        model_title = config.get('title', model_key)
        axs[0, idx + 2].set_title(model_title, fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    dataset_names = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir-SEG"]

    best_model_dir = "Best Models"

    test_image_dirs = {
        "CVC-ClinicDB": "Datasets/CVC-ClinicDB/test/images",
        "CVC-ColonDB": "Datasets/CVC-ColonDB/test/images",
        "ETIS-LaribPolypDB": "Datasets/ETIS-LaribPolypDB/test/images",
        "Kvasir-SEG": "Datasets/Kvasir-SEG/test/images"
    }

    test_mask_dirs = {
        "CVC-ClinicDB": "Datasets/CVC-ClinicDB/test/masks",
        "CVC-ColonDB": "Datasets/CVC-ColonDB/test/masks",
        "ETIS-LaribPolypDB": "Datasets/ETIS-LaribPolypDB/test/masks",
        "Kvasir-SEG": "Datasets/Kvasir-SEG/test/masks"
    }

    model_configs = {
        "ResUNet++_Model1": {
            "model_class": build_resunetplusplus,
            "checkpoint_path": f"{best_model_dir}/best_model_Combined_50_0.0001_Adam_352_32_1748 - 1.pth",
            "filter_size": 32,
            "image_size": 352,
            "title": "ResUNet++ \n(352×352, f=32)"
        },

        "ResUNet++_Model2": {
            "model_class": build_resunetplusplus,
            "checkpoint_path": f"{best_model_dir}/best_model_Combined_50_0.0001_Adam_352_32_5244 - 2.pth",
            "filter_size": 32,
            "image_size": 352,
            "title": "ResUNet++, Increased Train \nSize (352×352, f=32)"
        },

        "HybridNet": {
            "model_class": HybridNet,
            "checkpoint_path": f"{best_model_dir}/best_model_Combined_50_0.0001_Adam_352_32_5244 - 3.pth",
            "filter_size": 32,
            "image_size": 352,
            "title": "HybridNet, Increased Train \nSize (352×352, f=32)"
        },
        
    }

    visualize_predictions(
        dataset_names = dataset_names,
        model_configs = model_configs,
        test_image_dirs = test_image_dirs,
        test_mask_dirs = test_mask_dirs
    )