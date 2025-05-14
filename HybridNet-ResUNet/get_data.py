from glob import glob
from torch.utils.data import DataLoader, ConcatDataset

from prepare_data_loaders import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil


def get_all_dataset_paths(base_path, dataset_names):

    all_dataset_paths = []

    for dataset in dataset_names:
        dataset_path = os.path.join(base_path, dataset)
        paths = {
            'name': dataset,
            'train': {
                'images': os.path.join(dataset_path, 'train', 'images'),
                'masks': os.path.join(dataset_path, 'train', 'masks')
            },
            'val': {
                'images': os.path.join(dataset_path, 'validation', 'images'),
                'masks': os.path.join(dataset_path, 'validation', 'masks')
            },
            'test': {
                'images': os.path.join(dataset_path, 'test', 'images'),
                'masks': os.path.join(dataset_path, 'test', 'masks')
            }
        }
        all_dataset_paths.append(paths)

    for dataset in all_dataset_paths:
        print(f"\nDATASET: {dataset['name']}")
        for split in ['train', 'val', 'test']:
            print(f"{split.upper()} IMAGES PATH: {dataset[split]['images']}")
            print(f"{split.upper()} MASKS PATH:  {dataset[split]['masks']}")

    return all_dataset_paths

def preview_samples(all_dataset_paths, splits):

    for dataset in all_dataset_paths:
        for split in splits:
            print(f"\nDATASET: {dataset['name']}")

            image_dir = dataset[split]['images']
            mask_dir = dataset[split]['masks']

            image_files = sorted(os.listdir(image_dir))
            mask_files = sorted(os.listdir(mask_dir))

            sample_idx = min(30, len(image_files) - 1)

            image_path = os.path.join(image_dir, image_files[sample_idx])
            mask_path = os.path.join(mask_dir, mask_files[sample_idx])

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            plt.figure(figsize=(10, 4))
            plt.suptitle(f"{split.upper()} SET SAMPLE", fontsize=14)

            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.title("Mask")
            plt.axis("off")

            plt.show()