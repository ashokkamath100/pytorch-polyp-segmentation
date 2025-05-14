import os
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

DEFAULT_IMAGE_SIZE = (352, 352)

class DuckDataset(Dataset):
    def __init__(self, dataset_path:str, input_channels:int, transform=None):
        if transform is None:
            transform = T.Compose([
                T.Resize(DEFAULT_IMAGE_SIZE),
                T.ToTensor()
            ])
        self.data = []
        image_path = os.path.join(dataset_path, 'images')
        mask_path = os.path.join(dataset_path, 'masks')
        image_filenames = os.listdir(image_path)
        for image_filename in tqdm(image_filenames):
            if not image_filename.endswith(('png', 'jpg')):
                continue
            mask_filename = image_filename  # masks are .jpg too

            image = ImageOps.exif_transpose(Image.open(os.path.join(image_path, image_filename)))
            if input_channels == 1:
                image = image.convert('L')
            elif input_channels == 3:
                image = image.convert('RGB')
            else:
                raise ValueError('Input channels must be 1 or 3')

            mask = ImageOps.exif_transpose(Image.open(os.path.join(mask_path, mask_filename))).convert('L')

            self.data.append({
                'image': transform(image),
                'mask': transform(mask)
            })
            if input_channels == 1:
                image = image.convert('L')
            elif input_channels == 3:
                image = image.convert('RGB')
            else:
                raise ValueError('Input channels must be 1 or 3')
            mask = ImageOps.exif_transpose(Image.open(os.path.join(mask_path, mask_filename))).convert('L')
            self.data.append(
                {
                    'image': transform(image),
                    'mask': transform(mask)
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def get_dataloader(dataset_path:str, input_channels:int, batch_size:int, shuffle:bool, transform=None, num_workers:int=4):
    dataset = DuckDataset(dataset_path, input_channels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == '__main__':
    dataset_path = 'data/train'
    batch_size = 4
    shuffle = True
    num_workers = 4
    train_dl = get_dataloader(dataset_path, batch_size, shuffle, num_workers=num_workers)
    for batch in train_dl:
        image, mask = batch['image'], batch['mask']
        image = image.permute(0, 2, 3, 1).numpy()
        mask = mask.permute(0, 2, 3, 1).numpy()
        image_pil = Image.fromarray((image[0] * 255).astype('uint8'))
        mask_pil = Image.fromarray((mask[0] * 255).astype('uint8').squeeze())
        image_pil.save('image.jpg')
        mask_pil.save('mask.png')
        print(batch['image'].shape, batch['mask'].shape)
        break