import torch
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from model import DuckNet
from utils.config import Configs


def load_model(config:Configs, model_path:str):
    model = DuckNet(config.input_channels, config.num_classes, config.num_filters)
    model.load_state_dict(torch.load(model_path))
    return model


def predict(model:DuckNet, image_path:str, device:torch.device):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    image = np.array(image.resize((352, 352)))
    image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)

    output_image = output.squeeze().cpu().numpy()
    input_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return input_image, output_image


def main(config:Configs, model_path:str, image_path:str, output_path:str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = load_model(config, model_path)
    model.to(device)
    print(f'Model loaded from {model_path}')

    input_image, output_image = predict(model, image_path, device)
    
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title('Input Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(output_image, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f'Prediction saved at {output_path}')


if __name__ == '__main__':
    config = Configs(num_filters=17)
    model_path = 'checkpoints/best_model.pt'
    image_path = '0.jpg'
    output_path = 'output_1.jpg'
    main(config, model_path, image_path, output_path)