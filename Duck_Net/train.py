import math
import os
from tqdm import tqdm
import torch

from model import DuckNet
from utils.dataloader import get_dataloader
from utils.losses import criterion
from utils.metrics import dice_coef, jaccard_index, precision, recall, accuracy, iou
from utils.config import Configs


def train_epoch(model:DuckNet, train_loader, optimizer:torch.optim.Optimizer, device:torch.device):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader):
        images, masks = batch['image'], batch['mask']
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(masks, outputs)
        loss.backward()
        
        optimizer.step()

        train_loss += loss.item()
    return train_loss / len(train_loader)


@torch.no_grad()
def valid_epoch(model:DuckNet, valid_loader, device:torch.device):
    model.eval()
    valid_loss = 0
    dice = 0
    jaccard = 0
    prec = 0
    rec = 0
    acc = 0
    IoU = 0
    for batch in valid_loader:
        images, masks = batch['image'], batch['mask']
        images, masks = images.to(device), masks.to(device)
        
        outputs = model(images)
        
        loss = criterion(masks, outputs)
        valid_loss += loss.item()
        
        dice += dice_coef(masks, outputs)
        jaccard += jaccard_index(masks, outputs)
        prec += precision(masks, outputs)
        rec += recall(masks, outputs)
        acc += accuracy(masks, outputs)
        IoU += iou(masks, outputs)
    return valid_loss / len(valid_loader), dice / len(valid_loader), jaccard / len(valid_loader), prec / len(valid_loader), rec / len(valid_loader), acc / len(valid_loader), IoU / len(valid_loader)


def main(config:Configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model = DuckNet(input_channels=config.input_channels, num_classes=config.num_classes, num_filters=config.num_filters)
    model.to(device)
    print('Model created')

    dataset = os.path.join(config.ROOT_DIR, 'data', config.train_dataset, 'images', 'train')

    train_dl = get_dataloader(os.path.join(config.ROOT_DIR, dataset), config.input_channels, config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_dl = get_dataloader(os.path.join(config.ROOT_DIR, dataset), config.input_channels, config.batch_size, shuffle=False, num_workers=config.num_workers)
    print('DataLoaders created')

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    os.makedirs(os.path.join(config.ROOT_DIR, config.save_dir, config.train_dataset), exist_ok=True)

    best_valid_loss = math.inf
    early_stop_patience = config.early_stop_patience
    print('Starting training')
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_dl, optimizer, device)
        valid_loss, dice, jaccard, prec, rec, acc, iou = valid_epoch(model, valid_dl, device)
        scheduler.step()
        msg = f'Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f} - IoU: {iou:.4f} - Dice: {dice:.4f} - Jaccard: {jaccard:.4f} - Precision: {prec:.4f} - Recall: {rec:.4f} - Accuracy: {acc:.4f}'

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(config.ROOT_DIR, config.save_dir, config.train_dataset, 'best_model.pt'))
            msg += ' - Model saved'
            early_stop_patience = 0
        else:
            early_stop_patience += 1
            if early_stop_patience >= config.early_stopping:
                msg += ' - Early stopping'
                break
        print(msg)
    print('Training completed')


if __name__ == '__main__':
    config = Configs()
    config.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    main(config)