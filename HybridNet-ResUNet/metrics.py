import torch
import torch.nn as nn

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

bce_loss = nn.BCEWithLogitsLoss()

def combined_loss(pred, target):
    return bce_loss(pred, target) + dice_loss(pred, target)

def iou_score(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds) > threshold
    masks = masks > 0.5
    intersection = (preds & masks).float().sum((1, 2, 3))
    union = (preds | masks).float().sum((1, 2, 3))
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()

def compute_metrics(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds) > threshold
    masks = masks > 0.5

    preds = preds.view(-1)
    masks = masks.view(-1)

    TP = (preds & masks).sum().float()
    FP = (preds & ~masks).sum().float()
    FN = (~preds & masks).sum().float()
    TN = (~preds & ~masks).sum().float()

    epsilon = 1e-6
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    dice = 2 * TP / (2 * TP + FP + FN + epsilon)
    jaccard = TP / (TP + FP + FN + epsilon)

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'accuracy': accuracy.item(),
        'dice': dice.item(),
        'jaccard': jaccard.item()
    }