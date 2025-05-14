import os
import torch
from tqdm import tqdm

from model import DuckNet
from utils.config import Configs
from utils.dataloader import get_dataloader
from utils.metrics import dice_coef, jaccard_index, precision, recall, accuracy, iou

DATASETS = ['CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir-SEG']

def evaluate_on_dataset(model, dataloader, device):
    model.eval()
    dice = iou = jacc = prec = rec = acc = loss = 0
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            dice += dice_coef(masks, preds)
            iou += iou(masks, preds)
            jacc += jaccard_index(masks, preds)  # Jaccard ≈ IoU unless split
            prec += precision(masks, preds)
            rec += recall(masks, preds)
            acc += accuracy(masks, preds)
            count += 1

    return {
        'Dice': dice.item() / count,
        'IoU': iou.item() / count,
        'Jaccard': jacc.item() / count,
        'Precision': prec.item() / count,
        'Recall': rec.item() / count,
        'Accuracy': acc.item() / count,
    }


def main():
    config = Configs(num_filters=17)
    model_path = os.path.join('checkpoints', config.train_dataset, 'best_model.pt')
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")

    model = DuckNet(config.input_channels, config.num_classes, config.num_filters)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    results = []
    for dataset in DATASETS:
        test_path = os.path.join(config.ROOT_DIR, 'data', dataset, 'test')
        dataloader = get_dataloader(
            dataset_path=test_path,
            input_channels=config.input_channels,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers
        )
        print(f'\n--- Evaluating on {dataset} ---')
        metrics = evaluate_on_dataset(model, dataloader, device)
        results.append((config.train_dataset, dataset, metrics))

    print('\n📊 Final Cross-Dataset Testing Summary:\n')
    print(f"{'Trained On':<20} {'Tested On':<20} {'Dice':<8} {'IoU':<8} {'Jaccard':<8} {'Precision':<10} {'Recall':<10} {'Accuracy':<10}")
    print('-'*90)
    for train_ds, test_ds, m in results:
        print(f"{train_ds:<20} {test_ds:<20} {m['Dice']:.4f}   {m['IoU']:.4f}   {m['Jaccard']:.4f}   {m['Precision']:.4f}   {m['Recall']:.4f}   {m['Accuracy']:.4f}")


if __name__ == '__main__':
    main()