from datetime import timedelta
from metrics import *
from tqdm import tqdm

import pandas as pd
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

def train_epoch(model, loader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    running_loss = 0
    for imgs, masks in tqdm(loader):
        imgs = imgs.cuda() if torch.cuda.is_available() else imgs
        masks = masks.cuda() if torch.cuda.is_available() else masks
        preds = model(imgs)
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def validate_epoch(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0
    val_iou = 0
    metrics_accumulator = {'precision': 0, 'recall': 0, 'accuracy': 0, 'dice': 0, 'jaccard': 0}

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            val_loss += combined_loss(preds, masks).item()
            val_iou += iou_score(preds, masks)

            metrics = compute_metrics(preds, masks)
            for k in metrics_accumulator:
                metrics_accumulator[k] += metrics[k]

    avg_metrics = {k: v / len(loader) for k, v in metrics_accumulator.items()}
    avg_metrics['loss'] = val_loss / len(loader)
    avg_metrics['iou'] = val_iou / len(loader)

    return avg_metrics

def train_and_cross_test(model_class, all_data_loaders, filter = 16, num_epochs = 10, 
                         lr = 1e-4, optimizer_name = "Adam", patience = 10, 
                         image_size = 352, save_path_prefix = "best_model",
                         save_dir="saved_models"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_results = []
    session_results = []
    os.makedirs(save_dir, exist_ok=True)

    for i, train_val_data in enumerate(all_data_loaders):
        print(f"\nTraining on: {train_val_data['name']}")
        start_time = time.time()

        model = model_class().to(device)

        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        train_loader = train_val_data['train_loader']
        val_loader = train_val_data['val_loader']
        train_size = len(train_loader.dataset)

        best_val_metric = 0
        best_epoch = -1
        best_model_state = None
        epochs_no_improve = 0
        best_model_path = f"{save_dir}/best_model_{train_val_data['name'].replace(' ', '_')}.pth"

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss = train_epoch(model, train_loader, optimizer)
            val_metrics = validate_epoch(model, val_loader)

            val_score = val_metrics['iou']
            if val_score > best_val_metric:
                best_val_metric = val_score
                best_epoch = epoch + 1
                epochs_no_improve = 0
                best_model_state = {
                    'model_state_dict': model.state_dict(),
                    'epoch': best_epoch
                }
                torch.save(best_model_state, best_model_path)
                print(f"New best model saved")
            else:
                epochs_no_improve += 1
                print(f"Patience counter: {epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    print(f"Early stopping: No improvement for {patience} epochs.")
                    break

            print(f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"IoU: {val_metrics['iou']:.4f} | "
                  f"Dice: {val_metrics['dice']:.4f} | "
                  f"Jaccard: {val_metrics['jaccard']:.4f} | "
                  f"Precision: {val_metrics['precision']:.4f} | "
                  f"Recall: {val_metrics['recall']:.4f} | "
                  f"Accuracy: {val_metrics['accuracy']:.4f}")

        if best_model_state:
            torch.save(best_model_state, 
                       f"{save_dir}/{save_path_prefix}_{train_val_data['name']}_{num_epochs}_{lr}_{optimizer_name}_{image_size}_{filter}_{train_size}.pth")

        training_time = str(timedelta(seconds=int(time.time() - start_time)))

        print(f"\nTesting model trained on {train_val_data['name']} on all datasets...")

        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        test_results = []

        for j, test_data in enumerate(all_data_loaders[:-1]):
            test_loader = test_data['test_loader']
            test_metrics = validate_epoch(model, test_loader)

            result = {
                "Trained On": train_val_data['name'],
                "Tested On": test_data['name'],
                "Train Size": train_size,
                "Best Epoch": checkpoint['epoch'],
                "Filter Size": filter,
                "Optimizer": optimizer_name,
                "Dice": test_metrics['dice'],
                "Jaccard": test_metrics['jaccard'],
                "Precision": test_metrics['precision'],
                "Recall": test_metrics['recall'],
                "Accuracy": test_metrics['accuracy'],
                "Training Time": training_time,
                "Image Size": image_size
            }

            test_results.append(result)
            final_results.append(result)
            session_results.append(result)

        df = pd.DataFrame(test_results)
        df_with_avg = pd.concat([df, pd.DataFrame([{
            "Tested On": "Average",
            "Dice": df["Dice"].mean(),
            "Jaccard": df["Jaccard"].mean(),
            "Precision": df["Precision"].mean(),
            "Recall": df["Recall"].mean(),
            "Accuracy": df["Accuracy"].mean()
        }])], ignore_index=True)

        print("\nTest Results Across Datasets (with Average):")
        print(df_with_avg[["Tested On", "Dice", "Jaccard", "Precision", "Recall", "Accuracy"]].to_markdown(index=False))

        result_csv_path = f"{save_dir}/results_{train_val_data['name'].replace(' ', '_')}.csv"
        if os.path.exists(result_csv_path):
            existing = pd.read_csv(result_csv_path)
            existing = existing.dropna(axis=1, how='all')
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(result_csv_path, index=False)

    print("\nFinal Cross-Dataset Testing Summary:")
    all_df_path = f"{save_dir}/cross_dataset_summary.csv"
    if os.path.exists(all_df_path):
        all_existing = pd.read_csv(all_df_path)
        all_existing = all_existing.dropna(axis = 1, how = 'all')
        final_df = pd.concat([all_existing, pd.DataFrame(session_results)], ignore_index=True)
    else:
        final_df = pd.DataFrame(session_results)

    print(final_df.tail(20).to_markdown(index=False))
    final_df.to_csv(all_df_path, index=False)