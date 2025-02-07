import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from omegaconf import OmegaConf
import argparse
from configs.logging_config import logger
from data.dataset import get_data_loaders
from models.models import get_model
import time


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    logger.info(f"Checkpoint loaded from {checkpoint_path}, resuming at epoch {start_epoch}")
    return model, optimizer, scheduler, start_epoch


def train_model(model, train_loader, val_loader, device, training_config):

    epochs = training_config.epochs
    lr = training_config.lr
    weight_decay = training_config.weight_decay
    max_patience = training_config.max_patience
    checkpoint_interval = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=2)

    start_epoch = 1
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        latest_checkpoint = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")], reverse=True)
    else:
        latest_checkpoint = []

    if latest_checkpoint:
        model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler,
                                                                   os.path.join(checkpoint_dir, latest_checkpoint[0]),
                                                                   device)
        best_val_loss = float('inf')

    logger.info(f"Starting training from epoch {start_epoch} to {epochs} epochs")
    logger.info(f"Training configuration: lr={lr}, weight_decay={weight_decay}, max_patience={max_patience}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch}/{epochs}")

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()

            batch_end_time = time.time()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)} | Loss={loss.item():.4f} | Processed Images: {(batch_idx + 1) * train_loader.batch_size}/{len(train_loader.dataset)} | Batch Time: {batch_end_time - batch_start_time:.2f} seconds")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        logger.info(f"Epoch {epoch} completed | Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_corrects += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)
        logger.info(f"Validation Results - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        logger.info(
            f"Epoch [{epoch}/{epochs}] completed in {epoch_duration:.2f} seconds | Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f} | Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f} | Learning Rate: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.warning("Early stopping triggered!")
                save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir)
                break

        if epoch % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    torch.save(model.state_dict(), "model_best.pth")
    logger.info("Model saved in model_best.pth")


def train_pizza_classifier(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        logger.info("Loading data...")
        train_loader, val_loader = get_data_loaders(config.data)
        logger.info("Data loaded successfully.")

        logger.info("Initializing model...")
        model = get_model(config.model).to(device)
        logger.info("Model initialized.")

        train_model(model, train_loader, val_loader, device, config.training)

        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    logger.info("Configuration loaded:")
    logger.info(config)

    train_pizza_classifier(config)
