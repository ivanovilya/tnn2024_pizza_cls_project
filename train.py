import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from omegaconf import OmegaConf
import argparse

from data.dataset import get_data_loaders
from models.models import get_model


def train_model(model, train_loader, val_loader, device, training_config):
    criterion = getattr(nn, training_config.loss_function)()
    optimizer = optim.Adam(model.parameters(), lr=training_config.lr, weight_decay=training_config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=training_config.scheduler_mode,
                                                     factor=training_config.factor,
                                                     patience=training_config.scheduler_patience)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

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

        scheduler.step(val_loss)

        print(f"Epoch [{epoch}/{training_config.epochs}] | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= training_config.max_patience:
                print("Early stopping triggered!")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    torch.save(model.state_dict(), training_config.model_save_path)
    print(f"Model saved in {training_config.model_save_path}")


def train_pizza_classifier(config):
    device = torch.device(config.device)
    print("Using device:", device)

    train_loader, val_loader = get_data_loaders(config.data)
    model = get_model(config.model).to(device)

    train_model(model, train_loader, val_loader, device, config.training)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config', required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    print(config)

    train_pizza_classifier(config)
