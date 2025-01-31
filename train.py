import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import subprocess
from omegaconf import OmegaConf
import argparse

from data.dataset import get_data_loaders
from models.models import get_model

from datetime import datetime

str_dt = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
tensorboard_logger = SummaryWriter(f"logs/experiment-{str_dt}")


def train_model(model, train_loader, val_loader, device, training_config):
    epochs = training_config.epochs
    lr = training_config.lr
    weight_decay = training_config.weight_decay
    max_patience = training_config.max_patience
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=2)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        all_train_losses = []

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
            all_train_losses.append(loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        all_validation_losses = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_corrects += (preds == labels).sum().item()
                all_validation_losses.append(loss.item())

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        tensorboard_logger.add_histogram("Validation/Loss Distribution", np.array(all_validation_losses), epoch)
        tensorboard_logger.add_histogram("Train/Loss Distribution", np.array(all_train_losses), epoch)
        tensorboard_logger.add_scalar("Train/Loss", epoch_loss , epoch)
        tensorboard_logger.add_scalar("Validation/Loss", val_loss, epoch)
        tensorboard_logger.add_scalar("Train/Accuracy", epoch_acc, epoch)
        tensorboard_logger.add_scalar("Validation/Accuracy", val_acc, epoch)
        tensorboard_logger.add_scalar("Learning Rate", current_lr, epoch)


        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print("Early stopping triggered!")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    torch.save(model.state_dict(), "model_best.pth")
    print("Model saved in model_best.pth")
    # return model


def train_pizza_classifier(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = get_data_loaders(config.data)
    model = get_model(config.model).to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    tensorboard_logger.add_graph(model, dummy_input)

    train_model(model, train_loader, val_loader, device, config.training)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config')
    args = parser.parse_args()
    #args = OmegaConf.from_cli()

    config = OmegaConf.load(args.config)
    print(config)
    subprocess.Popen(["tensorboard", "--logdir", "logs", "--port", "6006"])
    train_pizza_classifier(config)
    tensorboard_logger.close()