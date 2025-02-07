import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from omegaconf import OmegaConf
import argparse
from torch.utils.tensorboard import SummaryWriter  # Импортируем для работы с TensorBoard

from data.dataset import get_data_loaders
from models.models import get_model


def train_model(model, train_loader, val_loader, device, training_config, writer):
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
        train_losses = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_losses.append(loss.item())
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_losses = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_losses.append(loss.item())
                preds = outputs.argmax(dim=1)
                val_corrects += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)

        # Логирование результатов валидации и learning rate в TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Логирование распределения потерь
        writer.add_histogram('Train/Loss Distribution', torch.tensor(train_losses), epoch)
        writer.add_histogram('Validation/Loss Distribution', torch.tensor(val_losses), epoch)

        scheduler.step(val_loss)

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

    # Инициализация TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir='runs')

    train_model(model, train_loader, val_loader, device, config.training, writer)

    # Закрытие writer после завершения тренировки
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    print(config)
    train_pizza_classifier(config)
