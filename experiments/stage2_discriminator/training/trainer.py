"""
Discriminator Trainer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class DiscriminatorTrainer:
    """判别器训练器"""

    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # 类别权重处理数据不平衡
        if 'class_weights' in config and config['class_weights']:
            class_weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay'])
        )

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = total_loss / len(self.val_loader)
        val_acc = correct / total
        return val_loss, val_acc

    def train(self, epochs, save_dir):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, 'best_model.pt')

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Acc: {val_acc:.4f}")

            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"保存最佳模型到 {best_model_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # 加载最佳模型
        self.model.load_state_dict(torch.load(best_model_path))
        return self.history

