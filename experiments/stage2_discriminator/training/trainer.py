"""
Discriminator Trainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
from tqdm import tqdm
from models.discriminator import SupConLoss

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss：专门处理类别不平衡，对难分样本加大惩罚"""

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class DiscriminatorTrainer:
    """判别器训练器"""

    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # 类别权重处理数据不平衡
        class_weights = None
        if 'class_weights' in config and config['class_weights']:
            class_weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(device)

        # 损失函数选择
        label_smoothing = config.get('label_smoothing', 0.0)
        loss_type = config.get('loss', 'cross_entropy')
        self.loss_type = loss_type
        self.supcon_weight = config.get('supcon_weight', 0.5)  # CE 和 SupCon 的混合比例

        if loss_type == 'focal':
            self.criterion = FocalLoss(gamma=2.0, weight=class_weights)
            self.supcon_criterion = None
        elif loss_type == 'supcon_ce':
            # 联合损失：CE + Supervised Contrastive
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=label_smoothing
            ) if class_weights is not None else nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.supcon_criterion = SupConLoss(temperature=config.get('supcon_temperature', 0.07))
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            ) if class_weights is not None else nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.supcon_criterion = None
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay'])
        )

        # Warmup + CosineAnnealing scheduler
        warmup_epochs = config.get('warmup_epochs', 3)
        total_epochs = config.get('epochs', 50)
        self.warmup_epochs = warmup_epochs

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.grad_clip = config.get('grad_clip', 1.0)

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, epoch, total_epochs):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            ce_loss = self.criterion(outputs, labels)

            if self.supcon_criterion is not None and hasattr(self.model, 'project'):
                proj = self.model.project(features)
                sc_loss = self.supcon_criterion(proj, labels)
                loss = (1 - self.supcon_weight) * ce_loss + self.supcon_weight * sc_loss
            else:
                loss = ce_loss

            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        train_loss = total_loss / len(self.train_loader)
        train_acc = correct / total
        return train_loss, train_acc

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

        epoch_pbar = tqdm(range(epochs), desc="  Overall", position=0, dynamic_ncols=True)
        for epoch in epoch_pbar:
            train_loss, train_acc = self.train_epoch(epoch, epochs)
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            lr_now = self.optimizer.param_groups[0]['lr']
            epoch_pbar.set_postfix(
                tLoss=f"{train_loss:.4f}", tAcc=f"{train_acc:.4f}",
                vLoss=f"{val_loss:.4f}", vAcc=f"{val_acc:.4f}",
                lr=f"{lr_now:.2e}"
            )
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {lr_now:.2e}")

            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # 加载最佳模型
        self.model.load_state_dict(torch.load(best_model_path))
        return self.history

