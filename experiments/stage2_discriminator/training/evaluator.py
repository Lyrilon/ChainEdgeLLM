"""
Discriminator Evaluator
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)


class DiscriminatorEvaluator:
    """判别器评估器"""

    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        """评估模型"""
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in self.test_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(features)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'auc': roc_auc_score(all_labels, all_probs),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }

        logger.info(f"Test Metrics: Acc={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")

        return metrics
