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
        all_attack_types = []

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
                all_attack_types.extend(batch['attack_type'])

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # 按攻击类型统计错误
        error_by_type = {}
        for attack_type in set(all_attack_types):
            mask = np.array([t == attack_type for t in all_attack_types])
            type_labels = all_labels[mask]
            type_preds = all_preds[mask]
            total = mask.sum()
            errors = (type_labels != type_preds).sum()
            error_by_type[attack_type] = {
                'total': int(total),
                'errors': int(errors),
                'error_rate': float(errors / total) if total > 0 else 0.0
            }

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'auc': roc_auc_score(all_labels, all_probs),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
            'error_by_attack_type': error_by_type
        }

        logger.info(f"Test Metrics: Acc={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        for attack_type, stats in sorted(error_by_type.items(), key=lambda x: -x[1]['error_rate']):
            logger.info(f"  {attack_type}: {stats['errors']}/{stats['total']} errors ({stats['error_rate']:.2%})")

        return metrics
