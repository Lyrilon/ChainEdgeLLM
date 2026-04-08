"""
Ensemble Evaluator for Stage 2 Discriminator
对同一层的多个最佳模型取 soft-voting，提升鲁棒性
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    """对多个模型的 softmax 输出取平均后再预测（soft voting）"""

    def __init__(self, models: list, test_loader, device='cuda'):
        self.models = [m.to(device) for m in models]
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        for m in self.models:
            m.eval()

        all_labels = []
        all_attack_types = []
        all_probs_list = [[] for _ in self.models]

        with torch.no_grad():
            for batch in self.test_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels']
                all_labels.extend(labels.numpy())
                all_attack_types.extend(batch['attack_type'])

                for i, model in enumerate(self.models):
                    outputs = model(features)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    all_probs_list[i].extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        # 软投票：各模型 positive 概率取平均
        ensemble_probs = np.mean([np.array(p) for p in all_probs_list], axis=0)
        all_preds = (ensemble_probs >= 0.5).astype(int)

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
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, ensemble_probs),
            'num_models': len(self.models),
            'error_by_attack_type': error_by_type
        }

        logger.info(f"[Ensemble {len(self.models)} models] Acc={metrics['accuracy']:.4f}, "
                    f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        for attack_type, stats in sorted(error_by_type.items(), key=lambda x: -x[1]['error_rate']):
            logger.info(f"  {attack_type}: {stats['errors']}/{stats['total']} errors ({stats['error_rate']:.2%})")

        return metrics
