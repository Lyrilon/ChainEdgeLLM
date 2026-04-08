"""
方向一：阈值优化
对 layer_21 所有已训练模型，在验证集上搜索最优阈值，汇报测试集结果
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../stage1_separability'))

import torch
import numpy as np
import yaml
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from models.discriminator import (
    DualStreamDiscriminator, GatedDualStreamDiscriminator,
    TripleStreamDiscriminator, StatEnhancedGatedDiscriminator
)
from data.dataset import DiscriminatorDataset
from data.attack_generator import LayerSkippingGenerator, PrecisionDowngradeGenerator, AdversarialPerturbationGenerator
from sample_cache import SampleCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def build_model(arch_name, half_dim):
    configs = {
        'dual_stream_m':    ('dual_stream',         {'hidden_dim': 512, 'dropout': 0.2}),
        'gated_dual_s':     ('gated_dual_stream',   {'hidden_dim': 256, 'dropout': 0.2}),
        'gated_dual_m':     ('gated_dual_stream',   {'hidden_dim': 512, 'dropout': 0.2}),
        'triple_stream_s':  ('triple_stream',       {'hidden_dim': 256, 'dropout': 0.2}),
        'triple_stream_m':  ('triple_stream',       {'hidden_dim': 512, 'dropout': 0.2}),
        'stat_enhanced_s':  ('stat_enhanced_gated', {'hidden_dim': 256, 'dropout': 0.2, 'proj_dim': 128}),
        'stat_enhanced_m':  ('stat_enhanced_gated', {'hidden_dim': 512, 'dropout': 0.2, 'proj_dim': 256}),
    }
    arch_type, kwargs = configs[arch_name]
    if arch_type == 'dual_stream':
        return DualStreamDiscriminator(half_dim, kwargs['hidden_dim'], kwargs['dropout'])
    elif arch_type == 'gated_dual_stream':
        return GatedDualStreamDiscriminator(half_dim, kwargs['hidden_dim'], kwargs['dropout'])
    elif arch_type == 'triple_stream':
        return TripleStreamDiscriminator(half_dim, kwargs['hidden_dim'], kwargs['dropout'])
    elif arch_type == 'stat_enhanced_gated':
        return StatEnhancedGatedDiscriminator(half_dim, kwargs['hidden_dim'], kwargs['dropout'], kwargs['proj_dim'])


def get_probs_and_labels(model, loader, device):
    model.eval()
    all_probs, all_labels, all_attack_types = [], [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch['features'].to(device)
            out = model(feats)
            probs = torch.softmax(out, dim=1)[:, 1]  # P(honest)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch['labels'].numpy())
            all_attack_types.extend(batch['attack_type'])
    return np.array(all_probs), np.array(all_labels), all_attack_types


def evaluate_at_threshold(probs, labels, attack_types, threshold):
    preds = (probs >= threshold).astype(int)
    error_by_type = {}
    for at in set(attack_types):
        mask = np.array([t == at for t in attack_types])
        errs = (labels[mask] != preds[mask]).sum()
        total = mask.sum()
        error_by_type[at] = {'errors': int(errs), 'total': int(total),
                              'error_rate': float(errs / total) if total > 0 else 0.0}
    overall_acc = (preds == labels).mean()
    return overall_acc, error_by_type


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(config['experiment']['seed'])

    # 加载 8000 样本缓存
    cache = SampleCache(cache_dir=config['experiment']['sample_cache']['cache_dir'])
    honest_samples = cache.load(
        model_name=config['model']['name'],
        num_samples=8000,
        target_layers=[7, 14, 21],   # 缓存包含所有层
        seed=config['experiment']['seed'],
        dataset_name=config['dataset']['name']
    )
    assert honest_samples is not None, "需要先运行主实验生成8000样本缓存"
    logger.info(f"加载 {len(honest_samples)} 个诚实样本")

    # 生成攻击样本
    attack_samples = []
    attack_samples.extend(LayerSkippingGenerator(honest_samples).generate())
    attack_samples.extend(PrecisionDowngradeGenerator(honest_samples, [6, 8, 4]).generate())
    attack_samples.extend(AdversarialPerturbationGenerator(honest_samples, [0.1, 0.5, 1.0]).generate())

    all_samples = honest_samples + attack_samples
    layer_samples = [s for s in all_samples if s.layer_idx == 21]

    # 按相同随机种子分割（与训练时一致）
    n = len(layer_samples)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    indices = np.random.permutation(n)
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    val_samples  = [layer_samples[i] for i in val_idx]
    test_samples = [layer_samples[i] for i in test_idx]

    val_dataset  = DiscriminatorDataset(val_samples)
    test_dataset = DiscriminatorDataset(test_samples)
    val_loader   = DataLoader(val_dataset,  batch_size=4096)
    test_loader  = DataLoader(test_dataset, batch_size=4096)

    # 使用 v8（7架构全比）的结果目录
    results_base = './results'
    v8_run = '20260407_203327'
    layer21_dir = os.path.join(results_base, v8_run, 'layer_21')
    logger.info(f"使用实验结果: {v8_run}")

    arch_names = ['dual_stream_m', 'gated_dual_s', 'gated_dual_m',
                  'triple_stream_s', 'triple_stream_m', 'stat_enhanced_s', 'stat_enhanced_m']

    thresholds = np.arange(0.05, 0.55, 0.05)

    print("\n" + "="*70)
    print("Layer 21 阈值搜索结果")
    print("="*70)

    best_results = []

    for arch_name in arch_names:
        model_path = os.path.join(layer21_dir, arch_name, 'best_model.pt')
        if not os.path.exists(model_path):
            logger.warning(f"{arch_name}: 模型文件不存在，跳过")
            continue

        model = build_model(arch_name, config['model']['hidden_dim'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)

        # 验证集搜索最优阈值（最小化 precision_downgrade 错误率）
        val_probs, val_labels, val_types = get_probs_and_labels(model, val_loader, device)

        best_thresh, best_pd_err = 0.5, 1.0
        for thresh in thresholds:
            _, err_by_type = evaluate_at_threshold(val_probs, val_labels, val_types, thresh)
            pd_err = err_by_type.get('precision_downgrade', {}).get('error_rate', 1.0)
            if pd_err < best_pd_err:
                best_pd_err, best_thresh = pd_err, thresh

        # 测试集评估：默认阈值 vs 最优阈值
        test_probs, test_labels, test_types = get_probs_and_labels(model, test_loader, device)

        _, default_errs = evaluate_at_threshold(test_probs, test_labels, test_types, 0.5)
        _, best_errs    = evaluate_at_threshold(test_probs, test_labels, test_types, best_thresh)

        default_pd = default_errs.get('precision_downgrade', {}).get('error_rate', 1.0)
        best_pd    = best_errs.get('precision_downgrade', {}).get('error_rate', 1.0)
        default_honest = default_errs.get('honest', {}).get('error_rate', 0.0)
        best_honest    = best_errs.get('honest', {}).get('error_rate', 0.0)
        default_adv    = default_errs.get('adversarial_perturbation', {}).get('error_rate', 0.0)
        best_adv       = best_errs.get('adversarial_perturbation', {}).get('error_rate', 0.0)
        auc = roc_auc_score(test_labels, test_probs)

        print(f"\n[{arch_name}]  最优阈值={best_thresh:.2f}  AUC={auc:.4f}")
        print(f"  {'指标':<25} {'阈值=0.5':>10} {'最优阈值':>10} {'改善':>10}")
        print(f"  {'precision_downgrade':<25} {default_pd:>9.2%} {best_pd:>9.2%} {default_pd-best_pd:>+9.2%}")
        print(f"  {'adversarial_perturb':<25} {default_adv:>9.2%} {best_adv:>9.2%} {default_adv-best_adv:>+9.2%}")
        print(f"  {'honest (误报率)':<25} {default_honest:>9.2%} {best_honest:>9.2%} {default_honest-best_honest:>+9.2%}")

        best_results.append({
            'arch': arch_name, 'threshold': best_thresh,
            'pd_default': default_pd, 'pd_best': best_pd,
            'adv_best': best_adv, 'honest_best': best_honest, 'auc': auc
        })

    # 汇总表
    print("\n" + "="*70)
    print("汇总：precision_downgrade 错误率（测试集）")
    print("="*70)
    print(f"{'架构':<22} {'阈值':>6} {'默认0.5':>9} {'最优':>9} {'改善':>9} {'honest误报':>10}")
    best_results.sort(key=lambda x: x['pd_best'])
    for r in best_results:
        print(f"{r['arch']:<22} {r['threshold']:>5.2f} "
              f"{r['pd_default']:>8.2%} {r['pd_best']:>8.2%} "
              f"{r['pd_default']-r['pd_best']:>+8.2%} {r['honest_best']:>9.2%}")

    print("\n最佳单模型:", best_results[0]['arch'],
          f"阈值={best_results[0]['threshold']:.2f}",
          f"precision_downgrade={best_results[0]['pd_best']:.2%}")


if __name__ == '__main__':
    main()
