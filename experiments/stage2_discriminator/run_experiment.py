"""
Stage 2 Discriminator Experiment
主实验流程
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # 添加当前目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../stage1_separability'))

import yaml
import torch
import numpy as np
import logging
from datetime import datetime
from torch.utils.data import DataLoader, random_split

from model_loader import ModelLoader
from data_generator import HonestSampleGenerator, DatasetLoader
from sample_cache import SampleCache
from data.attack_generator import LayerSkippingGenerator, PrecisionDowngradeGenerator, AdversarialPerturbationGenerator
from data.dataset import DiscriminatorDataset
from models.discriminator import Discriminator, CNNDiscriminator, AttentionDiscriminator
from training.trainer import DiscriminatorTrainer
from training.evaluator import DiscriminatorEvaluator
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    """加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_data(config):
    """准备数据"""
    logger.info("=" * 50)
    logger.info("步骤1: 加载模型和数据")
    logger.info("=" * 50)

    # 加载模型
    model_loader = ModelLoader(
        model_name=config['model']['name'],
        cache_dir=config['experiment']['cache_dir'],
        device=config['training']['device']
    )
    model, tokenizer = model_loader.load()
    model_info = model_loader.get_model_info()

    # 加载数据集
    dataset_loader = DatasetLoader(
        dataset_name=config['dataset']['name'],
        cache_dir=config['experiment']['cache_dir']
    )
    texts = dataset_loader.load(num_samples=config['dataset']['num_samples'])

    # 生成或加载诚实样本
    cache = SampleCache(cache_dir=config['experiment']['sample_cache']['cache_dir'])
    honest_samples = cache.load(
        model_name=config['model']['name'],
        num_samples=config['dataset']['num_samples'],
        target_layers=config['model']['target_layers'],
        seed=config['experiment']['seed'],
        dataset_name=config['dataset']['name']
    )

    if honest_samples is None:
        logger.info("缓存未命中，生成诚实样本...")
        honest_gen = HonestSampleGenerator(model, tokenizer, model_info)
        honest_samples = honest_gen.generate(texts, config['model']['target_layers'])
        cache.save(
            samples=honest_samples,
            model_name=config['model']['name'],
            num_samples=config['dataset']['num_samples'],
            target_layers=config['model']['target_layers'],
            seed=config['experiment']['seed'],
            dataset_name=config['dataset']['name']
        )
    else:
        logger.info(f"从缓存加载了 {len(honest_samples)} 个诚实样本")

    return honest_samples


def generate_attacks(honest_samples, config):
    """生成攻击样本"""
    logger.info("=" * 50)
    logger.info("步骤2: 生成攻击样本")
    logger.info("=" * 50)

    attack_samples = []

    # Layer Skipping
    if config['attacks']['layer_skipping']['enabled']:
        gen = LayerSkippingGenerator(honest_samples)
        attack_samples.extend(gen.generate())

    # Precision Downgrade
    if config['attacks']['precision_downgrade']['enabled']:
        gen = PrecisionDowngradeGenerator(honest_samples, config['attacks']['precision_downgrade']['bit_widths'])
        attack_samples.extend(gen.generate())

    # Adversarial Perturbation
    if config['attacks']['adversarial_perturbation']['enabled']:
        gen = AdversarialPerturbationGenerator(honest_samples, config['attacks']['adversarial_perturbation']['epsilon'])
        attack_samples.extend(gen.generate())

    logger.info(f"总计生成 {len(attack_samples)} 个攻击样本")
    return attack_samples


def generate_experiment_report(config, results, honest_samples, attack_samples, output_dir):
    """生成详细的实验报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': {
            'name': config['model']['name'],
            'hidden_dim': config['model']['hidden_dim'],
            'target_layers': config['model']['target_layers']
        },
        'dataset': {
            'name': config['dataset']['name'],
            'num_samples': config['dataset']['num_samples'],
            'honest_samples': len(honest_samples),
            'attack_samples': len(attack_samples),
            'total_samples': len(honest_samples) + len(attack_samples),
            'class_ratio': f"1:{len(attack_samples)//len(honest_samples)}"
        },
        'training': {
            'epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'class_weights': config['training'].get('class_weights', None),
            'early_stopping_patience': config['training']['early_stopping_patience']
        },
        'attacks': config['attacks'],
        'results': results
    }

    report_path = os.path.join(output_dir, 'experiment_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"实验报告已保存: {report_path}")

    # 生成 Markdown 报告
    _generate_markdown_report(report, output_dir)
    return report


def _generate_markdown_report(report, output_dir):
    """生成人类可读的 Markdown 报告"""
    lines = []
    lines.append(f"# Stage 2 判别器实验报告\n")
    lines.append(f"**时间**: {report['timestamp']}\n")

    # 模型与数据
    lines.append("## 实验配置\n")
    lines.append("### 模型")
    lines.append(f"- 名称: `{report['model']['name']}`")
    lines.append(f"- 隐藏层维度: {report['model']['hidden_dim']}")
    lines.append(f"- 目标层: {report['model']['target_layers']}\n")

    lines.append("### 数据集")
    d = report['dataset']
    lines.append(f"- 数据集: `{d['name']}`")
    lines.append(f"- 诚实样本: {d['honest_samples']}")
    lines.append(f"- 攻击样本: {d['attack_samples']}")
    lines.append(f"- 类别比例: {d['class_ratio']}\n")

    lines.append("### 训练参数")
    t = report['training']
    lines.append(f"- 学习率: {t['learning_rate']}")
    lines.append(f"- 批大小: {t['batch_size']}")
    lines.append(f"- 类别权重: {t['class_weights']}")
    lines.append(f"- Early Stopping: {t['early_stopping_patience']} epochs\n")

    # 模型对比表
    lines.append("## 模型性能对比\n")
    lines.append("| 架构 | 参数量 | Accuracy | Precision | Recall | F1 | AUC |")
    lines.append("|------|--------|----------|-----------|--------|----|-----|")
    for arch_name, result in report['results'].items():
        m = result['metrics']
        lines.append(
            f"| {arch_name} | {result['params']:,} "
            f"| {m['accuracy']:.4f} | {m['precision']:.4f} "
            f"| {m['recall']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} |"
        )
    lines.append("")

    # 各架构攻击类型错误分析
    lines.append("## 攻击类型错误分析\n")
    for arch_name, result in report['results'].items():
        lines.append(f"### {arch_name}\n")
        error_by_type = result['metrics'].get('error_by_attack_type', {})
        lines.append("| 攻击类型 | 总样本 | 错误数 | 错误率 |")
        lines.append("|---------|--------|--------|--------|")
        for attack_type, stats in sorted(error_by_type.items(), key=lambda x: -x[1]['error_rate']):
            lines.append(f"| {attack_type} | {stats['total']} | {stats['errors']} | {stats['error_rate']:.2%} |")
        lines.append("")

    md_path = os.path.join(output_dir, 'experiment_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"Markdown 报告已保存: {md_path}")


def main():
    """主函数"""
    config = load_config()
    np.random.seed(config['experiment']['seed'])
    torch.manual_seed(config['experiment']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['device'] == 'auto' else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['experiment']['output_dir'], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"实验结果将保存到: {output_dir}")

    # 准备数据
    honest_samples = prepare_data(config)
    attack_samples = generate_attacks(honest_samples, config)
    all_samples = honest_samples + attack_samples

    # 创建数据集
    dataset = DiscriminatorDataset(all_samples)
    train_size = int(config['training']['train_ratio'] * len(dataset))
    val_size = int(config['training']['val_ratio'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])

    logger.info(f"数据分割: Train={train_size}, Val={val_size}, Test={test_size}")

    # 训练所有架构
    results = {}
    for arch_config in config['discriminator']['architectures']:
        logger.info("=" * 50)
        logger.info(f"训练架构: {arch_config['name']}")
        logger.info("=" * 50)

        # 根据类型创建模型
        arch_type = arch_config.get('type', 'mlp')
        if arch_type == 'mlp':
            model = Discriminator(config['model']['hidden_dim'], arch_config['hidden_dims'], arch_config['dropout'])
        elif arch_type == 'cnn':
            model = CNNDiscriminator(config['model']['hidden_dim'], arch_config['channels'], arch_config['kernel_size'], arch_config['dropout'])
        elif arch_type == 'attention':
            model = AttentionDiscriminator(config['model']['hidden_dim'], arch_config['num_heads'], arch_config['hidden_dim'], arch_config['dropout'])
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

        logger.info(f"模型参数量: {model.count_parameters():,}")

        trainer = DiscriminatorTrainer(model, train_loader, val_loader, config['training'], device)
        save_dir = os.path.join(output_dir, arch_config['name'])
        history = trainer.train(config['training']['epochs'], save_dir)

        evaluator = DiscriminatorEvaluator(model, test_loader, device)
        metrics = evaluator.evaluate()

        results[arch_config['name']] = {'history': history, 'metrics': metrics, 'params': model.count_parameters()}

    # 生成实验报告
    generate_experiment_report(config, results, honest_samples, attack_samples, output_dir)

    # 保存结果摘要
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"实验完成！结果保存到 {output_dir}")


if __name__ == "__main__":
    main()



