#!/usr/bin/env python3
"""
Stage 1 Separability Experiment Runner
TS-ZRV 阶段一可分性实验主运行脚本

使用方法:
    python run_experiment.py --config config.yaml
    python run_experiment.py --quick-test  # 快速测试模式
"""

import argparse
import yaml
import torch
import numpy as np
import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# 导入自定义模块
from model_loader import ModelLoader
from data_generator import (
    DatasetLoader, HonestSampleGenerator, AttackSampleGenerator
)
from similarity_analyzer import (
    CosineSimilarityCalculator, SeparabilityAnalyzer, SimilarityResult
)
from visualizer import ResultVisualizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('experiment.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(config: dict, quick_test: bool = False) -> dict:
    """
    运行完整实验

    Args:
        config: 配置字典
        quick_test: 是否快速测试模式

    Returns:
        实验结果字典
    """
    # 设置随机种子
    set_seed(config['experiment']['seed'])

    # 创建输出目录
    output_dir = config['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 记录实验开始时间
    start_time = datetime.now()
    logger.info(f"实验开始: {start_time}")

    # ========== 步骤1: 加载模型 ==========
    logger.info("=" * 50)
    logger.info("步骤1: 加载模型")
    logger.info("=" * 50)

    model_config = config['model']
    model_loader = ModelLoader(
        model_name=model_config['name'],
        cache_dir=config['experiment']['cache_dir'],
        device=config['experiment']['device']
    )

    model, tokenizer = model_loader.load()
    model_info = model_loader.get_model_info()
    logger.info(f"模型信息: {model_info}")

    # 快速测试模式: 使用更小的模型和数据集
    if quick_test:
        logger.info("【快速测试模式】使用小规模数据")
        target_layers = [model_info['num_layers'] // 2]  # 只测试中间层
        num_samples = 50
        noise_levels = [1.0]
        replay_types = ['cross_sequence_same_layer']
    else:
        target_layers = model_config['target_layers']
        num_samples = config['dataset']['num_samples']
        noise_levels = config['attacks']['random_noise']['noise_levels']
        replay_types = config['attacks']['replay_attack']['replay_types']

    logger.info(f"目标层: {target_layers}")
    logger.info(f"样本数: {num_samples}")

    # ========== 步骤2: 加载数据集 ==========
    logger.info("=" * 50)
    logger.info("步骤2: 加载数据集")
    logger.info("=" * 50)

    dataset_loader = DatasetLoader(
        dataset_name=config['dataset']['name'],
        cache_dir=config['experiment']['cache_dir']
    )

    texts = dataset_loader.load(
        split=config['dataset']['subset'],
        num_samples=num_samples
    )

    logger.info(f"加载了 {len(texts)} 条文本")

    # ========== 步骤3: 生成诚实样本 ==========
    logger.info("=" * 50)
    logger.info("步骤3: 生成诚实样本")
    logger.info("=" * 50)

    honest_generator = HonestSampleGenerator(model, tokenizer, model_loader.device)
    honest_samples = honest_generator.generate(texts, target_layers)

    # 保存诚实样本用于后续攻击生成
    logger.info(f"生成了 {len(honest_samples)} 个诚实样本")

    # ========== 步骤4: 生成攻击样本 ==========
    logger.info("=" * 50)
    logger.info("步骤4: 生成攻击样本")
    logger.info("=" * 50)

    attack_generator = AttackSampleGenerator(
        honest_samples,
        seed=config['experiment']['seed']
    )

    all_samples = honest_samples.copy()
    attack_config = config['attacks']

    # 4.1 随机噪声攻击
    if attack_config['random_noise']['enabled']:
        logger.info("生成随机噪声攻击样本...")
        noise_samples = attack_generator.generate_random_noise(noise_levels)
        all_samples.extend(noise_samples)
        logger.info(f"  - 随机噪声样本: {len(noise_samples)}")

    # 4.2 重放攻击
    if attack_config['replay_attack']['enabled']:
        logger.info("生成重放攻击样本...")
        replay_samples = attack_generator.generate_replay_attacks(
            replay_types,
            pool_size=attack_config['replay_attack']['replay_pool_size']
        )
        all_samples.extend(replay_samples)
        logger.info(f"  - 重放攻击样本: {len(replay_samples)}")

    # 4.3 层跳过攻击 (Stage 1 无法检测，但生成用于对比)
    if attack_config.get('layer_skipping', {}).get('enabled', False):
        logger.info("生成层跳过攻击样本...")
        skip_samples = attack_generator.generate_layer_skipping()
        all_samples.extend(skip_samples)
        logger.info(f"  - 层跳过样本: {len(skip_samples)}")

    # 4.4 精度降级攻击 (Stage 1 无法检测，但生成用于对比)
    if attack_config.get('precision_downgrade', {}).get('enabled', False):
        logger.info("生成精度降级攻击样本...")
        downgrade_samples = attack_generator.generate_precision_downgrade()
        all_samples.extend(downgrade_samples)
        logger.info(f"  - 精度降级样本: {len(downgrade_samples)}")

    logger.info(f"总样本数: {len(all_samples)}")

    # ========== 步骤5: 计算余弦相似度 ==========
    logger.info("=" * 50)
    logger.info("步骤5: 计算余弦相似度")
    logger.info("=" * 50)

    logger.info("计算余弦相似度...")
    similarity_results = CosineSimilarityCalculator.compute_batch(all_samples)

    # 按标签统计
    label_counts = {}
    for r in similarity_results:
        label_counts[r.label] = label_counts.get(r.label, 0) + 1
    logger.info(f"相似度计算完成，标签分布: {label_counts}")

    # ========== 步骤6: 可分性分析 ==========
    logger.info("=" * 50)
    logger.info("步骤6: 可分性分析")
    logger.info("=" * 50)

    analyzer = SeparabilityAnalyzer(similarity_results)

    # 基础统计
    logger.info("基础统计信息:")
    stats = analyzer.compute_statistics()
    for label, stat in stats.items():
        logger.info(f"\n{label}:")
        logger.info(f"  样本数: {stat['count']}")
        logger.info(f"  均值: {stat['mean']:.4f}")
        logger.info(f"  标准差: {stat['std']:.4f}")
        logger.info(f"  范围: [{stat['min']:.4f}, {stat['max']:.4f}]")

    # 分离间隙
    logger.info("\n分离间隙 (Separation Gap):")
    gaps = analyzer.compute_separation_gap()
    for label, gap in gaps.items():
        logger.info(f"  {label}: {gap:.4f}")
        if gap > 0:
            logger.info(f"    -> 存在正分离间隙，可分性良好")
        else:
            logger.info(f"    -> 无分离间隙，可能存在重叠")

    # ROC-AUC
    logger.info("\nROC-AUC:")
    for label in analyzer.attack_scores.keys():
        auc = analyzer.compute_roc_auc(label)
        logger.info(f"  {label}: {auc:.4f}")

    # 最优阈值
    logger.info("\n最优阈值 (F1优化):")
    for label in analyzer.attack_scores.keys():
        optimal = analyzer.find_optimal_threshold(label, metric='f1')
        logger.info(f"  {label}:")
        logger.info(f"    阈值: {optimal['optimal_threshold']:.4f}")
        logger.info(f"    F1: {optimal['metrics_at_optimal']['F1']:.4f}")
        logger.info(f"    TPR: {optimal['metrics_at_optimal']['TPR']:.4f}")
        logger.info(f"    FPR: {optimal['metrics_at_optimal']['FPR']:.4f}")

    # 综合分析
    comprehensive = analyzer.comprehensive_analysis()

    # ========== 步骤7: 可视化 ==========
    logger.info("=" * 50)
    logger.info("步骤7: 生成可视化")
    logger.info("=" * 50)

    visualizer = ResultVisualizer(output_dir=output_dir)
    attack_labels = list(analyzer.attack_scores.keys())

    try:
        visualizer.plot_all(similarity_results, analyzer, attack_labels)
    except Exception as e:
        logger.error(f"可视化生成失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 步骤8: 保存结果 ==========
    logger.info("=" * 50)
    logger.info("步骤8: 保存结果")
    logger.info("=" * 50)

    # 保存分析结果
    result_data = {
        'experiment_info': {
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'model_name': model_config['name'],
            'target_layers': target_layers,
            'num_samples': num_samples,
            'quick_test': quick_test,
        },
        'model_info': model_info,
        'statistics': stats,
        'separation_gaps': gaps,
        'comprehensive_analysis': comprehensive,
    }

    visualizer.generate_report(result_data, 'analysis_report.json')

    # 保存原始相似度结果
    similarity_data = []
    for r in similarity_results:
        similarity_data.append({
            'sample_id': r.sample_id,
            'layer_idx': r.layer_idx,
            'label': r.label,
            'cosine_similarity': r.cosine_similarity,
        })

    similarity_path = os.path.join(output_dir, 'similarity_results.json')
    with open(similarity_path, 'w') as f:
        json.dump(similarity_data, f, indent=2)
    logger.info(f"相似度结果已保存: {similarity_path}")

    # 计算实验时间
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"实验结束: {end_time}")
    logger.info(f"总耗时: {duration}")

    return result_data


def print_summary(results: dict):
    """打印实验摘要"""
    print("\n" + "=" * 60)
    print("实验摘要")
    print("=" * 60)

    info = results['experiment_info']
    print(f"模型: {info['model_name']}")
    print(f"测试层: {info['target_layers']}")
    print(f"样本数: {info['num_samples']}")

    print("\n【关键发现】")

    gaps = results['separation_gaps']
    print("\n1. 分离间隙:")
    for label, gap in gaps.items():
        status = "✓ 可分" if gap > 0.1 else ("⚠ 边界" if gap > 0 else "✗ 重叠")
        print(f"   {label}: {gap:.4f} {status}")

    print("\n2. ROC-AUC:")
    for label, auc in results['comprehensive_analysis']['roc_auc'].items():
        status = "✓ 优秀" if auc > 0.95 else ("⚠ 良好" if auc > 0.8 else "✗ 较差")
        print(f"   {label}: {auc:.4f} {status}")

    print("\n3. 推荐阈值:")
    for label, opt in results['comprehensive_analysis']['optimal_thresholds'].items():
        threshold = opt['optimal_threshold']
        f1 = opt['metrics_at_optimal']['F1']
        print(f"   {label}: τ = {threshold:.4f} (F1={f1:.4f})")

    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)

    # 自动总结
    all_gaps_positive = all(g > 0 for g in gaps.values())
    all_auc_good = all(auc > 0.9 for auc in results['comprehensive_analysis']['roc_auc'].values())

    if all_gaps_positive and all_auc_good:
        print("✓ Stage 1 (Causal Binding Check) 可分性良好!")
        print("  - 余弦相似度能有效区分诚实样本和攻击样本")
        print("  - 可以可靠地检测随机噪声和重放攻击")
    elif all_gaps_positive:
        print("⚠ Stage 1 具有可分性，但部分攻击类型需要更精细的阈值")
    else:
        print("✗ Stage 1 可分性存在问题，可能需要:")
        print("  - 调整阈值策略")
        print("  - 结合 Stage 2 进行更精细的检测")

    print("\n详细结果请查看 results/ 目录")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stage 1 Separability Experiment for TS-ZRV'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='快速测试模式 (使用小规模数据)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录 (覆盖配置文件设置)'
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 覆盖输出目录
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir

    logger.info(f"配置加载完成: {args.config}")
    logger.info(f"输出目录: {config['experiment']['output_dir']}")

    # 运行实验
    try:
        results = run_experiment(config, quick_test=args.quick_test)
        print_summary(results)
        logger.info("实验完成!")
    except Exception as e:
        logger.error(f"实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
