"""
Sample Cache Module
样本缓存管理器 - 避免重复生成样本
"""

import os
import pickle
import hashlib
import json
from typing import List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SampleCache:
    """
    样本缓存管理器

    功能：
    - 根据模型配置和数据集生成唯一缓存标识
    - 保存和加载已生成的样本
    - 自动检测缓存是否过期
    """

    def __init__(self, cache_dir: str = "./data/sample_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _generate_cache_key(self, model_name: str, num_samples: int,
                           target_layers: List[int], seed: int,
                           dataset_name: str) -> str:
        """
        生成缓存标识（基于配置）

        Args:
            model_name: 模型名称
            num_samples: 样本数量
            target_layers: 目标层列表
            seed: 随机种子
            dataset_name: 数据集名称

        Returns:
            缓存键字符串
        """
        config_str = f"{model_name}_{num_samples}_{target_layers}_{seed}_{dataset_name}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"samples_{cache_key}.pkl")

    def _get_metadata_path(self, cache_key: str) -> str:
        """获取元数据文件路径"""
        return os.path.join(self.cache_dir, f"samples_{cache_key}.json")

    def save(self, samples: List, model_name: str, num_samples: int,
             target_layers: List[int], seed: int, dataset_name: str,
             cache_key: Optional[str] = None) -> str:
        """
        保存样本到缓存

        Args:
            samples: 样本列表
            model_name: 模型名称
            num_samples: 样本数量
            target_layers: 目标层列表
            seed: 随机种子
            dataset_name: 数据集名称
            cache_key: 可选，自定义缓存键

        Returns:
            缓存键
        """
        if cache_key == "auto" or cache_key is None:
            cache_key = self._generate_cache_key(
                model_name, num_samples, target_layers, seed, dataset_name
            )

        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        # 保存样本（使用pickle）
        with open(cache_path, 'wb') as f:
            pickle.dump(samples, f)

        # 保存元数据（方便查看）
        metadata = {
            'model_name': model_name,
            'num_samples': num_samples,
            'target_layers': target_layers,
            'seed': seed,
            'dataset_name': dataset_name,
            'cache_key': cache_key,
            'created_at': datetime.now().isoformat(),
            'num_cached_samples': len(samples),
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"样本已缓存: {cache_path}")
        logger.info(f"缓存元数据: {metadata_path}")
        return cache_key

    def load(self, model_name: str, num_samples: int,
             target_layers: List[int], seed: int,
             dataset_name: str,
             cache_key: Optional[str] = None) -> Optional[List]:
        """
        从缓存加载样本

        Args:
            model_name: 模型名称
            num_samples: 样本数量
            target_layers: 目标层列表
            seed: 随机种子
            dataset_name: 数据集名称
            cache_key: 可选，自定义缓存键

        Returns:
            样本列表，如果不存在则返回 None
        """
        if cache_key == "auto" or cache_key is None:
            cache_key = self._generate_cache_key(
                model_name, num_samples, target_layers, seed, dataset_name
            )

        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        # 检查缓存是否存在
        if not os.path.exists(cache_path):
            logger.info(f"缓存不存在: {cache_key}")
            return None

        # 检查元数据是否匹配
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 验证配置是否一致
            if (metadata.get('model_name') == model_name and
                metadata.get('num_samples') == num_samples and
                metadata.get('seed') == seed and
                metadata.get('dataset_name') == dataset_name):

                logger.info(f"找到有效缓存: {cache_key}")
                logger.info(f"  创建于: {metadata.get('created_at')}")
                logger.info(f"  样本数: {metadata.get('num_cached_samples')}")
            else:
                logger.warning(f"缓存配置不匹配: {cache_key}")
                return None
        else:
            logger.warning(f"缓存元数据缺失: {cache_key}")

        # 加载样本
        try:
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            logger.info(f"成功加载缓存样本: {len(samples)} 个")
            return samples
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            return None

    def clear_cache(self, cache_key: Optional[str] = None):
        """
        清除缓存

        Args:
            cache_key: 可选，特定缓存键；为None则清除所有缓存
        """
        if cache_key:
            cache_path = self._get_cache_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)

            for path in [cache_path, metadata_path]:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"已删除: {path}")
        else:
            # 清除所有缓存
            import glob
            for path in glob.glob(os.path.join(self.cache_dir, "samples_*")):
                os.remove(path)
                logger.info(f"已删除: {path}")
            logger.info("所有缓存已清除")

    def list_caches(self) -> List[dict]:
        """
        列出所有可用的缓存

        Returns:
            缓存元数据列表
        """
        import glob

        caches = []
        metadata_files = glob.glob(os.path.join(self.cache_dir, "samples_*.json"))

        for metadata_path in metadata_files:
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    caches.append(metadata)
            except Exception as e:
                logger.warning(f"读取缓存元数据失败: {metadata_path}, {e}")

        return sorted(caches, key=lambda x: x.get('created_at', ''), reverse=True)


def test_cache():
    """测试缓存功能"""
    import numpy as np
    from data_generator import Sample

    cache = SampleCache()

    # 创建测试样本
    samples = []
    for i in range(10):
        sample = Sample(
            sample_id=i,
            layer_idx=4,
            x_prev=np.random.randn(768).astype(np.float32),
            x_curr=np.random.randn(768).astype(np.float32),
            label='honest',
            metadata={'text': f'test_{i}'}
        )
        samples.append(sample)

    # 保存缓存
    cache_key = cache.save(
        samples,
        model_name="gpt2",
        num_samples=100,
        target_layers=[4, 8, 12],
        seed=42,
        dataset_name="wikitext"
    )
    print(f"缓存键: {cache_key}")

    # 加载缓存
    loaded_samples = cache.load(
        model_name="gpt2",
        num_samples=100,
        target_layers=[4, 8, 12],
        seed=42,
        dataset_name="wikitext"
    )
    print(f"加载样本数: {len(loaded_samples) if loaded_samples else 0}")

    # 列出缓存
    print("\n可用缓存:")
    for c in cache.list_caches():
        print(f"  - {c['cache_key']}: {c['model_name']}, {c['num_cached_samples']} samples")


if __name__ == "__main__":
    test_cache()
