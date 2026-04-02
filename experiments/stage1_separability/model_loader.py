"""
Model Loader Module
使用 ModelScope (魔塔) 作为中国大陆下载渠道
"""

import torch
from transformers import AutoModel, AutoTokenizer
from modelscope import snapshot_download
import os
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    模型加载器，支持从 ModelScope (魔塔) 下载和加载模型
    """

    # ModelScope 模型映射表 (更新模型列表)
    MODELSCOPE_MAP = {
        # === 新模型 (推荐) ===
        # Qwen2.5 系列 (阿里云, 支持中英文, 效果优秀)
        "qwen2.5-1.5b-instruct": "qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5-3b-instruct": "qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-7b-instruct": "qwen/Qwen2.5-7B-Instruct",

        # LLaMA 3.x 系列 (Meta, 最新架构)
        "llama3.2-1b": "LLM-Research/Llama-3.2-1B",
        "llama3.2-3b": "LLM-Research/Llama-3.2-3B-Instruct",
        "llama3.1-8b": "LLM-Research/Llama-3.1-8B-Instruct",

        # Gemma 系列 (Google, 轻量化)
        "gemma-2b": "AI-ModelScope/gemma-2b-it",
        "gemma-2b-v2": "AI-ModelScope/gemma-2-2b-it",
        "gemma-4b": "AI-ModelScope/gemma-2-4b-it",

        # Phi-3 系列 (Microsoft, 小模型强性能)
        "phi3-mini": "LLM-Research/Phi-3-mini-4k-instruct",
        "phi3.5-mini": "LLM-Research/Phi-3.5-mini-instruct",

        # === 经典模型 ===
        "gpt2": "AI-ModelScope/gpt2",
        "gpt2-medium": "AI-ModelScope/gpt2-medium",
        "llama2-7b": "modelscope/Llama-2-7b-ms",

        # BERT系列 (用于对比)
        "bert-base-uncased": "AI-ModelScope/bert-base-uncased",
    }

    def __init__(self, model_name: str, cache_dir: str = "./data/cache", device: str = "auto"):
        """
        初始化模型加载器

        Args:
            model_name: 模型名称或 ModelScope ID
            cache_dir: 模型缓存目录
            device: 计算设备 (auto/cpu/cuda)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None

        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)

    def _get_device(self, device: str) -> torch.device:
        """确定计算设备"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _get_modelscope_id(self, model_name: str) -> str:
        """
        获取 ModelScope 模型 ID

        Args:
            model_name: 本地配置名称或 ModelScope ID

        Returns:
            ModelScope 模型 ID
        """
        if model_name in self.MODELSCOPE_MAP:
            return self.MODELSCOPE_MAP[model_name]
        # 如果不在映射表中，假设用户直接提供了 ModelScope ID
        return model_name

    def download_model(self) -> str:
        """
        从 ModelScope 下载模型

        Returns:
            本地模型路径
        """
        model_id = self._get_modelscope_id(self.model_name)
        logger.info(f"正在从 ModelScope 下载模型: {model_id}")
        logger.info(f"缓存目录: {self.cache_dir}")

        try:
            local_path = snapshot_download(
                model_id,
                cache_dir=self.cache_dir,
                revision="master"
            )
            logger.info(f"模型下载完成: {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"从 ModelScope 下载失败: {e}")
            logger.info("尝试从 HuggingFace 加载...")
            return None

    def load(self) -> Tuple[AutoModel, AutoTokenizer]:
        """
        加载模型和 tokenizer

        Returns:
            (model, tokenizer) 元组
        """
        if self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer

        # 尝试从 ModelScope 下载
        local_path = self.download_model()

        model_id = self._get_modelscope_id(self.model_name)

        logger.info(f"正在加载 tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path if local_path else model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        # 设置 pad_token (某些模型如 GPT-2 没有默认 pad_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"正在加载模型: {model_id}")
        self.model = AutoModel.from_pretrained(
            local_path if local_path else model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"模型加载完成，使用设备: {self.device}")
        return self.model, self.tokenizer

    def get_model_info(self) -> dict:
        """获取模型信息"""
        if self.model is None:
            return {}

        config = self.model.config
        info = {
            "model_name": self.model_name,
            "num_layers": getattr(config, 'num_hidden_layers',
                                  getattr(config, 'n_layer', None)),
            "hidden_size": getattr(config, 'hidden_size',
                                   getattr(config, 'n_embd', None)),
            "num_heads": getattr(config, 'num_attention_heads',
                                getattr(config, 'n_head', None)),
            "vocab_size": config.vocab_size,
            "device": str(self.device),
        }
        return info


class HiddenStateExtractor:
    """
    隐藏状态提取器，用于提取 Transformer 中间层输出
    """

    def __init__(self, model: AutoModel, target_layers: list):
        """
        初始化提取器

        Args:
            model: 已加载的 Transformer 模型
            target_layers: 要提取的层索引列表
        """
        self.model = model
        self.target_layers = target_layers
        self.hidden_states = {}
        self.hooks = []

    def _register_hooks(self):
        """注册 forward hook 以捕获中间层输出"""
        def get_hook(layer_idx):
            def hook(module, input, output):
                # 对于 Transformer 层，output 通常是 (hidden_states, ...)
                if isinstance(output, tuple):
                    self.hidden_states[layer_idx] = output[0].detach().cpu()
                else:
                    self.hidden_states[layer_idx] = output.detach().cpu()
            return hook

        # 获取模型的 Transformer 层
        if hasattr(self.model, 'transformer'):
            # GPT-2 风格
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA 风格
            layers = self.model.model.layers
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            # BERT 风格
            layers = self.model.encoder.layer
        else:
            raise ValueError("不支持的模型架构")

        for layer_idx in self.target_layers:
            if layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(get_hook(layer_idx))
                self.hooks.append(hook)

    def remove_hooks(self):
        """移除所有注册的 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        提取隐藏状态

        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码

        Returns:
            隐藏状态字典 {layer_idx: hidden_state}
        """
        self.hidden_states = {}
        self._register_hooks()

        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        self.remove_hooks()
        return self.hidden_states.copy()


def test_model_loader():
    """测试模型加载器"""
    loader = ModelLoader("gpt2", cache_dir="./data/cache")
    model, tokenizer = loader.load()

    print("模型信息:")
    for key, value in loader.get_model_info().items():
        print(f"  {key}: {value}")

    # 测试前向传播
    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    print(f"\n测试输入: '{text}'")
    print(f"输出形状: {outputs.last_hidden_state.shape}")


if __name__ == "__main__":
    test_model_loader()
