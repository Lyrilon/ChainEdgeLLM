#!/usr/bin/env python3
"""
Test Runner for Stage 1 Separability Experiment
运行所有单元测试
"""

import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Stage 1 Separability Experiment - Unit Tests")
    print("=" * 60)
    print()

    # 发现并运行测试
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')

    if not os.path.exists(start_dir):
        print(f"错误: 测试目录不存在: {start_dir}")
        return False

    suite = loader.discover(start_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(
        verbosity=2,
        failfast=False
    )

    result = runner.run(suite)

    print()
    print("=" * 60)
    print("测试摘要")
    print("=" * 60)
    print(f"运行测试数: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print()

    if result.wasSuccessful():
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败")

    print("=" * 60)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
