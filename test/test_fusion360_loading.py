#!/usr/bin/env python3
"""
测试fusion360数据集加载功能
"""

import sys
import json
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append('.')

from benchmark import BenchmarkConfig, ModelConfig, BenchmarkRunner

def test_fusion360_loading():
    """测试fusion360数据集加载"""
    print("🔍 Testing fusion360 dataset loading...")
    
    # 检查必要的文件是否存在
    fusion360_dir = Path("fusion360dataset")
    reconstruction_dir = fusion360_dir / "reconstruction"
    train_test_json = fusion360_dir / "train_test.json"
    
    print(f"Checking paths:")
    print(f"  - fusion360dataset: {fusion360_dir.exists()}")
    print(f"  - reconstruction: {reconstruction_dir.exists()}")
    print(f"  - train_test.json: {train_test_json.exists()}")
    
    if not train_test_json.exists():
        print("❌ train_test.json not found")
        return
    
    # 查看train_test.json内容
    with open(train_test_json, 'r') as f:
        train_test_data = json.load(f)
    
    print(f"\nDataset splits:")
    for split, samples in train_test_data.items():
        print(f"  - {split}: {len(samples)} samples")
        if len(samples) > 0:
            print(f"    First few: {samples[:3]}")
    
    # 检查reconstruction目录中的文件
    if reconstruction_dir.exists():
        step_files = list(reconstruction_dir.glob("*.step"))
        print(f"\nReconstruction directory:")
        print(f"  - Total STEP files: {len(step_files)}")
        if len(step_files) > 0:
            print(f"  - First few files: {[f.name for f in step_files[:3]]}")
    
    # 测试数据加载
    try:
        config = BenchmarkConfig(
            models=[],  # 不需要模型，只测试数据加载
            dataset_type="fusion360",
            split="test",
            data_path="fusion360dataset",
            train_test_json="fusion360dataset/train_test.json"
        )
        
        # 创建一个临时的runner来测试数据加载
        class TestRunner:
            def __init__(self, config):
                self.config = config
            
            def _load_fusion360_data_index(self):
                # 使用BenchmarkRunner的方法
                runner = BenchmarkRunner.__new__(BenchmarkRunner)
                runner.config = self.config
                return runner._load_fusion360_data_index()
        
        test_runner = TestRunner(config)
        data_index = test_runner._load_fusion360_data_index()
        
        print(f"\n✅ Successfully loaded {len(data_index)} samples")
        if len(data_index) > 0:
            print(f"First sample: {data_index[0]}")
        
    except Exception as e:
        print(f"❌ Error loading fusion360 dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fusion360_loading()
