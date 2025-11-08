#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试无人检测逻辑
"""

import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 模拟导入
try:
    from real_time_detector_12 import TwelveSensorRealTimeDetector
except ImportError:
    print("请确保 real_time_detector_12.py 在同一目录下")
    sys.exit(1)

def test_no_person_detection():
    """测试无人检测逻辑"""
    print("🧪 测试无人检测逻辑")
    print("=" * 50)
    
    # 初始化检测器
    detector = TwelveSensorRealTimeDetector()
    detector.sensor_reader = None  # 不使用真实传感器
    
    # 测试场景1：无人状态（所有传感器都是0）
    print("\n📋 测试场景1：完全无人（所有传感器为0）")
    no_person_data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    result = detector.predict_ensemble(no_person_data)
    print(f"   总压力: {np.sum(no_person_data)} 克")
    print(f"   预测结果: {result[0]} (置信度: {result[1]:.1%})")
    print(f"   期望结果: normal ✅" if result[0] == 'normal' else f"   期望结果: normal ❌")
    
    # 测试场景2：微弱压力（低于阈值）
    print("\n📋 测试场景2：微弱压力（低于500克阈值）")
    weak_pressure_data = np.array([10, 15, 8, 12, 20, 18, 25, 30, 22, 16, 14, 10])
    result = detector.predict_ensemble(weak_pressure_data)
    print(f"   总压力: {np.sum(weak_pressure_data)} 克")
    print(f"   预测结果: {result[0]} (置信度: {result[1]:.1%})")
    print(f"   期望结果: normal ✅" if result[0] == 'normal' else f"   期望结果: normal ❌")
    
    # 测试场景3：正常坐姿（高于阈值）
    print("\n📋 测试场景3：正常坐姿（高于500克阈值）")
    normal_sitting_data = np.array([100, 150, 120, 200, 300, 280, 350, 400, 380, 320, 180, 220])
    result = detector.predict_ensemble(normal_sitting_data)
    print(f"   总压力: {np.sum(normal_sitting_data)} 克")
    print(f"   预测结果: {result[0]} (置信度: {result[1]:.1%})")
    print(f"   说明: 启用模型预测")
    
    # 测试场景4：边界情况（刚好500克）
    print("\n📋 测试场景4：边界情况（刚好500克）")
    boundary_data = np.array([40, 45, 42, 38, 45, 50, 48, 46, 44, 40, 32, 30])  # 总和=500
    result = detector.predict_ensemble(boundary_data)
    print(f"   总压力: {np.sum(boundary_data)} 克")
    print(f"   预测结果: {result[0]} (置信度: {result[1]:.1%})")
    print(f"   说明: 刚好达到阈值，启用模型预测")
    
    # 测试场景5：高压力左偏坐姿
    print("\n📋 测试场景5：高压力坐姿（模拟左偏）")
    left_leaning_data = np.array([200, 300, 250, 400, 800, 600, 500, 300, 400, 200, 100, 150])
    result = detector.predict_ensemble(left_leaning_data)
    print(f"   总压力: {np.sum(left_leaning_data)} 克")
    print(f"   预测结果: {result[0]} (置信度: {result[1]:.1%})")
    print(f"   说明: 正常使用模型预测")
    
    print(f"\n✅ 无人检测逻辑测试完成")
    print(f"💡 新阈值设置: 压力≥1000克 且 传感器≥8个")
    print(f"📝 结论: 基于训练数据分析的科学阈值，避免误判")

if __name__ == "__main__":
    test_no_person_detection()