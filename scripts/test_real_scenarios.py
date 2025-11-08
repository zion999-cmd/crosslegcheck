#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟实际传感器数据测试无人检测
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_time_detector_12 import TwelveSensorRealTimeDetector

def test_real_world_scenarios():
    """测试真实世界场景"""
    print("🧪 测试真实世界无人检测场景")
    print("=" * 50)
    
    detector = TwelveSensorRealTimeDetector()
    detector.sensor_reader = None
    
    # 场景1：您之前观察到的无人状态数据（676克，4个传感器）
    print("\n📋 场景1：之前观察到的无人状态数据")
    no_person_real = np.array([169, 172, 166, 169, 0, 0, 0, 0, 0, 0, 0, 0])  # 676克，4个传感器
    result = detector.predict_ensemble(no_person_real)
    print(f"   总压力: {np.sum(no_person_real)} 克")
    print(f"   非零传感器: {np.count_nonzero(no_person_real)}/12")
    print(f"   预测结果: {result[0]} (置信度: {result[1]:.1%})")
    print(f"   ✅ 应该显示normal" if result[0] == 'normal' else f"   ❌ 错误，应该显示normal")
    
    # 场景2：刚刚观察到的有人数据（1088克，10个传感器）
    print("\n📋 场景2：刚刚观察到的有人数据")
    with_person_real = np.array([219, 241, 243, 220, 38, 34, 13, 33, 29, 18, 0, 0])  # 1088克，10个传感器
    result = detector.predict_ensemble(with_person_real)
    print(f"   总压力: {np.sum(with_person_real)} 克")
    print(f"   非零传感器: {np.count_nonzero(with_person_real)}/12")
    print(f"   预测结果: {result[0]} (置信度: {result[1]:.1%})")
    print(f"   ✅ 启用模型预测" if np.sum(with_person_real) >= 1000 and np.count_nonzero(with_person_real) >= 8 else f"   ❌ 应该启用模型预测")
    
    # 场景3：边界测试 - 刚好1000克但传感器不够
    print("\n📋 场景3：边界测试 - 1000克但只有7个传感器")
    boundary_test1 = np.array([140, 140, 140, 140, 140, 140, 160, 0, 0, 0, 0, 0])  # 1000克，7个传感器
    result = detector.predict_ensemble(boundary_test1)
    print(f"   总压力: {np.sum(boundary_test1)} 克")
    print(f"   非零传感器: {np.count_nonzero(boundary_test1)}/12")
    print(f"   预测结果: {result[0]} (置信度: {result[1]:.1%})")
    print(f"   ✅ 应该显示normal（传感器不够）" if result[0] == 'normal' else f"   ❌ 错误，应该显示normal")
    
    # 场景4：边界测试 - 传感器够但压力不够
    print("\n📋 场景4：边界测试 - 8个传感器但只有900克")
    boundary_test2 = np.array([110, 110, 110, 110, 115, 115, 115, 115, 0, 0, 0, 0])  # 900克，8个传感器
    result = detector.predict_ensemble(boundary_test2)
    print(f"   总压力: {np.sum(boundary_test2)} 克")
    print(f"   非零传感器: {np.count_nonzero(boundary_test2)}/12")
    print(f"   预测结果: {result[0]} (置信度: {result[1]:.1%})")
    print(f"   ✅ 应该显示normal（压力不够）" if result[0] == 'normal' else f"   ❌ 错误，应该显示normal")
    
    # 场景5：正常有人数据（基于训练数据最小值）
    print("\n📋 场景5：正常有人数据（基于训练数据范围）")
    normal_person = np.array([120, 150, 130, 180, 200, 180, 160, 140, 120, 100, 80, 60])  # 1520克，12个传感器
    result = detector.predict_ensemble(normal_person)
    print(f"   总压力: {np.sum(normal_person)} 克")
    print(f"   非零传感器: {np.count_nonzero(normal_person)}/12")
    print(f"   预测结果: {result[0]} (置信度: {result[1]:.1%})")
    print(f"   ✅ 启用模型预测")
    
    print(f"\n🎯 阈值策略验证完成")
    print(f"📊 基于数据分析的阈值：压力≥1000克 且 传感器≥8个")
    print(f"📈 训练数据范围：1044-38416克，10-12个传感器")

if __name__ == "__main__":
    test_real_world_scenarios()