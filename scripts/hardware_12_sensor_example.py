#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12路传感器直接使用示例
展示如何在实际硬件中直接使用12路压力传感器数据
"""

import numpy as np
import sys
import os

# 添加脚本目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_time_detector_12_direct import DirectTwelveSensorDetector

def simulate_hardware_reading():
    """
    模拟硬件读取12路传感器数据
    在实际使用中，这个函数应该被替换为真实的硬件读取代码
    """
    # 模拟12路传感器的实际读数（单位：克）
    # 传感器位置对应原来256传感器阵列中的关键位置：
    # [48, 80, 112, 176, 87, 103, 88, 89, 104, 105, 91, 107]
    
    scenarios = {
        "无人": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "正常坐姿": [120, 150, 180, 200, 350, 400, 380, 390, 420, 430, 350, 380],
        "左倾": [200, 180, 160, 100, 450, 400, 420, 380, 350, 320, 400, 350],
        "右倾": [80, 120, 160, 250, 280, 350, 300, 320, 480, 500, 300, 450]
    }
    
    return scenarios

def direct_hardware_interface_example():
    """
    实际硬件接口示例
    这个函数展示了如何在实际硬件中使用检测器
    """
    print("🔧 12路传感器直接使用示例")
    print("=" * 50)
    
    # 创建检测器实例
    detector = DirectTwelveSensorDetector(infinite_mode=False)
    
    # 模拟硬件读取的数据
    scenarios = simulate_hardware_reading()
    
    print("\n📊 测试不同坐姿场景:")
    
    for scenario_name, sensor_readings in scenarios.items():
        print(f"\n🧪 测试场景: {scenario_name}")
        print(f"📡 传感器读数: {sensor_readings}")
        
        # 直接使用12路传感器数据进行预测
        result = detector.predict_ensemble(sensor_readings)
        
        # 显示结果
        print(f"🎯 预测结果: {result['prediction']}")
        print(f"📈 置信度: {result['confidence']:.1%}")
        print(f"⚖️ 总压力: {result['total_pressure']:.0f}g")
        print(f"🔢 活跃传感器: {result['nonzero_sensors']}/12")
        print(f"👤 无人检测: {'是' if result['is_no_person'] else '否'}")
        
        # 显示模型详情
        lr_label, lr_conf, lr_probs = result['lr_result']
        rf_label, rf_conf, rf_probs = result['rf_result']
        print(f"🧠 LR模型: {lr_label} ({lr_conf:.1%})")
        print(f"🌲 RF模型: {rf_label} ({rf_conf:.1%})")
        print("-" * 40)

def real_hardware_template():
    """
    真实硬件使用模板
    这是一个模板函数，展示在真实硬件中如何使用
    """
    print("""
🔧 真实硬件使用模板:

# 1. 初始化检测器
detector = DirectTwelveSensorDetector()

# 2. 在主循环中读取传感器数据
while True:
    # 读取12路传感器数据 (替换为实际硬件读取代码)
    sensor_data_12 = read_12_sensors_from_hardware()
    
    # 进行预测
    result = detector.predict_ensemble(sensor_data_12)
    
    # 处理结果
    if result['is_no_person']:
        print("无人坐着")
    else:
        posture = result['prediction']
        confidence = result['confidence']
        print(f"坐姿: {posture}, 置信度: {confidence:.1%}")
    
    # 控制检测频率
    time.sleep(0.5)  # 每0.5秒检测一次

⚠️  重要提醒:
1. 12路传感器的位置必须对应训练时的位置
2. 传感器索引对应: [48, 80, 112, 176, 87, 103, 88, 89, 104, 105, 91, 107]
3. 这些是16x16传感器阵列中的位置索引，不是物理坐标
4. 如果你的传感器排列不同，需要重新训练模型
""")

def sensor_mapping_info():
    """显示传感器映射信息"""
    print("📍 12路传感器位置映射信息")
    print("=" * 50)
    
    # 传感器在16x16阵列中的索引
    sensor_indices = [48, 80, 112, 176, 87, 103, 88, 89, 104, 105, 91, 107]
    
    print("🗺️  传感器在16x16阵列中的位置:")
    for i, idx in enumerate(sensor_indices):
        row = idx // 16
        col = idx % 16
        print(f"   传感器 {i+1:2d}: 索引 {idx:3d} -> 位置 ({row:2d}, {col:2d})")
    
    print(f"\n📊 总共使用: {len(sensor_indices)} 个传感器")
    print("💡 在实际硬件中，你只需要在这12个位置安装压力传感器")
    print("💡 数据格式: [sensor1, sensor2, ..., sensor12] (共12个数值)")
    print("💡 单位: 克 (g)")
    print("💡 范围: 0-1000g (根据实际传感器规格调整)")

def main():
    """主函数"""
    print("🎯 12路传感器直接使用指南")
    print("=" * 60)
    
    # 显示传感器映射信息
    sensor_mapping_info()
    
    print("\n" + "=" * 60)
    
    # 运行硬件接口示例
    direct_hardware_interface_example()
    
    print("\n" + "=" * 60)
    
    # 显示真实硬件使用模板
    real_hardware_template()

if __name__ == "__main__":
    main()