#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析12传感器数据的压力分布，确定合理的无人检测阈值
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_pressure_distribution():
    """分析压力分布"""
    print("📊 分析12传感器数据压力分布")
    print("=" * 50)
    
    # 加载数据
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(script_dir, 'data', 'dataset_12_sensors.csv')
    df = pd.read_csv(data_path)
    
    # 计算每个样本的总压力
    sensor_cols = [col for col in df.columns if col.startswith('Sensor_')]
    df['total_pressure'] = df[sensor_cols].sum(axis=1)
    
    # 统计各类别的压力分布
    print("\n📋 各类别压力统计:")
    for label in df['Label'].unique():
        subset = df[df['Label'] == label]
        pressure_stats = subset['total_pressure'].describe()
        print(f"\n{label.upper()} 类别:")
        print(f"   样本数: {len(subset)}")
        print(f"   最小值: {pressure_stats['min']:.0f} 克")
        print(f"   25%分位: {pressure_stats['25%']:.0f} 克") 
        print(f"   中位数: {pressure_stats['50%']:.0f} 克")
        print(f"   75%分位: {pressure_stats['75%']:.0f} 克")
        print(f"   最大值: {pressure_stats['max']:.0f} 克")
        print(f"   平均值: {pressure_stats['mean']:.0f} 克")
    
    # 找出最小的有效压力值
    min_valid_pressure = df['total_pressure'].min()
    max_valid_pressure = df['total_pressure'].max()
    
    print(f"\n🎯 数据集压力范围:")
    print(f"   最小有效压力: {min_valid_pressure:.0f} 克")
    print(f"   最大有效压力: {max_valid_pressure:.0f} 克")
    
    # 建议阈值
    # 取最小有效压力的80%作为安全阈值
    suggested_threshold = min_valid_pressure * 0.8
    print(f"\n💡 建议阈值:")
    print(f"   当前阈值: 500 克")
    print(f"   建议阈值: {suggested_threshold:.0f} 克 (最小值的80%)")
    
    # 分析低压力区间
    low_pressure_samples = df[df['total_pressure'] < 2000]
    print(f"\n🔍 低压力样本分析 (<2000克):")
    print(f"   样本数: {len(low_pressure_samples)}")
    if len(low_pressure_samples) > 0:
        print(f"   压力范围: {low_pressure_samples['total_pressure'].min():.0f} - {low_pressure_samples['total_pressure'].max():.0f} 克")
        print(f"   标签分布:")
        for label, count in low_pressure_samples['Label'].value_counts().items():
            print(f"     {label}: {count} 个")
    
    # 检查非零传感器数量
    print(f"\n🔍 非零传感器数量分析:")
    df['nonzero_sensors'] = (df[sensor_cols] > 0).sum(axis=1)
    
    for label in df['Label'].unique():
        subset = df[df['Label'] == label]
        nonzero_stats = subset['nonzero_sensors'].describe()
        print(f"\n{label.upper()} 类别非零传感器:")
        print(f"   最少: {nonzero_stats['min']:.0f} 个")
        print(f"   平均: {nonzero_stats['mean']:.1f} 个")
        print(f"   最多: {nonzero_stats['max']:.0f} 个")
    
    # 建议综合阈值策略
    print(f"\n🎯 建议阈值策略:")
    print(f"   方案1 - 单一压力阈值: {suggested_threshold:.0f} 克")
    print(f"   方案2 - 压力+传感器数量: 总压力 > 1500 克 且 非零传感器 >= 8 个")
    print(f"   方案3 - 保守策略: 总压力 > 1000 克")

if __name__ == "__main__":
    analyze_pressure_distribution()

    # ================== 16x16压力阈值表生成与分类 ==================
    import json

    def load_256_dataset(path):
        """读取16x16传感器数据集，假设最后一列为标签"""
        df = pd.read_csv(path, header=None)
        sensor_data = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values.astype(str)
        return sensor_data, labels

    def compute_stats(sensor_data, labels):
        stats = {}
        for label in np.unique(labels):
            data = sensor_data[labels == label]
            stats[label] = {
                'mean': np.mean(data, axis=0).tolist(),
                'std': np.std(data, axis=0).tolist()
            }
        return stats

    def generate_threshold_table(stats):
        threshold_table = {}
        for i in range(len(stats['normal']['mean'])):
            threshold_table[i] = {}
            for label in stats:
                mean = stats[label]['mean'][i]
                std = stats[label]['std'][i]
                threshold_table[i][label] = [mean - std, mean + std]
        return threshold_table

    def save_threshold_table(threshold_table, path):
        with open(path, 'w') as f:
            json.dump(threshold_table, f, indent=2)

    def load_threshold_table(path):
        with open(path, 'r') as f:
            return json.load(f)

    def classify(new_data, threshold_table):
        scores = {label: 0 for label in threshold_table[0].keys()}
        for i, value in enumerate(new_data):
            for label in scores:
                low, high = threshold_table[str(i)][label]
                if low <= value <= high:
                    scores[label] += 1
        return max(scores, key=scores.get)

    def main_256():
        DATASET_PATH = 'data/dataset.csv'
        THRESHOLD_PATH = 'data/pressure_thresholds.json'
        sensor_data, labels = load_256_dataset(DATASET_PATH)
        stats = compute_stats(sensor_data, labels)
        threshold_table = generate_threshold_table(stats)
        save_threshold_table(threshold_table, THRESHOLD_PATH)
        print(f'✅ 16x16压力阈值表已保存到 {THRESHOLD_PATH}')
        # 示例分类
        # new_data = sensor_data[0]
        # threshold_table = load_threshold_table(THRESHOLD_PATH)
        # result = classify(new_data, threshold_table)
        # print('预测类别:', result)

    # 如需运行，取消下行注释
    # main_256()