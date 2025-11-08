#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分布分析脚本
分析训练数据与测试数据的差异，找出13条数据表现不佳的原因
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

def pressure_to_image(pressure_data):
    """将256维压力数据转换为16x16图像"""
    image = pressure_data.reshape(16, 16).astype(np.float32)
    
    # 归一化到0-1范围
    if image.max() > 0:
        image = image / image.max()
    
    return image

def load_data(csv_file, label=None):
    """加载数据"""
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except:
        df = pd.read_csv(csv_file, encoding='gbk')
    
    if 'Label' in df.columns:
        labels = df['Label'].values
        features = df.drop('Label', axis=1).values
    else:
        labels = [label] * len(df) if label else ['unknown'] * len(df)
        features = df.values
    
    return features, labels

def calculate_statistics(features, labels, name):
    """计算数据统计信息"""
    print(f"\n📊 {name} 数据统计:")
    print(f"   - 样本数: {len(features)}")
    print(f"   - 特征维度: {features.shape[1]}")
    
    # 基本统计
    print(f"   - 数值范围: [{np.min(features):.2f}, {np.max(features):.2f}]")
    print(f"   - 均值: {np.mean(features):.2f}")
    print(f"   - 标准差: {np.std(features):.2f}")
    print(f"   - 中位数: {np.median(features):.2f}")
    
    # 零值统计
    zero_ratio = np.sum(features == 0) / features.size
    print(f"   - 零值比例: {zero_ratio:.3f} ({zero_ratio*100:.1f}%)")
    
    # 类别分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"   - 类别分布:")
    for label, count in zip(unique_labels, counts):
        print(f"     {label}: {count} 样本")
    
    return {
        'min': np.min(features),
        'max': np.max(features),
        'mean': np.mean(features),
        'std': np.std(features),
        'median': np.median(features),
        'zero_ratio': zero_ratio,
        'labels': dict(zip(unique_labels, counts))
    }

def analyze_pressure_patterns(features, labels, name):
    """分析压力分布模式"""
    print(f"\n🔍 {name} 压力模式分析:")
    
    # 转换为图像
    images = np.array([pressure_to_image(row) for row in features])
    
    # 计算每个样本的压力中心
    centers_x = []
    centers_y = []
    total_pressures = []
    
    for img in images:
        y, x = np.indices(img.shape)
        total_pressure = np.sum(img)
        
        if total_pressure > 0:
            center_x = np.sum(x * img) / total_pressure
            center_y = np.sum(y * img) / total_pressure
        else:
            center_x = center_y = 8  # 中心位置
        
        centers_x.append(center_x)
        centers_y.append(center_y)
        total_pressures.append(total_pressure)
    
    centers_x = np.array(centers_x)
    centers_y = np.array(centers_y)
    total_pressures = np.array(total_pressures)
    
    print(f"   - 压力中心X坐标: {np.mean(centers_x):.2f} ± {np.std(centers_x):.2f}")
    print(f"   - 压力中心Y坐标: {np.mean(centers_y):.2f} ± {np.std(centers_y):.2f}")
    print(f"   - 总压力: {np.mean(total_pressures):.2f} ± {np.std(total_pressures):.2f}")
    
    # 分析左右分布
    left_pressure = np.sum(images[:, :, :8], axis=(1, 2))
    right_pressure = np.sum(images[:, :, 8:], axis=(1, 2))
    
    lr_ratio = left_pressure / (left_pressure + right_pressure + 1e-8)
    print(f"   - 左右压力比例: {np.mean(lr_ratio):.3f} ± {np.std(lr_ratio):.3f}")
    print(f"     (0.5=平衡, <0.5=右偏, >0.5=左偏)")
    
    return {
        'center_x': (np.mean(centers_x), np.std(centers_x)),
        'center_y': (np.mean(centers_y), np.std(centers_y)),
        'total_pressure': (np.mean(total_pressures), np.std(total_pressures)),
        'lr_ratio': (np.mean(lr_ratio), np.std(lr_ratio)),
        'images': images
    }

def visualize_comparison(train_features, train_labels, test_features, test_labels):
    """可视化训练数据与测试数据的对比"""
    print(f"\n🎨 生成对比可视化...")
    
    # 转换为图像
    train_images = np.array([pressure_to_image(row) for row in train_features])
    test_images = np.array([pressure_to_image(row) for row in test_features])
    
    # 1. 显示每类的平均图像
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 训练数据的平均图像
    le = LabelEncoder()
    all_labels = np.concatenate([train_labels, test_labels])
    le.fit(all_labels)
    
    train_encoded = le.transform(train_labels)
    
    for i, class_name in enumerate(le.classes_):
        if i < 3:  # 只显示前3个类别
            # 训练数据平均
            class_mask = train_encoded == i
            if np.any(class_mask):
                avg_image = np.mean(train_images[class_mask], axis=0)
                axes[0, i].imshow(avg_image, cmap='hot', interpolation='nearest')
                axes[0, i].set_title(f'训练-{class_name}平均')
                axes[0, i].axis('off')
            
            # 测试数据平均 (如果是left类别)
            if class_name == 'left':
                avg_test = np.mean(test_images, axis=0)
                axes[1, i].imshow(avg_test, cmap='hot', interpolation='nearest')
                axes[1, i].set_title(f'测试-{class_name}平均')
                axes[1, i].axis('off')
    
    # 显示差异
    if 'left' in le.classes_:
        left_idx = list(le.classes_).index('left')
        train_left_mask = train_encoded == left_idx
        if np.any(train_left_mask):
            train_left_avg = np.mean(train_images[train_left_mask], axis=0)
            test_left_avg = np.mean(test_images, axis=0)
            diff = test_left_avg - train_left_avg
            
            axes[1, 3].imshow(diff, cmap='RdBu', interpolation='nearest', vmin=-0.5, vmax=0.5)
            axes[1, 3].set_title('测试-训练差异')
            axes[1, 3].axis('off')
            plt.colorbar(axes[1, 3].images[0], ax=axes[1, 3])
    
    plt.tight_layout()
    plt.savefig('data_comparison_images.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 特征分布对比
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 压力值分布
    axes[0, 0].hist(train_features.flatten(), bins=50, alpha=0.7, label='训练数据', density=True)
    axes[0, 0].hist(test_features.flatten(), bins=50, alpha=0.7, label='测试数据', density=True)
    axes[0, 0].set_title('压力值分布')
    axes[0, 0].set_xlabel('压力值')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 样本总压力分布
    train_totals = np.sum(train_features, axis=1)
    test_totals = np.sum(test_features, axis=1)
    
    axes[0, 1].hist(train_totals, bins=30, alpha=0.7, label='训练数据', density=True)
    axes[0, 1].hist(test_totals, bins=30, alpha=0.7, label='测试数据', density=True)
    axes[0, 1].set_title('样本总压力分布')
    axes[0, 1].set_xlabel('总压力')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].legend()
    
    # 零值比例分布
    train_zero_ratios = np.sum(train_features == 0, axis=1) / train_features.shape[1]
    test_zero_ratios = np.sum(test_features == 0, axis=1) / test_features.shape[1]
    
    axes[1, 0].hist(train_zero_ratios, bins=20, alpha=0.7, label='训练数据', density=True)
    axes[1, 0].hist(test_zero_ratios, bins=20, alpha=0.7, label='测试数据', density=True)
    axes[1, 0].set_title('零值比例分布')
    axes[1, 0].set_xlabel('零值比例')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].legend()
    
    # 最大值分布
    train_maxes = np.max(train_features, axis=1)
    test_maxes = np.max(test_features, axis=1)
    
    axes[1, 1].hist(train_maxes, bins=30, alpha=0.7, label='训练数据', density=True)
    axes[1, 1].hist(test_maxes, bins=30, alpha=0.7, label='测试数据', density=True)
    axes[1, 1].set_title('最大压力值分布')
    axes[1, 1].set_xlabel('最大压力值')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('data_distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化结果已保存:")
    print(f"   - data_comparison_images.png (图像对比)")
    print(f"   - data_distribution_comparison.png (分布对比)")

def statistical_tests(train_features, test_features):
    """进行统计检验"""
    print(f"\n🧮 统计检验结果:")
    
    # KS检验 - 检验分布是否相同
    train_flat = train_features.flatten()
    test_flat = test_features.flatten()
    
    # 随机采样以避免计算量过大
    if len(train_flat) > 10000:
        train_sample = np.random.choice(train_flat, 10000, replace=False)
    else:
        train_sample = train_flat
        
    if len(test_flat) > 10000:
        test_sample = np.random.choice(test_flat, 10000, replace=False)
    else:
        test_sample = test_flat
    
    ks_stat, ks_p = stats.ks_2samp(train_sample, test_sample)
    print(f"   - KS检验 (分布相似性):")
    print(f"     统计量: {ks_stat:.4f}, p值: {ks_p:.4e}")
    
    if ks_p < 0.01:
        print(f"     ❌ 分布显著不同 (p < 0.01)")
    elif ks_p < 0.05:
        print(f"     ⚠️  分布可能不同 (p < 0.05)")
    else:
        print(f"     ✅ 分布相似 (p >= 0.05)")
    
    # 均值检验
    train_means = np.mean(train_features, axis=1)
    test_means = np.mean(test_features, axis=1)
    
    t_stat, t_p = stats.ttest_ind(train_means, test_means)
    print(f"\n   - T检验 (均值差异):")
    print(f"     统计量: {t_stat:.4f}, p值: {t_p:.4f}")
    
    if t_p < 0.01:
        print(f"     ❌ 均值显著不同 (p < 0.01)")
    elif t_p < 0.05:
        print(f"     ⚠️  均值可能不同 (p < 0.05)")
    else:
        print(f"     ✅ 均值相似 (p >= 0.05)")

def main():
    """主分析函数"""
    print("🔍 开始数据分布分析...")
    
    # 加载数据
    print("\n📂 加载数据...")
    train_features, train_labels = load_data('dataset.csv')
    test_features, test_labels = load_data('/Users/bx/Desktop/tmp_left.csv', 'left')
    
    # 只分析训练数据中的left类别
    train_left_mask = np.array(train_labels) == 'left'
    train_left_features = train_features[train_left_mask]
    train_left_labels = np.array(train_labels)[train_left_mask]
    
    print(f"\n🎯 专门分析left类别:")
    print(f"   - 训练集left样本: {len(train_left_features)}")
    print(f"   - 测试集left样本: {len(test_features)}")
    
    # 计算统计信息
    train_stats = calculate_statistics(train_left_features, train_left_labels, "训练集(left)")
    test_stats = calculate_statistics(test_features, test_labels, "测试集(left)")
    
    # 分析压力模式
    train_patterns = analyze_pressure_patterns(train_left_features, train_left_labels, "训练集(left)")
    test_patterns = analyze_pressure_patterns(test_features, test_labels, "测试集(left)")
    
    # 对比分析
    print(f"\n📋 关键差异对比:")
    print(f"   数值范围:")
    print(f"     训练: [{train_stats['min']:.1f}, {train_stats['max']:.1f}]")
    print(f"     测试: [{test_stats['min']:.1f}, {test_stats['max']:.1f}]")
    
    print(f"   压力中心位置:")
    print(f"     训练: ({train_patterns['center_x'][0]:.2f}, {train_patterns['center_y'][0]:.2f})")
    print(f"     测试: ({test_patterns['center_x'][0]:.2f}, {test_patterns['center_y'][0]:.2f})")
    
    print(f"   左右压力比例:")
    print(f"     训练: {train_patterns['lr_ratio'][0]:.3f} ± {train_patterns['lr_ratio'][1]:.3f}")
    print(f"     测试: {test_patterns['lr_ratio'][0]:.3f} ± {test_patterns['lr_ratio'][1]:.3f}")
    
    # 可视化对比
    visualize_comparison(train_left_features, train_left_labels, test_features, test_labels)
    
    # 统计检验
    statistical_tests(train_left_features, test_features)
    
    # 结论和建议
    print(f"\n💡 分析结论:")
    
    # 数值范围差异
    range_diff = abs(test_stats['max'] - train_stats['max']) / train_stats['max']
    if range_diff > 0.2:
        print(f"   ❌ 数值范围差异较大 ({range_diff*100:.1f}%)")
        print(f"      建议: 检查传感器校准或数据采集环境")
    
    # 压力中心差异
    center_diff = np.sqrt((train_patterns['center_x'][0] - test_patterns['center_x'][0])**2 + 
                         (train_patterns['center_y'][0] - test_patterns['center_y'][0])**2)
    if center_diff > 1.0:
        print(f"   ⚠️  压力中心偏移较大 ({center_diff:.2f}像素)")
        print(f"      建议: 检查坐姿定义一致性")
    
    # 左右比例差异
    lr_diff = abs(train_patterns['lr_ratio'][0] - test_patterns['lr_ratio'][0])
    if lr_diff > 0.1:
        print(f"   ⚠️  左右压力分布差异明显 ({lr_diff:.3f})")
        print(f"      建议: 训练数据可能需要更多类似模式的样本")
    
    print(f"\n🎯 改进建议:")
    print(f"   1. 收集更多与测试数据相似的训练样本")
    print(f"   2. 使用数据增强技术增加训练数据多样性")  
    print(f"   3. 检查数据采集环境和标注一致性")
    print(f"   4. 考虑使用域适应技术处理分布差异")

if __name__ == "__main__":
    main()