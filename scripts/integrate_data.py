#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据整合脚本
整合所有可用的训练数据，提高模型性能
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_merge_datasets():
    """加载并整合所有数据集"""
    print("📂 加载并整合数据集...")
    
    all_data = []
    
    # 数据文件列表
    data_files = [
        ('../data/dataset.csv', '原始训练数据'),
        ('../data/test_dataset.csv', '测试数据'),
        ('../data/artificial_test_data.csv', '人工测试数据')
    ]
    
    for filename, description in data_files:
        try:
            # 尝试不同编码
            try:
                df = pd.read_csv(filename, encoding='utf-8')
            except:
                df = pd.read_csv(filename, encoding='gbk')
            
            print(f"   ✅ {description} ({filename}):")
            print(f"      - 样本数: {len(df)}")
            
            if 'Label' in df.columns:
                label_counts = df['Label'].value_counts()
                print(f"      - 标签分布: {dict(label_counts)}")
                all_data.append(df)
            else:
                print(f"      - ⚠️  没有Label列，跳过")
                
        except Exception as e:
            print(f"   ❌ 无法加载 {filename}: {e}")
    
    # 合并所有数据
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        print(f"\n📊 整合后的数据集:")
        print(f"   - 总样本数: {len(merged_df)}")
        print(f"   - 特征维度: {merged_df.shape[1] - 1}")  # 减去Label列
        
        label_counts = merged_df['Label'].value_counts()
        print(f"   - 标签分布: {dict(label_counts)}")
        
        return merged_df
    else:
        print("❌ 没有成功加载任何数据集")
        return None

def analyze_merged_data(df):
    """分析整合后的数据"""
    print(f"\n🔍 数据质量分析:")
    
    # 检查缺失值
    missing_values = df.isnull().sum().sum()
    print(f"   - 缺失值: {missing_values}")
    
    # 检查重复样本
    duplicates = df.duplicated().sum()
    print(f"   - 重复样本: {duplicates}")
    
    # 特征统计
    feature_cols = [col for col in df.columns if col != 'Label']
    features = df[feature_cols].values
    
    print(f"   - 特征范围: {features.min():.2f} ~ {features.max():.2f}")
    print(f"   - 特征均值: {features.mean():.2f}")
    print(f"   - 特征标准差: {features.std():.2f}")
    
    # 类别平衡性
    label_counts = df['Label'].value_counts()
    total_samples = len(df)
    
    print(f"\n📊 类别平衡性:")
    for label, count in label_counts.items():
        percentage = count / total_samples * 100
        print(f"   - {label}: {count} 样本 ({percentage:.1f}%)")
    
    # 可视化标签分布
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    label_counts.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('整合数据集 - 标签分布')
    plt.xlabel('类别')
    plt.ylabel('样本数')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
            colors=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('整合数据集 - 标签比例')
    
    plt.tight_layout()
    plt.savefig('../results/integrated_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def remove_duplicates(df):
    """去除重复样本"""
    print(f"\n🧹 数据清洗:")
    original_size = len(df)
    
    # 去除完全重复的行
    df_clean = df.drop_duplicates()
    after_dedup = len(df_clean)
    
    print(f"   - 原始样本: {original_size}")
    print(f"   - 去重后: {after_dedup}")
    print(f"   - 移除重复: {original_size - after_dedup}")
    
    return df_clean

def balance_dataset(df, method='oversample'):
    """平衡数据集"""
    print(f"\n⚖️  数据平衡 (方法: {method}):")
    
    label_counts = df['Label'].value_counts()
    max_count = label_counts.max()
    min_count = label_counts.min()
    
    print(f"   - 最大类别样本数: {max_count}")
    print(f"   - 最小类别样本数: {min_count}")
    print(f"   - 不平衡比例: {max_count/min_count:.2f}:1")
    
    if method == 'oversample':
        # 上采样到最大类别的样本数
        balanced_data = []
        
        for label in df['Label'].unique():
            label_data = df[df['Label'] == label]
            current_count = len(label_data)
            
            if current_count < max_count:
                # 随机重采样
                additional_samples = max_count - current_count
                resampled = label_data.sample(n=additional_samples, replace=True, random_state=42)
                balanced_data.append(pd.concat([label_data, resampled]))
            else:
                balanced_data.append(label_data)
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        
    elif method == 'undersample':
        # 下采样到最小类别的样本数
        balanced_data = []
        
        for label in df['Label'].unique():
            label_data = df[df['Label'] == label]
            sampled_data = label_data.sample(n=min_count, random_state=42)
            balanced_data.append(sampled_data)
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
    
    # 打乱数据
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   - 平衡后总样本: {len(balanced_df)}")
    balanced_counts = balanced_df['Label'].value_counts()
    for label, count in balanced_counts.items():
        print(f"   - {label}: {count} 样本")
    
    return balanced_df

def save_integrated_dataset(df, filename='../data/integrated_dataset.csv'):
    """保存整合后的数据集"""
    print(f"\n💾 保存整合数据集: {filename}")
    
    df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"   ✅ 已保存 {len(df)} 个样本到 {filename}")
    
    # 创建训练/验证分割
    features = df.drop('Label', axis=1)
    labels = df['Label']
    
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 保存训练集
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.to_csv('../data/integrated_train.csv', index=False, encoding='utf-8')
    
    # 保存验证集
    val_df = pd.concat([X_val, y_val], axis=1)
    val_df.to_csv('../data/integrated_val.csv', index=False, encoding='utf-8')
    
    print(f"   ✅ 训练集: {len(train_df)} 样本 → integrated_train.csv")
    print(f"   ✅ 验证集: {len(val_df)} 样本 → integrated_val.csv")

def main():
    """主函数"""
    print("🚀 开始数据整合流程...\n")
    
    # 1. 加载并合并数据
    merged_df = load_and_merge_datasets()
    if merged_df is None:
        return
    
    # 2. 分析数据
    analyzed_df = analyze_merged_data(merged_df)
    
    # 3. 清理数据
    clean_df = remove_duplicates(analyzed_df)
    
    # 4. 数据平衡 (可选)
    print(f"\n选择数据平衡策略:")
    print(f"   1. 不进行平衡 (保持原始分布)")
    print(f"   2. 上采样 (增加少数类样本)")
    print(f"   3. 下采样 (减少多数类样本)")
    
    choice = input("请选择 (1-3，默认1): ").strip()
    
    if choice == '2':
        final_df = balance_dataset(clean_df, 'oversample')
    elif choice == '3':
        final_df = balance_dataset(clean_df, 'undersample')
    else:
        final_df = clean_df
        print(f"\n📊 保持原始数据分布")
    
    # 5. 保存结果
    save_integrated_dataset(final_df)
    
    print(f"\n🎉 数据整合完成！")
    print(f"   - 可以使用 integrated_dataset.csv 进行训练")
    print(f"   - 或直接使用 integrated_train.csv 和 integrated_val.csv")

if __name__ == "__main__":
    main()