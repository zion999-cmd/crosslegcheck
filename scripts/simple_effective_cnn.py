#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于成功经验的简单有效CNN模型
参考MNIST的成功模式，但适配压力传感器数据
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def pressure_to_image(pressure_data):
    """将256维压力数据转换为16x16图像"""
    return pressure_data.reshape(16, 16)

def load_data(csv_file):
    """加载和预处理数据"""
    print(f"📂 加载数据: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except:
        df = pd.read_csv(csv_file, encoding='gbk')
    
    # 分离标签和特征
    labels = df['Label'].values
    features = df.drop('Label', axis=1).values
    
    print(f"   - 总样本数: {len(features)}")
    print(f"   - 特征维度: {features.shape[1]}")
    
    # 标签分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"   - {label}: {count} 样本")
    
    # 数据标准化 (关键改进！)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 转换为图像格式
    images = np.array([pressure_to_image(row) for row in features_scaled])
    
    # 标签编码
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    print(f"   - 标签编码: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    return images, encoded_labels, le, scaler

def create_simple_effective_model():
    """创建基于您成功经验的简单模型"""
    print("🏗️  创建简单有效模型...")
    
    model = keras.models.Sequential([
        # 输入层：16x16的压力图像
        keras.layers.Flatten(input_shape=(16, 16)),
        
        # 第一层：128个神经元 (和您的MNIST一样)
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        
        # 第二层：64个神经元 (适当减少)
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        
        # 输出层：3个类别 (left, normal, right)
        keras.layers.Dense(3)  # 不用激活函数，让SparseCategoricalCrossentropy处理
    ])
    
    print(f"   - 模型参数: {model.count_params():,}")
    return model

def train_simple_model(data_path='../data/dataset.csv'):
    """训练简单有效模型"""
    print("🚀 开始训练简单有效模型...\n")
    
    # 加载数据
    images, labels, le, scaler = load_data(data_path)
    
    # 划分训练和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"   - 训练集: {len(X_train)} 样本")
    print(f"   - 验证集: {len(X_val)} 样本")
    
    # 创建模型
    model = create_simple_effective_model()
    
    # 编译模型 (完全按照您的成功配置)
    model.compile(
        optimizer='adam',  # 使用默认的adam
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # 打印模型结构
    print(f"\n📋 模型结构:")
    model.summary()
    
    # 训练模型
    print(f"\n🔥 开始训练...")
    
    history = model.fit(
        X_train, y_train,
        epochs=50,  # 适中的训练轮数
        validation_data=(X_val, y_val),
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
    )
    
    # 评估模型
    print(f"\n📊 模型评估:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"   - 训练准确率: {train_acc:.4f}")
    print(f"   - 验证准确率: {val_acc:.4f}")
    
    # 保存模型和预处理器
    model_dir = '../models'
    os.makedirs(model_dir, exist_ok=True)
    
    model.save(f'{model_dir}/simple_effective_model.keras')
    
    # 保存预处理器
    import joblib
    joblib.dump(le, f'{model_dir}/simple_label_encoder.pkl')
    joblib.dump(scaler, f'{model_dir}/simple_scaler.pkl')
    
    print(f"   ✅ 模型已保存: simple_effective_model.keras")
    
    # 可视化训练过程
    plot_training_history(history)
    
    return model, le, scaler

def plot_training_history(history):
    """可视化训练过程"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../results/simple_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def predict_simple_model(test_csv, model_path='../models/simple_effective_model.keras'):
    """使用简单模型进行预测"""
    print(f"\n🔮 使用简单有效模型进行预测...")
    
    # 加载模型和预处理器
    model = keras.models.load_model(model_path)
    
    import joblib
    le = joblib.load('../models/simple_label_encoder.pkl')
    scaler = joblib.load('../models/simple_scaler.pkl')
    
    print(f"   - 已加载模型: {model_path}")
    
    # 加载测试数据
    try:
        df = pd.read_csv(test_csv, encoding='utf-8')
    except:
        df = pd.read_csv(test_csv, encoding='gbk')
    
    # 检查列数，处理多余列
    if df.shape[1] == 257:
        if 'Label' in df.columns:
            features = df.drop('Label', axis=1).values
        else:
            features = df.iloc[:, 1:].values
    else:
        features = df.values
    
    print(f"   - 测试样本数: {len(features)}")
    
    # 数据预处理
    features_scaled = scaler.transform(features)
    images = np.array([pressure_to_image(row) for row in features_scaled])
    
    # 预测
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # 获取预测概率
    probabilities = tf.nn.softmax(predictions).numpy()
    
    # 类别映射
    class_names = le.classes_
    
    # 显示预测结果
    print(f"\n📋 预测结果:")
    print(f"   样本编号 | 预测类别 | 置信度 | 各类别概率")
    print(f"   --------|---------|-------|------------------")
    
    correct_count = 0
    for i, (pred_class, prob) in enumerate(zip(predicted_classes, probabilities)):
        predicted_label = class_names[pred_class]
        confidence = prob[pred_class]
        
        # 假设测试数据都是left类别
        is_correct = predicted_label == 'left'
        if is_correct:
            correct_count += 1
        
        status = "✅" if is_correct else "❌"
        
        prob_str = " | ".join([f"{name}:{p:.3f}" for name, p in zip(class_names, prob)])
        print(f"   {i+1:7d} | {predicted_label:7s} | {confidence:.3f} | {prob_str} {status}")
    
    accuracy = correct_count / len(features)
    print(f"\n📊 预测统计:")
    print(f"   - 正确预测: {correct_count}/{len(features)} ({accuracy:.2%})")
    
    # 类别分布
    pred_counts = np.bincount(predicted_classes, minlength=len(class_names))
    for i, (name, count) in enumerate(zip(class_names, pred_counts)):
        print(f"   - 预测为{name}: {count} 样本 ({count/len(features):.1%})")
    
    return predictions, predicted_classes

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python simple_effective_cnn.py [train|predict] [文件路径]")
        return
    
    command = sys.argv[1]
    
    if command == 'train':
        data_path = sys.argv[2] if len(sys.argv) > 2 else '../data/dataset.csv'
        train_simple_model(data_path)
        
    elif command == 'predict':
        if len(sys.argv) < 3:
            print("请提供测试文件路径")
            return
        test_path = sys.argv[2]
        predict_simple_model(test_path)
        
    else:
        print("未知命令，请使用 'train' 或 'predict'")

if __name__ == "__main__":
    main()