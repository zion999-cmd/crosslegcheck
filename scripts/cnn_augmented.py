#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带数据增强的CNN模型
通过数据增强技术缓解样本不足问题
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
np.random.seed(42)
tf.random.set_seed(42)

def pressure_to_image(pressure_data):
    """将256维压力数据转换为16x16图像"""
    image = pressure_data.reshape(16, 16).astype(np.float32)
    
    # 归一化到0-1范围
    if image.max() > 0:
        image = image / image.max()
    
    return image

def augment_pressure_image(image, augment_params=None):
    """对压力图像进行数据增强
    
    Args:
        image: 16x16的压力图像
        augment_params: 增强参数字典
    
    Returns:
        增强后的图像
    """
    if augment_params is None:
        augment_params = {
            'noise_std': 0.02,
            'brightness_range': 0.1,
            'rotation_range': 5,
            'shift_range': 1,
            'zoom_range': 0.05,
        }
    
    aug_image = image.copy()
    
    # 1. 添加高斯噪声
    if np.random.random() < 0.7:
        noise = np.random.normal(0, augment_params['noise_std'], image.shape)
        aug_image = aug_image + noise
        aug_image = np.clip(aug_image, 0, 1)
    
    # 2. 亮度调整
    if np.random.random() < 0.5:
        brightness_factor = 1 + np.random.uniform(-augment_params['brightness_range'], 
                                                 augment_params['brightness_range'])
        aug_image = aug_image * brightness_factor
        aug_image = np.clip(aug_image, 0, 1)
    
    # 3. 轻微旋转 (通过TensorFlow实现)
    if np.random.random() < 0.4:
        angle = np.random.uniform(-augment_params['rotation_range'], 
                                augment_params['rotation_range'])
        aug_image = tf.image.rot90(aug_image, k=int(angle/90)) if abs(angle) > 45 else aug_image
    
    # 4. 平移
    if np.random.random() < 0.6:
        shift_x = int(np.random.uniform(-augment_params['shift_range'], 
                                       augment_params['shift_range']))
        shift_y = int(np.random.uniform(-augment_params['shift_range'], 
                                       augment_params['shift_range']))
        
        aug_image = np.roll(aug_image, shift_x, axis=0)
        aug_image = np.roll(aug_image, shift_y, axis=1)
    
    # 5. 缩放（通过裁剪和填充实现）
    if np.random.random() < 0.3:
        zoom_factor = 1 + np.random.uniform(-augment_params['zoom_range'], 
                                          augment_params['zoom_range'])
        
        if zoom_factor < 1:  # 缩小 - 周围填充0
            h, w = aug_image.shape
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            
            if new_h > 0 and new_w > 0:
                # 从中心裁剪
                start_h = (h - new_h) // 2
                start_w = (w - new_w) // 2
                
                cropped = aug_image[start_h:start_h+new_h, start_w:start_w+new_w]
                
                # 缩放回原尺寸
                aug_image = tf.image.resize(cropped[..., None], [h, w]).numpy()[..., 0]
    
    return aug_image

def create_augmented_dataset(X, y, augmentation_factor=3):
    """创建增强数据集"""
    print(f"   - 创建增强数据集 (增强倍数: {augmentation_factor}x)...")
    
    X_augmented = []
    y_augmented = []
    
    # 添加原始数据
    for i in range(len(X)):
        X_augmented.append(X[i])
        y_augmented.append(y[i])
    
    # 添加增强数据
    for i in range(len(X)):
        for _ in range(augmentation_factor):
            aug_image = augment_pressure_image(X[i])
            X_augmented.append(aug_image)
            y_augmented.append(y[i])
    
    # 打乱数据
    indices = np.random.permutation(len(X_augmented))
    X_augmented = np.array(X_augmented)[indices]
    y_augmented = np.array(y_augmented)[indices]
    
    print(f"   - 原始数据: {len(X)} 样本")
    print(f"   - 增强后: {len(X_augmented)} 样本")
    
    return X_augmented, y_augmented

def load_data(csv_file='../data/dataset.csv'):
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
    
    # 统计标签分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"   - {label}: {count} 样本")
    
    return features, labels

def create_enhanced_cnn(input_shape=(16, 16, 1), num_classes=3):
    """创建增强的CNN模型"""
    model = keras.models.Sequential([
        # 第一个卷积块
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # 第二个卷积块
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # 第三个卷积块
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling2D(),
        
        # 分类器
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_augmented_cnn(features, labels, model_path='../models/cnn_augmented_model.keras'):
    """训练带数据增强的CNN模型"""
    print(f"\n🏋️ 开始训练增强CNN模型...")
    
    # 转换为图像格式
    print(f"   - 转换为16x16图像格式...")
    images = np.array([pressure_to_image(row) for row in features])
    images = images[..., np.newaxis]  # 添加通道维度
    
    # 编码标签
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    print(f"   - 标签编码: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # 划分训练和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        images, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"   - 训练集: {len(X_train)} 样本")
    print(f"   - 验证集: {len(X_val)} 样本")
    
    # 创建增强数据集
    X_train_aug, y_train_aug = create_augmented_dataset(
        X_train.squeeze(), y_train, augmentation_factor=3
    )
    
    # 添加通道维度
    X_train_aug = X_train_aug[..., np.newaxis]
    
    # 创建模型
    model = create_enhanced_cnn()
    
    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 打印模型结构
    print(f"\n📋 模型结构:")
    model.summary()
    
    # 设置回调函数
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 训练模型
    print(f"\n🚀 开始训练...")
    
    history = model.fit(
        X_train_aug, y_train_aug,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    print(f"\n✅ 训练完成!")
    print(f"   - 最佳验证准确率: {max(history.history['val_accuracy']):.4f}")
    print(f"   - 模型已保存: {model_path}")
    
    # 可视化训练历史
    visualize_training_history(history)
    
    return model, le, history

def visualize_training_history(history):
    """可视化训练历史"""
    print(f"\n📊 生成训练历史图表...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 准确率
    ax1.plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    ax1.set_title('模型准确率 (带数据增强)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('准确率')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 损失
    ax2.plot(history.history['loss'], label='训练损失', linewidth=2)
    ax2.plot(history.history['val_loss'], label='验证损失', linewidth=2)
    ax2.set_title('模型损失 (带数据增强)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/cnn_augmented_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ 训练历史图已保存: cnn_augmented_training_history.png")

def predict_with_augmented_cnn(test_csv, model_path='../models/cnn_augmented_model.keras', expected_label=None):
    """使用增强CNN模型进行预测"""
    print(f"\n🔮 使用增强CNN模型进行预测...")
    
    # 加载模型
    model = keras.models.load_model(model_path)
    print(f"   - 已加载模型: {model_path}")
    
    # 加载测试数据
    try:
        df = pd.read_csv(test_csv, encoding='utf-8')
    except:
        df = pd.read_csv(test_csv, encoding='gbk')
    
    # 检查列数，如果有标签列则删除
    has_label = False
    true_labels = None
    
    if df.shape[1] == 257:
        if 'Label' in df.columns:
            true_labels = df['Label'].values
            features = df.drop('Label', axis=1).values
            has_label = True
        else:
            # 假设第一列是标签或索引
            features = df.iloc[:, 1:].values
    else:
        features = df.values
    
    print(f"   - 测试样本数: {len(features)}")
    print(f"   - 特征维度: {features.shape[1]}")
    
    # 转换为图像格式
    images = np.array([pressure_to_image(row) for row in features])
    images = images[..., np.newaxis]
    
    # 预测
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # 类别映射
    class_names = ['left', 'normal', 'right']
    
    # 显示预测结果
    print(f"\n📋 预测结果:")
    if has_label:
        print(f"   样本编号 | 真实标签 | 预测类别 | 置信度 | 各类别概率 | 结果")
        print(f"   --------|---------|---------|-------|-----------|----")
    elif expected_label:
        print(f"   样本编号 | 预测类别 | 置信度 | 各类别概率 | 结果(期望:{expected_label})")
        print(f"   --------|---------|-------|-----------|----------------")
    else:
        print(f"   样本编号 | 预测类别 | 置信度 | 各类别概率")
        print(f"   --------|---------|-------|------------------")
    
    correct_count = 0
    for i, (pred_class, prob) in enumerate(zip(predicted_classes, predictions)):
        predicted_label = class_names[pred_class]
        confidence = prob[pred_class]
        
        # 判断正确性
        if has_label:
            true_label = true_labels[i]
            is_correct = predicted_label == true_label
        elif expected_label:
            is_correct = predicted_label == expected_label
        else:
            is_correct = None
        
        if is_correct is not None:
            if is_correct:
                correct_count += 1
            status = "✅" if is_correct else "❌"
        else:
            status = ""
        
        prob_str = " | ".join([f"{name}:{p:.3f}" for name, p in zip(class_names, prob)])
        
        if has_label:
            print(f"   {i+1:7d} | {true_labels[i]:7s} | {predicted_label:7s} | {confidence:.3f} | {prob_str} | {status}")
        elif expected_label:
            print(f"   {i+1:7d} | {predicted_label:7s} | {confidence:.3f} | {prob_str} | {status}")
        else:
            print(f"   {i+1:7d} | {predicted_label:7s} | {confidence:.3f} | {prob_str}")
    
    # 统计结果
    if has_label or expected_label:
        accuracy = correct_count / len(features)
        print(f"\n📊 预测统计:")
        print(f"   - 正确预测: {correct_count}/{len(features)} ({accuracy:.2%})")
    else:
        print(f"\n📊 预测统计:")
    
    # 类别分布
    pred_counts = np.bincount(predicted_classes, minlength=3)
    for i, (name, count) in enumerate(zip(class_names, pred_counts)):
        print(f"   - 预测为{name}: {count} 样本 ({count/len(features):.1%})")
    
    return predictions, predicted_classes

def visualize_augmentation_samples():
    """可视化数据增强效果"""
    print(f"\n🎨 生成数据增强示例...")
    
    # 加载一个样本
    features, labels = load_data('../data/dataset.csv')
    
    # 找一个left类别的样本
    left_indices = np.where(np.array(labels) == 'left')[0]
    sample_idx = left_indices[0]
    sample_image = pressure_to_image(features[sample_idx])
    
    # 生成增强样本
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # 原始图像
    axes[0, 0].imshow(sample_image, cmap='hot', interpolation='nearest')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 增强样本
    augment_types = ['噪声', '亮度', '旋转', '平移', '缩放']
    
    for i in range(4):
        aug_image = augment_pressure_image(sample_image)
        axes[0, i+1].imshow(aug_image, cmap='hot', interpolation='nearest')
        axes[0, i+1].set_title(f'增强样本 {i+1}')
        axes[0, i+1].axis('off')
    
    # 显示不同增强类型的效果
    for i, aug_type in enumerate(augment_types):
        aug_image = augment_pressure_image(sample_image)
        axes[1, i].imshow(aug_image, cmap='hot', interpolation='nearest')
        axes[1, i].set_title(f'{aug_type}增强')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('../results/data_augmentation_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ 数据增强示例已保存: data_augmentation_samples.png")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("📖 使用方法:")
        print("   python cnn_augmented.py train                    - 训练模型")
        print("   python cnn_augmented.py predict <csv>           - 预测(自动判断)")
        print("   python cnn_augmented.py predict <csv> <label>   - 预测(指定期望标签)")
        print("   python cnn_augmented.py visualize               - 可视化增强效果")
        return
    
    command = sys.argv[1]
    
    if command == 'train':
        print("🔥 训练带数据增强的CNN模型")
        
        # 可视化增强效果
        visualize_augmentation_samples()
        
        # 加载数据并训练
        features, labels = load_data('../data/dataset.csv')
        model, le, history = train_augmented_cnn(features, labels)
        
        print(f"\n🎯 训练总结:")
        print(f"   - 有效训练样本: {len(features) * 4} (原始 + 3倍增强)")
        print(f"   - 最终验证准确率: {max(history.history['val_accuracy']):.4f}")
        print(f"   - 模型参数量: {model.count_params():,}")
        
    elif command == 'predict':
        if len(sys.argv) < 3:
            print("❌ 请提供CSV文件路径")
            return
        
        test_csv = sys.argv[2]
        expected_label = sys.argv[3] if len(sys.argv) > 3 else None
        predict_with_augmented_cnn(test_csv, expected_label=expected_label)
        
    elif command == 'visualize':
        visualize_augmentation_samples()
        
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == "__main__":
    main()