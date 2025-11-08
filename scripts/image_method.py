#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16x16图像方法验证 - 使用scikit-learn实现
将256维压力数据转换为16x16图像，然后使用不同的特征提取方法
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy import ndimage
import matplotlib.pyplot as plt
import warnings
import joblib

warnings.filterwarnings('ignore')

def pressure_to_image(pressure_data):
    """将256维压力数据转换为16x16图像"""
    image = pressure_data.reshape(16, 16).astype(np.float32)
    
    # 标准化到0-1范围
    if image.max() > 0:
        image = image / image.max()
    
    return image

def extract_image_features(image):
    """从16x16图像中提取特征"""
    features = []
    
    # 1. 基础统计特征
    features.extend([
        np.mean(image),      # 均值
        np.std(image),       # 标准差
        np.max(image),       # 最大值
        np.min(image),       # 最小值
        np.median(image),    # 中位数
    ])
    
    # 2. 几何特征
    # 重心位置
    y, x = np.indices(image.shape)
    total_mass = np.sum(image)
    if total_mass > 0:
        center_x = np.sum(x * image) / total_mass
        center_y = np.sum(y * image) / total_mass
    else:
        center_x = center_y = 8  # 中心位置
    
    features.extend([center_x, center_y])
    
    # 3. 区域特征
    # 左右半区的质量分布
    left_half = np.sum(image[:, :8])
    right_half = np.sum(image[:, 8:])
    top_half = np.sum(image[:8, :])
    bottom_half = np.sum(image[8:, :])
    
    total = left_half + right_half
    if total > 0:
        left_ratio = left_half / total
        right_ratio = right_half / total
    else:
        left_ratio = right_ratio = 0.5
    
    total_v = top_half + bottom_half
    if total_v > 0:
        top_ratio = top_half / total_v
        bottom_ratio = bottom_half / total_v
    else:
        top_ratio = bottom_ratio = 0.5
    
    features.extend([left_ratio, right_ratio, top_ratio, bottom_ratio])
    
    # 4. 纹理特征 - 使用简单的梯度统计
    # 水平梯度
    grad_x = np.abs(np.diff(image, axis=1))
    grad_y = np.abs(np.diff(image, axis=0))
    
    features.extend([
        np.mean(grad_x),     # 水平梯度均值
        np.std(grad_x),      # 水平梯度标准差
        np.mean(grad_y),     # 垂直梯度均值
        np.std(grad_y),      # 垂直梯度标准差
    ])
    
    # 5. 形状特征
    # 主要压力点的数量（阈值方法）
    threshold = np.mean(image) + np.std(image)
    high_pressure_points = np.sum(image > threshold)
    
    # 连通区域数量（简化版本）
    binary_image = image > (np.max(image) * 0.3)
    
    features.extend([
        high_pressure_points,
        np.sum(binary_image),  # 活跃像素数量
    ])
    
    # 6. 对称性特征
    # 左右对称性
    left_side = image[:, :8]
    right_side = np.fliplr(image[:, 8:])
    symmetry_lr = np.corrcoef(left_side.flatten(), right_side.flatten())[0, 1]
    if np.isnan(symmetry_lr):
        symmetry_lr = 0
    
    features.append(symmetry_lr)
    
    return np.array(features)

def load_and_extract_features(csv_file):
    """加载数据并提取图像特征"""
    print(f"🔄 加载数据: {csv_file}")
    
    # 读取CSV
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except:
        df = pd.read_csv(csv_file, encoding='gbk')
    
    print(f"   - 样本数: {len(df)}")
    
    # 分离标签和原始特征
    labels = df['Label'].values
    pressure_data = df.drop('Label', axis=1).values
    
    print(f"🖼️  转换为16x16图像并提取特征...")
    
    # 转换为图像并提取特征
    image_features = []
    images = []
    
    for i, row in enumerate(pressure_data):
        # 转换为16x16图像
        image = pressure_to_image(row)
        images.append(image)
        
        # 提取图像特征
        features = extract_image_features(image)
        image_features.append(features)
        
        if (i + 1) % 100 == 0:
            print(f"   处理进度: {i+1}/{len(pressure_data)}")
    
    image_features = np.array(image_features)
    images = np.array(images)
    
    # 编码标签
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"✅ 特征提取完成:")
    print(f"   - 图像特征维度: {image_features.shape[1]}")
    print(f"   - 类别数: {len(label_encoder.classes_)}")
    
    return image_features, encoded_labels, label_encoder, images

def visualize_pressure_images(images, labels, label_encoder, num_samples=6):
    """可视化压力图像"""
    plt.figure(figsize=(15, 10))
    
    samples_per_class = num_samples // len(label_encoder.classes_)
    plot_idx = 1
    
    for class_idx, class_name in enumerate(label_encoder.classes_):
        # 找到该类的样本
        class_indices = np.where(labels == class_idx)[0]
        
        for i in range(min(samples_per_class, len(class_indices))):
            sample_idx = class_indices[i]
            
            plt.subplot(len(label_encoder.classes_), samples_per_class, plot_idx)
            plt.imshow(images[sample_idx], cmap='viridis')
            plt.title(f'{class_name} - 样本{i+1}')
            plt.colorbar()
            plt.axis('off')
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('pressure_images_by_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 压力图像分类样例已保存: pressure_images_by_class.png")

def compare_methods():
    """比较原始方法vs图像方法"""
    print("🔄 开始方法对比实验...")
    
    # 加载原始数据（SVM方法）
    df = pd.read_csv('dataset.csv', encoding='utf-8')
    labels = df['Label'].values
    original_features = df.drop('Label', axis=1).values
    
    # 图像方法特征提取
    image_features, encoded_labels, label_encoder, images = load_and_extract_features('dataset.csv')
    
    # 数据分割
    X_orig_train, X_orig_test, y_train, y_test = train_test_split(
        original_features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    X_img_train, X_img_test, _, _ = train_test_split(
        image_features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"\n📊 数据分割:")
    print(f"   - 训练集: {len(X_orig_train)} 样本")
    print(f"   - 测试集: {len(X_orig_test)} 样本")
    print(f"   - 原始特征维度: {X_orig_train.shape[1]}")
    print(f"   - 图像特征维度: {X_img_train.shape[1]}")
    
    # 可视化压力图像
    train_indices = np.arange(len(X_orig_train))
    visualize_pressure_images(images[train_indices], y_train, label_encoder)
    
    results = {}
    
    # 1. 原始SVM方法（简化版，不用PCA）
    print(f"\n🔄 测试原始SVM方法...")
    scaler_orig = StandardScaler()
    X_orig_train_scaled = scaler_orig.fit_transform(X_orig_train)
    X_orig_test_scaled = scaler_orig.transform(X_orig_test)
    
    svm_orig = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_orig.fit(X_orig_train_scaled, y_train)
    
    orig_pred = svm_orig.predict(X_orig_test_scaled)
    orig_acc = accuracy_score(y_test, orig_pred)
    results['原始SVM'] = orig_acc
    
    print(f"   准确率: {orig_acc:.4f}")
    
    # 2. 图像特征 + SVM
    print(f"\n🔄 测试图像特征+SVM方法...")
    scaler_img = StandardScaler()
    X_img_train_scaled = scaler_img.fit_transform(X_img_train)
    X_img_test_scaled = scaler_img.transform(X_img_test)
    
    svm_img = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_img.fit(X_img_train_scaled, y_train)
    
    img_pred = svm_img.predict(X_img_test_scaled)
    img_acc = accuracy_score(y_test, img_pred)
    results['图像特征+SVM'] = img_acc
    
    print(f"   准确率: {img_acc:.4f}")
    
    # 3. 图像特征 + 随机森林
    print(f"\n🔄 测试图像特征+随机森林方法...")
    rf_img = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_img.fit(X_img_train, y_train)
    
    rf_pred = rf_img.predict(X_img_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    results['图像特征+RF'] = rf_acc
    
    print(f"   准确率: {rf_acc:.4f}")
    
    # 结果对比
    print(f"\n📊 方法对比结果:")
    for method, acc in results.items():
        print(f"   {method}: {acc:.4f} ({acc*100:.2f}%)")
    
    best_method = max(results, key=results.get)
    print(f"\n🏆 最佳方法: {best_method} ({results[best_method]*100:.2f}%)")
    
    # 保存最佳模型（图像特征版本）
    if 'SVM' in best_method:
        joblib.dump(svm_img, 'image_svm_model.pkl')
        joblib.dump(scaler_img, 'image_scaler.pkl')
        print(f"✅ 最佳SVM模型已保存")
    
    joblib.dump(label_encoder, 'image_label_encoder.pkl')
    
    return results

def predict_with_image_method(csv_file):
    """使用图像方法进行预测"""
    print(f"🔮 使用图像方法预测: {csv_file}")
    
    # 加载模型
    try:
        model = joblib.load('image_svm_model.pkl')
        scaler = joblib.load('image_scaler.pkl')
        label_encoder = joblib.load('image_label_encoder.pkl')
        print(f"✅ 图像方法模型加载成功")
    except:
        print(f"❌ 请先运行训练: python image_method.py compare")
        return
    
    # 加载测试数据
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except:
        df = pd.read_csv(csv_file, encoding='gbk')
    
    if 'Label' in df.columns:
        true_labels = df['Label'].values
        features = df.drop('Label', axis=1).values
        has_labels = True
    else:
        true_labels = None
        features = df.values
        has_labels = False
    
    print(f"   - 样本数: {len(features)}")
    
    # 提取图像特征
    print(f"🖼️  提取图像特征...")
    image_features = []
    for row in features:
        image = pressure_to_image(row)
        img_features = extract_image_features(image)
        image_features.append(img_features)
    
    image_features = np.array(image_features)
    
    # 标准化并预测
    image_features_scaled = scaler.transform(image_features)
    predictions = model.predict(image_features_scaled)
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    # 统计结果
    unique_pred, counts_pred = np.unique(predicted_labels, return_counts=True)
    pred_distribution = {label: int(count) for label, count in zip(unique_pred, counts_pred)}
    
    print(f"✅ 图像方法预测完成:")
    print(f"   - 预测分布: {pred_distribution}")
    
    if has_labels:
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"   - 图像方法准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print(f"\n📋 图像方法分类报告:")
        print(classification_report(true_labels, predicted_labels))
        
        # 错误分析
        errors = np.where(true_labels != predicted_labels)[0]
        if len(errors) > 0:
            print(f"\n❌ 错误预测 (共{len(errors)}个):")
            for idx in errors:
                print(f"   样本{idx}: {true_labels[idx]} → {predicted_labels[idx]}")
        else:
            print(f"\n✅ 完美预测，没有错误！")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python image_method.py compare             # 对比原始vs图像方法")
        print("  python image_method.py predict <csv_file>  # 使用图像方法预测")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == 'compare':
        print("🚀 开始方法对比实验...")
        compare_methods()
        print("\n🎉 实验完成！")
        
    elif mode == 'predict':
        if len(sys.argv) < 3:
            print("❌ 请提供CSV文件路径")
            sys.exit(1)
        
        csv_file = sys.argv[2]
        predict_with_image_method(csv_file)
    
    else:
        print(f"❌ 未知模式: {mode}")