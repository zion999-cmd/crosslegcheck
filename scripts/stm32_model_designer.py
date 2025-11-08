#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM32嵌入式模型设计
专为STM32H750设计的轻量级分类模型
"""

import numpy as np
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import joblib
import json
import time

class STM32ModelDesigner:
    """STM32嵌入式模型设计器"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoder = None
        self.feature_extractors = {}
        
    def extract_statistical_features(self, pressure_data):
        """提取统计特征 - 适合嵌入式计算"""
        features = []
        
        # 基础统计特征 (8个)
        features.extend([
            np.mean(pressure_data),      # 平均值
            np.std(pressure_data),       # 标准差
            np.min(pressure_data),       # 最小值
            np.max(pressure_data),       # 最大值
            np.median(pressure_data),    # 中位数
            np.sum(pressure_data),       # 总和
            np.count_nonzero(pressure_data),  # 非零点数量
            np.max(pressure_data) - np.min(pressure_data)  # 极差
        ])
        
        # 分布特征 (4个)
        q25, q75 = np.percentile(pressure_data, [25, 75])
        features.extend([
            q25,                         # 25%分位数
            q75,                         # 75%分位数
            q75 - q25,                   # 四分位距
            np.mean(pressure_data > np.mean(pressure_data))  # 超过均值的比例
        ])
        
        return np.array(features)
    
    def extract_spatial_features(self, pressure_data):
        """提取空间特征 - 基于16x16图像的空间分布"""
        # 重塑为16x16图像
        image = pressure_data.reshape(16, 16)
        features = []
        
        # 重心计算 (2个特征)
        total_pressure = np.sum(image)
        if total_pressure > 0:
            # 计算质心
            y_indices, x_indices = np.meshgrid(range(16), range(16), indexing='ij')
            center_x = np.sum(x_indices * image) / total_pressure
            center_y = np.sum(y_indices * image) / total_pressure
        else:
            center_x = center_y = 8.0  # 中心位置
        
        features.extend([center_x, center_y])
        
        # 区域压力分布 (4个特征)
        left_pressure = np.sum(image[:, :8])      # 左半部分
        right_pressure = np.sum(image[:, 8:])     # 右半部分
        top_pressure = np.sum(image[:8, :])       # 上半部分
        bottom_pressure = np.sum(image[8:, :])    # 下半部分
        
        if total_pressure > 0:
            features.extend([
                left_pressure / total_pressure,    # 左侧压力比例
                right_pressure / total_pressure,   # 右侧压力比例
                top_pressure / total_pressure,     # 上部压力比例
                bottom_pressure / total_pressure   # 下部压力比例
            ])
        else:
            features.extend([0.25, 0.25, 0.25, 0.25])
        
        # 对称性特征 (2个特征)
        left_right_ratio = left_pressure / (right_pressure + 1e-6)
        top_bottom_ratio = top_pressure / (bottom_pressure + 1e-6)
        features.extend([left_right_ratio, top_bottom_ratio])
        
        return np.array(features)
    
    def extract_peak_features(self, pressure_data):
        """提取峰值特征"""
        features = []
        
        # 峰值相关 (4个特征)
        sorted_data = np.sort(pressure_data)[::-1]  # 降序排列
        top_5_avg = np.mean(sorted_data[:5])        # 前5个最大值平均
        top_10_avg = np.mean(sorted_data[:10])      # 前10个最大值平均
        
        features.extend([
            top_5_avg,
            top_10_avg,
            top_5_avg / (np.mean(pressure_data) + 1e-6),  # 峰值与均值比
            np.max(pressure_data) / (top_5_avg + 1e-6)    # 最大值与top5比
        ])
        
        return np.array(features)
    
    def extract_all_features(self, pressure_data):
        """提取所有特征"""
        stat_features = self.extract_statistical_features(pressure_data)
        spatial_features = self.extract_spatial_features(pressure_data)
        peak_features = self.extract_peak_features(pressure_data)
        
        return np.concatenate([stat_features, spatial_features, peak_features])
    
    def load_data(self, csv_file='../data/dataset.csv'):
        """加载数据"""
        print(f"📂 加载数据: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except:
            df = pd.read_csv(csv_file, encoding='gbk')
        
        # 分离标签和特征
        labels = df['Label'].values
        pressure_data = df.drop('Label', axis=1).values
        
        print(f"   - 总样本数: {len(pressure_data)}")
        print(f"   - 原始特征维度: {pressure_data.shape[1]}")
        
        # 提取嵌入式友好的特征
        print(f"   - 提取嵌入式特征...")
        features = []
        for i, data in enumerate(pressure_data):
            if i % 1000 == 0:
                print(f"     处理进度: {i}/{len(pressure_data)}")
            feature_vector = self.extract_all_features(data)
            features.append(feature_vector)
        
        features = np.array(features)
        print(f"   - 提取特征维度: {features.shape[1]}")
        
        return features, labels
    
    def load_test_data(self, csv_file):
        """加载测试数据"""
        print(f"📂 加载测试数据: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except:
            df = pd.read_csv(csv_file, encoding='gbk')
        
        # 检查是否有Label列
        if 'Label' in df.columns:
            labels = df['Label'].values
            pressure_data = df.drop('Label', axis=1).values
        else:
            labels = None
            pressure_data = df.values
        
        print(f"   - 总样本数: {len(pressure_data)}")
        print(f"   - 原始特征维度: {pressure_data.shape[1]}")
        print(f"   - 有标签: {'是' if labels is not None else '否'}")
        
        # 提取嵌入式友好的特征
        print(f"   - 提取嵌入式特征...")
        features = []
        for i, data in enumerate(pressure_data):
            if i % 100 == 0 and len(pressure_data) > 100:
                print(f"     处理进度: {i}/{len(pressure_data)}")
            feature_vector = self.extract_all_features(data)
            features.append(feature_vector)
        
        features = np.array(features)
        print(f"   - 提取特征维度: {features.shape[1]}")
        
        return features, labels
    
    def train_lightweight_models(self, features, labels):
        """训练多个轻量级模型"""
        print(f"\n🏋️ 训练轻量级模型...")
        
        # 编码标签
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # 定义候选模型
        candidate_models = {
            'decision_tree': DecisionTreeClassifier(
                max_depth=8, 
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'random_forest_small': RandomForestClassifier(
                n_estimators=10,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'knn_3': KNeighborsClassifier(n_neighbors=3),
            'knn_5': KNeighborsClassifier(n_neighbors=5),
        }
        
        # 训练和评估每个模型
        results = {}
        
        for name, model in candidate_models.items():
            print(f"\n📊 训练 {name}...")
            
            # 选择合适的数据（KNN和Logistic回归需要标准化）
            if name in ['logistic_regression', 'knn_3', 'knn_5']:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test
            
            # 训练
            start_time = time.time()
            model.fit(X_train_use, y_train)
            train_time = time.time() - start_time
            
            # 预测
            start_time = time.time()
            y_pred = model.predict(X_test_use)
            predict_time = time.time() - start_time
            
            # 评估
            accuracy = accuracy_score(y_test, y_pred)
            
            # 估算模型大小
            model_size = self.estimate_model_size(model, name)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'train_time': train_time,
                'predict_time': predict_time,
                'model_size_kb': model_size,
                'predictions': y_pred
            }
            
            print(f"   ✅ 准确率: {accuracy:.4f}")
            print(f"   ⏱️ 训练时间: {train_time:.2f}s")
            print(f"   🚀 预测时间: {predict_time*1000:.2f}ms")
            print(f"   💾 估计大小: {model_size:.1f} KB")
        
        # 详细报告
        print(f"\n📋 详细分类报告:")
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print(classification_report(y_test, result['predictions'], 
                                      target_names=self.label_encoder.classes_))
        
        self.models = {name: result['model'] for name, result in results.items()}
        return results
    
    def estimate_model_size(self, model, model_name):
        """估算模型大小（KB）"""
        if model_name == 'decision_tree':
            # 决策树大小估算
            tree = model.tree_
            n_nodes = tree.node_count
            # 每个节点大约需要：特征索引(4字节) + 阈值(4字节) + 左右子节点(8字节) = 16字节
            return n_nodes * 16 / 1024
        
        elif 'random_forest' in model_name:
            # 随机森林大小估算
            total_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
            return total_nodes * 16 / 1024
        
        elif model_name == 'logistic_regression':
            # 逻辑回归：权重矩阵大小
            n_features = model.coef_.shape[1]
            n_classes = model.coef_.shape[0]
            return (n_features * n_classes + n_classes) * 4 / 1024  # 4字节/float
        
        elif 'knn' in model_name:
            # KNN：存储训练数据
            n_samples, n_features = model._fit_X.shape
            return n_samples * n_features * 4 / 1024
        
        else:
            return 0
    
    def generate_c_code(self, model_name='decision_tree', output_dir='../embedded/'):
        """生成C代码用于STM32"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if model_name not in self.models:
            print(f"❌ 模型 {model_name} 不存在")
            return
        
        model = self.models[model_name]
        
        if model_name == 'decision_tree':
            self.generate_decision_tree_c_code(model, output_dir)
        elif model_name == 'logistic_regression':
            self.generate_logistic_regression_c_code(model, output_dir)
        else:
            print(f"❌ 暂不支持 {model_name} 的C代码生成")
    
    def generate_decision_tree_c_code(self, model, output_dir):
        """生成决策树的C代码"""
        tree = model.tree_
        
        # 生成头文件
        header_content = f"""
#ifndef PRESSURE_CLASSIFIER_H
#define PRESSURE_CLASSIFIER_H

#include <stdint.h>

// 特征数量
#define N_FEATURES {tree.n_features}

// 类别数量
#define N_CLASSES {tree.n_classes[0]}

// 节点数量
#define N_NODES {tree.node_count}

// 类别标签
typedef enum {{
    CLASS_LEFT = 0,
    CLASS_NORMAL = 1,
    CLASS_RIGHT = 2
}} class_t;

// 决策树节点结构
typedef struct {{
    int16_t feature;        // 特征索引 (-1表示叶子节点)
    float threshold;        // 阈值
    int16_t left_child;     // 左子节点索引
    int16_t right_child;    // 右子节点索引
    int16_t class_id;       // 类别ID（仅叶子节点有效）
}} tree_node_t;

// 函数声明
float extract_statistical_features(const uint16_t* pressure_data, float* features);
float extract_spatial_features(const uint16_t* pressure_data, float* features);
float extract_peak_features(const uint16_t* pressure_data, float* features);
void extract_all_features(const uint16_t* pressure_data, float* features);
class_t classify_pressure_data(const uint16_t* pressure_data);
int predict_with_confidence(const uint16_t* pressure_data, float* confidence);

#endif // PRESSURE_CLASSIFIER_H
"""
        
        # 生成C实现文件
        c_content = f"""
#include "pressure_classifier.h"
#include <math.h>
#include <string.h>

// 决策树节点数据
static const tree_node_t tree_nodes[N_NODES] = {{
"""
        
        # 生成节点数据
        for i in range(tree.node_count):
            feature = tree.feature[i]
            threshold = tree.threshold[i]
            left_child = tree.children_left[i]
            right_child = tree.children_right[i]
            
            if feature == -2:  # 叶子节点
                class_id = np.argmax(tree.value[i][0])
                c_content += f"    {{-1, 0.0f, -1, -1, {class_id}}}, // Node {i} (leaf)\n"
            else:
                c_content += f"    {{{feature}, {threshold:.6f}f, {left_child}, {right_child}, -1}}, // Node {i}\n"
        
        c_content += f"""
}};

// 提取统计特征
void extract_statistical_features(const uint16_t* pressure_data, float* features) {{
    float sum = 0, sum_sq = 0;
    uint16_t min_val = 65535, max_val = 0;
    uint16_t non_zero_count = 0;
    
    // 计算基础统计量
    for (int i = 0; i < 256; i++) {{
        uint16_t val = pressure_data[i];
        sum += val;
        sum_sq += val * val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        if (val > 0) non_zero_count++;
    }}
    
    float mean = sum / 256.0f;
    float variance = (sum_sq / 256.0f) - (mean * mean);
    float std_dev = sqrtf(variance);
    
    features[0] = mean;                    // 平均值
    features[1] = std_dev;                 // 标准差
    features[2] = min_val;                 // 最小值
    features[3] = max_val;                 // 最大值
    features[4] = sum;                     // 总和
    features[5] = non_zero_count;          // 非零点数量
    features[6] = max_val - min_val;       // 极差
    
    // 计算中位数（简化版本）
    features[7] = mean;  // 用均值近似中位数
}}

// 提取空间特征
void extract_spatial_features(const uint16_t* pressure_data, float* features) {{
    float total_pressure = 0;
    float center_x = 0, center_y = 0;
    
    // 计算总压力和重心
    for (int y = 0; y < 16; y++) {{
        for (int x = 0; x < 16; x++) {{
            float val = pressure_data[y * 16 + x];
            total_pressure += val;
            center_x += x * val;
            center_y += y * val;
        }}
    }}
    
    if (total_pressure > 0) {{
        center_x /= total_pressure;
        center_y /= total_pressure;
    }} else {{
        center_x = center_y = 8.0f;
    }}
    
    features[0] = center_x;
    features[1] = center_y;
    
    // 区域压力分布
    float left_pressure = 0, right_pressure = 0;
    float top_pressure = 0, bottom_pressure = 0;
    
    for (int y = 0; y < 16; y++) {{
        for (int x = 0; x < 16; x++) {{
            float val = pressure_data[y * 16 + x];
            if (x < 8) left_pressure += val;
            else right_pressure += val;
            if (y < 8) top_pressure += val;
            else bottom_pressure += val;
        }}
    }}
    
    if (total_pressure > 0) {{
        features[2] = left_pressure / total_pressure;
        features[3] = right_pressure / total_pressure;
        features[4] = top_pressure / total_pressure;
        features[5] = bottom_pressure / total_pressure;
    }} else {{
        features[2] = features[3] = features[4] = features[5] = 0.25f;
    }}
    
    // 对称性特征
    features[6] = left_pressure / (right_pressure + 1e-6f);
    features[7] = top_pressure / (bottom_pressure + 1e-6f);
}}

// 提取峰值特征  
void extract_peak_features(const uint16_t* pressure_data, float* features) {{
    // 简化的峰值特征提取
    uint16_t max_val = 0;
    float sum = 0;
    
    for (int i = 0; i < 256; i++) {{
        if (pressure_data[i] > max_val) max_val = pressure_data[i];
        sum += pressure_data[i];
    }}
    
    float mean = sum / 256.0f;
    
    features[0] = max_val;                    // 最大值
    features[1] = max_val;                    // top5平均（简化为最大值）
    features[2] = max_val / (mean + 1e-6f);   // 峰值与均值比
    features[3] = 1.0f;                       // 简化为1
}}

// 提取所有特征
void extract_all_features(const uint16_t* pressure_data, float* features) {{
    extract_statistical_features(pressure_data, features);
    extract_spatial_features(pressure_data, features + 8);
    extract_peak_features(pressure_data, features + 16);
}}

// 分类函数
class_t classify_pressure_data(const uint16_t* pressure_data) {{
    float features[20];  // 总共20个特征
    extract_all_features(pressure_data, features);
    
    // 遍历决策树
    int node_id = 0;  // 从根节点开始
    
    while (tree_nodes[node_id].feature != -1) {{  // 不是叶子节点
        int feature_idx = tree_nodes[node_id].feature;
        float threshold = tree_nodes[node_id].threshold;
        
        if (features[feature_idx] <= threshold) {{
            node_id = tree_nodes[node_id].left_child;
        }} else {{
            node_id = tree_nodes[node_id].right_child;
        }}
    }}
    
    return (class_t)tree_nodes[node_id].class_id;
}}

// 带置信度的预测
int predict_with_confidence(const uint16_t* pressure_data, float* confidence) {{
    class_t result = classify_pressure_data(pressure_data);
    
    // 简化的置信度计算（基于数据质量）
    float sum = 0;
    for (int i = 0; i < 256; i++) {{
        sum += pressure_data[i];
    }}
    
    if (sum > 10000) {{
        *confidence = 0.9f;  // 高质量数据
    }} else if (sum > 1000) {{
        *confidence = 0.7f;  // 中等质量数据
    }} else {{
        *confidence = 0.5f;  // 低质量数据
    }}
    
    return result;
}}
"""
        
        # 保存文件
        with open(f"{output_dir}/pressure_classifier.h", 'w') as f:
            f.write(header_content)
        
        with open(f"{output_dir}/pressure_classifier.c", 'w') as f:
            f.write(c_content)
        
        print(f"✅ C代码已生成:")
        print(f"   - {output_dir}/pressure_classifier.h")
        print(f"   - {output_dir}/pressure_classifier.c")
        print(f"   - 树节点数量: {tree.node_count}")
        print(f"   - 特征数量: {tree.n_features}")
    
    def save_models(self, output_dir='../embedded/'):
        """保存模型文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存sklearn模型
        for name, model in self.models.items():
            model_file = f"{output_dir}/{name}_model.pkl"
            joblib.dump(model, model_file)
            print(f"✅ 已保存: {model_file}")
        
        # 保存标准化器
        if self.scalers:
            scaler_file = f"{output_dir}/scaler.pkl"
            joblib.dump(self.scalers['standard'], scaler_file)
            print(f"✅ 已保存: {scaler_file}")
        
        # 保存标签编码器
        if self.label_encoder:
            le_file = f"{output_dir}/label_encoder.pkl"
            joblib.dump(self.label_encoder, le_file)
            print(f"✅ 已保存: {le_file}")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("🔧 STM32嵌入式模型设计工具")
        print("=" * 50)
        print("📖 使用方法:")
        print("   python stm32_model_designer.py train     - 训练轻量级模型")
        print("   python stm32_model_designer.py generate  - 生成C代码")
        print("   python stm32_model_designer.py test      - 测试模型")
        return
    
    command = sys.argv[1]
    designer = STM32ModelDesigner()
    
    if command == 'train':
        print("🚀 训练STM32嵌入式模型")
        
        # 加载数据
        features, labels = designer.load_data('../data/dataset.csv')
        
        # 训练模型
        results = designer.train_lightweight_models(features, labels)
        
        # 保存模型
        designer.save_models()
        
        # 显示总结
        print(f"\n🎯 训练总结:")
        print(f"{'模型名称':<20} {'准确率':<10} {'大小(KB)':<10} {'预测时间(ms)':<15}")
        print("-" * 55)
        for name, result in results.items():
            print(f"{name:<20} {result['accuracy']:<10.3f} {result['model_size_kb']:<10.1f} {result['predict_time']*1000:<15.2f}")
        
        # 推荐最佳模型
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🌟 推荐模型: {best_model[0]} (准确率: {best_model[1]['accuracy']:.3f})")
        
    elif command == 'generate':
        print("🔧 生成C代码")
        
        # 先尝试加载已训练的模型
        try:
            designer.models['decision_tree'] = joblib.load('../embedded/decision_tree_model.pkl')
            designer.generate_c_code('decision_tree')
        except:
            print("❌ 请先运行训练: python stm32_model_designer.py train")
    
    elif command == 'test':
        if len(sys.argv) < 3:
            print("❌ 用法: python stm32_model_designer.py test <测试数据文件> [期望标签]")
            sys.exit(1)
        
        test_file = sys.argv[2]
        expected_label = sys.argv[3] if len(sys.argv) > 3 else None
        
        print(f"🧪 测试模型 - 数据文件: {test_file}")
        if expected_label:
            print(f"   期望标签: {expected_label}")
        
        # 加载测试数据
        test_features, test_labels = designer.load_test_data(test_file)
        print(f"📊 测试数据: {len(test_features)} 个样本")
        
        # 如果没有真实标签但提供了期望标签，创建标签数组
        if test_labels is None and expected_label:
            test_labels = [expected_label] * len(test_features)
            print(f"   使用期望标签 '{expected_label}' 作为真实标签")
        
        # 测试所有可用的模型
        results = {}
        
        # 1. 尝试加载并测试决策树模型
        try:
            decision_tree = joblib.load('../embedded/decision_tree_model.pkl')
            label_encoder = joblib.load('../embedded/label_encoder.pkl')
            
            predictions = decision_tree.predict(test_features)
            
            if test_labels is not None:
                # 编码真实标签进行比较
                encoded_test_labels = label_encoder.transform(test_labels)
                accuracy = accuracy_score(encoded_test_labels, predictions)
                results['决策树'] = accuracy
                print(f"✅ 决策树模型准确率: {accuracy:.3f}")
            else:
                print(f"✅ 决策树模型预测结果:")
            
            # 显示详细统计
            from collections import Counter
            # 解码预测结果以便显示
            decoded_predictions = label_encoder.inverse_transform(predictions)
            pred_counts = Counter(decoded_predictions)
            print(f"   预测分布: {dict(pred_counts)}")
            
            if test_labels is not None:
                true_counts = Counter(test_labels)
                print(f"   真实分布: {dict(true_counts)}")
            
        except Exception as e:
            print(f"❌ 决策树模型加载失败: {e}")
        
        # 2. 尝试加载并测试统计特征模型  
        try:
            # 首先需要加载标签编码器和缩放器
            label_encoder = joblib.load('../embedded/label_encoder.pkl')
            scaler = joblib.load('../embedded/scaler.pkl')
            
            # 查找统计特征模型
            import os
            stat_models = [f for f in os.listdir('../embedded') if 'model.pkl' in f and f != 'decision_tree_model.pkl']
            
            for model_file in stat_models:
                try:
                    model_name = model_file.replace('_model.pkl', '').replace('_', ' ')
                    stat_model = joblib.load(f'../embedded/{model_file}')
                    
                    # 标准化特征
                    test_features_scaled = scaler.transform(test_features)
                    predictions = stat_model.predict(test_features_scaled)
                    
                    if test_labels is not None:
                        # 编码真实标签进行比较
                        encoded_test_labels = label_encoder.transform(test_labels)
                        accuracy = accuracy_score(encoded_test_labels, predictions)
                        results[model_name] = accuracy
                        print(f"✅ {model_name}模型准确率: {accuracy:.3f}")
                    else:
                        print(f"✅ {model_name}模型预测结果:")
                        # 解码预测结果
                        decoded_predictions = label_encoder.inverse_transform(predictions)
                        pred_counts = Counter(decoded_predictions)
                        print(f"   预测分布: {dict(pred_counts)}")
                    
                except Exception as e:
                    print(f"❌ {model_name}模型测试失败: {e}")
                    
        except Exception as e:
            print(f"❌ 统计特征模型相关文件加载失败: {e}")
        
        # 3. 如果有原始CNN模型，也进行对比测试
        try:
            import tensorflow as tf
            cnn_model = tf.keras.models.load_model('../models/cnn_augmented_model.keras')
            
            # 重塑数据以适应CNN输入格式 (原始256维压力数据)
            # 需要使用原始压力数据而不是提取的特征
            df_raw = pd.read_csv(test_file)
            raw_data = df_raw.values  # 原始256维数据
            test_features_cnn = raw_data.reshape(-1, 256, 1)
            
            predictions = cnn_model.predict(test_features_cnn, verbose=0)
            predictions = (predictions > 0.5).astype(int).flatten()
            
            if test_labels is not None:
                # CNN期望的是二进制标签
                binary_test_labels = [1 if label == 'crossleg' else 0 for label in test_labels]
                accuracy = accuracy_score(binary_test_labels, predictions)
                results['原始CNN'] = accuracy
                print(f"✅ 原始CNN模型准确率: {accuracy:.3f}")
            else:
                print(f"✅ 原始CNN模型预测结果:")
                pred_labels = ['crossleg' if p == 1 else 'normal' for p in predictions]
                pred_counts = Counter(pred_labels)
                print(f"   预测分布: {dict(pred_counts)}")
            
        except Exception as e:
            print(f"❌ CNN模型加载失败: {e}")
        
        # 显示测试总结
        if results:
            print(f"\n📊 测试总结:")
            print(f"{'模型类型':<15} {'准确率':<10}")
            print("-" * 25)
            for model_name, accuracy in results.items():
                print(f"{model_name:<15} {accuracy:<10.3f}")
            
            # 找出最佳模型
            best_model = max(results.items(), key=lambda x: x[1])
            print(f"\n🌟 最佳模型: {best_model[0]} (准确率: {best_model[1]:.3f})")
            
            # 如果准确率较低，给出建议
            if best_model[1] < 0.8:
                print(f"\n⚠️  准确率较低，建议:")
                print(f"   1. 检查测试数据质量")
                print(f"   2. 增加训练数据")
                print(f"   3. 调整模型参数")
        else:
            print("\n❌ 没有可用的训练模型，请先运行训练")
    
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == "__main__":
    main()