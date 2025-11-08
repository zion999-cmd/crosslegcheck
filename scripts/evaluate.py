#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐姿分类模型评估脚本
整合了数据集分析、模型性能评估、测试集验证等功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PostureEvaluator:
    """坐姿分类评估器"""
    
    def __init__(self, model_type='standard'):
        """
        初始化评估器
        
        Args:
            model_type: 模型类型 ('standard' 或 'improved')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.pca = None
        
        # 设置文件路径
        suffix = '_improved' if model_type == 'improved' else ''
        self.model_path = f'model_svm{suffix}.pkl'
        self.scaler_path = f'scaler{suffix}.pkl'
        self.pca_path = f'pca{suffix}.pkl' if model_type == 'improved' else 'pca.pkl'
    
    def load_models(self):
        """加载模型和预处理器"""
        try:
            if Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                print(f"✅ SVM模型加载成功: {self.model_path}")
            else:
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            if Path(self.scaler_path).exists():
                self.scaler = joblib.load(self.scaler_path)
                print(f"✅ 标准化器加载成功: {self.scaler_path}")
            else:
                raise FileNotFoundError(f"标准化器文件不存在: {self.scaler_path}")
            
            if Path(self.pca_path).exists():
                self.pca = joblib.load(self.pca_path)
                print(f"✅ PCA降维器加载成功: {self.pca_path}")
            else:
                raise FileNotFoundError(f"PCA文件不存在: {self.pca_path}")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def load_data(self, csv_file):
        """加载数据"""
        try:
            # 尝试不同编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    print(f"✅ 使用 {encoding} 编码成功读取文件: {csv_file}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("无法读取CSV文件")
            
            print(f"📊 数据加载成功:")
            print(f"   - 总样本数: {len(df)}")
            print(f"   - 特征维度: {df.shape[1] - 1}")
            
            # 分离特征和标签
            if 'Label' in df.columns:
                X = df.drop('Label', axis=1).values
                y = df['Label'].values
            else:
                raise ValueError("数据必须包含Label列")
            
            # 统计类别分布
            unique, counts = np.unique(y, return_counts=True)
            print(f"   - 类别分布:")
            for label, count in zip(unique, counts):
                print(f"     {label}: {count} 样本 ({count/len(y)*100:.1f}%)")
            
            return X, y
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return None, None
    
    def analyze_dataset(self, csv_file):
        """分析数据集"""
        print(f"🔍 数据集分析: {csv_file}")
        
        X, y = self.load_data(csv_file)
        if X is None:
            return
        
        # 基本统计
        print(f"\n📊 数据统计:")
        print(f"   - 特征均值: {np.mean(X):.2f}")
        print(f"   - 特征标准差: {np.std(X):.2f}")
        print(f"   - 特征最小值: {np.min(X):.2f}")
        print(f"   - 特征最大值: {np.max(X):.2f}")
        
        # 检查数据质量
        zero_features = np.sum(np.all(X == 0, axis=0))
        constant_features = np.sum(np.var(X, axis=0) == 0)
        
        print(f"   - 全零特征: {zero_features}")
        print(f"   - 常数特征: {constant_features}")
        
        # 类别平衡性
        unique, counts = np.unique(y, return_counts=True)
        max_ratio = np.max(counts) / np.min(counts)
        print(f"   - 类别不平衡比例: {max_ratio:.2f}")
        
        if max_ratio > 2:
            print("   ⚠️  数据不平衡，建议使用平衡技术")
        else:
            print("   ✅ 数据相对平衡")
        
        # 绘制数据分布图
        self.plot_data_distribution(X, y, csv_file)
    
    def plot_data_distribution(self, X, y, title_suffix=""):
        """绘制数据分布图"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 类别分布
            unique, counts = np.unique(y, return_counts=True)
            axes[0, 0].bar(unique, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0, 0].set_title('类别分布')
            axes[0, 0].set_xlabel('类别')
            axes[0, 0].set_ylabel('样本数')
            
            # 2. 特征分布热图（前100个特征）
            sample_features = X[:, :100] if X.shape[1] > 100 else X
            im = axes[0, 1].imshow(sample_features[:50].T, cmap='viridis', aspect='auto')
            axes[0, 1].set_title('特征热图 (前50样本)')
            axes[0, 1].set_xlabel('样本')
            axes[0, 1].set_ylabel('特征')
            plt.colorbar(im, ax=axes[0, 1])
            
            # 3. 特征统计
            feature_means = np.mean(X, axis=0)
            axes[1, 0].plot(feature_means)
            axes[1, 0].set_title('各特征均值')
            axes[1, 0].set_xlabel('特征索引')
            axes[1, 0].set_ylabel('均值')
            
            # 4. 类别间特征对比（使用前10个特征的均值）
            for label in unique:
                mask = y == label
                class_means = np.mean(X[mask, :10], axis=0)
                axes[1, 1].plot(class_means, label=f'{label}', marker='o')
            
            axes[1, 1].set_title('类别间特征对比 (前10个特征)')
            axes[1, 1].set_xlabel('特征索引')
            axes[1, 1].set_ylabel('均值')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # 保存图片
            filename = f'data_analysis_{title_suffix}.png'.replace('.csv', '').replace(' ', '_')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ 数据分析图已保存: {filename}")
            
        except Exception as e:
            print(f"   ⚠️  数据分布图绘制失败: {e}")
    
    def evaluate_on_dataset(self, csv_file):
        """在指定数据集上评估模型"""
        print(f"\n🎯 模型评估: {csv_file}")
        
        # 加载模型
        self.load_models()
        
        # 加载数据
        X, y = self.load_data(csv_file)
        if X is None:
            return
        
        # 预处理数据
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # 预测
        y_pred = self.model.predict(X_pca)
        
        # 计算准确率
        accuracy = accuracy_score(y, y_pred)
        print(f"\n📈 评估结果:")
        print(f"   总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 详细分类报告
        print(f"\n📋 详细分类报告:")
        print(classification_report(y, y_pred))
        
        # 混淆矩阵
        self.plot_confusion_matrix(y, y_pred, csv_file)
        
        # 错误分析
        self.analyze_errors(y, y_pred)
        
        # 泛化能力评估
        self.assess_generalization(accuracy)
        
        return accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred, title_suffix=""):
        """绘制混淆矩阵"""
        try:
            cm = confusion_matrix(y_true, y_pred, labels=['left', 'normal', 'right'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['left', 'normal', 'right'],
                       yticklabels=['left', 'normal', 'right'])
            plt.title(f'混淆矩阵 - {title_suffix}')
            plt.ylabel('实际类别')
            plt.xlabel('预测类别')
            
            filename = f'confusion_matrix_{title_suffix}.png'.replace('.csv', '').replace(' ', '_')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ 混淆矩阵已保存: {filename}")
            
        except Exception as e:
            print(f"   ⚠️  混淆矩阵绘制失败: {e}")
    
    def analyze_errors(self, y_true, y_pred):
        """分析预测错误"""
        errors = np.where(y_true != y_pred)[0]
        
        if len(errors) == 0:
            print(f"\n✅ 完美预测，没有错误！")
            return
        
        print(f"\n❌ 错误分析 (共{len(errors)}个错误):")
        
        # 统计错误类型
        error_types = {}
        for idx in errors:
            error_key = f"{y_true[idx]} → {y_pred[idx]}"
            error_types[error_key] = error_types.get(error_key, 0) + 1
        
        # 按错误次数排序
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        
        for error_type, count in sorted_errors:
            percentage = count / len(errors) * 100
            print(f"   {error_type}: {count} 次 ({percentage:.1f}%)")
        
        # 分析最常见的错误
        if sorted_errors:
            most_common_error = sorted_errors[0]
            print(f"\n💡 最常见错误: {most_common_error[0]} (占{most_common_error[1]/len(errors)*100:.1f}%)")
            
            # 给出改进建议
            error_from, error_to = most_common_error[0].split(' → ')
            if error_from in ['left', 'right'] and error_to in ['left', 'right']:
                print("   建议: left和right类别容易混淆，可能需要更多区分性特征")
            elif 'normal' in [error_from, error_to]:
                print("   建议: normal类别的界定可能需要调整")
    
    def assess_generalization(self, accuracy):
        """评估泛化能力"""
        print(f"\n🔍 泛化能力评估:")
        
        if accuracy >= 0.99:
            print("   ⚠️  准确率过高，可能存在过拟合")
            print("   建议: 使用更多独立测试数据验证")
        elif accuracy >= 0.95:
            print("   ✅ 泛化能力优秀")
        elif accuracy >= 0.90:
            print("   ✅ 泛化能力良好")
        elif accuracy >= 0.80:
            print("   ⚡ 泛化能力中等，有改进空间")
            print("   建议: 调整模型参数或增加训练数据")
        else:
            print("   ❌ 泛化能力较差，需要重新训练")
            print("   建议: 检查数据质量、特征工程或模型选择")
    
    def compare_models(self, test_file):
        """比较standard和improved模型"""
        print(f"🆚 模型对比分析")
        
        results = {}
        
        for model_type in ['standard', 'improved']:
            try:
                print(f"\n--- 评估 {model_type} 模型 ---")
                evaluator = PostureEvaluator(model_type)
                accuracy = evaluator.evaluate_on_dataset(test_file)
                results[model_type] = accuracy
                
            except Exception as e:
                print(f"❌ {model_type} 模型评估失败: {e}")
                results[model_type] = None
        
        # 结果对比
        print(f"\n📊 模型对比结果:")
        for model_type, accuracy in results.items():
            if accuracy is not None:
                print(f"   {model_type}: {accuracy:.4f} ({accuracy*100:.2f}%)")
            else:
                print(f"   {model_type}: 评估失败")
        
        # 推荐最佳模型
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_model = max(valid_results, key=valid_results.get)
            print(f"\n🏆 推荐使用模型: {best_model}")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python evaluate.py analyze <csv_file>           # 数据集分析")
        print("  python evaluate.py test <csv_file>              # 模型测试")
        print("  python evaluate.py compare <csv_file>           # 模型对比")
        print("  python evaluate.py [--improved] <mode> <file>   # 使用改进模型")
        return
    
    # 解析参数
    args = sys.argv[1:]
    model_type = 'standard'
    
    if args[0] == '--improved':
        model_type = 'improved'
        args = args[1:]
    
    if len(args) < 2:
        print("❌ 缺少参数")
        return
    
    mode = args[0]
    csv_file = args[1]
    
    if not Path(csv_file).exists():
        print(f"❌ 文件不存在: {csv_file}")
        return
    
    try:
        evaluator = PostureEvaluator(model_type)
        
        if mode == 'analyze':
            # 数据集分析
            evaluator.analyze_dataset(csv_file)
            
        elif mode == 'test':
            # 模型测试
            evaluator.evaluate_on_dataset(csv_file)
            
        elif mode == 'compare':
            # 模型对比
            evaluator.compare_models(csv_file)
            
        else:
            print(f"❌ 未知的评估模式: {mode}")
            
    except Exception as e:
        print(f"❌ 评估失败: {e}")

if __name__ == "__main__":
    main()