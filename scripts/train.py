#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐姿分类模型训练脚本
整合了基础训练和改进训练功能
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PostureTrainer:
    """坐姿分类训练器"""
    
    def __init__(self, data_file='dataset.csv', mode='standard'):
        """
        初始化训练器
        
        Args:
            data_file: 训练数据文件路径
            mode: 训练模式 ('standard' 或 'improved')
        """
        self.data_file = data_file
        self.mode = mode
        self.model = None
        self.scaler = None
        self.pca = None
        
    def load_data(self):
        """加载训练数据"""
        try:
            # 尝试不同编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(self.data_file, encoding=encoding)
                    print(f"✅ 使用 {encoding} 编码成功读取数据")
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
                print(f"     {label}: {count} 样本")
            
            return X, y
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return None, None
    
    def preprocess_data(self, X, y, test_size=0.2):
        """数据预处理"""
        print(f"\n🔄 数据预处理 (模式: {self.mode})...")
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # PCA降维
        if self.mode == 'improved':
            # 改进模式：保留更多方差，避免信息丢失
            n_components = min(50, X_train_scaled.shape[1])
        else:
            # 标准模式：原始设置
            n_components = 20
            
        self.pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"   ✅ PCA降维: {X.shape[1]} → {n_components} 维")
        print(f"   ✅ 保留方差: {explained_variance:.3f} ({explained_variance*100:.1f}%)")
        
        return X_train_pca, X_test_pca, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """训练模型"""
        print(f"\n🤖 模型训练 (模式: {self.mode})...")
        
        if self.mode == 'improved':
            # 改进模式：网格搜索最优参数
            print("   🔍 网格搜索最优参数...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            }
            
            svm = SVC(random_state=42)
            grid_search = GridSearchCV(
                svm, param_grid, cv=5, scoring='accuracy',
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            print(f"   ✅ 最优参数: {grid_search.best_params_}")
            print(f"   ✅ 交叉验证得分: {grid_search.best_score_:.4f}")
            
        else:
            # 标准模式：固定参数
            self.model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
            self.model.fit(X_train, y_train)
            print("   ✅ 使用固定参数训练完成")
        
        return self.model
    
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """评估模型"""
        print(f"\n📊 模型评估...")
        
        # 训练集评估
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # 测试集评估
        y_test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"   训练集准确率: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"   测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # 过拟合检测
        overfitting = train_accuracy - test_accuracy
        print(f"   过拟合程度: {overfitting:.4f}")
        
        if overfitting > 0.1:
            print("   ⚠️  模型可能过拟合")
        elif overfitting > 0.05:
            print("   ⚡ 轻微过拟合")
        else:
            print("   ✅ 泛化能力良好")
        
        # 详细报告
        print(f"\n📋 分类报告:")
        print(classification_report(y_test, y_test_pred))
        
        # 混淆矩阵
        self.plot_confusion_matrix(y_test, y_test_pred)
        
        return train_accuracy, test_accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['left', 'normal', 'right'],
                       yticklabels=['left', 'normal', 'right'])
            plt.title('混淆矩阵')
            plt.ylabel('实际类别')
            plt.xlabel('预测类别')
            
            filename = f'confusion_matrix_{self.mode}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ 混淆矩阵已保存: {filename}")
            
        except Exception as e:
            print(f"   ⚠️  混淆矩阵绘制失败: {e}")
    
    def save_models(self):
        """保存模型和预处理器"""
        print(f"\n💾 保存模型...")
        
        suffix = '_improved' if self.mode == 'improved' else ''
        
        try:
            # 保存模型
            model_file = f'model_svm{suffix}.pkl'
            joblib.dump(self.model, model_file)
            print(f"   ✅ SVM模型已保存: {model_file}")
            
            # 保存标准化器
            scaler_file = f'scaler{suffix}.pkl'
            joblib.dump(self.scaler, scaler_file)
            print(f"   ✅ 标准化器已保存: {scaler_file}")
            
            # 保存PCA
            pca_file = f'pca{suffix}.pkl'
            if self.mode == 'improved':
                joblib.dump(self.pca, pca_file)
                print(f"   ✅ PCA降维器已保存: {pca_file}")
            else:
                joblib.dump(self.pca, 'pca.pkl')
                print(f"   ✅ PCA降维器已保存: pca.pkl")
                
        except Exception as e:
            print(f"   ❌ 模型保存失败: {e}")
    
    def train_full_pipeline(self):
        """完整训练流程"""
        print(f"🚀 开始坐姿分类模型训练 (模式: {self.mode})...")
        
        # 1. 加载数据
        X, y = self.load_data()
        if X is None:
            return False
        
        # 2. 预处理
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
        
        # 3. 训练模型
        self.train_model(X_train, y_train)
        
        # 4. 评估模型
        self.evaluate_model(X_train, X_test, y_train, y_test)
        
        # 5. 保存模型
        self.save_models()
        
        print(f"\n🎉 训练完成！")
        return True

def main():
    """主函数"""
    import sys
    
    # 解析命令行参数
    mode = 'standard'
    data_file = 'dataset.csv'
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['standard', 'improved']:
            mode = sys.argv[1]
        else:
            print("使用方法: python train.py [standard|improved] [data_file]")
            return
    
    if len(sys.argv) > 2:
        data_file = sys.argv[2]
    
    print(f"训练模式: {mode}")
    print(f"数据文件: {data_file}")
    
    # 开始训练
    trainer = PostureTrainer(data_file=data_file, mode=mode)
    success = trainer.train_full_pipeline()
    
    if success:
        print(f"\n✅ 训练成功完成!")
        if mode == 'improved':
            print("💡 建议使用 python evaluate.py 进行进一步评估")
    else:
        print(f"\n❌ 训练失败!")

if __name__ == "__main__":
    main()