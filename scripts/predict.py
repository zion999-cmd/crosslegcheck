#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐姿分类预测脚本
整合了单个预测、批量预测、CSV文件预测等功能
"""

import numpy as np
import joblib
import pandas as pd
from pathlib import Path
import warnings
import sys

warnings.filterwarnings('ignore')

class PosturePredictor:
    """坐姿分类预测器"""
    
    def __init__(self, model_type='standard'):
        """
        初始化预测器
        
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
        
        # 加载模型
        self.load_models()
    
    def load_models(self):
        """加载训练好的模型和预处理器"""
        try:
            # 加载SVM模型
            if Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                print(f"✅ SVM模型加载成功: {self.model_path}")
            else:
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 加载标准化器
            if Path(self.scaler_path).exists():
                self.scaler = joblib.load(self.scaler_path)
                print(f"✅ 标准化器加载成功: {self.scaler_path}")
            else:
                raise FileNotFoundError(f"标准化器文件不存在: {self.scaler_path}")
            
            # 加载PCA
            if Path(self.pca_path).exists():
                self.pca = joblib.load(self.pca_path)
                print(f"✅ PCA降维器加载成功: {self.pca_path}")
            else:
                raise FileNotFoundError(f"PCA文件不存在: {self.pca_path}")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def preprocess_data(self, pressure_data):
        """预处理单个样本数据"""
        try:
            # 确保输入是numpy数组
            if isinstance(pressure_data, list):
                pressure_data = np.array(pressure_data)
            
            # 检查维度
            if pressure_data.shape[0] != 256:
                raise ValueError(f"输入数据维度错误，期望256维，实际{pressure_data.shape[0]}维")
            
            # 重塑为2D数组
            pressure_data = pressure_data.reshape(1, -1)
            
            # 标准化
            pressure_data_scaled = self.scaler.transform(pressure_data)
            
            # PCA降维
            pressure_data_pca = self.pca.transform(pressure_data_scaled)
            
            return pressure_data_pca
            
        except Exception as e:
            print(f"数据预处理失败: {e}")
            return None
    
    def predict_single(self, pressure_data):
        """
        预测单个样本
        
        Args:
            pressure_data: 256维压力数据数组
            
        Returns:
            tuple: (预测类别, 置信度)
        """
        try:
            # 预处理数据
            processed_data = self.preprocess_data(pressure_data)
            if processed_data is None:
                return None, None
            
            # 进行预测
            prediction = self.model.predict(processed_data)[0]
            
            # 获取预测概率（置信度）
            if hasattr(self.model, 'decision_function'):
                decision_scores = self.model.decision_function(processed_data)[0]
                confidence = np.max(decision_scores)
            else:
                confidence = 1.0
            
            return prediction, confidence
            
        except Exception as e:
            print(f"预测失败: {e}")
            return None, None
    
    def predict_batch(self, pressure_data_list):
        """
        批量预测多个样本
        
        Args:
            pressure_data_list: 包含多个256维压力数据的数组或列表
            
        Returns:
            list: 预测结果列表，每个元素为(预测类别, 置信度)
        """
        results = []
        for i, pressure_data in enumerate(pressure_data_list):
            try:
                prediction, confidence = self.predict_single(pressure_data)
                results.append((prediction, confidence))
            except Exception as e:
                print(f"预测第{i+1}个样本时出错: {e}")
                results.append((None, None))
        
        return results
    
    def predict_from_csv(self, csv_file):
        """
        从CSV文件读取数据并进行预测
        支持两种格式：
        1. 有Label列：Label,F1(g),F2(g),...,F256(g)
        2. 无Label列：F1(g),F2(g),...,F256(g) 或 时间戳,F1(g),F2(g),...,F256(g)
        
        Args:
            csv_file: CSV文件路径
            
        Returns:
            dict: 预测结果字典
        """
        try:
            # 尝试不同编码读取文件
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
                raise ValueError("无法读取CSV文件，请检查编码格式")
            
            print(f"📊 文件信息:")
            print(f"   - 样本数: {len(df)}")
            print(f"   - 列数: {df.shape[1]}")
            
            # 判断文件格式并提取特征数据
            has_label = 'Label' in df.columns
            
            if has_label:
                # 格式1：有Label列
                true_labels = df['Label'].values
                feature_data = df.drop('Label', axis=1).values
                print(f"   - 检测到Label列，将进行对比预测")
            else:
                # 格式2：无Label列
                true_labels = None
                if df.shape[1] == 256:
                    # 纯特征数据
                    feature_data = df.values
                elif df.shape[1] == 257:
                    # 可能有时间戳列，取后256列
                    feature_data = df.iloc[:, -256:].values
                else:
                    raise ValueError(f"数据列数错误，期望256或257列，实际{df.shape[1]}列")
                
                print(f"   - 未检测到Label列，进行纯预测")
            
            # 检查特征维度
            if feature_data.shape[1] != 256:
                raise ValueError(f"特征维度错误，期望256维，实际{feature_data.shape[1]}维")
            
            # 批量预测
            print(f"🔮 开始预测...")
            predictions = self.predict_batch(feature_data)
            
            # 统计结果
            pred_labels = [pred[0] for pred in predictions if pred[0] is not None]
            confidences = [pred[1] for pred in predictions if pred[1] is not None]
            
            result = {
                'total_samples': len(df),
                'successful_predictions': len(pred_labels),
                'failed_predictions': len(df) - len(pred_labels),
                'predictions': pred_labels,
                'confidences': confidences
            }
            
            # 统计预测分布
            unique_preds, counts = np.unique(pred_labels, return_counts=True)
            pred_distribution = dict(zip(unique_preds, counts))
            result['prediction_distribution'] = pred_distribution
            
            print(f"✅ 预测完成:")
            print(f"   - 成功预测: {result['successful_predictions']}/{result['total_samples']}")
            print(f"   - 预测分布: {pred_distribution}")
            
            # 如果有真实标签，计算准确率
            if has_label and result['successful_predictions'] > 0:
                from sklearn.metrics import accuracy_score, classification_report
                
                # 过滤掉预测失败的样本
                valid_indices = [i for i, pred in enumerate(predictions) if pred[0] is not None]
                filtered_true = true_labels[valid_indices]
                
                accuracy = accuracy_score(filtered_true, pred_labels)
                result['accuracy'] = accuracy
                
                print(f"   - 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"\n📋 详细分类报告:")
                print(classification_report(filtered_true, pred_labels))
            
            return result
            
        except Exception as e:
            print(f"❌ CSV预测失败: {e}")
            return None
    
    def interactive_predict(self):
        """交互式预测"""
        print(f"🎯 交互式坐姿预测 (模型: {self.model_type})")
        print("输入256个压力值（用逗号分隔），或输入 'quit' 退出")
        
        while True:
            try:
                user_input = input("\n请输入压力数据: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见!")
                    break
                
                # 解析输入
                pressure_values = [float(x.strip()) for x in user_input.split(',')]
                
                if len(pressure_values) != 256:
                    print(f"❌ 输入维度错误，期望256个值，实际{len(pressure_values)}个")
                    continue
                
                # 预测
                prediction, confidence = self.predict_single(pressure_values)
                
                if prediction is not None:
                    print(f"🎯 预测结果: {prediction}")
                    print(f"📊 置信度: {confidence:.3f}")
                else:
                    print("❌ 预测失败")
                    
            except ValueError:
                print("❌ 输入格式错误，请输入数字，用逗号分隔")
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python predict.py single <pressure_data>     # 单个预测")
        print("  python predict.py csv <csv_file>             # CSV文件预测")
        print("  python predict.py interactive               # 交互式预测")
        print("  python predict.py [--improved] <mode> ...   # 使用改进模型")
        return
    
    # 解析参数
    args = sys.argv[1:]
    model_type = 'standard'
    
    if args[0] == '--improved':
        model_type = 'improved'
        args = args[1:]
    
    if not args:
        print("❌ 缺少预测模式参数")
        return
    
    mode = args[0]
    
    try:
        # 初始化预测器
        predictor = PosturePredictor(model_type=model_type)
        
        if mode == 'single':
            # 单个预测
            if len(args) < 2:
                print("❌ 缺少压力数据参数")
                return
            
            pressure_data = [float(x) for x in args[1].split(',')]
            prediction, confidence = predictor.predict_single(pressure_data)
            
            if prediction is not None:
                print(f"🎯 预测结果: {prediction}")
                print(f"📊 置信度: {confidence:.3f}")
            else:
                print("❌ 预测失败")
        
        elif mode == 'csv':
            # CSV文件预测
            if len(args) < 2:
                print("❌ 缺少CSV文件路径")
                return
            
            csv_file = args[1]
            result = predictor.predict_from_csv(csv_file)
            
            if result is None:
                print("❌ CSV预测失败")
        
        elif mode == 'interactive':
            # 交互式预测
            predictor.interactive_predict()
        
        else:
            print(f"❌ 未知的预测模式: {mode}")
            
    except Exception as e:
        print(f"❌ 预测器初始化失败: {e}")

if __name__ == "__main__":
    main()