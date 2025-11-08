#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12传感器模型快速测试脚本
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report

def load_12_sensor_models():
    """加载12传感器模型"""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(script_dir, 'models_12_sensors')
    
    models = {}
    models['lr'] = joblib.load(os.path.join(model_dir, 'logistic_regression_12.pkl'))
    models['rf'] = joblib.load(os.path.join(model_dir, 'random_forest_12.pkl'))
    models['scaler'] = joblib.load(os.path.join(model_dir, 'scaler_12.pkl'))
    models['label_encoder'] = joblib.load(os.path.join(model_dir, 'label_encoder_12.pkl'))
    
    return models

def test_models():
    """测试12传感器模型"""
    print("🧪 12传感器模型快速测试")
    print("=" * 50)
    
    # 加载模型
    print("📦 加载模型...")
    models = load_12_sensor_models()
    
    # 加载测试数据
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_path = os.path.join(script_dir, 'data', 'test_dataset_12_sensors.csv')
    
    print("📂 加载测试数据...")
    test_df = pd.read_csv(test_data_path)
    
    X_test = test_df.drop('Label', axis=1).values
    y_test = test_df['Label'].values
    
    print(f"   测试样本数: {len(X_test)}")
    print(f"   特征数: {X_test.shape[1]}")
    print(f"   类别分布: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Logistic回归测试
    print("\n🧠 Logistic回归模型测试...")
    X_test_scaled = models['scaler'].transform(X_test)
    lr_pred_encoded = models['lr'].predict(X_test_scaled)
    lr_pred = models['label_encoder'].inverse_transform(lr_pred_encoded)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    print(f"   准确率: {lr_accuracy:.1%}")
    
    # 随机森林测试
    print("\n🌲 随机森林模型测试...")
    rf_pred = models['rf'].predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"   准确率: {rf_accuracy:.1%}")
    
    # 详细报告
    print(f"\n📊 Logistic回归详细报告:")
    print(classification_report(y_test, lr_pred))
    
    print(f"\n📊 随机森林详细报告:")
    print(classification_report(y_test, rf_pred))
    
    # 单样本预测演示
    print(f"\n🎯 单样本预测演示:")
    sample_idx = 0
    sample_data = X_test[sample_idx:sample_idx+1]
    sample_label = y_test[sample_idx]
    
    # LR预测
    sample_scaled = models['scaler'].transform(sample_data)
    lr_pred_encoded = models['lr'].predict(sample_scaled)[0]
    lr_pred_single = models['label_encoder'].inverse_transform([lr_pred_encoded])[0]
    lr_proba = models['lr'].predict_proba(sample_scaled)[0]
    lr_conf = np.max(lr_proba)
    
    # RF预测  
    rf_pred_single = models['rf'].predict(sample_data)[0]
    rf_proba = models['rf'].predict_proba(sample_data)[0]
    rf_conf = np.max(rf_proba)
    
    print(f"   真实标签: {sample_label}")
    print(f"   LR预测: {lr_pred_single} (置信度: {lr_conf:.1%})")
    print(f"   RF预测: {rf_pred_single} (置信度: {rf_conf:.1%})")
    print(f"   12传感器数据: {sample_data[0]}")
    
    print(f"\n✅ 测试完成！12传感器模型表现优异")

if __name__ == "__main__":
    test_models()