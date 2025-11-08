#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12传感器实时坐姿检测器 - 直接读取版本
直接使用12路压力传感器数据，无需256传感器扩展
"""

import threading
import queue
import time
import numpy as np
import pandas as pd
import joblib
from collections import deque, Counter
import sys
import os

class DirectTwelveSensorDetector:
    """直接使用12路传感器的实时坐姿检测器"""
    
    def __init__(self, infinite_mode=True):
        """初始化检测器"""
        self.infinite_mode = infinite_mode
        
        # 数据队列和线程控制
        self.data_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=50)
        self.stop_event = threading.Event()
        
        # 结果统计
        self.prediction_history = deque(maxlen=10)
        self.detection_stats = {
            'total_detections': 0,
            'left_count': 0,
            'normal_count': 0,
            'right_count': 0,
            'no_person_count': 0
        }
        
        # 加载训练好的模型
        self.load_models()
        
    def load_models(self):
        """加载训练好的模型"""
        try:
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(script_dir, 'models_12_sensors')
            
            # 加载Logistic回归模型
            lr_model_path = os.path.join(models_dir, 'logistic_regression_12.pkl')
            self.lr_model = joblib.load(lr_model_path)
            print("✅ Logistic回归模型加载成功")
            
            # 加载随机森林模型
            rf_model_path = os.path.join(models_dir, 'random_forest_12.pkl')
            self.rf_model = joblib.load(rf_model_path)
            print("✅ 随机森林模型加载成功")
            
            # 加载标准化器和标签编码器
            scaler_path = os.path.join(models_dir, 'scaler_12.pkl')
            self.scaler = joblib.load(scaler_path)
            print("✅ 数据标准化器加载成功")
            
            label_encoder_path = os.path.join(models_dir, 'label_encoder_12.pkl')
            self.label_encoder = joblib.load(label_encoder_path)
            print("✅ 标签编码器加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)
    
    def predict_ensemble(self, sensor_data_12):
        """集成两个模型的预测结果"""
        # 计算总压力和非零传感器数量
        total_pressure = np.sum(sensor_data_12)
        nonzero_sensors = np.count_nonzero(sensor_data_12)
        
        # 无人检测策略 - 基于训练数据分析
        pressure_threshold = 1000  # 总压力阈值（克）
        sensor_count_threshold = 8  # 最少传感器数量
        
        # 判断是否无人
        is_no_person = (total_pressure < pressure_threshold) or (nonzero_sensors < sensor_count_threshold)
        
        if is_no_person:
            return {
                'prediction': 'normal',  # 无人时返回正常状态
                'confidence': 1.0,
                'ensemble_confidence': 1.0,
                'lr_result': ('normal', 1.0, [0, 1, 0]),
                'rf_result': ('normal', 1.0, [0, 1, 0]),
                'total_pressure': total_pressure,
                'nonzero_sensors': nonzero_sensors,
                'is_no_person': True
            }
        
        # 有人情况下进行正常预测
        try:
            # Logistic回归预测
            sensor_data_scaled = self.scaler.transform([sensor_data_12])
            lr_prediction = self.lr_model.predict(sensor_data_scaled)[0]
            lr_probabilities = self.lr_model.predict_proba(sensor_data_scaled)[0]
            lr_predicted_label = self.label_encoder.inverse_transform([lr_prediction])[0]
            lr_confidence = np.max(lr_probabilities)
            
            # 随机森林预测
            rf_prediction = self.rf_model.predict([sensor_data_12])[0]
            rf_probabilities = self.rf_model.predict_proba([sensor_data_12])[0]
            rf_confidence = np.max(rf_probabilities)
            
            # 集成策略：如果两个模型一致，使用该结果；否则选择置信度更高的
            if lr_predicted_label == rf_prediction:
                final_prediction = lr_predicted_label
                ensemble_confidence = (lr_confidence + rf_confidence) / 2
            else:
                if lr_confidence > rf_confidence:
                    final_prediction = lr_predicted_label
                    ensemble_confidence = lr_confidence
                else:
                    final_prediction = rf_prediction
                    ensemble_confidence = rf_confidence
            
            return {
                'prediction': final_prediction,
                'confidence': ensemble_confidence,
                'ensemble_confidence': ensemble_confidence,
                'lr_result': (lr_predicted_label, lr_confidence, lr_probabilities),
                'rf_result': (rf_prediction, rf_confidence, rf_probabilities),
                'total_pressure': total_pressure,
                'nonzero_sensors': nonzero_sensors,
                'is_no_person': False
            }
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'ensemble_confidence': 0.0,
                'lr_result': ('unknown', 0.0, [0.33, 0.33, 0.34]),
                'rf_result': ('unknown', 0.0, [0.33, 0.33, 0.34]),
                'total_pressure': total_pressure,
                'nonzero_sensors': nonzero_sensors,
                'is_no_person': False
            }
    
    def process_sensor_data(self, sensor_data_12):
        """处理12路传感器数据"""
        if not self.data_queue.full():
            self.data_queue.put({
                'timestamp': time.time(),
                'sensor_data_12': sensor_data_12,
                'sample_count': getattr(self, 'sample_count', 0)
            })
            self.sample_count = getattr(self, 'sample_count', 0) + 1
    
    def prediction_worker(self):
        """预测处理线程"""
        print("🧠 启动预测处理线程...")
        
        while not self.stop_event.is_set() or not self.data_queue.empty():
            try:
                # 从队列获取数据，超时1秒
                data_item = self.data_queue.get(timeout=1.0)
                
                # 预测坐姿
                result = self.predict_ensemble(data_item['sensor_data_12'])
                
                # 添加时间戳等信息
                result.update({
                    'timestamp': data_item['timestamp'],
                    'raw_data': data_item['sensor_data_12']
                })
                
                # 将结果放入结果队列
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
                # 更新统计
                self.update_stats(result)
                
                # 标记任务完成
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 预测处理错误: {e}")
                continue
        
        print("🧠 预测处理线程结束")
    
    def display_worker(self):
        """结果显示线程"""
        print("📊 启动结果显示线程...")
        
        while not self.stop_event.is_set() or not self.result_queue.empty():
            try:
                # 从结果队列获取数据
                result = self.result_queue.get(timeout=1.0)
                
                # 显示结果
                self.display_result(result)
                
                # 标记任务完成
                self.result_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 结果显示错误: {e}")
                continue
        
        print("📊 结果显示线程结束")
    
    def display_result(self, result):
        """显示预测结果"""
        # 格式化时间戳
        timestamp = time.strftime('%H:%M:%S', time.localtime(result['timestamp']))
        
        # 获取状态显示
        if result['is_no_person']:
            status_emoji = "👤"
            status_text = "无人坐着"
            confidence_text = f"置信度: {result['confidence']:.1%}"
        else:
            status_map = {
                'left': ("⬅️", "左倾"),
                'normal': ("✅", "正常坐姿"),
                'right': ("➡️", "右倾")
            }
            status_emoji, status_text = status_map.get(result['prediction'], ("❓", "未知状态"))
            confidence_text = f"置信度: {result['confidence']:.1%}"
        
        # 传感器数据摘要
        sensor_summary = f"总压力: {result['total_pressure']:.0f}g, 活跃传感器: {result['nonzero_sensors']}/12"
        
        # 显示主要结果
        print(f"🕐 {timestamp} | {status_emoji} {status_text} | {confidence_text} | {sensor_summary}")
        
        # 详细模型结果（可选）
        if hasattr(self, 'verbose') and self.verbose:
            lr_label, lr_conf, lr_probs = result['lr_result']
            rf_label, rf_conf, rf_probs = result['rf_result']
            print(f"   📈 LR: {lr_label}({lr_conf:.1%}) | RF: {rf_label}({rf_conf:.1%})")
    
    def update_stats(self, result):
        """更新检测统计"""
        self.detection_stats['total_detections'] += 1
        
        if result['is_no_person']:
            self.detection_stats['no_person_count'] += 1
        else:
            prediction = result['prediction']
            if prediction == 'left':
                self.detection_stats['left_count'] += 1
            elif prediction == 'normal':
                self.detection_stats['normal_count'] += 1
            elif prediction == 'right':
                self.detection_stats['right_count'] += 1
        
        # 更新预测历史
        self.prediction_history.append(result['prediction'])
    
    def show_stats(self):
        """显示检测统计"""
        stats = self.detection_stats
        total = stats['total_detections']
        
        if total == 0:
            print("📊 暂无检测统计")
            return
        
        print(f"\n📊 检测统计 (总计: {total})")
        print(f"   👤 无人: {stats['no_person_count']} ({stats['no_person_count']/total:.1%})")
        print(f"   ⬅️ 左倾: {stats['left_count']} ({stats['left_count']/total:.1%})")
        print(f"   ✅ 正常: {stats['normal_count']} ({stats['normal_count']/total:.1%})")
        print(f"   ➡️ 右倾: {stats['right_count']} ({stats['right_count']/total:.1%})")
        
        # 最近趋势
        if len(self.prediction_history) > 0:
            recent_predictions = list(self.prediction_history)[-5:]
            print(f"   🔄 最近5次: {' → '.join(recent_predictions)}")
    
    def run_demo_mode(self):
        """演示模式 - 使用模拟数据"""
        print("🎲 启动演示模式（使用模拟数据）")
        
        # 启动处理线程
        prediction_thread = threading.Thread(target=self.prediction_worker)
        display_thread = threading.Thread(target=self.display_worker)
        
        prediction_thread.start()
        display_thread.start()
        
        # 生成模拟数据
        try:
            # 尝试加载真实测试数据
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            test_data_path = os.path.join(script_dir, 'data', 'test_dataset_12_sensors.csv')
            test_df = pd.read_csv(test_data_path)
            test_data = test_df.drop('Label', axis=1).values
            test_labels = test_df['Label'].values
            
            print(f"📋 加载了 {len(test_data)} 个测试样本")
            
            sample_idx = 0
            self.sample_count = 0
            
            while not self.stop_event.is_set():
                # 循环使用测试数据
                if sample_idx < len(test_data):
                    sensor_data_12 = test_data[sample_idx]
                    true_label = test_labels[sample_idx]
                else:
                    # 数据用完后生成随机数据
                    sensor_data_12 = np.random.randint(0, 1000, 12)
                    true_label = 'random'
                
                # 添加一些随机噪声使数据更真实
                noise = np.random.normal(0, 10, len(sensor_data_12))
                noisy_data = np.maximum(0, sensor_data_12 + noise)
                
                # 处理数据
                self.process_sensor_data(noisy_data.astype(int))
                
                sample_idx += 1
                
                # 限制演示模式运行时间
                if not self.infinite_mode and sample_idx >= 20:
                    print("\n📊 演示模式完成")
                    break
                
                time.sleep(1.0)  # 每秒一个样本
                
        except Exception as e:
            print(f"❌ 演示数据加载失败: {e}")
            print("🎲 使用随机数据演示")
            
            # 生成随机数据演示
            for i in range(10 if not self.infinite_mode else 10000):
                if self.stop_event.is_set():
                    break
                
                # 生成12个随机传感器值
                random_data = np.random.randint(0, 1000, 12)
                self.process_sensor_data(random_data)
                
                time.sleep(1.0)
        
        # 等待处理完成
        self.stop_event.set()
        prediction_thread.join(timeout=5)
        display_thread.join(timeout=5)
        
        # 显示最终统计
        self.show_stats()
    
    def run_with_custom_data(self, sensor_data_list):
        """使用自定义数据运行检测"""
        print(f"🔍 处理 {len(sensor_data_list)} 个自定义数据样本")
        
        # 启动处理线程
        prediction_thread = threading.Thread(target=self.prediction_worker)
        display_thread = threading.Thread(target=self.display_worker)
        
        prediction_thread.start()
        display_thread.start()
        
        # 处理自定义数据
        self.sample_count = 0
        for sensor_data_12 in sensor_data_list:
            if self.stop_event.is_set():
                break
            
            self.process_sensor_data(sensor_data_12)
            time.sleep(0.5)  # 稍微延迟让线程处理
        
        # 等待处理完成
        time.sleep(2)
        self.stop_event.set()
        prediction_thread.join(timeout=5)
        display_thread.join(timeout=5)
        
        # 显示最终统计
        self.show_stats()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='12传感器直接检测器')
    parser.add_argument('--demo', action='store_true', help='演示模式')
    parser.add_argument('--full', action='store_true', help='无限循环模式')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = DirectTwelveSensorDetector(infinite_mode=args.full)
    
    if args.verbose:
        detector.verbose = True
    
    try:
        if args.demo:
            detector.run_demo_mode()
        else:
            print("💡 使用 --demo 参数运行演示模式")
            print("💡 使用 --full 参数启用无限循环模式")
            
            # 示例：处理一些自定义数据
            example_data = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 无人
                [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650],  # 正常坐姿
                [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325],   # 轻压正常
                [800, 750, 700, 200, 100, 50, 25, 10, 5, 0, 0, 0],           # 左倾
                [0, 0, 0, 50, 100, 200, 400, 600, 700, 750, 800, 850]        # 右倾
            ]
            
            detector.run_with_custom_data(example_data)
    
    except KeyboardInterrupt:
        print("\n⏹️  检测已停止")
        detector.stop_event.set()
    except Exception as e:
        print(f"❌ 运行错误: {e}")


if __name__ == "__main__":
    main()