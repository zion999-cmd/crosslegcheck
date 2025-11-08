#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12传感器实时坐姿检测器
基于优化的12传感器模型进行实时坐姿检测
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

# 添加当前目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from serial_sensor_reader import PressureSensorReader
except ImportError:
    print("⚠️  未找到 serial_sensor_reader.py，将使用模拟数据模式")
    PressureSensorReader = None

class TwelveSensorRealTimeDetector:
    """12传感器实时坐姿检测器"""
    
    def __init__(self, port='/dev/cu.usbserial-14220', baudrate=115200, infinite_mode=True):
        """初始化检测器"""
        self.port = port
        self.baudrate = baudrate
        self.infinite_mode = infinite_mode  # 是否无限循环模式
        
        # 12个关键传感器位置（对应原256传感器的索引）
        self.key_sensor_positions = [48, 80, 112, 176, 87, 103, 88, 89, 104, 105, 91, 107]
        
        # 数据队列和线程控制
        self.data_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=50)
        self.stop_event = threading.Event()
        
        # 结果统计
        self.prediction_history = deque(maxlen=10)  # 保存最近10次预测
        self.detection_stats = {
            'total_detections': 0,
            'left_count': 0,
            'normal_count': 0,
            'right_count': 0,
            'avg_confidence': 0.0,
            'start_time': time.time()
        }
        
        # 加载模型
        self.load_models()
        
        # 传感器读取器
        self.sensor_reader = None
        if PressureSensorReader:
            try:
                self.sensor_reader = PressureSensorReader(port, baudrate)
                print(f"✅ 12传感器读取器初始化成功: {port}")
            except Exception as e:
                print(f"❌ 传感器读取器初始化失败: {e}")
                print("将使用模拟数据模式")
        
    def load_models(self):
        """加载12传感器模型"""
        try:
            # 获取脚本的绝对路径，然后构建模型目录路径
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(script_dir, 'models_12_sensors')
            
            # 加载模型和预处理器
            self.lr_model = joblib.load(os.path.join(model_dir, 'logistic_regression_12.pkl'))
            self.rf_model = joblib.load(os.path.join(model_dir, 'random_forest_12.pkl'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler_12.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder_12.pkl'))
            
            print("✅ 12传感器模型加载成功")
            print(f"   - Logistic回归模型: {self.lr_model}")
            print(f"   - 随机森林模型: {self.rf_model}")
            print(f"   - 标签映射: {self.label_encoder.classes_}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("请确保已运行 create_12_sensor_model.py 生成模型")
            raise
    
    def extract_12_sensor_data(self, full_sensor_data):
        """从256传感器数据中提取12个关键传感器数据"""
        if len(full_sensor_data) != 256:
            raise ValueError(f"期望256个传感器数据，但收到{len(full_sensor_data)}个")
        
        # 提取12个关键位置的数据
        key_data = [full_sensor_data[pos] for pos in self.key_sensor_positions]
        return np.array(key_data)
    
    def predict_posture_lr(self, sensor_data_12):
        """使用Logistic回归模型预测坐姿"""
        try:
            # 标准化数据
            sensor_data_scaled = self.scaler.transform([sensor_data_12])
            
            # 预测
            prediction = self.lr_model.predict(sensor_data_scaled)[0]
            probabilities = self.lr_model.predict_proba(sensor_data_scaled)[0]
            
            # 解码预测结果
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            
            # 计算置信度
            confidence = np.max(probabilities)
            
            return predicted_label, confidence, probabilities
            
        except Exception as e:
            print(f"❌ Logistic回归预测失败: {e}")
            return 'unknown', 0.0, [0.33, 0.33, 0.34]
    
    def predict_posture_rf(self, sensor_data_12):
        """使用随机森林模型预测坐姿"""
        try:
            # 随机森林直接使用原始数据，无需标准化
            prediction = self.rf_model.predict([sensor_data_12])[0]
            probabilities = self.rf_model.predict_proba([sensor_data_12])[0]
            
            # 计算置信度
            confidence = np.max(probabilities)
            
            return prediction, confidence, probabilities
            
        except Exception as e:
            print(f"❌ 随机森林预测失败: {e}")
            return 'unknown', 0.0, [0.33, 0.33, 0.34]
    
    def predict_ensemble(self, sensor_data_12):
        """集成两个模型的预测结果"""
        # 计算总压力和非零传感器数量
        total_pressure = np.sum(sensor_data_12)
        nonzero_sensors = np.count_nonzero(sensor_data_12)
        
        # 更精确的无人检测策略
        # 基于训练数据分析：最小有效压力1044克，最少非零传感器10个
        pressure_threshold = 1000  # 略低于训练数据最小值
        sensor_count_threshold = 8  # 要求至少8个传感器有读数
        
        # 多重条件判断无人状态
        is_no_person = (total_pressure < pressure_threshold) or (nonzero_sensors < sensor_count_threshold)
        
        if is_no_person:
            # 没人坐着的情况，直接返回正常
            return 'normal', 0.95, 'normal', 'normal', 0.95, 0.95
        
        # Logistic回归预测
        lr_label, lr_conf, lr_probs = self.predict_posture_lr(sensor_data_12)
        
        # 随机森林预测
        rf_label, rf_conf, rf_probs = self.predict_posture_rf(sensor_data_12)
        
        # 简单的投票集成
        if lr_label == rf_label:
            # 两个模型一致，使用平均置信度
            final_label = lr_label
            final_confidence = (lr_conf + rf_conf) / 2
        else:
            # 两个模型不一致，选择置信度高的
            if lr_conf > rf_conf:
                final_label = lr_label
                final_confidence = lr_conf
            else:
                final_label = rf_label
                final_confidence = rf_conf
        
        return final_label, final_confidence, lr_label, rf_label, lr_conf, rf_conf
    
    def sensor_data_collector(self):
        """传感器数据采集线程"""
        print("📡 12传感器数据采集线程启动")
        
        if self.sensor_reader:
            if self.infinite_mode:
                # 真实传感器数据采集 - 无限循环模式
                self.continuous_collect_infinite()
            else:
                # 真实传感器数据采集 - 限制100个样本模式  
                self.continuous_collect_limited()
        else:
            # 模拟数据模式
            self.generate_simulation_data()
    
    def continuous_collect_limited(self):
        """限制采集100个样本"""
        print("📊 进入限制采集模式（100个样本）...")
        
        sample_count = 0
        for sensor_data in self.sensor_reader.continuous_collect(sample_count=100):
            if self.stop_event.is_set() or sample_count >= 100:
                break
                
            try:
                if len(sensor_data) == 256:
                    # 提取12个关键传感器数据
                    key_sensor_data = self.extract_12_sensor_data(sensor_data)
                    
                    if not self.data_queue.full():
                        self.data_queue.put({
                            'timestamp': time.time(),
                            'sensor_data_12': key_sensor_data,
                            'full_data': sensor_data,
                            'sample_count': sample_count
                        })
                        sample_count += 1
                    else:
                        print("⚠️  数据队列已满，跳过数据")
                        
            except Exception as e:
                print(f"❌ 数据处理错误: {e}")
                continue
        
        print(f"📊 限制采集完成，共采集 {sample_count} 个样本")
        print("💡 如需持续运行，请使用 --full 参数")
        
        # 等待一段时间让其他线程处理完剩余数据
        time.sleep(5)
        self.stop_event.set()
    
    def continuous_collect_infinite(self):
        """无限循环采集传感器数据，直到停止信号"""
        print("📡 进入无限循环数据采集模式...")
        
        sample_count = 0
        start_time = time.time()
        last_data_time = start_time
        reconnect_count = 0
        
        while not self.stop_event.is_set():
            try:
                # 连续读取数据
                raw_data = self.sensor_reader.read_raw_data(512)
                
                if raw_data:
                    # 解析数据帧
                    frames = self.sensor_reader.parse_hex_data(raw_data)
                    if frames:
                        last_data_time = time.time()
                        reconnect_count = 0  # 重置重连计数
                        
                        for frame in frames:
                            if self.stop_event.is_set():
                                break
                                
                            sample_count += 1
                            
                            if len(frame) == 256:
                                # 提取12个关键传感器数据
                                key_sensor_data = self.extract_12_sensor_data(frame)
                                
                                if not self.data_queue.full():
                                    self.data_queue.put({
                                        'timestamp': time.time(),
                                        'sensor_data_12': key_sensor_data,
                                        'full_data': frame,
                                        'sample_count': sample_count
                                    })
                                else:
                                    print("⚠️  数据队列已满，跳过数据")
                            else:
                                print(f"⚠️  收到异常数据长度: {len(frame)}")
                
                # 检查是否长时间没有数据
                current_time = time.time()
                if current_time - last_data_time > 5:  # 5秒没有有效数据
                    reconnect_count += 1
                    print(f"\n⚠️  长时间无有效数据 (第{reconnect_count}次)，尝试重新连接...")
                    
                    if reconnect_count > 3:  # 重连超过3次，进入模拟模式
                        print("❌ 多次重连失败，切换到模拟数据模式")
                        self.sensor_reader = None
                        self.generate_simulation_data()
                        return
                    
                    self.sensor_reader.disconnect()
                    time.sleep(1)
                    if not self.sensor_reader.connect():
                        print("❌ 重新连接失败")
                        continue
                    last_data_time = time.time()
                
                # 短暂休眠避免CPU占用过高
                time.sleep(0.001)
                
            except KeyboardInterrupt:
                print("\n🛑 数据采集线程收到停止信号")
                self.stop_event.set()
                break
            except Exception as e:
                print(f"❌ 数据采集异常: {e}")
                time.sleep(0.1)
                continue
        
        print(f"📊 数据采集完成，总共采集 {sample_count} 个样本")
    
    def generate_simulation_data(self):
        """生成模拟的12传感器数据"""
        print("🎲 使用模拟数据模式")
        
        # 加载一些真实的12传感器数据作为模拟基础
        try:
            # 获取脚本的绝对路径，然后构建数据文件路径
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            test_data_path = os.path.join(script_dir, 'data', 'test_dataset_12_sensors.csv')
            test_df = pd.read_csv(test_data_path)
            test_data = test_df.drop('Label', axis=1).values
            test_labels = test_df['Label'].values
            
            sample_idx = 0
            while not self.stop_event.is_set():
                # 循环使用测试数据
                sensor_data_12 = test_data[sample_idx % len(test_data)]
                true_label = test_labels[sample_idx % len(test_data)]
                
                # 添加一些随机噪声
                noise = np.random.normal(0, 10, len(sensor_data_12))
                noisy_data = np.maximum(0, sensor_data_12 + noise)
                
                if not self.data_queue.full():
                    self.data_queue.put({
                        'timestamp': time.time(),
                        'sensor_data_12': noisy_data.astype(int),
                        'true_label': true_label,  # 仅用于模拟模式验证
                        'sample_idx': sample_idx
                    })
                
                sample_idx += 1
                time.sleep(0.5)  # 每0.5秒一个样本
                
        except Exception as e:
            print(f"❌ 模拟数据生成失败: {e}")
            # 生成简单的随机数据
            while not self.stop_event.is_set():
                # 生成12个随机传感器值
                random_data = np.random.randint(0, 1000, 12)
                
                if not self.data_queue.full():
                    self.data_queue.put({
                        'timestamp': time.time(),
                        'sensor_data_12': random_data,
                        'random': True
                    })
                
                time.sleep(1.0)
    
    def prediction_processor(self):
        """预测处理线程"""
        print("🧠 12传感器预测处理线程启动")
        
        while not self.stop_event.is_set():
            try:
                # 获取数据（超时1秒）
                data_item = self.data_queue.get(timeout=1.0)
                sensor_data_12 = data_item['sensor_data_12']
                timestamp = data_item['timestamp']
                
                # 数据质量检查
                total_pressure = np.sum(sensor_data_12)
                if total_pressure < 100:  # 压力太低，可能是无效数据
                    continue
                
                # 预测坐姿
                start_time = time.time()
                final_label, final_confidence, lr_label, rf_label, lr_conf, rf_conf = self.predict_ensemble(sensor_data_12)
                prediction_time = (time.time() - start_time) * 1000  # 转换为毫秒
                
                # 保存预测结果
                result = {
                    'timestamp': timestamp,
                    'predicted_posture': final_label,
                    'confidence': final_confidence,
                    'lr_prediction': lr_label,
                    'rf_prediction': rf_label,
                    'lr_confidence': lr_conf,
                    'rf_confidence': rf_conf,
                    'prediction_time_ms': prediction_time,
                    'sensor_data_12': sensor_data_12,
                    'total_pressure': total_pressure,
                    'true_label': data_item.get('true_label', 'unknown')  # 仅模拟模式有效
                }
                
                # 添加到结果队列
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
                # 更新统计信息
                self.update_statistics(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 预测处理错误: {e}")
                continue
    
    def update_statistics(self, result):
        """更新检测统计信息"""
        self.detection_stats['total_detections'] += 1
        
        posture = result['predicted_posture']
        if posture == 'left':
            self.detection_stats['left_count'] += 1
        elif posture == 'normal':
            self.detection_stats['normal_count'] += 1
        elif posture == 'right':
            self.detection_stats['right_count'] += 1
        
        # 更新平均置信度
        total = self.detection_stats['total_detections']
        old_avg = self.detection_stats['avg_confidence']
        new_conf = result['confidence']
        self.detection_stats['avg_confidence'] = (old_avg * (total - 1) + new_conf) / total
        
        # 保存到历史记录
        self.prediction_history.append(posture)
    
    def display_results(self):
        """显示结果线程"""
        print("📊 12传感器结果显示线程启动")
        
        last_display_time = 0
        display_interval = 2.0  # 每2秒显示一次
        
        while not self.stop_event.is_set():
            current_time = time.time()
            
            if current_time - last_display_time >= display_interval:
                try:
                    # 显示最新结果
                    if not self.result_queue.empty():
                        latest_result = None
                        # 获取队列中最新的结果
                        while not self.result_queue.empty():
                            latest_result = self.result_queue.get_nowait()
                        
                        if latest_result:
                            self.display_single_result(latest_result)
                    
                    last_display_time = current_time
                    
                except Exception as e:
                    print(f"❌ 显示错误: {e}")
            
            time.sleep(0.1)
    
    def display_single_result(self, result):
        """显示单个检测结果"""
        # 清屏
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🪑 12传感器实时坐姿检测系统")
        print("=" * 60)
        
        # 当前检测结果
        posture = result['predicted_posture']
        confidence = result['confidence']
        total_pressure = result['total_pressure']
        sensor_data = result['sensor_data_12']
        nonzero_sensors = np.count_nonzero(sensor_data)
        
        # 判断是否无人状态 (使用与predict_ensemble相同的逻辑)
        is_no_person = (total_pressure < 1000) or (nonzero_sensors < 8)
        
        # 使用表情符号显示坐姿
        if is_no_person:
            posture_display = "🪑 无人 (显示正常)"
            status_note = f"💡 检测到无人状态 (压力:{total_pressure:.0f}克, 传感器:{nonzero_sensors}/12)"
        else:
            posture_emoji = {
                'left': '👈 左偏 (翘左二郎腿)',
                'normal': '✅ 正常坐姿',
                'right': '👉 右偏 (翘右二郎腿)',
                'unknown': '❓ 未知'
            }
            posture_display = posture_emoji.get(posture, posture)
            status_note = f"👤 有人就座 (压力:{total_pressure:.0f}克, 传感器:{nonzero_sensors}/12)"
        
        print(f"🎯 检测结果: {posture_display} (置信度: {confidence:.1%})")
        print(f"📌 状态说明: {status_note}")
        
        # 显示阈值信息
        print(f"🔧 检测阈值: 压力≥1000克 且 传感器≥8个")
        
        # 显示模型对比（只在有人时显示详细对比）
        if not is_no_person:
            print(f"\n📊 模型对比:")
            print(f"   Logistic回归: {result['lr_prediction']} (置信度: {result['lr_confidence']:.1%})")
            print(f"   随机森林:     {result['rf_prediction']} (置信度: {result['rf_confidence']:.1%})")
        else:
            print(f"\n📊 无人检测:")
            print(f"   训练数据最小压力: 1044 克")
            print(f"   训练数据最少传感器: 10 个")
            print(f"   当前状态: 低于训练数据范围")
        
        # 显示传感器数据概览
        
        print(f"\n📡 12传感器数据:")
        print(f"   总压力: {total_pressure:,} 克")
        print(f"   最大值: {np.max(sensor_data):,} 克")
        print(f"   非零传感器: {nonzero_sensors}/12")
        
        # 显示关键传感器数据
        print(f"\n🔍 关键传感器读数:")
        sensor_names = ['左上', '左中上', '左中', '左下', '左内1', '左内2', 
                       '中央1', '中央2', '中央3', '中央4', '右内1', '右内2']
        
        for i, (name, value) in enumerate(zip(sensor_names, sensor_data)):
            if i % 4 == 0:
                print()
            print(f"   {name}: {value:4d}", end="  ")
        print()
        
        # 显示性能信息
        print(f"\n⚡ 性能信息:")
        print(f"   预测时间: {result['prediction_time_ms']:.1f} ms")
        
        # 模拟模式显示真实标签对比
        if 'true_label' in result and result['true_label'] != 'unknown':
            true_label = result['true_label']
            is_correct = (posture == true_label)
            status = "✅ 正确" if is_correct else "❌ 错误"
            print(f"   真实标签: {true_label} | 预测结果: {status}")
        
        # 显示统计信息
        stats = self.detection_stats
        total = stats['total_detections']
        runtime = time.time() - stats['start_time']
        
        print(f"\n📈 运行统计:")
        print(f"   总检测次数: {total}")
        print(f"   运行时间: {runtime:.1f} 秒")
        print(f"   检测频率: {total/runtime:.1f} 次/秒")
        print(f"   平均置信度: {stats['avg_confidence']:.1%}")
        
        # 显示坐姿分布
        if total > 0:
            left_pct = stats['left_count'] / total * 100
            normal_pct = stats['normal_count'] / total * 100
            right_pct = stats['right_count'] / total * 100
            
            print(f"\n📊 坐姿分布:")
            print(f"   👈 左偏: {stats['left_count']} ({left_pct:.1f}%)")
            print(f"   ✅ 正常: {stats['normal_count']} ({normal_pct:.1f}%)")
            print(f"   👉 右偏: {stats['right_count']} ({right_pct:.1f}%)")
        
        # 显示最近预测趋势
        if len(self.prediction_history) > 0:
            recent_counter = Counter(list(self.prediction_history))
            print(f"\n📉 最近{len(self.prediction_history)}次预测:")
            
            # 定义表情符号映射
            posture_emoji_map = {
                'left': '👈 左偏 (翘左二郎腿)',
                'normal': '✅ 正常坐姿',
                'right': '👉 右偏 (翘右二郎腿)',
                'unknown': '❓ 未知'
            }
            
            for posture_type, count in recent_counter.most_common():
                emoji = posture_emoji_map.get(posture_type, posture_type)
                print(f"   {emoji}: {count}")
        
        print(f"\n💡 按 Ctrl+C 停止检测")
    
    def run(self):
        """启动实时检测"""
        print("🚀 启动12传感器实时坐姿检测系统")
        
        try:
            # 启动线程
            threads = [
                threading.Thread(target=self.sensor_data_collector, name="SensorCollector"),
                threading.Thread(target=self.prediction_processor, name="PredictionProcessor"),
                threading.Thread(target=self.display_results, name="ResultDisplay")
            ]
            
            for thread in threads:
                thread.daemon = True
                thread.start()
            
            print("✅ 所有线程已启动，开始检测...")
            
            # 主线程等待
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 接收到停止信号，正在关闭...")
            self.stop_event.set()
            
            # 等待线程结束
            for thread in threads:
                thread.join(timeout=2)
            
            print("✅ 12传感器实时检测系统已停止")
        
        except Exception as e:
            print(f"❌ 系统错误: {e}")
            self.stop_event.set()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='12传感器实时坐姿检测')
    parser.add_argument('--port', default='/dev/cu.usbserial-14220', 
                       help='串口设备路径 (默认: /dev/cu.usbserial-14220)')
    parser.add_argument('--baudrate', type=int, default=115200,
                       help='波特率 (默认: 115200)')
    parser.add_argument('--demo', action='store_true',
                       help='演示模式（使用模拟数据）')
    parser.add_argument('--full', action='store_true',
                       help='完整运行模式（无限循环直到Ctrl+C停止）')
    
    args = parser.parse_args()
    
    print("🪑 12传感器实时坐姿检测系统")
    print("=" * 50)
    print("基于优化的12传感器模型进行实时检测")
    print("预期性能：95.2%准确率，<1ms预测时间")
    
    if args.full:
        print("🔄 完整运行模式：无限循环直到手动停止")
    else:
        print("📊 标准模式：采集100个样本后停止")
    
    print()
    
    if args.demo:
        print("🎲 演示模式：使用测试数据集模拟传感器输入")
        detector = TwelveSensorRealTimeDetector(infinite_mode=args.full)
        detector.sensor_reader = None  # 强制使用模拟模式
    else:
        print(f"📡 硬件模式：连接到 {args.port} (波特率: {args.baudrate})")
        detector = TwelveSensorRealTimeDetector(args.port, args.baudrate, infinite_mode=args.full)
        
        if not args.full:
            print("⚠️  使用标准模式，将采集100个样本后停止")
            print("💡 要持续运行请使用 --full 参数")
    
    detector.run()

if __name__ == "__main__":
    main()