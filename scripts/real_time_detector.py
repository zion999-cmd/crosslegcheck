#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时压力传感器数据采集与识别系统
结合串口采集和CNN模型预测，实现实时分类
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import threading
import queue
import time
from datetime import datetime
import sys
import os
import warnings
from collections import deque

# 导入现有模块
from serial_sensor_reader import PressureSensorReader
from cnn_augmented import pressure_to_image

warnings.filterwarnings('ignore')

class RealTimeDetector:
    """实时检测器类"""
    
    def __init__(self, model_path='../models/cnn_augmented_model.keras'):
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        self.class_names = ['left', 'normal', 'right']
        
        # 数据队列和处理
        self.data_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=50)
        
        # 控制标志
        self.running = False
        self.sensor_thread = None
        self.detector_thread = None
        
        # 传感器读取器
        self.sensor_reader = PressureSensorReader()
        
        # 统计信息
        self.stats = {
            'total_samples': 0,
            'predictions': {'left': 0, 'normal': 0, 'right': 0},
            'start_time': None,
            'last_result': None,
            'confidence_history': deque(maxlen=10)
        }
    
    def load_model_async(self):
        """异步加载模型"""
        def load():
            print(f"🔄 正在加载模型: {self.model_path}")
            try:
                self.model = keras.models.load_model(self.model_path)
                self.model_loaded = True
                print(f"✅ 模型加载完成")
                print(f"   - 模型参数量: {self.model.count_params():,}")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                self.model_loaded = False
        
        # 在后台线程中加载模型
        model_thread = threading.Thread(target=load, daemon=True)
        model_thread.start()
        return model_thread
    
    def sensor_data_collector(self):
        """传感器数据收集线程"""
        print(f"📡 启动数据采集线程...")
        
        # 连接传感器
        if not self.sensor_reader.connect():
            print(f"❌ 传感器连接失败")
            return
        
        print(f"✅ 传感器连接成功，开始数据采集...")
        
        # 等待连接稳定
        time.sleep(2)
        self.sensor_reader.serial_conn.flushInput()
        self.sensor_reader.serial_conn.flushOutput()
        
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            try:
                # 读取原始数据
                raw_data = self.sensor_reader.read_raw_data(512)
                
                if raw_data:
                    # 解析数据帧
                    frames = self.sensor_reader.parse_hex_data(raw_data)
                    
                    if frames:
                        consecutive_failures = 0
                        for frame in frames:
                            if not self.data_queue.full():
                                timestamp = datetime.now()
                                self.data_queue.put((timestamp, frame))
                            else:
                                # 队列满了，丢弃最旧的数据
                                try:
                                    self.data_queue.get_nowait()
                                    self.data_queue.put((timestamp, frame))
                                except queue.Empty:
                                    pass
                    else:
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1
                
                # 检查连续失败次数
                if consecutive_failures >= max_failures:
                    print(f"⚠️  连续{max_failures}次获取数据失败，尝试重新连接...")
                    self.sensor_reader.disconnect()
                    time.sleep(1)
                    if self.sensor_reader.connect():
                        consecutive_failures = 0
                        print(f"✅ 重新连接成功")
                    else:
                        print(f"❌ 重新连接失败")
                        break
                
                # 短暂休眠
                time.sleep(0.001)
                
            except Exception as e:
                print(f"❌ 数据采集错误: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
        
        # 清理
        self.sensor_reader.disconnect()
        print(f"🔌 数据采集线程已停止")
    
    def prediction_processor(self):
        """预测处理线程"""
        print(f"🧠 启动预测处理线程...")
        
        # 等待模型加载
        while self.running and not self.model_loaded:
            print(f"⏳ 等待模型加载...")
            time.sleep(1)
        
        if not self.model_loaded:
            print(f"❌ 模型未加载，预测线程退出")
            return
        
        print(f"✅ 模型已就绪，开始实时预测...")
        
        while self.running:
            try:
                # 从队列获取数据
                timestamp, pressure_data = self.data_queue.get(timeout=1)
                
                # 转换为图像格式
                image = pressure_to_image(pressure_data)
                image = image[np.newaxis, ..., np.newaxis]  # 添加batch和channel维度
                
                # 进行预测
                prediction = self.model.predict(image, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                predicted_label = self.class_names[predicted_class]
                
                # 更新统计信息
                self.stats['total_samples'] += 1
                self.stats['predictions'][predicted_label] += 1
                self.stats['last_result'] = {
                    'timestamp': timestamp,
                    'label': predicted_label,
                    'confidence': confidence,
                    'probabilities': prediction[0],
                    'data_stats': {
                        'min': pressure_data.min(),
                        'max': pressure_data.max(),
                        'mean': pressure_data.mean(),
                        'non_zero': np.count_nonzero(pressure_data)
                    }
                }
                self.stats['confidence_history'].append(confidence)
                
                # 将结果放入结果队列
                if not self.result_queue.full():
                    self.result_queue.put(self.stats['last_result'])
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                print(f"❌ 预测处理错误: {e}")
                time.sleep(0.1)
        
        print(f"🧠 预测处理线程已停止")
    
    def display_results(self, update_interval=0.5):
        """显示实时结果"""
        print(f"📊 启动结果显示...")
        print(f"=" * 80)
        print(f"实时压力传感器状态检测系统")
        print(f"按 Ctrl+C 停止检测")
        print(f"=" * 80)
        
        last_display = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # 检查是否有新结果
                try:
                    result = self.result_queue.get_nowait()
                    
                    # 实时显示每个预测结果
                    timestamp = result['timestamp'].strftime('%H:%M:%S.%f')[:-3]
                    label = result['label']
                    confidence = result['confidence']
                    
                    # 根据置信度显示不同的指示符
                    if confidence > 0.8:
                        indicator = "🟢"  # 高置信度
                    elif confidence > 0.6:
                        indicator = "🟡"  # 中等置信度
                    else:
                        indicator = "🔴"  # 低置信度
                    
                    # 状态映射到中文
                    status_map = {'left': '左偏', 'normal': '正常', 'right': '右偏'}
                    status_cn = status_map.get(label, label)
                    
                    print(f"[{timestamp}] {indicator} 检测结果: {status_cn:4s} | 置信度: {confidence:.3f} | "
                          f"数据范围: {result['data_stats']['min']}-{result['data_stats']['max']}")
                
                except queue.Empty:
                    pass
                
                # 定期显示统计信息
                if current_time - last_display > update_interval * 10:  # 每5秒显示一次统计
                    self.display_statistics()
                    last_display = current_time
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 显示错误: {e}")
                time.sleep(0.1)
    
    def display_statistics(self):
        """显示统计信息"""
        if self.stats['total_samples'] == 0:
            return
        
        print(f"\n" + "─" * 60)
        print(f"📈 实时统计 (总样本: {self.stats['total_samples']})")
        
        # 预测分布
        total = sum(self.stats['predictions'].values())
        if total > 0:
            for label, count in self.stats['predictions'].items():
                percentage = count / total * 100
                status_cn = {'left': '左偏', 'normal': '正常', 'right': '右偏'}[label]
                print(f"   {status_cn}: {count:4d} ({percentage:5.1f}%)")
        
        # 最近的置信度
        if self.stats['confidence_history']:
            avg_confidence = np.mean(self.stats['confidence_history'])
            print(f"   平均置信度: {avg_confidence:.3f}")
        
        # 运行时间
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            rate = self.stats['total_samples'] / elapsed if elapsed > 0 else 0
            print(f"   检测速率: {rate:.1f} 样本/秒")
        
        print(f"─" * 60)
    
    def start(self):
        """启动实时检测"""
        print(f"🚀 启动实时压力传感器检测系统...")
        
        # 设置开始时间
        self.stats['start_time'] = time.time()
        self.running = True
        
        # 异步加载模型
        model_thread = self.load_model_async()
        
        # 启动数据采集线程
        self.sensor_thread = threading.Thread(target=self.sensor_data_collector, daemon=True)
        self.sensor_thread.start()
        
        # 启动预测处理线程
        self.detector_thread = threading.Thread(target=self.prediction_processor, daemon=True)
        self.detector_thread.start()
        
        try:
            # 主线程显示结果
            self.display_results()
        except KeyboardInterrupt:
            print(f"\n⏹️  用户停止检测")
        finally:
            self.stop()
    
    def stop(self):
        """停止检测"""
        print(f"\n🛑 正在停止检测系统...")
        self.running = False
        
        # 等待线程结束
        if self.sensor_thread and self.sensor_thread.is_alive():
            self.sensor_thread.join(timeout=3)
        
        if self.detector_thread and self.detector_thread.is_alive():
            self.detector_thread.join(timeout=3)
        
        # 显示最终统计
        print(f"\n📊 最终统计:")
        self.display_statistics()
        
        print(f"✅ 检测系统已停止")

class SimpleRealTimeDetector:
    """简化版实时检测器（单线程）"""
    
    def __init__(self, model_path='../models/cnn_augmented_model.keras'):
        self.model_path = model_path
        self.model = None
        self.class_names = ['left', 'normal', 'right']
        self.sensor_reader = PressureSensorReader()
        
        # 统计信息
        self.stats = {
            'total_samples': 0,
            'predictions': {'left': 0, 'normal': 0, 'right': 0},
            'start_time': time.time()
        }
    
    def load_model(self):
        """加载模型"""
        print(f"🔄 加载模型: {self.model_path}")
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ 模型加载完成")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def run_simple_detection(self, max_samples=100):
        """运行简化检测"""
        print(f"🚀 启动简化实时检测 (最多{max_samples}个样本)...")
        
        # 加载模型
        if not self.load_model():
            return
        
        # 连接传感器
        if not self.sensor_reader.connect():
            print(f"❌ 传感器连接失败")
            return
        
        print(f"✅ 开始实时检测...")
        print(f"按 Ctrl+C 停止")
        print(f"=" * 70)
        
        try:
            # 等待连接稳定
            time.sleep(2)
            self.sensor_reader.serial_conn.flushInput()
            self.sensor_reader.serial_conn.flushOutput()
            
            sample_count = 0
            
            while sample_count < max_samples:
                # 读取数据
                raw_data = self.sensor_reader.read_raw_data(512)
                
                if raw_data:
                    frames = self.sensor_reader.parse_hex_data(raw_data)
                    
                    if frames:
                        for frame in frames:
                            sample_count += 1
                            
                            # 转换为图像并预测
                            image = pressure_to_image(frame)
                            image = image[np.newaxis, ..., np.newaxis]
                            
                            prediction = self.model.predict(image, verbose=0)
                            predicted_class = np.argmax(prediction[0])
                            confidence = prediction[0][predicted_class]
                            predicted_label = self.class_names[predicted_class]
                            
                            # 更新统计
                            self.stats['total_samples'] += 1
                            self.stats['predictions'][predicted_label] += 1
                            
                            # 显示结果
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            status_map = {'left': '左偏', 'normal': '正常', 'right': '右偏'}
                            status_cn = status_map[predicted_label]
                            
                            if confidence > 0.8:
                                indicator = "🟢"
                            elif confidence > 0.6:
                                indicator = "🟡"
                            else:
                                indicator = "🔴"
                            
                            print(f"[{timestamp}] {indicator} 样本{sample_count:3d}: {status_cn} "
                                  f"(置信度: {confidence:.3f}) | "
                                  f"数据: {frame.min()}-{frame.max()}")
                            
                            if sample_count >= max_samples:
                                break
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print(f"\n⏹️  用户停止检测")
        finally:
            self.sensor_reader.disconnect()
            
            # 显示统计
            print(f"\n📊 检测完成统计:")
            total = sum(self.stats['predictions'].values())
            for label, count in self.stats['predictions'].items():
                status_cn = {'left': '左偏', 'normal': '正常', 'right': '右偏'}[label]
                percentage = count / total * 100 if total > 0 else 0
                print(f"   {status_cn}: {count} 次 ({percentage:.1f}%)")
            
            elapsed = time.time() - self.stats['start_time']
            rate = total / elapsed if elapsed > 0 else 0
            print(f"   检测速率: {rate:.1f} 样本/秒")

def test_model_only():
    """仅测试模型加载和预测功能"""
    print("🧪 测试模型功能...")
    
    # 加载模型
    model_path = '../models/cnn_augmented_model.keras'
    try:
        model = keras.models.load_model(model_path)
        print(f"✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 生成测试数据
    test_data = np.random.randint(0, 1000, (3, 256))
    class_names = ['left', 'normal', 'right']
    
    print(f"🔮 测试预测...")
    for i, data in enumerate(test_data):
        image = pressure_to_image(data)
        image = image[np.newaxis, ..., np.newaxis]
        
        prediction = model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        predicted_label = class_names[predicted_class]
        
        print(f"   测试样本{i+1}: {predicted_label} (置信度: {confidence:.3f})")
    
    print(f"✅ 模型测试完成")

def demo_with_saved_data():
    """使用保存的数据进行演示"""
    print("🎬 使用保存数据进行演示...")
    
    # 检查是否有保存的测试数据
    test_files = ['test_quick.csv', '../data/test_quick.csv', 'real_time_data.csv']
    data_file = None
    
    for file in test_files:
        if os.path.exists(file):
            data_file = file
            break
    
    if not data_file:
        print("❌ 未找到测试数据文件")
        return
    
    print(f"📂 使用数据文件: {data_file}")
    
    # 加载模型
    model_path = '../models/cnn_augmented_model.keras'
    try:
        model = keras.models.load_model(model_path)
        print(f"✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 加载数据
    try:
        import pandas as pd
        df = pd.read_csv(data_file)
        features = df.values
        print(f"📊 加载了 {len(features)} 个样本")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 模拟实时检测
    print(f"🎮 开始模拟实时检测...")
    print(f"=" * 60)
    
    class_names = ['left', 'normal', 'right']
    stats = {'left': 0, 'normal': 0, 'right': 0}
    
    for i, data in enumerate(features):
        # 预测
        image = pressure_to_image(data)
        image = image[np.newaxis, ..., np.newaxis]
        
        prediction = model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        predicted_label = class_names[predicted_class]
        
        stats[predicted_label] += 1
        
        # 显示结果
        timestamp = datetime.now().strftime('%H:%M:%S')
        status_map = {'left': '左偏', 'normal': '正常', 'right': '右偏'}
        status_cn = status_map[predicted_label]
        
        if confidence > 0.8:
            indicator = "🟢"
        elif confidence > 0.6:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f"[{timestamp}] {indicator} 样本{i+1:3d}: {status_cn} "
              f"(置信度: {confidence:.3f}) | "
              f"数据范围: {data.min()}-{data.max()}")
        
        # 模拟实时间隔
        time.sleep(0.2)
    
    # 统计结果
    print(f"\n📈 演示完成统计:")
    total = sum(stats.values())
    for label, count in stats.items():
        status_cn = {'left': '左偏', 'normal': '正常', 'right': '右偏'}[label]
        percentage = count / total * 100 if total > 0 else 0
        print(f"   {status_cn}: {count} 次 ({percentage:.1f}%)")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("🚀 实时压力传感器检测系统")
        print("=" * 50)
        print("📖 使用方法:")
        print("   python real_time_detector.py full              - 完整多线程实时检测")
        print("   python real_time_detector.py simple [数量]      - 简化单线程检测")
        print("   python real_time_detector.py test              - 测试模型功能")
        print("   python real_time_detector.py demo              - 使用保存数据演示")
        print("")
        print("💡 推荐使用:")
        print("   python real_time_detector.py simple 20         - 快速检测20个样本")
        print("   python real_time_detector.py demo              - 查看演示效果")
        print("   python real_time_detector.py test              - 验证模型状态")
        return
    
    mode = sys.argv[1]
    
    if mode == 'full':
        # 完整多线程检测
        detector = RealTimeDetector()
        detector.start()
        
    elif mode == 'simple':
        # 简化单线程检测
        max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        detector = SimpleRealTimeDetector()
        detector.run_simple_detection(max_samples)
        
    elif mode == 'test':
        # 测试模型功能
        test_model_only()
        
    elif mode == 'demo':
        # 演示模式
        demo_with_saved_data()
        
    else:
        print(f"❌ 未知模式: {mode}")
        print(f"使用 'python real_time_detector.py' 查看帮助")

if __name__ == "__main__":
    main()