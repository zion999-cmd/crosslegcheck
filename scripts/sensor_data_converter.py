#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器数据格式转换工具
支持12路传感器数据与256维数组之间的转换
"""

import numpy as np

class SensorDataConverter:
    """传感器数据格式转换器"""
    
    def __init__(self):
        # 12个关键传感器在256维数组中的位置索引
        self.key_sensor_positions = [48, 80, 112, 176, 87, 103, 88, 89, 104, 105, 91, 107]
        
        # 传感器索引到数组索引的映射 (注意：这里是1-based到0-based的转换)
        # 原数据文件中F1对应索引0，F48对应索引47，所以需要减1
        self.sensor_to_array_mapping = [pos - 1 for pos in [49, 81, 113, 177, 88, 104, 89, 90, 105, 106, 92, 108]]
        
        # 验证映射
        print("传感器位置映射 (0-based索引):")
        for i, pos in enumerate(self.sensor_to_array_mapping):
            print(f"  传感器 {i+1}: 数组索引 {pos}")
    
    def expand_12_to_256(self, sensor_data_12):
        """
        将12路传感器数据扩展到256维数组
        
        Args:
            sensor_data_12: 12路传感器数据 [s1, s2, ..., s12]
            
        Returns:
            256维数组，只有12个位置有数据，其余为0
        """
        if len(sensor_data_12) != 12:
            raise ValueError(f"期望12个传感器数据，但收到{len(sensor_data_12)}个")
        
        # 创建256维的零数组
        expanded_data = np.zeros(256, dtype=float)
        
        # 将12路传感器数据填充到对应位置
        for i, sensor_value in enumerate(sensor_data_12):
            array_index = self.sensor_to_array_mapping[i]
            expanded_data[array_index] = sensor_value
        
        return expanded_data
    
    def extract_12_from_256(self, sensor_data_256):
        """
        从256维数组中提取12路关键传感器数据
        
        Args:
            sensor_data_256: 256维传感器数组
            
        Returns:
            12路关键传感器数据
        """
        if len(sensor_data_256) != 256:
            raise ValueError(f"期望256个传感器数据，但收到{len(sensor_data_256)}个")
        
        # 提取12个关键位置的数据
        extracted_data = []
        for array_index in self.sensor_to_array_mapping:
            extracted_data.append(sensor_data_256[array_index])
        
        return np.array(extracted_data)
    
    def validate_conversion(self, original_12_data):
        """
        验证转换的正确性
        
        Args:
            original_12_data: 原始12路传感器数据
            
        Returns:
            转换是否正确
        """
        # 扩展到256维再提取回12维
        expanded = self.expand_12_to_256(original_12_data)
        extracted = self.extract_12_from_256(expanded)
        
        # 检查是否一致
        is_consistent = np.allclose(original_12_data, extracted)
        
        if is_consistent:
            print("✅ 数据转换验证通过")
        else:
            print("❌ 数据转换验证失败")
            print(f"原始数据: {original_12_data}")
            print(f"转换后数据: {extracted}")
        
        return is_consistent
    
    def get_active_positions_in_256(self, sensor_data_12):
        """
        获取12路传感器在256维数组中的活跃位置信息
        
        Args:
            sensor_data_12: 12路传感器数据
            
        Returns:
            活跃位置的详细信息
        """
        active_positions = []
        
        for i, sensor_value in enumerate(sensor_data_12):
            if sensor_value > 0:  # 只考虑有读数的传感器
                array_index = self.sensor_to_array_mapping[i]
                row = array_index // 16  # 假设是16x16的网格
                col = array_index % 16
                
                active_positions.append({
                    'sensor_id': i + 1,
                    'array_index': array_index,
                    'position': (row, col),
                    'value': sensor_value
                })
        
        return active_positions

class HardwareAdapterFor256Model:
    """
    适用于期望256维输入的模型的硬件适配器
    """
    
    def __init__(self, model_path, requires_256_input=False):
        self.converter = SensorDataConverter()
        self.requires_256_input = requires_256_input
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载模型"""
        import joblib
        try:
            self.model = joblib.load(model_path)
            
            # 检查模型期望的输入维度
            if hasattr(self.model, 'n_features_in_'):
                expected_features = self.model.n_features_in_
                print(f"模型期望输入维度: {expected_features}")
                
                if expected_features == 256:
                    self.requires_256_input = True
                    print("✅ 检测到256维输入模型，将自动进行数据转换")
                elif expected_features == 12:
                    self.requires_256_input = False
                    print("✅ 检测到12维输入模型，直接使用原始数据")
                else:
                    print(f"⚠️  未识别的输入维度: {expected_features}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
    
    def predict_from_hardware(self, sensor_data_12):
        """
        从硬件传感器数据进行预测
        
        Args:
            sensor_data_12: 12路硬件传感器读数
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未加载")
        
        # 根据模型类型决定是否需要转换数据格式
        if self.requires_256_input:
            # 将12维数据扩展到256维
            model_input = self.converter.expand_12_to_256(sensor_data_12)
            print(f"🔄 数据已从12维扩展到256维")
        else:
            # 直接使用12维数据
            model_input = np.array(sensor_data_12)
            print(f"📊 直接使用12维数据")
        
        # 进行预测
        try:
            prediction = self.model.predict([model_input])
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([model_input])
                return prediction[0], probabilities[0]
            else:
                return prediction[0], None
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None, None

def demo_conversion():
    """演示数据转换功能"""
    print("🔧 传感器数据转换演示")
    print("=" * 50)
    
    converter = SensorDataConverter()
    
    # 模拟12路传感器数据
    test_scenarios = {
        "无人": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "正常坐姿": [120, 150, 180, 200, 350, 400, 380, 390, 420, 430, 350, 380],
        "左倾": [200, 180, 160, 100, 450, 400, 420, 380, 350, 320, 400, 350],
        "右倾": [80, 120, 160, 250, 280, 350, 300, 320, 480, 500, 300, 450]
    }
    
    for scenario_name, sensor_data_12 in test_scenarios.items():
        print(f"\n📋 测试场景: {scenario_name}")
        print(f"原始12路数据: {sensor_data_12}")
        
        # 转换为256维
        expanded_256 = converter.expand_12_to_256(sensor_data_12)
        print(f"扩展到256维: 非零位置数 = {np.count_nonzero(expanded_256)}")
        
        # 提取回12维
        extracted_12 = converter.extract_12_from_256(expanded_256)
        print(f"提取回12维: {extracted_12.astype(int).tolist()}")
        
        # 验证一致性
        is_consistent = np.allclose(sensor_data_12, extracted_12)
        print(f"数据一致性: {'✅ 通过' if is_consistent else '❌ 失败'}")
        
        # 显示活跃位置
        active_positions = converter.get_active_positions_in_256(sensor_data_12)
        if active_positions:
            print("活跃传感器位置:")
            for pos_info in active_positions:
                print(f"  传感器{pos_info['sensor_id']}: 位置{pos_info['position']}, 值{pos_info['value']}")
        
        print("-" * 40)

def create_hardware_interface_template():
    """创建硬件接口模板"""
    template = '''
# 硬件接口模板 - 支持256维模型

from sensor_data_converter import HardwareAdapterFor256Model
import time

class HardwareSensorInterface:
    def __init__(self, model_path):
        # 创建适配器，自动检测模型输入维度
        self.adapter = HardwareAdapterFor256Model(model_path)
    
    def read_12_sensors_from_hardware(self):
        """
        从实际硬件读取12路传感器数据
        返回: [sensor1, sensor2, ..., sensor12] (单位: 克)
        """
        # 这里替换为实际的硬件读取代码
        # 例如通过串口、I2C、SPI等接口读取12路传感器
        
        # 示例：串口读取
        # serial_data = self.serial_port.readline()
        # sensor_values = parse_sensor_data(serial_data)
        
        # 示例：模拟数据
        import random
        return [random.randint(0, 500) for _ in range(12)]
    
    def run_detection_loop(self):
        """运行检测循环"""
        while True:
            try:
                # 读取12路传感器数据
                sensor_data_12 = self.read_12_sensors_from_hardware()
                
                # 使用适配器进行预测（自动处理数据格式转换）
                prediction, probabilities = self.adapter.predict_from_hardware(sensor_data_12)
                
                if prediction is not None:
                    if probabilities is not None:
                        confidence = max(probabilities)
                        print(f"坐姿: {prediction}, 置信度: {confidence:.1%}")
                    else:
                        print(f"坐姿: {prediction}")
                
                time.sleep(0.5)  # 每0.5秒检测一次
                
            except KeyboardInterrupt:
                print("检测已停止")
                break
            except Exception as e:
                print(f"检测错误: {e}")
                time.sleep(1)

# 使用示例
if __name__ == "__main__":
    # 指定模型路径
    model_path = "/path/to/your/model.pkl"
    
    # 创建硬件接口
    hardware = HardwareSensorInterface(model_path)
    
    # 运行检测
    hardware.run_detection_loop()
'''
    
    with open('/Users/bx/Workspace/crosslegcheck/scripts/hardware_interface_template.py', 'w', encoding='utf-8') as f:
        f.write(template)
    
    print("✅ 硬件接口模板已保存到 hardware_interface_template.py")

def main():
    """主函数"""
    print("🎯 传感器数据格式转换工具")
    print("=" * 60)
    
    # 演示转换功能
    demo_conversion()
    
    print("\n" + "=" * 60)
    
    # 创建硬件接口模板
    create_hardware_interface_template()
    
    print(f"\n💡 使用说明:")
    print(f"1. 如果你的模型期望12维输入：直接使用12路传感器数据")
    print(f"2. 如果你的模型期望256维输入：使用expand_12_to_256()转换")
    print(f"3. HardwareAdapterFor256Model会自动检测并处理数据转换")

if __name__ == "__main__":
    main()