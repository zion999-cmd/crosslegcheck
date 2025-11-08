# 12路压力传感器直接使用指南

## 概述

如果你在实际使用中只有12路压力传感器，**不需要扩展数组到256**。我们提供了专门的直接使用版本，可以直接处理12路传感器数据。

## 文件说明

### 主要文件
- `real_time_detector_12_direct.py` - 直接使用12路传感器的检测器
- `hardware_12_sensor_example.py` - 硬件使用示例和模板

### 模型文件（已训练好）
- `models_12_sensors/logistic_regression_12.pkl` - Logistic回归模型
- `models_12_sensors/random_forest_12.pkl` - 随机森林模型  
- `models_12_sensors/scaler_12.pkl` - 数据标准化器
- `models_12_sensors/label_encoder_12.pkl` - 标签编码器

## 传感器位置映射

### 12路传感器对应的原256传感器阵列位置：
```
传感器 1: 索引 48  -> 位置 (3, 0)   # 左上角
传感器 2: 索引 80  -> 位置 (5, 0)   # 左中上
传感器 3: 索引 112 -> 位置 (7, 0)   # 左中
传感器 4: 索引 176 -> 位置 (11, 0)  # 左下
传感器 5: 索引 87  -> 位置 (5, 7)   # 左内1
传感器 6: 索引 103 -> 位置 (6, 7)   # 左内2
传感器 7: 索引 88  -> 位置 (5, 8)   # 中央1
传感器 8: 索引 89  -> 位置 (5, 9)   # 中央2
传感器 9: 索引 104 -> 位置 (6, 8)   # 中央3
传感器 10: 索引 105 -> 位置 (6, 9)  # 中央4
传感器 11: 索引 91  -> 位置 (5, 11) # 右内1
传感器 12: 索引 107 -> 位置 (6, 11) # 右内2
```

**重要**: 这些是数据数组的索引位置，不是物理坐标。在实际硬件布局中，你需要按照这12个关键位置来安装压力传感器。

## 快速使用

### 1. 演示模式
```bash
python real_time_detector_12_direct.py --demo
```

### 2. 在代码中直接使用
```python
from real_time_detector_12_direct import DirectTwelveSensorDetector

# 创建检测器
detector = DirectTwelveSensorDetector()

# 12路传感器数据 (单位：克)
sensor_data_12 = [120, 150, 180, 200, 350, 400, 380, 390, 420, 430, 350, 380]

# 进行预测
result = detector.predict_ensemble(sensor_data_12)

# 获取结果
posture = result['prediction']        # 'left', 'normal', 'right'
confidence = result['confidence']     # 置信度 0.0-1.0
is_no_person = result['is_no_person'] # 是否无人
total_pressure = result['total_pressure']   # 总压力(克)
active_sensors = result['nonzero_sensors']  # 活跃传感器数量
```

## 硬件集成模板

### 基本使用模板
```python
import time
from real_time_detector_12_direct import DirectTwelveSensorDetector

# 初始化检测器
detector = DirectTwelveSensorDetector()

def read_12_sensors_from_hardware():
    """
    从硬件读取12路传感器数据
    返回: [sensor1, sensor2, ..., sensor12] (单位:克)
    """
    # 这里替换为实际的硬件读取代码
    # 例如通过串口、I2C、SPI等接口读取
    pass

# 主循环
while True:
    try:
        # 读取传感器数据
        sensor_data_12 = read_12_sensors_from_hardware()
        
        # 进行检测
        result = detector.predict_ensemble(sensor_data_12)
        
        # 处理结果
        if result['is_no_person']:
            print("无人坐着")
        else:
            posture = result['prediction']
            confidence = result['confidence']
            print(f"坐姿: {posture}, 置信度: {confidence:.1%}")
            
            # 根据结果执行相应动作
            if posture == 'left':
                print("检测到左倾，建议调整坐姿")
            elif posture == 'right':
                print("检测到右倾，建议调整坐姿")
            else:
                print("坐姿正常")
        
        # 控制检测频率
        time.sleep(0.5)  # 每0.5秒检测一次
        
    except KeyboardInterrupt:
        print("检测已停止")
        break
    except Exception as e:
        print(f"检测错误: {e}")
        time.sleep(1)
```

### Arduino/STM32 集成示例
```python
import serial
import time
from real_time_detector_12_direct import DirectTwelveSensorDetector

class ArduinoInterface:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.serial = serial.Serial(port, baudrate)
        self.detector = DirectTwelveSensorDetector()
    
    def read_sensors(self):
        """从Arduino读取12路传感器数据"""
        try:
            # 发送读取命令
            self.serial.write(b'READ_SENSORS\\n')
            
            # 读取响应 (假设格式: "123,456,789,...")
            response = self.serial.readline().decode().strip()
            sensor_values = [int(x) for x in response.split(',')]
            
            if len(sensor_values) == 12:
                return sensor_values
            else:
                print(f"数据长度错误: {len(sensor_values)}")
                return None
                
        except Exception as e:
            print(f"读取传感器失败: {e}")
            return None
    
    def run_detection(self):
        """运行检测循环"""
        while True:
            sensor_data = self.read_sensors()
            if sensor_data:
                result = self.detector.predict_ensemble(sensor_data)
                
                # 发送结果回Arduino
                posture = result['prediction']
                confidence = int(result['confidence'] * 100)
                
                command = f"RESULT:{posture},{confidence}\\n"
                self.serial.write(command.encode())
                
                print(f"坐姿: {posture}, 置信度: {confidence}%")
            
            time.sleep(0.5)

# 使用示例
if __name__ == "__main__":
    arduino = ArduinoInterface('/dev/ttyUSB0')
    arduino.run_detection()
```

## 数据格式要求

### 输入数据格式
- **数据类型**: Python列表或NumPy数组
- **长度**: 正好12个元素
- **数值范围**: 0-1000 (单位：克)
- **数据顺序**: 必须按照传感器1-12的顺序排列

### 示例数据
```python
# 无人状态
no_person = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 正常坐姿
normal_posture = [120, 150, 180, 200, 350, 400, 380, 390, 420, 430, 350, 380]

# 左倾
left_lean = [200, 180, 160, 100, 450, 400, 420, 380, 350, 320, 400, 350]

# 右倾  
right_lean = [80, 120, 160, 250, 280, 350, 300, 320, 480, 500, 300, 450]
```

## 检测阈值

### 无人检测阈值
- **总压力阈值**: 1000克
- **活跃传感器阈值**: 8个传感器

当满足以下任一条件时，判定为无人状态：
1. 总压力 < 1000克
2. 活跃传感器数量 < 8个

### 模型置信度
- **高置信度**: > 80%
- **中等置信度**: 60% - 80%  
- **低置信度**: < 60%

## 性能指标

### 模型准确率
- **Logistic回归**: 95.2%
- **随机森林**: 97.2%
- **集成模型**: 约97%

### 检测速度
- **单次预测时间**: < 20ms
- **推荐检测频率**: 0.5-2秒/次
- **内存占用**: < 10MB

## 常见问题

### Q: 我的传感器位置与训练数据不同怎么办？
A: 需要重新训练模型。使用 `create_12_sensor_model.py` 脚本，但要修改传感器位置索引。

### Q: 检测结果不准确怎么办？
A: 
1. 检查传感器位置是否正确
2. 确认数据单位是否为克
3. 检查传感器是否正常工作
4. 可能需要重新校准或训练

### Q: 可以使用更少的传感器吗？
A: 可以，但需要重新训练模型。12个传感器是经过优化的最佳配置。

### Q: 支持其他编程语言吗？
A: 目前提供Python版本。C语言版本请查看 `embedded_12_sensors/` 目录。

## 总结

使用12路压力传感器的优势：
1. **硬件成本低** - 只需要12个传感器而不是256个
2. **数据处理简单** - 直接处理12个数值，无需复杂的数组操作
3. **实时性好** - 数据量小，处理速度快
4. **准确率高** - 达到97%的检测准确率
5. **集成简单** - 提供了完整的接口和示例代码

**重要提醒**: 传感器的物理位置布局必须与训练时的位置一致，这是保证检测准确性的关键。