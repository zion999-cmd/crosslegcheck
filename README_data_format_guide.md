# 12路传感器数据格式处理指南

## 问题解答

### 问题：模型输入形状是否需要扩展到256维？

**答案**：取决于你使用的模型类型。

## 两种情况分析

### 情况1：模型期望12维输入（当前推荐）
- **模型文件**：`models_12_sensors/` 目录下的模型
- **输入格式**：直接使用12路传感器数据 `[sensor1, sensor2, ..., sensor12]`
- **硬件要求**：只需要12路压力传感器
- **优势**：硬件成本低，数据处理简单，实时性好

```python
# 直接使用12路数据
sensor_data_12 = [120, 150, 180, 200, 350, 400, 380, 390, 420, 430, 350, 380]
result = detector.predict_ensemble(sensor_data_12)
```

### 情况2：模型期望256维输入（兼容性支持）
- **模型文件**：原始的256传感器训练的模型
- **输入格式**：需要扩展到256维数组，其中12个位置有数据，244个位置为0
- **硬件要求**：仍然只需要12路压力传感器
- **转换方法**：使用 `sensor_data_converter.py`

```python
from sensor_data_converter import SensorDataConverter

converter = SensorDataConverter()

# 12路传感器数据
sensor_data_12 = [120, 150, 180, 200, 350, 400, 380, 390, 420, 430, 350, 380]

# 扩展到256维
sensor_data_256 = converter.expand_12_to_256(sensor_data_12)

# 使用256维数据进行预测
result = model.predict([sensor_data_256])
```

## 模型输入维度检查

### 检查现有模型的输入维度
```python
import joblib

# 加载模型
model = joblib.load('your_model.pkl')

# 检查期望的输入维度
print(f"模型期望输入维度: {model.n_features_in_}")

# 256维模型：model.n_features_in_ = 256
# 12维模型：model.n_features_in_ = 12
```

### 自动适配器
```python
from sensor_data_converter import HardwareAdapterFor256Model

# 创建适配器，自动检测模型类型
adapter = HardwareAdapterFor256Model(model_path)

# 直接输入12路数据，适配器会自动处理转换
prediction, probabilities = adapter.predict_from_hardware(sensor_data_12)
```

## 传感器位置映射

### 12路传感器在256维数组中的位置
```
传感器1  -> 数组索引48  -> 网格位置(3,0)   # 左上
传感器2  -> 数组索引80  -> 网格位置(5,0)   # 左中上  
传感器3  -> 数组索引112 -> 网格位置(7,0)   # 左中
传感器4  -> 数组索引176 -> 网格位置(11,0)  # 左下
传感器5  -> 数组索引87  -> 网格位置(5,7)   # 左内1
传感器6  -> 数组索引103 -> 网格位置(6,7)   # 左内2
传感器7  -> 数组索引88  -> 网格位置(5,8)   # 中央1
传感器8  -> 数组索引89  -> 网格位置(5,9)   # 中央2
传感器9  -> 数组索引104 -> 网格位置(6,8)   # 中央3
传感器10 -> 数组索引105 -> 网格位置(6,9)   # 中央4
传感器11 -> 数组索引91  -> 网格位置(5,11)  # 右内1
传感器12 -> 数组索引107 -> 网格位置(6,11)  # 右内2
```

**注意**：这些是基于16x16网格的索引位置，从0开始计数。

## 实际使用场景

### 场景1：新项目（推荐）
- 使用12维模型 (`models_12_sensors/`)
- 直接处理12路传感器数据
- 使用 `real_time_detector_12_direct.py`

### 场景2：已有256维模型
- 继续使用现有的256维模型
- 使用转换器扩展12路数据到256维
- 使用 `sensor_data_converter.py` 的转换功能

### 场景3：混合环境
- 使用自动适配器 `HardwareAdapterFor256Model`
- 自动检测模型类型并处理数据转换
- 支持两种模型的无缝切换

## 代码示例

### 方案1：直接使用12维模型
```python
from real_time_detector_12_direct import DirectTwelveSensorDetector

detector = DirectTwelveSensorDetector()

def hardware_reading_loop():
    while True:
        # 读取12路传感器
        sensor_data_12 = read_12_sensors_from_hardware()
        
        # 直接预测
        result = detector.predict_ensemble(sensor_data_12)
        
        print(f"坐姿: {result['prediction']}, 置信度: {result['confidence']:.1%}")
        time.sleep(0.5)
```

### 方案2：256维模型兼容
```python
from sensor_data_converter import SensorDataConverter
import joblib

# 加载256维模型
model_256 = joblib.load('model_256.pkl')
converter = SensorDataConverter()

def hardware_reading_loop_256():
    while True:
        # 读取12路传感器
        sensor_data_12 = read_12_sensors_from_hardware()
        
        # 扩展到256维
        sensor_data_256 = converter.expand_12_to_256(sensor_data_12)
        
        # 使用256维模型预测
        prediction = model_256.predict([sensor_data_256])
        
        print(f"坐姿: {prediction[0]}")
        time.sleep(0.5)
```

### 方案3：自动适配（推荐）
```python
from sensor_data_converter import HardwareAdapterFor256Model

# 自动适配任何模型
adapter = HardwareAdapterFor256Model('your_model.pkl')

def hardware_reading_loop_auto():
    while True:
        # 读取12路传感器
        sensor_data_12 = read_12_sensors_from_hardware()
        
        # 自动处理并预测
        prediction, probabilities = adapter.predict_from_hardware(sensor_data_12)
        
        if probabilities is not None:
            confidence = max(probabilities)
            print(f"坐姿: {prediction}, 置信度: {confidence:.1%}")
        else:
            print(f"坐姿: {prediction}")
        
        time.sleep(0.5)
```

## 性能对比

| 方案 | 数据维度 | 内存占用 | 处理速度 | 硬件要求 | 推荐度 |
|------|----------|----------|----------|----------|---------|
| 12维直接 | 12 | 低 | 快 | 简单 | ⭐⭐⭐⭐⭐ |
| 256维转换 | 256 | 高 | 较慢 | 简单 | ⭐⭐⭐ |
| 自动适配 | 自动 | 中等 | 中等 | 简单 | ⭐⭐⭐⭐ |

## 总结

1. **当前最佳实践**：使用12维模型，直接处理12路传感器数据
2. **兼容性考虑**：如果必须使用256维模型，使用转换器扩展数据
3. **灵活性方案**：使用自动适配器，支持两种模型类型
4. **硬件简化**：无论哪种方案，都只需要12路压力传感器

**关键点**：数据扩展只是软件层面的处理，硬件上仍然只需要12路传感器。扩展到256维是为了兼容原始模型的输入格式要求。