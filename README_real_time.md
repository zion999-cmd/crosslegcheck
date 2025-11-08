# 实时压力传感器检测系统使用指南

## 🚀 系统概述

实时压力传感器检测系统是一个集成了硬件数据采集和AI智能识别的完整解决方案。系统能够：

- 📡 **实时采集**: 通过串口实时获取256路压力传感器数据
- 🧠 **智能识别**: 使用CNN深度学习模型识别压力分布状态
- ⚡ **高效处理**: 支持单线程和多线程两种处理模式
- 📊 **实时反馈**: 提供实时状态显示和统计分析

## 🛠️ 核心组件

### 1. 数据采集模块 (`serial_sensor_reader.py`)
负责通过串口采集压力传感器数据：
- 串口通信 (115200波特率, 8N1)
- 数据帧解析 (AA AB AC帧头)
- 连续/定时采集模式
- 数据导出CSV格式

### 2. AI识别模块 (`cnn_augmented.py`)
基于CNN的压力状态识别：
- 16×16压力图像转换
- 数据增强技术
- 3分类识别: 左偏/正常/右偏
- 置信度评估

### 3. 实时检测系统 (`real_time_detector.py`)
集成采集和识别的实时系统：
- 多线程并行处理
- 实时状态显示
- 统计分析功能
- 错误恢复机制

## 📚 使用方法

### 🔧 环境准备

```bash
# 1. 确保Python环境已配置
conda activate crosslegcheck

# 2. 检查依赖包
pip list | grep -E "(tensorflow|numpy|pandas|pyserial)"

# 3. 检查模型文件
ls -la ../models/cnn_augmented_model.keras
```

### 📡 数据采集

#### 基础采集
```bash
# 监控模式 - 实时查看原始数据
python serial_sensor_reader.py monitor

# 采集指定数量样本
python serial_sensor_reader.py collect 50 my_data.csv

# 查看帮助
python serial_sensor_reader.py --help
```

#### 采集参数
- **端口**: `/dev/cu.usbserial-14220`
- **波特率**: `115200`
- **数据格式**: `8N1`
- **帧长度**: `515字节` (3字节头 + 512字节数据)
- **超时时间**: `30秒`

### 🧠 模型预测

#### 使用保存的数据
```bash
# 预测CSV文件
python cnn_augmented.py predict ../data/test_data.csv

# 预测并指定期望标签
python cnn_augmented.py predict ../data/test_data.csv normal

# 重新训练模型
python cnn_augmented.py train
```

### ⚡ 实时检测

#### 推荐使用方式

```bash
# 1. 快速测试模型
python real_time_detector.py test

# 2. 演示模式 (使用已保存数据)
python real_time_detector.py demo

# 3. 简化实时检测 (推荐)
python real_time_detector.py simple 20

# 4. 完整多线程检测
python real_time_detector.py full
```

#### 检测模式说明

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| `test` | 仅测试模型加载和预测 | 验证环境配置 |
| `demo` | 使用保存数据模拟检测 | 演示系统效果 |
| `simple` | 单线程实时检测 | 日常使用，稳定可靠 |
| `full` | 多线程并行检测 | 高性能需求场景 |

## 📊 输出说明

### 实时检测显示格式
```
[07:27:08] 🟢 样本  1: 正常 (置信度: 0.964) | 数据: 0-57824
[07:27:08] 🟡 样本  2: 左偏 (置信度: 0.652) | 数据: 0-45231
[07:27:08] 🔴 样本  3: 右偏 (置信度: 0.421) | 数据: 0-38754
```

### 指示符含义
- 🟢 **高置信度** (>0.8): 预测结果非常可靠
- 🟡 **中等置信度** (0.6-0.8): 预测结果较为可靠
- 🔴 **低置信度** (<0.6): 预测结果需要谨慎对待

### 状态分类
- **正常**: 压力分布均匀，状态良好
- **左偏**: 压力重心偏向左侧
- **右偏**: 压力重心偏向右侧

## 🔧 故障排除

### 常见问题

#### 1. 串口连接失败
```bash
# 检查串口设备
ls /dev/cu.usb*

# 检查串口权限
sudo chmod 666 /dev/cu.usbserial-14220
```

#### 2. 模型加载失败
```bash
# 检查模型文件
ls -la ../models/cnn_augmented_model.keras

# 重新训练模型
python cnn_augmented.py train
```

#### 3. 数据采集无响应
- 检查硬件连接
- 重启串口设备
- 确认波特率设置
- 查看终端错误信息

#### 4. 预测结果异常
- 检查输入数据范围
- 验证模型文件完整性
- 确认数据格式正确

### 性能优化

#### 1. 提高采集速度
```python
# 在 serial_sensor_reader.py 中调整
time.sleep(0.001)  # 减少延迟
```

#### 2. 优化预测性能
```python
# 批量预测
predictions = model.predict(batch_images, batch_size=32)
```

#### 3. 内存管理
```python
# 限制队列大小
data_queue = queue.Queue(maxsize=50)
```

## 📈 系统性能

### 基准测试结果
- **采集速率**: ~3-5 样本/秒
- **预测延迟**: ~100-200ms
- **模型准确率**: >95% (验证集)
- **内存占用**: ~500MB
- **CPU占用**: ~20-30%

### 优化建议
1. 使用SSD硬盘提高I/O性能
2. 增加内存缓解队列阻塞
3. 使用GPU加速模型推理
4. 调整线程数匹配CPU核心数

## 🔄 系统集成

### 与其他系统集成
```python
# 导入检测器
from real_time_detector import SimpleRealTimeDetector

# 自定义回调函数
def on_detection_result(result):
    print(f"检测到: {result['label']}, 置信度: {result['confidence']}")

# 集成到自己的系统
detector = SimpleRealTimeDetector()
detector.result_callback = on_detection_result
```

### API接口
系统提供了清晰的模块化接口，便于集成到更大的系统中：

- `PressureSensorReader`: 硬件数据采集
- `pressure_to_image()`: 数据格式转换
- `predict_with_augmented_cnn()`: AI模型预测
- `RealTimeDetector`: 完整实时检测系统

## 📝 开发说明

### 代码结构
```
scripts/
├── serial_sensor_reader.py     # 串口数据采集
├── cnn_augmented.py           # CNN模型训练和预测
├── real_time_detector.py      # 实时检测系统
├── simple_effective_cnn.py    # 简化CNN模型
└── README_real_time.md        # 本文档
```

### 扩展开发
1. **新增传感器支持**: 修改 `serial_sensor_reader.py` 中的数据解析逻辑
2. **模型优化**: 在 `cnn_augmented.py` 中调整网络结构
3. **新增状态类型**: 扩展分类数量和标签定义
4. **界面开发**: 基于现有接口开发GUI界面

## 🎯 最佳实践

### 1. 日常使用流程
```bash
# 步骤1: 系统检查
python real_time_detector.py test

# 步骤2: 快速检测
python real_time_detector.py simple 20

# 步骤3: 数据分析 (如需要)
python serial_sensor_reader.py collect 100 analysis_data.csv
```

### 2. 数据质量保证
- 定期校验传感器硬件
- 收集不同状态的标准样本
- 定期重新训练模型
- 监控预测置信度趋势

### 3. 系统维护
- 每周备份模型文件
- 定期清理日志文件
- 监控系统资源使用
- 更新依赖包版本

---

## 📞 技术支持

如有问题，请查看：
1. 终端错误信息
2. 系统日志输出
3. 模型训练历史
4. 硬件连接状态

系统经过充分测试，在正常环境下运行稳定可靠。