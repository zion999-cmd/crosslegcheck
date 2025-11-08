# 🚀 CrossLegCheck 快速上手指南

## 🎯 推荐使用：12传感器系统

### ⚡ 1分钟快速体验

```bash
# 1. 激活环境
conda activate crosslegcheck

# 2. 演示模式 - 立即体验效果
python scripts/real_time_detector_12.py --demo --full

# 3. 硬件模式 - 连接真实传感器
python scripts/real_time_detector_12.py --port /dev/cu.usbserial-14220 --full

# 4. 快速测试 - 验证模型性能
python scripts/test_12_sensor_quick.py
```

### 🎛️ 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--demo` | 演示模式，使用测试数据 | `--demo` |
| `--port` | 串口设备路径 | `--port /dev/cu.usbserial-14220` |
| `--full` | 完整模式，无限循环直到Ctrl+C | `--full` |
| `--baudrate` | 波特率设置 | `--baudrate 115200` |

### 📊 界面说明

运行后会看到实时更新的界面：
```
🪑 12传感器实时坐姿检测系统
============================================================
🎯 检测结果: ✅ 正常 (置信度: 97.4%)

📊 模型对比:
   Logistic回归: normal (置信度: 95.7%)
   随机森林:     normal (置信度: 99.0%)

📡 12传感器数据:
   总压力: 8,571 克
   最大值: 1,203 克
   非零传感器: 12/12

⚡ 性能信息:
   预测时间: 15.6 ms

📈 运行统计:
   总检测次数: 148
   检测频率: 4.9 次/秒
   平均置信度: 97.4%

💡 按 Ctrl+C 停止检测
```

## 🔧 故障排除

### ❌ 常见问题

1. **环境未激活**
   ```bash
   # 错误提示：ModuleNotFoundError: No module named 'numpy'
   # 解决方案：
   conda activate crosslegcheck
   ```

2. **串口连接失败**
   ```bash
   # 错误提示：串口连接失败
   # 解决方案：检查串口设备路径
   ls /dev/cu.*  # macOS
   ls /dev/tty*  # Linux
   ```

3. **模型文件缺失**
   ```bash
   # 错误提示：FileNotFoundError: ...models_12_sensors/...
   # 解决方案：重新生成模型
   python scripts/create_12_sensor_model.py
   ```

### ✅ 验证安装

```bash
# 检查Python环境
python --version  # 应该显示 Python 3.x

# 检查依赖包
python -c "import numpy, pandas, sklearn, joblib; print('✅ 所有依赖已安装')"

# 检查模型文件
ls models_12_sensors/  # 应该显示 .pkl 文件
```

## 📈 预期效果

### 🎯 正常运行指标
- **准确率**: 95%-97%
- **预测时间**: <20ms
- **检测频率**: 2-5次/秒
- **置信度**: >90%

### 🎛️ 坐姿检测
- **👈 左偏**: 身体向左倾斜
- **✅ 正常**: 身体居中平衡  
- **👉 右偏**: 身体向右倾斜

## 🔄 常用命令

```bash
# 🎯 最常用：完整实时检测
python scripts/real_time_detector_12.py --port /dev/cu.usbserial-14220 --full

# 🎲 演示模式
python scripts/real_time_detector_12.py --demo --full

# 📊 性能测试
python scripts/test_12_sensor_quick.py

# 🏗️ 重新训练模型
python scripts/create_12_sensor_model.py

# 🔍 检查传感器连接
python scripts/serial_sensor_reader.py check /dev/cu.usbserial-14220
```

## 📚 更多信息

- **详细文档**: [README_12_sensors.md](README_12_sensors.md)
- **项目结构**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **STM32部署**: `embedded_12_sensors/` 目录

## 🎉 成功标志

当你看到这样的输出时，说明系统运行正常：

```
✅ 12传感器模型加载成功
✅ 12传感器读取器初始化成功
🚀 启动12传感器实时坐姿检测系统
📡 进入无限循环数据采集模式...
🧠 12传感器预测处理线程启动
📊 12传感器结果显示线程启动
✅ 所有线程已启动，开始检测...
```

---

🎯 **开始您的智能坐姿检测之旅吧！**