# 坐姿分类系统使用说明

本系统包含3个核心脚本，分别用于训练、预测和评估坐姿分类模型。

## 📁 文件结构

```
crosslegcheck/
├── train.py          # 训练脚本
├── predict.py         # 预测脚本  
├── evaluate.py        # 评估脚本
├── dataset.csv        # 训练数据集
├── test_dataset.csv   # 测试数据集
└── model_*.pkl        # 训练生成的模型文件
```

## 🚀 快速开始

### 1. 训练模型

```bash
# 标准训练模式
python train.py standard dataset.csv

# 改进训练模式（网格搜索最优参数）
python train.py improved dataset.csv
```

**训练模式说明：**
- `standard`: 使用固定参数快速训练
- `improved`: 使用网格搜索找到最优参数（耗时较长但效果更好）

### 2. 模型预测

```bash
# CSV文件批量预测
python predict.py csv test_dataset.csv

# 使用改进模型预测
python predict.py --improved csv test_dataset.csv

# 交互式预测
python predict.py interactive

# 单个样本预测（256个压力值用逗号分隔）
python predict.py single "67,0,0,0,..."
```

### 3. 模型评估

```bash
# 在测试集上评估模型
python evaluate.py test test_dataset.csv

# 数据集分析
python evaluate.py analyze dataset.csv

# 对比标准模型和改进模型
python evaluate.py compare test_dataset.csv

# 使用改进模型评估
python evaluate.py --improved test test_dataset.csv
```

## 📊 数据格式要求

### CSV文件格式
```
Label,F1(g),F2(g),F3(g),...,F256(g)
left,67,0,0,...,0
normal,45,12,8,...,15
right,23,89,45,...,67
```

- 第一列：`Label` (必须，值为 left/normal/right)
- 后256列：`F1(g)` 到 `F256(g)` (压力传感器数据)

### 预测时也支持无标签格式
```
F1(g),F2(g),F3(g),...,F256(g)
67,0,0,...,0
45,12,8,...,15
```

## 🎯 功能特点

### 训练脚本 (train.py)
- ✅ 支持标准和改进两种训练模式
- ✅ 自动数据预处理（标准化、PCA降维）
- ✅ 模型性能评估和可视化
- ✅ 自动保存模型文件
- ✅ 过拟合检测

### 预测脚本 (predict.py)
- ✅ 支持单个样本、批量、CSV文件预测
- ✅ 交互式预测界面
- ✅ 自动编码检测
- ✅ 支持标准和改进模型
- ✅ 预测结果统计分析

### 评估脚本 (evaluate.py)
- ✅ 详细的模型性能评估
- ✅ 数据集质量分析
- ✅ 错误分析和改进建议
- ✅ 混淆矩阵可视化
- ✅ 模型对比功能

## 📈 模型性能

当前模型在测试集上的表现：
- **总体准确率**: 95.67%
- **left类别**: 精确率90%, 召回率99%
- **normal类别**: 精确率99%, 召回率95%  
- **right类别**: 精确率97%, 召回率94%

## 🔧 生成的文件

训练后会生成以下文件：
- `model_svm.pkl` / `model_svm_improved.pkl` - SVM模型
- `scaler.pkl` / `scaler_improved.pkl` - 数据标准化器
- `pca.pkl` / `pca_improved.pkl` - PCA降维器
- `confusion_matrix*.png` - 混淆矩阵图
- `data_analysis*.png` - 数据分析图

## 💡 使用建议

1. **首次使用**：先用标准模式快速训练验证效果
2. **性能优化**：使用改进模式进行参数优化
3. **模型验证**：在独立测试集上评估真实性能
4. **部署应用**：使用预测脚本进行实际应用

## ⚠️ 注意事项

- 确保CSV文件编码正确（支持UTF-8、GBK等）
- 训练数据必须包含Label列
- 预测数据必须是256维压力值
- 建议在训练前分析数据集质量

## 🔍 故障排除

1. **编码错误**：脚本会自动尝试多种编码
2. **维度错误**：检查数据是否为256维
3. **模型文件缺失**：确保先执行训练
4. **内存不足**：减少数据量或使用标准模式

---

如有问题，请检查错误提示信息，或查看脚本内的详细注释。