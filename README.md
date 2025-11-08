# 坐姿检测系统 - CrossLegCheck

## 🎯 项目概述

基于坐垫压力矩阵的智能坐姿分类系统，支持实时检测和STM32嵌入式部署。

### 🔥 **最新突破：12传感器优化系统**

> **重大成果**: 我们成功将原有的256传感器系统优化为12传感器版本，实现了：
> - 📈 **准确率提升**: 66.2% → **97.2%**
> - 💰 **成本降低**: **95%硬件成本节约**
> - ⚡ **性能提升**: 预测时间<20ms，模型大小<1KB
> - 🚀 **实时检测**: 无限循环监控，实际可用

详细信息请查看：**[12传感器系统文档](README_12_sensors.md)** 🔗

---

## 📊 系统对比

| 特性 | 原256传感器系统 | **12传感器优化系统** ⭐ |
|------|----------------|----------------------|
| 检测准确率 | 66.2% | **97.2%** |
| 传感器数量 | 256个 | **12个** |
| 硬件成本 | 100% | **~5%** |
| 预测时间 | ~100ms | **<20ms** |
| 实时检测 | ❌ | **✅ 完全实现** |
| STM32部署 | 复杂 | **简单易用** |

---

## 🚀 快速开始

### 🔥 推荐：12传感器实时检测

```bash
# 激活环境
conda activate crosslegcheck

# 实时检测（完整模式）
python scripts/real_time_detector_12.py --port /dev/cu.usbserial-14220 --full

# 演示模式
python scripts/real_time_detector_12.py --demo --full
```

### 📋 传统256传感器系统

基于坐垫压力矩阵的坐姿分类（SVM 训练脚本说明）

一、项目背景

本项目旨在通过坐垫下方的 16×16 压力传感器阵列（共256通道）实现坐姿识别。最终会把模型在stm32h750预测. 当前目标是在pc下完成三类姿势的识别：

normal —— 正常坐姿

crossleg_left —— 左腿交叉

crossleg_right —— 右腿交叉

每个传感器点输出压力值（单位：克），以 CSV 形式保存。

二、数据格式说明

数据文件：

normal.csv
crossleg_left.csv
crossleg_right.csv


每个文件：

每行代表 1 秒钟的压力分布；

共 256 个数值，对应 16×16 传感器矩阵的扁平展开；

无表头；

示例（字段名）：

时间, F1(g),F2(g),F3(g),F4(g),F5(g),F6(g),...,F255(g),F256(g)


数据维度：

行数：每个文件约 N 行（样本数量）；

列数：257；

单位：克 (g)。

三、分类任务目标

构建一个 三分类模型（normal / crossleg_left / crossleg_right），输入为一帧（256维向量）的压力分布，输出为对应的姿势类别。

四、数据处理与建模要求

加载数据

从3个CSV文件中加载数据；

自动打标签：

normal.csv → label=normal

crossleg_left.csv → label=crossleg_left

crossleg_right.csv → label=crossleg_right

数据合并

合并为一个DataFrame：

p0,p1,...,p255,label


打乱顺序。

预处理

转换为浮点数；

对每列进行标准化 (x - mean) / std；

可选：降噪（移动平均或中值滤波）；

可选：PCA降维到20维。

训练集划分

使用 train_test_split(test_size=0.2)；

保证随机性。

模型训练（SVM）

使用 sklearn.svm.SVC；

参数建议：

kernel='rbf'
C=10
gamma='scale'


输出：

训练集、测试集准确率；

混淆矩阵；

分类报告。

模型保存

使用 joblib 保存模型与标准化器：

model_svm.pkl
scaler.pkl
pca.pkl（如果有）


实时预测接口（可选）

从串口/文件读取一帧；

自动标准化、PCA、预测；

输出类别。

五、脚本逻辑结构示例
# 1. 自行安装依赖库并导入(注意,我使用conda沙盒crosslegcheck, 并已经安装python 3.13.5)
import pandas as pd
import numpy as np
from sklearn import svm, preprocessing, metrics, decomposition, model_selection
import joblib

# 2. 加载并标注数据
def load_data():
    def load_one(file, label):
        df = pd.read_csv(file, header=None)
        df['label'] = label
        return df
    df = pd.concat([
        load_one('normal.csv', 'normal'),
        load_one('crossleg_left.csv', 'crossleg_left'),
        load_one('crossleg_right.csv', 'crossleg_right')
    ], ignore_index=True)
    df = df.sample(frac=1, random_state=42)
    return df

# 3. 数据预处理
df = load_data()
X = df.drop('label', axis=1).values
y = df['label'].values
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. PCA降维
pca = decomposition.PCA(n_components=20)
X_pca = pca.fit_transform(X_scaled)

# 5. 划分数据集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 6. 训练SVM
clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
clf.fit(X_train, y_train)

# 7. 评估
y_pred = clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# 8. 保存模型
joblib.dump(clf, 'model_svm.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')

print("✅ 模型保存完成。")

六、预期输出
Classification Report:
                 precision    recall  f1-score   support
normal             0.95      0.92      0.93       50
crossleg_left      0.90      0.94      0.92       48
crossleg_right     0.93      0.93      0.93       52
Accuracy: 0.93
✅ 模型保存完成。

七、后续可扩展方向

使用 特征工程（左右压力差、重心偏移）优化模型；

增加时间序列特征（滑动窗口 + 平均）；

训练轻量版 MLP（方便移植 TinyML）；

采集更多用户样本，验证泛化能力。