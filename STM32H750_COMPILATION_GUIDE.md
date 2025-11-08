# STM32H750项目编译完整指南

## 🎯 项目现状
您的项目位于 `/Users/bx/Workspace/crosslegcheck/7-H750-LCD130H/`，这是一个STM32CubeMX生成的项目，包含：
- ✅ `H750.ioc` - CubeMX配置文件
- ✅ `MDK-ARM/` - Keil项目文件
- ✅ `Core/`, `Drivers/`, `HARDWARE/` - 源代码目录

## 🛠️ 编译方式选择（推荐顺序）

### 方式1: 使用Keil MDK-ARM（最简单）⭐️

#### 1.1 安装Keil MDK
```bash
# 下载Keil MDK-ARM（Windows环境）
# 访问：https://www.keil.com/download/product/
# 选择：MDK-ARM（支持STM32H7系列）
```

#### 1.2 打开项目
1. 启动Keil μVision
2. 打开项目文件：`7-H750-LCD130H/MDK-ARM/H750.uvprojx`
3. 项目会自动加载所有文件

#### 1.3 配置编译器
- Target: STM32H750VBTx
- Compiler: ARM Compiler V6 (推荐) 或 V5
- Optimization: -O2

#### 1.4 编译步骤
```
Project → Build Target (F7)
或者点击工具栏的锤子图标
```

### 方式2: 使用STM32CubeIDE（免费推荐）⭐️⭐️

#### 2.1 安装STM32CubeIDE
```bash
# 下载STM32CubeIDE（免费，跨平台）
# 访问：https://www.st.com/en/development-tools/stm32cubeide.html
# 支持 Windows、Linux、macOS
```

#### 2.2 导入项目
1. 启动STM32CubeIDE
2. File → Import → General → Existing Projects into Workspace
3. 选择项目根目录：`7-H750-LCD130H/`
4. 导入项目

#### 2.3 编译项目
```
Project → Build Project (Ctrl+B)
或者右键项目 → Build Project
```

### 方式3: 使用命令行编译（macOS推荐）⭐️⭐️⭐️

#### 3.1 安装ARM工具链
```bash
# 安装ARM GCC工具链
brew install armmbed/formulae/arm-none-eabi-gcc

# 或者下载官方版本
# https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm
```

#### 3.2 生成Makefile项目
由于您有CubeMX项目，可以重新生成支持Makefile的项目：

1. 安装STM32CubeMX（免费）：
```bash
# 下载：https://www.st.com/en/development-tools/stm32cubemx.html
```

2. 打开项目并重新生成：
```bash
# 打开 H750.ioc 文件
# Project Manager → Toolchain/IDE → 选择 "Makefile"
# 点击 "GENERATE CODE"
```

#### 3.3 编译命令
```bash
cd /Users/bx/Workspace/crosslegcheck/7-H750-LCD130H
make clean
make -j4  # 使用4个线程编译
```

## 📝 添加我们的代码

### 1. 复制文件到项目
```bash
# 将我们创建的文件复制到正确位置

# 传感器接收器
cp HARDWARE/SENSOR_RECEIVER/* 7-H750-LCD130H/HARDWARE/SENSOR_RECEIVER/

# 姿态显示（如果已存在POSTURE_DISPLAY目录）
cp HARDWARE/POSTURE_DISPLAY/* 7-H750-LCD130H/HARDWARE/POSTURE_DISPLAY/

# 主程序
cp Core/Src/main_sensor_module.c 7-H750-LCD130H/Core/Src/
```

### 2. 修改项目配置

#### 2.1 添加包含路径
在Keil或CubeIDE中添加：
```
HARDWARE/SENSOR_RECEIVER
HARDWARE/POSTURE_DISPLAY
HARDWARE/LCD130H
HARDWARE/delay
```

#### 2.2 添加源文件
确保以下文件被包含在编译中：
```
HARDWARE/SENSOR_RECEIVER/sensor_data_receiver.c
HARDWARE/POSTURE_DISPLAY/posture_display.c
HARDWARE/LCD130H/lcd130h.c
HARDWARE/delay/delay.c
Core/Src/main_sensor_module.c
```

### 3. 修改主文件
如果项目有现有的 `main.c`，您需要：
1. 备份原 `main.c` 为 `main_backup.c`
2. 将 `main_sensor_module.c` 重命名为 `main.c`

## 🔧 可能遇到的问题和解决方案

### 问题1: 缺少HAL库
```c
// 错误：找不到 stm32h7xx_hal.h
// 解决：确保Drivers目录包含STM32H7xx_HAL_Driver
```

### 问题2: 时钟配置错误
```c
// 错误：SystemClock_Config未定义
// 解决：使用CubeMX重新生成时钟配置
```

### 问题3: 缺少外设配置
```c
// 错误：huart1 未定义
// 解决：在CubeMX中配置USART1
```

## 🎯 快速开始（推荐步骤）

### 对于macOS用户（您的情况）:

#### 1. 安装STM32CubeIDE（最简单）
```bash
# 下载STM32CubeIDE for macOS
# https://www.st.com/en/development-tools/stm32cubeide.html
```

#### 2. 打开现有项目
```bash
# 启动STM32CubeIDE
# File → Open Projects from File System
# 选择：/Users/bx/Workspace/crosslegcheck/7-H750-LCD130H
```

#### 3. 检查项目配置
- 确认Target是STM32H750VBTx
- 检查USART1、SPI2、GPIO配置
- 验证时钟配置（480MHz）

#### 4. 添加我们的代码
- 将SENSOR_RECEIVER文件夹复制到HARDWARE目录
- 更新包含路径和源文件列表

#### 5. 编译测试
```
Project → Build Project
检查编译输出，解决任何错误
```

## 🔍 编译输出文件

成功编译后，您会得到：
```
MDK-ARM/H750/H750.hex      # 十六进制固件文件
MDK-ARM/H750/H750.bin      # 二进制固件文件  
MDK-ARM/H750/H750.elf      # ELF调试文件
```

## 📱 烧录和调试

### 使用ST-Link烧录：
```bash
# 使用STM32CubeProgrammer
# 或者在CubeIDE中直接Run/Debug
```

### 串口调试：
```bash
# 连接USART1 (PA9-TX) 到USB-TTL转换器
# 波特率：115200
# 使用串口工具查看调试输出
```

---

您想选择哪种编译方式？我建议从STM32CubeIDE开始，因为它是免费的且对初学者最友好。