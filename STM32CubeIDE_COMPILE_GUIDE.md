# STM32CubeIDE 编译STM32H750项目指南

## 🚀 项目导入和编译步骤

### 步骤1: 启动STM32CubeIDE
1. 打开STM32CubeIDE应用程序
2. 选择工作空间位置（可以使用默认位置）

### 步骤2: 导入现有项目
1. **File** → **Import**
2. 选择 **General** → **Existing Projects into Workspace**
3. 点击 **Next**
4. 在 **Select root directory** 中，浏览并选择：
   ```
   /Users/bx/Workspace/crosslegcheck/7-H750-LCD130H
   ```
5. 确保项目被勾选（应该会显示项目名称）
6. 点击 **Finish**

### 步骤3: 配置项目设置
导入后，可能需要确认以下设置：

#### 3.1 确认目标芯片
- 右键项目 → **Properties**
- **C/C++ Build** → **Settings**
- **Tool Settings** 标签下确认：
  - Target: STM32H750VBTx
  - Float ABI: Hard
  - FPU Type: fpv5-d16

#### 3.2 检查包含路径
在 **Properties** → **C/C++ Build** → **Settings** → **Tool Settings**：
- **MCU GCC Compiler** → **Include paths**
- 确认包含这些路径：
  ```
  Core/Inc
  Drivers/STM32H7xx_HAL_Driver/Inc
  Drivers/STM32H7xx_HAL_Driver/Inc/Legacy
  Drivers/CMSIS/Device/ST/STM32H7xx/Include
  Drivers/CMSIS/Include
  HARDWARE/SENSOR_RECEIVER
  HARDWARE/POSTURE_DISPLAY
  HARDWARE/LCD130H
  HARDWARE/delay
  HARDWARE/usart
  HARDWARE/LED
  ```

### 步骤4: 编译项目
1. **方法1**: 点击工具栏的 🔨 (Build) 图标
2. **方法2**: 右键项目名 → **Build Project**
3. **方法3**: 菜单 **Project** → **Build Project**
4. **快捷键**: `Ctrl+B` (Windows/Linux) 或 `Cmd+B` (Mac)

### 步骤5: 查看编译结果
- 在底部的 **Console** 窗口查看编译输出
- 如果成功，会显示类似：
  ```
  Finished building target: H750.elf
  arm-none-eabi-size H750.elf
  text    data     bss     dec     hex filename
  xxxxx   xxxx    xxxx   xxxxx   xxxxx H750.elf
  Finished building: default target
  ```

### 步骤6: 找到编译输出文件
编译成功后，输出文件位于：
```
/Users/bx/Workspace/crosslegcheck/7-H750-LCD130H/Debug/
├── H750.elf    # 调试文件
├── H750.bin    # 二进制固件文件
├── H750.hex    # 十六进制固件文件
└── H750.map    # 内存映射文件
```

## 🔧 可能遇到的问题和解决方案

### 问题1: 编译错误 "No such file or directory"
**解决**: 检查文件路径，确保所有源文件都存在

### 问题2: 类型定义冲突错误
**解决**: 
1. 右键项目 → **Properties**
2. **C/C++ Build** → **Settings** → **Tool Settings**
3. **MCU GCC Compiler** → **Preprocessor**
4. 在 **Defined symbols** 中添加：
   ```
   USE_HAL_DRIVER
   STM32H750xx
   ```

### 问题3: 链接器错误
**解决**: 
1. **Properties** → **C/C++ Build** → **Settings**
2. **MCU GCC Linker** → **General**
3. 确认链接脚本路径正确

### 问题4: 固件包版本问题
如果提示固件包版本不匹配：
1. **Help** → **Manage Embedded Software Packages**
2. 找到 **STM32Cube MCU Package for STM32H7 Series**
3. 安装或更新到版本 1.12.1

## 🎯 编译成功后的下一步

### 1. 连接硬件
- 连接ST-Link调试器到STM32H750开发板
- 连接USB线到电脑

### 2. 烧录程序
1. 右键项目 → **Run As** → **STM32 C/C++ Application**
2. 或者点击工具栏的 ▶️ (Run) 按钮
3. 首次运行会要求配置调试器设置

### 3. 调试程序
1. 右键项目 → **Debug As** → **STM32 C/C++ Application**
2. 或者点击工具栏的 🐛 (Debug) 按钮

### 4. 串口监控
- 连接USART1 (PA9-TX) 到USB-TTL转换器
- 使用串口工具监控调试输出
- 波特率：115200

## ✅ 验证清单

- [ ] 项目成功导入到CubeIDE
- [ ] 编译无错误完成
- [ ] 生成了.elf, .bin, .hex文件
- [ ] 文件大小合理（通常几十KB到几百KB）
- [ ] 准备好硬件连接进行测试

---

如果遇到任何问题，请截图错误信息，我会帮您具体解决！