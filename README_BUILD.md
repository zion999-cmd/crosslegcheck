# 构建说明（macOS + STM32CubeIDE / arm-none-eabi）

这个仓库的 `cubeide` 子目录由 STM32CubeIDE 生成。你可以用两种常见方式来构建：

A) 使用 STM32CubeIDE GUI：
   1. 用 STM32CubeIDE 打开工程（导入或直接打开 `cubeide` 项目）。
   2. 选择 Debug / Release 配置并点击 Build。

B) 在终端使用 CubeIDE 自带的工具链或系统安装的 arm-none-eabi 工具链（推荐用于自动化）：

   我在仓库根目录提供了一个脚本 `build_with_cubeide_toolchain.sh`，它会：
   - 尝试自动定位 `STM32CubeIDE.app` 并找到内置 `arm-none-eabi-*` 工具链；
   - 如果找不到，会尝试查找系统 PATH 中的 `arm-none-eabi-gcc`；
   - 将工具链路径加入 PATH 并在 `cubeide` 子目录运行 `make clean && make -jN all`（并打印结果）。

   使用方法（在仓库根目录）：

   ```zsh
   # 让脚本可执行一次
   chmod +x build_with_cubeide_toolchain.sh
   # 运行（脚本会提示并显示要用的工具链路径）
   ./build_with_cubeide_toolchain.sh
   ```

   脚本会自动使用你的 CPU 核心数作为并行编译参数。如果你想手动指定并行度：

   ```zsh
   BUILD_JOBS=4 ./build_with_cubeide_toolchain.sh
   ```

注意事项：
- 脚本不会安装任何工具链。如果你的机器没有 STM32CubeIDE，请先安装 STM32CubeIDE（推荐）或单独安装 GNU Arm Embedded Toolchain（请参考 ARM 官方或 Homebrew）。
- 如果项目使用了 CubeIDE 的特殊构建步骤（例如自定义 pre-build 操作、外部工具），建议仍使用 STM32CubeIDE GUI 来构建以确保一致性。

如果你希望，我可以把 `ENABLE_USB_DEBUG_OUTPUT` 的编译开关直接写入项目的 Makefile / IDE 设置的预处理宏里，或者把脚本改为在 release 构建时自动传递 `-DENABLE_USB_DEBUG_OUTPUT=0`。告诉我你更偏好哪种方式。