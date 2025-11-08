# 调试历史与当前状态（摘录）

日期: 2025-11-07

概述
- 目标: 使 ATmega32U4 在 STM32H7 上作为 USB CDC 设备被主机（STM32H7 USB Host）枚举并绑定（“调通 CDC”）。
- 手段: 增加 USB Host 低层日志、在板上实现环形日志缓冲并通过 USART2（蓝牙模块）转发实时日志，避免 LCD 刷新对 USB 枚举造成干扰。

关键改动（代码位置）
- `cubeide/Core/Src/usb_dbg.c` / `usb_dbg_uart.c`:
  - 实现了 printf 重定向到环形缓冲 `usb_log_lines[]`，並提供 weak hook `usb_dbg_uart_send()`。
  - 增加了非阻塞的 UART 转发实现（环形缓冲 + DMA 优先，IT 回退）。

- `cubeide/Core/Src/main.c`:
  - 初始化 `USART2`（PA2=TX, PA3=RX），添加启动时与周期性识别包发送（ASCII 测试行 + 识别包 0x55 0xAA 0xF1 0xF2 0x0D 0x0A）。
  - HUD/LCD 原先用于显示完整日志，后改为最小心跳显示以减少对 USB 枚举的影响（仅一行状态，每秒更新）。
  - `MX_USART2_UART_Init()` 波特率已修改为 9600（与蓝牙模块匹配）。

- `cubeide/Core/Src/stm32h7xx_hal_msp.c`:
  - 手动添加并初始化 `hdma_usart2_tx`（DMA1 Stream0, Request USART2_TX），并链接到 `huart2.hdmatx`，同时启用相关 DMA IRQ（手动补丁，推荐后续用 CubeMX 验证或调整 Stream/Request）。

- `cubeide/Core/Src/stm32h7xx_it.c`:
  - 添加 `USART2_IRQHandler()` 并增加 `DMA1_Stream0_IRQHandler()` 调用 `HAL_DMA_IRQHandler(&hdma_usart2_tx)`。

测试与排查过程（已完成）
1. 物理接线检查：确认 MCU PA2 (TX) -> 模块 RX，PA3 (RX) <- 模块 TX；共地。
2. 初始波特问题：最开始 MCU 使用 115200 导致 CoolTerm 上看到大量错帧字节（如 0xFF 交替模式）。
3. 环回测试：短接 PA2<->PA3 并观察输出，确认 MCU 能正确发送 ASCII 测试行与识别包。
4. 将 MCU 波特改为 9600 并在蓝牙端用 9600 捕获，识别包与 ASCII 文本可见，通信稳定。

当前观察
- 板子启动后几秒会显示 `cdc nord h:13 e:7 a: 1` 表示 USB 枚举快完成但 CDC 类注册未成功（Appli_state 达到 READY，但 CDC 未绑定）。
- LCD 现在仅显示最小心跳：用于确认板卡未死机。
- 日志通过蓝牙串口输出并在主机（Mac）终端上捕获。

下一步计划（今天目标）
1. 使用蓝牙日志（USART2）收集 USBH 相关详细日志：设备描述符、控制传输（GetDescriptor）、接口/端点信息、CDC 类初始化失败前后的日志。
2. 分析日志以确定 CDC 注册失败的根因：设备端（ATmega32U4 描述符/类实现问题）或主机端（驱动/时序/中断影响）。
3. 根据分析结果：修改 USB Host 侧（修复类选择逻辑/调整超时/重试）或建议检查/修改设备固件（ATmega32U4）。
4. 若日志量较大且需要长期追踪，可在 `MX_USART2_UART_Init` 上使用 CubeMX 启用 TX DMA（已手动启用一个 DMA stream 作为临时方案）。推荐最终使用 CubeMX 配置以避免手动映射错误。

已完成的 TODO（主要项）
- 初始化 USART2 并实现日志转发钩子：已完成
- 非阻塞环形缓冲（DMA 优先，IT 回退）：已实现
- 启用 USART2 TX DMA（手动补丁）：已实现（建议用 CubeMX 校验）
- 将 LCD 刷新降到最小心跳：已实现
- 识别包补丁并测试：已完成（发送 ASCII + 55 AA F1 F2 0D 0A）

注意事项与风险
- 手动为 DMA 选择 Stream/Request 在不同封装/版本上可能不同；若 DMA 未正确映射，推荐使用 CubeMX 自动生成并合并改动。
- CubeMX 再生成会覆盖 `MX_USART2_UART_Init` 和 MSP 相关代码；在使用 CubeMX 前请备份或用本仓库的 `history.md` 做记录。

日志与交付
- 本文档即为当前调试历史快照 `history.md`。
- 建议后续在新会话窗口继续，本文件和改动已保存于仓库。

---

(若你需要把 `hdma` 更改回由 CubeMX 管理，或需要我生成备份/恢复脚本以方便 CubeMX 再生成，我可以马上添加。)

注意: 我为手动启用 USART2 TX DMA 添加了补丁，但出于兼容性和避免编译失败的考虑，这段手动 DMA 代码已被包裹在宏保护中：
`ENABLE_USART2_MANUAL_DMA`。默认未启用，以避免不同封装/芯片上 DMA 流/请求映射不一致导致编译或运行时问题。

要启用临时手动 DMA：
- 在 `Core/Inc/main.h` 的 `/* USER CODE BEGIN Includes */` 区段或在项目的编译器预处理器宏中添加 `#define ENABLE_USART2_MANUAL_DMA`。
- 注意：启用前请确认你的具体器件 (part number) 使用 `DMA1_Stream0` 和 `DMA_REQUEST_USART2_TX` 是正确的映射，或替换为 CubeMX 给出的 stream/request。

## CDC 修改摘要（重点说明）

下面是本次工程中针对 USB CDC 的所有关键修改点与实现要点，便于后续排查和回退：

- 放宽 CDC 协议匹配
  - 在 USB Host CDC 类初始化/枚举阶段放宽了对 Interface Protocol 字段的检测（接受 Protocol == 0x00），解决了部分使用 ATmega32U4 等微控制器因协议字节不同未被绑定的问题。
  - 相关代码位置：`USB_HOST/App/` 下 CDC 相关 glue/回调处与 `USB_HOST/Target/usbh_conf.h`（宏保护）附近。

- 主动完成 CDC 控制序列以启动设备发送
  - 在 CDC 类激活（class active / APPLICATION_READY）时，增加了以下控制传输顺序：
    1. GetLineCoding（读取当前行编码，作诊断用）
    2. SetLineCoding（将设备行编码设置为 115200, 8N1；以确保设备以预期波特发送）
    3. SetControlLineState（置位 DTR/RTS 等控制线，含简单重试机制）
  - 这些改动解决了许多设备需由主机先行设置行编码或控制线以触发数据发送的兼容问题。
  - 相关代码位置：`USB_HOST/App/usb_host.c`（或同目录下的 CDC class 激活回调实现）。

- 非阻塞接收：USB 回调最小化处理
  - `USBH_CDC_ReceiveCallback()` 在收到数据时不做长时间处理，仅将 `USBH` 提供的接收缓冲（`CDC_RX_Buffer[512]`）里新到的字节快速复制到本地环形缓冲 `cdc_stream_buf`（默认 1024 字节），更新 head/tail 索引并设置可用标志，然后立即重新调用 `USBH_CDC_Receive()` 以续订接收。
  - 当本地环形缓冲满时，策略为丢弃最旧数据以保证回调短小并快速返回（可在以后改为计数并上报统计）。
  - 相关文件：`Core/Src/main.c`（回调实现和环形缓冲定义）。

- 主循环分块转发（保证透明性与低延时）
  - 主循环中的 `process_cdc_stream()` 以小块（CHUNK_SZ，默认 64 字节）从 `cdc_stream_buf` 取出字节，并直接调用 `usb_dbg_uart_send()` 将原始字节透传到蓝牙串口（`USART2`）。
  - 这里保持“透明转发”原则：不对收到的字节做任何解析、修改或过滤，确保 Mac 端收到与设备发送一致的 CSV 字节流。
  - 相关文件：`Core/Src/main.c`。

- 非阻塞 UART 发送实现（避免在 USB 路径阻塞）
  - 把原来阻塞式或弱符号占位的 `usb_dbg_uart_send()` 替换为一个基于 TX 环形缓冲（默认 4 KiB）的非阻塞实现：优先使用 DMA (`HAL_UART_Transmit_DMA`) 发送，若无 DMA 则回退到 IT（中断）方式。
  - `HAL_UART_TxCpltCallback()` 被用来在传输完成后推进缓冲尾并触发下一段发送，保证在高负载下仍然不会阻塞 USB 的回调或主循环。
  - 相关文件：`Core/Src/usb_dbg_uart.c`（实现），`Core/Src/usb_dbg.c`（将发送函数声明为 extern 并使用）。

- 运行时与编译时日志调整
  - 增加运行时日志类别掩码（`USB_LOG_CAT_ENUM`、`USB_LOG_CAT_CDC`、`USB_LOG_CAT_DATA`），并在 `Core/Src/usb_dbg.c` 中将默认 `usb_log_mask` 设为 0（启动时禁用），防止大量数据日志干扰主流数据转发。
  - 对 `USBH_DEBUG_LEVEL`、`USBH_RUNTIME_LOG_ENABLE_AT_BOOT` 等宏添加 `#ifndef` 守护以消除重复定义警告（修改 `USB_HOST/Target/usbh_conf.h` 与 `USB_HOST/App/usb_host.h`）。

- 错误与兼容修复
  - 修复了开发过程中因多次编辑引入的语法错误与链接缺少符号问题（例如 `usb_last_event` 等变量定义缺失）。
  - 清理了早期用于调试的 CSV 解析路径，保留了日志采集点便于按需打开（运行时掩码控制）。

- 调优与后续改进点（备注）
  - 如果目标蓝牙模块波特率低（例如 9600）但 CDC 设备以更高速度发送（例如 115200），将导致 TX 环形缓冲持续增长。优选方案是把蓝牙波特率提升到与设备匹配；替代方案是增大环形缓冲或降低 CHUNK 调度开销。
  - 推荐在现场验证时用低频率的统计日志（例如每 10s 报告一次环缓占用率与丢弃计数）来判断是否需要扩大缓冲或更改策略。

以上变更均以“最小侵入、保证 USB 回调短小与数据透明”为设计准则。若需要，我可以：
- 把 `CHUNK_SZ`、`cdc_stream_buf`、`USB_DBG_UART_RING_SZ` 这些尺寸改为更保守的默认值（例如将主环扩到 4KiB），并提交补丁；或
- 添加一条在蓝牙串口上可交互的命令，用于运行时打开/关闭 `usb_log_mask`（无需刷机）。
# 调试历史与当前状态（摘录）

日期: 2025-11-07

概述
- 目标: 使 ATmega32U4 在 STM32H7 上作为 USB CDC 设备被主机（STM32H7 USB Host）枚举并绑定（“调通 CDC”）。
- 手段: 增加 USB Host 低层日志、在板上实现环形日志缓冲并通过 USART2（蓝牙模块）转发实时日志，避免 LCD 刷新对 USB 枚举造成干扰。

关键改动（代码位置）
- `cubeide/Core/Src/usb_dbg.c` / `usb_dbg_uart.c`:
  - 实现了 printf 重定向到环形缓冲 `usb_log_lines[]`，并提供 weak hook `usb_dbg_uart_send()`。
  - 增加了非阻塞的 UART 转发实现（环形缓冲 + DMA 优先，IT 回退）。

- `cubeide/Core/Src/main.c`:
  - 初始化 `USART2`（PA2=TX, PA3=RX），添加启动时与周期性识别包发送（ASCII 测试行 + 识别包 0x55 0xAA 0xF1 0xF2 0x0D 0x0A）。
  - HUD/LCD 原先用于显示完整日志，后改为最小心跳显示以减少对 USB 枚举的影响（仅一行状态，每秒更新）。
  - `MX_USART2_UART_Init()` 波特率已修改为 9600（与蓝牙模块匹配）。

- `cubeide/Core/Src/stm32h7xx_hal_msp.c`:
  - 手动添加并初始化 `hdma_usart2_tx`（DMA1 Stream0, Request USART2_TX），并链接到 `huart2.hdmatx`，同时启用相关 DMA IRQ（手动补丁，推荐后续用 CubeMX 验证或调整 Stream/Request）。

- `cubeide/Core/Src/stm32h7xx_it.c`:
  - 添加 `USART2_IRQHandler()` 并增加 `DMA1_Stream0_IRQHandler()` 调用 `HAL_DMA_IRQHandler(&hdma_usart2_tx)`。

测试与排查过程（已完成）
1. 物理接线检查：确认 MCU PA2 (TX) -> 模块 RX，PA3 (RX) <- 模块 TX；共地。
2. 初始波特问题：最开始 MCU 使用 115200 导致 CoolTerm 上看到大量错帧字节（如 0xFF 交替模式）。
3. 环回测试：短接 PA2<->PA3 并观察输出，确认 MCU 能正确发送 ASCII 测试行与识别包。
4. 将 MCU 波特改为 9600 并在蓝牙端用 9600 捕获，识别包与 ASCII 文本可见，通信稳定。

当前观察
- 板子启动后几秒会显示 `cdc nord h:13 e:7 a: 1` 表示 USB 枚举快完成但 CDC 类注册未成功（Appli_state 达到 READY，但 CDC 未绑定）。
- LCD 现在仅显示最小心跳：用于确认板卡未死机。
- 日志通过蓝牙串口输出并在主机（Mac）终端上捕获。

下一步计划（今天目标）
1. 使用蓝牙日志（USART2）收集 USBH 相关详细日志：设备描述符、控制传输（GetDescriptor）、接口/端点信息、CDC 类初始化失败前后的日志。
2. 分析日志以确定 CDC 注册失败的根因：设备端（ATmega32U4 描述符/类实现问题）或主机端（驱动/时序/中断影响）。
3. 根据分析结果：修改 USB Host 侧（修复类选择逻辑/调整超时/重试）或建议检查/修改设备固件（ATmega32U4）。
4. 若日志量较大且需要长期追踪，可在 `MX_USART2_UART_Init` 上使用 CubeMX 启用 TX DMA（已手动启用一个 DMA stream 作为临时方案）。推荐最终使用 CubeMX 配置以避免手动映射错误。

已完成的 TODO（主要项）
- 初始化 USART2 并实现日志转发钩子：已完成
- 非阻塞环形缓冲（DMA 优先，IT 回退）：已实现
- 启用 USART2 TX DMA（手动补丁）：已实现（建议用 CubeMX 校验）
- 将 LCD 刷新降到最小心跳：已实现
- 识别包补丁并测试：已完成（发送 ASCII + 55 AA F1 F2 0D 0A）

注意事项与风险
- 手动为 DMA 选择 Stream/Request 在不同封装/版本上可能不同；若 DMA 未正确映射，推荐使用 CubeMX 自动生成并合并改动。
- CubeMX 再生成会覆盖 `MX_USART2_UART_Init` 和 MSP 相关代码；在使用 CubeMX 前请备份或用本仓库的 `history.md` 做记录。

日志与交付
- 本文档即为当前调试历史快照 `history.md`。
- 建议后续在新会话窗口继续，本文件和改动已保存于仓库。

---

(若你需要把 `hdma` 更改回由 CubeMX 管理，或需要我生成备份/恢复脚本以方便 CubeMX 再生成，我可以马上添加。)

注意: 我为手动启用 USART2 TX DMA 添加了补丁，但出于兼容性和避免编译失败的考虑，这段手动 DMA 代码已被包裹在宏保护中：
`ENABLE_USART2_MANUAL_DMA`。默认未启用，以避免不同封装/芯片上 DMA 流/请求映射不一致导致编译或运行时问题。

要启用临时手动 DMA：
- 在 `Core/Inc/main.h` 的 `/* USER CODE BEGIN Includes */` 区段或在项目的编译器预处理器宏中添加 `#define ENABLE_USART2_MANUAL_DMA`。
- 注意：启用前请确认你的具体器件 (part number) 使用 `DMA1_Stream0` 和 `DMA_REQUEST_USART2_TX` 是正确的映射，或替换为 CubeMX 给出的 stream/request。

长期推荐：在 CubeMX 的 .ioc 中为 USART2 启用 TX DMA 并重新生成代码，以确保 MSP/DMA 映射与 HAL 配置一致。
