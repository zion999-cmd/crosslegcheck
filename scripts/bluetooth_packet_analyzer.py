
import serial
import time
from collections import Counter

# 配置参数
SERIAL_PORT = 'cu.BTCOM'  # macOS下蓝牙串口名
BAUDRATE = 9600            # 与STM32和蓝牙模块一致
TIMEOUT = 1
BUFFER_SIZE = 4096         # 数据缓冲区大小
HEADER_CANDIDATE_LEN = 32  # 检查前32字节是否有稳定包头
MIN_STABLE_COUNT = 50      # 某字节稳定出现次数阈值

print(f"连接蓝牙串口: {SERIAL_PORT} 速率: {BAUDRATE}")
ser = serial.Serial(f'/dev/{SERIAL_PORT}', BAUDRATE, timeout=TIMEOUT)

buffer = bytearray()
history = [[] for _ in range(HEADER_CANDIDATE_LEN)]  # 每个位置的历史值

try:
    while True:
        data = ser.read(BUFFER_SIZE)
        if data:
            buffer.extend(data)
            # 记录每个位置的历史值
            for i in range(HEADER_CANDIDATE_LEN):
                values = [buffer[j*HEADER_CANDIDATE_LEN + i] for j in range(len(buffer)//HEADER_CANDIDATE_LEN)]
                history[i] = values[-MIN_STABLE_COUNT:] if len(values) >= MIN_STABLE_COUNT else values

            # 检查每个位置的稳定值和波动
            print("\n=== 字节位置统计 ===")
            max_var = 0
            max_var_pos = -1
            for i in range(HEADER_CANDIDATE_LEN):
                if not history[i]:
                    continue
                most_common, count = Counter(history[i]).most_common(1)[0]
                value_range = max(history[i]) - min(history[i]) if len(history[i]) > 1 else 0
                stable = count > MIN_STABLE_COUNT * 0.9
                print(f"位置{i:02d}: ", end="")
                print(f"稳定值: 0x{most_common:02X} ({count}次)", end="; " if stable else "; ")
                print(f"波动范围: {value_range}", end="")
                if stable:
                    print("  <-- 长期未变化（包头候选）", end="")
                if value_range > max_var:
                    max_var = value_range
                    max_var_pos = i
                print()
            if max_var_pos >= 0:
                print(f"\n波动最大的位置: {max_var_pos}，波动范围: {max_var}")
            print("="*40)

            # 控制缓冲区大小
            if len(buffer) > BUFFER_SIZE * 10:
                buffer = buffer[-BUFFER_SIZE*5:]
        time.sleep(0.2)
except KeyboardInterrupt:
    print("采集结束")
    ser.close()
