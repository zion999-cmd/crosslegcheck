#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时压力传感器数据读取脚本
通过串口读取16进制数据，数据帧以AA AB AC开头
"""

import serial
import time
import numpy as np
import struct
from datetime import datetime
import sys

class PressureSensorReader:
    def __init__(self, port='/dev/cu.usbserial-14220', baudrate=115200, 
                 timeout=1.0, frame_header=b'\xAA\xAB\xAC'):
        """
        初始化压力传感器读取器
        
        Args:
            port: 串口设备路径
            baudrate: 波特率
            timeout: 超时时间(秒)
            frame_header: 数据帧头部标识
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.frame_header = frame_header
        self.header_length = len(frame_header)
        self.serial_conn = None
        self.buffer = bytearray()
        
        # 压力数据参数
        self.sensor_count = 256  # 256个压力传感器
        self.frame_length = None  # 待确定的数据帧长度
        
    def connect(self):
        """连接串口"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            print(f"✅ 成功连接到串口: {self.port}")
            print(f"   - 波特率: {self.baudrate}")
            print(f"   - 参数: 8N1")
            print(f"   - 超时: {self.timeout}s")
            return True
        except Exception as e:
            print(f"❌ 串口连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开串口连接"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("🔌 串口已断开")
    
    def read_raw_data(self, size=1024):
        """读取原始数据"""
        if not self.serial_conn or not self.serial_conn.is_open:
            return None
        
        try:
            data = self.serial_conn.read(size)
            return data
        except Exception as e:
            print(f"❌ 读取数据失败: {e}")
            return None
    
    def find_frame_start(self, data):
        """在数据中查找帧头位置"""
        return data.find(self.frame_header)
    
    def parse_hex_data(self, raw_data):
        """解析16进制数据"""
        if not raw_data:
            return None
        
        # 将数据添加到缓冲区
        self.buffer.extend(raw_data)
        
        frames = []
        
        while len(self.buffer) >= self.header_length:
            # 查找帧头
            start_idx = self.find_frame_start(self.buffer)
            
            if start_idx == -1:
                # 没有找到帧头，只保留最后512字节，避免缓冲区过大
                if len(self.buffer) > 512:
                    self.buffer = self.buffer[-512:]
                break
            
            if start_idx > 0:
                # 移除帧头之前的数据
                self.buffer = self.buffer[start_idx:]
            
            # 动态检测数据帧长度
            if self.frame_length is None:
                # 尝试不同的帧长度
                possible_lengths = [
                    self.header_length + self.sensor_count * 2,  # 每个传感器2字节 = 515字节
                    self.header_length + self.sensor_count * 4,  # 每个传感器4字节 = 1027字节
                    516,  # 常见长度
                    520,  # 包含校验位
                ]
                
                for length in possible_lengths:
                    if len(self.buffer) >= length:
                        # 检查这个长度是否合理
                        if length >= self.header_length + 256 * 2:  # 至少256个传感器，每个2字节
                            self.frame_length = length
                            print(f"🔍 使用数据帧长度: {self.frame_length} 字节")
                            break
                
                # 如果还是没确定，使用默认长度
                if self.frame_length is None and len(self.buffer) >= 515:
                    self.frame_length = 515  # 3字节头 + 256*2字节数据
                    print(f"🔍 使用默认数据帧长度: {self.frame_length} 字节")
            
            # 如果确定了帧长度，解析数据
            if self.frame_length and len(self.buffer) >= self.frame_length:
                frame_data = self.buffer[:self.frame_length]
                self.buffer = self.buffer[self.frame_length:]
                
                # 解析压力数据
                pressure_data = self.parse_pressure_frame(frame_data)
                if pressure_data is not None:
                    frames.append(pressure_data)
            else:
                # 如果缓冲区太大但还是没有完整帧，可能数据有问题
                if len(self.buffer) > 2048:
                    print("⚠️  缓冲区过大，清理旧数据")
                    self.buffer = self.buffer[-1024:]
                break
        
        return frames
    
    def parse_pressure_frame(self, frame_data):
        """解析单个压力数据帧"""
        try:
            # 跳过帧头
            payload = frame_data[self.header_length:]
            
            # 根据帧长度确定解析方式
            if self.frame_length == self.header_length + self.sensor_count * 2:
                # 每个传感器2字节
                pressure_values = []
                for i in range(0, len(payload), 2):
                    if i + 1 < len(payload):
                        # 大端序或小端序，根据实际情况调整
                        value = struct.unpack('>H', payload[i:i+2])[0]  # 大端序
                        # value = struct.unpack('<H', payload[i:i+2])[0]  # 小端序
                        pressure_values.append(value)
                
            elif self.frame_length == self.header_length + self.sensor_count * 4:
                # 每个传感器4字节
                pressure_values = []
                for i in range(0, len(payload), 4):
                    if i + 3 < len(payload):
                        value = struct.unpack('>I', payload[i:i+4])[0]  # 大端序
                        # value = struct.unpack('<I', payload[i:i+4])[0]  # 小端序
                        pressure_values.append(value)
            
            else:
                # 其他格式，暂时按2字节处理
                pressure_values = []
                for i in range(0, min(len(payload), self.sensor_count * 2), 2):
                    if i + 1 < len(payload):
                        value = struct.unpack('>H', payload[i:i+2])[0]
                        pressure_values.append(value)
            
            # 确保有256个值
            if len(pressure_values) < self.sensor_count:
                pressure_values.extend([0] * (self.sensor_count - len(pressure_values)))
            elif len(pressure_values) > self.sensor_count:
                pressure_values = pressure_values[:self.sensor_count]
            
            return np.array(pressure_values)
            
        except Exception as e:
            print(f"⚠️  解析压力数据失败: {e}")
            return None
    
    def monitor_data(self, duration=60, display_interval=0.1):
        """监控数据流 - 连续高速获取"""
        print(f"\n🔍 开始连续数据监控 (持续{duration}秒)...")
        print(f"   - 数据帧头: {self.frame_header.hex().upper()}")
        print(f"   - 期望传感器数量: {self.sensor_count}")
        print(f"   - 连续模式: 无延迟高速采集")
        
        start_time = time.time()
        last_display = 0
        last_data_time = start_time
        frame_count = 0
        no_data_count = 0
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # 连续读取原始数据
            raw_data = self.read_raw_data(512)
            
            if raw_data:
                # 减少显示频率，避免输出太快
                if current_time - last_display > display_interval:
                    print(f"\n📡 [{datetime.now().strftime('%H:%M:%S')}] 接收到 {len(raw_data)} 字节:")
                    hex_str = raw_data.hex().upper()
                    
                    # 分行显示，每行32字节
                    for i in range(0, len(hex_str), 64):
                        line = hex_str[i:i+64]
                        formatted_line = ' '.join([line[j:j+2] for j in range(0, len(line), 2)])
                        print(f"   {formatted_line}")
                    
                    # 查找帧头
                    header_pos = self.find_frame_start(raw_data)
                    if header_pos != -1:
                        print(f"   🎯 找到帧头位置: {header_pos}")
                    
                    last_display = current_time
                
                # 解析数据帧
                frames = self.parse_hex_data(raw_data)
                if frames:
                    last_data_time = current_time
                    no_data_count = 0
                    for frame in frames:
                        frame_count += 1
                        print(f"\n✅ 第{frame_count}帧压力数据:")
                        print(f"   - 数据范围: {frame.min()} ~ {frame.max()}")
                        print(f"   - 平均值: {frame.mean():.2f}")
                        print(f"   - 非零值数量: {np.count_nonzero(frame)}")
                        
                        # 显示前10个和后10个值作为示例
                        print(f"   - 前10个值: {frame[:10].tolist()}")
                        print(f"   - 后10个值: {frame[-10:].tolist()}")
                else:
                    no_data_count += 1
            else:
                no_data_count += 1
            
            # 检查是否长时间没有数据
            if current_time - last_data_time > 10:  # 10秒没有有效数据
                print(f"\n⚠️  长时间无有效数据，检查连接状态...")
                last_data_time = current_time
            
            # 短暂休眠避免CPU占用过高
            time.sleep(0.001)
        
        print(f"\n✅ 监控完成，共处理 {frame_count} 帧数据")
    
    def continuous_collect(self, sample_count=100, save_file=None, timeout=30):
        """连续采集指定数量的压力数据样本"""
        print(f"\n🚀 开始连续采集 {sample_count} 个样本...")
        print(f"   - 无延迟高速模式")
        print(f"   - 超时时间: {timeout}秒")
        if save_file:
            print(f"   - 保存到: {save_file}")
        
        collected_samples = []
        sample_num = 0
        start_time = time.time()
        last_data_time = start_time
        no_data_count = 0
        
        while sample_num < sample_count:
            # 检查超时
            current_time = time.time()
            if current_time - start_time > timeout:
                print(f"\n⏰ 采集超时 ({timeout}秒)，已采集 {sample_num} 个样本")
                break
            
            # 连续读取数据
            raw_data = self.read_raw_data(512)
            
            if raw_data:
                # 解析数据帧
                frames = self.parse_hex_data(raw_data)
                if frames:
                    last_data_time = current_time
                    no_data_count = 0
                    for frame in frames:
                        sample_num += 1
                        collected_samples.append(frame)
                        
                        # 实时显示进度
                        if sample_num % 10 == 0 or sample_num <= 5:
                            print(f"   📊 已采集: {sample_num}/{sample_count} 样本")
                            print(f"      范围: {frame.min()} ~ {frame.max()}, 平均: {frame.mean():.1f}")
                        
                        if sample_num >= sample_count:
                            break
                else:
                    no_data_count += 1
            else:
                no_data_count += 1
            
            # 检查是否长时间没有数据
            if current_time - last_data_time > 5:  # 5秒没有有效数据
                print(f"\n⚠️  长时间无有效数据，尝试重新连接...")
                self.disconnect()
                time.sleep(1)
                if not self.connect():
                    print("❌ 重新连接失败")
                    break
                last_data_time = time.time()
            
            # 短暂休眠避免CPU占用过高
            time.sleep(0.001)
        
        # 保存数据
        if save_file and collected_samples:
            try:
                import pandas as pd
                # 转换为DataFrame
                df = pd.DataFrame(collected_samples)
                df.to_csv(save_file, index=False)
                print(f"✅ 已保存 {len(collected_samples)} 个样本到 {save_file}")
            except ImportError:
                # 使用numpy保存
                import numpy as np
                np.savetxt(save_file, collected_samples, delimiter=',', fmt='%d')
                print(f"✅ 已保存 {len(collected_samples)} 个样本到 {save_file}")
        
        return collected_samples

def main():
    """主函数"""
    import sys
    
    print("🚀 压力传感器数据检测器")
    print("=" * 50)
    
    # 创建读取器
    reader = PressureSensorReader()
    
    # 解析命令行参数
    mode = "monitor"  # 默认监控模式
    sample_count = 100
    save_file = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "collect":
            mode = "collect"
            if len(sys.argv) > 2:
                sample_count = int(sys.argv[2])
            if len(sys.argv) > 3:
                save_file = sys.argv[3]
        elif sys.argv[1] == "monitor":
            mode = "monitor"
    
    print(f"🔧 运行模式: {mode}")
    if mode == "collect":
        print(f"   - 采集样本数: {sample_count}")
        if save_file:
            print(f"   - 保存文件: {save_file}")
    
    try:
        # 连接串口
        if not reader.connect():
            return
        
        # 等待连接稳定
        print("⏳ 等待连接稳定...")
        time.sleep(2)
        
        # 清空缓冲区
        reader.serial_conn.flushInput()
        reader.serial_conn.flushOutput()
        
        # 根据模式执行不同操作
        print("\n按 Ctrl+C 停止")
        if mode == "collect":
            reader.continuous_collect(sample_count, save_file)
        else:
            reader.monitor_data(duration=3600)  # 监控1小时
        
    except KeyboardInterrupt:
        print("\n\n⏹️  用户停止")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        reader.disconnect()

if __name__ == "__main__":
    # 使用说明
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("🚀 压力传感器数据检测器")
        print("=" * 50)
        print("使用方法:")
        print("  python serial_sensor_reader.py                    # 监控模式（默认）")
        print("  python serial_sensor_reader.py monitor            # 监控模式")
        print("  python serial_sensor_reader.py collect [数量]      # 连续采集模式")
        print("  python serial_sensor_reader.py collect 50 data.csv # 采集50个样本并保存")
        print("")
        print("示例:")
        print("  python serial_sensor_reader.py collect 100 real_data.csv")
        print("  python serial_sensor_reader.py monitor")
        sys.exit(0)
    
    main()