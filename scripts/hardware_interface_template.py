
# 硬件接口模板 - 支持256维模型

from sensor_data_converter import HardwareAdapterFor256Model
import time

class HardwareSensorInterface:
    def __init__(self, model_path):
        # 创建适配器，自动检测模型输入维度
        self.adapter = HardwareAdapterFor256Model(model_path)
    
    def read_12_sensors_from_hardware(self):
        """
        从实际硬件读取12路传感器数据
        返回: [sensor1, sensor2, ..., sensor12] (单位: 克)
        """
        # 这里替换为实际的硬件读取代码
        # 例如通过串口、I2C、SPI等接口读取12路传感器
        
        # 示例：串口读取
        # serial_data = self.serial_port.readline()
        # sensor_values = parse_sensor_data(serial_data)
        
        # 示例：模拟数据
        import random
        return [random.randint(0, 500) for _ in range(12)]
    
    def run_detection_loop(self):
        """运行检测循环"""
        while True:
            try:
                # 读取12路传感器数据
                sensor_data_12 = self.read_12_sensors_from_hardware()
                
                # 使用适配器进行预测（自动处理数据格式转换）
                prediction, probabilities = self.adapter.predict_from_hardware(sensor_data_12)
                
                if prediction is not None:
                    if probabilities is not None:
                        confidence = max(probabilities)
                        print(f"坐姿: {prediction}, 置信度: {confidence:.1%}")
                    else:
                        print(f"坐姿: {prediction}")
                
                time.sleep(0.5)  # 每0.5秒检测一次
                
            except KeyboardInterrupt:
                print("检测已停止")
                break
            except Exception as e:
                print(f"检测错误: {e}")
                time.sleep(1)

# 使用示例
if __name__ == "__main__":
    # 指定模型路径
    model_path = "/path/to/your/model.pkl"
    
    # 创建硬件接口
    hardware = HardwareSensorInterface(model_path)
    
    # 运行检测
    hardware.run_detection_loop()
