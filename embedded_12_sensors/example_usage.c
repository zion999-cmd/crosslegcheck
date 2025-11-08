// 12传感器坐姿检测使用示例
#include "posture_classifier_12.h"
#include <stdio.h>

int main() {
    // 示例：12个传感器的数据
    uint16_t sensor_readings[N_SENSORS_12] = {
        // 对应传感器: 48, 80, 112, 176, 87, 103, 88, 89, 104, 105, 91, 107
        250, 335, 191, 346, 667, 660, 1484, 2160, 1676, 2016, 893, 946
    };
    
    // 简单分类
    posture_class_12_t posture = classify_posture_12_sensors(sensor_readings);
    printf("检测到的坐姿: %d\n", posture);
    
    // 带置信度的分类
    prediction_result_12_t result = predict_posture_12_with_confidence(sensor_readings);
    printf("坐姿: %d, 置信度: %.3f\n", result.predicted_class, result.confidence);
    printf("各类别概率: L=%.3f, N=%.3f, R=%.3f\n", 
           result.class_probabilities[0], 
           result.class_probabilities[1], 
           result.class_probabilities[2]);
    
    return 0;
}
