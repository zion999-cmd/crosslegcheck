#ifndef POSTURE_CLASSIFIER_12_H
#define POSTURE_CLASSIFIER_12_H

#include <stdint.h>
#include <math.h>

// 12传感器配置
#define N_SENSORS_12 12
#define N_FEATURES_12 12
#define N_CLASSES_12 3

// 传感器映射（对应原256传感器的索引）
static const uint16_t sensor_mapping[N_SENSORS_12] = {
    48, 80, 112, 176, 87, 103, 88, 89, 104, 105, 91, 107
};

// 类别定义
typedef enum {
    CLASS_LEFT_12 = 0,
    CLASS_NORMAL_12 = 1,
    CLASS_RIGHT_12 = 2
} posture_class_12_t;

// 预测结果结构
typedef struct {
    posture_class_12_t predicted_class;
    float confidence;
    float class_probabilities[N_CLASSES_12];
} prediction_result_12_t;

// 函数声明
posture_class_12_t classify_posture_12_sensors(const uint16_t* sensor_data_12);
prediction_result_12_t predict_posture_12_with_confidence(const uint16_t* sensor_data_12);
void normalize_features_12(const uint16_t* sensor_data, float* normalized_features);
void softmax_12(const float* input, float* output, int size);

#endif // POSTURE_CLASSIFIER_12_H
