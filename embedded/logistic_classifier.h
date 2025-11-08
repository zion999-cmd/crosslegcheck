#ifndef LOGISTIC_CLASSIFIER_H
#define LOGISTIC_CLASSIFIER_H

#include <stdint.h>
#include <math.h>

// 特征数量和类别数量
#define N_FEATURES 24
#define N_CLASSES 3

// 类别标签
typedef enum {
    CLASS_LEFT = 0,
    CLASS_NORMAL = 1,
    CLASS_RIGHT = 2
} posture_class_t;

// 预测结果结构
typedef struct {
    posture_class_t predicted_class;
    float confidence;
    float class_probabilities[N_CLASSES];
} prediction_result_t;

// 函数声明
void extract_statistical_features(const uint16_t* pressure_data, float* features);
void extract_spatial_features(const uint16_t* pressure_data, float* features);
void extract_peak_features(const uint16_t* pressure_data, float* features);
void extract_all_features(const uint16_t* pressure_data, float* features);
void normalize_features(float* features);
void softmax(const float* input, float* output, int size);
posture_class_t classify_posture_lr(const uint16_t* pressure_data);
prediction_result_t predict_posture_with_confidence(const uint16_t* pressure_data);
float get_prediction_confidence(const float* probabilities);

#endif // LOGISTIC_CLASSIFIER_H