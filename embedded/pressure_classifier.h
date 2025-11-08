
#ifndef PRESSURE_CLASSIFIER_H
#define PRESSURE_CLASSIFIER_H

#include <stdint.h>

// 特征数量
#define N_FEATURES 24

// 类别数量
#define N_CLASSES 3

// 节点数量
#define N_NODES 115

// 类别标签
typedef enum {
    CLASS_LEFT = 0,
    CLASS_NORMAL = 1,
    CLASS_RIGHT = 2
} class_t;

// 决策树节点结构
typedef struct {
    int16_t feature;        // 特征索引 (-1表示叶子节点)
    float threshold;        // 阈值
    int16_t left_child;     // 左子节点索引
    int16_t right_child;    // 右子节点索引
    int16_t class_id;       // 类别ID（仅叶子节点有效）
} tree_node_t;

// 函数声明
float extract_statistical_features(const uint16_t* pressure_data, float* features);
float extract_spatial_features(const uint16_t* pressure_data, float* features);
float extract_peak_features(const uint16_t* pressure_data, float* features);
void extract_all_features(const uint16_t* pressure_data, float* features);
class_t classify_pressure_data(const uint16_t* pressure_data);
int predict_with_confidence(const uint16_t* pressure_data, float* confidence);

#endif // PRESSURE_CLASSIFIER_H
