#include "logistic_classifier.h"
#include <string.h>

// Logistic回归模型参数
#define N_FEATURES 24
#define N_CLASSES 3

// 权重矩阵 [features][classes]
static const float weights[N_FEATURES][N_CLASSES] = {
    {0.256576f, 0.219145f, -0.475721f},
    {-1.229315f, 0.891140f, 0.338176f},
    {0.000000f, 0.000000f, 0.000000f},
    {0.802899f, -0.537630f, -0.265269f},
    {-1.374289f, 5.590487f, -4.216197f},
    {0.256576f, 0.219145f, -0.475721f},
    {-0.114205f, -0.183234f, 0.297439f},
    {0.802899f, -0.537630f, -0.265269f},
    {0.000000f, 0.000000f, 0.000000f},
    {-0.231417f, -1.741875f, 1.973291f},
    {-0.231417f, -1.741875f, 1.973291f},
    {2.049310f, -1.056033f, -0.993276f},
    {-1.411648f, -1.762587f, 3.174235f},
    {4.807968f, 0.831264f, -5.639232f},
    {-0.330124f, 0.597015f, -0.266892f},
    {0.330124f, -0.597015f, 0.266892f},
    {0.762513f, 0.283662f, -1.046176f},
    {-0.762513f, -0.283662f, 1.046176f},
    {-3.135723f, 1.524279f, 1.611444f},
    {0.507045f, -0.921773f, 0.414728f},
    {0.109084f, -0.079798f, -0.029286f},
    {-0.098426f, 0.484060f, -0.385634f},
    {-1.031947f, -0.304272f, 1.336219f},
    {-0.280625f, 0.062421f, 0.218204f}
};

// 偏置向量
static const float bias[N_CLASSES] = {1.403346f, -0.118985f, -1.284360f};

// 标准化参数
static const float feature_mean[N_FEATURES] = {
    215.854141f,
    451.670885f,
    0.000000f,
    3654.455773f,
    32.741056f,
    55124.652424f,
    79.023481f,
    3654.455773f,
    8.007089f,
    8.005079f,
    0.337831f,
    0.365134f,
    0.148517f,
    0.148517f,
    0.925970f,
    0.999962f,
    3654.455773f,
    3654.455773f,
    10.364851f,
    1.000000f,
};

static const float feature_scale[N_FEATURES] = {
    22.201696f,
    119.166381f,
    1.000000f,
    1793.586396f,
    37.155926f,
    26922.582031f,
    104.354462f,
    1793.586396f,
    0.001411f,
    0.001451f,
    0.170644f,
    0.177754f,
    0.063772f,
    0.070516f,
    0.436764f,
    0.455071f,
    1793.586396f,
    1793.586396f,
    6.582718f,
    0.000000f,
};

// 提取统计特征
void extract_statistical_features(const uint16_t* pressure_data, float* features) {
    float sum = 0, sum_sq = 0;
    uint16_t min_val = 65535, max_val = 0;
    uint16_t non_zero_count = 0;
    
    // 计算基础统计量
    for (int i = 0; i < 256; i++) {
        uint16_t val = pressure_data[i];
        sum += val;
        sum_sq += val * val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        if (val > 0) non_zero_count++;
    }
    
    float mean = sum / 256.0f;
    float variance = (sum_sq / 256.0f) - (mean * mean);
    float std_dev = sqrtf(variance);
    
    features[0] = mean;                    // 平均值
    features[1] = std_dev;                 // 标准差
    features[2] = min_val;                 // 最小值
    features[3] = max_val;                 // 最大值
    features[4] = mean;                    // 中位数（用均值近似）
    features[5] = sum;                     // 总和
    features[6] = non_zero_count;          // 非零点数量
    features[7] = max_val - min_val;       // 极差
}

// 提取空间特征
void extract_spatial_features(const uint16_t* pressure_data, float* features) {
    float total_pressure = 0;
    float center_x = 0, center_y = 0;
    
    // 计算总压力和重心
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            float val = pressure_data[y * 16 + x];
            total_pressure += val;
            center_x += x * val;
            center_y += y * val;
        }
    }
    
    if (total_pressure > 0) {
        center_x /= total_pressure;
        center_y /= total_pressure;
    } else {
        center_x = center_y = 8.0f;
    }
    
    features[8] = center_x;
    features[9] = center_y;
    
    // 区域压力分布
    float left_pressure = 0, right_pressure = 0;
    float top_pressure = 0, bottom_pressure = 0;
    
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            float val = pressure_data[y * 16 + x];
            if (x < 8) left_pressure += val;
            else right_pressure += val;
            if (y < 8) top_pressure += val;
            else bottom_pressure += val;
        }
    }
    
    if (total_pressure > 0) {
        features[10] = left_pressure / total_pressure;
        features[11] = right_pressure / total_pressure;
        features[12] = top_pressure / total_pressure;
        features[13] = bottom_pressure / total_pressure;
    } else {
        features[10] = features[11] = features[12] = features[13] = 0.25f;
    }
    
    // 对称性特征
    features[14] = left_pressure / (right_pressure + 1e-6f);
    features[15] = top_pressure / (bottom_pressure + 1e-6f);
}

// 提取峰值特征  
void extract_peak_features(const uint16_t* pressure_data, float* features) {
    // 简化的峰值特征提取
    uint16_t max_val = 0;
    float sum = 0;
    
    for (int i = 0; i < 256; i++) {
        if (pressure_data[i] > max_val) max_val = pressure_data[i];
        sum += pressure_data[i];
    }
    
    float mean = sum / 256.0f;
    
    features[16] = max_val;                    // 最大值
    features[17] = max_val;                    // top5平均（简化为最大值）
    features[18] = max_val / (mean + 1e-6f);   // 峰值与均值比
    features[19] = 1.0f;                       // 简化为1
}

// 提取所有特征
void extract_all_features(const uint16_t* pressure_data, float* features) {
    extract_statistical_features(pressure_data, features);
    extract_spatial_features(pressure_data, features + 8);
    extract_peak_features(pressure_data, features + 16);
}

// 标准化特征
void normalize_features(float* features) {
    for (int i = 0; i < N_FEATURES; i++) {
        if (feature_scale[i] > 0) {
            features[i] = (features[i] - feature_mean[i]) / feature_scale[i];
        }
    }
}

// Softmax函数
void softmax(const float* input, float* output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Logistic回归分类
posture_class_t classify_posture_lr(const uint16_t* pressure_data) {
    float features[N_FEATURES];
    float scores[N_CLASSES] = {0};
    
    // 特征提取
    extract_all_features(pressure_data, features);
    
    // 特征标准化
    normalize_features(features);
    
    // 计算得分
    for (int i = 0; i < N_FEATURES; i++) {
        for (int j = 0; j < N_CLASSES; j++) {
            scores[j] += features[i] * weights[i][j];
        }
    }
    
    // 添加偏置
    for (int j = 0; j < N_CLASSES; j++) {
        scores[j] += bias[j];
    }
    
    // 找到最高得分的类别
    int max_class = 0;
    for (int i = 1; i < N_CLASSES; i++) {
        if (scores[i] > scores[max_class]) {
            max_class = i;
        }
    }
    
    return (posture_class_t)max_class;
}

// 获取预测置信度
float get_prediction_confidence(const float* probabilities) {
    float max_prob = probabilities[0];
    float second_max = 0;
    
    for (int i = 1; i < N_CLASSES; i++) {
        if (probabilities[i] > max_prob) {
            second_max = max_prob;
            max_prob = probabilities[i];
        } else if (probabilities[i] > second_max) {
            second_max = probabilities[i];
        }
    }
    
    // 置信度 = 最大概率 - 第二大概率
    return max_prob - second_max;
}

// 带置信度的预测
prediction_result_t predict_posture_with_confidence(const uint16_t* pressure_data) {
    prediction_result_t result;
    float features[N_FEATURES];
    float scores[N_CLASSES] = {0};
    
    // 特征提取
    extract_all_features(pressure_data, features);
    
    // 特征标准化
    normalize_features(features);
    
    // 计算得分
    for (int i = 0; i < N_FEATURES; i++) {
        for (int j = 0; j < N_CLASSES; j++) {
            scores[j] += features[i] * weights[i][j];
        }
    }
    
    // 添加偏置
    for (int j = 0; j < N_CLASSES; j++) {
        scores[j] += bias[j];
    }
    
    // 计算概率
    softmax(scores, result.class_probabilities, N_CLASSES);
    
    // 找到最高概率的类别
    int max_class = 0;
    for (int i = 1; i < N_CLASSES; i++) {
        if (result.class_probabilities[i] > result.class_probabilities[max_class]) {
            max_class = i;
        }
    }
    
    result.predicted_class = (posture_class_t)max_class;
    result.confidence = get_prediction_confidence(result.class_probabilities);
    
    return result;
}