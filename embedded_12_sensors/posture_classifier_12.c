#include "posture_classifier_12.h"

// Logistic回归权重矩阵 [12传感器][3类别]
static const float weights_12[N_FEATURES_12][N_CLASSES_12] = {
    {-0.422222f, 0.428230f, -0.006008f},
    {-3.856235f, 1.736903f, 2.119331f},
    {-0.830034f, -1.414338f, 2.244372f},
    {4.094326f, 1.049823f, -5.144149f},
    {-0.230269f, 0.213609f, 0.016660f},
    {-0.100834f, -0.131286f, 0.232120f},
    {0.102387f, 0.041844f, -0.144231f},
    {-0.684050f, -0.284927f, 0.968978f},
    {0.278670f, 0.057863f, -0.336533f},
    {0.353034f, 0.410134f, -0.763169f},
    {1.077807f, -1.527204f, 0.449397f},
    {-0.498776f, 1.083779f, -0.585003f},

};

// 偏置向量
static const float bias_12[N_CLASSES_12] = {
    -1.021672f, 1.837654f, -0.815982f
};

// 标准化参数 - 均值
static const float feature_mean_12[N_FEATURES_12] = {
    254.590733f, 334.579621f, 191.352553f, 345.818954f, 667.449942f, 659.792955f, 1483.777021f, 2159.989831f, 1676.309781f, 2016.256840f, 892.858476f, 945.652899f
};

// 标准化参数 - 标准差
static const float feature_scale_12[N_FEATURES_12] = {
    181.262934f, 182.707407f, 139.695453f, 165.614927f, 380.874728f, 339.717960f, 1037.704777f, 1324.293933f, 1464.228150f, 1247.764526f, 755.112288f, 671.698616f
};

// 标准化12传感器数据
void normalize_features_12(const uint16_t* sensor_data, float* normalized_features) {
    for (int i = 0; i < N_FEATURES_12; i++) {
        if (feature_scale_12[i] > 0) {
            normalized_features[i] = ((float)sensor_data[i] - feature_mean_12[i]) / feature_scale_12[i];
        } else {
            normalized_features[i] = 0.0f;
        }
    }
}

// Softmax函数
void softmax_12(const float* input, float* output, int size) {
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

// 12传感器坐姿分类
posture_class_12_t classify_posture_12_sensors(const uint16_t* sensor_data_12) {
    float normalized_features[N_FEATURES_12];
    float scores[N_CLASSES_12] = {0};
    
    // 标准化输入数据
    normalize_features_12(sensor_data_12, normalized_features);
    
    // 计算线性组合
    for (int i = 0; i < N_FEATURES_12; i++) {
        for (int j = 0; j < N_CLASSES_12; j++) {
            scores[j] += normalized_features[i] * weights_12[i][j];
        }
    }
    
    // 添加偏置
    for (int j = 0; j < N_CLASSES_12; j++) {
        scores[j] += bias_12[j];
    }
    
    // 找到最高得分的类别
    int max_class = 0;
    for (int i = 1; i < N_CLASSES_12; i++) {
        if (scores[i] > scores[max_class]) {
            max_class = i;
        }
    }
    
    return (posture_class_12_t)max_class;
}

// 带置信度的预测
prediction_result_12_t predict_posture_12_with_confidence(const uint16_t* sensor_data_12) {
    prediction_result_12_t result;
    float normalized_features[N_FEATURES_12];
    float scores[N_CLASSES_12] = {0};
    
    // 标准化输入数据
    normalize_features_12(sensor_data_12, normalized_features);
    
    // 计算线性组合
    for (int i = 0; i < N_FEATURES_12; i++) {
        for (int j = 0; j < N_CLASSES_12; j++) {
            scores[j] += normalized_features[i] * weights_12[i][j];
        }
    }
    
    // 添加偏置
    for (int j = 0; j < N_CLASSES_12; j++) {
        scores[j] += bias_12[j];
    }
    
    // 计算概率
    softmax_12(scores, result.class_probabilities, N_CLASSES_12);
    
    // 找到最高概率的类别
    int max_class = 0;
    for (int i = 1; i < N_CLASSES_12; i++) {
        if (result.class_probabilities[i] > result.class_probabilities[max_class]) {
            max_class = i;
        }
    }
    
    result.predicted_class = (posture_class_12_t)max_class;
    
    // 计算置信度（最大概率与第二大概率的差值）
    float max_prob = result.class_probabilities[max_class];
    float second_max = 0;
    for (int i = 0; i < N_CLASSES_12; i++) {
        if (i != max_class && result.class_probabilities[i] > second_max) {
            second_max = result.class_probabilities[i];
        }
    }
    result.confidence = max_prob - second_max;
    
    return result;
}
