
#include "pressure_classifier.h"
#include <math.h>
#include <string.h>

// 决策树节点数据
static const tree_node_t tree_nodes[N_NODES] = {
    {13, 8.000772f, 1, 86, -1}, // Node 0
    {4, 33.250000f, 2, 59, -1}, // Node 1
    {13, 7.603573f, 3, 22, -1}, // Node 2
    {4, 19.250000f, 4, 13, -1}, // Node 3
    {23, 1.049816f, 5, 6, -1}, // Node 4
    {-1, 0.0f, -1, -1, 1}, // Node 5 (leaf)
    {18, 2.086131f, 7, 12, -1}, // Node 6
    {23, 1.066898f, 8, 9, -1}, // Node 7
    {-1, 0.0f, -1, -1, 2}, // Node 8 (leaf)
    {13, 7.332822f, 10, 11, -1}, // Node 9
    {-1, 0.0f, -1, -1, 2}, // Node 10 (leaf)
    {-1, 0.0f, -1, -1, 2}, // Node 11 (leaf)
    {-1, 0.0f, -1, -1, 2}, // Node 12 (leaf)
    {17, 0.445314f, 14, 19, -1}, // Node 13
    {9, 242.625000f, 15, 16, -1}, // Node 14
    {-1, 0.0f, -1, -1, 2}, // Node 15 (leaf)
    {10, 280.125000f, 17, 18, -1}, // Node 16
    {-1, 0.0f, -1, -1, 2}, // Node 17 (leaf)
    {-1, 0.0f, -1, -1, 2}, // Node 18 (leaf)
    {12, 8.227130f, 20, 21, -1}, // Node 19
    {-1, 0.0f, -1, -1, 0}, // Node 20 (leaf)
    {-1, 0.0f, -1, -1, 2}, // Node 21 (leaf)
    {23, 1.217021f, 23, 34, -1}, // Node 22
    {11, 0.230469f, 24, 27, -1}, // Node 23
    {18, 0.189964f, 25, 26, -1}, // Node 24
    {-1, 0.0f, -1, -1, 1}, // Node 25 (leaf)
    {-1, 0.0f, -1, -1, 1}, // Node 26 (leaf)
    {14, 0.337635f, 28, 31, -1}, // Node 27
    {19, 1.120274f, 29, 30, -1}, // Node 28
    {-1, 0.0f, -1, -1, 0}, // Node 29 (leaf)
    {-1, 0.0f, -1, -1, 0}, // Node 30 (leaf)
    {15, 0.193055f, 32, 33, -1}, // Node 31
    {-1, 0.0f, -1, -1, 1}, // Node 32 (leaf)
    {-1, 0.0f, -1, -1, 1}, // Node 33 (leaf)
    {3, 3893.000000f, 35, 44, -1}, // Node 34
    {16, 0.431266f, 36, 37, -1}, // Node 35
    {-1, 0.0f, -1, -1, 2}, // Node 36 (leaf)
    {23, 1.404924f, 38, 41, -1}, // Node 37
    {18, 1.131987f, 39, 40, -1}, // Node 38
    {-1, 0.0f, -1, -1, 0}, // Node 39 (leaf)
    {-1, 0.0f, -1, -1, 1}, // Node 40 (leaf)
    {13, 7.802117f, 42, 43, -1}, // Node 41
    {-1, 0.0f, -1, -1, 2}, // Node 42 (leaf)
    {-1, 0.0f, -1, -1, 0}, // Node 43 (leaf)
    {12, 7.754255f, 45, 52, -1}, // Node 44
    {22, 14.956011f, 46, 49, -1}, // Node 45
    {18, 0.549589f, 47, 48, -1}, // Node 46
    {-1, 0.0f, -1, -1, 0}, // Node 47 (leaf)
    {-1, 0.0f, -1, -1, 0}, // Node 48 (leaf)
    {6, 133.500000f, 50, 51, -1}, // Node 49
    {-1, 0.0f, -1, -1, 2}, // Node 50 (leaf)
    {-1, 0.0f, -1, -1, 1}, // Node 51 (leaf)
    {16, 0.519807f, 53, 56, -1}, // Node 52
    {13, 7.948599f, 54, 55, -1}, // Node 53
    {-1, 0.0f, -1, -1, 2}, // Node 54 (leaf)
    {-1, 0.0f, -1, -1, 0}, // Node 55 (leaf)
    {10, 258.375000f, 57, 58, -1}, // Node 56
    {-1, 0.0f, -1, -1, 0}, // Node 57 (leaf)
    {-1, 0.0f, -1, -1, 2}, // Node 58 (leaf)
    {10, 412.250000f, 60, 83, -1}, // Node 59
    {4, 46.250000f, 61, 74, -1}, // Node 60
    {0, 241.509766f, 62, 71, -1}, // Node 61
    {13, 7.521033f, 63, 68, -1}, // Node 62
    {9, 315.875000f, 64, 67, -1}, // Node 63
    {20, 3115.799927f, 65, 66, -1}, // Node 64
    {-1, 0.0f, -1, -1, 1}, // Node 65 (leaf)
    {-1, 0.0f, -1, -1, 1}, // Node 66 (leaf)
    {-1, 0.0f, -1, -1, 2}, // Node 67 (leaf)
    {0, 235.074219f, 69, 70, -1}, // Node 68
    {-1, 0.0f, -1, -1, 1}, // Node 69 (leaf)
    {-1, 0.0f, -1, -1, 1}, // Node 70 (leaf)
    {12, 8.394714f, 72, 73, -1}, // Node 71
    {-1, 0.0f, -1, -1, 1}, // Node 72 (leaf)
    {-1, 0.0f, -1, -1, 2}, // Node 73 (leaf)
    {23, 1.050578f, 75, 78, -1}, // Node 74
    {16, 0.532140f, 76, 77, -1}, // Node 75
    {-1, 0.0f, -1, -1, 1}, // Node 76 (leaf)
    {-1, 0.0f, -1, -1, 0}, // Node 77 (leaf)
    {12, 8.757280f, 79, 80, -1}, // Node 78
    {-1, 0.0f, -1, -1, 1}, // Node 79 (leaf)
    {3, 3693.000000f, 81, 82, -1}, // Node 80
    {-1, 0.0f, -1, -1, 1}, // Node 81 (leaf)
    {-1, 0.0f, -1, -1, 2}, // Node 82 (leaf)
    {20, 1687.900024f, 84, 85, -1}, // Node 83
    {-1, 0.0f, -1, -1, 1}, // Node 84 (leaf)
    {-1, 0.0f, -1, -1, 2}, // Node 85 (leaf)
    {12, 6.013616f, 87, 90, -1}, // Node 86
    {13, 8.550495f, 88, 89, -1}, // Node 87
    {-1, 0.0f, -1, -1, 1}, // Node 88 (leaf)
    {-1, 0.0f, -1, -1, 0}, // Node 89 (leaf)
    {4, 65.250000f, 91, 114, -1}, // Node 90
    {14, 0.197714f, 92, 101, -1}, // Node 91
    {13, 8.177919f, 93, 96, -1}, // Node 92
    {4, 1.750000f, 94, 95, -1}, // Node 93
    {-1, 0.0f, -1, -1, 1}, // Node 94 (leaf)
    {-1, 0.0f, -1, -1, 2}, // Node 95 (leaf)
    {16, 0.357577f, 97, 98, -1}, // Node 96
    {-1, 0.0f, -1, -1, 2}, // Node 97 (leaf)
    {0, 192.035156f, 99, 100, -1}, // Node 98
    {-1, 0.0f, -1, -1, 0}, // Node 99 (leaf)
    {-1, 0.0f, -1, -1, 0}, // Node 100 (leaf)
    {23, 1.078386f, 102, 105, -1}, // Node 101
    {23, 1.070413f, 103, 104, -1}, // Node 102
    {-1, 0.0f, -1, -1, 0}, // Node 103 (leaf)
    {-1, 0.0f, -1, -1, 2}, // Node 104 (leaf)
    {9, 169.375000f, 106, 107, -1}, // Node 105
    {-1, 0.0f, -1, -1, 0}, // Node 106 (leaf)
    {19, 0.892126f, 108, 111, -1}, // Node 107
    {13, 8.103023f, 109, 110, -1}, // Node 108
    {-1, 0.0f, -1, -1, 0}, // Node 109 (leaf)
    {-1, 0.0f, -1, -1, 0}, // Node 110 (leaf)
    {14, 0.434004f, 112, 113, -1}, // Node 111
    {-1, 0.0f, -1, -1, 0}, // Node 112 (leaf)
    {-1, 0.0f, -1, -1, 1}, // Node 113 (leaf)
    {-1, 0.0f, -1, -1, 1}, // Node 114 (leaf)

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
    features[4] = sum;                     // 总和
    features[5] = non_zero_count;          // 非零点数量
    features[6] = max_val - min_val;       // 极差
    
    // 计算中位数（简化版本）
    features[7] = mean;  // 用均值近似中位数
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
    
    features[0] = center_x;
    features[1] = center_y;
    
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
        features[2] = left_pressure / total_pressure;
        features[3] = right_pressure / total_pressure;
        features[4] = top_pressure / total_pressure;
        features[5] = bottom_pressure / total_pressure;
    } else {
        features[2] = features[3] = features[4] = features[5] = 0.25f;
    }
    
    // 对称性特征
    features[6] = left_pressure / (right_pressure + 1e-6f);
    features[7] = top_pressure / (bottom_pressure + 1e-6f);
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
    
    features[0] = max_val;                    // 最大值
    features[1] = max_val;                    // top5平均（简化为最大值）
    features[2] = max_val / (mean + 1e-6f);   // 峰值与均值比
    features[3] = 1.0f;                       // 简化为1
}

// 提取所有特征
void extract_all_features(const uint16_t* pressure_data, float* features) {
    extract_statistical_features(pressure_data, features);
    extract_spatial_features(pressure_data, features + 8);
    extract_peak_features(pressure_data, features + 16);
}

// 分类函数
class_t classify_pressure_data(const uint16_t* pressure_data) {
    float features[20];  // 总共20个特征
    extract_all_features(pressure_data, features);
    
    // 遍历决策树
    int node_id = 0;  // 从根节点开始
    
    while (tree_nodes[node_id].feature != -1) {  // 不是叶子节点
        int feature_idx = tree_nodes[node_id].feature;
        float threshold = tree_nodes[node_id].threshold;
        
        if (features[feature_idx] <= threshold) {
            node_id = tree_nodes[node_id].left_child;
        } else {
            node_id = tree_nodes[node_id].right_child;
        }
    }
    
    return (class_t)tree_nodes[node_id].class_id;
}

// 带置信度的预测
int predict_with_confidence(const uint16_t* pressure_data, float* confidence) {
    class_t result = classify_pressure_data(pressure_data);
    
    // 简化的置信度计算（基于数据质量）
    float sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += pressure_data[i];
    }
    
    if (sum > 10000) {
        *confidence = 0.9f;  // 高质量数据
    } else if (sum > 1000) {
        *confidence = 0.7f;  // 中等质量数据
    } else {
        *confidence = 0.5f;  // 低质量数据
    }
    
    return result;
}
