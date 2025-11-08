# ========== 选取各类别压力均值最大点并可视化 ========== 
def get_top_mean_points_by_class(groups, top_n=50):
    """
    分别选出每个类别压力均值最大的top_n个点，返回所有点的索引集合和坐标列表
    """
    class_points = {}
    for label in ['left', 'right', 'normal']:
        mean_arr = np.mean(groups[label], axis=0)
        top_idx = np.argsort(mean_arr)[::-1][:top_n]
        class_points[label] = set(top_idx)
    # 合并去重
    all_points = set.union(*class_points.values())
    # 坐标列表（物理布局：row = idx // 16, col = idx % 16）
    coords = [(idx // 16, idx % 16) for idx in all_points]
    print(f"\n【各类别压力均值最大前{top_n}点合并后共 {len(all_points)} 个点")
    print(f"点位索引: {sorted(list(all_points))}")
    print(f"点位坐标(row, col): {sorted(coords)}")
    return all_points

def plot_global_points(points, title='合并重要点位热力图'):
    """
    在全局热力图上标注指定点位
    """
    # 取所有样本均值作为底图
    global_mean = np.mean(sensor_data, axis=0)
    plot_matrix.important_points = list(points)
    plot_matrix(global_mean, title=title)
# 项目方法：基于256个传感器点的压力数据，计算每个类别（normal/left/right）的均值和标准差，
# 生成压力阈值表（均值±标准差），并用简单的
# 阈值表是基于训练集均值±标准差，未考虑特征间相关性，对复杂分布或边界样本不敏感。
# 传感器点位有冗余或噪声，部分点对分类贡献小甚至有干扰。
# 类别分布不均或数据本身有偏移，导致阈值不适用于新样本。
# 提升方法（不使用深度学习）：
# 特征选择/降维
# 只用“重要点位”或高区分度的传感器（计算分析的高频点），去掉无用点。
# 可以用方差、互信息、相关性等方法筛选。
# 更智能的阈值策略
# 不用均值±标准差，可以用分位数（如25%~75%区间）或自适应分割。
# 可以针对每个类别分别设定更严格或更宽松的阈值。
# 后期集成多种规则
# 例如：总压力、重要点位压力、非零点数量等多规则组合。


# 基础库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from matplotlib.colors import ListedColormap
import time

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
    

# 读取数据集（跳过表头）
df = pd.read_csv('../data/dataset.csv', header=None, skiprows=1)
labels = df.iloc[:, 0].astype(str).values  # shape: (样本数,)
sensor_data = df.iloc[:, 1:257].astype(float).values  # shape: (样本数, 256)

# 按类别分组
groups = {}
for label in np.unique(labels):
    groups[label] = sensor_data[labels == label]

# 计算每个类别的均值和标准差
stats = {}
for label, data in groups.items():
    stats[label] = {
        'mean': np.mean(data, axis=0),
        'std': np.std(data, axis=0)
    }


# 生成阈值表（均值±标准差）
threshold_table = {}
for i in range(256):
    threshold_table[i] = {
        'normal': (stats['normal']['mean'][i] - stats['normal']['std'][i],
                   stats['normal']['mean'][i] + stats['normal']['std'][i]),
        'left': (stats['left']['mean'][i] - stats['left']['std'][i],
                 stats['left']['mean'][i] + stats['left']['std'][i]),
        'right': (stats['right']['mean'][i] - stats['right']['std'][i],
                  stats['right']['mean'][i] + stats['right']['std'][i])
    }


# ================== 区域选点法与区域均值分类 ==================
def select_key_points_by_region():
    """
    按区域选出12个关键点，覆盖左腿、右腿、屁股
    返回点位索引列表
    """
    # 假设16x16布局，左腿区域：第4~7行, 2~6列；右腿区域：第4~7行, 10~14列；屁股区域：第10~13行, 5~11列
    left_leg = [(r, c) for r in range(4, 8) for c in range(2, 6)]
    right_leg = [(r, c) for r in range(4, 8) for c in range(10, 14)]
    butt = [(r, c) for r in range(10, 14) for c in range(5, 11)]
    # 各区域各选4个点（可随机/均匀选取）
    random.seed(42)
    left_leg_idx = random.sample(left_leg, 4)
    right_leg_idx = random.sample(right_leg, 4)
    butt_idx = random.sample(butt, 4)
    # 转为一维索引
    def rc_to_idx(r, c): return r * 16 + c
    selected_idx = [rc_to_idx(r, c) for (r, c) in left_leg_idx + right_leg_idx + butt_idx]
    return selected_idx

def classify_by_region_pressure(pressure_row, selected_idx, threshold=0.1):
    """
    按区域均值分类
    """
    # 还原二维
    image = pressure_row.reshape(16, 16)
    # 区域索引
    left_leg = selected_idx[:4]
    right_leg = selected_idx[4:8]
    butt = selected_idx[8:]
    # 计算均值
    left_mean = np.mean([pressure_row[i] for i in left_leg])
    right_mean = np.mean([pressure_row[i] for i in right_leg])
    butt_mean = np.mean([pressure_row[i] for i in butt])
    # 分类规则
    if left_mean > right_mean + threshold:
        return 'right'
    elif right_mean > left_mean + threshold:
        return 'left'
    elif left_mean > threshold and right_mean > threshold and butt_mean > threshold:
        return 'normal'
    else:
        return 'unknown'

# ================== 原始阈值表分类（全点/部分点） ==================
def classify(new_data, threshold_table):
    scores = {'normal': 0, 'left': 0, 'right': 0}
    for i, value in enumerate(new_data):
        for label in scores:
            low, high = threshold_table[i][label]
            if low <= value <= high:
                scores[label] += 1
    return max(scores, key=scores.get)

# new_data = [传感器1, 传感器2, ..., 传感器256]
# result = classify(new_data, threshold_table)
# print('预测类别:', result)

# ================== 重要点位分析 ==================

# ================== 重要点位分析（原有方法，保留对比） ==================
def analyze_important_points(sensor_data, labels):
    """
    极差法+代表性法自动选点（原有方法，便于对比）
    """
    total_samples = sensor_data.shape[0]
    print('\n【全体样本】每个点位压力均值:')
    mean_values = np.mean(sensor_data, axis=0)
    top_indices = np.argsort(mean_values)[::-1][:20]
    print('\n【全体均值最高的前20个点位下标】:')
    print(top_indices.tolist())

    # 计算每个点在3类中的均值
    class_means = []
    for label in np.unique(labels):
        class_data = sensor_data[labels == label]
        class_mean = np.mean(class_data, axis=0)
        class_means.append(class_mean)
    class_means = np.array(class_means)  # shape: (3, 256)
    # 对每个点，计算3类均值的极差
    diff = np.max(class_means, axis=0) - np.min(class_means, axis=0)
    top_diff_indices = np.argsort(diff)[::-1][:12]
    print(f'\n【区分力最强的12个点位下标（类别均值极差最大）】: {top_diff_indices.tolist()}')
    for idx in top_diff_indices:
        print(f'点位{idx}: normal={class_means[0,idx]:.2f}, left={class_means[1,idx]:.2f}, right={class_means[2,idx]:.2f}, 极差={diff[idx]:.2f}')

    # 每个类别选均值最大的4个点，合并去重
    rep_points = set()
    for i, class_mean in enumerate(class_means):
        rep_points.update(np.argsort(class_mean)[::-1][:4])
    # 合并极差法和代表性法，得到最终12个点
    final_points = list(set(top_diff_indices.tolist()) | rep_points)
    final_points = sorted(final_points)[:12]
    print(f'\n【最终选出的12个点位下标（极差法+代表性法）】: {final_points}')
    return final_points

    # 输出每类10个重要点
    for i, cp in enumerate(class_points):
        print(f'类别{i}的10个重要点位: {sorted(list(cp))}')
    # 自动扩展点数直到能覆盖所有类别的10个重要点
    all_points = set.union(*class_points)
    import itertools
    min_cover = 12
    found = False
    while min_cover <= len(all_points):
        for comb in itertools.combinations(all_points, min_cover):
            ok = True
            for cp in class_points:
                if not cp.issubset(set(comb)):
                    ok = False
                    break
            if ok:
                final_points = sorted(list(comb))
                found = True
                break
        if found:
            break
        min_cover += 1
    if not found:
        print(f'警告：无法用{min_cover-1}个点完全覆盖所有类别的10个重要点，已选最大覆盖！')
        # 贪心补齐法
        final_points = []
        covered_points = set()
        from collections import Counter
        freq = Counter()
        for cp in class_points:
            freq.update(cp)
        for pt, _ in freq.most_common():
            if pt not in final_points:
                final_points.append(pt)
            covered_points = set(final_points)
            if len(final_points) >= min_cover-1:
                break
        final_points = sorted(final_points[:min_cover-1])
    print(f'\n【最小覆盖的{len(final_points)}个点位下标】: {final_points}')
    # 显示每个类别的覆盖情况和未覆盖点
    for i, cp in enumerate(class_points):
        covered = set(final_points) & cp
        missed = cp - set(final_points)
        print(f'类别{i}的10个重要点位被覆盖了 {len(covered)}/10 个，未覆盖: {sorted(list(missed))}')
    return final_points


# ================== 16x16热力图 ==================
def plot_matrix(data, title='压力热力图'):
    """
    绘制16x16压力热力图，并标注重要点位
    """
    matrix = np.zeros((16, 16))
    # 物理布局：第1行下标为 1,17,33,...，即 matrix[row, col] = data[row + col * 16]
    for row in range(16):
        for col in range(16):
            idx = row + col * 16
            matrix[row, col] = data[idx]

    hot_cmap = plt.get_cmap('jet')
    colors = hot_cmap(np.linspace(0, 1, 256))
    colors[0] = [0, 0, 0.5, 1]
    custom_cmap = ListedColormap(colors)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=custom_cmap, interpolation='nearest', aspect=3/4)
    plt.title(title)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    # 标注重要点位（白色菱形），与物理布局一致
    if hasattr(plot_matrix, 'important_points') and plot_matrix.important_points:
        for idx in plot_matrix.important_points:
            row = idx % 16
            col = idx // 16
            plt.scatter(col, row, marker='D', s=120, c='white', edgecolors='black', linewidths=1.5, zorder=10)
    plt.show()

    hot_cmap = plt.get_cmap('jet')
    colors = hot_cmap(np.linspace(0, 1, 256))
    colors[0] = [0, 0, 0.5, 1]
    custom_cmap = ListedColormap(colors)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=custom_cmap, interpolation='nearest', aspect=3/4)
    plt.title(title)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    # 标注重要点位（白色菱形）
    if hasattr(plot_matrix, 'important_points') and plot_matrix.important_points:
        for idx in plot_matrix.important_points:
            row = idx // 16
            col = idx % 16
            plt.scatter(col, row, marker='D', s=120, c='white', edgecolors='black', linewidths=1.5, zorder=10)
    plt.show()


def show_random_heatmap(mode, important_points=None):
    """
    随机展示某类别样本的热力图，并标注重要点位
    """
    mode_map = {'n': 'normal', 'l': 'left', 'r': 'right'}
    if mode not in mode_map:
        print('参数错误，只能输入 n/l/r')
        return
    label = mode_map[mode]
    idxs = np.where(labels == label)[0]
    if len(idxs) == 0:
        print(f'没有找到类别 {label} 的数据')
        return
    rand_idx = random.choice(idxs)
    data = sensor_data[rand_idx]
    # 标注重要点位
    plot_matrix.important_points = important_points if important_points else []
    plot_matrix(data, title=f'{label} 随机样本热力图')

# 示例：输入 n/l/r 生成热力图
# show_random_heatmap('n')

# ================== 预测评估函数 ==================

def evaluate_classifier_with_features(sensor_data, labels, threshold_table, feature_indices, sample_size=100):
    """
    只用指定点位进行分类评估（原有阈值法）
    feature_indices: 选用的点位下标列表
    """
    np.random.seed(int(time.time()))
    total = min(sample_size, len(labels))
    indices = np.random.choice(len(labels), total, replace=False)
    correct = 0
    wrong = 0
    for idx in indices:
        true_label = labels[idx]
        sample = sensor_data[idx][feature_indices]
        # 构造只用特征点的阈值表
        sub_threshold_table = {i: threshold_table[i] for i in feature_indices}
        # 分类函数
        scores = {'normal': 0, 'left': 0, 'right': 0}
        for j, i in enumerate(feature_indices):
            value = sample[j]
            for label in scores:
                low, high = sub_threshold_table[i][label]
                if low <= value <= high:
                    scores[label] += 1
        pred_label = max(scores, key=scores.get)
        if pred_label == true_label:
            correct += 1
        else:
            wrong += 1
    print(f'【仅用指定点位】预测总数: {total}')
    print(f'正确数: {correct}, 正确率: {correct/total:.2%}')
    print(f'错误数: {wrong}, 错误率: {wrong/total:.2%}')

def evaluate_classifier_by_region(sensor_data, labels, selected_idx, sample_size=100, threshold=0.1):
    """
    只用区域选点法进行分类评估（认知友好）
    """
    np.random.seed(int(time.time()))
    total = min(sample_size, len(labels))
    indices = np.random.choice(len(labels), total, replace=False)
    correct = 0
    wrong = 0
    unknown = 0
    for idx in indices:
        true_label = labels[idx]
        pred_label = classify_by_region_pressure(sensor_data[idx], selected_idx, threshold)
        if pred_label == true_label:
            correct += 1
        elif pred_label == 'unknown':
            unknown += 1
        else:
            wrong += 1
    print(f'【区域均值法】预测总数: {total}')
    print(f'正确数: {correct}, 正确率: {correct/total:.2%}')
    print(f'错误数: {wrong}, 错误率: {wrong/total:.2%}')
    print(f'无法判定: {unknown}, 占比: {unknown/total:.2%}')
    
def evaluate_classifier(sensor_data, labels, threshold_table, sample_size=100):
    np.random.seed(int(time.time()))
    total = min(sample_size, len(labels))
    indices = np.random.choice(len(labels), total, replace=False)
    correct = 0
    wrong = 0
    for idx in indices:
        true_label = labels[idx]
        pred_label = classify(sensor_data[idx], threshold_table)
        if pred_label == true_label:
            correct += 1
        else:
            wrong += 1
    print(f'预测总数: {total}')
    print(f'正确数: {correct}, 正确率: {correct/total:.2%}')
    print(f'错误数: {wrong}, 错误率: {wrong/total:.2%}')

# 运行评估

if __name__ == '__main__':
    # 选取各类别压力均值最大点，合并集合并可视化
    merged_points = get_top_mean_points_by_class(groups, top_n=50)
    plot_global_points(merged_points, title=f'合并重要点位热力图（共{len(merged_points)}点）')
    print('【重要点位选取与分类评估脚本】')
    # 1. 区域选点法（认知友好，覆盖大腿和屁股）
    region_indices = select_key_points_by_region()
    region_indices = [48, 80, 112, 176, 87, 103, 88, 89, 104, 105, 91, 107]
    print(f'区域选点法选出的12个点位下标（覆盖两腿+屁股）: {region_indices}')
    # 2. 极差法+代表性法（原有自动选点，便于对比）
    feature_indices = analyze_important_points(sensor_data, labels)
    print(f'极差法+代表性法选出的12个点位下标: {feature_indices}')

    # 3. 热力图展示（分别标注两种选点）
    # 随机选择一个坐姿类别
    np.random.seed(int(time.time()))
    pose_choice = random.choice(['n', 'l', 'r'])
    pose_map = {'n': 'normal', 'l': 'left', 'r': 'right'}
    print(f"\n本次热力图随机选择坐姿类别: '{pose_choice}' ({pose_map[pose_choice]})")
    print('\n--- 热力图（区域选点法）---')
    show_random_heatmap(pose_choice, important_points=region_indices)

    # 4. 分类器评估
    print('\n--- 分类器评估（全部点）---')
    test_df = pd.read_csv('../data/test_dataset.csv', header=None, skiprows=1)
    test_labels = test_df.iloc[:, 0].astype(str).values
    test_sensor_data = test_df.iloc[:, 1:257].astype(float).values
    evaluate_classifier(test_sensor_data, test_labels, threshold_table, sample_size=100)

    print('\n--- 分类器评估（仅用12个区分力最强点）---')
    evaluate_classifier_with_features(test_sensor_data, test_labels, threshold_table, feature_indices, sample_size=100)

    print('\n--- 分类器评估（区域均值法，认知友好）---')
    evaluate_classifier_by_region(test_sensor_data, test_labels, region_indices, sample_size=100, threshold=0.1)
