# -*- coding: UTF-8 -*-
# @Time : 2022/11/15 19:13
# @Author : Yao
# @File : recognition.py
# @Software : PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
import sklearn.metrics
from scipy import stats
import csv
from scipy.stats import norm


# 下钻图形 5
#            →4
#   |---------------------/
# 3 |                    /  ↓1
# ↑ |                   /
#   |__________________/
#           ←2

# 起钻图形 4
#      ←3
#   |----------|
#   |          |
# 4 |          |
# ↓ |          |  ↑2
#   |       /
#   |     /
#   |  /
#   |/
#         →1

# 计算斜率
def calculate_slope(x, y):
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
    return slope


# Mann-Kendall趋势检验
def mk(data, alpha=0.05):  # 0<alpha<0.5 1-alpha/2为置信度
    n = len(data)
    # 计算S的值
    s = 0
    for j in range(n - 1):
        for i in range(j + 1, n):
            s += np.sign(data[i] - data[j])
    # 判断x里面是否存在重复的数，输出唯一数队列unique_x,重复数数量队列tp
    unique_x, tp = np.unique(data, return_counts=True)
    g = len(unique_x)
    # 计算方差VAR(S)
    if n == g:  # 如果不存在重复点
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:
        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18
    # 计算z_value
    if n <= 10:  # n<=10属于特例
        z = s / (n * (n - 1) / 2)
    else:
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
    # 计算p_value，可以选择性先对p_value进行验证
    p = 2 * (1 - norm.cdf(abs(z)))
    # 计算Z(1-alpha/2)
    h = abs(z) > norm.ppf(1 - alpha / 2)
    # 趋势判断，-1下降，1上升，0保持
    if (z < 0) and h:
        trend = -1
    elif (z > 0) and h:
        trend = 1
    else:
        trend = 0
    return trend


# 判断是否符合形状，1符合，0不符合
def judge_slope(x, y, flag):
    # 下钻斜边
    if flag == 1:
        return mk(y) == -1
    # 下钻下横边
    elif flag == 2:
        return mk(x) == -1
    # 下钻竖边
    elif flag == 3:
        return mk(y) == 1
    # 下钻上横边
    elif flag == 4:
        return mk(x) == 1
    # 起钻斜边
    elif flag == -1:
        return mk(x) == 1
    # 起钻右竖边
    elif flag == -2:
        return mk(y) == 1
    # 起钻横边
    elif flag == -3:
        return mk(x) == -1
    # 起钻左竖边
    elif flag == -4:
        return mk(y) == -1


def recognize(length, step, datapath, x):
    down_slope_zuo = x[0]
    down_slope_you = x[1]
    down_horizontal_down_zuo = x[2]
    down_horizontal_down_you = x[3]
    down_vertical_zuo = x[4]
    down_vertical_you = x[5]
    down_horizontal_up_zuo = x[6]
    down_horizontal_up_you = x[7]
    up_slope_zuo = x[8]
    up_slope_you = x[9]
    up_vertical_right_zuo = x[10]
    up_vertical_right_you = x[11]
    up_horizontal_zuo = x[12]
    up_horizontal_you = x[13]
    up_vertical_left_zuo = x[14]
    up_vertical_left_you = x[15]
    recognition = []
    # jie_vertical_zuo = x[16]
    # jie_vertical_you = x[17]
    # jie_horizontal_zuo = x[18]
    # jie_horizontal_you = x[19]
    data = pd.read_csv(datapath, encoding="gbk")
    for i in range(0, length):
        recognition.append(0)
    data['大钩负荷(KN)'] = pd.to_numeric(data['大钩负荷(KN)'], errors='coerce')
    data['大钩位置(m)'] = pd.to_numeric(data['大钩位置(m)'], errors='coerce')
    x = data['大钩负荷(KN)'].tolist()
    y = data['大钩位置(m)'].tolist()
    state = 0
    start = 0
    flag = 0
    for i in range(length, len(data), step):
        sx = x[i - length: i]
        sy = y[i - length: i]
        recognition.append(0)
        try:
            slope = calculate_slope(sx, sy)
        except ValueError:
            continue
        # 状态0
        if state == 0:
            # 出现起钻斜边，表示此时可能可似起钻
            if up_slope_zuo >= slope >= up_slope_you and judge_slope(sx, sy, -1):
                # 进入状态-1
                state = -1
                # 记录可能开始起钻的时间点
                start = i
                continue
            # 其出现下钻斜边，此时表示可能开始下钻
            elif down_slope_zuo >= slope >= down_slope_you and judge_slope(sx, sy, 1):
                # 进入状态1
                state = 1
                # 记录可能开始下钻的时间点
                start = i
                continue
            else:
                continue

        # 状态1下钻
        if state == 1:
            # 状态1且出现横边
            if down_horizontal_down_zuo >= slope >= down_horizontal_down_you and judge_slope(sx, sy, 2):
                # 进入状态2
                state = 2
                continue
            # 还在斜边状态
            elif down_slope_zuo >= slope >= down_slope_you and judge_slope(sx, sy, 1):
                continue
            # 状态1且在改变过程中
            elif down_slope_you > slope > down_horizontal_down_zuo:
                continue
            # 状态1出现错误
            else:
                if flag == 1:
                    for j in range(start, i):
                        recognition[j] = 5
                    flag = 0
                # 清空当前状态
                state = 0
                continue

        # 状态2
        if state == 2:
            # 状态2且出现竖边
            if slope <= down_vertical_zuo or slope >= down_vertical_you and judge_slope(sx, sy, 3):
                # 进入状态3
                state = 3
                continue
            # 还在横边状态
            elif down_horizontal_down_zuo >= slope >= down_horizontal_down_you and judge_slope(sx, sy, 2):
                continue
            # 状态2且在改变过程中
            elif down_horizontal_down_you > slope > down_vertical_zuo:
                continue
            # 状态2且出现错误
            else:
                if flag == 1:
                    for j in range(start, i):
                        recognition[j] = 5
                    flag = 0
                # 清空当前状态
                state = 0
                continue

        # 状态3
        if state == 3:
            # 状态3且出现横边
            if down_horizontal_up_zuo >= slope >= down_horizontal_up_you and judge_slope(sx, sy, 4):
                # 进入状态4
                state = 4
                continue
            # 还在竖边状态
            elif (slope <= down_vertical_zuo or slope >= down_vertical_you) and judge_slope(sx, sy, 3):
                continue
            # 状态3且在改变过程中
            elif down_vertical_you > slope > down_horizontal_up_zuo:
                continue
            # 状态3且出现错误
            else:
                if flag == 1:
                    for j in range(start, i):
                        recognition[j] = 5
                    flag = 0
                # 清空当前状态
                state = 0
                continue

        # 状态4
        if state == 4:
            # 状态4且出现斜边
            if down_slope_zuo >= slope >= down_slope_you and judge_slope(sx, sy, 1):
                # 进入状态1
                state = 1
                flag = 1
                # 完成转一个梯形
                for j in range(start, i):
                    recognition[j] = 5
                start = i
                continue
            # 且还在横边
            elif down_horizontal_up_zuo >= slope >= down_horizontal_up_you and judge_slope(sx, sy, 4):
                continue
            # 状态4在改变过程中
            elif not (down_slope_you >= slope >= down_horizontal_up_zuo):
                continue
            # 状态4且出现错误
            else:
                if flag == 1:
                    for j in range(start, i):
                        recognition[j] = 5
                    flag = 0
                # 清空当前状态
                state = 0
                continue

        # 起钻
        # 状态-1
        if state == -1:
            # 状态-1且出现竖边
            if slope <= up_vertical_right_zuo or slope >= up_vertical_right_you and judge_slope(sx, sy, -2):
                # 进入状态2
                state = -2
                continue
            # 还在斜边状态
            elif up_slope_zuo >= slope >= up_slope_you and judge_slope(sx, sy, -1):
                continue
            # 状态1且在改变过程中
            elif up_vertical_right_you > slope > up_slope_zuo:
                continue
            else:
                if flag == 1:
                    for j in range(start, i):
                        recognition[j] = 4
                    flag = 0
                # 清空当前状态
                state = 0
                continue

        # 状态-2
        if state == -2:
            # 状态-2且出现横边
            if up_horizontal_zuo >= slope >= up_horizontal_you and judge_slope(sx, sy, -3):
                # 进入状态3
                state = -3
                continue
            # 还在竖边状态
            elif (slope <= up_vertical_right_zuo or slope >= up_vertical_right_you) and judge_slope(sx, sy, -2):
                continue
            # 状态2且在改变过程中
            elif up_horizontal_you > slope > up_vertical_right_zuo:
                continue
            # 状态2且出现错误
            else:
                if flag == 1:
                    for j in range(start, i):
                        recognition[j] = 4
                    flag = 0
                # 清空当前状态
                state = 0
                continue

        # 状态-3
        if state == -3:
            # 状态-3且出现竖边
            if (slope <= up_vertical_left_zuo or slope >= up_vertical_left_you) and judge_slope(sx, sy, -4):
                # 进入状态4
                state = -4
                continue
            # 还在横边状态
            elif up_horizontal_zuo >= slope >= up_horizontal_you and judge_slope(sx, sy, -3):
                continue
            # 状态3且在改变过程中
            elif up_vertical_left_you > slope > up_horizontal_zuo:
                continue
            # 状态3且出现错误
            else:
                if flag == 1:
                    for j in range(start, i):
                        recognition[j] = 4
                    flag = 0
                # 清空当前状态
                state = 0
                continue

        # 状态4
        if state == -4:
            # 状态4且出现斜边
            if down_slope_zuo >= slope >= down_slope_you and judge_slope(sx, sy, -1):
                # 进入状态-1
                state = -1
                flag = 1
                # 完成转一个梯形
                for j in range(start, i):
                    recognition[j] = 4
                start = i
                continue
            # 且还在竖边
            elif (slope <= up_vertical_left_zuo or slope >= up_vertical_left_you) and judge_slope(sx, sy, -4):
                continue
            # 在转化过程中
            elif not (up_vertical_left_you > slope > up_slope_zuo):
                continue
            # 状态-4且出现错误
            else:
                if flag == 1:
                    for j in range(start, i):
                        recognition[j] = 4
                    flag = 0
                # 清空当前状态
                state = 0
                continue
    if flag == 1 and state != 0:
        for i in range(start, len(data)):
            recognition[i] = 5 if state in (1, 2, 3, 4) else 4
    return recognition


# 矫正识别数组
def correct_results(results, length):
    for d in range(0, len(results)):
        if 0 <= d < length:
            results[d] = results[length]
        elif results[d] != 0:
            continue
        else:
            left = results[d - 1]
            flag = len(results)
            right = -1
            for i in range(d + 1, len(results)):
                if results[i] != 0:
                    right = results[i]
                    flag = i
                    break
            if flag != len(results):
                if left == right:
                    for i in range(d, flag):
                        results[i] = left
                else:
                    for i in range(d, flag):
                        results[i] = right
            else:
                for i in range(d, flag):
                    results[i] = left
    return results


# 写入csv
def write_csv(result, path):
    result = list(map(lambda x: [x], result))
    f1 = open(path, 'w', newline="")
    csv_write = csv.writer(f1)
    for data in result:
        csv_write.writerow(data)
    f1.close()


# 计算准确度
def accuracy(prediction, test):
    labels = [0, 1, 2, 3, 4, 5, 6]
    return sklearn.metrics.precision_score(test, prediction, labels=labels,
                                           average='micro'), sklearn.metrics.recall_score(test, prediction,
                                                                                          labels=labels,
                                                                                          average='micro')


# 获取测试集
def get_test(data_path):
    with open(data_path, 'r', encoding="gbk") as cvsfile:
        reader = csv.reader(cvsfile)
        column = [row[0] for row in reader]
        column = list(map(int, column))
        return column


if __name__ == "__main__":
    train_path = r"./csv/train.csv"
    test_path = r'./csv/test.csv'
    recognition = []
    # 最初
    # x = [0.67532432, -2.00188218, 0.32985325, -0.3815734, 1.86937267, 2.46096856, -0.39712865, -0.67755756,
    #      -0.61794174, -1.54333804, 0.65326676, -2.53248301, 1.1898823, -0.74424434, -0.03722278, -0.49302816]
    # 1
    # x = [1.83439792, -1.1241716, 1.47109978, -1.85462222, -1.90941997, 1.02621688, -0.5408754, -1.03062812,
    # 0.44301909, -2.54828274, -2.91633602, 0.89952303, -0.17010565, 0.69362846, 2.6398917, -3.81385013]
    # 2
    # x = [3.511622, 0.15508649, -0.89534738, -3.45228784, 3.10539463, 0.58945733, 0.80664312, -0.69320675, 1.73901615,
    #      -0.74083321, 3.25133309, 1.07802334, 1.8479336, -0.26953004, 1.35123651, 0.95691367]
    # 3
    # x = [3.38916915, 0.19569176, -0.84858362, -3.45060192, 2.70040587, 0.87800947, 0.95534692, -0.53937446, 2.02712219,
    #      -0.53002209, 3.15764821, 1.10675355, 1.38627878, -0.45928462, 1.08220193, 0.53545313]
    # 31
    # x = [3.42995676, 0.26343698, -1.0300656, -3.42901042, 2.71240413, 0.83320584, 0.90754251, -0.4536412, 2.33303444,
    #      -0.0385356, 3.22903472, 1.11663346, 1.48939839, -0.78142092, 1.08901266, 0.70727005]
    # 50
    # x = [3.97648449, 0.26223076, - 1.03095176, - 3.42857434, 2.73062409, 0.83339226, 0.90897497, - 0.45409935,
    #      2.33310466, - 0.0385631, 3.2296796, 1.11524181, 1.48279906, - 0.78161647, 0.84541654, 0.71627685]
    # 51
    x = [3.92078648, 0.26429379, -1.03046161, -3.42242777, 2.77149018, 0.831477, 0.88262926, -0.45372691, 2.33303734,
         -0.03881193, 3.22804183, 1.11785215, 1.48648021, -0.97836924, 0.97712656, 0.71583385]
    recognition = recognize(35, 1, train_path, x)
    test = get_test(test_path)
    recognition = correct_results(recognition, 35)
    print(recognition)
    ps, rs = accuracy(recognition, test)
    print(ps, rs)
    save_path = r'./csv/results.csv'
    write_csv(recognition, save_path)
