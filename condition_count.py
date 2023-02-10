# -*- coding: UTF-8 -*-
# @Time : 2023/2/7 15:03
# @Author : Yao
# @File : condition_count.py
# @Software : PyCharm
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
import sklearn.metrics
from scipy import stats
import csv
import sklearn.metrics
import os
from scipy.stats import norm


# 计算准确度
def accuracy(prediction, test):
    labels = [0, 1, 2, 3, 4, 5, 6]
    return sklearn.metrics.precision_score(test, prediction, labels=labels, average='micro')


def get_test(Path):
    with open(Path, 'r') as cvsfile:
        reader = csv.reader(cvsfile)
        column = [row[0] for row in reader]
        column = list(map(int, column))
        return column


# 矫正识别数组
def correct_results(results, length):
    for d in range(0, len(results)):
        if results[d] != 0:
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


# 梯形识别
class Trapezoid:
    def calculate_slope(self, x, y):
        slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
        return slope

    # Mann-Kendall趋势检验
    def mk(self, data, alpha=0.1):  # 0<alpha<0.5 1-alpha/2为置信度
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
    def judge_slope(self, x, y, flag):
        # 下钻斜边
        if flag == 1:
            return self.mk(y) == -1
        # 下钻下横边
        elif flag == 2:
            return self.mk(x) == -1
        # 下钻竖边
        elif flag == 3:
            return self.mk(y) == 1
        # 下钻上横边
        elif flag == 4:
            return self.mk(x) == 1

        # 起钻斜边
        elif flag == -1:
            return self.mk(x) == 1
        # 起钻右竖边
        elif flag == -2:
            return self.mk(y) == 1
        # 起钻横边
        elif flag == -3:
            return self.mk(x) == -1
        # 起钻左竖边
        elif flag == -4:
            return self.mk(y) == -1

        # 接单根下横边
        elif flag == 10:
            return self.mk(x) == -1
        # 接单根左竖边
        elif flag == 20:
            return self.mk(y) == 1
        # 接单根上横边
        elif flag == 30:
            return self.mk(x) == 1
        # 接单根右竖边
        elif flag == 40:
            return self.mk(y) == -1

    def recognize(self, length, step, datapath, x):
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
        jie_vertical_zuo = x[16]
        jie_vertical_you = x[17]
        jie_horizontal_zuo = x[18]
        jie_horizontal_you = x[19]
        recognition = []
        try:
            data = pd.read_csv(datapath, encoding="gbk")
        except UnicodeDecodeError:
            data = pd.read_csv(datapath, encoding="utf-8")
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
                slope = self.calculate_slope(sx, sy)
            except ValueError:
                slope = 0
                continue
            # 状态0
            if state == 0:
                # 出现接单根横边
                if jie_horizontal_zuo >= slope >= jie_horizontal_you:
                    # 进入状态10
                    state = 10
                    # 记录可能开始接单根的时间点
                    start = i
                    continue
                # 出现起钻斜边，表示此时可能可似起钻
                if up_slope_zuo >= slope >= up_slope_you and self.judge_slope(sx, sy, -1):
                    # 进入状态-1
                    state = -1
                    # 记录可能开始起钻的时间点
                    start = i
                    continue
                # 其出现下钻斜边，此时表示可能开始下钻
                elif down_slope_zuo >= slope >= down_slope_you and self.judge_slope(sx, sy, 1):
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
                if down_horizontal_down_zuo >= slope >= down_horizontal_down_you and self.judge_slope(sx, sy, 2):
                    # 进入状态2
                    state = 2
                    continue
                # 还在斜边状态
                elif down_slope_zuo >= slope >= down_slope_you and self.judge_slope(sx, sy, 1):
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
                if slope <= down_vertical_zuo or slope >= down_vertical_you and self.judge_slope(sx, sy, 3):
                    # 进入状态3
                    state = 3
                    continue
                # 还在横边状态
                elif down_horizontal_down_zuo >= slope >= down_horizontal_down_you and self.judge_slope(sx, sy, 2):
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
                if down_horizontal_up_zuo >= slope >= down_horizontal_up_you and self.judge_slope(sx, sy, 4):
                    # 进入状态4
                    state = 4
                    continue
                # 还在竖边状态
                elif (slope <= down_vertical_zuo or slope >= down_vertical_you) and self.judge_slope(sx, sy, 3):
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
                if down_slope_zuo >= slope >= down_slope_you and self.judge_slope(sx, sy, 1):
                    # 进入状态1
                    state = 1
                    flag = 1
                    # 完成转一个梯形
                    for j in range(start, i):
                        recognition[j] = 5
                    start = i
                    continue
                # 且还在横边
                elif down_horizontal_up_zuo >= slope >= down_horizontal_up_you and self.judge_slope(sx, sy, 4):
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
                if slope <= up_vertical_right_zuo or slope >= up_vertical_right_you and self.judge_slope(sx, sy, -2):
                    # 进入状态2
                    state = -2
                    continue
                # 还在斜边状态
                elif up_slope_zuo >= slope >= up_slope_you and self.judge_slope(sx, sy, -1):
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
                if up_horizontal_zuo >= slope >= up_horizontal_you and self.judge_slope(sx, sy, -3):
                    # 进入状态3
                    state = -3
                    continue
                # 还在竖边状态
                elif (slope <= up_vertical_right_zuo or slope >= up_vertical_right_you) and self.judge_slope(sx, sy,
                                                                                                             -2):
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
                if (slope <= up_vertical_left_zuo or slope >= up_vertical_left_you) and self.judge_slope(sx, sy, -4):
                    # 进入状态4
                    state = -4
                    continue
                # 还在横边状态
                elif up_horizontal_zuo >= slope >= up_horizontal_you and self.judge_slope(sx, sy, -3):
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
                if down_slope_zuo >= slope >= down_slope_you and self.judge_slope(sx, sy, -1):
                    # 进入状态-1
                    state = -1
                    flag = 1
                    # 完成转一个梯形
                    for j in range(start, i):
                        recognition[j] = 4
                    start = i
                    continue
                # 且还在竖边
                elif (slope <= up_vertical_left_zuo or slope >= up_vertical_left_you) and self.judge_slope(sx, sy, -4):
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

            # 状态10 接单根横边
            if state == 10:
                # 出现竖边
                if jie_vertical_zuo >= slope or slope >= jie_vertical_you:
                    # 进入状态20
                    state = 20
                    continue
                # 还在横边
                elif jie_horizontal_zuo >= slope >= jie_horizontal_you:
                    continue
                # 转化过程中
                elif not (jie_vertical_you > slope > jie_horizontal_zuo):
                    continue
                else:
                    state = 0
                    continue

            # 状态20 接单根竖边
            if state == 20:
                # 出现横边
                if jie_horizontal_zuo >= slope >= jie_horizontal_you:
                    # 进入状态30
                    state = 30
                    continue
                # 还在竖边
                elif jie_vertical_zuo >= slope or slope >= jie_vertical_you:
                    continue
                # 转化过程中
                elif not (jie_horizontal_you > slope > jie_vertical_zuo):
                    continue
                else:
                    state = 0
                    continue

            # 状态30 接单根横边
            if state == 30:
                # 出现竖边
                if jie_vertical_zuo >= slope or slope >= jie_vertical_you:
                    state = 40
                    continue
                # 还在横边
                elif jie_horizontal_zuo >= slope >= jie_horizontal_you:
                    continue
                # 转化过程中
                elif not (jie_vertical_you > slope > jie_horizontal_zuo):
                    continue
                else:
                    state = 0
                    continue

            # 状态40 接单根竖边
            if state == 40:
                # 出现横边，完成一个接单根工况
                if jie_horizontal_zuo >= slope >= jie_horizontal_you:
                    state = 0
                    for j in range(start, i):
                        recognition[j] = 3
                    continue
                # 还在竖边
                elif jie_vertical_zuo >= slope or slope >= jie_vertical_you:
                    continue
                # 转化过程中
                elif not (jie_horizontal_you > slope > jie_horizontal_zuo):
                    continue
                else:
                    state = 0
                    continue

        if flag == 1 and state != 0:
            for i in range(start, len(data)):
                recognition[i] = 5 if state in (1, 2, 3, 4) else 4
        return recognition


# 阈值法识别
class Threshold:
    # 计算斜率
    def calculate_slope(self, data):
        reg = linear_model.LinearRegression()
        reg.fit(np.array(range(len(data))).reshape(-1, 1), np.array(data).reshape(-1, 1))
        slope = reg.coef_  # 斜率
        intercept = reg.intercept_  # 截距
        slope = slope[0][0]
        intercept = intercept[0]
        return slope

    # 阈值法
    def condition(self, DataPath, x):
        MinDepthDifference = x[0]
        MinIngressFlow = x[1]
        MinIngressFlowRate = x[2]
        MinHookLoad = x[3]
        MinHookLoadRate = x[4]
        MinHookHeight = x[5]
        MaxHookHeight = x[6]
        MinHookHeightRate = x[7]
        MinDrillDepthDifference = x[8]
        ConditionResult = []  # 工况识别结果
        length = x[9]  # 计算斜率步长
        try:
            info = pd.read_csv(DataPath, encoding="gbk")
        except UnicodeDecodeError:
            info = pd.read_csv(DataPath, encoding="utf-8")
        info['Time'] = pd.to_numeric(info['Time'], errors='coerce')
        info['井深(m)'] = pd.to_numeric(info['井深(m)'], errors='coerce')
        info['钻头深度(m)'] = pd.to_numeric(info['钻头深度(m)'], errors='coerce')
        info['平均钻压(KN)'] = pd.to_numeric(info['平均钻压(KN)'], errors='coerce')
        info['大钩负荷(KN)'] = pd.to_numeric(info['大钩负荷(KN)'], errors='coerce')
        info['大钩位置(m)'] = pd.to_numeric(info['大钩位置(m)'], errors='coerce')
        info['立压log(MPa)'] = pd.to_numeric(info['立压log(MPa)'], errors='coerce')
        info['出口流量log(%)'] = pd.to_numeric(info['出口流量log(%)'], errors='coerce')
        info['转盘转速(rpm)'] = pd.to_numeric(info['转盘转速(rpm)'], errors='coerce')
        info['入口流量log(L/s)'] = pd.to_numeric(info['入口流量log(L/s)'], errors='coerce')
        for index, row in info.iterrows():
            stage = []
            ConditionResult.append(0)
            # 空井工况
            if row['钻头深度(m)'] == 0 and row['立压log(MPa)'] == 0 and row['出口流量log(%)'] == 0 and row[
                '转盘转速(rpm)'] == 0 and row['钻头深度(m)'] == 0:
                continue

            # 其他工况
            if row['立压log(MPa)'] == 0 and row['出口流量log(%)'] == 0 and row['转盘转速(rpm)'] == 0 and row[
                '钻头深度(m)'] != 0:
                ConditionResult[index] = 6
                continue

            # 钻头深度与井深差
            # 有立压有入口流量有钻压且钻深与井深差距小
            if abs(row['钻头深度(m)'] - row['井深(m)']) < MinDepthDifference and row['平均钻压(KN)'] > 0 and row[
                '立压log(MPa)'] > 0 and row['入口流量log(L/s)'] > 0:
                # 钻进工况
                ConditionResult[index] = 1
                continue

            # 抛弃前面时间窗口大小个数据不计算斜率
            if index < length:
                continue

            # 入口流量大于最小入口流量
            if row['入口流量log(L/s)'] > MinIngressFlow:
                # 入口流量变化率
                IngressFlowRate = self.calculate_slope(info.loc[index - length: index + 1, '入口流量log(L/s)'].tolist())
                # 入口流量变化率小于最小入口流量变化率
                if abs(IngressFlowRate) < MinIngressFlowRate and row['立压log(MPa)'] > 0:
                    if row['平均钻压(KN)'] == 0:
                        # 循环工况
                        ConditionResult[index] = 2
                        continue
                    # 循环工况
                    ConditionResult[index] = 2
                elif IngressFlowRate > MinIngressFlow:
                    # 开泵过程
                    stage.append(1)
                else:
                    # 停泵过程
                    stage.append(2)
            # 入口流量大于等于最小入口流量
            else:
                # 停泵状态
                stage.append(3)

            # 如果大钩负荷小于最小大钩负荷
            if row['大钩负荷(KN)'] < MinHookLoad:
                # 大钩空载
                stage.append(4)
            # 大钩负荷大于等于最小大钩负荷
            # 表示大钩非空载，带钻柱
            else:
                # 大钩负荷变化率
                HookLoadRate = self.calculate_slope(info.loc[index - length: index + 1, '大钩负荷(KN)'].tolist())
                # 大钩负荷变化率小于最小大钩负荷变化率
                if abs(HookLoadRate) < MinHookLoadRate:
                    # 大钩负荷稳定
                    stage.append(5)
                # 大钩负荷变化剧烈
                else:
                    if HookLoadRate > MinHookLoadRate:
                        # 大钩负荷升高
                        stage.append(6)
                    else:
                        # 大钩负荷降低
                        stage.append(7)

            # 大钩高度位置小于最小大钩高度
            if row['大钩位置(m)'] < MinHookHeight:
                # 大钩高度为最低值
                stage.append(8)
            # 大钩高度位置大于等于最小大钩高度
            else:
                if row['大钩位置(m)'] > MaxHookHeight:
                    # 大钩高度为最高值
                    stage.append(9)
                else:
                    # 大钩高度变化率
                    HookHeightRate = self.calculate_slope(info.loc[index - length: index + 1, '大钩位置(m)'])
                    if abs(HookHeightRate) > MinHookHeightRate:
                        # 大钩高度变化剧烈
                        if HookHeightRate > MinHookHeightRate:
                            # 大钩高度升高
                            stage.append(10)
                        else:
                            # 大钩高度降低
                            stage.append(11)
                    else:
                        # 大钩高度基本不变
                        stage.append(12)

            # 钻头深度差
            DrillDepthDifference = row['钻头深度(m)'] - info.loc[index - 1, '钻头深度(m)']
            # 钻头深度差小于最小钻头深度差
            if abs(DrillDepthDifference) < MinDrillDepthDifference:
                # 钻头深度不变
                stage.append(13)
            else:
                if DrillDepthDifference > 0:
                    # 钻头深度下降
                    stage.append(14)
                elif DrillDepthDifference < 0:
                    # 钻头深度上升
                    stage.append(15)
            # 接单根工况
            if (2 in stage and 10 in stage and 7 in stage) or (3 in stage and 12 in stage and 4 in stage) or (
                    3 in stage and 10 in stage and 4 in stage) or (1 in stage and 11 in stage and 6 in stage):
                ConditionResult[index] = 3
            # 起钻工况
            elif (3 in stage and 10 in stage and 14 in stage) or (3 in stage and 12 in stage and 13 in stage) or (
                    3 in stage and 11 in stage and 13 in stage):
                ConditionResult[index] = 4
            # 下钻工况
            elif (3 in stage and 10 in stage and 13 in stage) or (3 in stage and 12 in stage and 13 in stage) or (
                    3 in stage and 11 in stage and 15 in stage):
                ConditionResult[index] = 5
            else:
                ConditionResult[index] = ConditionResult[index - 1]
        return ConditionResult


# 合并识别
def Unite(re1, re2):
    recognition_list = []
    for i in range(0, len(re1)):
        if re1[i] == re2[i]:
            recognition_list.append(re1[i])
        else:
            if re1[i] in [0, 1, 2, 6]:
                recognition_list.append(re1[i])
            else:
                recognition_list.append(re2[i])
    return recognition_list


# 写入csv
def write_csv(result, path):
    result = list(map(lambda x: [x], result))
    f1 = open(path, 'w', newline="")
    csv_write = csv.writer(f1)
    for data in result:
        csv_write.writerow(data)
    f1.close()


if __name__ == '__main__':
    threshold = Threshold()
    test_path = r'E:\program file(x86)\project\工况识别\csv\jie\jie_test.csv'
    train_path = r'E:\program file(x86)\project\工况识别\csv\jie\jie.csv'
    test = get_test(test_path)
    x = [2.45240286e-01, 1.53321564e-01, 6.51335232e-02, 300,
         7.47433354e-01, 1.5, 35, 1.20129655e-01,
         1.62397371e-01, 20]
    recognition1 = threshold.condition(train_path, x)
    trapezoid = Trapezoid()
    x = [2.63823889, -1.22163172, 2.70066099, -0.76321839, -1.96321717, -2.8072773, 0.81127697, -0.87252804, 1.30414409,
         0.09385573, -2.06653517, -2.05836141, 3.23514693, 0.5938165, 2.33840449, 0.82033646, -1.11494547, 0.16846922,
         0.4577732, -0.38045619]
    recognition2 = trapezoid.recognize(35, 1, train_path, x)
    recognition2 = correct_results(recognition2, 35)
    recognition = Unite(recognition1, recognition2)
    f = accuracy(recognition1, test)
    print(recognition1)
    print(f)
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    f = accuracy(recognition2, test)
    print(recognition2)
    print(f)
    print('-------------------------------------')
    print('-------------------------------------')
    print('-------------------------------------')
    f = accuracy(recognition, test)
    print(recognition)
    print(f)
    host = host_subplot(111, axes_class=axisartist.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    # par2 = host.twinx()

    # par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(60, 0))

    par1.axis["right"].toggle(all=True)
    # par2.axis["right"].toggle(all=True)
    index = range(1, len(recognition) + 1, 1)

    p1, = host.plot(index, recognition, label="pre_con")
    p2, = par1.plot(index, test, label="con")

    host.set_ylim(0, 6)
    par1.set_ylim(0, 6)

    host.set_xlabel("index")
    host.set_ylabel("pre_con")
    par1.set_ylabel("con")
    # par2.set_ylabel("IOU (%)")

    host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    plt.show()
    plt.plot()
