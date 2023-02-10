# -*- coding: UTF-8 -*-
# @Time : 2022/12/27 20:23
# @Author : Yao
# @File : Unite_csv.py
# @Software : PyCharm
import pandas as pd
import os

Folder_Path = r'E:\program file(x86)\project\工况识别\data'  # 要拼接的文件夹及其完整路径，注意不要包含中文
SaveFile_Path = r'E:\program file(x86)\project\工况识别\data'  # 拼接后要保存的文件路径
SaveFile_Name = r'train.csv'  # 合并后要保存的文件名

# 修改当前工作目录
os.chdir(Folder_Path)
# 将该文件夹下的所有文件名存入一个列表
file_list = os.listdir()

# 读取第一个CSV文件并包含表头
try:
    df = pd.read_csv(Folder_Path + '\\' + file_list[0], encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(Folder_Path + '\\' + file_list[0], encoding='gbk')
# 将读取的第一个CSV文件写入合并后的文件保存
df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf-8", index=False)

# 循环遍历列表中各个CSV文件名，并追加到合并后的文件
for i in range(1, len(file_list)):
    try:
        df = pd.read_csv(Folder_Path + '\\' + file_list[i], encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(Folder_Path + '\\' + file_list[i], encoding='gbk')
    df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf-8", index=False, header=False, mode='a+')
