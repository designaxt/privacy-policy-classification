import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
import csv
import json
import re

from bs4 import BeautifulSoup
from tensorflow import keras


#%%

# 读取csv文件获取文件名列表


input_dir = r'C:\python_project\ML\OPP-115\consolidation\threshold-0.5-overlap-similarity'
get_filename = tf.io.gfile.listdir(input_dir)  # 得到csv文件名列表
filenames = []
path_format = os.path.join(input_dir, '{}')
for filename in get_filename:
    print(filename)
    part_csv = path_format.format(filename)
    filenames.append(part_csv)


#%%

# 逐个读取csv文件将segment_id, type和value三列取出放入type_value列表中

type_value = []
for filename in filenames:
    with open(filename, 'r', encoding='gbk') as rf:
        reader = csv.reader(rf, dialect=csv.excel)
        temp_value = []
        for row in reader:
            temp_value.append([row[4], row[5], row[6], row[8]])
        type_value.append(temp_value)

for filename in type_value:
    print(filename)

#%%

# 将Other类改为Other类下的具体取值，即10分类改为12分类

for i in range(len(type_value)):
    for j in range(len(type_value[i])):
        if type_value[i][j][1] == 'Other':
            data = json.loads(type_value[i][j][2])
            get_value = data['Other Type']
            print(get_value['value'])
            type_value[i][j][1] = get_value['value']

for value in type_value:
    print(value)

#%%

# 获取具体privacy policy html文件列表
input_dir = r'C:\python_project\ML\OPP-115\sanitized_policies'
get_filename = os.listdir(input_dir)
html_filenames = []
path_format = os.path.join(input_dir, '{}')
for filename in get_filename:
    print(filename)
    part_csv = path_format.format(filename)
    html_filenames.append(part_csv)

#%%

# get privacy policy segment

segments = []
for filename in html_filenames:
    soup = BeautifulSoup(open(filename), 'html.parser')
    segments.append(re.split('\|\|\|', soup.get_text()))
    print(filename)
    print(len(re.split('\|\|\|', soup.get_text())))

#%%

# type_value: [[['segment_id', 'type', 'attribute']]]
# segments: [[segment]]
# len(type_values) == len(segments)
# merge(type_values, segments) use segment_id -> dataset

dataset = []
for i in range(len(type_value)):
    data = []
    for j in range(len(type_value[i])):
        for z in range(len(segments[i])):
            x = z
            y = int(type_value[i][j][0])
            if x == y:
                data.append([type_value[i][j][0],
                             segments[i][z],
                             type_value[i][j][1],
                             type_value[i][j][2],
                             type_value[i][j][3]])
    dataset.append(data)

#%%

# 根据需求写不同csv文件

output_dir = "clean_data"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

with open(r'clean_data\clean_data_0.5.csv', 'w', encoding='gbk', newline='') as f:
    writer = csv.writer(f, dialect=csv.excel, delimiter=',')
    for data in dataset:
        for single_data in data:
            writer.writerow(single_data)

#%%

with open(r'clean_data\clean_data_0.5_no_attribute.csv', 'w', encoding='gbk', newline='') as f:
    writer = csv.writer(f, dialect=csv.excel, delimiter=',')
    for data in dataset:
        for single_data in data:
            writer.writerow(single_data)


#%%

# 提取文件名中网站名称的部分用作比较

compare = []
for i in range(len(filenames)):
    item = [re.split(r'\.html', re.split(r'\\', html_filenames[i])[5])[0],
            re.split(r'\.csv', re.split(r'\\', filenames[i])[6])[0]
            ]
    print(item)
    compare.append(item)

#%%
#  读取之前生成的文件
text = []
with open(r'clean_data\clean_data_0.5_no_attribute.csv', 'r', encoding='gbk') as rf:
    reader = csv.reader(rf, dialect=csv.excel)
    for row in reader:
        text.append(row)

#%%
#  去重，合并同segment的不同type
dataset = []
typeset = set()
typeset.add(text[0][2])
for i in range(len(text)-1):
    if text[i][0] == text[i+1][0]:
        typeset.add(text[i+1][2])
    else:
        dataset.append([text[i][3], text[i][0], text[i][1], typeset])
        typeset = set()
        typeset.add(text[i+1][2])
dataset.append([text[len(text)-1][3], text[len(text)-1][0], text[len(text)-1][1], typeset])

#%%
#  形成二进制向量的type并去除‘Other’类与空集
type_dic = {'Introductory/Generic': 0, 'Practice not covered': 1,
            'Privacy contact information': 2, 'User Access, Edit and Deletion': 3,
            'Data Security': 4, 'International and Specific Audiences': 5,
            'Do Not Track': 6, 'User Choice/Control': 7,
            'Data Retention': 8, 'Policy Change': 9,
            'First Party Collection/Use': 10, 'Third Party Sharing/Collection': 11}
for i in range(len(dataset)):
    init_type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    init_type_list = []
    dataset[i][3] = dataset[i][3] - {'Other'}
    for j in range(len(list(dataset[i][3]))):
        init_type[type_dic[list(dataset[i][3])[j]]] = 1
        init_type_list.append(type_dic[list(dataset[i][3])[j]])
    dataset[i].append(init_type)
    dataset[i].append(init_type_list)

final_dataset = []
for data in dataset:
    if len(data[3]) != 0:
        final_dataset.append(data)


#%%
#  导出csv文件
with open(r'clean_data\clean_data_0.5_no_repeat.csv', 'w', encoding='gbk', newline='') as f:
    writer = csv.writer(f, dialect=csv.excel, delimiter=',')
    for data in final_dataset:
        writer.writerow(data)

        
#%%

''' 处理APP350数据 '''
#  APP350数据处理
#  读取每一篇隐私政策
input_dir = r'C:\python_project\ML\original_documents'
get_filename = tf.io.gfile.listdir(input_dir)  # 得到csv文件名列表
filenames = []
path_format = os.path.join(input_dir, '{}')
for filename in get_filename:
    print(filename)
    part_csv = path_format.format(filename)
    filenames.append(part_csv)


#%%
#  获取隐私政策纯文本内容
segments = []
for filename in filenames:
    file = open(filename, encoding='ISO-8859-1', errors='ignore')
    soup = BeautifulSoup(file, 'html.parser')
    segments.append(soup.get_text())

#%%
#  将数据写入csv文件
with open(r'clean_data\APP350.csv', 'w', encoding='ISO-8859-1', errors='ignore', newline='') as f:
    writer = csv.writer(f, dialect=csv.excel, delimiter=',')
    for data in segments:
        writer.writerow(data)