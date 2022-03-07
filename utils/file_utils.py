# -*- coding: utf-8 -*-
"""
File precessing tools.

Created on 07/12/21 11:40 AM

@file: heart_rate_variability.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""

# 读数据
# def read_data(file_name):
#     input_data = ""
#     with open(file_name, "r") as feature_file:
#         line = feature_file.readline()
#         while line:
#             input_data = input_data + line
#             line = feature_file.readline()
#     return input_data[2:-2]

def read_data(file_name):
    input_data = ""
    with open(file_name, "r") as feature_file:
        line = feature_file.readline()
        while line:
            input_data = input_data + line
            line = feature_file.readline()
    return eval(input_data)


def str_hex2dec(input_string):
    input_string = input_string.split(",")
    input_string = input_string[1:-1]  # 我也不知道为什么第一个是双引号。。。
    input_string = map(lambda x: int(x, 10), input_string)
    return list(input_string)
