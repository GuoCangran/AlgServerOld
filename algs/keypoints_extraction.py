# -*- coding: utf-8 -*-
"""
Extract key points of ppg signal.

Created on 07/12/21 11:40 AM

@file: keypoints_extraction.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""


import numpy as np


def extract_keypoints(data):
    feature = list()
    # data = bp_filter(data)
    #    print('datalen',len(data))
    #    plt.figure()
    #    plt.plot(data[0:1000])
    #    plt.show()
    downsample = 2
    if len(data) > 10000:
        downsample = 4
    if len(data) > 20000:
        downsample = 8
    if len(data) > 50000:
        downsample = 16
    if len(data) > 100000:
        downsample = 32

    D = abs(np.fft.fft(data[::downsample]))
    D = D.tolist()
    # 计算出周期和幅值
    Dlowgate = int(len(D)*downsample/150)
    D = D[Dlowgate:int(len(D) / 2)]
    # D = D[0:int(len(D) / 2)]
    #    plt.plot(D)
    tmp = max(D)
    # tmpT = D.index(tmp)
    tmpT = D.index(tmp) + Dlowgate
    T = len(D) / tmpT * downsample * 2
    count = 0
    iternum = 0
    flag = 0
    # max_slope = tmp * downsample / 5000
    max_slope = tmp * downsample / 20000
    maxupper = max_slope * 1.5

    datadiff = [0] * (len(data) + 1)
    for i in range(0, len(data)):
        if i <= 0:
            datadiff[i] = data[i] / 4
        elif i <= 2:
            datadiff[i] = (2 * data[i] + data[i - 1]) / 8
        elif i <= 3:
            datadiff[i] = (2 * data[i] + data[i - 1] - data[i - 3]) / 8
        else:
            datadiff[i] = (2 * data[i] + data[i - 1] - data[i - 3] - 2 * data[i - 4]) / 8
    #    plt.plot(datadiff[0:500])
    # 从平滑斜率中找上升最大点
    featuretmp = [-200 for j in range(0, 10)]
    for i in range(0, len(data)):
        if i >= 70:
            maxdatadiff = max(datadiff[i - 70:i])
            idmax = (datadiff[i - 70:i]).index(maxdatadiff)
            if idmax == 35 and (maxdatadiff > max_slope * 0.7):
                max_slope = min(maxdatadiff, maxupper)
                j = i - 35
                while datadiff[j] > 0 and j > 0:
                    j = j - 1
                if flag == 2 and count >= 1:
                    featuretmp[8] = j - 2
                    featuretmp[9] = data[j - 2]
                    feature.append(featuretmp)
                count = count + 1
                featuretmp = [-200 for j in range(0, 10)]
                featuretmp[0] = j - 2
                featuretmp[1] = data[j - 2]
                j = i - 35
                while datadiff[j] > 0 and j < len(data):
                    j = j + 1
                    iternum = iternum + 1
                featuretmp[2] = j - 2
                featuretmp[3] = data[j - 2]
                flag = 1

            if flag == 1 and i > featuretmp[0] + T * 0.35 + 10 and i < featuretmp[0] + T * 0.8 + 10:

                maxdatadiff = max(datadiff[i - 20:i])
                idmax = (datadiff[i - 20:i]).index(maxdatadiff)
                if idmax == 10:
                    if maxdatadiff <= 0:
                        featuretmp[4] = i - 12
                        featuretmp[5] = data[i - 12]
                        featuretmp[6] = i - 12
                        featuretmp[7] = data[i - 12]
                        flag = 2
                    else:
                        j = i - 10
                        while datadiff[j] > 0 and j > featuretmp[2]:
                            j = j - 1
                        featuretmp[4] = j - 2
                        featuretmp[5] = data[j - 2]
                        j = i - 10
                        while datadiff[j] > 0 and j < len(data):
                            j = j + 1
                        featuretmp[6] = j - 2
                        featuretmp[7] = data[j - 2]
                        flag = 2
        #print('originalfeaturepoint:',len(feature))
    return feature


if __name__ == "__main__":
    pass