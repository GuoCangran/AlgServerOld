# -*- coding: utf-8 -*-
"""
Some function for ppg signal preprocessing.

Created on 07/12/21 11:40 AM

@file: filters.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""

import numpy as np
from algs.filter import lowpass_filter, highpass_filter


# band-pass filter to elimate the nosie
def bp_filter(data, lowcoef=lowpass_filter, highcoef=highpass_filter):
    data = np.convolve(data, lowcoef, 'same')
    data = np.convolve(data, highcoef, 'same')
    return data

# de-mean data to center 0
def demean(data):
    return data - np.mean(data)

# median filter
def median_filter(x, winlen):
    y = list()
    for i in range(0, len(x)):
        if i < winlen:
            y.append(np.median(x[0:i + 1]))
        else:
            if i >= len(x) - winlen:
                y.append(np.median(x[i:len(x)]))
            else:
                y.append(np.median(x[i:i + winlen]))
    return y


if __name__ == '__main__':
    data = np.loadtxt("../tstdata/ha_ppg_20211104.txt", delimiter=',')
    data = data.flatten()
    print(['data length: ', len(data)])
    pdata = preprocess(data)
    print(['pdata length: ', len(pdata)])
