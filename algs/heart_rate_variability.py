# -*- coding: utf-8 -*-
"""
Compute hrv by ppg signal.

Created on 07/12/21 11:40 AM

@file: heart_rate_variability.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""

import os
import sys
import time
import traceback, json, logging
import numpy as np
from scipy.fftpack import fft
from scipy.signal import convolve
from scipy.signal import resample

from utils.time_utils import fn_timer
from algs.preprocessing import bp_filter, demean
from algs.keypoints_extraction import extract_keypoints

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT)

FS = 200  # 重采样后的采样频率
TOTAL_TIME = 5 # 总数据时间长度(min)
ITV_TIME = TOTAL_TIME/5 # 数据分段时间长度 

def beat_interval(feature, Fs=200):	
    NN = []
    for i in range(len(feature)):
        P = feature[i]
        if len(P) < 10:
            continue
        Ta = P[0]/Fs
        Te = P[8]/Fs
        T = (Te-Ta)*1000.  # ms
        NN.append(T) 
    return NN

@fn_timer
def heart_rate_var(data0, fs):
    # 计算时域心率变异性系数
    # 输入：
    #   data0: 脉搏波
    #   fs: 实际的采样频率
    # 输出：
    #   result: 7个时域心率变异性系数
    # 参考文献：孙瑞龙等，心率变异性检测临床应用的建议，中华心血管病杂志，1998,26.4

    logging.info("计算心率变异性系数...")
    result = dict()
    try:
        # 从100Hz插值成200Hz
        data0 = resample(data0, round(FS/fs*len(data0)))
        if len(data0)/FS/60 < TOTAL_TIME: # 判断信号长度
            raise ValueError("采集时间少于指定时间{}分钟".format(TOTAL_TIME))
        data0 = data0[FS:len(data0)-FS]  # 去除前后1秒的数据
        ## 带通滤波/去均值
        data0 = demean(data0)
        data0 = bp_filter(data0)

        Np = int(FS*60*ITV_TIME)  # 信号分段
        NNs = []
        NN_mean = []
        NN_std = []
        for ii in range(int(len(data0)/Np)):
            data = data0[ii*Np:(ii+1)*Np]
            kps = extract_keypoints(data)
            NN = beat_interval(kps)
            NNs.extend(NN)
            NN_mean.append(np.mean(NN))
            NN_std.append(np.std(NN))
        data = data0[(ii+1)*Np:]
        kps = extract_keypoints(data)
        NN = beat_interval(kps)
        NNs.extend(NN)
        NN_mean.append(np.mean(NN))
        NN_std.append(np.std(NN))

        NNs_diff = np.diff(NNs)

        SDNN = np.std(NNs)
        SDANN = np.nanstd(NN_mean)
        RMSSD = np.sqrt(np.sum(np.square(NNs_diff))/(len(NNs_diff)))
        SDNN_index = np.nanmean(NN_std)
        SDSD = np.std(NNs_diff)
        NN50 = np.where(NNs_diff>0.050,1,0).sum()
        PNN50 = NN50/len(NNs)*100

        print("Heart rate variability coefficients are: {}"\
            .format([SDNN,SDANN,RMSSD,SDNN_index,SDSD,NN50,PNN50]))

        result["isSuccessful"] = True
        result["SDNN"] = str(SDNN)
        result["SDANN"] = str(SDANN)
        result["RMSSD"] = str(RMSSD)
        result["SDNN_index"] = str(SDNN_index)
        result["SDSD"] = str(SDSD)
        result["NN50"] = str(NN50)
        result["PNN50"] = str(PNN50)
        return json.dumps(result)
    except Exception as e:
        logging.error(repr(e))
        logging.error(traceback.format_exc())
        result["isSuccessful"] = False
        return json.dumps(result)

if __name__ == "__main__":
    ppg_dir = "ring/algorithm/ppg_bp_/test_data/"
    file_names = os.listdir(ppg_dir)
    for name in file_names:
        if name == "." or name == "..":
            continue
        ppg_path = os.path.join(ppg_dir,name)
        print(name)
        data0 = np.loadtxt(ppg_path)
        heart_rate_var(data0, 100)

        # plt.plot(data0)
        # plt.savefig(ppg_path+".png")

    # data0 = np.random.randn(6*60*100)
    # heart_rate_var(data0)
