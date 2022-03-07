# @yongshuai
import numpy as np
from scipy.fftpack import fft
from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt

Fs = 200  # 采样频率


def fea_trans(fea): # TODO: 什么意思？   
    # 从提取的特征点中回复出脉搏波的波谷、波峰、降中峡、重搏波和波谷
    kp = [0,0]
    kp.append(fea[1]*Fs)
    kp.append(fea[5])
    kp.append((fea[1]+fea[4])*Fs)
    kp.append(fea[6])
    kp.append((fea[1]+fea[3])*Fs)
    kp.append(fea[7])
    kp.append((fea[1]+fea[2])*Fs)
    kp.append(0)
    return kp


def extract_attributes(feature):
    tmp = []
    for i in range(len(feature)): # 本循环含义见：北交论文
        fea = feature[i]

        # TODO: why use fea_trans?
        P = fea_trans(fea)

        Ta = P[0]/Fs
        Tb = P[2]/Fs
        Tc = P[4]/Fs
        Td = P[6]/Fs
        Te = P[8]/Fs
        T1 = Tb-Ta
        T2 = Te-Tb
        T3 = Tc-Tb
        T4 = Td-Tb
        T5 = Td-Tc
        T = Te-Ta
        if T == 0: continue

        H = P[3]-P[1]
        if H == 0: continue
        Hc = (P[5]-P[1])/H
        Hd = (P[7]-P[1])/H

        Beats = 1/T*60
        SV = T / T2  # 每分搏出量
        CC = T / T1  # 心排出量
        CO = Beats*SV  # 心搏出量
        
        if not P[4] == P[6]:
            x = [P[0],P[2],P[4],P[6],P[8]]
            y = [P[1],P[3],P[5],P[7],P[9]]
        else:
            x = [P[0],P[2],P[4],P[8]]
            y = [P[1],P[3],P[5],P[9]]

        func = interp1d(x,y,"quadratic")
        newx = np.arange(P[0],P[8])
        ppg1 = func(newx)

        # delta = (P[9]-P[1])/(P[8]-P[0])
        # bs = [P[1]+delta*i for i in range(P[8]-P[0])]
        # assert(len(ppg1) == len(bs))
        # ppg1 = [p-b for p,b in zip(ppg1, bs)]
        ppg1 = ppg1/max(ppg1)
        data = np.tile(ppg1,10)
        
        L = len(data)
        FFT = abs(fft(data,L))
        FFT = FFT[0:int(L/2)]
        # 寻找峰值
        FFTdif = FFT[1:]-FFT[0:len(FFT)-1]
        Fre = []
        for i in range(0,len(FFTdif)-1):
            if FFTdif[i]*FFTdif[i+1] < -1:
                Fre.append(i+1)
        AmpT = FFT[Fre]
        
        if len(Fre) < 7:
            continue
        idx = np.argsort(AmpT)
        idx = idx[-7:]
        idx.sort()
        # idx = list(idx)
        # idx.reverse()
        Amp = [AmpT[idx[i]] for i in range(7)]
        Fre = [Fre[idx[i]] for i in range(7)]
        
        # plt.figure(1)
        # plt.plot(FFT)
        # plt.scatter(Fre,Amp)
        # plt.show()
        
        Fre = Fs*np.array(Fre)/L
        Fre = list(Fre)
        # tmpT = [Beats,SV,CC,CO,Hc,H]
        tmpT = [T1,T2,T3,T4,T5,T,Hc,Hd,Beats,SV,CC,CO]
        tmpT.extend(Fre)
        tmpT.extend(Amp)

        tmp.append(tmpT)
        
    attribute = []
    if len(tmp) > 0: # TODO: 这部分做什么？
        M = np.mean(tmp,0)
        STD = np.std(tmp,0)
        # print(tmp-np.tile(M,[len(tmp),1]))
        # print(type(STD))
        # print(type(tmp))
        Bias = (tmp - np.tile(M,[len(tmp),1])) / np.tile(STD,[len(tmp),1])
        # print(Bias)
        # print(type(Bias))
        Idx = []
        for i in range(len(Bias)):
            if abs(Bias[i].any()) >= 2:
                Idx.append(i)
        tmp = np.delete(tmp,Idx,axis=0)
        R = len(tmp)
        if R <= 20:
            tmpT = np.mean(tmp,0)
            attribute.append(list(tmpT))
        else:
            for j in range(R-19):
                tmpT = np.mean(tmp[j:j+19],0)
                attribute.append(list(tmpT))
    return attribute


# 以下代码由李永帅于2021年11月2日补充

def isPPGWave(kpts):
    # 判断输入波形是噪声还是脉搏波信号
    # 输入
        # kpts: 输入的特征点
    # 输出
        # 脉搏波信号的特征点

    thre = 0.8   # 判断相邻波形为脉搏波波形的阈值

    curr_ppg = np.zeros(Fs)
    true_ind = []

    kpt = kpts[0]
    P = fea_trans(kpt)
    if not P[4] == P[6]:
        x = [P[0],P[2],P[4],P[6],P[8]]
        y = [P[1],P[3],P[5],P[7],P[9]]
    else:
        x = [P[0],P[2],P[4],P[8]]
        y = [P[1],P[3],P[5],P[9]]
    func = interp1d(x,y,"quadratic")
    newx = np.arange(P[0],P[8])
    ppg = func(newx)
    if len(ppg) <= Fs:
        curr_ppg[:len(ppg)] = ppg
    else:
        curr_ppg = ppg[:Fs]
    pre_ppg = curr_ppg

    for i, kpt in enumerate(kpts):
        P = fea_trans(kpt)
        if not P[4] == P[6]:
            x = [P[0],P[2],P[4],P[6],P[8]]
            y = [P[1],P[3],P[5],P[7],P[9]]
        else:
            x = [P[0],P[2],P[4],P[8]]
            y = [P[1],P[3],P[5],P[9]]
        func = interp1d(x,y,"quadratic")
        newx = np.arange(P[0],P[8])
        ppg = func(newx)
        if len(ppg) <= Fs:
            curr_ppg[:len(ppg)] = ppg
        else:
            curr_ppg = ppg[:Fs]
        rao = corrcoef(curr_ppg, pre_ppg)[0][1]  # 计算相邻波形的相关系数
        if rao > thre:  # 保留大于阈值的波形
            true_ind.append(i)
        # pre_ppg = curr_ppg
        pre_ppg = (pre_ppg * i + curr_ppg) / (i+1)  # 更新预存的脉搏波波形
    return kpts[true_ind]
