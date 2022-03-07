# -*- coding: utf-8 -*-
"""
Client to call function on servers using zerorpc.

Created on 07/12/21 11:40 AM

@file: alg_client.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""

import sys
import numpy as np
import alg_client as alg

def test_rpc_test():
    print(sys._getframe().f_code.co_name)
    print(alg.rpc_test("RPC"))

def test_get_activation_code():
    req_code = 'WSSsJIzDg5okdiQKuNHSm47KglaYlgRlAJITYoXYewBPechSyE3sPJ+F\/TbAoUcd'
    act_code = alg.get_activation_code(req_code)
    print('activation code: %s' % act_code)
    fact_code = 'jsqCVpiWBGUAkhNihdh7AH8i/SNUtbSXJdFjBj5+Af0Yeb4V9VH8J42WMPUGOpPFSr/4nb2YF1qyW5TX4mOMGg=='
    assert act_code == fact_code

def test_get_heart_rate():
    print(sys._getframe().f_code.co_name)
    ppg_data = np.loadtxt("./tstdata/ute_ppg_data.txt", delimiter=',')
    ppg_data = ppg_data.tolist()
    print(['ppg data length: ', len(ppg_data)])
    result = alg.get_heart_rate(pulse_kpts, type='ppg')
    print(['get_heart_rate result: ', result])

    mid_data = np.loadtxt("./tstdata/ute_ppg_data_midvalue.txt")
    mid_data = mid_data.tolist()
    print(['mid data length: ', len(mid_data)])
    result2 = alg.get_heart_rate(mid_data)
    print(['get_heart_rate result: ', result2])

def test_get_heart_rate_variability():
    print(sys._getframe().f_code.co_name)
    ppg_data = np.loadtxt("./tstdata/ha_ppg_20211012.txt", delimiter=',')
    print(['ppg data shape: ', ppg_data.shape])
    ppg_data = ppg_data.flatten()
    ppg_data = ppg_data.tolist()
    print(['ppg data length: ', len(ppg_data)])
    result = alg.get_heart_rate_variability(ppg_data)
    print(['get_heart_rate_variability result: ', result])

def test_get_blood_pressure():
    print(sys._getframe().f_code.co_name)
    ppg_data = np.loadtxt("./tstdata/ha_ppg_20211012.txt", delimiter=',')
    print(['ppg data shape: ', ppg_data.shape])
    ppg_data = ppg_data.flatten()
    ppg_data = ppg_data.tolist()
    print(['ppg data length: ', len(ppg_data)])
    result = alg.get_blood_pressure(ppg_data, '13521291211')
    print(['get_blood_pressure result: ', result])

def test_get_emotion():
    print(sys._getframe().f_code.co_name)
    ppg_data = np.loadtxt("./tstdata/ha_ppg_20211012.txt", delimiter=',')
    print(['ppg data shape: ', ppg_data.shape])
    ppg_data = ppg_data.flatten()
    ppg_data = ppg_data.tolist()
    print(['ppg data length: ', len(ppg_data)])
    result = alg.get_emotion(ppg_data, type='ppg')
    print(['get_emotion result: ', result])

def test_get_emotion_status():
    print(sys._getframe().f_code.co_name)
    ppg_data = np.loadtxt("./tstdata/ha_ppg_20211012.txt", delimiter=',')
    print(['ppg data shape: ', ppg_data.shape])
    ppg_data = ppg_data.flatten()
    ppg_data = ppg_data.tolist()
    print(['ppg data length: ', len(ppg_data)])
    result = alg.get_emotion_status(ppg_data, type='ppg')
    print(['get_emotion_status result: ', result])



if __name__ == '__main__':
    test_rpc_test()
    #test_get_emotion()
    test_get_activation_code()
    test_get_emotion_status()
    test_get_blood_pressure()
    test_get_heart_rate_variability()


