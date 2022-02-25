# -*- coding: utf-8 -*-
"""
Client to call function on servers using zerorpc.

Created on 07/12/21 11:40 AM

@file: alg_client.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""

import zerorpc

c = zerorpc.Client(timeout=60, heartbeat=30)
c.connect("tcp://127.0.0.1:4242")

def rpc_test(name):
    return c.rpc_test(name)

def get_activation_code(reqcode):
    return c.get_activation_code(reqcode)

def get_heart_rate(data, type='midvalue'):
    return c.get_heart_rate(data, type)

def get_heart_rate_variability(data):
    return c.get_heart_rate_variability(data)

def get_blood_pressure(data, user):
    return c.get_blood_pressure(data, user)

def get_emotion(data, type='midvalue', user=None):
    return c.get_emotion(data, type, user)  

def get_study_status(data, type='midvalue', user=None):
    return c.get_study_status(data, type, user)

def get_sunny_index(data, type='midvalue', user=None):
    return c.get_sunny_index(data, type, user)

def get_emo_index(data, type='midvalue', user=None):
    return c.get_emo_index(data, type, user)

def get_depression_index(data, type='midvalue', user=None):
    return c.get_depression_index(data, type, user)

def get_anxiety_index(data, type='midvalue', user=None):
    return c.get_anxiety_index(data, type, user)

def get_pressure_index(data, type='midvalue', user=None):
    return get_pressure_index(data, type, user)

def get_fatigue_index(data, type='midvalue', user=None):
    return c.get_fatigue_index(data, type, user)

def get_sunshine_index(data, type='midvalue', user=None):
    return c.get_sunshine_index(data, type, user)

def get_emotional_coordinates_emo(data, type='midvalue', user=None):
    return c.get_emotional_coordinates_emo(data, type, user)

def get_emotion_status(data, type='midvalue', user=None):
    return c.get_emotion_status(data, type, user)

if __name__ == '__main__':
    print(rpc_test("RPC"))