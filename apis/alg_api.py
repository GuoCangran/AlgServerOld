# -*- coding: utf-8 -*-
"""
Wrap ppg algrithm use unique interface.
Input: parameter of client requestion.
Output: tuple of result after compution.

Created on 17/12/21 11:40 AM

@file: alg_api.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""

import time
import logging
import random
import utils.dbs_utils as dbs
import utils.alg_utils as alg

from datetime import datetime, timedelta
from utils.response import corr_response, err_response


# 创建一个logger
logger = logging.getLogger('root')
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] '
                           '- %(levelname)s: %(message)s',level=logging.INFO)

model = dbs.read_model_from_dbs(owner='general')

class AlgorithmRPC(object):
    def rpc_test(self, name):
        return "Hello, %s" % name

    def get_activation_code(self, reqcode):
        keycode = alg.get_keycode(reqcode)
        return keycode

    def get_heart_rate(self, data, type='midvalue'):
        status, hr = alg.get_hr(data, type)
        return hr

    def get_heart_rate_variability(self, data, fs=100):
        return alg.get_hrv(data, fs)

    def get_blood_pressure(self, data, user):
        return alg.get_bp(data, user)

    def get_emotion(self, data, type='midvalue', user=None):
        if user is None:
            use_model = model       
        else:
            use_model = dbs.read_model_from_dbs(owner=user)

        status, emo_label = alg.get_emotion(use_model, data, type)

        return emo_label

    def get_study_status(self, data, type='midvalue', user=None):
        emo_label = self.get_emotion(data, type, user)
        hr_value = self.get_heart_rate(data, type)
        emo = [emo_label.count(i) / len(emo_label) for i in range(0, 7)] if emo_label else []
        return alg.get_study_status(emo, hr_value)

    def get_sunny_index(self, data, type='midvalue', user=None):
        emo_label = self.get_emotion(data, type, user)
        hr_value = self.get_heart_rate(data, type)
        emo = [emo_label.count(i) / len(emo_label) for i in range(0, 7)] if emo_label else []
        return alg.get_sunny_index(emo, hr_value)

    def get_emo_index(self, data, type='midvalue', user=None):
        emo_label = self.get_emotion(data, type, user)
        emo = [emo_label.count(i) / len(emo_label) for i in range(0, 7)] if emo_label else []
        return alg.get_emo_index(emo)

    def get_depression_index(self, data, type='midvalue', user=None):
        emo_label = self.get_emotion(data, type, user)
        emo = [emo_label.count(i) / len(emo_label) for i in range(0, 7)] if emo_label else []
        return alg.get_depression_index(emo)

    def get_anxiety_index(self, data, type='midvalue', user=None):
        emo_label = self.get_emotion(data, type, user)
        emo = [emo_label.count(i) / len(emo_label) for i in range(0, 7)] if emo_label else []
        return alg.get_anxiety_index(emo)

    def get_pressure_index(self, data, type='midvalue', user=None): 
        hr_value = self.get_heart_rate(data, type)
        pi1 = 0.05 * hr_value + 44.7 - 0.1 * (float)(random.randint(0, 150) + 1)
        emo_label = self.get_emotion(data, type, user)
        emo = [emo_label.count(i) / len(emo_label) for i in range(0, 7)] if emo_label else []
        pi2 = alg.get_pressure_index(emo)
        return (pi1 + pi2) / 2

    def get_fatigue_index(self, data, type='midvalue', user=None):
        emo_label = self.get_emotion(data, type, user)
        logger.info('emo_label: {}'.format(emo_label))
        emo = [emo_label.count(i) / len(emo_label) for i in range(0, 7)] if emo_label else []
        return alg.get_fatigue_index(emo)

    def get_sunshine_index(self, data, type='midvalue', user=None):
        emo_label = self.get_emotion(data, type, user)
        emo = [emo_label.count(i) / len(emo_label) for i in range(0, 7)] if emo_label else []
        return alg.get_sunshine_index(emo)

    def get_emotional_coordinates_emo(self, data, type='midvalue', user=None):
        emo_label = self.get_emotion(data, type, user)
        hr_value = self.get_heart_rate(data, type)
        emo = [emo_label.count(i) / len(emo_label) for i in range(0, 7)] if emo_label else []
        return alg.get_emotional_coordinates_emo(emo, hr_value)

    def get_emotion_status(self, data, type='midvalue', user=None):
        emo_label = self.get_emotion(data, type, user)
        hr_value = self.get_heart_rate(data, type)
        emo = [emo_label.count(i) / len(emo_label) for i in range(0, 7)] if emo_label else []

        print("after get emotion and hrs")
        ss = alg.get_study_status(emo_label, hr_value)
        si = alg.get_sunny_index(emo_label, hr_value)
        ei = alg.get_emo_index(emo_label)
        di = alg.get_depression_index(emo_label)
        ai = alg.get_anxiety_index(emo_label)
        pi = alg.get_pressure_index(emo_label)
        fi = alg.get_fatigue_index(emo_label)
        ssi = alg.get_sunshine_index(emo_label)
        ece = alg.get_emotional_coordinates_emo(emo_label, hr_value)
        print("computed all indexes!")

        result = dict({'study_status': ss, 'sunny_index': si, 'emo_index': ei, 'depression_index': di,
                       'anxiety_index': ai, 'pressure_index': pi, 'fatigue_index': fi, 'sunshine_index': ssi,
                       'emotional_coordinates_emo': ece})
        #ss, si, ei, di, ai, pi, fi, ssi, ece
        return result
        







