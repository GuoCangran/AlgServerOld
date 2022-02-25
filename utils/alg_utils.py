# -*- coding: utf-8 -*-
"""
Tools of interaction with database (mongodb).

Created on 07/12/21 11:40 AM

@file: heart_rate_variability.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""


import logging
import algs.library_activation as act
import algs.keypoints_extraction as kpe
import algs.heart_rate_variability as hrv
import algs.emotion_classification as emo_app
import algs.blood_pressure_detection as bp_app

from utils.time_utils import fn_timer
from algs.preprocessing import bp_filter
from algs.emotion_classification import is_emotion_correct


# 创建一个logger
logger = logging.getLogger('root')
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] '
                           '- %(levelname)s: %(message)s',level=logging.INFO)

def get_keycode(reqcode):
    return act.generate_activation_code(reqcode)


def get_hrv(data, fs=100):
    return hrv.heart_rate_var(data, fs)


def get_emotion_feature(data, type="midvalue", emo=0):
    # judge input data is midvalue or ppg, and transform it to a sample (feature).
    status = ""
    if type == "midvalue":
        fm = emo_app.temperal_feature_extender(data)
    elif type == "ppg":
        fm = emo_app.feature_transformer_temperal(kpe.extract_keypoints(data))
    else:
        logging.error("input type error!")
        # raise ValueError("input type error")
        status = "fail"
        return status
    fre_fm, _ = emo_app.feature_transformer_frequential(fm, emo)
    status = "success"
    return status, fre_fm

@fn_timer
def get_emotion(model, data, type="midvalue"):
    # get emotion features of input data
    status, feature = get_emotion_feature(data, type)
    pca = model['pca']
    lda = model['lda']
    clf = model['clf']

    # predict which emotion state the sample is
    label = emo_app.emotion_classifer(feature, pca, lda, clf)

    # map the label predicted to emotion map
    emo_map = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    trans_fun_label = emo_app.emo_transfer_function(emo_map)
    emo = emo_app.emo_transfer(label, trans_fun_label)
    emo = list(int(i) for i in emo)
    status = "success"

    return status, emo

@fn_timer
def get_hr(data, type="midvalue"):
    # judge input data is midvalue or ppg, and transform it to a sample (feature).
    status = ""
    if type == "midvalue":
        fm = emo_app.temperal_feature_extender(data)
    elif type == "ppg":
        kpts = kpe.extract_keypoints(bp_filter(data))
        print(['key points length: ', len(kpts)])
        fm = emo_app.feature_transformer_temperal(kpts)
        print(['fm len: ', len(fm)])
        fm = emo_app.temperal_feature_extender(fm)
        #logger.info("fm: {}".format(fm))
    else:
        logging.error("input type error!")
        status = "fail"
        return status
    if len(fm) == 0:
        status = "fail"
        return status
    # compute the heart rate
    hr = emo_app.heart_rate(fm)
    status = "success"

    return status, hr

@fn_timer
def get_bp(data, user_id):
    return bp_app.predict_blood_pressure(data, user_id)


def get_study_status(emo, hr_result):
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6喜悦
    # 学习状态：=（平静%*100+惊奇%*83.3+愉悦%*57.4+愤怒%*27.8+恐惧%*37+悲伤%*13.9+厌恶%*9.3）0.5+（100-(hr-60)）0.5
    x = 100 if hr_result < 60 else (0 if hr_result > 160 else 160 - hr_result)
    return round(emo[1] * 50 + emo[2] * 41.65 + emo[6] * 28.7 + emo[0] * 13.9 + emo[3] * 18.5 + emo[4] * 6.95 +
                 emo[5] * 4.65 + x * 0.5, 2) if len(emo) > 0 else 0


def get_sunny_index(emo, hr_result):
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6喜悦
    # 阳光指数a =（愉悦%*100+惊奇%*83.3+平静%*57.4+愤怒%*27.8+恐惧%*37+悲伤%*13.9+厌恶%*9.3）0.8+（100-(hr - 60)）0.2
    # 定义：心率相关系数 = 100 - (hr - 60)
    # 备注：如果心率hr < 60，心率相关系数 = 100，如果心率值 > 160, 心率相关系数 = 0
    x = 100 if hr_result < 60 else (0 if hr_result > 160 else 160 - hr_result)
    return round(emo[6] * 80 + emo[2] * 66.64 + emo[1] * 45.92 + emo[0] * 22.24 + emo[3] * 29.6 + emo[4] * 11.12 +
                 emo[5] * 7.44 + x * 0.2, 2) if len(emo) > 0 else 0

def get_emo_index(emo):
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6喜悦
    # 压力度 = 100 - 情绪指数
    return round(emo[0] * 27.8 + emo[1] * 83.3 + emo[2] * 57.4 + emo[3] * 37 + emo[4] * 13.9 + emo[5] * 9.3 +
                 emo[6] * 100, 2) if len(emo) > 0 else 0


def get_depression_index(emo):
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6喜悦
    # 抑郁度 = (0 * 愉悦 + 8 * 平静 + 2 * 惊奇 + 23 * 愤怒 + 66 * 伤心 + 25 * 厌恶 + 29 * 恐惧） * 100 / 66
    return round((emo[6] * 0 + emo[1] * 8 + emo[2] * 2 + emo[0] * 23 + emo[4] * 66 + emo[5] * 25 +
                 emo[3] * 29)*100/66, 2) if len(emo) > 0 else 0

def get_anxiety_index(emo):
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6喜悦
    # 焦虑度 = (0 * 愉悦 + 1 * 平静 + 9 * 惊奇 + 61 * 愤怒 + 26 * 伤心 + 30 * 厌恶 + 20 * 恐惧） * 100 / 61
    return round((emo[6] * 0 + emo[1] * 1 + emo[2] * 9 + emo[0] * 61 + emo[4] * 26 + emo[5] * 30 +
                 emo[3] * 20)*100/66, 2) if len(emo) > 0 else 0

def get_pressure_index(emo):
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6喜悦
    # 压力度 = 100 - (27.8 * 愤怒 + 83.3 * 平静 + 57.4 * 惊奇 + 37 * 恐惧 + 13.9 * 伤心 + 9.3 * 厌恶 + 100 * 喜悦)
    return round(100-(emo[0] * 27.8 + emo[1] * 83.3 + emo[2] * 57.4 + emo[3] * 37 + emo[4] * 13.9 + emo[5] * 9.3 +
                 emo[6] * 100), 2) if len(emo) > 0 else 0

def get_fatigue_index(emo):
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6喜悦
    # 疲劳度 = (0 * 愉悦 + 24 * 平静 + 2 * 惊奇 + 1 * 愤怒 + 41 * 伤心 + 54 * 厌恶 + 5 * 恐惧） * 100 / 54
    return round((emo[6] * 0 + emo[1] * 24 + emo[2] * 2 + emo[0] * 1 + emo[4] * 41 + emo[5] * 54 +
                 emo[3] * 5)*100/54, 2) if len(emo) > 0 else 0

def get_sunshine_index(emo):
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6喜悦
    # 压力度 = (27.8 * 愤怒 + 57.4 * 平静 + 83.3 * 惊奇 + 37 * 恐惧 + 13.9 * 伤心 + 9.3 * 厌恶 + 100 * 喜悦) * 0.8
    return round(0.8 * (emo[0] * 27.8 + emo[1] * 83.3 + emo[2] * 57.4 + emo[3] * 37 + emo[4] * 13.9 + emo[5] * 9.3 +
                        emo[6] * 100), 2) if len(emo) > 0 else 0


def get_emotional_coordinates(emo,hr_result):
    # 返回情绪坐标[0.5,44.0]
    # return [round(emo.count(max(emo, key=emo.count))/7,2), 0 if hr_result <= 40 else 100 if hr_result >=100 else hr_result-40] if len(emo) > 0 else [0,0]
    return [round(max(emo),2), 0 if hr_result <= 40 else 100 if hr_result >=100 else hr_result-40] if len(emo) > 0 else [0,0]


def get_emotional_coordinates_emo(emo,hr_result):
    # 返回情绪坐标及对应情绪{1：[0.5,44.0]}
    is_active = emo.index(max(emo)) if len(emo) > 0 else -1
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6愉悦
    # 愉悦，惊奇第一象限，愤怒，恐惧第二象限，伤心，厌恶第三象限，平静第4象限
    if is_active in [2,6]:
        return {str(emo.index(max(emo))): [round(max(emo),2), 0 if hr_result <= 40 else 100 if hr_result >= 100 else hr_result-40]} if len(emo) > 0 else {-1:[0,0]}
    elif is_active in [0,3]:
        return {str(emo.index(max(emo))): [-round(max(emo),2), 0 if hr_result <= 40 else 100 if hr_result >= 100 else hr_result-40]} if len(emo) > 0 else {-1:[0,0]}
    elif is_active in [4,5]:
        return {str(emo.index(max(emo))): [-round(max(emo),2), 0 if hr_result <= 40 else -100 if hr_result >= 100 else 40-hr_result]} if len(emo) > 0 else {-1:[0,0]}
    else:
        return {str(emo.index(max(emo))): [round(max(emo),2), 0 if hr_result <= 40 else -100 if hr_result >= 100 else 40-hr_result]} if len(emo) > 0 else {-1:[0,0]}


def low_pressure(bp_data):
    # 从如"bp_min" : "75|117"中取出低压
    if len(bp_data.split('|')) > 1:
        low = bp_data.split('|')[0]
        return low
    else:
        return 0
def high_pressure(bp_data):
    # 从如"bp_min" : "75|117"中取出高压
    if len(bp_data.split('|')) > 1:
        high = bp_data.split('|')[1]
        return high
    else:
        return 0


# 获取情绪效价(emo_list)分析结果
def get_emotional_valence_analysis(emo_list,name,start,end):
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6喜悦
    # 积极情绪：2，6；消极情绪：0，3，4，5；中性情绪：1
    global result, status
    res_list = []  # 存放结果列表
    for it in emo_list:
        for chi in it.keys():
            res_list.append(int(chi))
    if res_list == [] or list(set(res_list))[0] == -1 and len(list(set(res_list))) == 1:
        return 5, ""
    # print('res_list:',res_list)
    a, b, c, d = 0, 0, 0, 0
    for i in res_list:
        if i in [2, 6]:
            a += 1
        elif i in [0, 3]:
            b += 1
        elif i in [4, 5]:
            c += 1
        elif i in [1]:
            d += 1
        else:
            pass
    sum1 = a + b + c + d
    li = [a, b, c, d]
    # print('liaaa:',li)
    l_new = list(i / sum1 for i in li)  # [0.09375, 0.15625, 0.1875, 0.5625]
    if all(v < 0.3 for v in l_new):
        status = 5
        result = "%s在%s至%s时间段内情绪未激发状态" % (name, start, end)
    else:
        li_pre1 = [i for i in l_new if i >= 0.7]
        li_pre2 = [i for i in l_new if i >= 0.5]
        li_pre3 = [i for i in l_new if i >= 0.3]
        # 找出l_new中>=0.5(如果有),或>0.4(如果有)或>0.3那个最大元素所对应的索引
        index = l_new.index(max(li_pre1 if len(li_pre1) > 0 else li_pre2 if len(li_pre2) > 0 else li_pre3))
        weight = '高度' if len(li_pre1) > 0 else '中度' if len(li_pre2) > 0 else '轻度'
        if li.index(a) == index:
            status = 1
            result = "%s在%s至%s处于%s积极兴奋态。积极兴奋态：是一种积极情绪下的兴奋状态，表示在第一象限的情绪点数的占比最多。" \
                     "第一象限情绪点越多位置越靠右表示越积极，位置越靠上表示越兴奋，这是一种非常好的状态，表示%s在该时间段内心情很好，" \
                     "可能专注某种事物并有了较大感兴趣,或者表示%s在某种事物上很有天赋，找到了成就感，从而获得了较高的愉悦感或者快感。"%(name,start,end,weight,name,name)
        elif li.index(b) == index:
            status = 2
            result = "%s在%s至%s处于%s消极兴奋态。消极兴奋态：是一种消极情绪下的兴奋状态，表示在第二象限的情绪点数的占比最多。" \
                     "第二象限情绪点越多且位置越靠左表示越消极，位置越靠上表示越兴奋，这是一种不太好的状态，表示%s在该时间段内处于焦虑、暴躁、紧张甚至是恐惧状态，" \
                     "可能是压力比较大导致，或者受到某种威胁，如果长期处于该种情绪状态下，对身心健康不利。"%(name,start,end,weight,name)
        elif li.index(c) == index:
            status = 3
            result = "%s在%s至%s处于%s消极抑郁态。消极抑郁态：是一种消极情绪下的抑郁状态，表示在第三象限的情绪点数的占比最多。" \
                     "第三象限情绪点越多且位置越靠左表示越消极，位置越靠下表示越压抑，这是一种非常不好的状态，表示%s在该段时间内处于消极、颓废、抑郁、甚至是厌世状态，" \
                     "可能是遭遇某种挫折或者经历了一些不好的事情，导致%s对一些人或者事情失望、伤心甚至厌烦，如果长期处于该种情绪状态容易导致抑郁，" \
                     "甚至是引发抑郁症，需要引起相关人的格外关注。"%(name,start,end,weight,name,name)
        elif li.index(d) == index:
            status = 4
            result = "%s在%s至%s处于%s积极平和态。积极平和态：是一种积极情绪下的平和状态，表示在第四象限的情绪点数的占比最多。" \
                     "第四象限情绪点越多且位置越靠右表示越积极，位置越靠下表示越平和，这是一种相对较好的状态，" \
                     "表示%s在该段时间内比较安静、平和，安静、平和的状态比较适合学习。" %(name,start,end,weight,name)
    # 第二次逻辑
    # if all(v < 0.5 for v in l_new):
    #     result = "%s在%s至%s时间段内情绪未激发状态" % (name, start, end)
    # else:
    #     # 找出l_new中>=0.5那个元素所对应的索引（已知l_new中只有一个>=0.5的元素）
    #     index = l_new.index([i for i in l_new if i >= 0.5][0])
    #     if li.index(a) == index:
    #         result = "%s在%s至%s处于积极兴奋态。积极兴奋态：是一种积极情绪下的兴奋状态，表示在第一象限的情绪点数的占比最多。" \
    #                  "第一象限情绪点越多位置越靠右表示越积极，位置越靠上表示越兴奋，这是一种非常好的状态，表示%s在该时间段内心情很好，" \
    #                  "可能专注某种事物并有了较大感兴趣,或者表示%s在某种事物上很有天赋，找到了成就感，从而获得了较高的愉悦感或者快感。"%(name,start,end,name,name)
    #     elif li.index(b) == index:
    #         result = "%s在%s至%s处于消极兴奋态。消极兴奋态：是一种消极情绪下的兴奋状态，表示在第二象限的情绪点数的占比最多。" \
    #                  "第二象限情绪点越多且位置越靠左表示越消极，位置越靠上表示越兴奋，这是一种不太好的状态，表示%s在该时间段内处于焦虑、暴躁、紧张甚至是恐惧状态，" \
    #                  "可能是压力比较大导致，或者受到某种威胁，如果长期处于该种情绪状态下，对身心健康不利。"%(name,start,end,name)
    #     elif li.index(c) == index:
    #         result = "%s在%s至%s处于消极抑郁态。消极抑郁态：是一种消极情绪下的抑郁状态，表示在第三象限的情绪点数的占比最多。" \
    #                  "第三象限情绪点越多且位置越靠左表示越消极，位置越靠下表示越压抑，这是一种非常不好的状态，表示%s在该段时间内处于消极、颓废、抑郁、甚至是厌世状态，" \
    #                  "可能是遭遇某种挫折或者经历了一些不好的事情，导致%s对一些人或者事情失望、伤心甚至厌烦，如果长期处于该种情绪状态容易导致抑郁，" \
    #                  "甚至是引发抑郁症，需要引起相关人的格外关注。"%(name,start,end,name,name)
    #     elif li.index(d) == index:
    #         result = "%s在%s至%s处于积极平和态。积极平和态：是一种积极情绪下的平和状态，表示在第四象限的情绪点数的占比最多。" \
    #                  "第四象限情绪点越多且位置越靠右表示越积极，位置越靠下表示越平和，这是一种相对较好的状态，" \
    #                  "表示%s在该段时间内比较安静、平和，安静、平和的状态比较适合学习。" % (name, start, end,name)
    # 前一次逻辑
    # if res_list == [] or list(set(res_list))[0] == -1 and len(list(set(res_list))) == 1:
    #     result = ""
    # elif a == max(a, b, c, d):
    #     result = "%s在%s至%s处于积极兴奋态。积极兴奋态：是一种积极情绪下的兴奋状态，表示在第一象限的情绪点数的占比最多。" \
    #              "第一象限情绪点越多位置越靠右表示越积极，位置越靠上表示越兴奋，这是一种非常好的状态，表示%s在该时间段内心情很好，" \
    #              "可能专注某种事物并有了较大感兴趣,或者表示%s在某种事物上很有天赋，找到了成就感，从而获得了较高的愉悦感或者快感。"%(name,start,end,name,name)
    # elif b == max(a, b, c, d):
    #     result = "%s在%s至%s处于消极兴奋态。消极兴奋态：是一种消极情绪下的兴奋状态，表示在第二象限的情绪点数的占比最多。" \
    #              "第二象限情绪点越多且位置越靠左表示越消极，位置越靠上表示越兴奋，这是一种不太好的状态，表示%s在该时间段内处于焦虑、暴躁、紧张甚至是恐惧状态，" \
    #              "可能是压力比较大导致，或者受到某种威胁，如果长期处于该种情绪状态下，对身心健康不利。"%(name,start,end,name)
    # elif c == max(a, b, c, d):
    #     result = "%s在%s至%s处于消极抑郁态。消极抑郁态：是一种消极情绪下的抑郁状态，表示在第三象限的情绪点数的占比最多。" \
    #              "第三象限情绪点越多且位置越靠左表示越消极，位置越靠下表示越压抑，这是一种非常不好的状态，表示%s在该段时间内处于消极、颓废、抑郁、甚至是厌世状态，" \
    #              "可能是遭遇某种挫折或者经历了一些不好的事情，导致%s对一些人或者事情失望、伤心甚至厌烦，如果长期处于该种情绪状态容易导致抑郁，" \
    #              "甚至是引发抑郁症，需要引起相关人的格外关注。"%(name,start,end,name,name)
    # elif d == max(a, b, c, d):
    #     result = "%s在%s至%s处于积极平和态。积极平和态：是一种积极情绪下的平和状态，表示在第四象限的情绪点数的占比最多。" \
    #              "第四象限情绪点越多且位置越靠右表示越积极，位置越靠下表示越平和，这是一种相对较好的状态，" \
    #              "表示%s在该段时间内比较安静、平和，安静、平和的状态比较适合学习。" % (name, start, end,name)
    # else:
    #     result = ''
    return status, result


# 获取emo_list中积极，消极，中性的情绪比例
def get_emo_proport(emo_list):
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6喜悦
    # 积极情绪：2，6；消极情绪：0，3，4，5；中性情绪：1
    res_list = []  # 存放结果列表
    for it in emo_list:
        for chi in it.keys():
            res_list.append(int(chi))
    positive, negative, neutral = 0, 0, 0
    for i in res_list:
        if i in [1, 2, 6]:
            positive += 1
        elif i in [0, 3, 4, 5]:
            negative += 1
        # elif i in [1]:
        #     neutral += 1
        else:
            pass
    sum = positive + negative + neutral
    if sum == 0:
        positive_per, negative_per, neutral_per = 0, 0, 0
    else:
        positive_per, negative_per = round((positive/sum)*100), round((negative/sum)*100)
        neutral_per = round(100 - positive_per - negative_per)
    # max_emo_per = int(max(positive_per, negative_per, neutral_per))
    return positive_per, negative_per, neutral_per

# 获取emo_list中占比第一和第二的情绪名称
def get_fir_sec_emo(emo_list):
    # emo_list:[{"2": [0.39,35.150684931506845 ]},{"3": [-0.29,37.263157894736835]}]
    res_list = []  # 存放结果列表
    count_list = []  # 存放各情绪出现次数
    # 将emo_list中所有字典的情绪放到res_list中
    for it in emo_list:     # 遍历emo_list中一个个字典
        for chi in it.keys():   # 遍历字典中的key
            res_list.append(int(chi))  # [2, 3, 2, -1, 6, 6]
    # 去除res_list中的所有-1(直接remove只是移除最先找到的-1)
    # 方式一：for循环遍历
    # for i in range(len(res_list)-1,-1,-1):
    #     if res_list[i] == -1:
    #         res_list.remove(-1)  # [2, 3, 2, 6, 6]
    # 方式二：列表解析
    res_list = [res_list[i] for i in range(0, len(res_list)) if res_list[i] != -1] # [2, 3, 2, 6, 6]
    # 方式三：利用filter和lambda表达式
    # new_list = list(filter(lambda x: x != -1, res_list))
    # 计算出各情绪以及此情绪在情绪列表中出现的次数
    for i in set(res_list):  # {2, 3, 6}
        count_list.append(res_list.count(i))  # [2, 1, 2]
    li = list(set(res_list))  # [2, 3, 6]
    zip_list = list(zip(count_list, li))  # 顺序不能反[(2, 2), (1, 3), (2, 6)]
    # 对zip以情绪出现次数进行从大到小排序
    sort_zip = sorted(zip_list, reverse=True)  # [(2, 6), (2, 2), (1, 3)]
    # # 与 zip 相反，*sort_zip 可理解为解压，返回二维矩阵式
    a = list(zip(*sort_zip))  # [(2, 2, 1), (6, 2, 3)]
    # 取出出现次数排名前2所对应的情绪
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6喜悦
    dic = {0: "愤怒", 1: "平静", 2: "惊奇", 3: "恐惧", 4: "悲伤", 5: "厌恶", 6: "喜悦"}
    if len(a[1]) == 0:
        first_emo, second_emo = '', ''
    elif len(a[1]) == 1:
        first_emo, second_emo = dic[a[1][0]], ''
    else:
        first_emo = dic[a[1][0]]  # 次数最多所对应的情绪
        second_emo = dic[a[1][1]]  # 次数次多所对应的情绪
    return first_emo,second_emo     #('喜悦', '惊奇')

# 获取情绪状态
def get_emotional_status(emo_list):
    res_list = []  # 存放结果列表
    for it in emo_list:
        for chi in it.keys():
            res_list.append(int(chi))
    a, b, c, d = 0, 0, 0, 0
    for i in res_list:
        if i in [2, 6]:
            a += 1
        elif i in [0, 3]:
            b += 1
        elif i in [4, 5]:
            c += 1
        elif i in [1]:
            d += 1
        else:
            pass
    if res_list == [] or list(set(res_list))[0] == -1 and len(list(set(res_list))) == 1:
        result = ""
    elif a == max(a, b, c, d):
        result = "积极兴奋态"
    elif b == max(a, b, c, d):
        result = "消极兴奋态"
    elif c == max(a, b, c, d):
        result = "消极抑郁态"
    elif d == max(a, b, c, d):
        result = "积极平和态"
    else:
        result = ''
    return result

# 获取情绪名字
def get_emo_name(emo_list):
    positive_per, negative_per, neutral_per = get_emo_proport(emo_list)
    max_emo_per = max(positive_per, negative_per, neutral_per)
    res_list = []  # 存放结果列表
    for it in emo_list:
        for chi in it.keys():
            res_list.append(int(chi))
    if res_list == [] or list(set(res_list))[0] == -1 and len(list(set(res_list))) == 1:
        result = ""
    elif int(positive_per) == max_emo_per:
        result = '积极情绪'
    elif int(negative_per) == max_emo_per:
        result = '消极情绪'
    elif int(neutral_per) == max_emo_per:
        result = '中性情绪'
    else:
        result = ""
    return result,max_emo_per

def simple_report(emo_list,name):
    res_list = []  # 存放结果列表
    for it in emo_list:
        for chi in it.keys():
            res_list.append(int(chi))
    if res_list == [] or list(set(res_list))[0] == -1 and len(list(set(res_list))) == 1:
        result = ''
    else:
        first_emo, second_emo = get_fir_sec_emo(emo_list)
        main_emo, max_emo_per = get_emo_name(emo_list)
        emo_status = get_emotional_status(emo_list)
        result = "{0}的基本情绪中主要情绪有{1}、{2}，{3}比例为{4}%占据主要地位，当下{5}处于{6}。详情请查看详细报告。" \
            .format(name, first_emo, second_emo, main_emo, max_emo_per, name, emo_status)
    return result




if __name__ == '__main__':
    simple_report()
