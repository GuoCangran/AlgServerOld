# -*- coding: utf-8 -*-
"""
Client to call function on servers using zerorpc.

Created on 07/12/21 11:40 AM

@file: alg_client.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""

def upload_train_data_first(params):
    """
    上传第一次标定训练数据
    :params(字典）:
        | 参数         | 必选  | 类型     | 说明                     |
        | ------------ | ---- | -------- | -------------------------|
        | user_id      | true | string   | 用户ID                    |
        | data         | true | list     | 数据                      |
        | data_category| true | string   | 数据类别(ppg or midvalue) |
        | start_time   | true | Datetime | 采集开始时间              |
        | end_time     | true | Datetime | 采集结束时间              |
        | imei_id      | true | string   | 设备码，长度17，0-9A-F   |
        | video_id     | true | string   | 视频id                 |
        | emo_category | true | integer  | 情绪类型(0-6)          |
        | emo_level    | true | integer  | 情绪强度(1-5)          |
        | is_cover     | true | boolean  | 是否覆盖上传,1覆盖     |

        - is_cover是为了保证第一次标定数据有效且唯一(方便第一次建模) 如果用户重复观看视频就需要将它设为1，覆盖之前的数据
    :returns(字典）:
        | 返回字段 | 字段类型 | 说明                   |
        | -------- | -------- | ---------------------- |
        | status   | bool     | 返回结果状态true/false |
    """
    # todo:数据校验，结束时间必须比开始时间大三分钟以上，
    #  用户存在性校验，得根据mysql的用户表校验，包括设备号
    #  也是，然后视频分类也得根据mysql记录查询
    logger.info('data: {} '.format(params))
    user = params['user_id']
    is_cover = params['is_cover']
    calibration_data = dbs.retrieve_mongos("calibration_data", 
                                    {user_id: user,
                                     similarity: 0,
                                     emo_category: params['emo_category']})
    logger.info('calibration_data: {} \n'.format(calibration_data))
    # 保证基本模型每个情绪有且仅有一条
    if is_cover and calibration_data.count() > 0:
        dbs.delete_mongos('calibration_data', {user_id: user})
    if (not is_cover) and calibration_data.count() > 0:
        return err_response('1015', ' 绪基本模型数据已存在')

    # 训练数据过来先进行特征提取并保存结果
    start_time = time.clock()
    status, feature = alg.get_feature(params['data'], params['data_category'], params['emo_category'])
    finish_time = time.clock()
    feature_exectime = finish_time - start_time
    logger.info('user: {}   emo: {} length: {} feature generator time: {}'.format(
        user, params['emo_category'], len(feature), feature_exectime))

    time_now = datetime.now()
    time_now.strftime('%Y-%m-%d')

    params['is_used'] = 0
    params['similarity'] = 0      # 基本模型的数据都设为0，便于管理以及区分二次标定的数据
    params['feature'] = feature
    logger.info('data: {} '.format(params))
    try:
        dbs.create_mongos('calibration_data', params)
    except Exception:
        return err_response('1103', 'mongo保存数据失败')

    return corr_response()


def upload_train_data_second(params):
    """
    上传第二次标定数据,这里会对该情绪数据和第一次上传的其他情绪数据重新建模
    然后校验之前上传的该情绪的所有数据，找出两组具有一定相似性的数据当作最终
    模型的标准数据，可多次上传
    :params(字典）:
        | 参数         | 必选  | 类型     | 说明                     |
        | ------------ | ---- | -------- | -------------------------|
        | user_id      | true | string   | 用户ID                    |
        | model_times  | true | integer  | 0, 1 or 2                |
        | data         | true | list     | 数据                      |
        | data_category| true | string   | 数据类别(ppg or midvalue) |
        | start_time   | true | Datetime | 采集开始时间              |
        | end_time     | true | Datetime | 采集结束时间              |
        | device_code  | true | string   | 设备码，长度17，0-9A-F   |
        | video_id     | true | string   | 视频id                 |
        | emo_category | true | integer  | 情绪类型(0-6)          |
        | emo_level    | true | integer  | 情绪强度(1-5)          |
    :returns(字典）:
        | 返回字段 | 字段类型 | 说明                   |
        | -------- | -------- | ---------------------- |
        | status   | bool     | 返回结果状态true/false |
    """
    logger.info('data: {} '.format(params))
    user = dbs.retrieve_mongos("user", {user_id: params['user_id']})
    logger.info('user_id: {} \n'.format(user.user_id))
    if user['model_times'] != 1:
        return err_response('1016', '请先建立基础模型再进行第二次标定')
    # 1.提取特征
    start_time = time.clock()
    try:
        status, feature = alg.get_feature(params['data'], params['data_category'], params['emo_category'])
        logger.info('feature: {} \n'.format(feature))
    except Exception:
        logger.info('Exception: {} \n'.format(Exception))
        return err_response('1014', '特征提取失败， 数据无效, 请重新采集')
    finish_time = time.clock()
    feature_exectime = finish_time - start_time
    logger.info('user: %s emo: %s length: %s feature generator time: %s'
                % (user, params['emo_category'], len(feature), feature_exectime))

    time_now = datetime.now()
    time_now.strftime('%Y-%m-%d')

    # 直接读模型校验看看
    try:
        emotion_model = dbs.retrieve_mongos('emotion_model', {'user_id': user.user_id})
        pca, lda, clf = dbs.loads_model(emotion_model['pca'], emotion_model['lda'], emotion_model['clf'])
        logger.info('pca: {} \n'.format(pca))
    except Exception:
        logger.info('Exception: {} \n'.format(Exception))
        return err_response('1014', '模型加载失败')
    try:
        flag = alg.is_emotion_correct(feature, pca, lda, clf, params['emo_category'])
        logger.info('flag : {} \n'.format(flag))
    except Exception:
        logger.info('Exception: {} \n'.format(Exception))
        return err_response('1014', 'is_emotion_correct调用出错')
    if flag:
        params['feature'] = feature
        params['is_used'] = 0
        params['similarity'] = 0
        try:
            dbs.create_mongos('calibration_data', params)
        except Exception:
            return err_response('1103', 'mongo保存数据失败')
        logger.info('一次就校验成功了 \n')
        return corr_response()

    # 2.找出基本模型的其他情绪特征值
    other_emo = dbs.retrieve_mongos("calibration_data", 
                             {user_id: user_id,
                              is_used: True,
                              similarity: 0, 
                              emo_category: {'$ne': params['emo_category']}}, 
                              {'feature': 1, 'emo_category': 1})

    FmFre = feature
    FmEmo = [params['emo_category'] for i in range(len(feature))]
    for emo in other_emo:
        FmFre.extend(emo.feature)
        FmEmo.extend([emo.emo_category for i in range(len(emo.feature))])
    logger.info('2.找出基本模型的其他情绪特征值 \n')
    # 3.建模
    pca, lda, clf = alg.train_model(FmFre, FmEmo)
    logger.info('3.建模 \n')

    # 4.找出该情绪历史特征数据
    history_data = dbs.retrieve_mongos("calibration_data", 
                             {user_id: user_id,
                              emo_category: params['emo_category']}, 
                              {'feature': 1, 'emo_category': 1, 'similarity': 1, 'is_used': 1})
    logger.info('4.找出该情绪历史特征数据 \n')
    # 5.校验
    params['feature'] = feature
    params['user_id'] = user
    params['is_used'] = 0
    for emo_data in history_data:
        if emo_data['similarity'] == 0:
            model_data = emo_data
            break
    params['similarity'] = 0
    for emo_data in history_data:
        if alg.is_emotion_correct(emo_data.feature, pca, lda, clf, params['emo_category']):
            # 如果校验成功，更改这两条数据状态similarity=0,这样就得到了一组具有相似性的数据
            
            
            # 将匹配的历史数据交换基础模型的数据,并保存最新提交的数据
            try:
                if emo_data.similarity != 0:
                    emo_data.similarity, model_data.similarity = model_data.similarity, emo_data.similarity
                    emo_data.is_used, model_data.is_used = model_data.is_used, emo_data.is_used
                    model_data.save()
                    emo_data.save()
                dbs.create_mongos('calibration_data', params)
            except Exception:
                return err_response('1103', 'mongo保存数据失败')
            return corr_response()
    logger.info('5.校验 \n')
    # 6.校验未通过，将情绪状态设为该情绪历史标定数据条数之和
    params['similarity'] = len(history_data)
    logger.info('data: {} \n'.format(data))
    try:
        dbs.create_mongos('calibration_data', params)
    except Exception:
        return err_response('1017', '保存数据失败')
    return err_response('1017', '校验未通过，请继续观看该情绪其他视频')



def create_first_model(params):
    """
    建立基础模型
    :param request:
    :param data:
        | 参数  | 必选 | 类型   | 说明 |
        | ----- | ---- | ------ | ---- |
        | token | true | string | 令牌 |
    :return:
        | 返回字段 | 字段类型 | 说明                   |
        | -------- | -------- | ---------------------- |
        | status   | bool     | 返回结果状态true/false |
    """
    logger.info('params:{}\n'.format(params))
    user = dbs.retrieve_mongos("user", {user_id: params['user']})
    user_id = user.user_id
    if user.model_times != 0:
        return err_response('1020', '已存在基础模型，无需重复创建')

    base_model = dbs.retrieve_mongos("calibration_data", 
                             {user_id: user_id,
                              is_used: False,
                              similarity: 0}, 
                              {'feature': 1, 'emo_category': 1})
    # logger.info('base_model: {} \n'.format(base_model))
    logger.info('base_model length: %s \n' % (len(base_model)))
    if len(base_model) != 7:
        return err_response('1018', '请先采集完七种情绪的有效数据再建立基础模型')
    else:
        FmFre, FmEmo = list(), list()
        # base_model = base_model.only('feature', 'emo_category')
        # start_time = timezone.now()
        for emo in base_model:
            FmFre.extend(emo.feature)
            FmEmo.extend([emo.emo_category for i in range(len(emo.feature))])
        logger.info('FmFre: {} FmEmo:{}'.format(FmFre, FmEmo))
        # end_time = timezone.now()
        # print((end_time - start_time).total_seconds(), len(FmFre))
        try:
            pca, lda, clf = alg.train_model(FmFre, FmEmo)
        except Exception as ex:
            logger.info('alg.train_model err {}'.format(ex))
        try:
            pca_bin, lda_bin, clf_bin = dbs.dumps_model(pca, lda, clf)
            emotion_model = dbs.create_mongos('emotion_model',
                                             {'user_id': user.user_id, 
                                              'pca': pca_bin, 'lda': lda_bin, 'clf': clf_bin})
        except Exception as ex:
            logger.info(' dbs.create_mongos err {} \n'.format(ex))
            return err_response('1105', '创建个人基础模型失败，请重新创建')

    # 创建成功，修改对应数据状态
    dbs.update_mongos("calibration_data",
                      {user_id: user_id,
                      is_used: False,
                      similarity: 0},
                      {'$set': {is_used: True}})
    dbs.update_mongos("user",
                      {user_id: user_id},
                      {'$set': {model_times: 1}})

    return corr_response()


def create_second_model(params):
    """
    创建第二次标定模型（完整模型）
    :param request:
    :param data:
        | 参数  | 必选 | 类型   | 说明 |
        | ----- | ---- | ------ | ---- |
        | token | true | string | 令牌 |
    :return:
        | 返回字段 | 字段类型 | 说明                   |
        | -------- | -------- | ---------------------- |
        | status   | bool     | 返回结果状态true/false |
    """
    logger.info('params:{}\n'.format(params))
    user = dbs.retrieve_mongos("user", {user_id: params['user']})
    user_id = user.user_id

    if user.model_times != 1:
        return err_response('1016', '请先建立基础模型')
    else:
        model_data = dbs.retrieve_mongos("calibration_data", 
                             {user_id: user_id,
                              similarity: 0}, 
                              {'feature': 1, 'emo_category': 1, 'is_used': 1})

        second_data = [data for data in model_data if data['is_used'] == False]
        if len(second_data) < 7:
            logger.info('second_data.count() < 7 提升关尚有未校验成功的情绪 \n')
            return err_response('1019', '提升关尚有未校验成功的情绪')
        else:
            FmFre, FmEmo = list(), list()
            for emo in model_data:
                FmFre.extend(emo.feature)
                FmEmo.extend([emo.emo_category for i in range(len(emo.feature))])
            pca, lda, clf = alg.train_model(FmFre, FmEmo)
            try:
                pca_bin, lda_bin, clf_bin = dbs.dumps_model(pca, lda, clf)
                emotion_model = dbs.create_mongos('emotion_model',
                                             {'user_id': user.user_id, 
                                              'pca': pca_bin, 'lda': lda_bin, 'clf': clf_bin})
            except Exception:
                return err_response('1105', '创建个人模型失败，请重新创建')

    dbs.update_mongos("user",
                      {user_id: user_id},
                      {'$set': {model_times: 2}})

    return corr_response()


def get_train_status(params):
    """
    获取标定状态
    :param request:
    :param data:
        | 参数  | 必选 | 类型   | 说明 |
        | ----- | ---- | ------ | ---- |
        | token | true | string | 令牌 |
    :return:
        | 返回字段      | 字段类型 | 说明                   |
        | ------------- | -------- | ---------------------- |
        | status        | bool     | 返回结果状态true/false |
        | first_status  | list     | 第一次标定状态         |
        | second_status | list     | 第二次标定状态         |
    """

    logger.info('params:{}\n'.format(params))
    user = dbs.retrieve_mongos("user", {user_id: params['user']})
    user_id = user.user_id
    logger.info(' user {} \n'.format(user))
    first_status = [0 for i in range(8)]
    second_status = [0 for i in range(8)]
    train_data = CalibrationData.objects.filter(user_id=user_id) \
        .only('emo_category', 'is_used', 'similarity')
    train_data = dbs.retrieve_mongos("calibration_data", 
                             {user_id: user_id}, 
                              {'emo_category': 1, 'is_used': 1, 'similarity': 1})

    if user.model_times == 2:  # 建模完成
        logger.info('建模完成 \n')
        first_status = [1 for i in range(8)]
        second_status = [1 for i in range(8)]
    elif user.model_times == 1:  # 基本模型
        logger.info('基本模型 \n')
        first_status = [1 for i in range(8)]
        # 基本模型已存在，就找出第二次标定已成功的情绪
        second_data = [data for data in train_data if train_data['is_used'] == False and train_data['similarity'] == 0]
        for second in second_data:
            second_status[second.emo_category] = 1
    elif user.model_times == 0:  # 未建模
        logger.info('未建模 \n')
        # start_time = timezone.now()
        first_train = [data for data in train_data if train_data['similarity'] == 0]
        for first in first_train:
            first_status[first.emo_category] = 1
        # end_time = timezone.now()
        # print((end_time - start_time).total_seconds(), (end_time-second_time).total_seconds())

    return corr_response({'first_status': first_status, 'second_status': second_status})


def upload_new_predict_data(params):
    """
        ##### 参数
    | 参数        | 必选 | 类型     | 说明                 |
    | ----------- | ---- | -------- | -------------------- |
    | token       | true | string   | 令牌                 |
    | start_time  | true | datetime | 数据采集开始时间     |
    | end_time    | true | datetime | 数据采集结束时间     |
    | data        | true | list     | 数据(二维数组)                |
    | device_code | true | string   | 设备码长度17，0-9A-F |
    | step_count  | true | integer  | 步数                 |
    | tp          | true |  float   | 体温                 |
    ##### 返回参数

    | 返回字段     | 字段类型 | 说明                                      |
    | ------------ | -------- | ----------------------------------------- |
    | status       | bool     | 返回结果状态true/false                    |
    | emo          | str    | 情绪预测结果 |
    | hr           | float    | 心率                                      |
    | bp           | str      | 血压                                      |
    | study_status | float    | 学习状态                                  |
    """
    logger.info('params:{}\n'.format(params))
    user1, device_code, start_time, end_time, feature_data, pulse_data, tp, step_count = params['user'], params['device_code'], params['start_time'], params['end_time'], params['data'], params['pulse_wave'], params['tp'], params['step_count']
    tester_id = user1.user_id
    user = dbs.retrieve_mongos("user", {phone: device_code})
    if len(user) == 0 or device_code == '':
        user = user1
    else:
        user = user[0]
   
    # 获取情绪结果数据
    flag, emo_result = alg.get_emotion(pulse_data, type="ppg")
    # 0愤怒，1平静，2惊奇，3恐惧，4悲伤，5厌恶，6喜悦
    # 将惊奇换成愉悦(因为目前惊奇计算不准确)
    emo_result = [6 if i == 2 else i for i in emo_result]
    logger.info('flag: {} user: {} fm_ext: {} length: {} emo_result: {}'.format(flag, user.user_id, fm, len(fm), emo_result))
    # 获取血压
    try:
        bp_result = alg.predict_bp(pulse_data, user.user_id)
        bp_result = json.loads(bp_result)
        # logger.info('bp_result: {}'.format(bp_result))
        if bp_result['isSuccessful']:
            bp_result = str(int(float(bp_result['dbp']))) + '|' + str(int(float(bp_result['sbp'])))
        else:
            bp_result = '0|0'
    except Exception as ex:
        logger.error('new_predict_bp_result_error: {}', ex)
        bp_result = '0|0'
    # 获取心率
    try:
        hr_result = alg.get_hr(pulse_data, type="ppg")
        # hr_result会出现Infinity无穷大的情况
        hr_result = hr_result if hr_result <= 200 else 0
        # logger.info('fm: {} hr_result: {}'.format(fm, hr_result))
    except Exception as ex:
        logger.error('new_predict_hr_error: {}', ex)
        hr_result = 0
    # 获取tp，step_count，emo
    try:
        hr_result = float(hr_result)
        emo_result_str = ''.join([str(i) for i in emo_result]) if flag else ''
        user_id = user.user_id
        step_count = step_count if step_count != None else 0
        tp = tp if tp != None else 0.0
        # logger.info('tp: {}'.format(tp))
        # if len(emo_result) == 0:
        #     emo_l = [0, 0, 0, 0, 0, 0, 0]
        # else:
        #     emo_l = [emo_result.count(i) / len(emo_result) for i in range(0, 7)]
        emo_l = [emo_result.count(i) / len(emo_result) for i in range(0, 7)] if flag else []
        # 情绪效价
        emo_dict = alg.get_emotional_coordinates_emo(emo_l, hr_result)
        emo_dict = {'status': int(list(emo_dict.keys())[0]), 'list': list(emo_dict.values())[0]}
        # （安卓需求）如果status为-1，返回None
        if emo_dict['status'] == -1:
            emo_dict = None
        # 焦虑-压力-抑郁值
        anxiety = alg.get_anxiety_index(emo_l)
        pressure = alg.get_pressure_index(emo_l)
        depression = alg.get_depression_index(emo_l)
        health_min_data = {'user_id': user_id, 'tester_id': tester_id, 'start_time': start_time,
                           'end_time': end_time,
                           'emo_min': emo_l if flag else [],
                           'bp_min': bp_result, 'hr_min': hr_result, 'step_count': step_count, 'tp_min': tp, 'emo_str': emo_result_str}
        emo = health_min_data['emo_min']
        study_status = alg.get_study_status(emo, hr_result)
        health_min_data['study_status'] = study_status
        dbs.create_mongos('min_health', health_min_data)
    except Exception as ex:
        logger.info('new_predict_minhealth_save_error: {} \n'.format(ex))
        return err_response('1103', 'mongo保存数据失败')
    # 获取阳光指数
    hr_radio = 0 if hr_result >= 160 else 100 if hr_result <= 60 else 100 - (hr_result - 60)
    sunshine_index = round(0.2 * hr_radio + 0.8 * alg.get_sunshine_index(emo_l))
    # 获取学习指数
    try:
        study_status_data = dbs.retrieve_mongos("study_status", {'user_id': user_id, 'tester_id': tester_id})
        status = study_status_data.study_status
        rate = study_status_data.count / (study_status_data.count + 1)
        study_status_data.study_status = status * rate + study_status * (1 - rate)
        study_status_data.count += 1
    except Exception as ex:
        logger.info('new_predict_study_creat_error: {} \n'.format(ex))
    try:
        dbs.update_mongos("study_status",
                      {'user_id': user_id, 'tester_id': tester_id},
                      {'$set': study_status_data})
    except Exception as ex:
        logger.info('new_predict_study_save_error: {} \n'.format(ex))
        return err_response('1103', 'mongo保存数据失败')
    # 获取hrv
    # 只有传了原始脉搏波才开始计算心率变异性
    hrv_list = []
    if pulse_data:
        # 20秒一次的hrv就不保存了，只有在综合测量的1分钟一次的数据才保存
        if end_time - start_time > timedelta(seconds=50):
            # 心率变异性（用原始脉搏波）
            data_list = [{'start_time': start_time, 'end_time': end_time, 'data': pulse_data}]
            try:
                data_queue = dbs.retrieve_mongos("data_queue", {'phone': user.phone})
                data_queue_l = data_queue.data_list
                data_queue_append(data_queue_l, data_list[0], 'aaa')
                data_queue_l_len = len(data_queue_l)
                if data_queue_l_len == 6 or data_queue_l_len == 7:
                    if data_queue_l_len == 7:
                        data_queue_l.pop(0)
                        # with open(result,"a+") as fo:
                        #    fo.write("test zhy")
                    data_l = []
                    for i in data_queue_l:
                        data_l.extend(i['data'])
                    # fs=100hz
                    hrv_list = json.loads(heart_rate_var(data_l, 100))
                    # hrv_list[6]=9
                    hrv_list = [my_round(hrv_list[i], 2) for i in hrv_l] if hrv_list['isSuccessful'] else []
                try:
                    dbs.update_mongos("study_status",
                      {'user_id': user_id, 'tester_id': tester_id},
                      {'$set': data_queue_l})  # 用户数据队列更新后保存
                except Exception as ex:
                    logger.info('new_predict_hrv_error: {} \n'.format(ex))
                    return err_response('1103', 'mongo保存数据失败')
            except DoesNotExist:
                data_queue = DataQueue(phone=user.phone, data_list=data_list)
                try:
                    dbs.update_mongos("study_status",
                      {'user_id': user_id},
                      {'$set': data_list})  # 用户数据队列更新后保存
                except Exception as ex:
                    logger.info('new_predict_queue_save_error: {} \n'.format(ex))
                    return err_response('1103', 'DataQueue保存数据失败')
            timestamp = int(time.mktime(start_time.timetuple()))
            data = Data(phone=user.phone, start_time=start_time, end_time=end_time, timestamp=timestamp, hrv_list=hrv_list,
                        sbp=high_pressure(bp_result), dbp=low_pressure(bp_result), emo=emo)
            try:
                data.save()  # 保存这次数据血压，心率变异性等
            except Exception:
                return err_response('1103', 'mongo保存数据失败')

    result = {'emo': emo_result_str, 'emo_dict': emo_dict, 'hr': hr_result, 'bp': bp_result,
              'dbp': low_pressure(bp_result), 'sbp': high_pressure(bp_result), 'anxiety': anxiety,
              'pressure': pressure, 'depression': depression, 'study_status': study_status,
              'sunshine_index': sunshine_index, 'hrv_list': hrv_list}
    return corr_response(result)


def data_queue_append(data_queue, item, target):
    if len(data_queue) == 0:
        data_queue.append(item)
        return
    if target == 'ars':
        # hours,days,weeks
        if item['start_time'] - data_queue[-1]['end_time'] <= timedelta(weeks=52):
            data_queue.clear()
            data_queue.append(item)
        else:
            data_queue.append(item)
    else:
        if data_queue[-1]['end_time'] != item['start_time']:
            data_queue.clear()
            data_queue.append(item)
        else:
            data_queue.append(item)

