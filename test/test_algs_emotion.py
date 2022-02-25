# -*- coding: utf-8 -*-
"""
Algorithm of emotion calssification.

Created on 01/03/22 22:40 PM

@file: tes_algs_emotion.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""


import os
import re
import random
import pandas as pd
import numpy as np
import algs.keypoints_extraction as kpe
import algs.emotion_classification as emo_app
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utils.dbs_utils import save_classifer


def load_files(data_dir, pattern=""):
    """
    Load files filtered by pattern from database folder.
    
    Parameters
    ----------
    data_dir : string
        Root directory of database.
    pattern : string, optional
        A regular expression pattern being matched.

    Returns
    -------
        file_list : list
            list of file name filtered by pattern.
    """
    file_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            if re.search(pattern, path) is not None:
                file_list.append(path)
    return file_list


# read raw data or feature data in time domain from txt file.
# the file save str data looking like list.
def read_data(file_name):
    """
    Read data from a file including direct data expressions.
    
    Parameters
    ----------
    file_name : string
        Path of file been read.

    Returns
    -------
        data : list
            list of data read from the data file.
    """
    data = ""
    with open(file_name, "r") as file:
        line = file.readline()
        while line:
            data = data + line
            line = file.readline()
    if re.search("^\[\[.*\]\]$", data) is not None:
        return eval(data)
    else:
        return []

    ##### get all data in <newtrain>
def generate_datasets_all(dbdir):
    angry_data = []
    calm_data = []
    surprise_data = []
    fear_data = []
    sorrow_data = []   
    disgust_data = []   
    joy_data = []
    
    angry_files = load_files("./newtrain", "愤怒.*Fm_result")
    calm_files = load_files("./newtrain", "平静.*Fm_result")
    surprise_files = load_files("./newtrain", "惊奇.*Fm_result")
    fear_files = load_files("./newtrain", "恐惧.*Fm_result")
    sorrow_files = load_files("./newtrain", "悲伤.*Fm_result|伤心.*Fm_result")
    disgust_files = load_files("./newtrain", "厌恶.*Fm_result")
    joy_files = load_files("./newtrain", "愉悦.*Fm_result")
    
    for file in angry_files:
        tmp_data = read_data(file)
        angry_data += tmp_data
    
    for file in calm_files:
        tmp_data = read_data(file)
        calm_data += tmp_data
    
    for file in surprise_files:
        tmp_data = read_data(file)
        surprise_data += tmp_data
        
    for file in fear_files:
        tmp_data = read_data(file)
        fear_data += tmp_data
    
    for file in sorrow_files:
        tmp_data = read_data(file)
        sorrow_data += tmp_data   
    
    for file in disgust_files:
        tmp_data = read_data(file)
        disgust_data += tmp_data
        
    for file in joy_files:
        tmp_data = read_data(file)
        joy_data += tmp_data
    
    data_fm = [angry_data, calm_data, surprise_data, fear_data, sorrow_data, disgust_data, joy_data]
      
    data_X = []
    data_y = []
    
    for i in range(7):
        data_FmFre, data_emol = emo_app.feature_transformer_frequential(data_fm[i], i)
        data_X += data_FmFre
        data_y += data_emol

    return data_X, data_y

##### filtered data by 校验数据.xlsx
def generate_datasets_excel(dbdir):
    df = pd.read_excel("./newtrain/校验数据.xlsx")
    filter_folders = list(df.iloc[:, 2].values)
    filter_files = []
    for folder in filter_folders:
        filter_files += load_files(dbdir, str(folder)+".*Fm_result")
    
    angry_data = []
    calm_data = []
    surprise_data = []
    fear_data = []
    sorrow_data = []
    disgust_data = []
    joy_data = []
    
    for file in filter_files:
        tmp_data = read_data(file)
        if "愤怒" in file:
            angry_data += tmp_data
        if "平静" in file:
            calm_data += tmp_data
        if "惊奇" in file:
            surprise_data += tmp_data       
        if "恐惧" in file:
            fear_data += tmp_data
        if "悲伤" in file or "伤心" in file:
            sorrow_data += tmp_data        
        if "厌恶" in file:
            disgust_data += tmp_data
        if "愉悦" in file:
            joy_data += tmp_data
               
    data_fm = [angry_data, calm_data, surprise_data, fear_data, sorrow_data, disgust_data, joy_data] 
    data_X = []
    data_y = []    
    for i in range(7):
        data_FmFre, data_emol = emo_app.feature_transformer_frequential(data_fm[i], i)
        data_X += data_FmFre
        data_y += data_emol

    return data_X, data_y


##### filtered data in 2019
def generate_datasets_2019(dbdir):
    filter_files = load_files(dbdir,  "2019"+".*Fm_result")
    
    angry_data = []
    calm_data = []
    surprise_data = []
    fear_data = []
    sorrow_data = []   
    disgust_data = []   
    joy_data = []
    
    for file in filter_files:
        tmp_data = read_data(file)
        if "愤怒" in file:
            angry_data += tmp_data
        if "平静" in file:
            calm_data += tmp_data
        if "惊奇" in file:
            surprise_data += tmp_data       
        if "恐惧" in file:
            fear_data += tmp_data
        if "悲伤" in file:
            sorrow_data += tmp_data        
        if "厌恶" in file:
            disgust_data += tmp_data
        if "愉悦" in file:
            joy_data += tmp_data

    data_fm = [angry_data, calm_data, surprise_data, fear_data, sorrow_data, disgust_data, joy_data]
    data_X = []
    data_y = [] 
    for i in range(7):
        data_FmFre, data_emol = emo_app.feature_transformer_frequential(data_fm[i], i)
        data_X += data_FmFre
        data_y += data_emol

    return data_X, data_y

def save_datasets(filename, data_X, data_y):
    np.savez(filename, data_X=data_X, data_y=data_y)

def load_datasets(filename):
    datasets = np.load(filename)
    return datasets['data_X'], datasets['data_y']

def train_classifier(train_data, train_label):
    pca, lda, clf = emo_app.emotion_classifier_trainers(np.array(train_data), np.array(train_label))
    save_classifer(pca, lda, clf, './models/model_excel_aver_6/')
    return dict({'pca':pca, 'lda':lda, 'clf':clf})

def test_classifier(model, test_data, test_label):
    pca = model['pca']
    lda = model['lda']
    clf = model['clf']
    predlabel = emo_app.emotion_classifer(np.array(test_data), pca, lda, clf)
    print('total testing accuracy', (predlabel==np.array(test_label)).sum()/len(test_label))
    
    test_result = {}
    for i in set(test_label):
        test_result[i] = test_label.tolist().count(i)
    print('test_emo_label:', test_result)
    print('number of test emtion labels', len(test_label))
    
    pred_result = {}
    for i in set(predlabel):
        pred_result[i] = predlabel.count(i)
    print('predlabel:', pred_result)
    print('number of predicted', len(predlabel))
    
    cfm = confusion_matrix(test_label, predlabel, labels=[0,1,2,3,4,5,6])
    n = len(cfm)
    text_labels = ['angry:', 'calm:', 'surprise:', 'fear:', 'sorrow:', 'disgust:', 'joy:']
    for i in range(len(cfm[0])):
        rowsum, colsum = sum(cfm[i]), sum(cfm[r][i] for r in range(n))
        try:
            print(text_labels[i],'precision: %s' % (cfm[i][i]/float(colsum)), 'recall: %s' % (cfm[i][i]/float(rowsum)))
        except ZeroDivisionError:
            print('precision: %s' % 0, 'recall: %s' % 0)



if __name__ == '__main__':
    #data_X, data_y = generate_datasets_excel('./newtrain')
    #save_datasets("./datasets/flt_ds_excel.npz", data_X, data_y)
    #print("save datasets successfully!")
    #BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_X, data_y = load_datasets("./datasets/flt_ds_excel.npz")
    y, y_num = np.unique(data_y, return_counts=True)
    print('numbers of samples in each class: \n', y, '\n', y_num)

    idx2 = np.where(y==4)
    y = np.delete(y, idx2[0])
    y_num = np.delete(y_num, idx2[0])

    cls_len = np.min(y_num)

    #cls_len_min = np.argmin(y_num)
    #y_num_new = np.delete(y_num, cls_len_min)
    #cls_len = np.min(y_num_new)

    data_X_adj = data_X
    data_y_adj = data_y

    #data_X_adj = None
    #data_y_adj = []
    #for i, label in enumerate(y):
    #    data = data_X[data_y==label]
    #    if y_num[i] < cls_len:
    #        use_cls_len = y_num[i]
    #    else:
    #        use_cls_len = cls_len
    #    if data_X_adj is None:
    #        data_X_adj = data[np.random.choice(data.shape[0], use_cls_len, False)]
    #    else:
    #        data_X_adj = np.r_[data_X_adj,  data[np.random.choice(data.shape[0], use_cls_len, False)]]
    #    data_y_adj += [label]*use_cls_len
    #data_y_adj = np.array(data_y_adj)
    #y, y_num = np.unique(data_y_adj, return_counts=True)
    #print('numbers of adjust samples in each class: \n', y, '\n', y_num)


    X_train, X_test, y_train, y_test = train_test_split(data_X_adj, data_y_adj, test_size=0.3, random_state=12)
    y, y_num = np.unique(y_train, return_counts=True)
    print('numbers of adjust samples in train sets: \n', y, '\n', y_num)
    y, y_num = np.unique(y_test, return_counts=True)
    print('numbers of adjust samples in test sets: \n', y, '\n', y_num)
    model2 = train_classifier(X_train, y_train)
    test_classifier(model2, X_test, y_test)


