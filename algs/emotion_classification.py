# -*- coding: utf-8 -*-
"""
Algorithm of emotion calssification.

Created on 07/12/21 11:40 AM

@file: emotion_classification.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""

## includes all the function needed in ppg emotion model training and predict
import numpy as np
#import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm
from algs.keypoints_extraction import extract_keypoints
from algs.preprocessing import median_filter

#@fn_timer
def feature_transformer_temperal(feature):
    count = 0
    FM = list()
    for i in range(0, len(feature) - 1):
        featuretmp = feature[i]
        if sum([tmp == 'inf' for tmp in featuretmp]) == 0:
            count = count + 1
            if feature[i + 1][4] - feature[i][4] > 150 or feature[i + 1][4] - feature[i][4] < 30:
                count = count - 1
                continue
            Fmtmp = list()
            ##pluse rate between the main peaks 0
            Fmtmp.append((feature[i + 1][2] - feature[i][2]) / 100)
            ##time of the rising part 1
            Fmtmp.append((feature[i][2] - feature[i][0]) / 100)
            ##time of the descending branch 2
            Fmtmp.append((feature[i][8] - feature[i][2]) / 100)
            ##time between the main peak and the mid peak 3
            Fmtmp.append((feature[i][6] - feature[i][2]) / 100)
            ##time between tha main peak and the mid valley 4
            Fmtmp.append((feature[i][4] - feature[i][2]) / 100)
            ## 5
            Fmtmp.append((feature[i][4] - feature[i][0]) / 100)
            ## 6
            Fmtmp.append((feature[i][6] - feature[i][0]) / 100)
            ## 7
            Fmtmp.append((feature[i][8] - feature[i][0]) / 100)

            bottom_of_main = feature[i][1] + (feature[i][9] - feature[i][1]) / (feature[i][8] - feature[i][0]) * (
                        feature[i][2] - feature[i][0])
            bottom_of_midvalley = feature[i][1] + (feature[i][9] - feature[i][1]) / (feature[i][8] - feature[i][0]) * (
                        feature[i][4] - feature[i][0])
            bottom_of_midpeak = feature[i][1] + (feature[i][9] - feature[i][1]) / (feature[i][8] - feature[i][0]) * (
                        feature[i][6] - feature[i][0])
            ## 8
            Fmtmp.append(feature[i][3] - bottom_of_main)
            ## 9
            Fmtmp.append(feature[i][5] - bottom_of_midvalley)
            ## 10
            Fmtmp.append(feature[i][7] - bottom_of_midpeak)
            ## 11
            Fmtmp.append(Fmtmp[9] / Fmtmp[8])
            ## 12
            Fmtmp.append(Fmtmp[10] / Fmtmp[8])
            ## 13
            Fmtmp.append(Fmtmp[1] / Fmtmp[0])
            ## 14
            Fmtmp.append(Fmtmp[2] / Fmtmp[0])
            ## 15
            Fmtmp.append(Fmtmp[5] / Fmtmp[0])
            ## 16
            Fmtmp.append(Fmtmp[6] / Fmtmp[0])
            FM.append(Fmtmp)
    # delete abnormal data
    #    print('original time feature:',len(FM))
    if len(FM) == 0:
        return FM

    tmp = np.array(FM)
    tmpstd = [np.std(tmp[:, i]) for i in range(0, 12)]
    tmpmean = [np.mean(tmp[:, i]) for i in range(0, 12)]
    tmp1 = [tmpmean[i] - 3 * tmpstd[i] for i in range(0, 12)]
    tmp2 = [tmpmean[i] + 3 * tmpstd[i] for i in range(0, 12)]
    Fm = list()
    for i in range(0, len(FM)):
        if sum([tmp[i, j] < tmp1[j] for j in range(0, 12)]) == 0 and sum(
                [tmp[i, j] > tmp2[j] for j in range(0, 12)]) == 0:
            Fm.append(FM[i])

    return Fm


# @fn_timer
def heart_rate(Fm):
    Fm = np.array(Fm)
    RR = np.mean(Fm[:, 0])
    return round(60 / RR)

def temperal_feature_extender(Fm):
    Fm_Ext = list()
    for i in range(0,len(Fm)):
        tmp = list()
        tmp.append(Fm[i][0])
        tmp.append(Fm[i][1])
        tmp.append(Fm[i][2])
        tmp.append(Fm[i][3])
        tmp.append(Fm[i][4])
        tmp.append(Fm[i][4]+Fm[i][1])
        tmp.append(Fm[i][3]+Fm[i][1])
        tmp.append(Fm[i][2]+Fm[i][1])
        tmp.append(Fm[i][5])
        tmp.append(Fm[i][6])
        tmp.append(Fm[i][7])
        tmp.append(Fm[i][6]/Fm[i][5])
        tmp.append(Fm[i][7]/Fm[i][5])
        tmp.append(Fm[i][1]/Fm[i][0])
        tmp.append(Fm[i][2]/Fm[i][0])
        tmp.append(tmp[5]/tmp[0])
        tmp.append(tmp[6]/tmp[0])
        Fm_Ext.append(tmp)
    return Fm_Ext

# @fn_timer
def feature_transformer_frequential(Fm, emo):
    #    print('final time feature:',len(Fm))
    if len(Fm) <= 2:
        FmFre = list()
        emol = list()
        return FmFre, emol
    # transform temperal features into frequential features
    i = 25
    ii = 5
    FM = np.array(Fm)
    FmFre = list()
    RRtmp = FM[:, 0]  # [60/(x+0.001) for x in FM[:,0]]
    hRtmp = [x for x in FM[:, 0]]
    PRtmp = FM[:, 1]
    RTtmp = FM[:, 16]
    time = list()
    time.append(0)
    for j in range(0, len(RRtmp) - 1):
        time.append(time[j] + round(RRtmp[j + 1] * 10))
    # print(int(time[len(time)-1]))
    timeall = np.linspace(0, int(time[len(time) - 1]), int(time[len(time) - 1]) + 1)
    f1 = interp1d(time, hRtmp, kind='linear')
    f2 = interp1d(time, PRtmp, kind='linear')
    f3 = interp1d(time, RTtmp, kind='linear')
    RRtmpall = f1(timeall)
    PRtmpall = f2(timeall)
    RTtmpall = f3(timeall)
    RRtmpall_med = RRtmpall - median_filter(RRtmpall, 50)
    PRtmpall_med = PRtmpall - median_filter(PRtmpall, 50)
    RTtmpall_med = RTtmpall - median_filter(RTtmpall, 50)

    while i + 400 <= len(RRtmpall) - 50:
        RRtmpwin = RRtmpall_med[i:i + 400]
        PRtmpwin = PRtmpall_med[i:i + 400]
        RTtmpwin = RTtmpall_med[i:i + 400]
        FreRR = abs(np.fft.fft(RRtmpwin))
        FrePR = abs(np.fft.fft(PRtmpwin))
        FreRT = abs(np.fft.fft(RTtmpwin))
        FreRR = FreRR[0:20]
        FrePR = FrePR[0:20]
        FreRT = FreRT[0:20]

        FreRR = FreRR / np.sqrt(sum(np.square(FreRR)))
        FrePR = FrePR / np.sqrt(sum(np.square(FrePR)))
        FreRT = FreRT / np.sqrt(sum(np.square(FreRT)))

        Fmtmp = list()
        FreRR = FreRR.tolist()
        FreRR.append(np.mean(RRtmpall[i:i + 400]))
        Fmtmp.append(FreRR)
        FrePR = FrePR.tolist()
        FrePR.append(np.mean(PRtmpall[i:i + 400]))
        Fmtmp.append(FrePR)
        FreRT = FreRT.tolist()
        FreRT.append(np.mean(RTtmpall[i:i + 400]))
        Fmtmp.append(FreRT)

        FmFre.append(Fmtmp)
        i = i + 100
    #        ii = round(i/10)
    emol = [emo for i in range(0, len(FmFre))]
    return FmFre, emol


# @fn_timer
# emo is given according to the emotion type of the video when training, and for predicting, emo can be given as 0
def feature_generator(data, emo):
    feature = extract_keypoints(data)
    Fm = feature_transformer_temperal(feature)
    FmFre, emol = feature_transformer_frequential(Fm, emo)
    #    print(FmFre)
    return FmFre, emol


# @fn_timer
# for example
def feature_generator_application(data1, data2):
    feature = extract_keypoints(data1)
    Fm1 = feature_transformer_temperal(feature)
    feature = extract_keypoints(data2)
    Fm2 = feature_transformer_temperal(feature)
    Fm = Fm1 + Fm2
    FmFre, emol = feature_transformer_frequential(Fm, 0)
    return FmFre


def emotion_classifier_trainers(FmFre, Emo, clfn="svm", ispca=True, islda=True):
    Fm = np.array(FmFre)
    Emo = np.array(Emo)
    Fm1 = Fm[:, 0, :]
    Fm2 = Fm[:, 1, :]
    Fm3 = Fm[:, 2, :]

    if ispca:
        pca1 = PCA(n_components=10)
        pca1.fit(Fm1)
        Fm1_pca = pca1.transform(Fm1)
        pca2 = PCA(n_components=10)
        pca2.fit(Fm2)
        Fm2_pca = pca2.transform(Fm2)
        pca3 = PCA(n_components=10)
        pca3.fit(Fm3)
        Fm3_pca = pca3.transform(Fm3)
        pca = list()
        pca.append(pca1)
        pca.append(pca2)
        pca.append(pca3)
    else:
        Fm1_pca = Fm1
        Fm2_pca = Fm2
        Fm3_pca = Fm3
        pca = None

    Fm_pca = np.hstack((Fm1_pca, Fm2_pca, Fm3_pca))
    if islda:
        lda = LinearDiscriminantAnalysis(n_components=5)
        lda.fit(Fm_pca, Emo)
        Fm_pca_lda = lda.transform(Fm_pca)
    else:
        Fm_pca_lda = Fm_pca
        lda = None

    clfs = list()
    for i in range(0, 7):
        for j in range(0, 7):
            if i == j:
                continue
            label1 = [k for k in range(0, len(Emo)) if Emo[k] == i]
            label2 = [k for k in range(0, len(Emo)) if Emo[k] == j]
            label = label1 + label2
            train_data = Fm_pca_lda[label, :]
            train_label = Emo[label]
            if len(label1) == 0 or len(label2) == 0:
                clfs.append(0)
            else:
                if clfn == "svm":
                    clf = SVC(gamma='scale')
                elif clfn == "knn":
                    clf = KNeighborsClassifier()
                elif clfn == "lda":
                    clf = LinearDiscriminantAnalysis()
                elif clfn == "cart":
                    clf = DecisionTreeClassifier()
                elif clfn == "nb":
                    clf = GaussianNB()
                elif clfn == "lr":
                    clf = LogisticRegression()
                else:
                    clf = None
                if clf is None:
                    clfs.append(0)
                else:
                    clf.fit(train_data, train_label)
                    clfs.append(clf)

    # predict
    count = 0
    predlabel = np.zeros((len(Emo), 7), dtype=float)
    for i in range(0, 7):
        for j in range(0, 7):
            if i == j:
                continue
            clf = clfs[count]
            count = count + 1
            if clf == 0:
                continue
            emopred = clf.predict(Fm_pca_lda)
            #            print(len(emopred))
            predlabel[emopred == i, i] = predlabel[emopred == i, i] + 1
    predlabel = [np.argmax(predlabel[i, :]) for i in range(0, len(Emo))]
    # print('total training accuracy', (predlabel == Emo).sum() / len(Emo))

    return pca, lda, clfs



# function of training
# @fn_timer
def emotion_classifier_trainer(FmFre, Emo):
    Fm = np.array(FmFre)
    Emo = np.array(Emo)
    Fm1 = Fm[:, 0, :]
    Fm2 = Fm[:, 1, :]
    Fm3 = Fm[:, 2, :]

    pca1 = PCA(n_components=10)
    pca1.fit(Fm1)
    Fm1_pca = pca1.transform(Fm1)
    pca2 = PCA(n_components=10)
    pca2.fit(Fm2)
    Fm2_pca = pca2.transform(Fm2)
    pca3 = PCA(n_components=10)
    pca3.fit(Fm3)
    Fm3_pca = pca3.transform(Fm3)

    pca = list()
    pca.append(pca1)
    pca.append(pca2)
    pca.append(pca3)

    Fm_pca = np.hstack((Fm1_pca, Fm2_pca, Fm3_pca))
    lda = LinearDiscriminantAnalysis(n_components=5)
    lda.fit(Fm_pca, Emo)
    Fm_pca_lda = lda.transform(Fm_pca)

    clf = list()
    for i in range(0, 7):
        for j in range(0, 7):
            if i == j:
                continue
            label1 = [k for k in range(0, len(Emo)) if Emo[k] == i]
            label2 = [k for k in range(0, len(Emo)) if Emo[k] == j]
            label = label1 + label2
            train_data = Fm_pca_lda[label, :]
            train_label = Emo[label]
            if len(label1) == 0 or len(label2) == 0:
                clf.append(0)
            else:
                clfsvm = svm.SVC(gamma='scale')
                clfsvm.fit(train_data, train_label)
                clf.append(clfsvm)
    # predict
    count = 0
    predlabel = np.zeros((len(Emo), 7), dtype=float)
    for i in range(0, 7):
        for j in range(0, 7):
            if i == j:
                continue
            clfsvm = clf[count]
            count = count + 1
            if clfsvm == 0:
                continue
            emopred = clfsvm.predict(Fm_pca_lda)
            #            print(len(emopred))
            predlabel[emopred == i, i] = predlabel[emopred == i, i] + 1
    predlabel = [np.argmax(predlabel[i, :]) for i in range(0, len(Emo))]
    # print('total training accuracy', (predlabel == Emo).sum() / len(Emo))

    return pca, lda, clf


# @fn_timer
def emotion_classifer(FmFre, pca, lda, clf):
    Fm = np.array(FmFre)
    Fm1 = Fm[:, 0, :]
    Fm2 = Fm[:, 1, :]
    Fm3 = Fm[:, 2, :]

    pca1 = pca[0]
    Fm1_pca = pca1.transform(Fm1)
    pca2 = pca[1]
    Fm2_pca = pca2.transform(Fm2)
    pca3 = pca[2]
    Fm3_pca = pca3.transform(Fm3)

    Fm_pca = np.hstack((Fm1_pca, Fm2_pca, Fm3_pca))
    Fm_pca_lda = lda.transform(Fm_pca)

    # predict
    count = 0
    predlabel = np.zeros((len(FmFre), 7), dtype=float)
    for i in range(0, 7):
        for j in range(0, 7):
            if i == j:
                continue
            clfsvm = clf[count]
            count = count + 1
            if clfsvm == 0:
                continue
            emopred = clfsvm.predict(Fm_pca_lda)
            #            print(len(emopred))
            predlabel[emopred == i, i] = predlabel[emopred == i, i] + 1

    predlabel = [np.argmax(predlabel[i, :]) for i in range(0, len(FmFre))]

    return predlabel


# trans_fun_label is the 7X1 np.array
# MAP is the 7x7 np.array
def emo_transfer_function(MAP):
    MAP = np.array(MAP)
    M = np.array(MAP.shape)
    M = MAP + 0
    trans_fun_label = np.array([x for x in range(0, 7)])
    label_i, label_j = np.where(M == np.max(M))
    label_i = label_i[0]
    label_j = label_j[0]
    max_d_cos = M[label_i, label_j]
    M[label_i, :] = np.min(M)
    M[:, label_j] = np.min(M)

    while 1:
        tmp2 = trans_fun_label[label_j]
        trans_fun_label[trans_fun_label == label_i] = tmp2
        trans_fun_label[label_j] = label_i

        label_i, label_j = np.where(M == np.max(M))
        label_i = label_i[0]
        label_j = label_j[0]

        max_d_cos = M[label_i, label_j]
        M[label_i, :] = np.min(M)
        M[:, label_j] = np.min(M)

        if max_d_cos == np.min(M):
            break

    return trans_fun_label


def emo_transfer(P, trans_fun_label):
    emo = np.zeros(len(P))
    for i in range(0, len(P)):
        emo[i] = trans_fun_label[P[i]]

    return emo


def general_emotion_classifier(FmFre, lda, clf):
    global pca_me
    return emotion_classifer(FmFre, pca_me, lda, clf)
    # if len(FmFre) == 0:
    #     return list()
    # Fm = np.array(FmFre)
    # # Fm = FmFre
    # Fm1 = Fm[:, 0, 0:-1]
    # Fm2 = Fm[:, 1, 0:-1]
    # Fm3 = Fm[:, 2, 0:-1]
    # Fm = np.hstack((Fm1, Fm2, Fm3))
    # Fm_lda = lda.transform(Fm)
    # count = 0
    # predlabel = np.zeros((len(Fm), 7), dtype=float)
    # for i in range(0, 7):
    #     for j in range(0, 7):
    #         if i == j:
    #             continue
    #         clfsvm = clf[count]
    #         count = count + 1
    #         if clfsvm == 0:
    #             continue
    #         emopred = clfsvm.predict(Fm_lda)
    #         for k in range(0, 7):
    #             predlabel[emopred == k, k] = predlabel[emopred == k, k] + 1
    # P_label = [np.argmax(predlabel[i, :]) for i in range(0, len(Fm))]
    # return P_label


# @fn_timer
def is_emotion_correct(FmFre, pca, lda, clf, emol):
    predlabel = emotion_classifer(FmFre, pca, lda, clf)
    rate = predlabel.count(emol) / len(predlabel)
    return rate > 0.25


if __name__ == "__main__":
    # ## for test
    # traindir = './new_data_100hz'
    # #    traindir = 'H:/文档/软件下载/微信/WeChat Files/wxid_ovy5w8b4phan22/FileStorage/File/2019-08'
    # testdir = './new_data_100hz'
    # trainlist = os.listdir(traindir)
    # FmFreall = list()
    # Emo = list()
    # for i in range(0, len(trainlist)):
    #     path = os.path.join(traindir, trainlist[i])
    #     if os.path.isfile(path):
    #         emostr = trainlist[i][len(trainlist[i]) - 6:len(trainlist[i]) - 4]
    #         #            print(trainlist[i])
    #         emo = 0
    #         if emostr == '平静':
    #             emo = 0
    #         if emostr == '惊奇':
    #             emo = 1
    #         if emostr == '愉悦':
    #             emo = 2
    #         if emostr == '悲伤':
    #             emo = 3
    #         if emostr == '愤怒':
    #             emo = 4
    #         if emostr == '恐惧':
    #             emo = 5
    #         if emostr == '厌恶':
    #             emo = 6

    #         data = read_data(path)
    #         data = str_hex2dec(data)

    #         ## test
    #         FmFre, emol = feature_generator(data, emo)
    #         if i == 0:
    #             feature1 = extract_keypoints(data)
    #         if i == 1:
    #             feature2 = extract_keypoints(data)
    #         FmFreall = FmFreall + FmFre
    #         Emo = Emo + emol
            
    # server test
    # logger = logging.getLogger('root')
    # passwd = parse.quote("dongni$mongodb@211022")
    # client = pymongo.MongoClient('mongodb://{}:{}@{}:{}/{}'.format("algorith_root",passwd,"dds-2ze1843ca7f0dfc4-pub.mongodb.rds.aliyuncs.com","3717","algorithfile"))
	
    # lda, clf = load_general_classifer('./models/ppg_model_jay')
    # # pca, lda, clf = load_classifer('./ppg_model_new1')
    print("success!")

    # print("LDA:", lda)
    # print("clf:", clf)
    # print("pca_me:", pca_me)
    # file = "./test_row_data"
    # tmp_data = read_data(file)
    # FmFre, emol = feature_generator(tmp_data, 0)
    
    # p_label = general_emotion_classifier(FmFre, lda, clf)
    # trans_fun_label = emo_transfer_function(np.zeros((7, 7)))
    # emo_result = emo_transfer(p_label, trans_fun_label)
    # emo_result = list(int(i) for i in emo_result)
    # print("result:", emo_result)
    

#    pca,lda,clf = emotion_classifier_trainer(np.array(FmFreall),np.array(Emo))
#    save_classifer(pca,lda,clf,'./ppg_model')
#    pca,lda,clf = load_classifer('./ppg_model')
#    # test
#    testlist = os.listdir(testdir)
#    FmFreall_test = list()
#    Emo_test = list()
#    for i in range(0,len(testlist)):
#        path = os.path.join(testdir,testlist[i])
#        if os.path.isfile(path):
#            emostr = testlist[i][len(testlist[i])-6:len(testlist[i])-4]
#            if emostr=='平静':
#                emo = 0
#            if emostr=='惊奇':
#                emo = 1
#            if emostr=='愉悦':
#                emo = 2
#            if emostr=='悲伤':
#                emo = 3
#            if emostr=='愤怒':
#                emo = 4
#            if emostr=='恐惧':
#                emo = 5
#            if emostr=='厌恶':
#                emo = 6
#
#            data = read_data(path)
#            data = str_hex2dec(data)
#            ## tests
#            FmFre,emol = feature_generator(data,emo)
#
#            FmFreall_test = FmFreall_test + FmFre
#            Emo_test = Emo_test + emol
#
#    predlabel = emotion_classifer(np.array(FmFreall_test),pca,lda,clf)
#    print('total testing accuracy',(predlabel==np.array(Emo_test)).sum()/len(Emo_test))
