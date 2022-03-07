import os

import numpy as np
import traceback,json,logging

from algs.extract_attributes_kpts import extract_attributes
from algs.auxiliaries_lyd import merge_samples_atts
from algs.auxiliaries_lyd import ATT, N_INCEPT_CLF, HID_CLF, IN, HID_DNN, MID, \
            N_INCEPT_NRM, N_INCEPT_HI, MIDe, NORM
# from utils.isppg import isPPGWave, plot_ppg_kpts


# import pdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT)


# 神经网络模型
class Inception(nn.Module):
    def __init__(self, n_in, n_mid):
        super(Inception, self).__init__()
        self.extend_add = nn.Sequential(
            nn.Linear(n_in, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_in),
        )

    def forward(self, x):
        x = nn.functional.relu(self.extend_add(x) + x)
        return x


class DNN(nn.Module):
    def __init__(self, n_in, n_mid, n_hidden, n_inception):
        super(DNN, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU()
        )

        self.inceptions = nn.Sequential(
            *[Inception(n_hidden, n_mid) for _ in range(n_inception)]
        )

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_hidden, 2)
        )
        
    def do_incept(self, x):
        x = self.input(x)
        x = self.inceptions(x)
        return x

    def do_output(self, x):
        x = self.output(x)
        return x

    def forward(self, x):
        x = self.do_incept(x)
        x = self.do_output(x)
        return x


class ExtendNN(nn.Module):
    def __init__(self, n_in, n_mid, dnn):
        super(ExtendNN, self).__init__()
        # load atts from mass atts data
        self.inception_out = nn.Sequential(
            *[Inception(n_in, n_mid) for _ in range(2)],
            nn.Linear(n_in, 2)
        )

        self.dnn = dnn

    def train(self):
        self.inception_out.train()

    def parameters(self):
        return self.inception_out.parameters()

    def forward(self, x):
        x = self.dnn.do_incept(x)
        x = self.inception_out(x)
        return x

    def save(self, save_loc):
        self.dnn = None
        torch.save(self.state_dict(), save_loc)


class MyDataset():
    def __init__(self, atts, bps):
        super(MyDataset, self).__init__()
        self.atts = atts
        self.bps = bps

    def __getitem__(self, index):
        return torch.Tensor(self.atts[index]), torch.Tensor(self.bps[index])

    def __len__(self):
        return self.atts.shape[0]


current_path = os.path.dirname(__file__)
clf_loc = os.path.join(current_path, "support/clf_kpts.pkl")
highpr_loc = os.path.join(current_path, "support/highpr_model_kpts.pkl")
normal_loc = os.path.join(current_path, "support/normal_model_kpts.pkl")
dims_std_normal = os.path.join(current_path, "support/std_dims_normal.txt")
dims_std_highpr = os.path.join(current_path, "support/std_dims_highpr.txt")
dims_mean_normal = os.path.join(current_path, "support/mean_dims_normal.txt")
dims_mean_highpr = os.path.join(current_path, "support/mean_dims_highpr.txt")
mass_ppg_loc = os.path.join(current_path, "support/mass_ppg_kpts.txt")
mass_bp_loc = os.path.join(current_path, "support/mass_bp_kpts.txt")


def feed_np_get_np(module, x):
    x = torch.Tensor(x)
    x = module(x)
    x = x.detach().numpy()
    return x

LR = 1e-4
BS = 50
EPO = 100
lossF = torch.nn.MSELoss()

# [用于核心运算的代码]: atts =[model]=> bp
def update_model_from_atts_bp(atts_private, bps_private, extend_loc):
    assert len(atts_private) == len(bps_private)    # 2021/11/16, li yongshuai
    # classify
    clf = DNN(ATT, MID, HID_CLF, N_INCEPT_CLF)
    clf.load_state_dict(torch.load(clf_loc))
    groups = np.argmax(feed_np_get_np(clf, atts_private), axis=1)

    # Both groups can update models
    # # fitting
    # if group == 0:
    #     return
    atts_private = atts_private[:, :IN]

    # 初始化dnn和enn, 2021/11/16, li yongshuai    
    if groups[-1] == 0:  # normal
        dnn = DNN(IN, MID, HID_DNN, N_INCEPT_NRM)
        dnn.load_state_dict(torch.load(normal_loc))

        atts_private = atts_private[groups == 0]
        bps_private = bps_private[groups == 0]

        std_dims = np.loadtxt(dims_std_normal)
        mean_dims = np.loadtxt(dims_mean_normal)
    else:  # highpr
        dnn = DNN(IN, MID, HID_DNN, N_INCEPT_HI)
        dnn.load_state_dict(torch.load(highpr_loc))

        atts_private = atts_private[groups == 1]
        bps_private = bps_private[groups == 1]

        std_dims = np.loadtxt(dims_std_highpr)
        mean_dims = np.loadtxt(dims_mean_highpr)
    enn = ExtendNN(HID_DNN, MIDe, None)
    if os.path.isfile(extend_loc):
        enn.load_state_dict(torch.load(extend_loc))
    enn.dnn = dnn

    atts_mass = np.loadtxt(mass_ppg_loc)
    bps_mass = np.loadtxt(mass_bp_loc)

    # get atts_[0,1,...,IN-1], concatenate, normalize
    atts = (np.concatenate([atts_mass, atts_private], axis=0) - mean_dims) / std_dims
    bps = np.concatenate([bps_mass, bps_private], axis=0) / NORM

    # assigning sampling weights
    num_samples = len(atts_mass) + len(atts_private)
    cards = [len(atts_mass), len(atts_private)]
    label_weights = [1./card for card in cards]
    weights = np.concatenate(
                [np.ones(cards[0]) * label_weights[0],
                 np.ones(cards[1]) * label_weights[1]])
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples)

    dataset = MyDataset(atts, bps)
    dataLoader = DataLoader(dataset, batch_size=BS, sampler=sampler)
    # TODO: training code
    optimizer = torch.optim.Adam(enn.parameters(), lr=LR)
    enn.train()
    for e in range(EPO):
        for i_batch, batch_data in enumerate(dataLoader):
            bps_out = enn(batch_data[0])
            loss = lossF(bps_out, batch_data[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    enn.save(extend_loc)


def predict_bp_from_atts(atts, extend_loc):
    # classify
    clf = DNN(ATT, MID, HID_CLF, N_INCEPT_CLF)
    clf.load_state_dict(torch.load(clf_loc))
    group = np.argmax(feed_np_get_np(clf, atts))

    # regress
    ## normal
    if group == 0:
        ### dnn: normalize input, feed, de-normalize output
        dims_std = np.loadtxt(dims_std_normal)
        dims_mean = np.loadtxt(dims_mean_normal)
        atts = (atts[:IN] - dims_mean) / dims_std

        dnn = DNN(IN, MID, HID_DNN, N_INCEPT_NRM)
        dnn.load_state_dict(torch.load(normal_loc))

        #### using extend model, 2021/11/16, li yongshuai.
        if extend_loc is not None:
            extend = ExtendNN(HID_DNN, MIDe, None)
            extend.load_state_dict(torch.load(extend_loc))
            extend.dnn = dnn
            _pre_bp = feed_np_get_np(extend, atts)
            pre_bp = NORM * _pre_bp
        else:   #### using basic model, 2021/11/16, li yongshuai
            _pre_bp = feed_np_get_np(dnn, atts)
            pre_bp = NORM * _pre_bp
    ## highpr
    else:
        dims_std = np.loadtxt(dims_std_highpr)
        dims_mean = np.loadtxt(dims_mean_highpr)
        atts = (atts[:IN] - dims_mean) / dims_std
        
        dnn = DNN(IN, MID, HID_DNN, N_INCEPT_HI)
        dnn.load_state_dict(torch.load(highpr_loc))

        ### extend
        if extend_loc is not None:
            extend = ExtendNN(HID_DNN, MIDe, None)
            extend.load_state_dict(torch.load(extend_loc))
            extend.dnn = dnn
            _pre_bp = feed_np_get_np(extend, atts)
            pre_bp = NORM * _pre_bp
        ### dnn: normalize input, feed, de-normalize output
        else:
            _pre_bp = feed_np_get_np(dnn, atts)
            pre_bp = NORM * _pre_bp
        
    return pre_bp


# [采样点 -> 关键点]
FS = 81
def run_find_keypoints(samples, fs=FS):
    kps_person = []

    # [e] find keypoints in line
    ## clear previous results
    os.system("rm {}".format(os.path.join(os.path.dirname(__file__), 
                                          "cpp-find-keypoints/fea.txt")))

    # for each second, run cpp-find-keypoint/ppg =append mode=> cpp-find-keypoints/fea.txt
    SEG = fs * 60
    n_segs = len(samples) // SEG
    for i_seg in range(n_segs+1):
        ## dump samples -> input.txt
        fp = open(os.path.join(os.path.dirname(__file__), 
                               "cpp-find-keypoints/input.txt"), "w")
        fp.write(
            "\n".join(
                [str(val) for val in samples[i_seg * SEG : (i_seg+1) * SEG]])
            + "\n")

        ## run ./ppg, appends results
        os.system("{}".format(os.path.join(os.path.dirname(__file__), 
                                           "cpp-find-keypoints/ppg")))

    ## load keypoints from "fea.txt"
    line_kps = []
    fp = open(os.path.join(os.path.dirname(__file__), 
                           "cpp-find-keypoints/fea.txt"))
    for line in fp:
        line_kps.append([float(sv) for sv in line.split()])

    return line_kps


# [对外接口]
def update_model(pulse_data, user_name, sbp, dbp):  # 输入单个PPG信号, 2021/11/16, li yongshuai
# def update_model(pulse_data_list, user_name, sbp, dbp): 
    # user_name=="" => train a base model on public data
    logging.info("更新血压模型...")
    result = {}

    try:
        assert(len(user_name) > 0)
        # kpts_data = [run_find_keypoints(pulse_data) for pulse_data in pulse_data_list]
        kpts_data = run_find_keypoints(pulse_data)
        curr_atts = np.array([merge_samples_atts(extract_attributes(kpts_data))])
        curr_bp  = np.transpose([[sbp], [dbp]])

        ## 读取个人数据
        # N_secs x 52 <- N_secs x (N_samples x 52) <- N_secs x (N_samples x LEN_FEA)
        # p_atts = np.array(([merge_samples_atts(extract_attributes(sec_kpts)) for sec_kpts in kpts_data]))
        # p_bp  = np.transpose([sbp, dbp])  # N_secs x 2
        atts_path = os.path.join(current_path, 'model', user_name, 'atts.txt')
        bp_path = os.path.join(current_path, 'model', user_name, 'bp.txt')
        if os.path.isfile(atts_path):
            prev_atts = np.reshape(np.loadtxt(atts_path), (-1, 52))
            prev_bp = np.reshape(np.loadtxt(bp_path), (-1, 2))

            private_atts = np.concatenate([prev_atts, curr_atts], axis=0)
            private_bp = np.concatenate([prev_bp, curr_bp], axis=0)
        else:
            private_atts = curr_atts
            private_bp = curr_bp

        ## (创建?)拓展模型
        mdl_dir = os.path.join(current_path, "model", user_name)
        if not os.path.exists(mdl_dir):
            os.makedirs(mdl_dir)
        extend_loc = os.path.join(current_path, "model", user_name, "extend.pkl")

        ## 更新模型
        update_model_from_atts_bp(private_atts, private_bp, extend_loc)

        ## 保存本次标注数据
        with open(atts_path, 'ab') as f:
            np.savetxt(f, curr_atts)
        with open(bp_path, 'ab') as f:
            np.savetxt(f, curr_bp)

        result["isSuccessful"] = True
        return json.dumps(result)
    except Exception as e:
        logging.error(repr(e))
        logging.error(traceback.format_exc())

        result["isSuccessful"] = False
        return json.dumps(result)


def predict_blood_pressure(pulse_data, user_name):  # 用脉搏波预测血压,参数是脉搏波数据的列表
    logging.info("预测血压...")
    pulse_data = np.array(pulse_data)
    result = dict()
    try:
        assert(len(user_name) > 0)
        kpts_data = run_find_keypoints(pulse_data)
        # kpts_data = isPPGWave(pulse_data, kpts_data)

        ## 读取个人历史数据: (1 x ATT) <- (N_kpts x ATT) <- (N_kpts x LEN_KPT)
        atts = merge_samples_atts(extract_attributes(kpts_data))

        ## 模型路径：拓展模型
        extend_loc = os.path.join(current_path, "model", user_name, "extend.pkl")
        if not os.path.isfile(extend_loc):
            extend_loc = None

        pre_bp = predict_bp_from_atts(atts, extend_loc)
        pre_sbp = pre_bp[0]
        pre_dbp = pre_bp[1]

        ## 打印结果
        logging.info('收缩压预测值：{}'.format(round(pre_sbp)))
        logging.info('舒张压预测值：{}'.format(round(pre_dbp)))

        # 范围判断
        if (pre_sbp < 80 or pre_sbp > 200 or pre_dbp < 40 or pre_dbp > 120 
                or pre_dbp >= pre_sbp or pre_sbp - pre_dbp > 100):
            result["error"] = "Unexpected blood pressure, try again!"
            raise TypeError("血压测试异常，请重新测试！")

        # 结果
        result["isSuccessful"] = True
        result["sbp"] = str(round(pre_sbp))
        result["dbp"] = str(round(pre_dbp))
        return json.dumps(result)
    except Exception as e:
        logging.error(repr(e))
        logging.error(traceback.format_exc())
        result["isSuccessful"] = False
        return json.dumps(result)


if __name__ == "__main__":
    pass    
