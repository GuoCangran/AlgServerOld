
import numpy as np

""" data format #1:
    row_data = <line, style=row1> \n <line, style=row1> \n ...
        <line, style=row1> = {'phone': <id>, 
                              'fs': <int>, 
                              'pluse_data': [int, int, ...], 
                              'end_time': <int>,
                              'start_time': <int>}

    tp = <line, style=tp> \n <line, style=tp> \n ...
        <line, style=tp> = <time>  <sbp: int>  <dbp: int>  <int>

    correspondence:
        |- row_data[i]<line, style=row1> ~ tp[j]<line=tp>
        |    if (a) 0s < row_data[i]['end_time'] - tp[j].<time> < 30s
        |       (b) row_data[i-1]['end_time'] < row_data[i]['end_time']
        |       (c) tp[j-1].<time> < tp[j].<time>
        |- skip
        |-   otherwise
        # (a): whether in same measure slot
        # (b,c): skip dis-ordered frames
"""
dirs_sample_inputs = "../../../../raw_data/data_ppg_kpts_202108/"


""" data format #2:
    row_data = <line, style=row2> \n <line, style=row2> \n ...
        <line, style=row2> = {'time': <str>, 
                              'row_data': [<list, dtype=float>, ...]}
            <list, dtype=float> = [float, ......, float]

    bp = <line, style=bp> \n <line, style=bp> ...
        <line, style=bp> = {'time': <str>, 'sbp': <int>, 'dbp': <int>}

    correspondence: <line, style=row2> ~ <line=bp>
"""
dirs_kpts_inputs = "../../../../raw_data/data_ppg_kpts/"


###
MIN_ATTS = 3
ATT = 52
HID_CLF = 100
IN = 26
HID_DNN = 150
MID = 15
N_INCEPT_CLF = 8
N_INCEPT_NRM = 3
N_INCEPT_HI = 25
MIDe = 15

NORM = 100
EPS = 1e-5

def mean_std(atts_list):
    mean = np.mean(atts_list, axis=0)
    std  = np.std(atts_list, axis=0)
    return np.concatenate([mean, std])

def average_atts(atts_list):
    mean = np.mean(atts_list, axis=0)
    return mean
###

def merge_samples_atts(atts):
    if len(atts) < MIN_ATTS: return None
    return mean_std(atts)


# import os
# import re

# def get_persons_names(i_data_source):
#     p_data_dir = os.path.join(os.path.dirname(__file__), INPUTS[i_data_source])
#     fnames = os.listdir(p_data_dir)
#     id_format = "1[0-9]+"
#     matcher = re.compile(id_format)
#     return p_data_dir, [s for s in fnames if matcher.match(s)]

