# -*- coding: utf-8 -*-
"""
Time compution of running program.

Created on 07/12/21 11:40 AM

@file: heart_rate_variability.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""

import time
from functools import wraps

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        #    print(function)
        # print(str(t1 - t0))
        print ("Total time running %s: %s seconds" % (function.__name__, str(t1-t0)))
        return result

    return function_timer

