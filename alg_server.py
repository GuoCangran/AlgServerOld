# -*- coding: utf-8 -*-
"""
Server of algorithm.

Created on 07/12/21 11:40 AM

@file: alg_server.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""

import zerorpc
import apis.alg_api as api

s = zerorpc.Server(api.AlgorithmRPC(), heartbeat=30)
s.bind("tcp://0.0.0.0:4242")
#zerorpc.gevent.spawn(s.run)
#while True:
    #zerorpc.gevent.sleep(10)

s.run()