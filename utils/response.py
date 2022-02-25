# -*- coding: utf-8 -*-
"""
Response of client request.

Created on 17/12/21 11:40 AM

@file: response.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""


def corr_response(data=None):
    """正确时的返回

    :param data: 字典
    """
    # str_data = str(data) if data is not None else ''
    # logger.info(len(str_data))

    if data is None:
        params = {
            "status": True,
        }
    else:
        params = {
            "status": True,
            "data": data
        }
    return params


def err_response(err_code, description):
    """发生错误时的返回

    :param err_code: 错误码字符串
    :param description: 错误描述

    """
    # logger.info('error response')

    params = {
        "status": False,
        "err_code": err_code,
        "description": description
    }
    return params


if __name__ == "__main__":
    a = corr_response([])
    print("asdas")

