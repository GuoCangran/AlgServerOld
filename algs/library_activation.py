# -*- coding: utf-8 -*-
"""
Extract key points of ppg signal.

Created on 07/12/21 11:40 AM

@file: keypoints_extraction.py
@author: hl <hl@hengaigaoke.com>
@copyright(C), 2020-2022, Hengai Gaoke Tech. Co. Ltd.
"""

import base64
import binascii
from Crypto.Cipher import AES


def encrypt(data, password, iv):
    bs = 16
    BLOCK_SIZE = 16
    #pad = lambda s: s + (bs - len(s) % bs) * chr(bs - len(s) % bs)
    print('iv len: %s' % (len(iv)))
    cipher = AES.new(password, AES.MODE_CBC, iv)
    print('data: %s' % (data))
    x = data.encode("utf-8")
    x = x + ((BLOCK_SIZE - len(x) % BLOCK_SIZE) * chr(BLOCK_SIZE - len(x) % BLOCK_SIZE)).encode("utf-8")
    print('x len: %s' % (len(x)))
    #x = pad(data)
    rdata = cipher.encrypt(x)
    print('rdata: %s' % (rdata))
    #rdata = iv + rdata
    return (rdata)


def decrypt(data, password):
    bs = AES.block_size
    if len(data) <= bs:
        return (data)
    unpad = lambda s : s[0:-s[-1]]
    iv = data[:bs]
    cipher = AES.new(password, AES.MODE_CBC, iv)
    data  = unpad(cipher.decrypt(data[bs:]))
    return (data)


def generate_activation_code(request_code):
    #先base64解码
    try:
        encrypt_data = base64.b64decode(request_code)
    except Exception as e:
        print('error1:', e)
    #前16字节是iv
    random = encrypt_data[:16]
    random1 = binascii.b2a_hex(random)
    print('random1 :%s' % (random1))
    randomstr = random1.decode('ascii')
    print('randomstr :%s' % (randomstr))
    #密钥
    password = b'78f40f2c57eee727'      #16,24,32位长的密码
    try:
        decrypt_data = decrypt(encrypt_data, password)
        #print ('decrypt_data11:', decrypt_data)
        print('decrypt_data11: %s' % (decrypt_data))
        param = decrypt_data.decode("utf8","ignore")
        print('param1: %s' % (param))
        listparam = param.split("-")
        fact = listparam[0]
        imei = listparam[1]
        plain_text = fact+"-"+imei+"-"+randomstr
        print('plain_text: %s' % (plain_text))
        activation_code = encrypt(plain_text, password,random)
        print('license_code1: %s' % (activation_code))
        activation_code = base64.b64encode(activation_code).decode("utf8","ignore")
    except Exception as e:
        print('error2:', e)
        activation_code = ""

    return activation_code


def test_generate_activation_code():
    req_code = 'WSSsJIzDg5okdiQKuNHSm47KglaYlgRlAJITYoXYewBPechSyE3sPJ+F\/TbAoUcd'
    act_code = generate_activation_code(req_code)
    print('activation code: %s' % act_code)
    fact_code = 'jsqCVpiWBGUAkhNihdh7AH8i/SNUtbSXJdFjBj5+Af0Yeb4V9VH8J42WMPUGOpPFSr/4nb2YF1qyW5TX4mOMGg=='
    assert act_code == fact_code


if __name__ == '__main__':
    test_generate_activation_code()
