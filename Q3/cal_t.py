import cv2
import numpy as np
from math import exp
from cal_d import estimate_depth
from cal_beta import estimate_beta
def estimate_t(image):
    # zeros = np.zeros(image.shape[:2], dtype="uint8")
    
    # image=image.astype(np.float64)
    # image=image/255
    B, G, R = cv2.split(image)
    

    #计算B通道的平均强度
    depthimg=estimate_depth(image)
    avg_B = np.mean(B)
    EPXL =1-avg_B/128

    bbr,bbg=estimate_beta(image)

    tbx=np.exp(-EPXL*depthimg)
    trx=np.power(tbx,bbr)
    tgx=np.power(tbx,bbg)
    #下面对img进行分通道处理
    #对B通道进行处理，将B通道的值每个值都除以tbx中对应的值
    Bc = np.divide(B, np.maximum(tbx,0.1))
    #对G通道进行处理，将G通道的值每个值都除以tgx中对应的值
    Gc = np.divide(G, np.maximum(tgx,0.1))
    #对R通道进行处理，将R通道的值每个值都除以trx中对应的值
    Rc = np.divide(R, np.maximum(trx,0.1))
    # 对每个通道进行归一化处理
    Bc = cv2.normalize(Bc, None, 0, 255, cv2.NORM_MINMAX)
    Gc = cv2.normalize(Gc, None, 0, 255, cv2.NORM_MINMAX)
    Rc = cv2.normalize(Rc, None, 0, 255, cv2.NORM_MINMAX)

    Bc = Bc.astype(np.uint8)
    Gc = Gc.astype(np.uint8)
    Rc = Rc.astype(np.uint8)

    zeros = np.zeros_like(Rc)
    red_channel = cv2.merge([zeros, zeros, Rc])
    green_channel = cv2.merge([zeros, Gc, zeros])
    blue_channel = cv2.merge([Bc, zeros, zeros])
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Red Channel', red_channel)
    # cv2.imshow('Green Channel', green_channel)
    # cv2.imshow('Blue Channel', blue_channel)





    #将处理后的通道按0-255进行合并
    img = cv2.merge([Bc, Gc, Rc])

    return  img,EPXL
