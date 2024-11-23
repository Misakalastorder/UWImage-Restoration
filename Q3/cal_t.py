import cv2
import numpy as np
from math import exp
from cal_d import estimate_depth
from cal_beta import estimate_beta
import matplotlib.pyplot as plt


def estimate_t(image):
    # zeros = np.zeros(image.shape[:2], dtype="uint8")
    
    depthimg,depth_map0=estimate_depth(image)
 

    B, G, R = cv2.split(image)
    #计算B通道的平均强度
    avg_B = np.mean(B)/255
    # EPXL  = 1-avg_B/128
    EPXL  = 1-avg_B/128
    
    # avg_G = np.mean(G)
    # EPXLG = 1-avg_G/128
    bbr,bbg,bl,outputimage_mark,selected_img=estimate_beta(image,depthimg)
    
    tbx=np.exp(-EPXL*depthimg)
    


    trx=np.power(tbx,bbr)

    # tgx=np.exp(-EPXLG*depthimg)
    tgx=np.power(tbx,bbg)

    #对tbx,trx,tgx进行归一化处理
    outb=cv2.normalize(tbx, None, 0, 255, cv2.NORM_MINMAX)
    outr=cv2.normalize(trx, None, 0, 255, cv2.NORM_MINMAX)
    outg=cv2.normalize(tgx, None, 0, 255, cv2.NORM_MINMAX)
    outb = outb.astype(np.uint8)
    outr = outr.astype(np.uint8)
    outg = outg.astype(np.uint8)

    

    #下面对img进行分通道处理
    # B = B.astype(np.float64)/255
    # G = G.astype(np.float64)/255
    # R = R.astype(np.float64)/255
    # #对B通道进行处理，将B通道的值每个值都除以tbx中对应的值
    # Bc = np.divide(B, np.maximum(tbx,0.1))
    # # #对G通道进行处理，将G通道的值每个值都除以tgx中对应的值
    # Gc = np.divide(G, np.maximum(tgx,0.1))
    # # #对R通道进行处理，将R通道的值每个值都除以trx中对应的值
    # Rc = np.divide(R, np.maximum(trx,0.1))
    Bc = np.divide(B, np.maximum(tbx,0.1))
    # #对G通道进行处理，将G通道的值每个值都除以tgx中对应的值
    Gc = np.divide(G, np.maximum(tgx,0.1))
    # #对R通道进行处理，将R通道的值每个值都除以trx中对应的值
    Rc = np.divide(R, np.maximum(trx,0.1))
    
    #将处理后的通道按0-255进行归一化
    Bc = cv2.normalize(Bc, None, 0, 255, cv2.NORM_MINMAX)
    Gc = cv2.normalize(Gc, None, 0, 255, cv2.NORM_MINMAX)
    Rc = cv2.normalize(Rc, None, 0, 255, cv2.NORM_MINMAX)

#     red_threshold = 70
#     bg_threshold = 150

# # 检测红色通道的值，并在红色通道的值大于阈值且蓝色和绿色通道的值均小于最大值时，给蓝色和绿色通道加上这个红色点的值
#     mask = (Rc > red_threshold) & (Bc < bg_threshold) & (Gc < bg_threshold)
#     Bc[mask] = Rc[mask]
#     Gc[mask] = Rc[mask]
    
#     Bc = B
#    # 对G通道进行处理，将G通道的值每个值都除以tgx中对应的值
#     Gc = G
#     #对R通道进行处理，将R通道的值每个值都除以trx中对应的值
#     Rc = R
    
    # 对每个通道进行解归一化处理
    #创建一个3维数组，其中第一个维度是B通道，第二个维度是G通道，第三个维度是R通道
    # BIGmatrix = np.zeros((Bc.shape[0], Bc.shape[1], 3))
    # BIGmatrix[:, :, 0] = Bc
    # BIGmatrix[:, :, 1] = Gc
    # BIGmatrix[:, :, 2] = Rc
    # #对此3维数组进行整体归一化处理
    # BIGmatrix = cv2.normalize(BIGmatrix, None, 0, 255, cv2.NORM_MINMAX)
    # #BIGmatrix转为unit8格式
    # BIGmatrix = BIGmatrix.astype(np.uint8)
    # #将BIGmatrix转换为图像RGB格式

    # Bc = BIGmatrix[:, :, 0]
    # Gc = BIGmatrix[:, :, 1]
    # Rc = BIGmatrix[:, :, 2]
    

    Bc = Bc.astype(np.uint8)
    Gc = Gc.astype(np.uint8)
    Rc = Rc.astype(np.uint8)
    img_cshift = cv2.merge([Bc, Gc, Rc])
    
    
    # # cv2.imshow('Original', image)
    # # cv2.waitKey(0)
    # # 对YUV通道进行自适应直方图均衡化
    imgYUV = cv2.cvtColor(img_cshift, cv2.COLOR_BGR2YCrCb)
    # cv2.imshow("src", img)
    channelsYUV = list(cv2.split(imgYUV))  # 将元组转换为列表

    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(16, 16))
    channelsYUV[0] = clahe.apply(channelsYUV[0])

    channels = cv2.merge(channelsYUV)
    img1 = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
    # cv2.imshow('Equalized', img1)

    
    
    
    # cv2.waitKey(0)
    #对三个通道做彩色直方图均衡化
    # Bc = cv2.equalizeHist(Bc)
    # Gc = cv2.equalizeHist(Gc)
    # Rc = cv2.equalizeHist(Rc)
    # img1 = cv2.merge([Bc, Gc, Rc])

 
 


    


    

    #此处展示三通道按亮度统计的直方图
    # Display histograms for each channel
    # Rc, Gc, Bc = cv2.split(img1)
    # cv2.waitKey(0)
    # plt.figure(figsize=(10, 5))
 
    # plt.subplot(131)
    # plt.hist(Rc.ravel(), 256, [0, 256], color='r')

    # plt.subplot(132)
    # plt.hist(Gc.ravel(), 256, [0, 256], color='g')
    
    # plt.subplot(133)
    # plt.hist(Bc.ravel(), 256, [0, 256], color='b')

    # plt.tight_layout()
    # plt.show()
    # cv2.imshow('output', img1)



    # zeros = np.zeros_like(Rc)
    # red_channel = cv2.merge([zeros, zeros, Rc])
    # green_channel = cv2.merge([zeros, Gc, zeros])
    # blue_channel = cv2.merge([Bc, zeros, zeros])

    # zeros = np.zeros_like(R)
    # red_channel1 = cv2.merge([zeros, zeros, R])
    # green_channel1 = cv2.merge([zeros, G,zeros])
    # blue_channel1 = cv2.merge([B, zeros, zeros])

    # cv2.imshow('Original Image', image)

    # cv2.imshow('Red Channel-', red_channel1)
    # cv2.imshow('Green Channel-', green_channel1)
    # cv2.imshow('Blue Channel-', blue_channel1)

    # cv2.imshow('Red Channel0', red_channel)
    # cv2.imshow('Green Channel0', green_channel)
    # cv2.imshow('Blue Channel0', blue_channel)




    
    #将处理后的通道按0-255进行合并
    #合成需按RGB顺序
    
    
    
    # img = BIGmatrix
    return  img1,depth_map0,outputimage_mark,EPXL,selected_img,outb,outg,outr,img_cshift
