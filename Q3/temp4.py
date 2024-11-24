import cv2
import math
import numpy as np
import os
import csv

def cot (img):
    image = cv2.imread(img)#图片路径
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # RGB转为HSV
    H, S, V = cv2.split(hsv)
    delta = np.std(H) /180  #色度的标准差
    mu = np.mean(S) /255  #饱和度的平均值
    #求亮度对比值
    n, m = np.shape(V)
    number = math.floor(n*m/10000)  #所需像素的个数
    Maxsum, Minsum = 0, 0
    V1, V2 = V /255, V/255

    for i in range(1, number+1):
        Maxvalue = np.amax(np.amax(V1))
        x, y = np.where(V1 == Maxvalue)
        Maxsum = Maxsum + V1[x[0],y[0]]
        V1[x[0],y[0]] = 0

    top = Maxsum/number

    for i in range(1, number+1):
        Minvalue = np.amin(np.amin(V2))
        X, Y = np.where(V2 == Minvalue)
        Minsum = Minsum + V2[X[0],Y[0]]
        V2[X[0],Y[0]] = 1

    bottom = Minsum/number

    conl = top-bottom
    ###对比度
    uciqe = 0.4680*delta + 0.2745*conl + 0.2576*mu
    print(uciqe)
    return uciqe

def process_folder(path):
    results = []
    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(path, filename)
            print(img_path)
            uciqe_value = cot(img_path)
            results.append((filename, uciqe_value))

    with open('D:/2024/pccup/Q3/uciqe_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "UCIQE Value"])
        writer.writerows(results)

process_folder("D:/2024/pccup/Q3/output/lowlight/未提前归一化")