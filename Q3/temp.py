# 引入使用的库
import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def RL_deconvblind(img, PSF, iterations):
    img = img.astype(np.float64)
    PSF = PSF.astype(np.float64)
    init_img = img
    PSF_hat = flip180(PSF)
    for i in range(iterations):
        est_conv = convolve2d(init_img, PSF, 'same')
        relative_blur = img / est_conv
        error_est = convolve2d(relative_blur, PSF_hat, 'same')
        init_img = init_img * error_est
    return np.uint8(normal(init_img))

def fspecial_Gaussian(KernelWH, sigma):
    r, c = KernelWH
    return np.multiply(cv2.getGaussianKernel(r, sigma), (cv2.getGaussianKernel(c, sigma)).T)

def bluredImg(src):
    GausBlurImg = cv2.GaussianBlur(src, (7, 7), 3)
    return GausBlurImg

def normal(img):
    return (img - img.min()) / (img.max() - img.min()) * 255

if __name__ == '__main__':
    path = r"D:\2024\pccup\Q3\Attachment\image_001.png"
    image1 = cv2.imread(path)
    image = bluredImg(image1)
    b_gray, g_gray, r_gray = cv2.split(image.copy())
 
    Result1 = []
    iterations = 20    #迭代次数
    PSF = fspecial_Gaussian((5, 5), 0)
    for gray in [b_gray, g_gray, r_gray]:
        channel1 = RL_deconvblind(gray, PSF, iterations)
        Result1.append(channel1)
 
    init_img_all = cv2.merge([Result1[0], Result1[1], Result1[2]])
    #展示三个通道的图像
    cv2.imshow("b_gray", Result1[0])
    cv2.imshow("g_gray", Result1[1])
    cv2.imshow("r_gray", Result1[2])
    
    
    #用cv2展示init_img_all
    cv2.imshow("init_img_all", init_img_all)

    plt.figure(figsize=(8, 5))
    plt.gray()
    imgNames = {"Original_Image": image,
                "init_img_all": init_img_all,
                }
    for i, (key, imgName) in enumerate(imgNames.items()):
        plt.subplot(121 + i)
        plt.xlabel(key)
        plt.imshow(np.flip(imgName, axis=2))
    plt.show()
