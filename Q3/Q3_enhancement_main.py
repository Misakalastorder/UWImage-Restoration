import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def enhance(img):
    # 估计高斯模糊参数
    kernel_size, sigma = estimate_gaussian_blur_params(img)
    
    # 去模糊处理
    en_img = deblur_image(img, kernel_size, sigma)
    
    return en_img

def estimate_gaussian_blur_params(image):
    # 估计高斯模糊的卷积核大小和标准差
    # 这里使用一个简单的估计方法，你可以根据需要调整
    
    kernel_size = estimate_kernel_size(image)
    # kernel = cv2.getGaussianKernel(kernel_size, 0)
    # sigma=np.sqrt(np.sum(kernel**2))
    #根据kernel_size计算sigma
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    return kernel_size, sigma

def deblur_image(image, kernel_size, sigma):
    # 创建高斯模糊核
    
    # 调用 OpenCV 的 getGaussianKernel 函数生成一个一维的高斯核。
    # kernel_size 是核的大小，sigma 是高斯分布的标准差。这个函数返回一个列向量。
    # np.outer(kernel, kernel)：使用 NumPy 的 outer 函数计算两个向量的外积。
    # 这里将一维的高斯核与自身进行外积运算，生成一个二维的高斯核矩阵
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel)
    
    # 进行逆滤波
    # 这是 OpenCV 提供的一个函数，用于对图像进行二维卷积操作。
    # image：这是输入图像，类型为 cv2.typing.MatLike，即 OpenCV 的图像矩阵。
    # -1：这是 ddepth 参数，表示输出图像的深度。-1 表示输出图像的深度与输入图像相同。
    # kernel：这是卷积核，类型为 cv2.typing.MatLike，用于定义卷积操作的权重。
    # 这行代码的作用是对输入图像 image 应用卷积核 kernel，并将结果存储在 
    # deblurred 变量中。卷积操作通常用于图像处理中的滤波、边缘检测、去噪等任务。
    deblurred = cv2.filter2D(image, -1, kernel)
    return deblurred

def estimate_kernel_size(image):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算图像的傅里叶变换
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    
    # 计算幅度谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    rows, cols = gray.shape
    x = np.linspace(0, cols, cols)
    y = np.linspace(0, rows, rows)
    x, y = np.meshgrid(x, y)

    #用三维空间绘出矩阵magnitude_spectrum
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, magnitude_spectrum, cmap='rainbow')
    plt.show()



    # 计算幅度谱的中心区域的平均值
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    center_region = magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30]
    mean_val = np.mean(center_region)
    
    # 根据平均值估计卷积核大小
    if mean_val < 10:
        kernel_size = 3
    elif mean_val < 20:
        kernel_size = 5
    elif mean_val < 30:
        kernel_size = 7
    else:
        kernel_size = 9
    
    return kernel_size

def dehaze(originPath,savePath):
    '''originaPath:文件夹的路径，图片上一级
       savePath：同理'''
    for image_name in os.listdir(originPath):
        image_path = os.path.join(originPath,image_name)
        print(image_path)
        img = cv2.imread(image_path)
         
        deblurred_image = enhance(img)
      
        cv2.imshow('Original Image', img)
        cv2.imshow('Deblurred Image', deblurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # cv2.imshow("source",img)
    # cv2.imshow("result", result)


def deflur(image):
    #来源https://blog.51cto.com/u_12204/8983821
    en_img=image
    return en_img


if __name__ == '__main__':
    dehaze(r'./Attachment2','./forty/enhancement')



