#本文件用于展示原始图像的直方图和计算图像的平均亮度
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm

def show_histogram_and_calculate_brightness(image_path):
    # 设置字体以支持中文显示
    font_path = "C:/Windows/Fonts/simhei.ttf"  # 你可以根据实际情况选择其他支持中文的字体
    font_prop = fm.FontProperties(fname=font_path)
    
    # 读取图像
    image = cv2.imread(image_path)
    
    # 检查图像是否成功加载
    if image is None:
        raise ValueError("图像加载失败，请检查路径是否正确")
    
    # 分离图像的三个通道
    channels = cv2.split(image)
    colors = ('b', 'g', 'r')
    
    # 绘制直方图
    # plt.figure()
    # plt.title("三通道直方图", fontproperties=font_prop)
    # plt.xlabel("像素值", fontproperties=font_prop)
    # plt.ylabel("像素数", fontproperties=font_prop)
    
    # for (channel, color) in zip(channels, colors):
    #     hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    #     plt.plot(hist, color=color)
    #     plt.xlim([0, 256])
    
    # plt.show()
    
    # 计算平均亮度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray_image)
    
    return average_brightness