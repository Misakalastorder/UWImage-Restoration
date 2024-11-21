# Function: 展示图片三通道直方图，并计算图片的平均亮度和判断为暗图的概率
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import multivariate_normal

#计算暗图可能性函数
def calculate_darkness_probability(image_path):
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
    
    # 计算暗的概率
    darkness_threshold = 100  # 你可以根据实际情况调整阈值
    darkness_probability = max(0.0, min(1.0, (darkness_threshold - average_brightness) / darkness_threshold))
    
    return darkness_probability

#计算模糊可能性函数
def is_image_blurry(image_path, blur_threshold=100):
    """
    判断图像是否模糊，并返回拉普拉斯方差。
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"图像加载失败，请检查路径是否正确: {image_path}")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    return laplacian_var

#计算色偏函数
def calculate_hsv_features(image_path):
    """
    计算图像的亮度和颜色偏移度（基于HSV）。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"图像加载失败，请检查路径是否正确: {image_path}")
    #转换为HSV颜色空间
# 在图像处理中，将图像从BGR（或RGB）颜色空间转换到HSV颜色空间有几个重要的原因，尤其是在处理颜色和亮度时：
# 分离亮度和颜色信息：HSV颜色空间将颜色信息（色调H和饱和度S）与亮度信息（明度V）分离开来。
# 这使得我们可以单独处理亮度和颜色。例如，在你的代码中，v_channel 提取了亮度信息，而 h_channel 提取了色调信息。
# 更直观的颜色表示：在HSV颜色空间中，色调（H）直接表示颜色类型（如红色、绿色、蓝色等），饱和度（S）表示颜色的纯度或强度，
# 而明度（V）表示颜色的亮度。这种表示方式更符合人类的视觉感知。
# 简化颜色检测和处理：在HSV颜色空间中，检测特定颜色或调整颜色范围变得更加简单。例如，检测红色只需要
# 检查色调（H）是否在红色的范围内，而不需要考虑亮度和饱和度。
# 在你的代码中，转换到HSV颜色空间后，你提取了亮度通道（v_channel）来计算图像的平均亮度，并提取了色调通道（h_channel）
# 来计算颜色偏差。这些操作在HSV颜色空间中更容易实现和理解。
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 从HSV图像中提取亮度通道和色调通道
    v_channel = hsv_image[:, :, 2]
# 亮度通道是HSV图像的第三个通道（索引为2）   
    h_channel = hsv_image[:, :, 0]
    # 在HSV颜色空间中，色调（H）通道的取值范围是0到360度
    # 色调通道是HSV图像的第一个通道（索引为0）

    # brightness = np.mean(v_channel)
    h_center = 0
    color_deviation = np.mean(np.abs(h_channel - h_center))

    return color_deviation

def calculate_color_bias(image_path):
    """
    判断图像是否偏蓝或偏绿，并返回一个衡量值。
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"图像加载失败，请检查路径是否正确: {image_path}")

    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 提取色调通道
    h_channel = hsv_image[:, :, 0]
    #提取亮度通道
    v_channel = hsv_image[:, :, 2].astype(np.float32) / 255.0
                                                                                             
    
    average_hue = np.mean(h_channel)
    average_hue = average_hue + 75
    # 判断偏蓝或偏绿
    if 90 <= average_hue <= 150:
        color_bias = "-1"
    elif 150 < average_hue <= 270:
        color_bias = "1"
    else:
        color_bias = "0"

    return color_bias, average_hue
