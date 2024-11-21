# Function: 展示图片三通道直方图，并计算图片的平均亮度和判断为暗图的概率
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm

def show_histogram_and_calculate_darkness_probability(image_path):
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



#下文注释 无用
# def process_images():
#     attachment_folder = 'Attachment'
#     try:
#         files = os.listdir(attachment_folder)
        
#         # 仅处理前两张图片
#         for i, filename in enumerate(files):
#             if i >= 2:
#                 break
#             file_path = os.path.join(attachment_folder, filename)
#             try:
#                 darkness_probability = show_histogram_and_calculate_darkness_probability(file_path)
#                 print(f"Image: {filename}, Darkness Probability: {darkness_probability}")
#             except ValueError as e:
#                 print(f"Error processing {filename}: {e}")
    
#     except FileNotFoundError:
#         print(f"The folder '{attachment_folder}' does not exist.")

# process_images()