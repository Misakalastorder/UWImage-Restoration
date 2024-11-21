import cv2 
import numpy as np 
def estimate_background_light_local(img, window_size=15): 
    #将img读取为numpy数组
    image = np.array(img)
    # 转换为灰度图像 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    #展示这张gray图像
    # cv2.imshow('gray',gray)
    # #保存上一个窗口的图像
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # # 计算局部最大值 
    local_max = cv2.dilate(gray, np.ones((window_size, window_size), np.uint8))
    #  # 找到亮度最高的区域 
    max_value = np.max(local_max) 
    max_indices = np.where(local_max == max_value) 
    # 
    # 
    # # 选择该区域的平均值作为背景光 
    background_light = np.mean(image[max_indices[0], max_indices[1]]) 
    return 0 # 读取图像 

    return background_light # 读取图像 

#计算衰减系数
def calculate_attenuation_coefficient(beta,d=10): 
    
    # 计算衰减系数 ,其为自然对数的底的-beta次方
    attenuation_coefficient = np.exp(-beta*d)
  
    return attenuation_coefficient

#测试代码
# image = cv2.imread('standrd.png') 
#     # # 估计背景光 
# background_light = estimate_background_light_local(image) 
# print(f'Estimated Background Light: {background_light}')

