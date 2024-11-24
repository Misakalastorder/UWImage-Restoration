
from PIL import Image
import sys
import numpy as np
from Q2_co_formuparrma import estimate_background_light_local,calculate_attenuation_coefficient
import cv2
def apply_color_shift(img):
    
    beta_value = (1,0,0)  
    # Example shift values for RGB channels
    # 将图像转换为RGB模式
    b, g, r = cv2.split(img)
    # 获取三色通道的图像
    
  
    # 估计背景光
    r_background_light=estimate_background_light_local(r)
    g_background_light=estimate_background_light_local(g)
    b_background_light=estimate_background_light_local(b)
 
    r_attenuation_coefficient=calculate_attenuation_coefficient(beta_value[0])
    g_attenuation_coefficient=calculate_attenuation_coefficient(beta_value[1])
    b_attenuation_coefficient=calculate_attenuation_coefficient(beta_value[2])

    # 对每个通道进行偏移
    # r = r.point(lambda i: min(255, max(0, i + i * shift_value[0])))
    # g = g.point(lambda i: min(255, max(0, i + i * shift_value[1])))
    # b = b.point(lambda i: min(255, max(0, i + i * shift_value[2])))
    r = np.maximum(0,r*r_attenuation_coefficient+r_background_light*(1-r_attenuation_coefficient))
    g = np.maximum(0,g*g_attenuation_coefficient+g_background_light*(1-g_attenuation_coefficient))
    b = np.maximum(0,b*b_attenuation_coefficient+b_background_light*(1-b_attenuation_coefficient))
    #对每个通道转换到unit8
    r = np.uint8(r)
    g = np.uint8(g)
    b = np.uint8(b)

    # g = g.point(lambda i: min(255, max(0, i *g_attenuation_coefficient+g_background_light*(1-g_attenuation_coefficient))))
    # b = b.point(lambda i: min(255, max(0, i *b_attenuation_coefficient+b_background_light*(1-b_attenuation_coefficient))))
    
    # 合并通道
    img = cv2.merge([b, g, r])

    return img

