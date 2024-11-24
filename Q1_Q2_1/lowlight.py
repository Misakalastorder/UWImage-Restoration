from PIL import Image, ImageEnhance
import numpy as np
from depth_estimate import estimate_depth

def calculate_attenuation_coefficient(beta, d):
    
    # 计算衰减系数 ,其为自然对数的底的-beta次方
    attenuation_coefficient = np.exp(-beta * d)
    return attenuation_coefficient

# 对一张图片进行操作，使其整体光线变暗
def apply_lowlight_effect(img, beta_value, background_light):

    r, g, b = img.split()  
    d = estimate_depth(img)
    
    # 估计背景光
    # background_light = estimate_background_light_local(img)
    attenuation_coefficient = calculate_attenuation_coefficient(beta_value, d)
    
    # 将图像和深度图转换为NumPy数组
    r = np.array(r, dtype=np.float32)
    g = np.array(g, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    attenuation_coefficient = np.array(attenuation_coefficient, dtype=np.float32)
    attenuation_coefficient=attenuation_coefficient
    # r=r * attenuation_coefficient + background_light * (1 - np.mean(attenuation_coefficient))
    # g=g * attenuation_coefficient + background_light * (1 - np.mean(attenuation_coefficient))
    # b=b * attenuation_coefficient + background_light * (1 - np.mean(attenuation_coefficient))
    t=np.mean(attenuation_coefficient)
    r=r*t + background_light * (1 - attenuation_coefficient)
    g=g*t + background_light * (1 - attenuation_coefficient)
    b=b*t  + background_light * (1 - attenuation_coefficient)
    # 应用衰减系数和背景光
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)
    
    # 将结果转换回PIL图像
    r = Image.fromarray(r.astype(np.uint8))
    g = Image.fromarray(g.astype(np.uint8))
    b = Image.fromarray(b.astype(np.uint8))
    
    # 合并通道
    img = Image.merge("RGB", (r, g, b))
    return img

if __name__ == "__main__":
    img_path = 'reference/reference_image.png'
    img = Image.open(img_path)
    beta_value = 10
    background_light = 40
    lowlight_img = apply_lowlight_effect(img, beta_value, background_light)
    lowlight_img.show()
    lowlight_img.save('lowlight_result.jpg')