from PIL import Image, ImageEnhance
from Q2_co_formuparrma import estimate_background_light_local,calculate_attenuation_coefficient


#对一张图片进行操作，使其整体光线变暗
def apply_lowlight_effect(img,beta_value):
    
    # Example shift values for RGB channels
    # 将图像转换为RGB模式
    img = img.convert("RGB")
    r, g, b = img.split()  
    # 估计背景光
    background_light=estimate_background_light_local(img)
    attenuation_coefficient=calculate_attenuation_coefficient(beta_value,2)
    r = r.point(lambda i: min(255, max(0, i *attenuation_coefficient+background_light*(1-attenuation_coefficient))))
    g = g.point(lambda i: min(255, max(0, i *attenuation_coefficient+background_light*(1-attenuation_coefficient))))
    b = b.point(lambda i: min(255, max(0, i *attenuation_coefficient+background_light*(1-attenuation_coefficient))))
    # 合并通道
    img = Image.merge("RGB", (r, g, b))
    return img
        # # Initialize the enhancer
        # enhancer = ImageEnhance.Brightness(img)
        # # Apply the enhancement
        # img_darkened = enhancer.enhance(factor)
        
        # return  img_darkened
    