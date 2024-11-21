
from PIL import Image
import sys
from Q2_co_formuparrma import estimate_background_light_local,calculate_attenuation_coefficient
def apply_color_shift(img):
    
    beta_value = (1,0,0)  
    # Example shift values for RGB channels

    # 将图像转换为RGB模式
    img = img.convert("RGB")
    r, g, b = img.split()
    # 获取三色通道的图像
    r_img = Image.merge("RGB", (r, Image.new("L", r.size), Image.new("L", r.size)))
    g_img = Image.merge("RGB", (Image.new("L", r.size), g, Image.new("L", r.size)))
    b_img = Image.merge("RGB", (Image.new("L", r.size), Image.new("L", r.size), b))
  
    # 估计背景光
    r_background_light=estimate_background_light_local(r_img)
    g_background_light=estimate_background_light_local(g_img)
    b_background_light=estimate_background_light_local(b_img)
 
    r_attenuation_coefficient=calculate_attenuation_coefficient(beta_value[0])
    g_attenuation_coefficient=calculate_attenuation_coefficient(beta_value[1])
    b_attenuation_coefficient=calculate_attenuation_coefficient(beta_value[2])

    # 对每个通道进行偏移
    # r = r.point(lambda i: min(255, max(0, i + i * shift_value[0])))
    # g = g.point(lambda i: min(255, max(0, i + i * shift_value[1])))
    # b = b.point(lambda i: min(255, max(0, i + i * shift_value[2])))
    r = r.point(lambda i: min(255, max(0, i *r_attenuation_coefficient+r_background_light*(1-r_attenuation_coefficient))))
    g = g.point(lambda i: min(255, max(0, i *g_attenuation_coefficient+g_background_light*(1-g_attenuation_coefficient))))
    b = b.point(lambda i: min(255, max(0, i *b_attenuation_coefficient+b_background_light*(1-b_attenuation_coefficient))))
    
    # 合并通道
    img = Image.merge("RGB", (r, g, b))

    return img

