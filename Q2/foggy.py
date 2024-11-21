from PIL import Image, ImageFilter
import numpy as np
from cv2 import GaussianBlur

def apply_fog_effect(image):
    # foggy_image = image.filter(ImageFilter.GaussianBlur(2))  # 5为雾化强度

    #将image读取为numpy数组
    image0 = np.array(image)
    
    foggy_image0 = GaussianBlur(image0, (15, 15),0)
# 第一个参数 image0 是输入的图像数组。
# 第二个参数 (15, 15) 是高斯核的大小，表示模糊的范围。
# 第三个参数 0 是标准差（sigma），如果设置为 0，则根据核大小自动计算标准差。
# 返回值是模糊后的图像数组。

    #将numpy数组转换为image
    foggy_image = Image.fromarray(foggy_image0)

    return foggy_image
