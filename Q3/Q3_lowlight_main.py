import cv2
import numpy as np
import os

# 计算雾化图像的暗通道
def dark_channel(img, size = 15):
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))  # 取最暗通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img

# 估计全局大气光值
def get_atmo(img, percent = 0.001):
    mean_perpix = np.mean(img, axis = 2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)

# 估算透射率图
def get_trans(img, atom, w = 0.95):
    x = img / atom
    t = 1 - w * dark_channel(x, 15)
    return t

# 引导滤波
def guided_filter(p, i, r, e):
    """
    :param p: 输入图像
    :param i: 引导图像
    :param r: 半径
    :param e: 正则化参数
    :return: 滤波输出 q
    """
    # 1 计算均值
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    # 2 计算方差和协方差
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    # 3 计算系数 a 和 b
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    # 4 计算均值 a 和 b
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    # 5 计算输出 q
    q = mean_a * i + mean_b
    return q

# 去雾处理
def dehaze(im):
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
    atom = get_atmo(img)
    trans = get_trans(img, atom)
    trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
    trans_guided = cv2.max(trans_guided, 0.25)
    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
    return result * 255

# 批量去雾处理
def dehaze_V2(originPath, savePath):
    '''originaPath: 文件夹的路径，图片上一级
       savePath：同理'''
    for image_name in os.listdir(originPath):
        image_path = os.path.join(originPath, image_name)
        print(image_path)
        im = cv2.imread(image_path)
        img = im.astype('float64') / 255
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
        atom = get_atmo(img)
        trans = get_trans(img, atom)
        trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
        trans_guided = cv2.max(trans_guided, 0.25)
        result = np.empty_like(img)
        for i in range(3):
            result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
        oneSave = os.path.join(savePath, image_name)
        #对result归一化至0-255
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        result = result.astype(np.uint8)


        cv2.imwrite(oneSave, result)

    # cv2.imshow("source", img)
    # cv2.imshow("result", result)
    
    # cv2.waitKey(0)

if __name__ == '__main__':
    dehaze_V2(r'./Attachment1', './output/lowlight')

