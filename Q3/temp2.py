import cv2
import numpy as np
from scipy.signal import convolve2d

def richardson_lucy(image, psf, iterations=10):
    image = image.astype(np.float64)
    psf = psf.astype(np.float64)
    estimate = np.full(image.shape, 0.5)
    psf_mirror = psf[::-1, ::-1]
    
    for _ in range(iterations):
        relative_blur = image / convolve2d(estimate, psf, 'same')
        estimate *= convolve2d(relative_blur, psf_mirror, 'same')
    
    return estimate

# 定义卷积核大小
kernel_size = 5

# 定义高斯卷积核
kernel = np.zeros((kernel_size, kernel_size), np.float32)
sigma = 1.0
for i in range(kernel_size):
    for j in range(kernel_size):
        kernel[i, j] = np.exp(-((i - kernel_size // 2) ** 2 + (j - kernel_size // 2) ** 2) / (2 * sigma ** 2))
kernel = kernel / np.sum(kernel)

# 读取图像
blur_img = cv2.imread('D:/2024/pccup/Q3/Attachment/image_001.png')
#将图片分通道进行处理
b_gray, g_gray, r_gray = cv2.split(blur_img.copy())



if blur_img is None:
    print("无法加载图像，请检查文件路径和文件是否存在。")
    exit()

# 对模糊图像进行卷积操作
blurred_b = cv2.filter2D(b_gray, -1, kernel)
# 进行反卷积操作
deconvolved_b = richardson_lucy(blurred_b, kernel, iterations=20)
# 显示原始图像、模糊图像和复原的图像
cv2.imshow('Original', b_gray)
cv2.imshow('Blurred', blurred_b)
cv2.imshow('Deconvolved', np.uint8(deconvolved_b))
cv2.waitKey(0)
cv2.destroyAllWindows()

#对g_gray进行处理
blurred_g = cv2.filter2D(g_gray, -1, kernel)
deconvolved_g = richardson_lucy(blurred_g, kernel, iterations=20)
cv2.imshow('Original', g_gray)
cv2.imshow('Blurred', blurred_g)
cv2.imshow('Deconvolved', np.uint8(deconvolved_g))
cv2.waitKey(0)
cv2.destroyAllWindows()

#对r_gray进行处理
blurred_r = cv2.filter2D(r_gray, -1, kernel)
deconvolved_r = richardson_lucy(blurred_r, kernel, iterations=20)
cv2.imshow('Original', r_gray)
cv2.imshow('Blurred', blurred_r)
cv2.imshow('Deconvolved', np.uint8(deconvolved_r))
cv2.waitKey(0)
cv2.destroyAllWindows()

#将三个通道的图像合并
blurred_img_all = cv2.merge([blurred_b, blurred_g, blurred_r])
init_img_all = cv2.merge([np.uint8(deconvolved_b), np.uint8(deconvolved_g), np.uint8(deconvolved_r)])
cv2.imshow('Original', blur_img)
cv2.imshow('Blurred', blurred_img_all)
cv2.imshow("init_img_all", init_img_all)
cv2.waitKey(0)
cv2.destroyAllWindows()



