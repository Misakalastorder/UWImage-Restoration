#计算一张图片的深度图像
import cv2
import numpy as np

def estimate_depth(image):
    # Read the image
    
    
    # Split the image into R, G, B channels
    B, G, R = cv2.split(image)
    
    # Calculate the average intensity of each channel
    avg_B = np.mean(B)
    avg_G = np.mean(G)
    avg_R = np.mean(R)
    
    # Determine the channel with the maximum average intensity
    if avg_G > avg_B:
        max_channel = G
    else:
        max_channel = B
    
    # Calculate the depth map as the difference between the max channel and the R channel
    depth_map = cv2.absdiff(max_channel, R)
    #输出图像depth_map
    # cv2.imshow('depth_map',depth_map)
    #将深度图像的像素值归一化在0到255之间
    #将depth_map转换为float64格式的矩阵
    depth_map = depth_map.astype(np.float64)
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    # cv2.imshow('depth_map01',depth_map)
    return depth_map






