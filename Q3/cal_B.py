import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os
import csv

def estimate_background_light_local(img, size=[90, 90]):
    gray_img = rgb2gray(img)
    flat_gray = gray_img.flatten()
    
    # 去除亮度大小处于较大的前0.01%的点
    threshold_001 = np.percentile(flat_gray, 99.99)
    mask_001 = gray_img < threshold_001
    
    # 标记亮度大小处于前0.1%的点
    threshold_01 = np.percentile(flat_gray, 99.9)
    mask_01 = (gray_img >= threshold_01) & mask_001
    
    # 超像素分割
    segments = slic(img, n_segments=(img.shape[0] // size[0]) * (img.shape[1] // size[1]), compactness=10)
    
    min_gradient = float('inf')
    reference_pixels = []
    marked_segment = None

    for segment_val in np.unique(segments):
        mask = segments == segment_val
        segment_pixels = gray_img[mask]
        gradient = np.mean(np.gradient(segment_pixels))
        
        if gradient < min_gradient and np.any(mask_01[mask]):
            min_gradient = gradient
            reference_pixels = img[mask & mask_01]
            marked_segment = segment_val

    # 确保 reference_pixels 是 NumPy 数组
    reference_pixels = np.array(reference_pixels)

    # 检查 reference_pixels 是否为空
    if reference_pixels.size > 0:
        background_light = np.mean(reference_pixels)
    else:
        background_light = np.mean(img)  # 如果没有找到标记点，返回默认值

    color_background_light = np.mean(img, axis=(0, 1))

    if marked_segment is not None:
        mask = segments == marked_segment
        marked_points = img[mask & mask_01]
        
        # 确保 marked_points 是 NumPy 数组
        marked_points = np.array(marked_points)
        
        if marked_points.size > 0:
            mean_values = np.mean(marked_points, axis=0)
            color_background_light = mean_values
        else:
            color_background_light = np.mean(img, axis=(0, 1))
    else:
        color_background_light = np.mean(img, axis=(0, 1))

    # 标记点标红
    marked_img = img.copy()
    marked_img[mask_01] = [255, 0, 0]
    
    # 标出结果超像素块的轮廓
    # if marked_segment is not None:
    #     marked_img = mark_boundaries(marked_img, segments == marked_segment, color=(0, 1, 0))
    # 标出所有超像素块的轮廓
    marked_img = mark_boundaries(marked_img ,segments, color=(0, 1, 0))
    
    # plt.imshow(marked_img)
    # plt.title('Superpixel Segmentation with Marked Points')
    # plt.axis('off')
    # plt.show()
    
    # print(color_background_light)
    #计算结果超像素块所有的标记点三通道的均值

    return background_light, marked_img,color_background_light

# def process_images_and_save_results(folder_path):
    
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     csv_file = os.path.join(output_folder, 'background_light_results.csv')
#     with open(csv_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Image', 'Background Light'])
        
#         files = os.listdir(folder_path)
#         for i, file in enumerate(files):
#             # 执行多次后停止
#             if i >= 403:
#                 break
#             file_path = os.path.join(folder_path, file)
#             if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#                 image = cv2.imread(file_path)
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 background_light, marked_img, color_background_light = estimate_background_light_local(image)
                
#                 marked_img_path = os.path.join(output_folder, f'marked_{file}')
#                 plt.imsave(marked_img_path, marked_img)
                
#                 writer.writerow([file, background_light])
#                 print(f'Processed {file}, Background Light: {background_light}')




# if __name__ == "__main__":
#     folder_path = os.path.join(os.path.dirname(__file__), 'Attachment')
#     output_folder = os.path.join(os.path.dirname(__file__), 'lightestimate')
#     process_images_and_save_results(folder_path)
