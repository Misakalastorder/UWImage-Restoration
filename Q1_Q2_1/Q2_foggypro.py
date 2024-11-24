import os
import numpy as np
import cv2
from PIL import Image
from cv2 import GaussianBlur
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from Feature import calculate_blur  # 假设calculate_blur函数返回三个指标

def apply_fog_effect(image, fog_intensity, sigma):
    # 将image读取为numpy数组
    image0 = np.array(image)
    
    # 确保fog_intensity是一个正奇数
    if fog_intensity % 2 == 0:
        fog_intensity += 1
    ksize = (fog_intensity, fog_intensity)
    
    foggy_image0 = GaussianBlur(image0, ksize, sigma)
    # 将numpy数组转换为image
    foggy_image = Image.fromarray(foggy_image0)

    return foggy_image

def process_images(input_folder, output_folder, kernel_sizes, sigmas):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    csv_file_path = os.path.join(output_folder, 'blur_data.csv')
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['filename', 'kernel_size', 'sigma', 'lap_var_foggy', 'entropy_value_foggy', 'reblur_value_foggy'])
        
        for i, filename in enumerate(os.listdir(input_folder)):
            if i >= 5:
                break # 仅处理前五个文件
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path)
                
                for kernel_size in kernel_sizes:
                    for sigma in sigmas:
                        print(f"Processing {filename} with kernel_size={kernel_size} and sigma={sigma}")
                        foggy_img = apply_fog_effect(img, kernel_size, sigma)
                        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_ks_{kernel_size}_sigma_{sigma}.jpg")
                        foggy_img.save(output_path)
                        
                        # 将图像转换为BGR格式
                        foggy_img_bgr = cv2.cvtColor(np.array(foggy_img), cv2.COLOR_RGB2BGR)
                        
                        # 计算模糊图像的模糊度
                        lap_var_foggy, entropy_value_foggy, reblur_value_foggy = calculate_blur(foggy_img_bgr)
                        csv_writer.writerow([filename, kernel_size, sigma, lap_var_foggy, entropy_value_foggy, reblur_value_foggy])
                        print(f"{filename} (Kernel Size={kernel_size}, Sigma={sigma}): Foggy Blur = (lap_var={lap_var_foggy}, entropy_value={entropy_value_foggy}, reblur_value={reblur_value_foggy})")

def plot_blur_data(csv_file_path):
    filenames = []
    kernel_sizes = []
    sigmas = []
    lap_var_foggy = []
    entropy_value_foggy = []
    reblur_value_foggy = []
    
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # 跳过表头
        for row in csv_reader:
            filenames.append(row[0])
            kernel_sizes.append(int(row[1]))
            sigmas.append(float(row[2]))
            lap_var_foggy.append(float(row[3]))
            entropy_value_foggy.append(float(row[4]))
            reblur_value_foggy.append(float(row[5]))
    
    kernel_sizes = np.array(kernel_sizes)
    sigmas = np.array(sigmas)
    lap_var_foggy = np.array(lap_var_foggy)
    entropy_value_foggy = np.array(entropy_value_foggy)
    reblur_value_foggy = np.array(reblur_value_foggy)
    
    # 创建二维网格
    kernel_sizes_grid, sigmas_grid = np.meshgrid(np.unique(kernel_sizes), np.unique(sigmas))
    
    # 将数据转换为二维网格
    lap_var_foggy_grid = np.zeros_like(kernel_sizes_grid, dtype=np.float32)
    entropy_value_foggy_grid = np.zeros_like(kernel_sizes_grid, dtype=np.float32)
    reblur_value_foggy_grid = np.zeros_like(kernel_sizes_grid, dtype=np.float32)
    
    for i in range(kernel_sizes_grid.shape[0]):
        for j in range(kernel_sizes_grid.shape[1]):
            mask = (kernel_sizes == kernel_sizes_grid[i, j]) & (sigmas == sigmas_grid[i, j])
            if np.any(mask):
                lap_var_foggy_grid[i, j] = lap_var_foggy[mask][0]
                entropy_value_foggy_grid[i, j] = entropy_value_foggy[mask][0]
                reblur_value_foggy_grid[i, j] = reblur_value_foggy[mask][0]
    
    # 绘制第一个窗口
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(kernel_sizes_grid, sigmas_grid, lap_var_foggy_grid, cmap='viridis')
    ax1.set_xlabel('Kernel Size')
    ax1.set_ylabel('Sigma')
    ax1.set_zlabel('Lap Var')
    ax1.set_title('Lap Var vs Kernel Size and Sigma')
    plt.show()
    
    # 绘制第二个窗口
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(kernel_sizes_grid, sigmas_grid, entropy_value_foggy_grid, cmap='viridis')
    ax2.set_xlabel('Kernel Size')
    ax2.set_ylabel('Sigma')
    ax2.set_zlabel('Entropy Value')
    ax2.set_title('Entropy Value vs Kernel Size and Sigma')
    plt.show()
    
    # 绘制第三个窗口
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot_surface(kernel_sizes_grid, sigmas_grid, reblur_value_foggy_grid, cmap='viridis')
    ax3.set_xlabel('Kernel Size')
    ax3.set_ylabel('Sigma')
    ax3.set_zlabel('Reblur Value')
    ax3.set_title('Reblur Value vs Kernel Size and Sigma')
    plt.show()

if __name__ == "__main__":
    input_folder = 'reference'
    output_folder = 'Q2_foggy'
    
    # 设置高斯模糊核大小和方差的范围
    kernel_sizes = range(3, 24, 2)  # 只包含奇数
    a=np.linspace(0.1, 0.9, 81)
    b=np.linspace(1, 4, 61)


    sigmas = np.concatenate([a,b])  # 由密到稀疏的数列
    
    process_images(input_folder, output_folder, kernel_sizes, sigmas)
    
    csv_file_path = os.path.join(output_folder, 'blur_data.csv')
    plot_blur_data(csv_file_path)















# import os
# import numpy as np
# import cv2
# from PIL import Image
# from cv2 import GaussianBlur
# import matplotlib.pyplot as plt
# import csv
# from Feature import calculate_blur  # 假设calculate_blur函数返回三个指标

# def apply_fog_effect(image, fog_intensity, sigma):
#     # 将image读取为numpy数组
#     image0 = np.array(image)
    
#     foggy_image0 = GaussianBlur(image0, (fog_intensity, fog_intensity), sigma)
#     # 将numpy数组转换为image
#     foggy_image = Image.fromarray(foggy_image0)

#     return foggy_image

# def process_images(input_folder, output_folder, kernel_sizes, sigmas):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     csv_file_path = os.path.join(output_folder, 'blur_data.csv')
#     with open(csv_file_path, 'w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(['filename', 'kernel_size', 'sigma', 'lap_var_original', 'entropy_value_original', 'reblur_value_original', 'lap_var_foggy', 'entropy_value_foggy', 'reblur_value_foggy'])
        
#         for i, filename in enumerate(os.listdir(input_folder)):
#             if i >= 5:
#                 break # 仅处理前五个文件
#             if filename.endswith(('.png', '.jpg', '.jpeg')):
#                 img_path = os.path.join(input_folder, filename)
#                 img = Image.open(img_path)
                
#                 # 计算原始图像的模糊度
#                 img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#                 lap_var_original, entropy_value_original, reblur_value_original = calculate_blur(img_bgr)
                
#                 for kernel_size in kernel_sizes:
#                     for sigma in sigmas:
#                         print(f"Processing {filename} with kernel_size={kernel_size} and sigma={sigma}")
#                         foggy_img = apply_fog_effect(img, kernel_size, sigma)
#                         output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_ks_{kernel_size}_sigma_{sigma}.jpg")
#                         foggy_img.save(output_path)
                        
#                         # 将图像转换为BGR格式
#                         foggy_img_bgr = cv2.cvtColor(np.array(foggy_img), cv2.COLOR_RGB2BGR)
                        
#                         # 计算模糊图像的模糊度
#                         lap_var_foggy, entropy_value_foggy, reblur_value_foggy = calculate_blur(foggy_img_bgr)
#                         csv_writer.writerow([filename, kernel_size, sigma, lap_var_original, entropy_value_original, reblur_value_original, lap_var_foggy, entropy_value_foggy, reblur_value_foggy])
#                         print(f"{filename} (Kernel Size={kernel_size}, Sigma={sigma}): Original Blur = (lap_var={lap_var_original}, entropy_value={entropy_value_original}, reblur_value={reblur_value_original}), Foggy Blur = (lap_var={lap_var_foggy}, entropy_value={entropy_value_foggy}, reblur_value={reblur_value_foggy})")

# def plot_blur_data(csv_file_path):
#     filenames = []
#     kernel_sizes = []
#     sigmas = []
#     lap_var_original = []
#     entropy_value_original = []
#     reblur_value_original = []
#     lap_var_foggy = []
#     entropy_value_foggy = []
#     reblur_value_foggy = []
    
#     with open(csv_file_path, 'r') as csv_file:
#         csv_reader = csv.reader(csv_file)
#         next(csv_reader)  # 跳过表头
#         for row in csv_reader:
#             filenames.append(row[0])
#             kernel_sizes.append(int(row[1]))
#             sigmas.append(float(row[2]))
#             lap_var_original.append(float(row[3]))
#             entropy_value_original.append(float(row[4]))
#             reblur_value_original.append(float(row[5]))
#             lap_var_foggy.append(float(row[6]))
#             entropy_value_foggy.append(float(row[7]))
#             reblur_value_foggy.append(float(row[8]))
    
#     plt.figure(figsize=(18, 18))
    
#     plt.subplot(3, 1, 1)
#     plt.scatter(kernel_sizes, lap_var_foggy, c=sigmas, cmap='viridis', marker='x', label='Foggy Lap Var')
#     plt.axhline(y=np.mean(lap_var_original), color='r', linestyle='-', label='Original Lap Var')
#     plt.colorbar(label='Sigma')
#     plt.xlabel('Kernel Size')
#     plt.ylabel('Lap Var')
#     plt.title('Lap Var vs Kernel Size and Sigma')
#     plt.legend()
    
#     plt.subplot(3, 1, 2)
#     plt.scatter(kernel_sizes, entropy_value_foggy, c=sigmas, cmap='viridis', marker='x', label='Foggy Entropy Value')
#     plt.axhline(y=np.mean(entropy_value_original), color='r', linestyle='-', label='Original Entropy Value')
#     plt.colorbar(label='Sigma')
#     plt.xlabel('Kernel Size')
#     plt.ylabel('Entropy Value')
#     plt.title('Entropy Value vs Kernel Size and Sigma')
#     plt.legend()
    
#     plt.subplot(3, 1, 3)
#     plt.scatter(kernel_sizes, reblur_value_foggy, c=sigmas, cmap='viridis', marker='x', label='Foggy Reblur Value')
#     plt.axhline(y=np.mean(reblur_value_original), color='r', linestyle='-', label='Original Reblur Value')
#     plt.colorbar(label='Sigma')
#     plt.xlabel('Kernel Size')
#     plt.ylabel('Reblur Value')
#     plt.title('Reblur Value vs Kernel Size and Sigma')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     input_folder = 'reference'
#     output_folder = 'Q2_foggy'
    
#     # 设置高斯模糊核大小和方差的范围
#     kernel_sizes = [3, 5, 7]
#     sigmas = [1, 10]
    
#     process_images(input_folder, output_folder, kernel_sizes, sigmas)
    
#     csv_file_path = os.path.join(output_folder, 'blur_data.csv')
#     plot_blur_data(csv_file_path)