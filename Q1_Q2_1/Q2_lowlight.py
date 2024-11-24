import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image, ImageStat
import csv
from lowlight import apply_lowlight_effect
from mpl_toolkits.mplot3d import Axes3D

def process_images(input_folder, output_folder, beta_values, background_lights, max_num=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    csv_file_path = os.path.join(output_folder, 'image_parameters.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['filename', 'beta_value', 'background_light', 'brightness_normalized', 'psnr'])

        for beta_value in beta_values:
            for background_light in background_lights:
                for i, filename in enumerate(os.listdir(input_folder)):
                    if i >= max_num:
                        break
                    input_path = os.path.join(input_folder, filename)
                    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_beta{beta_value}_light{background_light}.png")
                    
                    # 读取图像并处理
                    img = Image.open(input_path)
                    lowlight_img = apply_lowlight_effect(img, beta_value, background_light)
                    lowlight_img.save(output_path)

                    # 计算亮度和PSNR
                    brightness = calculate_light(lowlight_img)
                    psnr_value = calculate_psnr(img, lowlight_img)
                    
                    # 归一化亮度
                    brightness_normalized = brightness / 255
                    
                    # 记录参数到CSV文件
                    csv_writer.writerow([filename, beta_value, background_light, brightness_normalized, psnr_value])
                    print(f"Saved {output_path} with brightness={brightness_normalized} and PSNR={psnr_value}")

def calculate_light(image):
    stat = ImageStat.Stat(image)
    R, G, B = stat.mean[:3]
    return 0.299*R + 0.587*G + 0.114*B

def calculate_psnr(original_image, processed_image):
    original = cv2.cvtColor(np.asarray(original_image), cv2.COLOR_RGB2BGR)
    processed = cv2.cvtColor(np.asarray(processed_image), cv2.COLOR_RGB2BGR)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def plot_results(csv_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 创建三维图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维面图
    ax.plot_trisurf(df['beta_value'], df['background_light'], df['psnr'], cmap='viridis', edgecolor='none')

    # 添加颜色条
    sc = ax.scatter(df['beta_value'], df['background_light'], df['psnr'], c=df['psnr'], cmap='viridis', marker='o')
    cbar = plt.colorbar(sc)
    cbar.set_label('PSNR')

    # 设置标签
    ax.set_title('PSNR vs The ambient light and Light intensity attenuation coefficient')
    ax.set_xlabel('Light intensity attenuation coefficient')
    ax.set_ylabel('The ambient light ')
    ax.set_zlabel('PSNR')

    plt.show()

# 示例用法
if __name__ == "__main__":
    input_folder0 = 'reference'
    output_folder0 = 'Q2_lowlight'
    max_num = 1
    script_dir = os.path.dirname(__file__)
    input_folder = os.path.join(script_dir, input_folder0)
    output_folder = os.path.join(script_dir, output_folder0)
    
    beta_values = np.linspace(1, 5, 41)  # 示例beta_value列表
    background_lights = np.linspace(0, 60, 61)  # 示例background_light列表

    process_images(input_folder, output_folder, beta_values, background_lights, max_num=max_num)
    
    csv_file_path = os.path.join(output_folder, 'image_parameters.csv')
    plot_results(csv_file_path)








    '''
    import os
import csv
import matplotlib.pyplot as plt
from PIL import Image, ImageStat
from lowlight import apply_lowlight_effect
import numpy as np
import cv2
def process_images(input_folder, output_folder, beta_values, background_lights, max_num=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    csv_file_path = os.path.join(output_folder, 'image_parameters.csv')
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['filename', 'beta_value', 'background_light',  'brightness', 'contrast'])
        
        for i, filename in enumerate(os.listdir(input_folder)):
            if i >= max_num:
                break # 仅处理前max_num个文件
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path)
                
                for beta_value in beta_values:
                    for background_light in background_lights:
                        lowlight_img = apply_lowlight_effect(img, beta_value, background_light)
                        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_beta_{beta_value}_bg_{background_light}.jpg")
                        lowlight_img.save(output_path)
                        
                        # 计算亮度和对比度
                        brightness = calculate_light(lowlight_img)
                        contrast = calculate_contrast(lowlight_img)
                        
                        # 归一化亮度和对比度
                        brightness_normalized = brightness / 255
                        contrast_normalized = contrast / 255
                        
                        # 记录参数到CSV文件
                        csv_writer.writerow([filename, beta_value, background_light, brightness_normalized, contrast_normalized])
                        print(f"Saved {output_path} with brightness={brightness_normalized} and contrast={contrast_normalized}")

def calculate_light(image):
    stat = ImageStat.Stat(image)
    R, G, B = stat.mean[:3]
    return 0.299*R + 0.587*G + 0.114*B

def calculate_contrast(img0_RGB):   
    img0=cv2.cvtColor(np.asarray(img0_RGB),cv2.COLOR_RGB2BGR)
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #彩色转为灰度图片
    m, n = img1.shape
    #图片矩阵向外扩展一个像素
    img1_ext = cv2.copyMakeBorder(img1,1,1,1,1,cv2.BORDER_REPLICATE) / 1.0   # 除以1.0的目的是uint8转为float型，便于后续计算
    rows_ext,cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            b += ((img1_ext[i,j]-img1_ext[i,j+1])**2 + (img1_ext[i,j]-img1_ext[i,j-1])**2 + 
                    (img1_ext[i,j]-img1_ext[i+1,j])**2 + (img1_ext[i,j]-img1_ext[i-1,j])**2)

    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4) #对应上面48的计算公式
    return cg

def plot_results(csv_file_path):
    beta_values = []
    background_lights = []
    brightness_values = []
    contrast_values = []
    
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # 跳过表头
        for row in csv_reader:
            beta_values.append(float(row[1]))
            background_lights.append(float(row[2]))
            brightness_values.append(float(row[3]))
            contrast_values.append(float(row[4]))
    
    plt.figure(figsize=(10, 18))
    
    plt.subplot(1, 2, 1)
    plt.scatter(beta_values, brightness_values, c=background_lights, cmap='viridis', marker='o')
    plt.colorbar(label='The ambient light')
    plt.title('Brightness vs Light intensity attenuation coefficient')
    plt.xlabel('Light intensity attenuation coefficient')
    plt.ylabel('Brightness (Normalized)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(beta_values, contrast_values, c=background_lights, cmap='viridis', marker='o')
    plt.colorbar(label='The ambient light')
    plt.title('Contrast vs Light intensity attenuation coefficient')
    plt.xlabel('Light intensity attenuation coefficient')
    plt.ylabel('Contrast (Normalized)')
    plt.show()

    plt.figure(figsize=(10, 18))
    plt.subplot(1, 2, 1)
    plt.scatter(background_lights, brightness_values, c=beta_values, cmap='viridis', marker='o')
    plt.colorbar(label='Light intensity attenuation coefficient')
    plt.title('Brightness vs The ambient light')
    plt.xlabel('The ambient light')
    plt.ylabel('Brightness (Normalized)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(background_lights, contrast_values, c=beta_values, cmap='viridis', marker='o')
    plt.colorbar(label='Light intensity attenuation coefficient')
    plt.title('Contrast vs The ambient light')
    plt.xlabel('The ambient light')
    plt.ylabel('Contrast (Normalized)')
    
    plt.tight_layout()
    plt.show()

# 示例用法
if __name__ == "__main__":
    input_folder0 = 'reference'
    output_folder0 = 'Q2_lowlight'
    max_num = 1
    script_dir = os.path.dirname(__file__)
    input_folder = os.path.join(script_dir, input_folder0)
    output_folder = os.path.join(script_dir, output_folder0)
    
    beta_values = np.linspace(3,10,1)  # 示例beta_value列表
    #生成等距的列表

    background_lights = np.linspace(1,150,4)  # 示例background_light列表

    process_images(input_folder, output_folder, beta_values, background_lights, max_num=max_num)
    
    csv_file_path = os.path.join(output_folder, 'image_parameters.csv')
    plot_results(csv_file_path)
    
    
    '''