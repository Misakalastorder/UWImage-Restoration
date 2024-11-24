import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import csv
import os


# 归一化函数
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# 1. 色彩直方图
def calculate_color_histogram_bias(image, plot_histograms=True):
    colors = ('b', 'g', 'r')
    histograms = []
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        histograms.append(hist)
        if plot_histograms:
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
    if plot_histograms:
        plt.title('Color Histogram')
        plt.show()
    
    # 计算每个通道的峰值
    peaks = [np.max(hist) for hist in histograms]
    
    # 计算所有通道的平均峰值
    mean_peak = np.mean(peaks)
    
    # 计算偏移指标
    bias_ratios = [peak / mean_peak for peak in peaks]
    color_bias = max(bias_ratios)
    
    # 归一化
    min_val = 1.0  # 偏移指标的最小值为1
    max_val = 3.0  # 偏移指标的最大值为3（假设）
    return normalize(color_bias, min_val, max_val)

# 3. 色彩平衡
def check_color_balance(image):
    mean_R = np.mean(image[:, :, 2])
    mean_G = np.mean(image[:, :, 1])
    mean_B = np.mean(image[:, :, 0])
    
    balance = np.array([mean_R, mean_G, mean_B])
    max_diff = np.max(balance) - np.min(balance)
    
    # 归一化
    min_val = 0.0  # 差值的最小值为0
    max_val = 255.0  # 差值的最大值为255
    return normalize(max_diff, min_val, max_val)

# 5. 白平衡
def check_white_balance(image):
    white_patch = image[100:200, 100:200]  # 假设图像的某个区域是白色
    mean_R = np.mean(white_patch[:, :, 2])
    mean_G = np.mean(white_patch[:, :, 1])
    mean_B = np.mean(white_patch[:, :, 0])
    
    # 归一化
    min_val = 0.0  # RGB值的最小值为0
    max_val = 255.0  # RGB值的最大值为255
    return normalize(mean_R, min_val, max_val), normalize(mean_G, min_val, max_val), normalize(mean_B, min_val, max_val)

# 主函数
def calculate_color(image):
    # 计算各项指标
    color_bias = calculate_color_histogram_bias(image, plot_histograms=False)
    color_balance = check_color_balance(image)
    white_R, white_G, white_B = check_white_balance(image)
    # 收集所有归一化后的指标
    normalized_metrics = [
        color_bias,
        color_balance,
        white_R, white_G, white_B
    ]
    
    return normalized_metrics

def calculate_light(image):
    # 1. 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 计算平均亮度并归一化
    mean_brightness = np.mean(gray_image)
    mean_brightness_normalized = (mean_brightness - 0) / (255 - 0)  # 假设灰度值范围是0-255

    # 5. 计算直方图峰度并归一化
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist_kurtosis = kurtosis(hist.flatten())
    kurtosis_normalized = (hist_kurtosis - (-3)) / (3 - (-3))  # 假设峰度范围是-3到3

    # 6. 计算像素强度分布并归一化
    dark_threshold = 60  # 可以根据具体需求调整
    total_pixels = gray_image.size
    dark_pixels = np.sum(hist[:dark_threshold])
    dark_ratio = dark_pixels / total_pixels
    dark_ratio_normalized = (dark_ratio - 0) / (1 - 0)  # 假设暗像素比例范围是0-1
# 返回所有指标,变成向量输出
    return [mean_brightness_normalized,  kurtosis_normalized, dark_ratio_normalized]



# 1. 拉普拉斯梯度
def laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 归一化
    min_val = 0.0  # 拉普拉斯方差的最小值为0
    max_val = 1000.0  # 拉普拉斯方差的最大值为1000（假设）
    return normalize(lap_var, min_val, max_val)

# 8. 熵函数
def entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy_value = -np.sum(hist * np.log2(hist + 1e-10))
    
    # 归一化
    min_val = 0.0  # 熵的最小值为0
    max_val = 8.0  # 熵的最大值为8（256个灰度级的熵）
    return normalize(entropy_value, min_val, max_val)

# 10. Reblur 二次模糊
def reblur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    diff = cv2.absdiff(gray, blurred)
    reblur_value = np.sum(diff)
    
    # 归一化
    min_val = 0.0  # Reblur值的最小值为0
    max_val = 255.0 * gray.size  # Reblur值的最大值
    return normalize(reblur_value, min_val, max_val)


def calculate_blur(image):
    # 计算各项指标
    lap_var = laplacian_variance(image)
    entropy_value = entropy(image)
    reblur_value = reblur(image)
    
    # 收集所有归一化后的指标
    normalized_metrics = [
        lap_var,
        entropy_value,
        reblur_value
    ]
    
    return normalized_metrics


def calculate_feature(image_BGR):
    feature=[]
    #统一变量类型
    
    feature+=calculate_color(image_BGR)
    feature+=calculate_light(image_BGR)
    feature+=calculate_blur(image_BGR)
    # 将feature中的所有元素转换为float类型
    feature = [float(f) for f in feature]
    return feature

def write_feature_file(image_folder,output_csv='features.csv',max_images=10):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]

    # 定义特征名称
    feature_names = [
        "color_bias",
        "color_balance",
        "white_R", "white_G", "white_B",
        "mean_brightness_normalized",
        "kurtosis_normalized",
        "dark_ratio_normalized",
        "lap_var",
        "entropy_value",
        "reblur_value"
    ]
    
    # 写入CSV文件
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # 写入表头
        writer.writerow(['image_name'] + feature_names)
        
        # 遍历所有图片文件并计算特征
        for image_file in image_files:
            # 控制图片数量
            
            if image_files.index(image_file) >= max_images:
                break

            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: The image file {image_file} was not found or could not be read.")
                continue
            feature = calculate_feature(image)
            
            # 写入特征值
            writer.writerow([image_file] + feature)
            print(f"Processed {image_file}")


if __name__ == '__main__':
    write_feature_file('Attachment',max_images=3000)
    # 获取Attachment文件夹中的所有图片文件
    # print(f"Mean Red: {mean_R}")
    # print(f"Mean Green: {mean_G}")
    # print(f"Mean Blue: {mean_B}")
    # print(f"Max Color Difference: {max_diff}")
    # print(f"Mean Gray (Gray World Assumption): {mean_gray}")
    # print(f"White Balance (R, G, B): ({white_R}, {white_G}, {white_B})")
    # print(f"Estimated Color Temperature: {color_temp}K")
    # print(f"Color Differences (RG, RB, GB): ({diff_RG}, {diff_RB}, {diff_GB})")


