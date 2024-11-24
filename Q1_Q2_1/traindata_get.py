import os
import cv2
import pandas as pd
from Feature import calculate_feature

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

def trans_data(base_path = 'Q1_train_data'):
    # 初始化数据列表
    data = []
    # 遍历每个类别文件夹
    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        if os.path.isdir(class_path):
            print(f"Processing class: {class_name}")
            # 遍历文件夹中的每个图像文件
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                if os.path.isfile(image_path):
                    # 使用cv2读取图像
                    image = cv2.imread(image_path)
                    if image is not None:
                        # 计算图像特征
                        features = calculate_feature(image)
                        # 将特征列表转换为字典
                        features_dict = {name: value for name, value in zip(feature_names, features)}
                        features_dict['Class'] = class_name
                        features_dict['image_name'] = image_name
                        data.append(features_dict)
                        #print(f"Processing class: {class_name}, Processed image: {image_name}")

    # 将数据转换为DataFrame
    df = pd.DataFrame(data)

    # 保存到CSV文件
    df.to_csv('train_data.csv', index=False)
    print("Features saved to train_data.csv")

if __name__ == '__main__':
    base_path = 'Q1_train_data'
    trans_data(base_path)