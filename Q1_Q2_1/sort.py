import os
import csv
from showraw import show_histogram_and_calculate_brightness
import judge



#读取Attachment文件夹下的文件，统计文件夹下的文件数目
def count_files_in_attachment():
    attachment_folder = 'Attachment'
    try:
        files = os.listdir(attachment_folder)
        file_count = len(files)
        print(f"Number of files in '{attachment_folder}' folder: {file_count}")
    
        # 打开 CSV 文件以写入结果
        with open('result.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入 CSV 文件的标题行
            writer.writerow(["Filename", "Average Brightness","darkness_probability","furryposibility","colorpref","colorprefnum"])
            
            # 遍历文件并处理图像
            for i, filename in enumerate(files):
                if i >= 400:
                    break # 仅处理前两个文件
                file_path = os.path.join(attachment_folder, filename)
                try:
                    brightness = show_histogram_and_calculate_brightness(file_path)
                    #调用show_histogram_and_calculate_brightness函数，计算图像的平均亮度

                    #调用show_histogram_and_calculate_darkness_probability函数，计算图像为暗的概率
                    darkness_probability = judge.calculate_darkness_probability(file_path)

                    furryposibility = judge.is_image_blurry(file_path)

                    # colorpref = judge.calculate_hsv_features(file_path)
                    colorpref, colorprefnum = judge.calculate_color_bias(file_path)

                    #将brightness和darkness_probability写入CSV文件
                    writer.writerow([filename, brightness, darkness_probability,furryposibility,colorpref,colorprefnum])
                except ValueError as e:
                    print(f"Error processing {filename}: {e}")
    
    except FileNotFoundError:
        print(f"The folder '{attachment_folder}' does not exist.")

count_files_in_attachment()
