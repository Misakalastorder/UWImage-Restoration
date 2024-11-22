import os
import csv
import cv2
from cal_B import estimate_background_light_local
from cal_t import estimate_t
import matplotlib.pyplot as plt
import cal_whitebalance as wb

def process_images_and_save_results(folder_path):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    csv_file = os.path.join(output_folder, 'output.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['Image', 'Background Light'])
        # 遍历文件夹中的所有文件
        files = os.listdir(folder_path)
        for i, file in enumerate(files):
            # 执行多次后停止
            if i >= 2:
                break
            file_path = os.path.join(folder_path, file)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                processed_img,temp1= estimate_t(image)
                marked_img=wb.white_balance_1(processed_img)


                marked_img_path = os.path.join(output_folder, f'fixed_{file}')
                plt.imsave(marked_img_path, marked_img)
                
                writer.writerow([file, temp1])
                # print(f'Processed {file}, Background Light: {temp}')




if __name__ == "__main__":
    folder_path = os.path.join(os.path.dirname(__file__), 'Attachment')
    output_folder = os.path.join(os.path.dirname(__file__), 'output')
    process_images_and_save_results(folder_path)
