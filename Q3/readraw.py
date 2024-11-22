#添加必要的库
import cv2
import numpy as np
import os
from cal_d import estimate_depth

def process_images_in_folder(folder_path):
    output_folder = 'Q3_depthestimate'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all files in the folder
    files = os.listdir(folder_path)
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        
        # Check if the file is an image
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            
            image = cv2.imread(file_path)
            depth_map = estimate_depth(image)
            
            # Save the depth map
            depth_map_path = os.path.join(output_folder, f'depth_{file}')
            cv2.imwrite(depth_map_path, depth_map)
            # print(f'Saved depth map for {file} at {depth_map_path}')

if __name__ == "__main__":
    folder_path = './Attachment'
    process_images_in_folder(folder_path)
