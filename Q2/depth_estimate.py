import os
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
    #将深度图像的像素值归一化在0到255之间
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    return depth_map


def process_images_in_folder(folder_path):
    output_folder = 'Q2_depthestimate'
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
            print(f'Saved depth map for {file} at {depth_map_path}')

if __name__ == "__main__":
    folder_path = './Attachment'
    process_images_in_folder(folder_path)





