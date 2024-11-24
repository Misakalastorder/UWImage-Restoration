import os
from colorshift import apply_color_shift
from PIL import Image
import cv2
#通过调用colorshift.py中的apply_color_shift函数，实现对指定文件夹中的图片进行颜色偏移处理，并将处理后的图片保存到指定文件夹中。
def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, filename in enumerate(os.listdir(input_folder)):
        if i >= 2:
            break # 仅处理前五个文件
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            #修改代码使用cv2读入
            image=cv2.imread(input_path)
            shifted_image = apply_color_shift(image)
            
            #调用cv2.imwrite函数保存图片
            cv2.imwrite(output_path, shifted_image)

            # shifted_image.save(output_path)

if __name__ == "__main__":
    input_folder = 'Attachment'
    output_folder = 'Q2_foggy'
    script_dir = os.path.dirname(__file__)
    input_folder = os.path.join(script_dir, 'Attachment')
    output_folder = os.path.join(os.path.dirname(__file__), 'Q2_colorshift')
    process_images(input_folder, output_folder)