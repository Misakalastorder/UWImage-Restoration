import os
from foggy import apply_fog_effect
from PIL import Image

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, filename in enumerate(os.listdir(input_folder)):
        if i >= 5:
            break # 仅处理前两个文件
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            foggy_img = apply_fog_effect(img)
            output_path = os.path.join(output_folder, filename)
            foggy_img.save(output_path)

if __name__ == "__main__":
    input_folder = 'Attachment'
    output_folder = 'Q2_foggy'
    script_dir = os.path.dirname(__file__)
    input_folder = os.path.join(script_dir, 'Attachment')
    output_folder = os.path.join(script_dir, 'Q2_foggy')
    process_images(input_folder, output_folder)
