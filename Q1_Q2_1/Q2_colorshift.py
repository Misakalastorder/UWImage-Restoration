import os
from colorshift import apply_color_shift
from PIL import Image
from Feature import write_feature_file
import matplotlib.pyplot as plt

#通过调用colorshift.py中的apply_color_shift函数，实现对指定文件夹中的图片进行颜色偏移处理，并将处理后的图片保存到指定文件夹中。
def process_images(input_folder, output_folder,c=[1,1,1],d=1,max_images=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, filename in enumerate(os.listdir(input_folder)):
        if i >= max_images:
            break # 仅处理前五个文件
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = Image.open(input_path)
            #颜色偏移RGB和距离设定
            shifted_image = apply_color_shift(image,c,d)
            shifted_image.save(output_path)
            print(f"colorshifted: {filename}")


def save_histograms(input_folder, histogram_folder):
    if not os.path.exists(histogram_folder):
        os.makedirs(histogram_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            image = Image.open(input_path).convert('RGB')
            histogram = image.histogram()

            # Split the histogram into RGB channels
            r = histogram[0:256]
            g = histogram[256:512]
            b = histogram[512:768]

            # Plot the histograms
            plt.figure()
            plt.title(f"Color Histogram for {filename}")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.xlim([0, 256])
            plt.plot(r, color='red', label='Red')
            plt.plot(g, color='green', label='Green')
            plt.plot(b, color='blue', label='Blue')
            plt.legend()

            # Save the histogram as an image
            histogram_path = os.path.join(histogram_folder, f"{os.path.splitext(filename)[0]}_histogram.png")
            plt.savefig(histogram_path)
            plt.close()


if __name__ == "__main__":
    #选择文件夹的图片进行颜色变换
    input_folder0 = 'reference'
    #直方图输出文件夹
    histograms_folder0 = 'Q2_histograms'
    #输出文件夹
    output_folder0 = 'Q2_colorshift'
    #颜色衰减
    c=[1,0.1,0]
    max_images=1
    
    script_dir = os.path.dirname(__file__)
    input_folder = os.path.join(script_dir, input_folder0)
    output_folder = os.path.join(os.path.dirname(__file__), output_folder0)
    
    process_images(input_folder, output_folder,c, d=1,max_images=max_images)
    write_feature_file(output_folder, 'Q2_colorshift_feature.csv', max_images=max_images)
    
    histogram_folder = os.path.join(os.path.dirname(__file__), histograms_folder0)
    save_histograms(output_folder, histogram_folder)
 
