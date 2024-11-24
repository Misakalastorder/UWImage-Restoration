import cv2
import numpy as np
from scipy.signal import wiener
import os


def process(img,output_path,file):
    # 读取图像
    flag=0
    
    # image_path = r'D:\2024\pccup\Q3\Attachment1\image_001.png'
    # image = cv2.imread(image_path)

    image=img

    # 检查图像是否成功读取
    if image is None:
        raise FileNotFoundError(f"无法打开或读取文件: {image}")

    #获取image三通道，并输出为三个图像
    b, g, r = cv2.split(image)
    if flag:
        cv2.imshow("Blue", b)
        cv2.imshow("Green", g)
        cv2.imshow("Red", r)


    # 显示原始图像
    if flag:
        cv2.namedWindow('Original Image')
        cv2.imshow('Original Image', image)
        cv2.waitKey(0)

    # 生成原始图像的灰度图像的边缘检测图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges1 = cv2.Canny(gray_image, 100, 200)

    if flag:
        cv2.namedWindow('Original Image Edges')
        cv2.imshow('Original Image Edges', edges1)

    # 生成高斯卷积核
    gaussian_kernel = cv2.getGaussianKernel(7, 1)
    gaussian_kernel = gaussian_kernel * gaussian_kernel.T

    # 对每个通道进行卷积，生成高斯模糊图像
    blurred_image = np.zeros_like(image)
    for i in range(3):
        blurred_image[:, :, i] = cv2.filter2D(image[:, :, i], -1, gaussian_kernel)

    # 叠加高斯噪声
    noise = np.random.normal(0, 0.5, image.shape)
    noisy_blurred_image = blurred_image + noise

    # 显示模糊图像
    if flag:
        cv2.namedWindow('Noisy Blurred Image')
        cv2.imshow('Noisy Blurred Image', noisy_blurred_image.astype(np.uint8))

    # 生成模糊图像的灰度图像的边缘检测图像
    gray_noisy_blurred_image = cv2.cvtColor(noisy_blurred_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edges2 = cv2.Canny(gray_noisy_blurred_image, 100, 200)
    if flag:
        cv2.namedWindow('Noisy Blurred Image Edges')
        cv2.imshow('Noisy Blurred Image Edges', edges2)

    # 估计初始点扩展函数h
    h = np.zeros((7, 7))
    h[3, :] = 1
    h[:, 3] = 1
    h = h / np.sum(h)

    # 对每个通道进行维纳滤波
    restored_image = np.zeros_like(image, dtype=np.float32)
    H = np.fft.fft2(h, s=image.shape[:2])
    S = np.abs(H)**2 / (np.abs(H)**2 + 0.5)

    for i in range(3):
        G = np.fft.fft2(noisy_blurred_image[:, :, i])
        restored_channel = np.fft.ifft2(S * G / H).real
        restored_image[:, :, i] = restored_channel

    # 二维中值滤波
    restored_image = cv2.medianBlur(restored_image.astype(np.uint8), 3)

    # 显示恢复图像
    if flag:
        cv2.namedWindow('Restored Image')
        cv2.imshow('Restored Image', restored_image)


    B,G,R=cv2.split(restored_image)


    # 生成恢复图像的灰度图像的边缘检测图像
    gray_restored_image = cv2.cvtColor(restored_image, cv2.COLOR_BGR2GRAY)
    edges3 = cv2.Canny(gray_restored_image, 100, 200)
    if flag:
        cv2.namedWindow('Restored Image Edges')
        cv2.imshow('Restored Image Edges', edges3)

    # 将窗口整齐排列
    if flag:
        cv2.moveWindow('Original Image', 0, 0)
        cv2.moveWindow('Original Image Edges', 400, 0)
        cv2.moveWindow('Noisy Blurred Image', 800, 0)
        cv2.moveWindow('Noisy Blurred Image Edges', 0, 400)
        cv2.moveWindow('Restored Image', 400, 400)
        cv2.moveWindow('Restored Image Edges', 800, 400)

    # 保存结果图像
    #将b、g、r三个通道保存
    print(file)
    #将output_path修改为output_path下的enhancement文件夹
    # output_path=os.path.join(output_path, 'enhancement')
    # ott=os.path.join(output_path, f'{os.path.splitext(file)[0]}_Blue_before{os.path.splitext(file)[1]}')
    # print(ott)
    print(output_path)
    if 1==1:
        #将file的名字与文件保存的名字拼接作为保存名字
        cv2.imwrite(os.path.join(output_path, f'{os.path.splitext(file)[0]}_Blue_before{os.path.splitext(file)[1]}'), b)
        cv2.imwrite(os.path.join(output_path, f'{os.path.splitext(file)[0]}_Green_before{os.path.splitext(file)[1]}'), g)
        cv2.imwrite(os.path.join(output_path, f'{os.path.splitext(file)[0]}_Red_before{os.path.splitext(file)[1]}'), r)

        cv2.imwrite(os.path.join(output_path, f'{os.path.splitext(file)[0]}_Blue_after{os.path.splitext(file)[1]}'), B)
        cv2.imwrite(os.path.join(output_path, f'{os.path.splitext(file)[0]}_Green_after{os.path.splitext(file)[1]}'), G)
        cv2.imwrite(os.path.join(output_path, f'{os.path.splitext(file)[0]}_Red_after{os.path.splitext(file)[1]}'), R)

        cv2.imwrite(os.path.join(output_path, f'{os.path.splitext(file)[0]}_original_image{os.path.splitext(file)[1]}'), image)
        cv2.imwrite(os.path.join(output_path, f'{os.path.splitext(file)[0]}_original_image_edges{os.path.splitext(file)[1]}'), edges1)
        cv2.imwrite(os.path.join(output_path, f'{os.path.splitext(file)[0]}_noisy_blurred_image{os.path.splitext(file)[1]}'), noisy_blurred_image.astype(np.uint8))
        cv2.imwrite(os.path.join(output_path, f'{os.path.splitext(file)[0]}_noisy_blurred_image_edges{os.path.splitext(file)[1]}'), edges2)
        cv2.imwrite(os.path.join(output_path, f'{os.path.splitext(file)[0]}_restored_image{os.path.splitext(file)[1]}'), restored_image)
        cv2.imwrite(os.path.join(output_path, f'{os.path.splitext(file)[0]}_restored_image_edges{os.path.splitext(file)[1]}'), edges3)

    if flag:
        # cv2.imwrite(r'D:\2024\pccup\Q3\output\enhancement\Blue_before.png', b)
        # cv2.imwrite(r'D:\2024\pccup\Q3\output\enhancement\Green_before.png', g)
        # cv2.imwrite(r'D:\2024\pccup\Q3\output\enhancement\Red_before.png', r)

        # cv2.imwrite(r'D:\2024\pccup\Q3\output\enhancement\Blue_after.png', B)
        # cv2.imwrite(r'D:\2024\pccup\Q3\output\enhancement\Green_after.png', G)
        # cv2.imwrite(r'D:\2024\pccup\Q3\output\enhancement\Red_after.png', R)

        # cv2.imwrite(r'D:\2024\pccup\Q3\output\enhancement\original_image.png', image)
        # cv2.imwrite(r'D:\2024\pccup\Q3\output\enhancement\original_image_edges.png', edges1)
        # cv2.imwrite(r'D:\2024\pccup\Q3\output\enhancement\noisy_blurred_image.png', noisy_blurred_image.astype(np.uint8))
        # cv2.imwrite(r'D:\2024\pccup\Q3\output\enhancement\noisy_blurred_image_edges.png', edges2)
        # cv2.imwrite(r'D:\2024\pccup\Q3\output\enhancement\restored_image.png', restored_image)
        # cv2.imwrite(r'D:\2024\pccup\Q3\output\enhancement\restored_image_edges.png', edges3)
        1==1

    if flag:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0

def process_images_and_save_results(folder_path, output_folder):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    files = os.listdir(folder_path)
    for i, file in enumerate(files):
        # 执行多次后停止
        if i >= 404:
            break
        file_path = os.path.join(folder_path, file)
        
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):            
            image = cv2.imread(file_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            num= process(image,output_folder,file)

if __name__ == "__main__":
    folder_path = os.path.join(os.path.dirname(__file__), 'Attachment2')
    output_folder = os.path.join(os.path.dirname(__file__), 'forty')
    process_images_and_save_results(folder_path,output_folder)

