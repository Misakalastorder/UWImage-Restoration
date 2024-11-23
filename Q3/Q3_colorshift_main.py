import os
import csv
import cv2
from cal_B import estimate_background_light_local
from cal_t import estimate_t
import matplotlib.pyplot as plt
import cal_whitebalance as wb

def process_images_and_save_results(folder_path, output_folder=None):
    
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
            if i >= 30:
                break
            file_path = os.path.join(folder_path, file)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                
                
                image = cv2.imread(file_path)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                processed_img,depthimg,outputimage_mark,temp1,selected_img,outb,outg,outr,img_cshift= estimate_t(image)
                # marked_img=wb.white_balance_3(processed_img)
                # cv2.imshow('outputimage_mark',selected_img)
                # cv2.waitKey(0)

                #使用cv2对img_cshift进行图像保存，保存在output文件夹中的img_cshift文件夹中,提取file的文件名，然后输出文件为文件名+img_cshift+后缀名
                img_cshift_folder = os.path.join(output_folder, 'img_cshift')
                if not os.path.exists(img_cshift_folder):
                    os.makedirs(img_cshift_folder)
                img_cshift_path = os.path.join(img_cshift_folder, f'{os.path.splitext(file)[0]}_img_cshift{os.path.splitext(file)[1]}')
                cv2.imwrite(img_cshift_path, img_cshift)

                #使用cv2对outb进行图像保存，保存在output文件夹中的outb文件夹中,提取file的文件名，然后输出文件为文件名+outb+后缀名
                outb_folder = os.path.join(output_folder, 'outb')
                if not os.path.exists(outb_folder):
                    os.makedirs(outb_folder)
                outb_path = os.path.join(outb_folder, f'{os.path.splitext(file)[0]}_outb{os.path.splitext(file)[1]}')
                cv2.imwrite(outb_path, outb)

                #使用cv2对outr进行图像保存，保存在output文件夹中的outr文件夹中
                outr_folder = os.path.join(output_folder, 'outr')
                if not os.path.exists(outr_folder):
                    os.makedirs(outr_folder)
                outr_path = os.path.join(outr_folder, f'{os.path.splitext(file)[0]}_outr{os.path.splitext(file)[1]}')
                cv2.imwrite(outr_path, outr)

                #使用cv2对outg进行图像保存，保存在output文件夹中的outg文件夹中
                outg_folder = os.path.join(output_folder, 'outg')
                if not os.path.exists(outg_folder):
                    os.makedirs(outg_folder)
                outg_path = os.path.join(outg_folder, f'{os.path.splitext(file)[0]}_outg{os.path.splitext(file)[1]}')
                cv2.imwrite(outg_path, outg)

                #使用cv2对selected_img进行图像保存，保存在output文件夹中的selected_img文件夹中
                selected_img_folder = os.path.join(output_folder, 'selected_img')
                if not os.path.exists(selected_img_folder):
                    os.makedirs(selected_img_folder)
                selected_img_path = os.path.join(selected_img_folder, f'{os.path.splitext(file)[0]}_selected_img{os.path.splitext(file)[1]}')
                cv2.imwrite(selected_img_path, selected_img)

                #使用cv2对outputimage_mark进行图像保存，保存在output文件夹中的marked文件夹中
                marked_folder = os.path.join(output_folder, 'marked')
                if not os.path.exists(marked_folder):
                    os.makedirs(marked_folder)
                marked_path = os.path.join(marked_folder, f'{os.path.splitext(file)[0]}_marked{os.path.splitext(file)[1]}')
                cv2.imwrite(marked_path, outputimage_mark)

                #使用cv2对processed_img进行图像保存，保存在output文件夹中的fixed文件夹中               
                fixed_folder = os.path.join(output_folder, 'fixed')
                if not os.path.exists(fixed_folder):
                    os.makedirs(fixed_folder)
                fixed_path = os.path.join(fixed_folder, f'{os.path.splitext(file)[0]}_fixed{os.path.splitext(file)[1]}')
                cv2.imwrite(fixed_path, processed_img)

                #使用cv2对depthimg进行图像保存，保存在output文件夹中的depthimg文件夹中
                depthimg_folder = os.path.join(output_folder, 'depthimg')
                if not os.path.exists(depthimg_folder):
                    os.makedirs(depthimg_folder)
                depthimg_path = os.path.join(depthimg_folder, f'{os.path.splitext(file)[0]}_depthimg{os.path.splitext(file)[1]}')
                cv2.imwrite(depthimg_path, depthimg)
                

                # plt.imsave(marked_img_path, marked_img)
                #注意plt按照RGB通道进行保存，而cv2按照BGR通道进行保存
                
                writer.writerow([file, temp1])
                # print(f'Processed {file}, Background Light: {temp}')




if __name__ == "__main__":
    folder_path = os.path.join(os.path.dirname(__file__), 'Attachment1')
    output_folder = os.path.join(os.path.dirname(__file__), 'output')
    process_images_and_save_results(folder_path,output_folder)


    
    
