"""
 > Script for testing .pth models  
    * set model_name ('funiegan'/'ugan') and  model path
    * set data_dir (input) and sample_dir (output) 
"""
# py libs
import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
import cv2

def resize(size0, intensor):

    # #获取intensor的尺寸并打印
    # print(f"Input tensor size: {intensor.size()}")
    outputtensor = intensor
    # intensor234= intensor
    #将intensor的第二第三和第四维度的值提取生成一个新的tensor
    # intensor234 = intensor[0, :, :, :]
    # # print(f"Input tensor size234: {intensor.size()}")
    # # # 提取intensor的第一个维度的值，生成一个新的tensor
    # # intensor1 = intensor[:, 0, 0, 0] 
    # # print(f"Input tensor size1: {intensor234.size()}")


    outputtensor = F.interpolate(intensor, size=size0, mode='bilinear', align_corners=False)
    # # #将intensor234赋值给intensor的第二和第三和第四个维度
    # outputtensor[0, :, :, :] = intensor234
    # # print(f"outputtensor size: {outputtensor.size()}")
    return outputtensor

## options
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="D:/2024/pccup/Q4/UW-net/data/test/A/")
# parser.add_argument("--data_dir", type=str, default="D:/2024/pccup/Q4/UW-net/data/Attachment/")

parser.add_argument("--sample_dir", type=str, default="D:/2024/pccup/Q4/UW-net/data/output/")
parser.add_argument("--model_name", type=str, default="funiegan") # or "ugan"
parser.add_argument("--model_path", type=str, default="D:/2024/pccup/Q4/UW-net/PyTorch/models/funie_generator.pth")
opt = parser.parse_args()

## checks
assert exists(opt.model_path), "model not found"
os.makedirs(opt.sample_dir, exist_ok=True)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

## model arch
if opt.model_name.lower()=='funiegan':
    from nets import funiegan
    model = funiegan.GeneratorFunieGAN()
elif opt.model_name.lower()=='ugan':
    from nets.ugan import UGAN_Nets
    model = UGAN_Nets(base_model='pix2pix').netG
else: 
    # other models
    pass

## load weights
model.load_state_dict(torch.load(opt.model_path))
if is_cuda: model.cuda()
model.eval()
print ("Loaded model from %s" % (opt.model_path))

## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)


## testing loop
times = []
test_files = sorted(glob(join(opt.data_dir, "*.*")))

#使下面循环最多执行3次

for i, path in enumerate(test_files):
    print("Tested: %s" % path)
    if i >= 404:
        break
    image = Image.open(path)
    #输出图像的通道数和通道类型
    # print(f"Image mode: {image.mode}")
    # 获取图像通道数
    channels = len(image.getbands())
    # print(f"Number of channels: {channels}")
    #检查image是否为RGB图像，如果不是则转换为RGB图像
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    #借助cv2获取image的尺寸
    #获取image的尺寸
    img_width, img_height = image.size
    # print(f"Image size: {img_width} x {img_height}")

    inp_img = transform(image)
    inp_img_size = inp_img.size()
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
    # generate enhanced image
    s = time.time()
    gen_img = model(inp_img)

    times.append(time.time()-s)
    # save output
    # img_sample = torch.cat((inp_img.data, gen_img.data), -1)
    img_sample = resize([img_height, img_width], gen_img)
    save_image(img_sample, join(opt.sample_dir, basename(path)), normalize=True)
    print("Tested: %s" % path)

## run-time    
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files)) 
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in in %s\n" %(opt.sample_dir))





