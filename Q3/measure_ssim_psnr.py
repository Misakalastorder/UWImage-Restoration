"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
## python libs
import numpy as np
from PIL import Image
from glob import glob
from os.path import join, dirname
from ntpath import basename
## local libs
from imqual_utils import getSSIM, getPSNR


## compares avg ssim and psnr 
def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256, 256)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated images 
    """
    print(gtr_dir)
    #读取gtr_dir下的所有文件，然后统计文件的数量
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    #统计gtr_dir下的文件数量
    print(len(gtr_paths))
    #读取gen_dir下的所有文件，然后统计文件的数量
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    #统计gen_dir下的文件数量
    print(len(gen_paths))
    
    ssims, psnrs = [], []
    #遍历gtr_paths和gen_paths
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        #输出gtr_path
        print(gtr_path)
        #输出gen_path
        print(gen_path)

        gen_f = basename(gen_path).split('.')[0]
        if (gtr_f==gen_f):
            # assumes same filenames
            r_im = Image.open(gtr_path).resize(im_res)
            g_im = Image.open(gen_path).resize(im_res)
            # get ssim on RGB channels
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
            # get psnt on L channel (SOTA norm)
            r_im = r_im.convert("L"); g_im = g_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)




        
    return np.array(ssims), np.array(psnrs)


"""
Get datasets from
 - http://irvlab.cs.umn.edu/resources/euvp-dataset
 - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
"""
gtr_dir = "D:/2024/pccup/Q3/Attachment2"
#gtr_dir = "/home/xahid/datasets/released/UFO-120/TEST/hr/"

## generated im paths
gen_dir = "D:/2024/pccup/Q3/forty/temp" 
#gen_dir = "eval_data/ufo_test/deep-sesr/" 


### compute SSIM and PSNR
SSIM_measures, PSNR_measures = SSIMs_PSNRs(gtr_dir, gen_dir)
#使用os库获取本文件相对路径，将其拼接为csv文件的输出路径outcsv_path
outcsv_path = join(dirname(__file__), "SSIM_PSNR.csv")

#将SSIM_measures和PSNR_measures写入一个csv文件，对应有表头,第一列为文件名，第二列为SSIM，第三列为PSNR

np.savetxt(outcsv_path, np.array([SSIM_measures, PSNR_measures]).T, delimiter=",", header="SSIM,PSNR", comments="")
print ("SSIM on {0} samples".format(len(SSIM_measures)))
print ("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
print ("PSNR on {0} samples".format(len(PSNR_measures)))
print ("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))



