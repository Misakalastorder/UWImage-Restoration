from cal_B import estimate_background_light_local
import cv2
def estimate_beta(img,depthimg):
    _,selected_img,bl,outputimage_mark=estimate_background_light_local(img,depthimg)
    #BGR通道的背景光
    print(bl)
    Bb=bl[0]
    Bg=bl[1]
    Br=bl[2]
    #用cv2展示outputimage_mark
    
    r=620
    g=540
    b=450
    #nm
    
    #代入公式(-0.00113*1+1.62517)
    betabetar=Bb*(-0.00113*r+1.62517)/(-0.00113*b+1.62517)/Br
    betabetag=Bb*(-0.00113*g+1.62517)/(-0.00113*b+1.62517)/Bg
    bl=bl/255

    return betabetar,betabetag,bl,outputimage_mark,selected_img