from cal_B import estimate_background_light_local

def estimate_beta(img):
    _,_,cl=estimate_background_light_local(img)
    
    Br=cl[0]
    Bg=cl[1]
    Bb=cl[2]
    r=620
    g=540
    b=450
    #代入公式(-0.00113*1+1.62517)
    betabetar=Bb*(-0.00113*r+1.62517)/(-0.00113*b+1.62517)/Br
    betabetag=Bg*(-0.00113*g+1.62517)/(-0.00113*b+1.62517)/Bg


    return betabetar,betabetag