# Single Image Haze Removal Using Dark Channel Prior
import cv2
import numpy as np

def compute_dark_channel(img,radius=7):
    h,w=img.shape[:2]
    window_size=2*radius+1
    b, g, r = cv2.split(img)
    bgr_min_img = cv2.min(cv2.min(b, g), r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size,window_size))
    dark_channel_img = cv2.erode(bgr_min_img, kernel)
    return dark_channel_img

def compute_atmosphere_light(img,dark_channel_img):
    h,w=dark_channel_img.shape[:2]
    num_of_candiate=int(0.001*h*w)
    dark_channel=dark_channel_img.reshape(-1,1)[:,0 ]
    arg_sorted=np.argsort(dark_channel)[::-1]
    img=img.astype(np.float32)
    atmosphere_light=np.zeros((3,))
    for i in range(num_of_candiate):
        index=arg_sorted[i]
        row_index=index//w
        col_index=index%w
        for c in range(3):
            atmosphere_light[c]=max(atmosphere_light[c],img[row_index,col_index][c])
    return atmosphere_light


def compute_transmission_rate(img,atmosphere_light_max,omega,dark_channel_img,guided_filter_radius,epsilon):
    h,w=img.shape[:2]
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    zero_mat=np.zeros((h,w))
    transmition_rate_est=cv2.max(zero_mat,np.ones_like(zero_mat)-omega*dark_channel_img/atmosphere_light_max)
    #transmission_rate = guied_filter(img_gray,transmition_rate_est, guided_filter_radius, epsilon)
    transmission_rate=cv2.ximgproc.guidedFilter(img_gray.astype(np.float32),transmition_rate_est.astype(np.float32),guided_filter_radius,epsilon)
    return transmission_rate


def guied_filter(I,P,radius,epsilon):
    window_size=2*radius+1
    meanI=cv2.blur(I,(window_size,window_size))
    meanP=cv2.blur(P,(window_size,window_size))
    II=I**2
    IP=I*P
    corrI=cv2.blur(II,(window_size,window_size))
    corrIP=cv2.blur(IP,(window_size,window_size))
    varI=corrI-meanI**2
    covIP=corrIP-meanI*meanP
    a=covIP/(varI+epsilon)
    b=meanP-a*meanI
    meanA=cv2.blur(a,(window_size,window_size))
    meanB=cv2.blur(b,(window_size,window_size))
    transmission_rate=meanA*I+meanB
    return transmission_rate

def dehaze(img,radius=7,omega=0.95,epsilon=0.0001,guided_filter_radius=25,transmission_thresh=0.1):
    dark_channel_img=compute_dark_channel(img,radius)
    cv2.imshow('dark',dark_channel_img)
    atmosphere_light=compute_atmosphere_light(img,dark_channel_img)
    transmission_rate=compute_transmission_rate(img,np.max(atmosphere_light),omega,dark_channel_img,guided_filter_radius=guided_filter_radius,epsilon=epsilon)
    transmission_rate[transmission_rate<transmission_thresh]=transmission_thresh
    cv2.imshow('t',transmission_rate)
    dehaze_img=np.zeros_like(img)
    for c in range(3):
        dehaze_img[:,:,c]=(img[:,:,c]-atmosphere_light[c])/transmission_rate+atmosphere_light[c]
    dehaze_img[dehaze_img>1]=1
    dehaze_img[dehaze_img<0]=0
    return dehaze_img

if __name__=='__main__':
    img=cv2.imread('haze_img/9.jpg')
    img=img.astype(np.float32)
    img/=255
    cv2.imshow('src',img)
    dehaze_img=dehaze(img,radius=3)
    cv2.imshow('dehaze',dehaze_img)
    cv2.waitKey(0)


