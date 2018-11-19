import cv2
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import colorsys

"""
python implemention of "Colorization Using Optimization"
http://www.cs.huji.ac.il/~yweiss/Colorization/
based on the matlab code
"""

def yiq_to_rgb(y, i, q):
    r = y + 0.9468822170900693*i + 0.6235565819861433*q
    g = y - 0.27478764629897834*i - 0.6356910791873801*q
    b = y - 1.1085450346420322*i + 1.7090069284064666*q
    r=np.clip(r,0.,1.)
    g=np.clip(g,0.,1.)
    b=np.clip(b,0.,1.)
    return (r,g,b)


def colorize(original_img,marked_img):
    original_img=original_img.astype(float)/255
    marked_img=marked_img.astype(float)/255
    is_colored = cv2.absdiff(original_img,marked_img).sum(2) > 0.01
    y,_,_=colorsys.rgb_to_yiq(original_img[:,:,2],original_img[:,:,1],original_img[:,:,0])
    _,u,v=colorsys.rgb_to_yiq(marked_img[:,:,2],marked_img[:,:,1],marked_img[:,:,0])
    ntsc_img=cv2.merge((y,u,v))
    """
    h,w=ntsc_img.shape[:2]
    max_d=np.floor(np.log(min(h,w))/np.log(2)-2)
    iu=int(np.floor(h/(2**(max_d-1)))*(2**(max_d-1)))
    ju=int(np.floor(w/(2**(max_d-1)))*(2**(max_d-1)))

    is_colored=is_colored[:iu,:ju]
    ntsc_img=ntsc_img[:iu,:ju,:]
    """
    h,w=ntsc_img.shape[:2]
    dst_img=np.zeros_like(ntsc_img)
    dst_img[:,:,0]=ntsc_img[:,:,0]
    inds_matrix=np.arange(h*w).reshape(h,w,order='F')
    labl_inds=np.nonzero(is_colored.reshape(h*w,order='F'))[0]

    wd=1
    len=0
    consts_len=0
    col_inds=np.zeros((h*w*(2*wd+1)**2,),dtype=np.int64)
    row_inds=np.zeros((h*w*(2*wd+1)**2,),dtype=np.int64)
    vals=np.zeros((h*w*(2*wd+1)**2,))

    for j in range(w):
        for i in range(h):
            if(not is_colored[i,j]):
                tlen=0
                gvals = np.zeros(((2 * wd + 1) ** 2,))
                for ii in range(max(0,i-wd),min(i+wd+1,h)):
                    for jj in range(max(0,j-wd),min(j+wd+1,w)):
                        if ii!=i or jj!=j:
                            row_inds[len]=consts_len
                            col_inds[len]=inds_matrix[ii,jj]
                            gvals[tlen]=ntsc_img[ii,jj,0]
                            len+=1
                            tlen+=1
                t_val=ntsc_img[i,j,0].copy()
                gvals[tlen]=t_val
                c_var=np.mean((gvals[:tlen+1]-np.mean(gvals[:tlen+1]))**2)
                csig=c_var*0.6
                mgv=min((gvals[:tlen+1]-t_val)**2)
                if csig<(-mgv/np.log(0.01)):
                    csig=-mgv/np.log(0.01)
                if csig<2e-5:
                    csig=2e-5
                gvals[:tlen]=np.exp(-(gvals[:tlen]-t_val)**2/csig)
                gvals[:tlen]=gvals[:tlen]/np.sum(gvals[:tlen])
                vals[len-tlen:len]=-gvals[:tlen]
            row_inds[len]=consts_len
            col_inds[len]=inds_matrix[i,j]
            vals[len]=1
            len += 1
            consts_len+=1

    vals=vals[:len]
    col_inds=col_inds[:len]
    row_inds=row_inds[:len]
    A=sparse.csr_matrix((vals,(row_inds,col_inds)),shape=(consts_len,h*w))
    b=np.zeros((A.shape[0]))
    for c in [1,2]:
        cur_img=ntsc_img[:,:,c].copy()
        b[labl_inds]=cur_img.reshape(h*w,order='F')[labl_inds]
        new_vals=linalg.spsolve(A,b)
        dst_img[:,:,c]=new_vals.reshape(h,w,order='F')
    r,g,b=yiq_to_rgb(dst_img[:,:,0],dst_img[:,:,1],dst_img[:,:,2])
    return cv2.merge((b,g,r))

if __name__=='__main__':
    src=cv2.imread('colorization_img/example.bmp',1)
    marked=cv2.imread('colorization_img/example_marked.bmp',1)
    cv2.imshow('src',src)
    cv2.imshow('marked',marked)
    dst_img=(colorize(src,marked)*255).astype(np.uint8)
    cv2.imshow('dst',dst_img)
    cv2.waitKey(0)
    cv2.imwrite('dst.jpg',dst_img)
