import cv2
import numpy as np
import matplotlib.pyplot as plt
class Decolor:
    def __init__(self,sigma=0.02):
        self.sigma=sigma

    def single_channel_grad_x(self,single_img):
        kernelx=np.array([[1.,-1.]])
        w=single_img.shape[1]
        k_rows,k_cols=kernelx.shape[:2]
        dest=cv2.filter2D(single_img,-1,kernelx,anchor=(k_cols-k_rows//2-1,k_rows-k_rows//2-1),delta=0.,borderType=cv2.BORDER_CONSTANT)
        dest[:,w-1]=0.
        print('dest.shape:',dest.shape)
        return dest

    def single_channel_grad_y(self,single_img):
        h=single_img.shape[0]
        kernely=np.array([[1.],[-1.]])
        k_rows,k_cols=kernely.shape[:2]
        dest=cv2.filter2D(single_img,-1,kernely,anchor=(k_cols-k_cols//2-1,k_rows-k_rows//2-1),delta=0.,borderType=cv2.BORDER_CONSTANT)
        dest[h-1,:]=0.
        return dest

    def cal_grad_vector_single_channel(self,single_img):
        grad_x=self.single_channel_grad_x(single_img)
        grad_y=self.single_channel_grad_y(single_img)
        print(grad_x.shape)
        return np.r_[grad_x.reshape(-1,),grad_y.reshape(-1,)]

    def cal_delta(self,src_img):
        L,a,b=cv2.split(cv2.cvtColor(src_img,cv2.COLOR_BGR2Lab))
        grad_L=self.cal_grad_vector_single_channel(L)
        grad_a=self.cal_grad_vector_single_channel(a)
        grad_b=self.cal_grad_vector_single_channel(b)
        return np.sqrt(grad_L**2+grad_a**2+grad_b**2)/100

    def cal_l(self,src_img):
        b, g, r = cv2.split(src_img)
        M = np.concatenate((r[:, :, np.newaxis], g[:, :, np.newaxis], b[:, :, np.newaxis],
                            (r * g)[:, :, np.newaxis], (r * b)[:, :, np.newaxis], (g * b)[:, :, np.newaxis],
                            (r * r)[:, :, np.newaxis], (g * g)[:, :, np.newaxis], (b * b)[:, :, np.newaxis]), axis=2)
        h,w=b.shape[:2]
        l=np.zeros((h*w*2,9))
        for i in range(9):
            l[:,i]=self.cal_grad_vector_single_channel(M[:,:,i])
        return l


    def weak_order(self,src_img):
        b, g, r = cv2.split(src_img)
        grad_b=self.cal_grad_vector_single_channel(b)
        grad_g=self.cal_grad_vector_single_channel(g)
        grad_r=self.cal_grad_vector_single_channel(r)
        level=1e-9
        t1=np.zeros_like(grad_b)
        t2=np.zeros_like(grad_b)
        t3=np.zeros_like(grad_b)
        tmp1=np.zeros_like(grad_b)
        tmp2=np.zeros_like(grad_b)
        tmp3=np.zeros_like(grad_b)
        t1[grad_b>level]=1.
        t2[grad_g>level]=1.
        t3[grad_r>level]=1.
        tmp1[grad_b<-level]=1.
        tmp2[grad_g<-level]=1.
        tmp3[grad_r<-level]=1.
        alpha=t1*t2*t3-tmp1*tmp2*tmp3
        return alpha

    def cal_energy(self,omega,delta,l):
        val=l.dot(omega)
        tmp=val-delta
        tmp1=val+delta
        energy=-1.*np.log(np.exp(-1.*tmp**2/self.sigma)+np.exp(-1.*tmp1**2/self.sigma))
        return np.mean(energy)

    def gray_contruct(self,omega,img):
        b, g, r = cv2.split(img)
        M = np.concatenate((r[:, :, np.newaxis], g[:, :, np.newaxis], b[:, :, np.newaxis],
                            (r * g)[:, :, np.newaxis], (r * b)[:, :, np.newaxis], (g * b)[:, :, np.newaxis],
                            (r * r)[:, :, np.newaxis], (g * g)[:, :, np.newaxis], (b * b)[:, :, np.newaxis]), axis=2)
        h,w=b.shape[:2]
        gray = M.dot(omega).reshape((h, w))
        min_g, max_g = np.min(gray), np.max(gray)
        gray = (((gray - min_g) / (max_g - min_g)) * 255).astype(np.uint8)
        return gray

    def update_omega(self,l,delta):
        B=np.zeros_like(l)
        for i in range(l.shape[1]):
            B[:,i]=l[:,i]*delta
        b=np.sum(B,axis=0)
        A=l.T.dot(l)
        omega=np.linalg.solve(A,b)
        print('omega:',omega)
        print(B.shape)
        _,X=cv2.solve(A,B.T,flags=cv2.DECOMP_NORMAL)
        return X

def decolor(src_img):
    max_iter=15
    tol=1e-4
    iter_count=0
    E=0.
    pre_E=np.inf
    src_img=src_img.astype(np.float32)/255
    obj=Decolor()
    alpha=obj.weak_order(src_img)
    l=obj.cal_l(src_img)
    delta=obj.cal_delta(src_img)

    Mt=obj.update_omega(l,delta)
    omega=np.zeros((9,))
    omega[:3]=0.33
    while np.sqrt((E-pre_E)**2>tol):
        iter_count+=1
        pre_E=E
        val = l.dot(omega)
        pos_G=((1+alpha)/2)*np.exp(-0.5*(val-delta)**2/(obj.sigma**2))
        neg_G=((1-alpha)/2)*np.exp(-0.5*(val+delta)**2/(obj.sigma**2))
        expSum=pos_G+neg_G
        tmp2=np.zeros_like(expSum)
        tmp2[expSum==0.]=1.
        expterm=(pos_G-neg_G)/(expSum+tmp2)
        print('Mt',Mt.shape)
        for i in range(9):
            omega[i]=np.sum(Mt[i,:]*expterm)
        E=obj.cal_energy(omega,delta,l)
        print(alpha)
        if iter_count>max_iter:
            break
    gray=obj.gray_contruct(omega,src_img)

    _,a,b=cv2.split(cv2.cvtColor((src_img*255).astype(np.uint8),cv2.COLOR_BGR2Lab))
    color_boost=cv2.cvtColor(cv2.merge((gray,a,b)),cv2.COLOR_Lab2BGR)
    return gray,color_boost

if __name__=='__main__':
    src_img = cv2.imread('decolor_img/6.png', 1)
    src_img=src_img
    gray,color_boost=decolor(src_img)
    hist=np.bincount(gray.reshape(-1,))
    plt.plot(range(len(hist)),hist)
    plt.show()
    plt.imshow(gray,cmap='gray')
    plt.show()

    plt.imshow(color_boost[:,:,::-1])
    plt.show()









