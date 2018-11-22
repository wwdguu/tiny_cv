import cv2
import numpy as np
import matplotlib.pyplot as plt


def cal_grad_vector_single_channel(single_img):
    h,w=single_img.shape[:2]
    kernelx = np.array([[1., -1.]])
    kx_rows, kx_cols = kernelx.shape[:2]
    grad_x = cv2.filter2D(single_img, -1, kernelx, anchor=(kx_cols - kx_rows // 2 - 1, kx_rows - kx_rows // 2 - 1), delta=0.,
                        borderType=cv2.BORDER_CONSTANT)
    grad_x[:, w - 1] = 0.

    kernely = np.array([[1.], [-1.]])
    ky_rows, ky_cols = kernely.shape[:2]
    grad_y= cv2.filter2D(single_img, -1, kernely, anchor=(ky_cols - ky_cols // 2 - 1, ky_rows - ky_rows // 2 - 1), delta=0.,
                        borderType=cv2.BORDER_CONSTANT)
    grad_y[h-1,:]=0.
    print(grad_x.shape)
    return np.r_[grad_x.reshape(-1, ), grad_y.reshape(-1, )]


def cal_delta(src_img):
    L, a, b = cv2.split(cv2.cvtColor(src_img, cv2.COLOR_BGR2Lab))
    grad_L = cal_grad_vector_single_channel(L)
    grad_a = cal_grad_vector_single_channel(a)
    grad_b = cal_grad_vector_single_channel(b)
    return np.sqrt(grad_L ** 2 + grad_a ** 2 + grad_b ** 2) / 100


def cal_l_M(src_img):
    b, g, r = cv2.split(src_img)
    M = np.concatenate((r[:, :, np.newaxis], g[:, :, np.newaxis], b[:, :, np.newaxis],
                        (r * g)[:, :, np.newaxis], (r * b)[:, :, np.newaxis], (g * b)[:, :, np.newaxis],
                        (r * r)[:, :, np.newaxis], (g * g)[:, :, np.newaxis], (b * b)[:, :, np.newaxis]), axis=2)
    h, w = b.shape[:2]
    l = np.zeros((h * w * 2, 9))
    for i in range(9):
        l[:, i] = cal_grad_vector_single_channel(M[:, :, i])
    return l,M


def weak_order(src_img):
    b, g, r = cv2.split(src_img)
    grad_b = cal_grad_vector_single_channel(b)
    grad_g = cal_grad_vector_single_channel(g)
    grad_r = cal_grad_vector_single_channel(r)
    level = 5e-2
    t1 = np.zeros_like(grad_b)
    t2 = np.zeros_like(grad_b)
    t3 = np.zeros_like(grad_b)
    tmp1 = np.zeros_like(grad_b)
    tmp2 = np.zeros_like(grad_b)
    tmp3 = np.zeros_like(grad_b)
    t1[grad_b > level] = 1.
    t2[grad_g > level] = 1.
    t3[grad_r > level] = 1.
    tmp1[grad_b < -level] = 1.
    tmp2[grad_g < -level] = 1.
    tmp3[grad_r < -level] = 1.
    alpha = t1 * t2 * t3 - tmp1 * tmp2 * tmp3
    return alpha


def cal_energy(omega, delta, l,sigma=0.02):
    val = l.dot(omega)
    energy = -1. * np.log(np.exp(-0.5 * (val-delta) ** 2 / sigma**2) + np.exp(-0.5 * (val+delta) ** 2 /sigma**2))
    return np.mean(energy)

def gray_contruct(omega,M):
    h, w = M.shape[:2]
    gray = M.dot(omega).reshape((h, w))
    min_g, max_g = np.min(gray), np.max(gray)
    gray = (((gray - min_g) / (max_g - min_g)) * 255).astype(np.uint8)
    return gray


def cal_X(l, delta):
    B = np.zeros_like(l)
    for i in range(l.shape[1]):
        B[:, i] = l[:, i] * delta
    A = l.T.dot(l)
    _, X = cv2.solve(A, B.T, flags=cv2.DECOMP_NORMAL)
    return X

def decolor(src_img,sigma=0.02):
    max_iter=15
    tol=1e-8
    iter_count=0
    E=0.
    pre_E=np.inf
    src_img=src_img.astype(np.float32)/255
    alpha=weak_order(src_img)
    l,M=cal_l_M(src_img)
    delta=cal_delta(src_img)
    Mt=cal_X(l,delta)
    omega=np.zeros((9,))
    omega[:3]=0.33
    while np.sqrt((E-pre_E)**2>tol):
        iter_count+=1
        pre_E=E
        val = l.dot(omega)
        pos_G=((1+alpha)/2)*np.exp(-0.5*(val-delta)**2/(sigma**2))
        neg_G=((1-alpha)/2)*np.exp(-0.5*(val+delta)**2/(sigma**2))
        eps=np.zeros_like(pos_G)
        eps[(pos_G+neg_G)==0.]=1.
        expterm=(pos_G-neg_G)/(pos_G+neg_G+eps)
        for i in range(9):
            omega[i]=np.sum(Mt[i,:]*expterm)
        E=cal_energy(omega,delta,l)
        if iter_count>max_iter:
            break
    gray=gray_contruct(omega,M)
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









