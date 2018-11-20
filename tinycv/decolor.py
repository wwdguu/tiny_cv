import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg

def compute_delta_alpha_l(src_img):
    src_img=src_img.astype(np.float32)/255
    Lab=cv2.cvtColor(src_img,cv2.COLOR_BGR2LAB)
    L,a,b=cv2.split(Lab)
    h,w=src_img.shape[:2]
    delta=np.zeros((2*h*w),)
    alpha=0.5*np.ones_like(delta)
    index=0
    l=np.zeros((2*h*w,9))
    b, g, r = cv2.split(src_img)
    M=np.concatenate((r[:,:,np.newaxis],g[:,:,np.newaxis],b[:,:,np.newaxis],
                      (r*g)[:,:,np.newaxis],(r*b)[:,:,np.newaxis],(g*b)[:,:,np.newaxis],
                      (r*r)[:,:,np.newaxis],(g*g)[:,:,np.newaxis],(b*b)[:,:,np.newaxis]),axis=2)

    for i in range(h):
        for j in range(w):
           if (i+1)<h:
               delta[index]=np.sqrt((L[i,j]-L[i+1,j])**2+(a[i,j]-a[i+1,j])**2+(b[i,j]-b[i+1,j])**2)
               if r[i,j]<=r[i+1,j] and g[i,j]<=g[i+1,j] and b[i,j]<=b[i+1,j]:
                   alpha[index]=1.
               l[index,:]=M[i,j]-M[i+1,j]
               index+=1
           #else:
           #     delta[index]=0.
           #     l[index,:]=0.
           #     index+=1
    for i in range(h):
        for j in range(w):
           if (j+1)<w:
               delta[index] = np.sqrt(
                   (L[i, j] - L[i, j+1]) ** 2 + (a[i, j] - a[i, j+1]) ** 2 + (b[i, j] - b[i, j+1]) ** 2)
               if r[i,j]<=r[i,j+1] and g[i,j]<=g[i,j+1] and b[i,j]<=b[i,j+1]:
                   alpha[index]=1.
               l[index, :] = M[i, j] - M[i, j+1]
               index += 1
           #else:
           #    delta[index]=0.
           #    l[index,:]=0.
           #    index+=1
    print(index)
    delta=delta[:index]/100
    alpha=alpha[:index]
    l=l[:index]
    return  delta,alpha,l,M

def compute_G(delta_g,delta,sigma):
    return np.exp(-((delta_g-delta)**2)/(sigma))

def update_beta(alpha,delta,omega,l):
    delta_g=l.dot(omega)
    sigma = float(0.69 * np.std(delta_g))
    sigma=2e-3
    p_G=compute_G(delta_g,delta,sigma)
    n_G=compute_G(delta_g,-delta,sigma)
    p_G[p_G<1e-5]=0.
    n_G[n_G<1e-5]=0.
    beta=alpha*p_G/(alpha*p_G+(1-alpha)*n_G+1e-20)
    print(beta[:10])
    return beta

def update_omega(delta,beta,l):
    A=l.T.dot(l)
    b=l.T.dot((2*beta-np.ones_like(beta))*(delta))
    omega=np.linalg.solve(A,b,)
    return omega



if __name__=='__main__':
    src_img=cv2.imread('decolor_img/6.png',1)
    h,w=src_img.shape[:2]
    delta,alpha,l,M=compute_delta_alpha_l(src_img)
    omega=np.zeros((9,))
    omega[:3]=0.33
    print(omega)
    for iter in range(15):
        beta=update_beta(alpha,delta,omega,l)
        omega=update_omega(delta,beta,l)
        print(omega)
    #print(omega)
    min_c,max_c=np.min(src_img),np.max(src_img)
    print(min_c,max_c)

    #omega = np.array([2.00, -3.32, 3.04, 6.10, -3.45, 2.94, -2.98, -1.67, -1.00])
    gray=M.dot(omega).reshape((h,w))
    #print(gray)
    min_g,max_g=np.min(gray),np.max(gray)
    gray=(((gray-min_g)/(max_g-min_g))*255).astype(np.uint8)
    hist=np.bincount(gray.reshape(-1,))
    plt.plot(range(len(hist)),hist)
    plt.show()
    plt.imshow(gray,cmap='gray')
    plt.show()
    print(gray.shape)

    gray2,_=cv2.decolor(src_img)
    hist = np.bincount(gray2.reshape(-1, ))
    plt.plot(range(len(hist)), hist)
    plt.show()
    min_c, max_c = np.min(gray2), np.max(gray2)
    print(min_c, max_c)
    plt.imshow(gray2,cmap='gray')
    plt.show()
    print(gray2.shape)




