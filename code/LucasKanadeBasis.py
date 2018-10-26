import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def SD_Q(x, y, grad_x, grad_y, bases):
    '''
        注意此时的点X已经是为index为1，2，3等的整数了
    '''
    n = bases.shape[2]
    e = np.zeros(2)
    d = np.zeros((n,2))

    for i in range(n):
        p = bases[:,:,i].reshape(-1)
        q0 = grad_x.reshape(-1)
        q1 = grad_y.reshape(-1)
        d[i,0] = np.dot(p,q0)
        d[i,1] = np.dot(p,q1)
        e = e + np.array([bases[x,y,i]*d[i,0], bases[x,y,i]*d[i,1]]).reshape(2)

    p = bases[x,y,:]
    q0 = d[:,0]
    q1 = d[:,1]
    e = np.array([np.dot(p,q0), np.dot(p,q0)])

    result = np.array([grad_x[x,y], grad_y[x,y]]) - e
    return result


def LucasKanadeBasis(It, It1, rect, bases, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here

    p = p0
    threshold = 0.001
    dp = np.array([200, 200])
    W = 55
    H = 47

    x = np.linspace(0, It.shape[1], It.shape[1])
    y = np.linspace(0, It.shape[0], It.shape[0])
    interp_spline_It = RectBivariateSpline(y, x, It)
    image_grad_x = cv2.Sobel(It, -1, 1, 0)
    image_grad_y = cv2.Sobel(It, -1, 0, 1)
    interp_gradx = RectBivariateSpline(y, x, image_grad_x)
    interp_grady = RectBivariateSpline(y, x, image_grad_y)

    x1 = np.linspace(0, It1.shape[1], It1.shape[1])
    y1 = np.linspace(0, It1.shape[0], It1.shape[0])
    interp_spline_It1 = RectBivariateSpline(y1, x1, It1)

    x = np.linspace(rect[0], rect[2], W)
    y = np.linspace(rect[1], rect[3], H)

    It_pixel = interp_spline_It(y, x).reshape((-1, 1))
    grad_x = interp_gradx(y,x)
    grad_y = interp_grady(y,x)


    H_Q = 0
    SD = np.zeros((W*H,2))


    for i in range(H):
        for j in range(W):

            SD[i*W+j] = SD_Q(i, j, grad_x, grad_y, bases)
            H_Q = H_Q + np.dot(SD[i*W+j], SD[i*W+j])


    while np.linalg.norm(dp) >= threshold:

        xt1 = x + p[0]
        yt1 = y + p[1]
        It1_pixel = interp_spline_It1(yt1, xt1).reshape((-1, 1))

        b = (It1_pixel-It_pixel).reshape(-1)

        dp = np.array([np.dot(b, SD[:,0]), np.dot(b, SD[:,1])]/H_Q)
        p = p - dp

    print(p)

    return p
    
