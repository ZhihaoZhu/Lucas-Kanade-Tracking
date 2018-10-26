import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2


def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    # Put your implementation here

    p = p0
    threshold = 0.001

    x = np.linspace(0, It.shape[1]-1, It.shape[1])
    y = np.linspace(0, It.shape[0]-1, It.shape[0])
    interp_spline_It = RectBivariateSpline(y, x, It)

    image_grad_x = cv2.Sobel(It1, -1, 1, 0)
    image_grad_y = cv2.Sobel(It1, -1, 0, 1)
    image_grad = np.concatenate((image_grad_x, image_grad_y), axis=-1)
    x1 = np.linspace(0, It1.shape[1]-1, It1.shape[1])
    y1 = np.linspace(0, It1.shape[0]-1, It1.shape[0])
    interp_spline = RectBivariateSpline(y1, x1, It1)
    interp_gradx = RectBivariateSpline(y1, x1, image_grad_x)
    interp_grady = RectBivariateSpline(y1, x1, image_grad_y)

    x = np.linspace(rect[0], rect[2], rect[2] - rect[0] + 1)
    y = np.linspace(rect[1], rect[3], rect[3] - rect[1] + 1)
    It_pixel = interp_spline_It(y, x)
    It_pixel = It_pixel.reshape((-1, 1))

    dp = np.array([200,200])

    while np.linalg.norm(dp)>=threshold:

        '''
            Get the It1_pixel
        '''
        xt1 = x + p[0]
        yt1 = y + p[1]
        It1_pixel = interp_spline(yt1, xt1)
        It1_pixel = It1_pixel.reshape((-1,1))

        '''
            Get the gradient
        '''
        grad_x = interp_gradx(yt1, xt1)
        grad_y = interp_grady(yt1, xt1)
        grad_x = grad_x.reshape((-1,1))
        grad_y = grad_y.reshape((-1,1))

        '''
            Get the A and b, p
        '''
        A = np.concatenate((grad_x,grad_y),axis=1)
        b = It_pixel-It1_pixel
        dp = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)), np.transpose(A)), b)
        dp = dp.reshape(2)
        p = p+dp


    return p
