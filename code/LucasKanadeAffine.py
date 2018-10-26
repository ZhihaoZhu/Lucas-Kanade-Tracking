import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage

def LucasKanadeAffine(It, It1):
    # Input:
    # 	It: template image
    # 	It1: Current image
    # Output:
    # 	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here

    p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    threshold = 0.02
    W,H = It.shape
    image_grad_y = cv2.Sobel(It1, -1, 1, 0, ksize=1)
    image_grad_x = cv2.Sobel(It1, -1, 0, 1, ksize=1)
    dp = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    mask = np.ones((W,H))

    y = np.linspace(0, W - 1, W)
    x = np.linspace(0, H - 1, H)
    X, Y = np.meshgrid(x, y)

    X_col = X.reshape(-1)
    Y_col = Y.reshape(-1)


    while np.linalg.norm(dp) >= threshold:
        maskWarp = scipy.ndimage.affine_transform(mask, M, output_shape=(W, H))
        pixel_It = It*maskWarp
        pixel_It = pixel_It.reshape(-1)
        pixel_It1 = scipy.ndimage.affine_transform(It1, M, output_shape=(W, H)).reshape(-1)
        grad_x = scipy.ndimage.affine_transform(image_grad_x, M, output_shape=(W, H)).reshape(-1)
        grad_y = scipy.ndimage.affine_transform(image_grad_y, M, output_shape=(W, H)).reshape(-1)
        A = np.array([np.multiply(grad_x,X_col), np.multiply(grad_x,Y_col), grad_x, np.multiply(grad_y,X_col), np.multiply(grad_y,Y_col), grad_y])
        A = A.transpose()
        b = pixel_It - pixel_It1
        dp = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)), np.transpose(A)), b)
        p = p+dp
        M = np.array([[1.0 + p[1], p[0], p[2]], [p[4], 1.0 + p[3], p[5]]])
    M = np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]]])

    return M
