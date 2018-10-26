import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt

def InverseCompositionAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M_initial = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    threshold = 0.02
    W, H = It.shape
    image_grad_y = cv2.Sobel(It1, -1, 1, 0, ksize=1)
    image_grad_x = cv2.Sobel(It1, -1, 0, 1, ksize=1)
    d_p = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    mask = np.ones((W, H))

    y = np.linspace(0, W - 1, W)
    x = np.linspace(0, H - 1, H)
    X, Y = np.meshgrid(x, y)
    X_col = X.reshape(-1)
    Y_col = Y.reshape(-1)

    while np.linalg.norm(d_p) >= threshold:
        maskWarp = scipy.ndimage.affine_transform(mask, M, output_shape=(W, H))
        pixel_It = It * maskWarp
        pixel_It = pixel_It.reshape(-1)
        grad_x = (image_grad_x*maskWarp).reshape(-1)
        grad_y = (image_grad_y*maskWarp).reshape(-1)
        pixel_It1 = scipy.ndimage.affine_transform(It1, M, output_shape=(W, H)).reshape(-1)
        A = np.array([np.multiply(grad_x,X_col), np.multiply(grad_x,Y_col), grad_x, np.multiply(grad_y,X_col), np.multiply(grad_y,Y_col), grad_y])
        A = A.transpose()
        b = pixel_It1-pixel_It
        d_p = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)), np.transpose(A)), b)
        dM = np.array([[1.0 + d_p[1], d_p[0], d_p[2]], [d_p[4], 1.0 + d_p[3], d_p[5]], [0.0, 0.0, 1.0]])
        M = np.matmul(M, np.linalg.inv(dM))

        print(M)

    return M

