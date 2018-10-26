import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
import cv2
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from InverseCompositionAffine import InverseCompositionAffine
import scipy.ndimage

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1
	# Output:
	#	mask: [nxm]
    # put your implementation here

    threshold = 0.25



    M = LucasKanadeAffine(image1, image2)
    print(M)

    '''
        Uncomment the following code to use the inverse composition method
    '''
    # M = InverseCompositionAffine(image1, image2)
    # print(M)


    image1_warped = scipy.ndimage.affine_transform(image2, M, output_shape=(image2.shape[0], image2.shape[1]))
    im_diff = np.absolute(image1_warped - image1)
    mask = im_diff > threshold
    mask = scipy.ndimage.binary_dilation(mask,structure=np.ones((10,10)))

    return mask




