import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeAffine import LucasKanadeAffine
import cv2
import scipy.ndimage as nd
from SubtractDominantMotion import SubtractDominantMotion
import os



aerialseq = np.load("../data/aerialseq.npy")


index = [30,60,90,120]
for i in index:
    print(i)
    image1 = aerialseq[:, :, i-1]
    image2 = aerialseq[:, :, i]
    mask = SubtractDominantMotion(image1, image2)

    dd = image2.copy()
    dd = np.stack((dd, dd, dd), axis=2) * 255.0
    dd[:, :, 2] += (mask.astype(np.float32)) * 100.0

    vis = np.clip(dd, 0, 255).astype(np.uint8)
    fig = plt.figure()
    plt.imshow(vis)
    plt.show()
    plt.close()


# for i in range(aerialseq.shape[2]-1):
#
#     image1 = aerialseq[:, :, i]
#     image2 = aerialseq[:, :, i+1]
#     mask = SubtractDominantMotion(image1, image2)
#
#     dd = image2.copy()
#     dd = np.stack((dd, dd, dd), axis=2) * 255.0
#     dd[:, :, 2] += (mask.astype(np.float32)) * 100.0
#
#     vis = np.clip(dd, 0, 255).astype(np.uint8)
#     fig = plt.figure()
#     plt.imshow(dd)
#     plt.pause(0.0001)
#     plt.close(1)


