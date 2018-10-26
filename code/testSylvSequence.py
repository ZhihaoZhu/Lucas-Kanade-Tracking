import numpy as np
import matplotlib.pyplot as plt
from LucasKanadeBasis import LucasKanadeBasis
from LucasKanade import LucasKanade
from matplotlib import animation
import matplotlib.patches as patches
import time

# write your script here, we recommend the above libraries for making your animation
sylvseq = np.load("../data/sylvseq.npy")
bases = np.load("../data/sylvbases.npy")

'''
    This is the pre-stored rect which is generated by traditional LK method
'''
rect_LK = np.load("../data/sylvseqrects_withLK.npy")


rect_0 = [101, 61, 155, 107]
rect = rect_0

rect_tostore = []
rect_tostore.append(rect_0)
p = np.zeros(2)

for i in range(sylvseq.shape[2]-1):
    print(i)
    p_tmp = LucasKanadeBasis(sylvseq[:, :, i], sylvseq[:, :, i+1], rect, bases, np.zeros(2))
    p = p_tmp + p

    rect = [rect_0[0]+p[0], rect_0[1]+p[1], rect_0[2]+p[0], rect_0[3]+p[1]]
    rect_tostore.append(rect)

    if i in [0,199,299,349,399]:
        im = sylvseq[:, :, i + 1]
        fig, ax = plt.subplots(1)
        plt.imshow(im, cmap='gray')

        rectangle = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0] + 1, rect[3] - rect[1] + 1, linewidth=1,
                                      edgecolor='y', facecolor='none')
        ax.add_patch(rectangle)

        rectangle_LK = patches.Rectangle((rect_LK[i + 1, 0], rect_LK[i + 1, 1]),
                                         rect_LK[i + 1, 2] - rect_LK[i + 1, 0] + 1,
                                         rect_LK[i + 1, 3] - rect_LK[i + 1, 1] + 1, linewidth=1,
                                                                       edgecolor='g', facecolor='none')
        ax.add_patch(rectangle_LK)
        plt.show()
        plt.close()


rect_tostore = np.array(rect_tostore)
np.save("../data/sylvseqrects.npy", rect_tostore)
print("done")



