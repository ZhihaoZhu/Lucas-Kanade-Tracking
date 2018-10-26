import numpy as np
import matplotlib.pyplot as plt
from LucasKanade import LucasKanade
from matplotlib import animation
import matplotlib.patches as patches
import time

# write your script here, we recommend the above libraries for making your animation
carseq = np.load("../data/carseq.npy")


rect_0 = [59,116,145,151]

rect = rect_0
rect_tostore = []
rect_tostore.append(rect_0)
p = np.zeros(2)

for i in range(carseq.shape[2]-1):
    p_tmp = LucasKanade(carseq[:, :, i], carseq[:, :, i+1], rect, np.zeros(2))
    p = p_tmp + p
    rect = [rect_0[0]+p[0], rect_0[1]+p[1], rect_0[2]+p[0], rect_0[3]+p[1]]
    rect_tostore.append(rect)


    if i in [0,99,199,299,399]:
        im = carseq[:, :, i + 1]
        fig, ax = plt.subplots(1)
        plt.imshow(im, cmap='gray')
        rectangle = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0] + 1, rect[3] - rect[1] + 1, linewidth=1,
                                      edgecolor='y', facecolor='none')
        ax.add_patch(rectangle)
        plt.show()
        time.sleep(3)
        plt.close()

    # im = carseq[:, :, i + 1]
    # fig, ax = plt.subplots(1)
    # plt.imshow(im, cmap='gray')
    # rectangle = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0] + 1, rect[3] - rect[1] + 1, linewidth=1,
    #                               edgecolor='y', facecolor='none')
    # ax.add_patch(rectangle)
    # plt.pause(0.00001)
    # plt.close(1)


rect_tostore = np.array(rect_tostore)
np.save("../data/carseqrects.npy", rect_tostore)
print("done")




