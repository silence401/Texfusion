import os
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    pth = sys.argv[1]
    qua = np.load(pth)
    plt.imshow(qua)
    plt.show()
