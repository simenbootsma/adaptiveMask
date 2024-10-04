import numpy as np
import cv2 as cv
from multiprocessing import Process, Queue
import time
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
import pickle

matplotlib.use('Qt5Agg')


def main():
    # test_data_from_contours()
    show_test_data(3)


def show_test_data(num):
    arr = np.load('test_data/test_data{:d}.npy'.format(num))

    plt.ion()
    fig, ax = plt.subplots()
    for n in range(arr.shape[2]):
        ax.clear()
        ax.imshow(255-arr[:, :, n], cmap='gray')
        ax.set_title(str(n))
        ax.invert_yaxis()
        plt.pause(.1)
    plt.ioff()
    plt.show()


def test_data_from_contours():
    resolution = (600, 1000)
    cntrs = [np.load('hc_contours/S10/t{:04d}.npy'.format(i)) for i in range(0, 1711, 10)]
    fac = 0.8 * min(resolution) / (np.max(cntrs[0][:, 0]) - np.min(cntrs[0][:, 0]))
    midpoint = np.array([resolution[1]/2, resolution[0]/2])
    cntrs = [(c * fac + midpoint).astype(np.int32) for c in cntrs]

    data = np.zeros((resolution[0], resolution[1], len(cntrs)), dtype=np.uint8)
    for n, c in enumerate(cntrs):
        for i, j in c:
            data[j, i, n] = 255
        seed_point = np.mean(c, axis=0, dtype=np.int32)
        _, img, _, _ = cv.floodFill(data[:, :, n].copy(), np.zeros((resolution[0] + 2, resolution[1] + 2), dtype=np.uint8), seed_point, 255)
        data[:, :, n] = img
        print("\r{:d}/{:d}".format(n, len(cntrs)), end='')

    np.save('test_data/test_data3.npy', data)




if __name__ == '__main__':
    main()

