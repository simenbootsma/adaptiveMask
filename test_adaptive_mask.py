import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
from AdaptiveMask import AdaptiveMask
from glob import glob

matplotlib.use('Qt5Agg')


def main():
    show_auto_contours("auto_contours/ice_contours20241119_104422/")
    # test_data_from_vc_contours()
    # show_test_data(4)
    # test_data_from_hc_contours()
    run_test(4)


def run_test(num):
    arr = np.load('test_data/test_data{:d}.npy'.format(num))

    for n in range(50):
        test0 = arr[:, :, 0]

        img = np.zeros((2000, 4000))
        img[:1000, 1800:2200] = np.flipud(test0)
        mask = 255 * np.ones(img.shape)
        mask[:1000, 1800:2200] = 0
        mask = cv.blur(mask, (51, 51))
        M = 255 - (mask + img)
        rgb = np.stack((M, M, M), axis=-1).astype(np.uint8)
        # plt.imshow(rgb)
        plt.imsave('test_folder/_{:03d}.jpg'.format(n), rgb)
    plt.show()


def show_auto_contours(folder):
    files = sorted(glob(folder + "*.npy"))
    contours = [np.load(f) for f in files]

    plt.figure()
    for c in contours:
        plt.plot(c[:, 0], c[:, 1])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.show()



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


def test_data_from_hc_contours():
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


def test_data_from_vc_contours():
    import pickle
    resolution = (1000, 400)
    with open('vc_contours/contours_a1.pkl', 'rb') as f:
        cntrs = pickle.load(f)
    fac = 0.8 * min(resolution[0]/(np.max(cntrs[0][:, 1]) - np.min(cntrs[0][:, 1])), resolution[1]/(np.max(cntrs[0][:, 0]) - np.min(cntrs[0][:, 0])))
    midpoint = np.array([resolution[1]/2 - fac * np.mean(cntrs[0][:, 0]), 0])
    cntrs = [(c * fac + midpoint).astype(np.int32) for c in cntrs]

    data = np.zeros((resolution[0], resolution[1], len(cntrs)), dtype=np.uint8)
    for n, c in enumerate(cntrs):
        for i, j in c:
            data[j, i, n] = 255
        seed_point = np.mean(c, axis=0, dtype=np.int32)
        _, img, _, _ = cv.floodFill(data[:, :, n].copy(), np.zeros((resolution[0] + 2, resolution[1] + 2), dtype=np.uint8), seed_point, 255)
        data[:, :, n] = np.flipud(img)
        print("\r{:d}/{:d}".format(n, len(cntrs)), end='')

    np.save('test_data/test_data4.npy', data)


if __name__ == '__main__':
    main()

