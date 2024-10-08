import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
from AdaptiveMask import AdaptiveMask

matplotlib.use('Qt5Agg')


def main():
    # test_data_from_contours()
    run_test(2)


def run_test(num):
    screen_res = (1920, 1080)
    calib_box = [500, 1400, 300, 840]  # part of the screen that is seen by camera, from calibration [xmin, xmax, ymin, ymax]
    s0, s1 = slice(calib_box[0], calib_box[1]), slice(calib_box[2], calib_box[3])

    tdata = np.load('test_data/test_data{:d}.npy'.format(num))
    mask = AdaptiveMask(1000, calib_box)

    # plt.ion()
    fig, axes = plt.subplots(2, 1)
    for n in range(tdata.shape[2]):
        # fig, ax = plt.subplots()
        screen = mask.screen

        cam = 255 - cv.resize(tdata[:, :, n], (calib_box[1]-calib_box[0], calib_box[3]-calib_box[2]))
        s = (cv.blur(screen[s1, s0], (21, 21))).astype(np.uint8)
        cam[cam == 255] = s[cam == 255]

        # screen[s1, s0] = cam

        axes[0].clear()
        axes[0].imshow(screen, cmap='gray')
        axes[0].set_title(str(n))
        axes[0].invert_yaxis()

        axes[1].clear()
        axes[1].imshow(cam, cmap='gray')
        axes[1].set_title(str(n))
        axes[1].invert_yaxis()

        mask.update(cam)

        # curve = mask.curve()
        # ax.plot(curve[:, 0], curve[:, 1], '-r')
        plt.pause(.1)
        # plt.show()
    plt.ioff()
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

