import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
from AdaptiveMask import AdaptiveMask

matplotlib.use('Qt5Agg')


def main():
    # test_data_from_contours()
    run_test(3)


def run_test(num):
    screen_res = (1920, 1080)
    calib_box = [500, 1400, 300, 840]  # part of the screen that is seen by camera, from calibration [xmin, xmax, ymin, ymax]
    s0, s1 = slice(calib_box[0], calib_box[1]), slice(calib_box[2], calib_box[3])

    tdata = np.load('test_data/test_data{:d}.npy'.format(num))
    mask = AdaptiveMask(100, calib_box)

    # plt.ion()
    fig, ax = plt.subplots()
    for n in range(tdata.shape[2]):
        # fig, ax = plt.subplots()
        # set up screen
        screen = np.zeros((screen_res[1], screen_res[0]), dtype=np.uint8)
        cam = 255 - cv.resize(tdata[:, :, n], (calib_box[1]-calib_box[0], calib_box[3]-calib_box[2]))
        screen[s1, s0] = cam

        # update screen with cpoints
        _, screen, _, _ = cv.floodFill(screen, np.zeros((screen.shape[0]+2, screen.shape[1]+2), dtype=np.uint8), (0, 0), 255)
        curve = mask.curve()
        for i in range(1, len(curve)+1):
            d = curve[i%len(curve)] - curve[i - 1]
            for x in np.linspace(0, 1, int(np.sqrt(d[0]**2 + d[1]**2)) * 2):
                jj, ii = int(curve[i-1, 0] + x * d[0]), int(curve[i-1, 1] + x * d[1])
                screen[ii, jj] = 0

        _, screen, _, _ = cv.floodFill(screen, np.zeros((screen.shape[0] + 2, screen.shape[1] + 2), dtype=np.uint8), (0, 0), 0)

        ax.clear()
        ax.imshow(screen, cmap='gray')
        ax.set_title(str(n))
        ax.invert_yaxis()

        # update control points
        mask.update(screen[s1, s0])

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

