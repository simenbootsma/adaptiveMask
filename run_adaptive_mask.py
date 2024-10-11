import numpy as np
import cv2 as cv
from multiprocessing import Process, Queue
import time
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
from AdaptiveMask import AdaptiveMask

matplotlib.use('Qt5Agg')

CALIBRATION_FILENAME = None  # leave None to use most recent


def main():
    q = Queue()
    p0 = Process(target=show_screen_worker, args=(q,))
    p1 = Process(target=masking_worker, args=(q,))

    p0.start()
    p1.start()
    p0.join()
    p1.join()

    repeat = q.get()
    if repeat:
        print('Repeating')
        main()
    print("Exiting...")


def show_screen_worker(queue):
    cv_window()
    img = np.ones((1000, 1000, 3))

    while True:
        if not queue.empty():
            pkg = queue.get()
            if type(pkg) is np.ndarray:
                img = pkg
            else:
                queue.put(pkg)
                break
        cv.imshow("window", img)
        key = cv.waitKey(100)
        if key == 27:
            break
    cv.destroyWindow("window")


def masking_worker(queue):
    calib = load_calibration()

    # initialize adaptive mask
    mask = AdaptiveMask(**calib)
    queue.put(mask.screen)

    print('press <any key> to start adaptive masking')
    cv.waitKey()

    st = time.time()
    while True:
        if queue.empty():
            img = take_image()
            mask.update(img)
            queue.put(mask.screen)
            st = time.time()
        else:
            time.sleep(0.1)
        if time.time() - st > 10:
            break  # quit if queue has not been emptied in 10 seconds


def load_calibration():
    global CALIBRATION_FILENAME
    if CALIBRATION_FILENAME is None:
        CALIBRATION_FILENAME = sorted(glob('calibration_files/*.txt'))[-1]
    data = {}
    with open(CALIBRATION_FILENAME, 'r') as f:
        for ln in f.read().splitlines():
            if ' : ' in ln:
                k, val = ln.split(' : ')
                data[k] = int(val) if '(' not in val else [int(v) for v in val[1:-1].split(', ')]
    return data


def cv_window():
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    # cv.moveWindow("window", 900, 900)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


def take_image():
    pass


if __name__ == '__main__':
    main()

