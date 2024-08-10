import numpy as np
import cv2 as cv
from multiprocessing import Process, Queue
import time
from glob import glob
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

USE_CAMERA = False  # is camera connected?
CALIBRATION_FILENAME = None  # leave None to use most recent


def main():
    q = Queue()
    p0 = Process(target=show_mask, args=(q,))
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


def show_mask(queue):
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


def masking_worker(queue, sz=400):
    calib = load_calibration()
    screen = np.zeros(calib['screen_size'] + [3])

    cv.circle(screen, calib['center'], int(min(calib['size']) / 2), (1, 1, 1), -1)
    queue.put(screen)

    print('press <any key> to start adaptive masking')
    cv.waitKey()

    dil_it = 5
    dil_sz = sz//dil_it
    st = time.time()
    while True:
        if queue.empty():
            img = take_image(screen)

            # plt.imshow(img)

            mask, ice = find_mask_and_ice(img)
            target = cv.dilate(ice, kernel=np.ones((dil_sz, dil_sz)), iterations=dil_it)

            plt.figure()
            plt.imshow(ice + target)
            plt.show()

            if calib['rotation'] > 0:
                target = cv.rotate(target, calib['rotation']//90 - 1)
            xslice = slice(calib['center'][0] - calib['size'][0]//2, calib['center'][0] + calib['size'][0]//2)
            yslice = slice(calib['center'][1] - calib['size'][1] // 2, calib['center'][1] + calib['size'][1] // 2)
            target = cv.cvtColor(target.astype(np.uint8), cv.COLOR_GRAY2RGB)

            screen = np.zeros(calib['screen_size'] + [3])
            screen[yslice, xslice] = cv.resize(target, calib['size'])

            queue.put(screen)
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


def take_image(screen=None):
    if not USE_CAMERA:
        return generate_synthetic_image(screen)


def generate_synthetic_image(screen):
    sz = (200, 500)
    # r = (np.random.randint(screen.shape[0]-sz[0]),  np.random.randint(screen.shape[1] - sz[1]))
    r = (200, 500)
    crop = screen[r[0]:(r[0]+sz[0]), r[1]:(r[1]+sz[1])]
    # cv.ellipse(crop, (sz[1]//2, sz[0]//2), (int(sz[0]/3), int(sz[0]/4)), 0, 0, 360, (0, 0, 0), -1)
    img = cv.resize(crop, dsize=(crop.shape[1]*6, crop.shape[0]*6))
    img = cv.blur(img, (31, 31))
    rot = cv.ROTATE_90_CLOCKWISE  # [None, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE][np.random.randint(4)]
    img = img if img is None else cv.rotate(img, rot)
    return img


def find_mask_and_ice(img):
    gray = np.mean(img, axis=2)
    blur = cv.GaussianBlur(gray, (5, 5), sigmaX=0)
    ret, otsu = cv.threshold(blur.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    s0, s1 = otsu.shape
    mask = np.zeros(otsu.shape)
    edges = [(i, j) for i in range(s0) for j in range(s1) if (i%(s0-1))*(j%(s1-1)) == 0]
    for i, j in edges:
        if mask[i, j] == 0 and otsu[i, j] == 0:
            empty_mat = np.zeros((s0 + 2, s1 + 2), dtype=np.uint8)
            _, _, m, _ = cv.floodFill(otsu.copy(), empty_mat, (j, i), 0)
            mask[m[1:-1, 1:-1] == 1] = 1
    mask = cv.dilate(mask, kernel=np.ones((31, 31)))
    ice = (1 - otsu.copy()/255)
    ice[mask==1] = 0
    return mask, ice


if __name__ == '__main__':
    main()

