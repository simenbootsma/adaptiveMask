import numpy as np
import cv2 as cv
from multiprocessing import Process, Queue
import time
from scipy.fft import fft, fftfreq
import os
from datetime import datetime

SCREEN_RES = (1200, 1920)
USE_CAMERA = False  # is camera connected?
SAVE_FOLDER = os.path.abspath(os.getcwd()) + '/calibration_files/'
man_img = 255 * np.ones((SCREEN_RES[0], SCREEN_RES[1], 3))
man_points = []


def main():
    global SAVE_FOLDER, man_img, man_points

    # Calibrate
    result = auto_calibration()
    print(result)
    while not is_result_ok(result):
        man_img = 255 * np.ones((SCREEN_RES[0], SCREEN_RES[1], 3))
        man_points = []
        result = manual_calibration()
        print(result)

    # Save result
    calib = 'ADAPTIVE MASK CALIBRATION\n' + datetime.today().ctime() + '\n' + '-'*50
    calib += '\nsize : ({:d}, {:d})'.format(*result[0])  # size (w, h) of view box on screen
    calib += '\ncenter : ({:d}, {:d})'.format(*result[1])  # center (x, y) of view box on screen
    calib += '\nrotation : {:d}'.format(result[2])  # rotation (in degrees) of view box relative to screen
    calib += '\nscreen_size : ({:d}, {:d})'.format(*SCREEN_RES)  # dimensions of the screen in pixels

    # size, center, rotation: how to transform such that it can be fitted in the screen

    tdy = datetime.today()
    filename = 'adaptive_mask_calibration_{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}.txt'.format(tdy.year, tdy.month, tdy.day, tdy.hour, tdy.minute, tdy.second)
    with open(SAVE_FOLDER + filename, 'w') as f:
        f.write(calib)
    if SAVE_FOLDER == '':
        SAVE_FOLDER = os.path.abspath(os.getcwd())
    print('Saved calibration as {:s}!'.format(SAVE_FOLDER + filename))

    polar_bear_check(result)


def auto_calibration():
    q = Queue()
    p0 = Process(target=show_mask, args=(q,))
    p1 = Process(target=auto_calib_worker, args=(q,))

    p0.start()
    p1.start()
    p0.join()
    p1.join()

    result = q.get()

    print("Calibration DONE!")
    return result


def auto_calib_worker(queue):
    # Grid for size calibration
    screen = np.ones(SCREEN_RES)
    sq_sz = 50
    for i in range(0, screen.shape[0], sq_sz):
        screen[i - 3:i + 3, :] = 0
    for j in range(0, screen.shape[1], sq_sz):
        screen[:, j-3:j+3] = 0
    queue.put(screen)

    time.sleep(1)

    img = take_image(screen)

    ret, otsu = cv.threshold(img.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    otsu = cv.blur(otsu, (13, 13))

    ffts = [np.abs(fft(np.sum(otsu, axis=i)))[1:otsu.shape[1-i]//2] for i in [0, 1]]
    fftfreqs = [fftfreq(otsu.shape[1-i], 1)[1:otsu.shape[1-i]//2] for i in [0, 1]]
    zoom = [1/f[np.argmax(y)]/sq_sz for y, f in zip(ffts, fftfreqs)]

    # Image for camera view localization
    screen = cv.imread('polar_bear_mosaic.jpg', cv.IMREAD_UNCHANGED)
    # screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)
    screen = cv.rotate(screen[50:-50, 50:612], cv.ROTATE_90_COUNTERCLOCKWISE)
    screen = cv.resize(screen, SCREEN_RES[::-1])

    queue.put(screen)

    time.sleep(1)

    img = take_image(screen)
    img = cv.resize(img, (int(img.shape[1]/zoom[0]), int(img.shape[0]/zoom[1])))  # use zoom to rescale
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    locs = []
    rotation_codes = [None, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]
    for rot in rotation_codes:
        rot_img = img if rot is None else cv.rotate(img, rot)
        conv = cv.matchTemplate(screen, rot_img, cv.TM_SQDIFF)
        locs.append(cv.minMaxLoc(conv))  # min_val, max_val, min_loc, max_loc
    locs_sorted = sorted(locs, key=lambda tup: tup[0])
    opt_rot = rotation_codes[locs.index(locs_sorted[0])]
    top_left = locs_sorted[0][2]
    rot_img = img if opt_rot is None else cv.rotate(img, opt_rot)
    size = (rot_img.shape[1], rot_img.shape[0])
    print(size)
    center = (top_left[0] + size[0]//2, top_left[1] + size[1]//2)
    # view_box = [list(top_left), [top_left[0], top_left[1] + h], [top_left[0] + w, top_left[1] + h], [top_left[0] + w, top_left[1]]]
    queue.put((size, center, opt_rot * 90))


def show_mask(queue):
    cv_window()
    img = np.ones(SCREEN_RES)
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


def manual_calibration():
    global man_img, man_points
    cv_window()

    print('Click to set 4 corner points')
    cv.setMouseCallback('window', draw_circle)
    while len(man_points) < 4:
        cv.imshow("window", man_img)
        key = cv.waitKey(10)
        if key == 27:
            break

    print('Click on top edge')
    cv.setMouseCallback('window', draw_square)
    while len(man_points) < 5:
        cv.imshow("window", man_img)
        key = cv.waitKey(10)
        if key == 27:
            break
    cv.destroyWindow("window")
    center = np.mean(np.array(man_points[:4]), axis=0).astype(np.int32)
    x, y = list(zip(*man_points[:4]))
    size = int(abs(np.mean(np.sort(x)[:2]) - np.mean(np.sort(x)[2:]))), int(abs(np.mean(np.sort(y)[:2]) - np.mean(np.sort(y)[2:])))
    top_vec = 4 * np.array(man_points[4]) - np.sum(np.array(man_points[:4]), axis=0)
    top_vec = top_vec / np.sqrt(np.sum(top_vec**2))
    rot = (90 - np.arctan2(top_vec[1], top_vec[0]) * 180/np.pi) % 360
    rot = int(np.round(rot/90)) * 90  # round to nearest multiple of 90
    return size, center, rot


def draw_circle(event, x, y, flags, param):
    global man_img, man_points
    if event == cv.EVENT_MOUSEMOVE:
        man_img = 255 * np.ones((SCREEN_RES[0], SCREEN_RES[1], 3))
        cv.circle(man_img, (x, y), 100, (255, 0, 0), -1)
        for p in man_points:
            cv.circle(man_img, p, 100, (0, 0, 255), 4)
            cv.circle(man_img, p, 10, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        man_points.append([x, y])


def draw_square(event, x, y, flags, param):
    global man_img, man_points
    if event == cv.EVENT_MOUSEMOVE:
        man_img = 255 * np.ones((SCREEN_RES[0], SCREEN_RES[1], 3))
        cv.rectangle(man_img, (x-30, y-30), (x+30, y+30), (0, 255, 0), -1)
        for p in man_points:
            cv.circle(man_img, p, 100, (0, 0, 255), 4)
            cv.circle(man_img, p, 10, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        man_points.append([x, y])


def is_result_ok(result):
    size, center, rotation = result
    img = 255 * np.ones((SCREEN_RES[0], SCREEN_RES[1], 3))
    corner_points = np.array(center).astype(np.int32) + np.array([[-size[0]/2, -size[1]/2], [-size[0]/2, size[1]/2], [size[0]/2, size[1]/2], [size[0]/2, -size[1]/2]]).astype(np.int32)
    for p in corner_points:
        cv.circle(img, p, 100, (0, 0, 255), 4)
        cv.circle(img, p, 10, (0, 0, 255), -1)
    cv.polylines(img, [np.reshape(np.array(corner_points, dtype=np.int32), (-1, 1, 2))], True, (0, 0, 0), 30)

    cv_window()
    result = None
    print("happy? (y/n)")
    while True:
        cv.imshow("window", img)
        key = cv.waitKey()
        if chr(key).lower() == 'y':
            result = True
            break
        elif chr(key).lower() == 'n':
            result = False
            break
        elif key == 27:
            break
    cv.destroyWindow("window")
    return result


def polar_bear_check(result):
    size, center, rotation = result
    screen = cv.imread('polar_bear_mosaic.jpg', cv.IMREAD_UNCHANGED)
    screen = cv.rotate(screen[50:-50, 50:612], cv.ROTATE_90_COUNTERCLOCKWISE)
    screen = cv.resize(screen, SCREEN_RES[::-1])

    img = take_image(screen)

    rotation += 90  # TODO: check this in setup
    if rotation > 0:
        img = cv.rotate(img, rotation // 90 - 1)
    img = cv.resize(img, size)
    xslice = slice(center[0] - size[0] // 2, int(np.ceil(center[0] + size[0] / 2)))
    yslice = slice(center[1] - size[1] // 2, int(np.ceil(center[1] + size[1] / 2)))
    screen[yslice, xslice] = img

    cv_window()
    while True:
        cv.imshow('window', screen)
        key = cv.waitKey()
        if key == 27:
            break
    cv.destroyWindow('window')


def cv_window():
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    # cv.moveWindow("window", 900, 900)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


def take_image(screen=None):
    if not USE_CAMERA:
        return generate_synthetic_image(screen)


def generate_synthetic_image(screen):
    sz = (200, 500)
    r = (200, 400) # (np.random.randint(screen.shape[0]-sz[0]),  np.random.randint(screen.shape[1] - sz[1]))
    crop = screen[r[0]:(r[0]+sz[0]), r[1]:(r[1]+sz[1])]
    img = cv.resize(crop, dsize=(crop.shape[1]*6, crop.shape[0]*6))
    img = cv.blur(img, (51, 51))
    rot = cv.ROTATE_90_CLOCKWISE  #[None, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE][np.random.randint(4)]
    img = img if img is None else cv.rotate(img, rot)
    return img


if __name__ == '__main__':
    main()


