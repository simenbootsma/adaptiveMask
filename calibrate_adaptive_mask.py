import numpy as np
import cv2 as cv
import matplotlib
from multiprocessing import Process, Queue
from datetime import datetime
import os.path
from Camera import Camera


matplotlib.use('Qt5Agg')
mask_points = []
SAVE_FOLDER = 'calibration_files/'
cam_settings = Camera.Settings(aperture=2.5, shutter_speed='1/5', iso=160)
cam_control_cmd_path = 'C:/Program Files (x86)/digiCamControl/CameraControlCmd.exe'
cam = Camera(cam_control_cmd_path, save_folder=SAVE_FOLDER)
cam.setup(cam_settings)


def main():
    screen = cv.imread('polar_bear_mosaic.jpg')

    # show screen and take photo simultaneously
    q = Queue()
    p0 = Process(target=show_screen, args=(screen, q))
    p1 = Process(target=take_photo, args=(q, ))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    cam = q.get()

    # find what part of the screen is viewed by camera
    calib = calibrate(screen, cam, use_mask=True)

    # save result
    save_calibration(screen.shape, calib)

    # show result
    check_calibration(screen, cam, calib)


def calibrate(screen, cam, use_mask=False):
    # assumption: cam image is larger than the same part displayed on screen
    # TODO: rotate cam image using EXIF when necessary

    if use_mask:
        mask0, mask1 = set_rect_mask(cam)
        cam = cam[mask0[1]:mask1[1], mask0[0]:mask1[0]]

    result = None
    pad_screen = np.random.randint(0, 255, (screen.shape[0]*2, screen.shape[1]*2, screen.shape[2]), dtype=np.uint8)
    pad_screen[screen.shape[0]//2:3*screen.shape[0]//2, screen.shape[1]//2:3*screen.shape[1]//2] = screen
    for scale in np.linspace(.02, 1.0, 20):
        resized = cv.resize(cam, (int(cam.shape[1] * scale), int(cam.shape[0] * scale)))

        if pad_screen.shape[0] < resized.shape[0] or pad_screen.shape[1] < resized.shape[1]:
            # cam image is larger than padded screen
            break

        conv = cv.matchTemplate(pad_screen, resized, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(conv)  # min_val, max_val, min_loc, max_loc

        if result is None or max_val > result[0]:
            result = (max_val, max_loc, scale)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (startX, startY) = result[1]
    (endX, endY) = (int(startX + cam.shape[1] * result[2]), int(startY + cam.shape[0] * result[2]))
    calib = [startX - screen.shape[1] // 2, endX - screen.shape[1] // 2, startY - screen.shape[0] // 2,
             endY - screen.shape[0] // 2]
    return calib


def check_calibration(screen, cam, view_box):
    # draw a bounding box around the detected result and display the image
    cv.rectangle(screen, (view_box[0], view_box[2]), (view_box[1], view_box[3]), (0, 0, 255), 2)
    resized_cam = cv.resize(cam, (view_box[1]-view_box[0], view_box[3]-view_box[2]))
    screen[view_box[2]:view_box[3], view_box[0]:view_box[1]] = resized_cam
    cv.imshow("Image", screen)
    cv.waitKey(0)


def show_screen(screen, queue):
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    # cv.moveWindow("window", 900, 0)
    cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow('window', screen)

    while queue.empty():
        key = cv.waitKey(10)
        if key == 27:
            break
    cv.destroyWindow('window')


def take_photo(queue):
    image_name = cam.collection_name + '_' + str(cam.image_index) + cam.image_type
    cam.capture_single_image(autofocus=False)
    photo = cv.imread(SAVE_FOLDER + image_name)
    queue.put(photo)


def set_rect_mask(cam):
    global mask_points
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    print('Click to set 2 corner points')
    cv.setMouseCallback('window', draw_circle, param=cam)
    while len(mask_points) < 2:
        cv.imshow("window", cam)
        key = cv.waitKey(10)
        if key == 27:
            break
    cv.destroyWindow('window')

    p0 = (min(mask_points[0][0], mask_points[1][0]), min(mask_points[0][1], mask_points[1][1]))
    p1 = (max(mask_points[0][0], mask_points[1][0]), max(mask_points[0][1], mask_points[1][1]))

    cv.namedWindow("window", cv.WINDOW_NORMAL)
    cv.rectangle(cam, p0, p1, (0, 0, 255), 2)
    cv.imshow('window', cam)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(10)
    return p0, p1


def draw_circle(event, x, y, flags, param):
    global mask_points
    if event == cv.EVENT_LBUTTONUP:
        mask_points.append([x, y])
        cv.circle(param, (x, y), 30, (0, 0, 255), -1)


def save_calibration(screen_size, view_box):
    global SAVE_FOLDER

    size = (view_box[1]-view_box[0], view_box[3]-view_box[2])
    center = ((view_box[1] + view_box[0])/2, (view_box[3] + view_box[2])/2)

    # Save result
    calib = 'ADAPTIVE MASK CALIBRATION\n' + datetime.today().ctime() + '\n' + '-'*50
    calib += '\nview_box : ({:d}, {:d})'.format(*view_box)  # coordinates (xmin, xmax, ymin, ymax) of view box on screen
    calib += '\nsize : ({:d}, {:d})'.format(*size)  # size (w, h) of view box on screen
    calib += '\ncenter : ({:d}, {:d})'.format(*center)  # center (x, y) of view box on screen
    calib += '\nscreen_size : ({:d}, {:d})'.format(*screen_size)  # dimensions of the screen in pixels

    tdy = datetime.today()
    filename = 'adaptive_mask_calibration_{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}.txt'.format(tdy.year, tdy.month, tdy.day, tdy.hour, tdy.minute, tdy.second)
    with open(SAVE_FOLDER + filename, 'w') as f:
        f.write(calib)
    if SAVE_FOLDER == '':
        SAVE_FOLDER = os.path.abspath(os.getcwd())
    print('Saved calibration as {:s}!'.format(SAVE_FOLDER + filename))


if __name__ == '__main__':
    main()


