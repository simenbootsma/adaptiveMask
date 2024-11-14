import numpy as np
import cv2 as cv
import matplotlib
from multiprocessing import Process, Queue
from datetime import datetime
import os.path
from Camera import Camera
import sys


matplotlib.use('Qt5Agg')
mask_points = []
SAVE_FOLDER = 'temp_files_calibration/'
cam_settings = Camera.Settings(aperture='2.5', shutter_speed='1/5', iso=160)
cam_control_cmd_path = 'C:/Program Files (x86)/digiCamControl/CameraControlCmd.exe'
cam = Camera(cam_control_cmd_path, save_folder=SAVE_FOLDER, image_type='NEF')
cam.setup(cam_settings)


def main(args):
    screen = cv.imread('polar_bear_mosaic.png')

    # show screen and take photo simultaneously
    q = Queue()
    p0 = Process(target=show_screen, args=(screen, q))
    p1 = Process(target=take_photo, args=(q, ))
    p0.start()
    p1.start()
    p0.join()
    cam_img = q.get()
    p1.join()

    print('took photo! starting calibration...')

    # find what part of the screen is viewed by camera
    use_mask = len(args) > 1 and ('mask' in args[1])
    calib = calibrate(screen, cam_img, use_mask=use_mask)

    # show result
    keep_sides = check_calibration(screen, cam_img, calib)

    # save result
    save_calibration(screen.shape, calib, keep_sides)


def calibrate_old(screen, cam_img, use_mask=False):
    # assumption: cam image is larger than the same part displayed on screen
    if use_mask:
        mask0, mask1 = set_rect_mask(cam_img)
        cam_img = cam_img[mask0[1]:mask1[1], mask0[0]:mask1[0]]

    result = None
    pad_screen = np.random.randint(0, 255, (screen.shape[0]*2, screen.shape[1]*2, screen.shape[2]), dtype=np.uint8)
    pad_screen[screen.shape[0]//2:3*screen.shape[0]//2, screen.shape[1]//2:3*screen.shape[1]//2] = screen
    for scale in np.linspace(.02, 1.0, 40):
        resized = cv.resize(cam_img, (int(cam_img.shape[1] * scale), int(cam_img.shape[0] * scale)))

        if pad_screen.shape[0] < resized.shape[0] or pad_screen.shape[1] < resized.shape[1]:
            # cam image is larger than padded screen
            break

        conv = cv.matchTemplate(pad_screen, resized, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(conv)  # min_val, max_val, min_loc, max_loc

        if result is None or max_val > result[0]:
            result = (max_val, max_loc, scale)

    (startX, startY) = result[1]
    (endX, endY) = (int(startX + cam_img.shape[1] * result[2]), int(startY + cam_img.shape[0] * result[2]))
    calib = [startX - screen.shape[1] // 2, endX - screen.shape[1] // 2, startY - screen.shape[0] // 2,
             endY - screen.shape[0] // 2]
    mask_box = [0, cam_img.shape[1], 0, cam_img.shape[0]] if not use_mask else [mask0[0], mask1[0], mask0[1], mask1[1]]
    return calib, mask_box


def calibrate(screen, cam_img, use_mask=False):
    # assumption: cam image is larger than the same part displayed on screen
    if use_mask:
        mask0, mask1 = set_rect_mask(cam_img)
        cam_img = cam_img[mask0[1]:mask1[1], mask0[0]:mask1[0]]

    max_scale = min(screen.shape[0]/cam_img.shape[0], screen.shape[1]/cam_img.shape[1])
    min_scale = max_scale / 10
    scale_step = 0.05

    # First run: large steps
    scales1 = np.arange(min_scale, max_scale, scale_step)
    values1 = np.zeros(scales1.size)
    for i in range(scales1.size):
        resized = cv.resize(cam_img, (int(cam_img.shape[1] * scales1[i]), int(cam_img.shape[0] * scales1[i])))
        conv = cv.matchTemplate(screen, resized, cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(conv)  # min_val, max_val, min_loc, max_loc
        values1[i] = max_val

    # Second run: refined steps
    opt_sc = scales1[np.argmax(values1)]
    scales2 = opt_sc + np.array([-1, -2/5, -1/5, 0, 1/5, 2/5, 1]) * scale_step
    values2 = np.nan * np.zeros(scales2.size)
    if np.argmax(values1) > 0:
        values2[0] = values1[np.argmax(values1)-1]
    if np.argmax(values1) < len(values1)-1:
        values2[-1] = values1[np.argmax(values1)+1]
    values2[scales2==0] = values1[np.argmax(values1)]
    for i in range(scales2.size):
        if np.isnan(values2[i]):
            resized = cv.resize(cam_img, (int(cam_img.shape[1] * scales2[i]), int(cam_img.shape[0] * scales2[i])))
            conv = cv.matchTemplate(screen, resized, cv.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv.minMaxLoc(conv)  # min_val, max_val, min_loc, max_loc
            values2[i] = max_val

    # Final run: find optimum
    opt_i = np.argmax(values2)
    assert 0 < opt_i < values2.size-1, "Peak on one of the sides"
    opt_sc, opt_val = parabolic_fit(scales2[opt_i-1:opt_i+2], values2[opt_i-1:opt_i+2])
    resized = cv.resize(cam_img, (int(cam_img.shape[1] * opt_sc), int(cam_img.shape[0] * opt_sc)))
    conv = cv.matchTemplate(screen, resized, cv.TM_CCOEFF_NORMED)
    _, _, _, opt_loc = cv.minMaxLoc(conv)  # min_val, max_val, min_loc, max_loc

    # plt.figure()
    # plt.plot(scales1, values1, '-o', markersize=2)
    # plt.plot(scales2, values2, '-o', markersize=2)
    # plt.plot(opt_sc, opt_val, '^g')
    #
    # plt.figure()
    # plt.imshow(resized)
    # plt.figure()
    # plt.imshow(screen)
    # plt.plot(opt_loc[0], opt_loc[1], 'or')
    # plt.show()

    (startX, startY) = opt_loc
    (endX, endY) = (int(startX + cam_img.shape[1] * opt_sc), int(startY + cam_img.shape[0] * opt_sc))
    calib = [startX, endX, startY, endY]
    mask_box = [0, cam_img.shape[1], 0, cam_img.shape[0]] if not use_mask else [mask0[0], mask1[0], mask0[1], mask1[1]]
    return calib, mask_box


def parabolic_fit(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

    xv = -B / (2*A)
    yv = C - B**2 / (4*A)
    return xv, yv


def check_calibration(screen, cam_img, calib):
    def on_click(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            xc, yc = (view_box[0] + view_box[1])/2, (view_box[2] + view_box[3])/2
            edge_centers = [np.array([view_box[0], yc]), np.array([xc, view_box[2]]),
                            np.array([view_box[1], yc]), np.array([xc, view_box[3]])]  # left, top, right, bottom
            ind = np.argsort([np.sum([(np.array([x, y] - ec/2)) ** 2]) for ec in edge_centers])[0]  # index of edge closest to clicked point
            keep_edge[ind] = not keep_edge[ind]  # toggle edge
            draw_edges()

    def draw_edges():
        # Redraw edges
        box_corners = [[view_box[0]//2, view_box[2]//2], [view_box[0]//2, view_box[3]//2],
                       [view_box[1]//2, view_box[2]//2], [view_box[1]//2, view_box[3]//2]]
        bc_ind = [[0, 1], [0, 2], [2, 3], [1, 3]]  # indices of box corners corresponding to edges: left, top, right, bottom
        for i in range(4):
            color = (0, 255, 0) if keep_edge[i] else (0, 0, 255)
            p1, p2 = box_corners[bc_ind[i][0]], box_corners[bc_ind[i][1]]
            cv.line(img, p1, p2, color=color, thickness=2)

    # draw a bounding box around the detected result and display the image
    screen_intensity = np.mean(screen)

    view_box, mask_box = calib
    cam_img = cam_img[mask_box[2]:mask_box[3], mask_box[0]:mask_box[1]]

    cam_img = cam_img.astype(np.float64) / np.mean(cam_img) * screen_intensity
    cam_img[cam_img > 255] = 255
    cam_img = cam_img.astype(np.uint8)

    # cv.rectangle(screen, (view_box[0], view_box[2]), (view_box[1], view_box[3]), (0, 0, 255), 2)
    resized_cam = cv.resize(cam_img, (view_box[1] - view_box[0], view_box[3] - view_box[2]))
    if (view_box[0] < 0) or (view_box[1] >= screen.shape[1]) or (view_box[2] < 0) or (view_box[3] >= screen.shape[0]):
        raise ValueError("View box lies outside of screen, use mask on camera image using the 'mask' flag")

    keep_edge = [True, True, True, True]  # will be changed by on_click()
    screen[view_box[2]:view_box[3], view_box[0]:view_box[1]] = resized_cam
    img = cv.resize(screen, (screen.shape[1] // 2, screen.shape[0] // 2))

    cv.namedWindow("Image", cv.WINDOW_NORMAL)
    cv.setMouseCallback('Image', on_click)
    draw_edges()
    while True:
        cv.imshow("Image", img)
        key = cv.waitKey(10)
        if key == 13 or key == 27 or key == 32:  # press enter/escape/space to quit
            break
    cv.destroyWindow("Image")
    return keep_edge


def show_screen(screen, queue):
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    cv.moveWindow("window", 900, 0)
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
    if photo is None:
        print("[take_photo]: ERROR! Photo is None, is camera connected?")
    queue.put(photo)


def set_rect_mask(cam_img):
    global mask_points
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    print('Click to set 2 corner points')
    cv.setMouseCallback('window', draw_circle, param=cam_img)
    while len(mask_points) < 2:
        cv.imshow("window", cam_img)
        key = cv.waitKey(10)
        if key == 27:
            break
    cv.destroyWindow('window')

    p0 = (min(mask_points[0][0], mask_points[1][0]), min(mask_points[0][1], mask_points[1][1]))
    p1 = (max(mask_points[0][0], mask_points[1][0]), max(mask_points[0][1], mask_points[1][1]))

    cv.namedWindow("window", cv.WINDOW_NORMAL)
    cv.rectangle(cam_img, p0, p1, (0, 0, 255), 2)
    cv.imshow('window', cam_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(10)
    return p0, p1


def draw_circle(event, x, y, flags, param):
    global mask_points
    if event == cv.EVENT_LBUTTONUP:
        mask_points.append([x, y])
        cv.circle(param, (x, y), 30, (0, 0, 255), -1)


def save_calibration(screen_size, calib, keep_sides):
    global SAVE_FOLDER

    view_box, mask_box = calib
    size = (view_box[1]-view_box[0], view_box[3]-view_box[2])
    center = (int((view_box[1] + view_box[0])/2), int((view_box[3] + view_box[2])/2))

    # Save result
    calib = 'ADAPTIVE MASK CALIBRATION\n' + datetime.today().ctime() + '\n' + '-'*50
    calib += '\nview_box : ({:d}, {:d}, {:d}, {:d})'.format(*view_box)  # coordinates (xmin, xmax, ymin, ymax) of view box on screen
    calib += '\nmask_box : ({:d}, {:d}, {:d}, {:d})'.format(*mask_box)  # coordinates (xmin, xmax, ymin, ymax) used to crop camera image
    calib += '\nsize : ({:d}, {:d})'.format(*size)  # size (w, h) of view box on screen
    calib += '\ncenter : ({:d}, {:d})'.format(*center)  # center (x, y) of view box on screen
    calib += '\nscreen_size : ({:d}, {:d})'.format(*screen_size)  # dimensions of the screen in pixels
    calib += '\nkeep_sides : ({:d}, {:d}, {:d}, {:d})'.format(*keep_sides)  # which sides to use in the masking 1 = use, 0 = ignore

    tdy = datetime.today()
    filename = 'adaptive_mask_calibration_{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}.txt'.format(tdy.year, tdy.month, tdy.day, tdy.hour, tdy.minute, tdy.second)
    with open(SAVE_FOLDER + filename, 'w') as f:
        f.write(calib)
    if SAVE_FOLDER == '':
        SAVE_FOLDER = os.path.abspath(os.getcwd())
    print('Saved calibration as {:s}!'.format(SAVE_FOLDER + filename))


if __name__ == '__main__':
    main(sys.argv)


