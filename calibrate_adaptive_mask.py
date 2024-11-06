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

    # perform dotted screen calibration
    screen = screen_for_dot_calibration(calib)

    q = Queue()
    p0 = Process(target=show_screen, args=(screen, q))
    p1 = Process(target=take_photo, args=(q, ))
    p0.start()
    p1.start()
    p0.join()
    q.get()
    p1.join()

    # save result
    save_calibration(screen.shape, calib, keep_sides)


def calibrate(screen, cam_img, use_mask=False):
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


def screen_for_dot_calibration(calib):
    screen = 255 * np.ones(calib['screen_size'])
    vx0, vx1, vy0, vy1 = calib['view_box']
    center, width, height = [(vx0 + vx1) // 2, (vy0 + vy1) // 2], vx1 - vx0, vy1 - vy0
    dot_radius = min(width, height) // 100
    dot_dist = min(width, height) // 10
    cv.rectangle(screen, [center[0] - dot_radius, center[1] - dot_radius],
                 [center[0] + dot_radius, center[1] + dot_radius], (0, 0, 0), -1)

    style = {'radius': dot_radius, 'color': (0, 0, 0), 'thickness': -1}
    for i in range(0, width // 2 // dot_dist + 1):
        for j in range(0, height // 2 // dot_dist + 1):
            cv.circle(screen, [center[0] + i * dot_dist, center[1] + j * dot_dist], **style)
            cv.circle(screen, [center[0] - i * dot_dist, center[1] + j * dot_dist], **style)
            cv.circle(screen, [center[0] + i * dot_dist, center[1] - j * dot_dist], **style)
            cv.circle(screen, [center[0] - i * dot_dist, center[1] - j * dot_dist], **style)
    # cv.rectangle(screen, [vx0, vy0], [vx1, vy1], (0, 0, 255), 2)
    return screen


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


