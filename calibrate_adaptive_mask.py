import numpy as np
import cv2 as cv
import matplotlib
from multiprocessing import Process, Queue
from datetime import datetime
import os.path
from Camera import Camera
import sys
from scipy.optimize import least_squares


matplotlib.use('Qt5Agg')
mask_points = []
SAVE_FOLDER = 'temp_files_calibration/'
cam_settings = Camera.Settings()
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
    cal = {'screen_size': screen.shape, 'view_box': calib[0]}
    screen = screen_for_dot_calibration(cal)

    q = Queue()
    p0 = Process(target=show_screen, args=(screen, q))
    p1 = Process(target=take_photo, args=(q, ))
    p0.start()
    p1.start()
    p0.join()
    cam_img = q.get()
    p1.join()

    dot_cal = dot_calibration(cam_img, cal)

    # save result
    save_calibration(screen.shape, calib, keep_sides, dot_cal)


def calibrate(screen, cam_img, use_mask=False):
    # assumption: cam image is larger than the same part displayed on screen
    if use_mask:
        mask0, mask1 = set_rect_mask(cam_img)
        cam_img = cam_img[mask0[1]:mask1[1], mask0[0]:mask1[0]]

    result = None
    pad_screen = np.random.randint(0, 255, (screen.shape[0]*2, screen.shape[1]*2, screen.shape[2]), dtype=np.uint8)
    pad_screen[screen.shape[0]//2:3*screen.shape[0]//2, screen.shape[1]//2:3*screen.shape[1]//2] = screen
    for scale in np.linspace(.2, 1.0, 40):
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


def dots_for_dot_calibration(calib):
    vx0, vx1, vy0, vy1 = calib['view_box']
    width, height = vx1 - vx0, vy1 - vy0
    dot_dist = min(width, height) // 10  # distance between dots in pixels
    dotx = np.arange(-width//(2*dot_dist)+1, width//(2*dot_dist)+1) * dot_dist
    doty = np.arange(-height//(2*dot_dist)+1, height//(2*dot_dist)+1) * dot_dist
    return np.array([[x, y] for x in dotx for y in doty if not (x == y == 0)], dtype=np.int32)


def screen_for_dot_calibration(calib):
    screen = 255 * np.ones(calib['screen_size'])
    vx0, vx1, vy0, vy1 = calib['view_box']
    center, width, height = [(vx0 + vx1) // 2, (vy0 + vy1) // 2], vx1 - vx0, vy1 - vy0
    dr = min(width, height) // 100  # dot radius
    pnts = np.array([
        [center[0] - dr, center[1] - dr], [center[0] + dr, center[1] - dr], [center[0] + 3*dr, center[1]],
        [center[0]+dr, center[1]+dr], [center[0], center[1] + 3 * dr], [center[0]-dr, center[1]+dr]
    ], dtype=np.int32)  # corner points of rectangle with triangles on bottom and right side, clockwise starting from top left
    cv.fillPoly(screen, [pnts],  (0, 0, 0))

    style = {'radius': dr, 'color': (0, 0, 0), 'thickness': -1}
    for dot in dots_for_dot_calibration(calib):
        cv.circle(screen, np.array(center) + dot, **style)
    return screen


def dot_calibration(cam_img, calib):
    screen_dots = dots_for_dot_calibration(calib)
    cam_dots = get_dot_locations(cam_img)

    assert len(cam_dots) == len(screen_dots), "Not all dots are visible on camera image!"

    def transform(x, cd):
        s1, s2, theta = x
        S = np.array([[s1, 0], [0, s2]])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return S.dot(R.dot(cd))

    approx_scale = (np.max(cam_dots, axis=0) - np.min(cam_dots, axis=0)) / (np.max(screen_dots, axis=0) - np.min(screen_dots, axis=0))
    cam_dots = cam_dots / approx_scale
    # TODO: add approx rotation using vector between mean and median of center marker

    # sort cam dots based on screen dots
    ind = []
    for d in screen_dots:
        ind.append(np.argmin(np.sum((d - cam_dots)**2, axis=1)))
    assert len(ind) == len(np.unique(ind)), "Non-unique mapping..."
    cam_dots = cam_dots[ind]

    # optimize parameters to minimize distance between cam dots and screen dots
    x0 = [approx_scale[0], approx_scale[1], 0]  # initial guess
    bounds = ([0, 0, -np.pi], [np.inf, np.inf, np.pi])  # limits
    res = least_squares(lambda x: np.sum((screen_dots.T - transform(x, cam_dots.T))**2), x0, bounds=bounds)
    trans_mat = transform(res.x, np.identity(2))  # transformation matrix
    return trans_mat


def get_dot_locations(img):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    ret, otsu = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blobs = find_blobs(255 - otsu)

    blob_sizes = [len(b) for b in blobs]
    square = blobs[np.argmax(blob_sizes)]
    blobs.pop(np.argmax(blob_sizes))
    mean_sz, std_sz = np.mean(blob_sizes), np.std(blob_sizes)
    blobs = [b for b in blobs if mean_sz - 2*std_sz < len(b) < mean_sz + 2*std_sz]  # remove noise

    sq_loc = np.mean(square, axis=0)
    dots = [np.mean(b, axis=0) - sq_loc for b in blobs]
    return np.array(dots)


def find_blobs(bin_img, min_area=10):
    def mask_image(img, box_size=1, start_pos=None):
        def get_neighbours(tup, shape):
            x, y = tup
            adjacent = [(x - 1, y), (x + 1, y), (x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y + 1),
                        (x, y + 1), (x + 1, y + 1)]
            return [(ix, iy) for ix, iy in adjacent if (0 <= ix < shape[0]) and (0 <= iy < shape[1])]

        if start_pos is None:
            start_pos = (0, img.shape[1] // 2)

        # Find edges of initial mask, if available
        mask0 = np.zeros(img.shape, dtype=np.uint8)
        q = [start_pos]

        # Mask image
        cnt = 0
        max_val = 255 * box_size ** 2
        while len(q) > 0:
            p = q.pop(0)
            mask0[p[0], p[1]] = 1
            neighbours = get_neighbours(p, shape=mask0.shape)
            q += [nb for nb in neighbours if
                  (img[nb[0], nb[1]] == max_val) and (not mask0[nb[0], nb[1]]) and (nb not in q)]
            cnt += 1
        return mask0

    blobs = []
    mask = np.zeros(bin_img.shape)
    for i in range(bin_img.shape[0]):
        print("\rFinding blobs: {:.0f}%".format(100*i/bin_img.shape[0]), end='')
        for j in range(bin_img.shape[1]):
            if bin_img[i, j] and not mask[i, j]:
                m = mask_image(bin_img, start_pos=(i, j))
                n_pixels_on = np.sum(m)
                if n_pixels_on >= min_area:
                    blobs.append(np.argwhere(m > 0))
                    # blob_x = np.sum(np.sum(m, axis=0) * np.arange(m.shape[1])) / n_pixels_on
                    # blob_y = np.sum(np.sum(m, axis=1) * np.arange(m.shape[0])) / n_pixels_on
                    # blob_radius = np.sqrt(n_pixels_on / np.pi)
                    # blobs.append({"x": blob_x, "y": blob_y, "r": blob_radius})
                mask += m
    print('\nDone!')

    return blobs


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


def save_calibration(screen_size, calib, keep_sides, transform):
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
    calib += '\ntransform : ({:.3f}, {:.3f}, {:.3f}, {:.3f})'.format(transform[0, 0], transform[0, 1], transform[1, 0], transform[1, 1])  # transformation matrix from camera to screen

    tdy = datetime.today()
    filename = 'adaptive_mask_calibration_{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}.txt'.format(tdy.year, tdy.month, tdy.day, tdy.hour, tdy.minute, tdy.second)
    with open(SAVE_FOLDER + filename, 'w') as f:
        f.write(calib)
    if SAVE_FOLDER == '':
        SAVE_FOLDER = os.path.abspath(os.getcwd())
    print('Saved calibration as {:s}!'.format(SAVE_FOLDER + filename))


if __name__ == '__main__':
    main(sys.argv)


