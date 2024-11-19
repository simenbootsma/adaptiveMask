import numpy as np
import cv2 as cv
import os.path
from ManualMask import Cylinder
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

matplotlib.use('Qt5Agg')


# IMG_FOLDER = 'C:/Users/local.la/Documents/Masking/adaptiveMask/auto_images/'  # folder where camera saves images
IMG_FOLDER = 'test_folder/'
ARROW_UP, ARROW_DOWN, ARROW_LEFT, ARROW_RIGHT = chr(0), chr(1), chr(2), chr(3)


def main(save_contours=True):
    # initialize
    cyl = Cylinder(resolution=(1920, 1080))
    cyl.sensitivity = 10  # sensitivity in screen pixels
    cyl.transpose()
    cv_window()
    log_file = open('logs/log' + datetime_string() + '.txt', 'w')

    if save_contours:
        ic_folder = "auto_contours/ice_contours" + datetime_string()
        os.mkdir(ic_folder)
    else:
        ic_folder = None

    # start program
    img_count = 0
    auto_enabled = True
    while True:
        img_path = IMG_FOLDER + "_{:03d}.jpg".format(img_count)
        if auto_enabled:# and os.path.exists(img_path):
            img = fake_img(cyl, img_count)
            # img = cv.imread(img_path)
            img_count += 1

            # auto-update screen
            actions = compute_actions(img, save_folder=ic_folder)
            print(actions)
            for a in actions:
                cyl.handle_key(a)
            log_actions(log_file, actions, auto=True)

        key = cv.waitKey(10)
        if key == 27:
            break
        elif key == ord('a'):
            auto_enabled = not auto_enabled
            line = "Auto mode {:s}".format("enabled" if auto_enabled else "disabled")
            print(line)
            log_file.write("[{:s}] ".format(datetime.now().ctime()) + line + "\n")
        elif key != -1:
            cyl.handle_key(key)
            log_actions(log_file, [chr(key)], auto=False)
        cv.imshow("window", cyl.get_img())
    cv.destroyWindow("window")
    log_file.close()


def compute_actions(img, save_folder=None):
    """ Find which buttons should be pressed to improve masking. Assumes vertical cylinder suspended from the top.
    Saves ice contours in save_folder. """

    # settings
    lr_thresh = 0.05  # minimum difference in white area between left and right before moving laterally
    w_thresh = 80  # maximum difference in mask and ice width in camera pixels
    h_thresh = 80  # maximum distance between mask and ice tip in camera pixels
    iw_ratio_thresh = 0.1  # maximum difference in ice area to white area between top and bottom half

    # setup
    actions = []
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img[:10, :] = 255  # make top edge white, assuming ice object suspended from top
    mask, ice = find_mask_and_ice(img)
    mask[:10, :] = 1  # add top edge back in mask
    ice_edges = find_edges(ice, largest_only=True)
    mask_edges = find_edges(mask, remove_outside=True)

    if len(ice_edges) == 0:
        print("[compute_actions]: no ice detected")
        return actions

    # Save ice contour
    now = datetime.now()
    fname = "contour_h{:02d}m{:02d}s{:02d}_us{:06d}.npy".format(now.hour, now.minute, now.second, now.microsecond)
    np.save(save_folder + '/' + fname, ice_edges)

    M = 1 - (mask + ice)  # regions of mask and ice are 0, rest is 1

    # Move left/right?
    cx = int(np.mean(ice_edges[:, 0]))
    x_diff = (np.sum(M[:, cx:]) - np.sum(M[:, :cx]))/np.sum(M)  # normalised difference in white area between left and right side of the cylinder
    if abs(x_diff) > lr_thresh:
        actions.append(ARROW_LEFT if x_diff > 0 else ARROW_RIGHT)

    # Adjust width
    min_ind, max_ind = int(0.02 * len(ice_edges)), int(0.98 * len(ice_edges))
    sorted_ice_edges_x = np.sort(ice_edges[:, 0])
    max_width = np.mean(sorted_ice_edges_x[max_ind:]) - np.mean(sorted_ice_edges_x[:min_ind])  # difference between average top and bottom 2% of x-coordinates
    max_width_mask = np.max(mask_edges[:, 0]) - np.min(mask_edges[:, 0])
    width_diff = max_width_mask - max_width
    if abs(width_diff) > w_thresh:
        actions.append("W" if width_diff > 0 else "w")

    # Adjust height
    tip_y = int(np.mean(np.sort(ice_edges[:, 1])[max_ind:]))
    mask_tip_y = np.max(mask_edges[:, 1])
    height_diff = mask_tip_y - tip_y
    if abs(height_diff) > h_thresh:
        actions.append("H" if height_diff > 0 else "h")

    # Adjust curvature
    cy = int(np.mean(ice_edges[:, 1]))
    iw_ratio_top = np.sum(ice[:cy, :]) / np.sum(M[:cy, :])  # ice area to white area ratio of top half
    iw_ratio_bot = np.sum(ice[cy:tip_y, :]) / np.sum(M[cy:tip_y, :])  # ice area to white area ratio of bottom half
    iw_ratio_diff = iw_ratio_top - iw_ratio_bot
    if abs(iw_ratio_diff) > iw_ratio_thresh and ("h" not in actions and "H" not in actions):
        actions.append("k" if iw_ratio_diff > 0 else "K")
    return actions


def find_mask_and_ice(img):
    # assumes gray image
    ret, otsu = cv.threshold(img.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    s0, s1 = otsu.shape
    mask = np.zeros(otsu.shape)
    edge_centers = [(0, s1//2), (s0//2, 0), (s0-1, s1//2), (s0//2, s1-1)]  # left, top, right, bottom
    for i, j in edge_centers:
        if mask[i, j] == 0 and otsu[i, j] == 0:
            empty_mat = np.zeros((s0 + 2, s1 + 2), dtype=np.uint8)
            _, _, m, _ = cv.floodFill(otsu.copy(), empty_mat, (j, i), 0)
            mask[m[1:-1, 1:-1] == 1] = 1
    ice = (1 - otsu.copy()/255)
    ice[mask==1] = 0
    return mask, ice


def find_edges(img, largest_only=False, remove_outside=False):
    if largest_only:
        cont, hierarchy = cv.findContours(img.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(cont) == 0:
            return None  # no contours found

        idx = np.argmax([len(c) for c in cont])  # index of largest contour
        edges = np.reshape(cont[idx], (cont[idx].shape[0], 2))
        return edges
    else:
        conts, hierarchy = cv.findContours(img.astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        if len(conts) == 0:
            return None  # no contours found

        # stack together
        edges = np.array([0, 0])
        for c in conts:
            edges = np.vstack((edges, np.reshape(c, (c.shape[0], 2))))

        # remove box edges
        if remove_outside:
            edges = edges[edges[:, 0] > 0]
            edges = edges[edges[:, 0] < img.shape[1]-1]
            edges = edges[edges[:, 1] > 0]
            edges = edges[edges[:, 1] < img.shape[0]-1]
        return edges[1:, :]


def cv_window():
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    # cv.moveWindow("window", 900, 900)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


def fake_img(cyl, n=0):
    arr = np.load('test_data/test_data4.npy')
    ice = arr[:, :, n]
    screen = cv.cvtColor(cyl.get_img(), cv.COLOR_RGB2GRAY)
    img = cv.resize(screen, (2 * cyl.resolution[0], 2*cyl.resolution[1]))
    img[:1000, 1800:2200] -= np.flipud(ice)
    return np.stack((img, img, img), axis=-1)


def datetime_string():
    now = datetime.now()
    return "{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)


def log_actions(file, actions, auto=False):
    now = datetime.now()
    arrows = {'\x00': '↑', '\x01': '↓', '\x02': '←', '\x03': '→'}
    for a in actions:
        if a in arrows:
            a = arrows[a]
        line = "[{:s}] ".format(now.ctime()) + ("Auto-pressed" if auto else "Pressed") + " {:s}\n".format(a)
        file.write(line)
    file.write("\n")


def write_log(file, line):
    file.write("[{:s}] ".format(datetime.now().ctime()) + line)


if __name__ == '__main__':
    main()

