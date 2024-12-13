import numpy as np
import cv2 as cv
import os.path
from ManualMask import Cylinder
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import time
from glob import glob
import shutil

matplotlib.use('Qt5Agg')

DEMO = False  # run mask with existing data
IMG_FOLDER = '/Users/simenbootsma/Documents/PhD/Work/Vertical cylinder/ColdRoom/ColdVC_20241212/'  # folder where images ares saved
ONEDRIVE_FOLDER = '/Users/simenbootsma/OneDrive - University of Twente/VC_coldroom/ColdVC_20241212/'  # folder for communicating with external computer
ARROW_UP, ARROW_DOWN, ARROW_LEFT, ARROW_RIGHT = 'u', 'd', 'M', 'm'
PREV_CONTOUR_LENGTH = None

TARGETS = {
    'm': 0,    # intended difference in white area between left and right in camera pixels
    'w': 350,  # intended difference in mask and ice width in camera pixels
    'h': 200,  # intended distance between mask and ice tip in camera pixels
    'k': 1,    # intended white area ratio between tip and full cylinder
}

THRESHOLDS = {
    'm': 50,   # minimum deviation from target in camera pixels
    'w': 50,  # maximum deviation from target in camera pixels
    'h': 50,  # maximum deviation from target in camera pixels
    'k': 0.5,  # maximum deviation from target ratio
}


def main(save_contours=True):
    global THRESHOLDS
    # initialize
    cyl = Cylinder(resolution=(1920, 1080))
    cyl.sensitivity = 10  # sensitivity in screen pixels
    cyl.transpose()
    cv_window()
    log_file = open('logs/log' + datetime_string() + '.txt', 'w')
    for s in ['jpg', 'updates', 'commands']:
        if not os.path.exists(ONEDRIVE_FOLDER + s):
            os.mkdir(ONEDRIVE_FOLDER + s)

    if save_contours:
        ic_folder = "auto_contours/ice_contours" + datetime_string()
        os.mkdir(ic_folder)
    else:
        ic_folder = None

    # start program
    img_count = 0
    auto_enabled = True
    img_paths = glob(IMG_FOLDER + '*.JPG')
    command_paths = glob(ONEDRIVE_FOLDER + 'commands/*.txt')

    if DEMO:
        plt.ion()
        fig, ax = plt.subplots()
    while True:
        new_images = sorted([fn for fn in glob(IMG_FOLDER + "*.JPG") if fn not in img_paths])
        if auto_enabled and (len(new_images) > 0 or DEMO):
            if DEMO:
                img = fake_img(cyl, img_count)  # for testing purposes
                ax.clear()
                ax.imshow(img)
                ax.set_title('Iteration {:d}'.format(img_count))
                plt.pause(0.01)
            else:
                time.sleep(.5)
                img = cv.imread(new_images[0])
                img_paths.append(new_images[0])
                shutil.copyfile(new_images[0], ONEDRIVE_FOLDER + 'jpg/IMG_{:05d}.jpg'.format(img_count))
            img_count += 1

            try:
                # auto-update screen
                auto_actions, errors = compute_actions_fuzzy(img, save_folder=ic_folder, count=img_count, return_errors=True)
                for a in auto_actions:
                    cyl.handle_key(a)
                log_actions(log_file, auto_actions, auto=True)
                if errors is not None and not DEMO:
                    give_update(errors, cyl, img_count)
            except:
                print("An error occurred in auto-updating the screen")

        try:
            # check for external commands and update screen
            command_files = [fpath for fpath in sorted(glob(ONEDRIVE_FOLDER + 'commands/*.txt')) if fpath not in command_paths]
            command_actions = []
            for cf in command_files:
                print('received file!')
                time.sleep(0.5)  # wait before reading to make sure file is closed
                command_actions += read_command(cf)
                command_paths.append(cf)
            if len(command_actions) > 0:
                print(command_actions)
            for a in command_actions:
                if len(a) > 1 and 'threshold' in a[0]:
                    THRESHOLDS[a[0][0]] = a[1]
                elif len(a) > 1 and 'target' in a[0]:
                    TARGETS[a[0][0]] = a[1]
                elif len(a) > 1:
                    match a[0]:
                        case 'm':
                            cyl.set_center(int(a[1]))
                        case 'w':
                            cyl.set_width(int(a[1]))
                        case 'h':
                            cyl.set_height(int(a[1]))
                        case 'k':
                            cyl.set_curvature(a[1])
                else:
                    cyl.handle_key(a)
            log_actions(log_file, command_actions, auto=False)
        except:
            print("An error occurred in updating the mask via commands")

        # handle key presses
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

        # show screen
        cv.imshow("window", cyl.get_img())

    cv.destroyWindow("window")
    log_file.close()


def compute_actions_fuzzy(img, save_folder=None, count=None, return_errors=False):
    global PREV_CONTOUR_LENGTH, THRESHOLDS, TARGETS

    """ Find which buttons should be pressed to improve masking.
    NOTE: Assumes vertical cylinder suspended from the top.
    Saves ice contours in save_folder. """

    # settings
    sensitivity_small = 2
    sensitivity_large = 10

    # setup
    actions = []
    # img = img[:, 500:-500]
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img[:10, :] = 255  # make top edge white, assuming ice object suspended from top
    mask, ice = find_mask_and_ice(img)
    mask[:10, :] = 1  # add top edge back in mask
    ice_edges = find_edges(ice, largest_only=True)
    mask_edges = find_edges(mask, remove_outside=True)

    if ice_edges is None or (PREV_CONTOUR_LENGTH is not None and len(ice_edges) < PREV_CONTOUR_LENGTH/4):
        print("[compute_actions]: no ice detected")
        if return_errors:
            return ['w', 'h', 'K'], None
        return ['w', 'h', 'K']  # if no ice is detected, increase width and height, decrease curvature

    PREV_CONTOUR_LENGTH = len(ice_edges)

    if save_folder is not None:
        # Save ice contour
        now = datetime.now()
        fname = "contour_h{:02d}m{:02d}s{:02d}_us{:06d}.npy".format(now.hour, now.minute, now.second, now.microsecond)
        np.save(save_folder + '/' + fname, ice_edges)

    # Adjust width
    xmean_ice = np.mean(ice_edges[:, 0])
    dy = 10  # bin size in pixels
    ice_bins = [ice_edges[np.abs(ice_edges[:, 1] - j * dy) <= dy / 2, :] for j in range(int(img.shape[0] / dy))]
    ice_bins = [[b[b[:, 0] < xmean_ice], b[b[:, 0] > xmean_ice]] for b in ice_bins]  # split into left and right
    ice_widths = [np.nan if (len(rb) == 0 or len(lb) == 0) else np.max(rb[:, 0]) - np.min(lb[:, 0]) for lb, rb in
                  ice_bins]
    mask_bins = [mask_edges[np.abs(mask_edges[:, 1] - j * dy) <= dy / 2, :] for j in range(int(img.shape[0] / dy))]
    mask_bins = [[b[b[:, 0] < xmean_ice], b[b[:, 0] > xmean_ice]] for b in mask_bins]  # split into left and right
    mask_widths = [np.nan if (len(rb) == 0 or len(lb) == 0) else np.max(rb[:, 0]) - np.min(lb[:, 0]) for lb, rb in
                   mask_bins]

    width_diffs = np.array(mask_widths) - np.array(ice_widths)
    width_diffs = width_diffs[~np.isnan(width_diffs)]
    w_diff = np.sort(width_diffs)[int(.02 * len(width_diffs))]  # take value at 2% instead of min to ignore extreme values

    # Adjust position
    white_space_left = [np.nan if (len(ice_bins[j][0]) == 0 or len(mask_bins[j][0]) == 0)
                        else np.abs(np.min(ice_bins[j][0][:, 0]) - np.min(mask_bins[j][0][:, 0]))
                        for j in range(len(ice_bins))]
    white_space_right = [np.nan if (len(ice_bins[j][1]) == 0 or len(mask_bins[j][1]) == 0)
                         else np.abs(np.min(ice_bins[j][1][:, 0]) - np.min(mask_bins[j][1][:, 0]))
                         for j in range(len(ice_bins))]
    x_diff = np.nanmean(white_space_right) - np.nanmean(white_space_left)

    # Adjust height
    min_ind, max_ind = int(0.02 * len(ice_edges)), int(0.98 * len(ice_edges))
    tip_y = int(np.mean(np.sort(ice_edges[:, 1])[max_ind:]))
    mask_tip_y = np.max(mask_edges[:, 1])
    h_diff = mask_tip_y - tip_y

    # Adjust curvature
    M = 1 - (mask + ice)  # regions of mask and ice are 0, rest is 1
    avg_iw_ratio = np.sum(ice[:tip_y, :]) / np.sum(M[:tip_y, :])
    tip_iw_ratio = np.sum(ice[int(.9*tip_y):tip_y, :], axis=1) / np.sum(M[int(.9*tip_y):tip_y, :], axis=1)
    if np.max(tip_iw_ratio) > avg_iw_ratio:
        k_diff = avg_iw_ratio/np.max(tip_iw_ratio) - 1
    else:
        k_diff = avg_iw_ratio / np.min(tip_iw_ratio) - 1

    errors = {'m': x_diff - TARGETS['m'], 'h': h_diff - TARGETS['h'], 'w': w_diff - TARGETS['w'], 'k': k_diff - TARGETS['k']}

    # Make action list
    for k in errors:
        if abs(errors[k]) > THRESHOLDS[k]:
            s = sensitivity_large if abs(errors[k]) > 2 * THRESHOLDS[k] else sensitivity_small
            actions.append((k.upper() if errors[k] > 0 else k.lower(), s))

    err_str = [
        " ok " if abs(errors[k]) <= THRESHOLDS[k] else "{:.02f}".format(errors[k] / THRESHOLDS[k]) for k in errors
    ]

    # add colours
    for i in range(len(err_str)):
        if err_str[i] == ' ok ':
            err_str[i] = '\033[32m ok \033[0m'
        elif abs(float(err_str[i])) <= 2:
            err_str[i] = "\033[33m" + err_str[i] + "\033[0m"
        else:
            err_str[i] = "\033[31m" + err_str[i] + "\033[0m"
    count_str = "" if count is None else "[IMG {:d}]".format(count)
    print("\r"+count_str+" Errors  |  x: {:s}  | w: {:s}  | h: {:s}  | k: {:s} ".format(*err_str), end='')

    if return_errors:
        err_dct = {k: (errors[k], THRESHOLDS[k], TARGETS[k]) for k in errors}
        return actions, err_dct
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
    cv.moveWindow("window", 900, 400)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


def fake_img(cyl, n=0):
    # return np.flipud(np.transpose(plt.imread('/Users/simenbootsma/OneDrive - University of Twente/VC_coldroom/ColdVC_20241128' + '/jpg/IMG_{:05d}.jpg'.format(n+9)), (1, 0, 2)))

    files = sorted(glob('auto_contours/ice_contours20241128_104800/*.npy'))
    n = min(len(files)-1, n)
    c = np.load(files[n])

    ice = 255 * np.ones((4128, 2752), np.uint8)
    ice = cv.fillPoly(ice, [c.astype(np.int32)], (0, 0, 0))

    # arr = np.load('test_data/test_data7.npy')
    # n = min(n//2, arr.shape[-1]-1)
    # ice = arr[:, :, n]
    screen = cv.cvtColor(cyl.get_img(), cv.COLOR_RGB2GRAY)
    screen = cv.resize(screen, (ice.shape[1], ice.shape[0]))

    ice[screen == 0] = 0
    img = ice

    # Triangle
    # img = cv.fillPoly(img, [np.fliplr(np.array([[0, 1850], [0, 2150], [900, 2000], [0, 1850]]))], color=(0, 0, 0))
    return np.stack((img, img, img), axis=-1)


def datetime_string():
    now = datetime.now()
    return "{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)


def log_actions(file, actions, auto=False):
    now = datetime.now()
    arrows = {'\x00': '↑', '\x01': '↓', '\x02': '←', '\x03': '→'}
    for a in actions:
        if type(a) is tuple:
            a = a[0]
        if a in arrows:
            a = arrows[a]
        line = "[{:s}] ".format(now.ctime()) + ("Auto-pressed" if auto else "Pressed") + " {:s}\n".format(a)
        file.write(line)
    file.write("\n")


def write_log(file, line):
    file.write("[{:s}] ".format(datetime.now().ctime()) + line)


def give_update(errors, cyl, img_count):
    """ Write update for external computer, containing current errors and parameter values """
    f = open(ONEDRIVE_FOLDER + 'updates/update_{:04d}.txt'.format(img_count), 'w')

    parameters = ['sensitivity', 'center', 'width', 'height', 'blur', 'curvature', 'flipped', 'transposed', 'contrast']
    parameters = [p + ': ' + str(cyl.__getattribute__(p)) for p in parameters]
    errors = ['err_' + key + ': ' + str(val) for key, val in errors.items()]

    f.write("\n".join(parameters + errors))
    f.close()


def read_command(filename):
    allowed_actions = [c for c in 'swhkmg'] + ['wt', 'ht', 'mt', 'kt']
    actions = []
    try:
        file = open(filename, 'r')
        for ln in file.readlines():
            ln = ln.replace('\n', '')
            if '(' in ln:
                for c in "'()":
                    ln = ln.replace(c, '')
                key, val = ln.split(', ')
                if key in allowed_actions:
                    actions.append((key, float(val)))
            else:
                actions += [c for c in ln if c in allowed_actions]
        file.close()
    except:
        print("Could not read command '{:s}'".format(filename))
    return actions


if __name__ == '__main__':
    main()

