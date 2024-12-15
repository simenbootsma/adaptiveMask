import numpy as np
import cv2 as cv
import os.path
from ManualMask import Cylinder
from AutoMask import AutoMask
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import time
from glob import glob
import shutil

matplotlib.use('Qt5Agg')

DEMO = False  # run mask with existing data
IMG_FOLDER = '/Users/simenbootsma/Documents/PhD/Work/Vertical cylinder/ColdRoom/ColdVC_20241215/'  # folder where images ares saved
ONEDRIVE_FOLDER = '/Users/simenbootsma/OneDrive - University of Twente/VC_coldroom/ColdVC_20241215/'  # folder for communicating with external computer
PREV_CONTOUR_LENGTH = None


def main(save_contours=True):
    # initialize
    mask = AutoMask()

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
                img = fake_img(mask, img_count)  # for testing purposes
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

            # TODO: handle auto-update
            mask.update(img)

        # TODO: handle commands

        # handle key presses
        key = cv.waitKey(10)
        if key == 27:
            break
        elif key == ord('a'):
            auto_enabled = not auto_enabled
            line = "Auto mode {:s}".format("enabled" if auto_enabled else "disabled")
            print(line)
            log_file.write("[{:s}] ".format(datetime.now().ctime()) + line + "\n")

        # show screen
        cv.imshow("window", mask.get_img())

    cv.destroyWindow("window")
    log_file.close()


def cv_window():
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    cv.moveWindow("window", 900, 400)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


def fake_img(cyl, n=0):
    # return np.flipud(np.transpose(plt.imread('/Users/simenbootsma/OneDrive - University of Twente/VC_coldroom/ColdVC_20241128' + '/jpg/IMG_{:05d}.jpg'.format(n+9)), (1, 0, 2)))

    # files = sorted(glob('auto_contours/ice_contours20241128_104800/*.npy'))
    files = sorted(glob('auto_contours/ice_contours20241213_173608/*.npy'))
    n = min(len(files)-1, n)
    c = np.load(files[n])

    # ice = 255 * np.ones((4128, 2752), np.uint8)
    ice = 255 * np.ones((8256, 5504), np.uint8)
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
    allowed_actions = [c for c in 'swhkmg']
    actions = []
    try:
        file = open(filename, 'r')
        for ln in file.readlines():
            ln = ln.replace('\n', '')
            if '(' in ln:
                for c in "'()":
                    ln = ln.replace(c, '')
                key, val = ln.split(', ')
                if (key in allowed_actions) or ('target' in key) or ('threshold' in key):
                    actions.append((key, float(val)))
            else:
                actions += [c for c in ln if c in allowed_actions]
        file.close()
    except:
        print("Could not read command '{:s}'".format(filename))
    return actions


if __name__ == '__main__':
    main()

