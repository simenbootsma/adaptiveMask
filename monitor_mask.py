import time
import numpy as np

from glob import glob
from datetime import datetime
import cv2 as cv
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
#['gtk3agg', 'gtk3cairo', 'gtk4agg', 'gtk4cairo', 'macosx', 'nbagg', 'notebook', 'qtagg', 'qtcairo', 'qt5agg', 'qt5cairo', 'tkagg', 'tkcairo', 'webagg', 'wx', 'wxagg', 'wxcairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']


FOLDER = "C:/Users/Simen/OneDrive - University of Twente/VC_coldroom/ColdVC_20241211/"  # must contain jpg, updates, commands folders
update_paths = glob(FOLDER + "updates/*.txt")


def main():
    img_paths = glob(FOLDER + "jpg/*.jpg")

    fig, axes = plt.subplots(1, 2)
    last_update = None

    cnt = 0
    while True:
        imgs = sorted(glob(FOLDER + "jpg/*.jpg"))
        if len(imgs) > 0 and (imgs[-1] not in img_paths or last_update is None):
            axes[0].clear()
            axes[1].clear()
            last_update = datetime.now()
            #img = cv.imread(imgs[-1]).transpose([1, 0, 2])
            img = cv.imread(imgs[-1])
            gray_img = np.mean(img, axis=2).astype(np.uint8)
            contour = find_edges(find_mask_and_ice(gray_img)[1], largest_only=True)
            axes[0].imshow(img)
            axes[1].imshow(img)
            axes[1].plot(contour[:, 0], contour[:, 1], '-r')
            axes[0].set_title(last_update.ctime())
            img_paths.append(imgs[-1])
            update_text = axes[0].text(70, 220, 'updated 0 s ago', va='center', ha='left', color=(.8, .8, .8), fontsize=12)

        read_update()

        mfc = 'r' if (cnt // 2) % 2 == 0 else 'k'
        axes[0].plot(100, 90, 'o', mec='r', mfc=mfc, markersize=10, mew=2)
        axes[0].text(200, 100, 'LIVE', va='center', ha='left', color='r', fontweight='bold', fontsize=16)
        if last_update is not None:
            diff = (datetime.now() - last_update).total_seconds()
            update_text.set_text('updated {:.0f} s ago'.format(diff))
        plt.pause(.1)
        cnt += 1


def find_mask_and_ice(img):
    # assumes gray image
    ret, otsu = cv.threshold(img.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    s0, s1 = otsu.shape
    mask = np.zeros(otsu.shape)
    edge_centers = [(0, s1//2), (s0//2, 0), (s0-1, s1//2), (s0//2, s1-1)]  # left, top, right, bottom
    corners = [(0, 0), (0, s1-1), (s0-1, 0), (s0-1, s1-1)]
    for i, j in corners:
        if mask[i, j] == 0 and otsu[i, j] == 0:
            empty_mat = np.zeros((s0 + 2, s1 + 2), dtype=np.uint8)
            _, _, m, _ = cv.floodFill(otsu.copy(), empty_mat, (j, i), 0)
            mask[m[1:-1, 1:-1] == 1] = 1
    ice = (1 - otsu/255).astype(np.uint8)
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


def write_command():
    N = len(glob(FOLDER + "commands/*.txt"))
    command = input("Write command here: ")
    with open(FOLDER + "commands/command_{:04d}.txt".format(N), 'w') as f:
        f.write(command)


def read_update():
    global update_paths
    updates = sorted(glob(FOLDER + 'updates/*.txt'))
    if len(updates) > 0 and updates[-1] not in update_paths:
        time.sleep(0.5)  # wait before reading to allow for writing time
        data = [line[:-1].split(': ') for line in open(updates[-1], 'r').readlines()]
        data = [val for val in data if len(val) == 2]
        dct = {key: val for key, val in data}
        err_dct = {key: str_to_tuple(val) for key, val in dct.items() if 'err' in key}
        relative_errors = [err_dct[key][1] for key in err_dct]
        err_str = [" ok " if abs(rel_err) <= 1 else "{:.02f}".format(rel_err) for rel_err in relative_errors]

        # add colours
        for i in range(len(err_str)):
            if err_str[i] == ' ok ':
                err_str[i] = '\033[32m ok \033[0m'
            elif abs(float(err_str[i])) <= 2:
                err_str[i] = "\033[33m" + err_str[i] + "\033[0m"
            else:
                err_str[i] = "\033[31m" + err_str[i] + "\033[0m"
        count_str = "[Update {:d}]".format(len(updates))
        print("\r" + count_str + " Errors  |  x: {:s}  | w: {:s}  | h: {:s}  | k: {:s} ".format(*err_str), end='')

        update_paths.append(updates[-1])


def str_to_tuple(s):
    v1, v2 = s.split(', ')
    return float(v1[1:]), float(v2[:-1])


if __name__ == '__main__':
    main()


