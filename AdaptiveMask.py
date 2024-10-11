import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from math import factorial


class AdaptiveMask:
    def __init__(self, view_box, screen_size=(1920, 1080), **kwargs):
        self.vbox = view_box  # [xmin, xmax, ymin, ymax] in screen coordinates of the field of view of the camera
        self.screen = np.zeros((screen_size[1], screen_size[0]), dtype=np.uint8)
        self.screen[self.vbox[2]:self.vbox[3], self.vbox[0]:self.vbox[1]] = 255
        self.p0 = 0.1  # factor that controls how dot size depends on distance to ice (aggressiveness)
        self.p1 = 10  # desired distance to ice in pixels
        self.p2 = 50  # dot size to expand the mask with when ice is not visible

    def update(self, cam):
        top_left = np.array([self.vbox[0], self.vbox[2]])
        mask, ice = find_mask_and_ice(cam)
        mask = fill_border(mask, 1)
        ice = fill_border(ice, 0, n=5)
        mask_edges = find_edges(mask)
        ice_edges = find_edges(ice)
        screen_edges = find_edges(self.screen)

        # plt.figure()
        # plt.imshow(ice)
        # plt.plot(ice_edges[:, 0], ice_edges[:, 1])
        # plt.show()

        # ice_edges += top_left
        # plt.plot(ice_edges[:, 0], ice_edges[:, 1])

        if ice_edges is None:
            # expand mask on all edge points
            for j, i in screen_edges:
                dsz = self.p2
                self.screen[i-dsz:i+dsz, j-dsz:j+dsz] = 255
        else:
            ice_edges += top_left
            mask_edges += top_left
            for j, i in screen_edges:
                cm = mask_edges[np.argmin(np.sum((np.array([j, i]) - mask_edges)**2, axis=1))]  # closest point on mask
                di = np.min(np.sqrt(np.sum((cm - ice_edges)**2, axis=1))) - self.p1  # distance to ice
                dsz = int(abs(self.p0 * di))
                self.screen[i-dsz:i+dsz, j-dsz:j+dsz] = 0 if di > 0 else 255

        # force any regions outside of camera view box to be black
        self.screen[:self.vbox[2], :] = 0
        self.screen[self.vbox[3]:, :] = 0
        self.screen[:, self.vbox[0]] = 0
        self.screen[:, self.vbox[1]:] = 0

        # smoothen
        self.screen = cv.blur(self.screen, (self.p1//2, self.p1//2))
        _, self.screen = cv.threshold(self.screen, 127, 255, cv.THRESH_BINARY)


def find_mask_and_ice(img):
    # assumes gray image
    # blur = cv.GaussianBlur(img, (5, 5), sigmaX=0)
    ret, otsu = cv.threshold(img.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    s0, s1 = otsu.shape
    mask = np.zeros(otsu.shape)
    # edges = [(i, j) for i in range(s0) for j in range(s1) if (i%(s0-1))*(j%(s1-1)) == 0]
    corners = [(0, 0), (s0-1, 0), (0, s1-1), (s0-1, s1-1)]
    for i, j in corners:
        if mask[i, j] == 0 and otsu[i, j] == 0:
            empty_mat = np.zeros((s0 + 2, s1 + 2), dtype=np.uint8)
            _, _, m, _ = cv.floodFill(otsu.copy(), empty_mat, (j, i), 0)
            mask[m[1:-1, 1:-1] == 1] = 1
    # mask = cv.dilate(mask, kernel=np.ones((31, 31)))
    ice = (1 - otsu.copy()/255)
    ice[mask==1] = 0
    return mask, ice


def find_edges(img, largest_only=False):
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
        return edges[1:, :]


def fill_border(arr, value, n=1):
    arr[:n, :] = value
    arr[-n:, :] = value
    arr[:, :n] = value
    arr[:, -n:] = value
    return arr

