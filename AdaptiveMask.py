import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from math import factorial


class AdaptiveMask:
    def __init__(self, n_points, view_box):
        self.np = n_points  # number of points around the edge of the contour
        self.vbox = view_box  # [xmin, xmax, ymin, ymax] in screen coordinates of the field of view of the camera
        self.cpoints = self.init_cpoints()  # control points
        self.prev_cpoints = self.cpoints.copy()
        self.p0 = 0.1  # multiplier, controls what fraction of the distance control points move towards the box center when the mask is not visible
        self.p1 = 0.3  # multiplier, controls what fraction of the distance control points move towards the ice
        self.p2 = 100  # sets the desired distance between mask and ice in number of pixels

    def init_cpoints(self):
        w, h = self.vbox[1]-self.vbox[0], self.vbox[3]-self.vbox[2]
        top_left = np.array([self.vbox[0], self.vbox[2]])
        dp = (2*w + 2*h) / self.np  # distance between points
        mw, mh = (w % dp)/2, (h % dp)/2  # margins for width and height
        x, y = np.arange(mw, w, dp), np.arange(mh, h, dp)  # point positions
        cpoints = np.hstack((
            np.vstack((np.zeros(len(y)), y)),  # left edge
            np.vstack((x, np.ones(len(x)) * h)),  # bottom edge
            np.vstack((np.ones(len(y)) * w, y[::-1])),  # right edge
            np.vstack((x[::-1], np.zeros(len(x)))),  # top edge
        )).T + top_left
        self.np = cpoints.shape[0]
        return cpoints

    def update(self, cam):
        self.prev_cpoints = self.cpoints.copy()

        mask, ice = find_mask_and_ice(cam)
        top_left = np.array([self.vbox[0], self.vbox[2]])
        ice_edges = find_edges(ice)

        if ice_edges is None:
            self.cpoints = self.prev_cpoints.copy()
            return
        else:
            ice_edges = ice_edges + top_left

        mask_edges = find_edges(mask)
        mask_edges = [p for p in mask_edges if (p[0] % (mask.shape[1]-1) > 0) and (p[1] % (mask.shape[0]-1) > 0)]
        mask_edges = 9e9 * np.ones((2, 2)) if len(mask_edges) == 0 else np.array(mask_edges) + top_left
        box_edges = np.hstack((
            np.vstack((np.zeros(mask.shape[0]), np.arange(mask.shape[0]))),  # left edge
            np.vstack((np.arange(mask.shape[1]), np.ones(mask.shape[1]) * (mask.shape[0]-1))),  # bottom edge
            np.vstack((np.ones(mask.shape[0]) * (mask.shape[1]-1), np.arange(mask.shape[0])[::-1])),  # right edge
            np.vstack((np.arange(mask.shape[1])[::-1], np.zeros(mask.shape[1]))),  # top edge
        )).T + top_left

        # # for debugging
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(mask)
        # ax[0].plot(mask_edges[:, 0], mask_edges[:, 1])
        # # ax[0].plot(box_edges[:, 0], box_edges[:, 1])
        # ax[1].imshow(ice)
        # # ax[1].plot(ice_edges[:, 0], ice_edges[:, 1])

        box_center = np.array([(self.vbox[1] + self.vbox[0]) / 2, (self.vbox[3] + self.vbox[2]) / 2])
        for i in range(self.np):
            mask_dist = np.sqrt(np.sum((mask_edges - self.cpoints[i]) ** 2, axis=1))
            box_dist = np.sqrt(np.sum((box_edges - self.cpoints[i]) ** 2, axis=1))
            if np.min(box_dist) < np.min(mask_dist):
                # box edge is closer than mask, so move towards center
                self.cpoints[i] = self.cpoints[i] + self.p0 * (box_center - self.cpoints[i])
                # plt.plot(self.cpoints[i, 0], self.cpoints[i, 1], 'xb')
            else:
                # mask is visible, assume control point is at the closest point to the mask. Move towards closest point of ice edge
                mp = mask_edges[np.argmin(mask_dist)]  # point on mask edge, closest to the i-th control point
                ice_dist = np.sqrt(np.sum((ice_edges - mp)**2, axis=1))
                ip = ice_edges[np.argmin(ice_dist)]  # point on ice edge, closest to the mask point
                dmp = ip + (mp - ip)/np.sqrt(np.sum((mp - ip)**2)) * self.p2  # desired point for the mask

                # plt.plot(self.cpoints[i, 0], self.cpoints[i, 1], 'ob')
                self.cpoints[i] = self.cpoints[i] + self.p1 * (dmp - mp)  # move towards desired mask point
                # plt.plot(mp[0], mp[1], 'or')
                # plt.plot(dmp[0], dmp[1], 'sr')
                # plt.plot(ip[0], ip[1], 'og')
                # plt.plot(self.cpoints[i, 0], self.cpoints[i, 1], 'sb')

    def curve(self):
        cp = np.vstack((self.cpoints, self.cpoints))
        return np.array([np.mean(cp[i:(i+10), :], axis=0) for i in range(len(self.cpoints)+1)])


def find_mask_and_ice(img):
    # assumes gray image
    # blur = cv.GaussianBlur(img, (5, 5), sigmaX=0)
    ret, otsu = cv.threshold(img.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    s0, s1 = otsu.shape
    mask = np.zeros(otsu.shape)
    edges = [(i, j) for i in range(s0) for j in range(s1) if (i%(s0-1))*(j%(s1-1)) == 0]
    for i, j in edges:
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


