import numpy as np
import cv2 as cv


class AdaptiveMask:
    def __init__(self, n_points, view_box):
        self.np = n_points  # number of points around the edge of the contour
        self.vbox = view_box  # [xmin, xmax, ymin, ymax] in screen coordinates of the field of view of the camera
        self.cpoints = self.init_cpoints()  # control points
        self.p0 = 0.01  # multiplier, controls what fraction of the distance control points move towards the center when the mask is not visible

    def init_cpoints(self):
        w, h = self.vbox[1]-self.vbox[0], self.vbox[3]-self.vbox[2]
        top_left = np.array([self.vbox[0], self.vbox[2]])
        dp = (2*w + 2*h) / self.np  # distance between points
        mw, mh = (w % dp)/2, (h % dp)/2  # margins for width and height
        x, y = np.arange(mw, w), np.arange(mh, h)  # point positions
        return np.hstack((
            np.vstack((np.zeros(len(y)), y)),  # left edge
            np.vstack((x, np.ones(len(x)) * h)),  # bottom edge
            np.vstack((np.ones(len(y)) * w, y[::-1])),  # right edge
            np.vstack((x[::-1], np.zeros(len(x)))),  # top edge
        )).T + top_left

    def update(self, cam):
        mask, ice = find_mask_and_ice(cam)
        ice_edges = find_edges(ice)
        mask_edges = find_edges(mask)

        # if mask is not visible around the edge, slowly move towards center
        if mask_edges is None:
            center = np.array([(self.vbox[1]-self.vbox[0])/2, (self.vbox[3]-self.vbox[2])/2])
            delta = center - self.cpoints
            self.cpoints = self.cpoints + self.p0 * delta


def find_mask_and_ice(img):
    # assumes gray image
    blur = cv.GaussianBlur(img, (5, 5), sigmaX=0)
    ret, otsu = cv.threshold(blur.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    s0, s1 = otsu.shape
    mask = np.zeros(otsu.shape)
    edges = [(i, j) for i in range(s0) for j in range(s1) if (i%(s0-1))*(j%(s1-1)) == 0]
    for i, j in edges:
        if mask[i, j] == 0 and otsu[i, j] == 0:
            empty_mat = np.zeros((s0 + 2, s1 + 2), dtype=np.uint8)
            _, _, m, _ = cv.floodFill(otsu.copy(), empty_mat, (j, i), 0)
            mask[m[1:-1, 1:-1] == 1] = 1
    mask = cv.dilate(mask, kernel=np.ones((31, 31)))
    ice = (1 - otsu.copy()/255)
    ice[mask==1] = 0
    return mask, ice


def find_edges(img):
    cont, hierarchy = cv.findContours(img.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(cont) == 0:
        return None  # no contours found

    idx = np.argmax([len(c) for c in cont])  # index of largest contour
    edges = np.reshape(cont[idx], (cont[idx].shape[0], 2))
    return edges


