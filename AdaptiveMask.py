import numpy as np
import cv2 as cv


class AdaptiveMask:
    def __init__(self, view_box, **kwargs):
        self.vbox = view_box  # [xmin, xmax, ymin, ymax] in screen coordinates of the field of view of the camera
        self.p0 = 3  # coarsening factor (works best if it is a divisor of p1)
        self.p1 = 30  # desired distance to ice in pixels
        self.p2 = 50  # dot size to expand the mask with when ice is not visible
        self.keep_sides = kwargs['keep_sides']  # left, top, right, bottom: which sides to keep white at all times
        self.screen = self.init_screen(kwargs['screen_size'])
        self.cam_crop = kwargs['mask_box']  # [xmin, xmax, ymin, ymax] in screen coordinates of the part of camera image that is on screen
        self.ice = None  # placeholder for binarised location of ice
        self.fail_count = 0

    def init_screen(self, ssz):
        screen = np.zeros((ssz[0], ssz[1]), dtype=np.uint8)
        screen[self.vbox[2]:self.vbox[3], self.vbox[0]:self.vbox[1]] = 255
        if not self.keep_sides[0]:
            screen[self.vbox[2]:self.vbox[3], :self.vbox[1]] = 255
        if not self.keep_sides[1]:
            screen[:self.vbox[3], self.vbox[0]:self.vbox[1]] = 255
        if not self.keep_sides[2]:
            screen[self.vbox[2]:self.vbox[3], self.vbox[0]:] = 255
        if not self.keep_sides[3]:
            screen[self.vbox[2]:, self.vbox[0]:self.vbox[1]] = 255
        return screen

    def update(self, cam):
        if self.cam_crop is not None:
            cam = cam[self.cam_crop[2]:self.cam_crop[3], self.cam_crop[0]:self.cam_crop[1]]
        cam = cv.resize(cam, (self.vbox[1]-self.vbox[0], self.vbox[3]-self.vbox[2]))
        cam = fill_border(cam, 255, n=10, skip=self.keep_sides)

        _, ice = find_mask_and_ice(cam)
        cfac = int(self.p1/self.p0)  # coarsening factor: relative size of the coarse image pixels

        if self.ice is not None and np.sum(ice > 0) < 0.5 * np.sum(self.ice > 0):
            print("[AdaptiveMask.update] Ice not detected! Dilating previous ice twice as much instead.")
            self.fail_count += 1
            ice = self.ice
            cfac *= self.fail_count * 2
        else:
            self.fail_count = 0
            self.ice = ice

        coarse_ice = np.zeros((ice.shape[0]//cfac, ice.shape[1]//cfac))
        for i in range(coarse_ice.shape[0]):
            for j in range(coarse_ice.shape[1]):
                coarse_ice[i, j] = 255 if np.any(ice[i*cfac:(i+1)*cfac, j*cfac:(j+1)*cfac]) else 0

        coarse_mask = cv.dilate(coarse_ice, kernel=np.ones((3, 3)), iterations=self.p0)
        mask = np.zeros(ice.shape)
        for i in range(coarse_ice.shape[0]):
            for j in range(coarse_ice.shape[1]):
                mask[i*cfac:(i+1)*cfac, j*cfac:(j+1)*cfac] = coarse_mask[i, j]
        self.screen[self.vbox[2]:self.vbox[3], self.vbox[0]:self.vbox[1]] = mask  # TODO: apply conversion map here


def find_mask_and_ice(img):
    # assumes gray image
    # blur = cv.GaussianBlur(img, (5, 5), sigmaX=0)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
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


def fill_border(arr, value, n=1, skip=None):
    if skip is None or not skip[0]:
        arr[:, :n] = value  # left
    if skip is None or not skip[1]:
        arr[:n, :] = value  # top
    if skip is None or not skip[2]:
        arr[:, -n:] = value  # right
    if skip is None or not skip[3]:
        arr[-n:, :] = value  # bottom 
    return arr

