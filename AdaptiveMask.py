import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from math import factorial


class AdaptiveMask:
    def __init__(self, view_box, **kwargs):
        self.vbox = view_box  # [xmin, xmax, ymin, ymax] in screen coordinates of the field of view of the camera
        self.p0 = 0.2  # factor that controls how dot size depends on distance to ice (aggressiveness)
        self.p1 = 10  # desired distance to ice in pixels
        self.p2 = 50  # dot size to expand the mask with when ice is not visible
        self.keep_sides = kwargs['keep_sides']  # left, top, right, bottom: which sides to keep white at all times
        self.screen = self.init_screen(kwargs['screen_size'])
        self.cam_crop = kwargs['mask_box']  # [xmin, xmax, ymin, ymax] in screen coordinates of the part of camera image that is on screen

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
        cam = cam[self.cam_crop[2]:self.cam_crop[3], self.cam_crop[0]:self.cam_crop[1]]
        cam = cv.resize(cam, (self.vbox[1]-self.vbox[0], self.vbox[3]-self.vbox[2]))
        cam = fill_border(cam, 255, n=10, skip=[not bl for bl in self.keep_white])
        
        top_left = np.array([self.vbox[0], self.vbox[2]])
        mask, ice = find_mask_and_ice(cam)
        mask = fill_border(mask, 1, skip=self.keep_white)
        ice = fill_border(ice, 0, n=5)
        mask_edges = find_edges(mask)
        ice_edges = find_edges(ice)
        screen_edges = find_edges(self.screen)

        # Remove box edges
        mask_edges = mask_edges[(mask_edges[:, 0] % mask.shape[1]-1) != 0]
        mask_edges = mask_edges[(mask_edges[:, 1] % mask.shape[0]-1) != 0]

        # ice_edges += top_left
        # plt.plot(ice_edges[:, 0], ice_edges[:, 1])

        #plt.figure()
        #mat = np.zeros(cam.shape[:2])
        #mat += mask
        #mat += 2 * ice
        #plt.imshow(mat)
        #if mask_edges is not None:
        #    plt.plot(mask_edges[:, 0], mask_edges[:, 1], '.b')
        #if ice_edges is not None:
        #    plt.plot(ice_edges[:, 0], ice_edges[:, 1], '-r')
     

        if ice_edges is None:
            # expand mask on all edge points
            for j, i in screen_edges:
                dsz = self.p2
                self.screen[i-dsz:i+dsz, j-dsz:j+dsz] = 255
        else:
            #plt.figure()
            #plt.imshow(cam)
            #plt.figure()
            #plt.imshow(mask)
            #plt.figure()
            #plt.imshow(ice)
            #plt.plot(ice_edges[:, 0], ice_edges[:, 1])
            #plt.show()
            ice_edges += top_left
            mask_edges += top_left

            target_points = []
            errors = []
            for j, i in screen_edges:
                cm = mask_edges[np.argmin(np.sum((np.array([j, i]) - mask_edges)**2, axis=1))]  # closest point on mask
                di = np.min(np.sqrt(np.sum((cm - ice_edges)**2, axis=1))) - self.p1  # distance to ice
                dsz = int(abs(self.p0 * di))
                self.screen[i-dsz:i+dsz, j-dsz:j+dsz] = 0 if di > 0 else 255

                ii = np.argmin(np.sqrt(np.sum((cm - ice_edges)**2, axis=1)))
                target_points.append(cm + (ice_edges[ii] - cm) * dsz/di - top_left)
                errors.append(di**2)
            target_points = np.array(target_points)
            #plt.plot(target_points[:, 0], target_points[:, 1], '.g')
            print("Min error: {:.0f} px  | Max error: {:.0f}  |  Mean error: {:.0f} pixels  |  Median error: {:.0f}".format(np.sqrt(np.min(errors)), np.sqrt(np.max(errors)), np.sqrt(np.mean(errors)), np.sqrt(np.median(errors))))
        #plt.show()

        # force any regions outside of camera view box to be black
        self.screen[:self.vbox[2], :] = 0
        self.screen[self.vbox[3]:, :] = 0
        self.screen[:, :self.vbox[0]] = 0
        self.screen[:, self.vbox[1]:] = 0

        # force pre-set regions to be white
        if self.keep_white[0]:
            self.screen[self.vbox[2]:self.vbox[3], :self.vbox[0]] = 255
        if self.keep_white[1]:
            self.screen[:self.vbox[2], self.vbox[0]:self.vbox[1]] = 255
        if self.keep_white[2]:
            self.screen[self.vbox[2]:self.vbox[3], self.vbox[1]:] = 255
        if self.keep_white[3]:
            self.screen[self.vbox[3]:, self.vbox[0]:self.vbox[1]] = 255

        # smoothen
        self.screen = cv.blur(self.screen, (self.p1//2, self.p1//2))
        _, self.screen = cv.threshold(self.screen, 127, 255, cv.THRESH_BINARY)


def find_mask_and_ice(img):
    # assumes gray image
    # blur = cv.GaussianBlur(img, (5, 5), sigmaX=0)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
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

