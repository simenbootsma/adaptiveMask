from ice_detection import *
import matplotlib.pyplot as plt
from ManualMask import Cylinder
import cv2 as cv


class AutoMask:
    def __init__(self):
        self.display = cv.cvtColor(init_manual_mask(), cv.COLOR_RGB2GRAY)
        self.sensitivity = 0.5  # multiplier for the error NOTE: should high enough, otherwise the mask might fail due to subpixel movement
        self.eta = 100  # target width in camera pixels
        self.ncp = 200  # number of control points  TODO: should depend on number of points in mask edge, each control point must have multiple corresponding mask edge points
        self.smth_n = 0#5  # number of control points to use in moving average smoothing
        self.mov_limit = 0#self.eta
        self.min_mov = 0  # minimum movement in camera pixels
        self.loc = None
        self.sz_fac = None

    def update(self, img, show=False):
        # (0) Extract ice and mask from image
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        mask, ice = find_mask_and_ice(gray)

        ice_edges = find_edges(ice, largest_only=True, remove_inside=False)
        mask_edges = find_edges(mask, remove_outside=True)

        if mask_edges is None:
            mask_edges = box_edges(mask.shape)

        # (1) Obtain target mask
        if ice_edges is None or np.sum(ice)/np.sum(mask) < 0.001:
            print('No ice!')

            if show:
                plt.ioff()
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(img)
                ax[1].imshow(mask)
                ax[2].imshow(ice)
                plt.show()

            self.display = cv.dilate(self.display, np.ones((self.eta//2, self.eta//2)))
            for j, i in box_edges(self.display.shape).astype(np.int32):
                self.display[i, j] = 0  # edges must remain black for defining control points
            return

        not_ice = np.ones(ice.shape)
        cv.fillPoly(not_ice, [ice_edges], (0, 0, 0))
        dmap_ice = cv.distanceTransform(not_ice.astype(np.uint8), cv.DIST_L2, 5)
        target = np.where(dmap_ice < self.eta, 0, 1)
        target_edges = find_edges(target, remove_outside=True)

        # (2) Vector of each point on mask edge to target
        dists = np.zeros(mask_edges.shape)
        for i in range(len(mask_edges)):
            d2 = np.sum((mask_edges[i, :] - target_edges)**2, axis=1)
            dists[i, :] = target_edges[np.argmin(d2), :] - mask_edges[i, :]  # shortest vector from mask to target

        # (3) Calibrate
        if self.loc is None or self.sz_fac is None:
            self.calibrate(img)
        # self.recalibrate(mask_edges)  # unstable..

        # (4) Define control points
        disp_edges = find_edges(self.display, remove_outside=True)
        control_points = disp_edges[::len(disp_edges)//self.ncp]

        # (5) Find movement for every control point
        mov = np.zeros(control_points.shape)
        cnt = np.zeros((control_points.shape[0], 1))
        sme = mask_edges / self.sz_fac + self.loc  # shifted mask edges
        for i in range(mask_edges.shape[0]):
            d = np.sum((sme[i, :] - control_points)**2, axis=1)
            mov[np.argmin(d), :] += dists[i, :] / self.sz_fac
            cnt[np.argmin(d)] += 1

        # print("minimum number of mask edge points for a control point: {:.0f}".format(np.min(cnt[cnt>0])))

        mov = mov / cnt
        mov[np.isnan(mov)] = 0
        if self.min_mov > 0:
            I = np.sqrt(np.sum(mov ** 2, axis=1)) < self.min_mov
            mov[I, :] = 0
        if self.mov_limit > 0:
            I = np.sqrt(np.sum(mov**2, axis=1)) > self.mov_limit
            mov[I, :] = mov[I, :] / np.reshape(np.sqrt(np.sum(mov[I, :]**2, axis=1)), (len(mov[I, :]), 1)) * self.mov_limit
        if self.smth_n > 1:
            mov[:, 0] = smoothen_array(mov[:, 0], n=self.smth_n)
            mov[:, 1] = smoothen_array(mov[:, 1], n=self.smth_n)

        # (6) Move control points
        control_points = control_points + mov * self.sensitivity

        if self.smth_n > 1:
            control_points[:, 0] = smoothen_array(control_points[:, 0], n=self.smth_n)
            control_points[:, 1] = smoothen_array(control_points[:, 1], n=self.smth_n)

        # (7) Generate new display
        new_display = np.zeros(self.display.shape)
        cv.fillPoly(new_display, [control_points.astype(np.int32)], (255, 255, 255))
        old_display = self.display.copy()
        self.display = new_display.astype(np.uint8)

        if show:
            plt.ioff()
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(mask)
            ax[0].plot(mask_edges[:, 0], mask_edges[:, 1], '-r')
            ax[1].imshow(target)
            ax[1].plot(target_edges[:, 0], target_edges[:, 1], '-r')
            ax[2].imshow(mask - target)
            scat = ax[2].scatter(mask_edges[:, 0], mask_edges[:, 1], 10, np.sqrt(np.sum(dists**2, axis=1)), cmap=plt.get_cmap('Blues'))

            for i in range(0, len(mask_edges), len(mask_edges)//20):
                ax[2].arrow(mask_edges[i, 0], mask_edges[i, 1], dists[i, 0]/2, dists[i, 1]/2, color='r', width=0.003, head_width=0.1, head_length=0.1)

            ax[0].set_title('mask')
            ax[1].set_title('target')
            ax[2].set_title('diff')

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(old_display)
            ax[0].add_artist(plt.Rectangle((self.loc[0], self.loc[1]), width=mask.shape[1]/self.sz_fac, height=mask.shape[0]/self.sz_fac, fc='none', ec='r'))
            ax[0].plot(control_points[:, 0], control_points[:, 1], '-og', markersize=3)
            ax[0].plot(sme[:, 0], sme[:, 1], '-m', markersize=3)
            for i in range(len(control_points)):
                ax[0].arrow(control_points[i, 0], control_points[i, 1], -mov[i, 0]*self.sensitivity, -mov[i, 1]*self.sensitivity)
            ax[0].set_title('display')

            ax[1].imshow(new_display)
            ax[1].set_title('new display')

            for a in ax:
                a.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
            plt.show()

    def set_eta(self, value):
        self.eta = int(value)

    def set_sensitivity(self, value):
        if 0 < value <= 1:
            self.sensitivity = value
        else:
            print("[AutoMask] warning: sensitivity must be in the range (0, 1]")

    def set_ncp(self, value):
        self.ncp = int(value)

    def get_img(self):
        return self.display

    def calibrate(self, img):
        # TODO
        #   - deal with white screen
        #   - add certainty (peak width)
        #   - adaptive calibration: try values close to peak

        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        mask, ice = find_mask_and_ice(gray)

        # Find location of image mask on screen mask
        factors = np.linspace(1.5, 2.5, 40)[::-1]
        factors = np.array([2])
        values = -1 * np.ones(factors.size)
        locations = np.zeros((factors.size, 2))
        for i, fac in enumerate(factors):
            templ = cv.resize((255 * mask).astype(np.uint8), (int(mask.shape[1] / fac), int(mask.shape[0] / fac)))
            if templ.shape[0] > self.display.shape[0] or templ.shape[1] > self.display.shape[1]:
                break
            res = cv.matchTemplate(255 - self.display, templ, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(res)
            values[i] = max_val
            locations[i, :] = max_loc

        self.loc = locations[np.argmax(values)]
        self.sz_fac = factors[np.argmax(values)]
        # print("size factor: {:.3f} +/- {:.3f}".format(self.sz_fac, factors[0]-factors[1]))

        # plt.figure()
        # plt.plot(factors, values, '-o')
        # plt.show()

    def recalibrate(self, mask_edges):
        disp_edges = find_edges(self.display, remove_outside=True)
        control_points = disp_edges[::len(disp_edges)//self.ncp]

        cmep = np.zeros(control_points.shape)
        cnt = np.zeros((control_points.shape[0], 1))
        sme = mask_edges / self.sz_fac + self.loc  # shifted mask edges
        for i in range(mask_edges.shape[0]):
            d = np.sum((sme[i, :] - control_points)**2, axis=1)
            cmep[np.argmin(d), :] += sme[i, :]
            cnt[np.argmin(d)] += 1
        cmep = cmep / cnt
        cp = control_points[~np.isnan(cmep[:, 0] + cmep[:, 1]), :]
        cmep = cmep[~np.isnan(cmep[:, 0] + cmep[:, 1]), :]

        diff = np.mean(cp - cmep, axis=0)

        cp_w, cp_h = (np.max(cp[:, 0]) - np.min(cp[:, 0])), (np.max(cp[:, 1]) - np.min(cp[:, 1]))
        cmep_w, cmep_h = np.max(cmep[:, 0]) - np.min(cmep[:, 0]), np.max(cmep[:, 1]) - np.min(cmep[:, 1])
        sz_fac_opt = self.sz_fac * (cmep_w * cmep_h) / (cp_w * cp_h)
        sz_fac_opt = self.sz_fac
        print("szf: {:.3f}  |  szf opt: {:.3f}".format(self.sz_fac, sz_fac_opt))
        print("loc: ({:.1f}, {:.1f})  |  loc opt: ({:.1f}, {:.1f})".format(self.loc[0], self.loc[1], self.loc[0] + diff[0], self.loc[1] + diff[1]))

        new_sme = sme + diff
        sme_c = np.mean(new_sme, axis=0)
        new_sme = (new_sme - sme_c) / sz_fac_opt * self.sz_fac + sme_c

        self.loc += diff

        # plt.ioff()
        # fig, ax = plt.subplots(1, 2)
        # ax[0].plot(cp[:, 0], cp[:, 1], 'o')
        # ax[0].plot(sme[:, 0], sme[:, 1], '-')
        # ax[1].plot(cp[:, 0], cp[:, 1], 'o')
        # ax[1].plot(new_sme[:, 0], new_sme[:, 1], '-')
        # plt.show()


def init_manual_mask():
    # im = np.zeros((2000, 1000, 3), dtype=np.uint8)
    # im[200:750, 500:900] = 255
    # # cv.circle(im, (500, 500), 400, color=(255, 255, 255), thickness=-1)
    # return im
    #
    # cyl = Cylinder(resolution=(8256, 5504))
    # cyl.transpose()
    # cyl.set_center(4000)
    # cyl.set_height(7500)
    # cyl.set_width(2500)
    # return cyl.get_img()

    obj = Cylinder()

    cv.namedWindow("init", cv.WND_PROP_FULLSCREEN)
    # cv.moveWindow("window", 2000, 100)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    while True:
        cv.imshow("init", obj.get_img())
        key = cv.waitKey()
        if key == 27:
            break
        obj.handle_key(key)
    cv.destroyWindow("init")
    return obj.get_img()


def smoothen_array(y, n):
    y = [np.nanmean(y[max(0, i - n // 2):min(len(y), i + n // 2)]) for i in range(len(y))]
    return np.array(y)


def box_edges(shape):
    return np.array([[0, i] for i in range(shape[0])]                       # left
                    + [[j, shape[0]-1] for j in range(shape[1])]            # bottom
                    + [[shape[1]-1, i] for i in range(shape[0])]            # right
                    + [[j, 0] for j in range(shape[1]-1, -1, -1)]           # top
                    )


