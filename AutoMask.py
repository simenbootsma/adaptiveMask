import cv2 as cv
from ice_detection import *
import matplotlib.pyplot as plt
from ManualMask import Cylinder


class AutoMask:
    def __init__(self):
        self.display = cv.cvtColor(init_manual_mask(), cv.COLOR_RGB2GRAY)
        self.sensitivity = 0.2  # multiplier for the error
        self.eta = 50  # target width in camera pixels
        self.loc = None
        self.sz_fac = None

    def update(self, img, show=False):
        # (0) Extract ice and mask from image
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        mask, ice = find_mask_and_ice(img)

        ice_edges = find_edges(ice, largest_only=True, remove_inside=False)
        mask_edges = find_edges(mask, remove_outside=True)

        if ice_edges is None:
            print('No ice!')
            return

        if mask_edges is None:
            mask_edges = box_edges(mask.shape)

        mat = mask.copy()
        cv.fillPoly(mat, [ice_edges], (2, 2, 2))

        # (1) Obtain target mask
        not_ice = 1 - (mat - mask)/2
        dmap_ice = cv.distanceTransform(not_ice.astype(np.uint8), cv.DIST_L2, 5)
        target = np.where(dmap_ice < self.eta, 0, 1)
        target_edges = find_edges(target, remove_outside=True)

        # (2) Vector of each point on mask edge to target
        dists = np.zeros(mask_edges.shape)
        for i in range(len(mask_edges)):
            d2 = np.sum((mask_edges[i, :] - target_edges)**2, axis=1)
            dists[i, :] = target_edges[np.argmin(d2), :] - mask_edges[i, :]  # shortest vector from mask to target

        if self.loc is None or self.sz_fac is None:
            # (3) Find location of image mask on screen mask TODO: deal with white screen
            factors = np.linspace(1.5, 2.5, 20)[::-1]
            values = -1 * np.ones(factors.size)
            locations = np.zeros((factors.size, 2))
            for i, fac in enumerate(factors):
                templ = cv.resize((255*mask).astype(np.uint8), (int(mask.shape[1] / fac), int(mask.shape[0] / fac)))
                if templ.shape[0] > self.display.shape[0] or templ.shape[1] > self.display.shape[1]:
                    break
                res = cv.matchTemplate(255 - self.display, templ, cv.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv.minMaxLoc(res)
                values[i] = max_val
                locations[i, :] = max_loc

            self.loc = locations[np.argmax(values)]
            self.sz_fac = factors[np.argmax(values)]

        # (4) Define control points
        ncp = 500  # number of control points
        disp_edges = find_edges(self.display, remove_outside=True)
        control_points = disp_edges[::len(disp_edges)//ncp]

        # (5) Find movement for every control point
        mov = np.zeros(control_points.shape)
        cnt = np.zeros((control_points.shape[0], 1))
        sme = mask_edges / self.sz_fac + self.loc  # shifted mask edges
        for i in range(mask_edges.shape[0]):
            d = np.sum((sme[i, :] - control_points)**2, axis=1)
            mov[np.argmin(d), :] += dists[i, :] / self.sz_fac
            cnt[np.argmin(d)] += 1

        mov = mov / cnt
        mov[np.isnan(mov)] = 0

        # (6) Move control points
        control_points = control_points + mov * self.sensitivity

        # (7) Generate new display
        new_display = np.zeros(self.display.shape)
        cv.fillPoly(new_display, [control_points.astype(np.int32)], (255, 255, 255))
        self.display = new_display.astype(np.uint8)

        if show:
            plt.ioff()
            if self.loc is None:
                plt.figure()
                plt.plot(factors, values, '-o')

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
            ax[0].imshow(self.display)
            ax[0].add_artist(plt.Rectangle((self.loc[0], self.loc[1]), width=mask.shape[1]/self.sz_fac, height=mask.shape[0]/self.sz_fac, fc='none', ec='r'))
            ax[0].plot(control_points[:, 0], control_points[:, 1], '-og', markersize=3)
            ax[0].plot(sme[:, 0], sme[:, 1], '-m', markersize=3)
            for i in range(len(control_points)):
                ax[0].arrow(control_points[i, 0], control_points[i, 1], mov[i, 0], mov[i, 1])
            ax[0].set_title('display')

            ax[1].imshow(new_display)
            ax[1].set_title('new display')

            for a in ax:
                a.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
            plt.show()

    def get_img(self):
        return self.display


def init_manual_mask():
    im = np.zeros((2000, 1000, 3), dtype=np.uint8)
    im[300:1150, 500:900] = 255
    # cv.circle(im, (500, 500), 400, color=(255, 255, 255), thickness=-1)
    return im

    cyl = Cylinder(resolution=(8256, 5504))
    cyl.transpose()
    cyl.set_center(4000)
    cyl.set_height(7500)
    cyl.set_width(2500)
    return cyl.get_img()

    obj = Cylinder()

    cv.namedWindow("window", cv.WND_PROP_FULLSCREEN)
    # cv.moveWindow("window", 2000, 100)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    while True:
        cv.imshow("window", obj.get_img())
        key = cv.waitKey()
        if key == 27:
            break
        obj.handle_key(key)
    cv.destroyWindow("window")
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


