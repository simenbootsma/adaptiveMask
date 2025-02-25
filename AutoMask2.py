import cv2 as cv
from ice_detection import *
import matplotlib.pyplot as plt
from ManualMask import Cylinder


class AutoMask:
    def __init__(self):
        self.mask = cv.cvtColor(init_manual_mask(), cv.COLOR_RGB2GRAY)
        self.sensitivity = 0.02  # multiplier for the error
        self.eta = 50  # target width in camera pixels

    def update(self, img, show=False):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        mask, ice = find_mask_and_ice(img)

        ice_edges = find_edges(ice, largest_only=True, remove_inside=False)

        if ice_edges is None:
            print('No ice!')
            return

        # mask_edges = find_edges(mask, remove_outside=True)
        mat = mask.copy()
        cv.fillPoly(mat, [ice_edges], (2, 2, 2))

        centerline = np.array([np.mean(np.where(ice[i, :] == 1)) for i in range(ice.shape[0])])
        idx = np.arange(len(centerline))
        centerline = np.interp(idx, idx[~np.isnan(centerline)], centerline[~np.isnan(centerline)])
        centerline = centerline.astype(np.int32)

        not_ice = 1 - (mat - mask)/2
        dmap_ice = cv.distanceTransform(not_ice.astype(np.uint8), cv.DIST_L2, 5)
        dmap_mask = cv.distanceTransform(mask.astype(np.uint8), cv.DIST_L2, 5)
        mask_edge_dist = np.where((dmap_mask > 0) * (dmap_mask < 4), dmap_ice, np.nan)

        target = np.where(dmap_ice < self.eta, 0, 1)
        diff = (target - mask)
        target_or_mask = (target + mask) > 0

        d_left = np.array([np.nanmean(mask_edge_dist[i, :centerline[i]]) for i in range(len(centerline))])
        d_left[np.isnan(d_left)] = 0
        dx_left = np.array([np.sum(diff[i, :centerline[i]]) for i in range(len(centerline))])

        d_right = np.array([np.nanmean(mask_edge_dist[i, centerline[i]:]) for i in range(len(centerline))])
        d_right[np.isnan(d_right)] = 0
        dx_right = np.array([np.sum(diff[i, centerline[i]:]) for i in range(len(centerline))])

        if show:
            plt.ioff()
            fig, ax = plt.subplots(1, 5)
            ax[0].imshow(mat)
            ax[1].imshow(ice)
            ax[1].plot(ice_edges[:, 0], ice_edges[:, 1], 'r')
            ax[2].imshow(target_or_mask)
            ax[3].imshow(target)
            ax[4].imshow(diff)
            ax[0].plot(centerline, np.arange(len(centerline)))
            ax[0].set_title('mask + 2*ice, mat')
            ax[1].set_title('L2 transform ice, ice')
            ax[2].set_title('target or mask')
            ax[3].set_title('target')
            ax[4].set_title('diff')
            for a in ax:
                a.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)

            fig, ax = plt.subplots(1, 2)
            ax[0].plot(dx_left, np.arange(len(dx_left)))
            ax[0].plot(dx_right, np.arange(len(dx_right)))

        # Redraw mask
        new_mask = 255 * np.ones(self.mask.shape, dtype=np.uint8)
        dm = np.zeros((self.mask.shape[0], self.mask.shape[1], 4))

        # TODO: include calibration here
        # dx_left = dx_left[::2]
        # d_left = d_left[::2]
        # dx_right = dx_right[::2]
        # d_right = d_right[::2]
        # # centerline = (centerline[::2]//2)

        top, bot = 500, 800
        # top, bot = 0, 1500

        x, xp = np.arange(self.mask.shape[0]), np.linspace(top, bot, img.shape[0])
        dx_left = np.interp(x, xp, dx_left)
        d_left = np.interp(x, xp, d_left)
        dx_right = np.interp(x, xp, dx_right)
        d_right = np.interp(x, xp, d_right)

        dx_left = np.sign(dx_left)
        dx_right = np.sign(dx_right)

        mask_cl = np.array([np.mean(np.where(self.mask[i, :] == 0)) for i in range(self.mask.shape[0])])
        idx = np.arange(len(mask_cl))
        mask_cl = np.interp(idx, idx[~np.isnan(mask_cl)], mask_cl[~np.isnan(mask_cl)])
        mask_cl = mask_cl.astype(np.int32)

        clw = np.array([np.sum(self.mask[i, :mask_cl[i]] == 0) for i in range(len(mask_cl))])
        crw = np.array([np.sum(self.mask[i, mask_cl[i]:] == 0) for i in range(len(mask_cl))])

        # clw = clw[::2]//2
        lw = clw + dx_left * np.abs(d_left - self.eta) * self.sensitivity
        lw = lw.astype(np.int32)
        for i in range(new_mask.shape[0]):
            if lw[i] >= new_mask.shape[1]:
                new_mask[i, :] = 0
                dm[i, :, 0] = 1
            elif lw[i] > 0:
                new_mask[i, :lw[i]] = 0
                dm[i, :lw[i], 0] = 1

        # crw = crw[::2]//2
        rw = crw + dx_right * np.abs(d_right - self.eta) * self.sensitivity
        rw = rw.astype(np.int32)
        for i in range(new_mask.shape[0]):
            if rw[i] >= new_mask.shape[1]:
                new_mask[i, :] = 0
                dm[i, :, 1] = 1
            elif rw[i] > 0:
                new_mask[i, -rw[i]:] = 0
                dm[i, -rw[i]:, 1] = 1

        if show:
            plt.figure()
            plt.plot(dx_left, label='dx left')
            plt.plot(d_left, label='d left')
            plt.plot(dx_left * np.abs(d_left - self.eta), label='dx * err')
            plt.plot(clw, label='clw')
            plt.plot(lw, '--', label='lw')
            plt.legend()

            plt.figure()
            plt.imshow(np.sum(dm, axis=-1))
            plt.show()

        self.mask = new_mask

    def update_old3(self, img, show=False):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        mask, ice = find_mask_and_ice(img)

        ice_edges = find_edges(ice, largest_only=True, remove_inside=True)

        if ice_edges is None:
            print('No ice!')
            return

        # mask_edges = find_edges(mask, remove_outside=True)
        mat = mask.copy()
        cv.fillPoly(mat, [ice_edges], (2, 2, 2))

        cx, cy = np.mean(ice_edges, axis=0).astype(np.int32)

        not_ice = 1 - (mat - mask)/2
        dmap_ice = cv.distanceTransform(not_ice.astype(np.uint8), cv.DIST_L2, 5)
        dmap_mask = cv.distanceTransform(mask.astype(np.uint8), cv.DIST_L2, 5)
        mask_edge_dist = np.where((dmap_mask > 0) * (dmap_mask < 4), dmap_ice, np.nan)

        target = np.where(dmap_ice < self.eta, 0, 1)
        diff = (target - mask)
        target_or_mask = (target + mask) > 0

        d_left = np.nanmean(mask_edge_dist[:, :cx], axis=1)
        d_left[np.isnan(d_left)] = 0
        dx_left = np.sum(diff[:, :cx], axis=1)

        d_right = np.nanmean(mask_edge_dist[:, cx:], axis=1)
        d_right[np.isnan(d_right)] = 0
        dx_right = np.sum(diff[:, cx:], axis=1)

        d_top = np.nanmean(mask_edge_dist[:cy, :], axis=0)
        d_top[np.isnan(d_top)] = 0
        dy_top = np.sum(diff[:cy, :], axis=0)

        d_bot = np.nanmean(mask_edge_dist[cy:, :], axis=0)
        d_bot[np.isnan(d_bot)] = 0
        dy_bot = np.sum(diff[cy:, :], axis=0)

        # smth_n = 11
        # dleft = smoothen_array(dleft, smth_n)
        # dright = smoothen_array(dright, smth_n)
        # dtop = smoothen_array(dtop, smth_n)
        # dbottom = smoothen_array(dbottom, smth_n)

        if show:
            plt.ioff()
            fig, ax = plt.subplots(1, 5)
            ax[0].imshow(mat)
            ax[1].imshow(dmap_ice)
            ax[2].imshow(target_or_mask)
            ax[3].imshow(mask_edge_dist)
            ax[4].imshow(diff)
            ax[0].set_title('mask + 2*ice')
            ax[1].set_title('L2 transform ice')
            ax[2].set_title('target or mask')
            ax[3].set_title('left cumsum')
            ax[4].set_title('diff')
            for a in ax:
                a.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)

            fig, ax = plt.subplots(1, 2)
            ax[0].plot(dx_left, np.arange(len(dx_left)))
            ax[0].plot(dx_right, np.arange(len(dx_right)))
            ax[1].plot(np.arange(len(dy_top)), dy_top)
            ax[1].plot(np.arange(len(dy_bot)), dy_bot)
            # plt.show()

        # Redraw mask
        new_mask = 255 * np.ones(self.mask.shape, dtype=np.uint8)
        dm = np.zeros((self.mask.shape[0], self.mask.shape[1], 4))

        # TODO: include calibration here

        dx_left = np.sign(dx_left)
        dx_right = np.sign(dx_right)
        dy_top = np.sign(dy_top)
        dy_bot = np.sign(dy_bot)

        clw = np.sum(self.mask[:, :cx] == 0, axis=1)
        lw = clw + dx_left * d_left * self.sensitivity
        lw = lw.astype(np.int32)
        for i in range(mask.shape[0]):
            if lw[i] >= mask.shape[1]:
                new_mask[i, :] = 0
                dm[i, :, 0] = 1
            elif lw[i] > 0:
                new_mask[i, :lw[i]] = 0
                dm[i, :lw[i], 0] = 1

        crw = np.sum(self.mask[:, cx:] == 0, axis=1)
        rw = crw + dx_right * d_right * self.sensitivity
        rw = rw.astype(np.int32)
        for i in range(mask.shape[0]):
            if rw[i] >= mask.shape[1]:
                new_mask[i, :] = 0
                dm[i, :, 1] = 1
            elif rw[i] > 0:
                new_mask[i, -rw[i]:] = 0
                dm[i, -rw[i]:, 1] = 1

        # ctw = np.sum(self.mask[:cy, :] == 0, axis=0)
        # tw = ctw + dy_top * d_top * self.sensitivity
        # tw = tw.astype(np.int32)
        # for j in range(mask.shape[1]):
        #     if tw[j] >= mask.shape[0]:
        #         new_mask[:, j] = 0
        #         dm[:, j, 2] = 1
        #     elif tw[j] > 0:
        #         new_mask[:tw[j], j] = 0
        #         dm[:tw[j], j, 2] = 1
        #
        # cbw = np.sum(self.mask[cy:, :] == 0, axis=0)
        # bw = cbw + dy_bot * d_bot * self.sensitivity
        # bw = bw.astype(np.int32)
        # for j in range(mask.shape[1]):
        #     if bw[j] >= mask.shape[0]:
        #         new_mask[:, j] = 0
        #         dm[:, j, 3] = 1
        #     elif bw[j] > 0:
        #         new_mask[-bw[j]:, j] = 0
        #         dm[-bw[j]:, j, 3] = 1

        if show:
            plt.figure()
            plt.plot(dx_left, label='dx left')
            plt.plot(d_left, label='d left')
            plt.plot(clw, label='clw')
            plt.plot(lw, label='lw')
            plt.legend()

            plt.figure()
            plt.imshow(np.sum(dm, axis=-1))
            plt.show()

        self.mask = new_mask

    def update_old2(self, img, show=False):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        mask, ice = find_mask_and_ice(img)

        ice_edges = find_edges(ice, largest_only=True, remove_inside=True)
        # mask_edges = find_edges(mask, remove_outside=True)
        mat = mask.copy()
        cv.fillPoly(mat, [ice_edges], (2, 2, 2))

        not_ice = 1 - (mat - mask)/2
        dmap_mask = cv.distanceTransform(mask.astype(np.uint8), cv.DIST_L2, 5)
        dmap_ice = cv.distanceTransform(not_ice.astype(np.uint8), cv.DIST_L2, 5)

        target = np.where(dmap_ice < self.eta, 0, 1)
        diff = (target - mask) * dmap_ice / self.eta
        target_or_mask = (target + mask) > 0

        left_cs = np.cumsum(np.abs(np.diff(target_or_mask, axis=1)), axis=1)
        left_dx = np.cumsum(np.abs(np.diff(mat, axis=1)), axis=1)
        dleft = np.sum(np.where(left_cs == 0, diff[:, :-1], 0), axis=1)
        dleft[np.sum(mask, axis=1) == mask.shape[1]] = 0

        right_cs = np.cumsum(np.abs(np.diff(target_or_mask[:, ::-1], axis=1)), axis=1)
        dright = np.sum(np.where(right_cs == 0, diff[:, :-1], 0), axis=1)
        dright[np.sum(mask, axis=1) == mask.shape[1]] = 0

        top_cs = np.cumsum(np.abs(np.diff(target_or_mask, axis=0)), axis=0)
        dtop = np.sum(np.where(top_cs == 0, diff[:-1, :], 0), axis=0)
        dtop[np.sum(mask, axis=0) == mask.shape[0]] = 0

        bottom_cs = np.cumsum(np.abs(np.diff(target_or_mask[::-1, :], axis=0)), axis=0)
        dbottom = np.sum(np.where(bottom_cs == 0, diff[:-1, :], 0), axis=0)
        dbottom[np.sum(mask, axis=0) == mask.shape[0]] = 0

        # smth_n = 11
        # dleft = smoothen_array(dleft, smth_n)
        # dright = smoothen_array(dright, smth_n)
        # dtop = smoothen_array(dtop, smth_n)
        # dbottom = smoothen_array(dbottom, smth_n)

        mask_edge_L2 = dmap_ice.copy()
        mask_edge_L2[dmap_mask == 0] = np.nan
        mask_edge_L2[dmap_mask > 2] = np.nan

        if show:
            plt.ioff()
            fig, ax = plt.subplots(1, 5)
            ax[0].imshow(mat)
            ax[1].imshow(dmap_ice)
            ax[2].imshow(target_or_mask)
            ax[3].imshow(left_dx)
            ax[4].imshow(diff)
            ax[0].set_title('mask + 2*ice')
            ax[1].set_title('L2 transform ice')
            ax[2].set_title('target or mask')
            ax[3].set_title('left cumsum')
            ax[4].set_title('diff')
            for a in ax:
                a.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)

            fig, ax = plt.subplots(1, 2)
            ax[0].plot(dleft, np.arange(len(dleft)))
            ax[0].plot(dright, np.arange(len(dright)))
            ax[1].plot(np.arange(len(dtop)), dtop)
            ax[1].plot(np.arange(len(dbottom)), dbottom)
            # plt.show()

        # Redraw mask
        new_mask = 255 * np.ones(self.mask.shape, dtype=np.uint8)

        # TODO: include calibration
        clw = np.sum(np.cumsum(np.diff(self.mask, axis=1), axis=1) == 0, axis=1)  # current mask width on left
        lw = clw + dleft * self.sensitivity
        lw = lw.astype(np.int32)
        lw[lw < 0] = 0
        lw[lw > mask.shape[1]] = mask.shape[1]
        for i in range(mask.shape[0]):
            new_mask[i, :lw[i]] = 0

        crw = np.sum(np.cumsum(np.diff(self.mask[:, ::-1], axis=1), axis=1) == 0, axis=1)  # current mask width on right
        rw = crw + dright * self.sensitivity
        rw = rw.astype(np.int32)
        rw[rw < 0] = 0
        rw[rw > mask.shape[1]] = mask.shape[1]
        for i in range(mask.shape[0]):
            new_mask[i, -rw[i]:] = 0

        ctw = np.sum(np.cumsum(np.diff(self.mask, axis=0), axis=0) == 0, axis=0)  # current mask width on top
        tw = ctw + dtop * self.sensitivity
        tw = tw.astype(np.int32)
        tw[tw < 0] = 0
        tw[tw > mask.shape[0]] = mask.shape[0]
        for j in range(mask.shape[1]):
            new_mask[:tw[j], j] = 0

        if show:
            plt.figure()
            plt.imshow(new_mask)

        cbw = np.sum(np.cumsum(np.diff(self.mask[::-1, :], axis=0), axis=0) == 0, axis=0)  # current mask width on bottom
        bw = cbw + dbottom * self.sensitivity
        bw = bw.astype(np.int32)
        bw[bw < 0] = 0
        bw[bw > mask.shape[0]] = mask.shape[0]
        for j in range(mask.shape[1]):
            new_mask[-bw[j]:, j] = 0

        if show:
            plt.figure()
            plt.imshow(new_mask)

            plt.figure()
            plt.plot(cbw)
            plt.plot(dbottom)
            plt.plot(bw)
            plt.show()

        self.mask = new_mask

    def update_old(self, img):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        mask, ice = find_mask_and_ice(img)

        ice_edges = find_edges(ice, largest_only=True, remove_inside=True)
        mask_edges = find_edges(mask, remove_outside=True)
        mat = mask.copy()
        cv.fillPoly(mat, [ice_edges], (2, 2, 2))

        # distmap = np.zeros(mat.shape)
        # dxmap = np.zeros(mat.shape)
        # dymap = np.zeros(mat.shape)
        # for i in range(mat.shape[0]):
        #     for j in range(mat.shape[1]):
        #         dx = ice_edges[:, 0] - i
        #         dy = ice_edges[:, 1] - j
        #         d = np.sqrt(dx**2 + dy**2)
        #         distmap[j, i] = np.min(d) - self.eta
        #         dxmap[j, i] = dx[np.argmin(d)]
        #         dymap[j, i] = dy[np.argmin(d)]
        # distmap[mat == 2] = 0
        # dxmap[mat == 2] = 0

        # from left
        ind = [i for i in range(mat.shape[0]) if np.any(mat[i, :] == 0)]
        start, end = min(ind), max(ind)
        dmat = np.cumsum(np.abs(np.diff(mat, axis=1)), axis=1)
        darr_left = np.zeros((mat.shape[0], 1))
        for i in range(start, end+1):
            j = np.sum(dmat[i, :] == 0)
            dx = ice_edges[:, 0] - j
            dy = ice_edges[:, 1] - i
            d = np.sqrt(dx**2 + dy**2)
            darr_left[i] = np.min(d) - self.eta

        # from top
        ind = [j for j in range(mat.shape[1]) if np.any(mat[:, j] == 0)]
        start, end = min(ind), max(ind)
        dmat = np.cumsum(np.abs(np.diff(mat, axis=0)), axis=0)
        darr_top = np.zeros((1, mat.shape[1]))
        for j in range(start, end+1):
            # d = [np.sqrt(np.sum(dmat[j, :] == 1)**2 + (i - j)**2) for j in range(i - self.eta, i+self.eta)]
            i = np.sum(dmat[:, j] == 0)
            dx = ice_edges[:, 0] - j
            dy = ice_edges[:, 1] - i
            d = np.sqrt(dx**2 + dy**2)
            darr_top[0, j] = np.min(d) - self.eta

        # from bottom
        ind = [j for j in range(mat.shape[1]) if np.any(mat[:, j] == 0)]
        start, end = min(ind), max(ind)
        dmat = np.cumsum(np.abs(np.diff(mat, axis=0)), axis=0)
        dmat = dmat[-1, :] - dmat
        darr_bot = np.zeros((1, mat.shape[1]))
        for j in range(start, end+1):
            i = mat.shape[0] - np.sum(dmat[:, j] == 0)
            dx = ice_edges[:, 0] - j
            dy = ice_edges[:, 1] - i
            d = np.sqrt(dx**2 + dy**2)
            darr_bot[0, j] = np.min(d) - self.eta

        # from right
        ind = [i for i in range(mat.shape[0]) if np.any(mat[i, :] == 0)]
        start, end = min(ind), max(ind)
        dmat = np.cumsum(np.abs(np.diff(mat, axis=1)), axis=1)
        dmat = dmat[:, -1:] - dmat
        darr_right = np.zeros((mat.shape[0], 1))
        for i in range(start, end+1):
            j = mat.shape[1] - np.sum(dmat[i, :] == 0)
            dx = ice_edges[:, 0] - j
            dy = ice_edges[:, 1] - i
            d = np.sqrt(dx**2 + dy**2)
            darr_right[i] = np.min(d) - self.eta

        distmap = cv.distanceTransform(255-self.mask, cv.DIST_L2, 5).astype(np.float64)
        distmap -= cv.distanceTransform(self.mask, cv.DIST_L2, 5).astype(np.float64)

        dm2 = distmap + darr_left * self.sensitivity + darr_top * self.sensitivity + darr_bot * self.sensitivity + darr_right * self.sensitivity
        self.mask = np.where(dm2 > 0, 0, 255).astype(np.uint8)

        # plt.ioff()
        # fig, ax = plt.subplots(1, 5)
        # ax[0].imshow(mask)
        # ax[1].imshow(ice)
        # ax[2].imshow(mat)
        # ax[3].imshow(distmap)
        # # ax[4].imshow(img)
        # # ax[4].plot(ice_edges[:, 0], ice_edges[:, 1])
        # # ax[4].plot(mask_edges[:, 0], mask_edges[:, 1])
        # ax[4].imshow(dm2)
        # ax[0].set_title('mask')
        # ax[1].set_title('ice')
        # ax[2].set_title('mat')
        # ax[3].set_title('distTransform')
        # ax[4].set_title('dm2')
        # for a in ax:
        #     a.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        #
        # plt.figure()
        # plt.plot(darr_left, np.arange(len(darr_left)))
        # plt.show()

    def control(self, error):
        # sgn = np.sign(error)
        # return 10 * sgn if abs(error)/self.eta > 1 else 2 * sgn if abs(error)/self.eta > 0.1 else 0  # fuzzy
        return self.sensitivity * error  # proportional

    def get_img(self):
        return self.mask


def init_manual_mask():
    im = np.zeros((1500, 500, 3), dtype=np.uint8)
    im[50:1450, 50:450] = 255
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

