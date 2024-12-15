from ice_detection import *
import matplotlib.pyplot as plt


class AutoMask:
    def __init__(self):
        pass

    def update(self, img):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        mask, ice = find_mask_and_ice(img)

        plt.figure()
        plt.imshow(mask)

        plt.figure()
        plt.imshow(ice)

        ice_edges = find_edges(ice, largest_only=True, remove_inside=True)
        mask_edges = find_edges(mask, remove_outside=True)

        mat = np.zeros(ice.shape, dtype=np.uint8)
        if mask_edges is not None:
            mat = cv.fillPoly(mat, [mask_edges.astype(np.int32)], color=(1, 1, 1))
            plt.plot(mask_edges[:, 0], mask_edges[:, 1], 'og')
        if ice_edges is not None:
            mat = cv.fillPoly(mat, [ice_edges.astype(np.int32)], color=(2, 2, 2))
            plt.plot(ice_edges[:, 0], ice_edges[:, 1], 'or')

        top = np.mean(mask_edges[mask_edges[:, 1] == np.min(mask_edges[:, 1])], axis=0)
        bottom = np.mean(mask_edges[mask_edges[:, 1] == np.max(mask_edges[:, 1])], axis=0)
        center_x = int((top[0] + bottom[0]) / 2)
        top_y = int(top[1])
        bot_y = int(bottom[1])

        plt.figure()
        plt.imshow(mat)

        plt.plot(top[0], top[1], '^')
        plt.plot(bottom[0], bottom[1], 'v')

        left = mat[:, :center_x]
        right = mat[:, center_x:]

        # left = mat.copy()
        # left[:, center_x:] = np.reshape(left[:, center_x], (left.shape[0], 1)).dot(np.ones((1, left.shape[1]-center_x)))

        maskbw_l = np.sum(left < 2, axis=1)  # width of mask black and white area on left side
        maskbw_r = np.sum(right < 2, axis=1)  # width of mask black and white area on right side
        maskb_l = np.sum(left == 0, axis=1)  # width of mask black area on left side
        maskb_r = np.sum(right == 0, axis=1)  # width of mask black area on right side
        # plt.figure()
        # plt.plot(widths_l, np.arange(len(widths_l)))
        # plt.plot(widths_r, np.arange(len(widths_r)))

        eta = 800  # target width in camera pixels
        r_l = np.nan * np.zeros((left.shape[0], 2*eta))
        r_r = np.nan * np.zeros((right.shape[0], 2*eta))

        r_l[:, 0] = maskbw_l - maskb_l
        r_r[:, 0] = maskbw_r - maskb_r
        for i in range(1, eta):
            r_l[:-i, -i] = np.sqrt((maskbw_l[i:] - maskb_l[:-i])**2 + i**2)
            r_l[i:, i] = np.sqrt((maskbw_l[:-i] - maskb_l[i:])**2 + i**2)

            r_r[:-i, -i] = np.sqrt((maskbw_r[i:] - maskb_r[:-i])**2 + i**2)
            r_r[i:, i] = np.sqrt((maskbw_r[:-i] - maskb_r[i:])**2 + i**2)

        error_l = np.nanmin(r_l, axis=1) - eta
        error_r = np.nanmin(r_r, axis=1) - eta

        plt.figure()
        plt.plot(error_l, np.arange(len(error_l)))
        plt.plot(error_r, np.arange(len(error_r)))

        new_l = left.copy()
        new_r = right.copy()
        for i in range(left.shape[0]):
            w_l = maskb_l[i] + error_l[i]
            if 0 < w_l <= new_l.shape[1]:
                new_l[i, :int(w_l)] += 3

            w_r = maskb_r[i] + error_r[i]
            if 0 < w_r <= new_r.shape[1]:
                new_r[i, -int(w_r):] += 3

        plt.figure()
        plt.imshow(new_l)
        plt.figure()
        plt.imshow(new_r)

        plt.figure()
        plt.plot(r_r[7940, :], '-o')
        print(error_r[7940])

        plt.show()


