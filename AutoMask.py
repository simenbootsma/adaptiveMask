from ice_detection import *
import matplotlib.pyplot as plt
from ManualMask import Cylinder


BIG_NUMBER = 10**9


class AutoMask:
    def __init__(self):
        self.mask = cv.cvtColor(init_manual_mask(), cv.COLOR_RGB2GRAY)
        self.sensitivity = 0.2  # multiplier for the error

    def update(self, img):
        # y, err_l, err_r = compute_errors_plotting(img)
        y, err_l, err_r = compute_errors(img)

        screen = self.mask.copy()
        screen[:10, :] = 0
        mask_edges = find_edges(screen, remove_outside=True)
        top = np.mean(mask_edges[mask_edges[:, 1] == np.min(mask_edges[:, 1])], axis=0)
        bottom = np.mean(mask_edges[mask_edges[:, 1] == np.max(mask_edges[:, 1])], axis=0)
        center_x = int((top[0] + bottom[0]) / 2)
        top_y = int(top[1])
        bot_y = int(bottom[1])

        # Convert to screen coordinates
        L = bot_y - top_y
        y = (top_y + y * L)
        err_l = err_l * L
        err_r = err_r * L

        I = np.arange(self.mask.shape[0])
        el = np.interp(I, y, err_l, left=0, right=0)
        er = np.interp(I, y, err_r, left=0, right=0)

        # Apply
        left_width = np.sum(self.mask[:, :center_x] == 0, axis=1)
        right_width = np.sum(self.mask[:, center_x:] == 0, axis=1)

        # plt.ioff()
        # plt.figure()
        # plt.plot(el, I)
        # plt.plot(er, I)
        # plt.figure()
        # plt.plot(left_width, I)
        # plt.plot(right_width, I)
        # plt.show()

        new_mask = 255 * np.ones(self.mask.shape, dtype=np.uint8)
        for i in range(new_mask.shape[0]):
            new_mask[i, :int(left_width[i] + el[i] * self.sensitivity)] = 0
            new_mask[i, -int(right_width[i] + er[i] * self.sensitivity):] = 0

        # plt.imshow(new_mask)
        # plt.show()
        self.mask = new_mask

    def get_img(self):
        return self.mask


def init_manual_mask():
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


def compute_errors(img):
    eta = 400  # target width in camera pixels

    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    mask, ice = find_mask_and_ice(img)

    ice_edges = find_edges(ice, largest_only=True, remove_inside=True)
    mask_edges = find_edges(mask, remove_outside=True)

    if ice_edges is None or mask_edges is None:
        return None

    mat = np.zeros(ice.shape, dtype=np.uint8)
    mat = cv.fillPoly(mat, [mask_edges.astype(np.int32)], color=(1, 1, 1))
    mat = cv.fillPoly(mat, [ice_edges.astype(np.int32)], color=(2, 2, 2))

    top = np.mean(mask_edges[mask_edges[:, 1] == np.min(mask_edges[:, 1])], axis=0)
    bottom = np.mean(mask_edges[mask_edges[:, 1] == np.max(mask_edges[:, 1])], axis=0)
    center_x = int((top[0] + bottom[0]) / 2)
    top_y = int(top[1])
    bot_y = int(bottom[1])

    left = mat[:, :center_x]
    right = mat[:, center_x:]

    maskbw_l = np.sum(left < 2, axis=1)  # width of mask black and white area on left side
    maskbw_r = np.sum(right < 2, axis=1)  # width of mask black and white area on right side
    maskb_l = np.sum(left == 0, axis=1)  # width of mask black area on left side
    maskb_r = np.sum(right == 0, axis=1)  # width of mask black area on right side
    ice_width_l = np.sum(left == 2, axis=1)  # width of ice on the left side
    ice_width_r = np.sum(right == 2, axis=1)  # width of ice on the right side

    # Add big number to ignore parts where there is no ice in the error computation. Might be a better way.....?
    maskbw_l[ice_width_l == 0] += BIG_NUMBER
    maskbw_r[ice_width_r == 0] += BIG_NUMBER

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

    # Subtract big number again
    error_r[error_r > BIG_NUMBER/2] -= BIG_NUMBER - eta
    error_l[error_l > BIG_NUMBER / 2] -= BIG_NUMBER - eta

    # Normalize coordinates
    L = (bot_y - top_y)  # length scale (height of the mask)
    y = (np.arange(mat.shape[0]) - top_y) / L
    error_l = error_l / L
    error_r = error_r / L
    return y, error_l, error_r


def compute_errors_plotting(img):
    plt.ioff()
    eta = 400  # target width in camera pixels

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

    # left = np.hstack((left, np.repeat(np.reshape(left[:, -1], (left.shape[0], 1)), eta, axis=1)))
    # right = np.hstack((np.repeat(np.reshape(right[:, 0], (right.shape[0], 1)), eta, axis=1), right))

    maskbw_l = np.sum(left < 2, axis=1)  # width of mask black and white area on left side
    maskbw_r = np.sum(right < 2, axis=1)  # width of mask black and white area on right side
    maskb_l = np.sum(left == 0, axis=1)  # width of mask black area on left side
    maskb_r = np.sum(right == 0, axis=1)  # width of mask black area on right side
    ice_width_l = np.sum(left == 2, axis=1)
    ice_width_r = np.sum(right == 2, axis=1)

    maskbw_l[ice_width_l == 0] += BIG_NUMBER
    maskbw_r[ice_width_r == 0] += BIG_NUMBER

    # plt.figure()
    # plt.plot(widths_l, np.arange(len(widths_l)))
    # plt.plot(widths_r, np.arange(len(widths_r)))

    r_l = np.nan * np.zeros((left.shape[0], 2 * eta))
    r_r = np.nan * np.zeros((right.shape[0], 2 * eta))

    r_l[:, 0] = maskbw_l - maskb_l
    r_r[:, 0] = maskbw_r - maskb_r
    for i in range(1, eta):
        r_l[:-i, -i] = np.sqrt((maskbw_l[i:] - maskb_l[:-i]) ** 2 + i ** 2)
        r_l[i:, i] = np.sqrt((maskbw_l[:-i] - maskb_l[i:]) ** 2 + i ** 2)

        r_r[:-i, -i] = np.sqrt((maskbw_r[i:] - maskb_r[:-i]) ** 2 + i ** 2)
        r_r[i:, i] = np.sqrt((maskbw_r[:-i] - maskb_r[i:]) ** 2 + i ** 2)

    error_l = np.nanmin(r_l, axis=1) - eta
    error_r = np.nanmin(r_r, axis=1) - eta

    error_r[error_r > BIG_NUMBER / 2] -= BIG_NUMBER - eta
    error_l[error_l > BIG_NUMBER / 2] -= BIG_NUMBER - eta

    # plt.figure()
    # plt.plot(error_l, np.arange(len(error_l)))
    # plt.plot(error_r, np.arange(len(error_r)))

    new_l = left.copy()
    new_r = right.copy()
    for i in range(left.shape[0]):
        w_l = maskb_l[i] + error_l[i]
        new_l[i, :int(w_l)] += 3

        w_r = maskb_r[i] + error_r[i]
        new_r[i, -int(w_r):] += 3

    # plt.figure()
    # plt.imshow(new_l)
    # plt.figure()
    # plt.imshow(new_r)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(new_l)
    ax[1].imshow(new_r)
    plt.show()

    # Normalize coordinates
    L = (bot_y - top_y)  # length scale (height of the mask)
    y = (np.arange(mat.shape[0]) - top_y) / L
    error_l = error_l / L
    error_r = error_r / L

    # plt.figure()
    # plt.plot(error_l)
    # plt.plot(error_r)
    # plt.show()
    return y, error_l, error_r

