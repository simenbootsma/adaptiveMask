import cv2 as cv
import numpy as np


def find_mask_and_ice(img):
    # assumes gray image and assumes object suspended from the top
    ret, otsu = cv.threshold(img.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    start_ind = [i for i in range(otsu.shape[0]) if np.any(otsu[i, :] > 0)][0]
    otsu = otsu[start_ind:, :]  # skip black part on top
    otsu[:10, :] = 255  # make top edge white

    s0, s1 = otsu.shape
    mask = np.zeros(otsu.shape)
    edge_centers = [(0, s1//2), (s0//2, 0), (s0-1, s1//2), (s0//2, s1-1)]  # left, top, right, bottom
    for i, j in edge_centers:
        if mask[i, j] == 0 and otsu[i, j] == 0:
            empty_mat = np.zeros((s0 + 2, s1 + 2), dtype=np.uint8)
            _, _, m, _ = cv.floodFill(otsu.copy(), empty_mat, (j, i), 0)
            mask[m[1:-1, 1:-1] == 1] = 1
    ice = np.where(otsu == 255, 0, 1)
    ice[mask==1] = 0
    mask[:10, :] = 1  # add top edge back in
    return mask, ice


def find_edges(img, largest_only=False, remove_outside=False, remove_inside=False):
    if largest_only:
        cont, hierarchy = cv.findContours(img.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(cont) == 0:
            return None  # no contours found

        idx = np.argmax([len(c) for c in cont])  # index of largest contour
        edges = np.reshape(cont[idx], (cont[idx].shape[0], 2))
        if remove_inside:
            return remove_inner_points(edges)
        return edges
    else:
        conts, hierarchy = cv.findContours(img.astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        if len(conts) == 0:
            return None  # no contours found

        # stack together
        edges = np.array([0, 0])
        for c in conts:
            edges = np.vstack((edges, np.reshape(c, (c.shape[0], 2))))

        # remove box edges
        if remove_outside:
            edges = edges[edges[:, 0] > 0]
            edges = edges[edges[:, 0] < img.shape[1]-1]
            edges = edges[edges[:, 1] > 0]
            edges = edges[edges[:, 1] < img.shape[0]-1]
        return edges[1:, :]


def remove_inner_points(edges):
    xmean = np.mean(edges[:, 0])
    nc_left, nc_right = [], []
    cl, cr = edges[edges[:, 0] < xmean], edges[edges[:, 0] >= xmean]
    for j in range(0, int(np.max(edges[:, 1])) + 1):
        cl1 = cl[cl[:, 1] == j]
        cr1 = cr[cr[:, 1] == j]
        if len(cl1) > 0:
            nc_left.append(cl1[np.argmax(np.abs(cl1[:, 0] - xmean))])
        if len(cr1) > 0:
            nc_right.insert(0, cr1[np.argmax(np.abs(cr1[:, 0] - xmean))])
    edges = np.array(nc_left + nc_right)
    return edges

