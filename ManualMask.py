import sys

import numpy as np
import cv2 as cv

"""
CONTROLS

    Esc         : exit
    Back        : reset
    f           : flip image
    left arrow  : move left
    right arrow : move right
    up arrow    : move up
    down arrow  : move down
    s (+ shift) : increase (decrease) sensitivity
    h (+ shift) : increase (decrease) height 
    w (+ shift) : increase (decrease) width
    b (+ shift) : increase (decrease) blur 
    k (+ shift) : increase (decrease) curvature
    c (+ shift) : increase (decrease) contrast
    g (+ shift) : increase (decrease) height of black space at cylinder base 

    NOTE: shift + <key> will perform the inverse operation (when available)
"""


class Cylinder:
    def __init__(self, resolution=(1920, 1080)):
        self.resolution = resolution
        self.sensitivity = int(resolution[1] / 500)
        self.center = resolution[1] // 2
        self.width = resolution[1] // 3
        self.height = int(resolution[0] * 3 / 4)
        self.bs_height = 0
        self.blur = 0
        self.curvature = 1
        self.flipped = False
        self.transposed = False
        self.contrast = 1.0
        self.color_idx = 0
        self.color = (255, 255, 255)
        self.random_dot = False

    def move_down(self):
        if self.transposed:
            self.move_right()
        else:
            self.center = min(self.resolution[1]-self.width//2, self.center + self.sensitivity)

    def move_up(self):
        if self.transposed:
            self.move_left()
        else:
            self.center = max(self.width//2, self.center - self.sensitivity)

    def move_right(self):
        if not self.transposed:
            self.move_up()
        else:
            self.center = min(self.resolution[0]-self.width//2, self.center + self.sensitivity)

    def move_left(self):
        if not self.transposed:
            self.move_down()
        else:
            self.center = max(self.width//2, self.center - self.sensitivity)

    def increase_width(self):
        self.width = min(self.resolution[1 - self.transposed], int(2*self.height/(self.curvature+1e-5)-1), self.width + self.sensitivity)

    def decrease_width(self):
        self.width = max(2, self.width - self.sensitivity)

    def increase_height(self):
        self.height = min(self.resolution[self.transposed], self.height + self.sensitivity)

    def decrease_height(self):
        min_height = int(self.curvature*self.width/2)+1
        while self.height - self.sensitivity < min_height:
            self.decrease_curvature()
            min_height = int(self.curvature * self.width / 2) + 1
        self.height = self.height - self.sensitivity

    def increase_bs_height(self):
        self.bs_height = min(self.bs_height + self.sensitivity, self.height)

    def decrease_bs_height(self):
        self.bs_height = max(self.bs_height - self.sensitivity, 0)

    def increase_blur(self):
        self.blur += 1

    def decrease_blur(self):
        self.blur = max(0, self.blur - 1)

    def increase_curvature(self):
        self.curvature = min(2*self.height/self.width, self.curvature + 2*self.sensitivity/self.width)

    def decrease_curvature(self):
        self.curvature = max(0, self.curvature - 2*self.sensitivity/self.width)

    def increase_contrast(self):
        self.contrast = min(max(0., self.contrast + 0.01), 1)

    def decrease_contrast(self):
        self.contrast = min(max(0., self.contrast - 0.01), 1)

    def increase_sensitivity(self):
        self.sensitivity = min(self.resolution[1] // 3, self.sensitivity + 1)
        print("set sensivity to {:d} px".format(self.sensitivity))

    def decrease_sensivity(self):
        self.sensitivity = max(1, self.sensitivity - 1)
        print("set sensivity to {:d} px".format(self.sensitivity))

    def flip(self):
        self.flipped = not self.flipped

    def transpose(self):
        self.transposed = not self.transposed
        res_ratio = self.resolution[0] / self.resolution[1]
        if self.transposed:
            self.height = int(self.height / res_ratio)
            self.center = int(self.center * res_ratio)
        else:
            self.height = int(self.height * res_ratio)
            self.center = int(self.center / res_ratio)

    def change_color(self):
        colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.color_idx = 0 if self.color_idx == len(colors)-1 else self.color_idx + 1
        self.color = colors[self.color_idx]

    def handle_key(self, key):
        # temporary hack to test proportionality stuff
        s = None
        if type(key) is tuple:
            s = self.sensitivity
            self.sensitivity = int(key[1])
            key = key[0]

        char = key if type(key) is str else chr(key)
        func_map = {chr(1): self.move_down, chr(0): self.move_up, 'M': self.move_left, 'm': self.move_right, "w": self.increase_width, "W": self.decrease_width,
                    "h": self.increase_height, "H": self.decrease_height, "b": self.increase_blur,
                    "B": self.decrease_blur, "t": self.transpose,
                    "k": self.increase_curvature, "K": self.decrease_curvature, "f": self.flip,
                    "c": self.increase_contrast, "C": self.decrease_contrast, chr(127): self.__init__,
                    "s": self.increase_sensitivity, "S": self.decrease_sensivity, "o": self.change_color,
                    "g": self.increase_bs_height, "G": self.decrease_bs_height}
        if char in func_map:
            func_map[char]()

        if s is not None:
            self.sensitivity = s

    def get_img(self):
        img = np.zeros((self.resolution[1], self.resolution[0], 3))
        slices = [slice(self.center - self.width // 2, self.center + self.width // 2),
                  slice(0, self.height - int(self.curvature * self.width / 2))]
        if self.transposed:
            slices = slices[::-1]
        img[slices[0], slices[1], :] = self.color
        if self.curvature > 0:
            pos = (self.height - int(self.curvature * self.width / 2), self.center)
            radii = (int(self.curvature * self.width / 2), self.width // 2 - 1)
            if self.transposed:
                pos = pos[::-1]
                radii = radii[::-1]
            img = cv.ellipse(img, pos, radii, 0, 0, 360, self.color, -1)
        if self.bs_height > 0:
            slices[1 - self.transposed] = slice(0, self.bs_height)
            img[slices[0], slices[1], :] = 0
        if self.blur > 0:
            img = cv.blur(img.astype(np.uint8), (self.blur, self.blur))
        if self.flipped:
            img = np.flipud(img) if self.transposed else np.fliplr(img)
        img = (img * self.contrast).astype(np.uint8)
        return img


class Sphere:
    def __init__(self, resolution=(1920, 1080)):
        self.resolution = resolution
        self.sensitivity = int(resolution[1] / 500)
        self.center = [resolution[0] // 2, resolution[1] // 2]
        self.radius = resolution[1] // 4
        self.blur = 0
        self.contrast = 1.0
        self.color_idx = 0
        self.color = (255, 255, 255)

    def move_down(self):
        self.center[1] = min(self.resolution[1]-self.radius, self.center[1] + self.sensitivity)

    def move_up(self):
        self.center[1] = max(self.radius, self.center[1] - self.sensitivity)

    def move_right(self):
        self.center[0] = min(self.resolution[0]-self.radius, self.center[0] + self.sensitivity)

    def move_left(self):
        self.center[0] = max(self.radius, self.center[0] - self.sensitivity)

    def increase_radius(self):
        self.radius += self.sensitivity

    def decrease_radius(self):
        self.radius = max(1, self.radius - self.sensitivity)

    def increase_blur(self):
        self.blur += 1

    def decrease_blur(self):
        self.blur = max(0, self.blur - 1)

    def increase_contrast(self):
        self.contrast = min(max(0., self.contrast + 0.01), 1)

    def decrease_contrast(self):
        self.contrast = min(max(0., self.contrast - 0.01), 1)

    def increase_sensitivity(self):
        self.sensitivity = min(self.resolution[1] // 3, self.sensitivity + 1)
        print("set sensivity to {:d} px".format(self.sensitivity))

    def decrease_sensivity(self):
        self.sensitivity = max(1, self.sensitivity - 1)
        print("set sensivity to {:d} px".format(self.sensitivity))

    def change_color(self):
        colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.color_idx = 0 if self.color_idx == len(colors)-1 else self.color_idx + 1
        self.color = colors[self.color_idx]

    def handle_key(self, key):
        char = chr(key)
        func_map = {chr(2): self.move_left, chr(3): self.move_right, chr(0): self.move_up, chr(1): self.move_down,
                    "r": self.increase_radius, "R": self.decrease_radius, "b": self.increase_blur,
                    "B": self.decrease_blur,
                    "c": self.increase_contrast, "C": self.decrease_contrast, chr(127): self.__init__,
                    "s": self.increase_sensitivity, "S": self.decrease_sensivity, "o": self.change_color}
        if char in func_map:
            func_map[char]()

    def get_img(self):
        img = np.zeros((self.resolution[1], self.resolution[0], 3))
        img = cv.ellipse(img, self.center, (self.radius, self.radius), 0, 0, 360, self.color, -1)
        if self.blur > 0:
            img = cv.blur(img.astype(np.uint8), (self.blur, self.blur))
        img = (img * self.contrast).astype(np.uint8)
        return img


def main(args):
    if len(args) < 2 or args[1] not in ['cylinder', 'sphere']:
        print("\033[95m warning: Neither 'cylinder' nor 'sphere' given as argument, defaulting to 'cylinder'. \033[0m")
        args = ['', 'cylinder']
    obj = {'cylinder': Cylinder, 'sphere': Sphere}[args[1]]()

    cv.namedWindow("window", cv.WND_PROP_FULLSCREEN)
    # cv.moveWindow("window", 2000, 100)
    cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    while True:
        cv.imshow("window", obj.get_img())
        key = cv.waitKey()
        if key == 27:
            break
        obj.handle_key(key)
    cv.destroyWindow("window")


if __name__ == "__main__":
    main(sys.argv)

