import numpy as np
import cv2 as cv
from multiprocessing import Process, Queue
import time
from glob import glob
from AdaptiveMask import AdaptiveMask
from Camera import Camera


SAVE_FOLDER = 'C:/Users/local.la/Documents/Masking/adaptiveMask/auto_images/'
CALIBRATION_FILENAME = None  # leave None to use most recent
cam_settings = Camera.Settings(aperture=2.5, shutter_speed='1/5', iso=160)
cam_control_cmd_path = 'C:/Program Files (x86)/digiCamControl/CameraControlCmd.exe'
cam = Camera(cam_control_cmd_path, save_folder=SAVE_FOLDER)
cam.setup(cam_settings)


def main():
    q = Queue()
    p0 = Process(target=show_screen_worker, args=(q,))
    p1 = Process(target=masking_worker, args=(q,))

    p0.start()
    p1.start()
    p0.join()
    repeat = q.get(timeout=10)
    p1.join(timeout=10)
    q.close()

    if repeat:
        print('Repeating')
        main()
    print("Exiting...")


def show_screen_worker(queue):
    cv_window()
    calib = load_calibration()
    img = np.ones((calib['screen_size'][1], calib['screen_size'][0], 3))

    while True:
        if not queue.empty():
            pkg = queue.get()
            if type(pkg) is np.ndarray:
                img = pkg
            else:
                queue.put(pkg)
                break
        cv.imshow("window", img)
        key = cv.waitKey(100)
        if key == 27:
            break
    cv.destroyWindow("window")


def masking_worker(queue):
    calib = load_calibration()

    # initialize adaptive mask
    mask = AdaptiveMask(**calib)
    queue.put(mask.screen)

    st = time.time()
    while True:
        if queue.empty():
            img = take_photo()
            mask.update(img)
            queue.put(mask.screen)
            st = time.time()
        else:
            time.sleep(0.1)
        if time.time() - st > 10:
            print("[masking_worker] received no image in 10 seconds, quitting...")
            break  # quit if queue has not been emptied in 10 seconds


def load_calibration():
    global CALIBRATION_FILENAME
    if CALIBRATION_FILENAME is None:
        CALIBRATION_FILENAME = sorted(glob('temp_files_calibration/*.txt'))[-1]
    data = {}
    with open(CALIBRATION_FILENAME, 'r') as f:
        for ln in f.read().splitlines():
            if ' : ' in ln:
                k, val = ln.split(' : ')
                data[k] = int(val) if '(' not in val else [int(v) for v in val[1:-1].split(', ')]
    data['transform'] = np.reshape(data['transform'], (2, 2))
    return data


def cv_window():
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    # cv.moveWindow("window", 900, 900)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


def take_photo():
    image_name = cam.collection_name + '_' + str(cam.image_index) + cam.image_type
    cam.capture_single_image(autofocus=False)
    #image_name = '_1.jpg'
    photo = cv.imread(SAVE_FOLDER + image_name)
    return photo


if __name__ == '__main__':
    main()

