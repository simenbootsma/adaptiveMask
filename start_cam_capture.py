import sys
from Camera import Camera
import time


SAVE_FOLDER = 'C:/Users/local.la/Documents/Simen/ColdRoom/working_folder/'
cam_control_cmd_path = 'C:/Program Files (x86)/digiCamControl/CameraControlCmd.exe'
cam = Camera(cam_control_cmd_path, save_folder=SAVE_FOLDER, image_type='NEF')


def main(args=None):
    if args is not None and len(args) > 1:
        dt = float(args[1])
    else:
        dt = 10  # default time interval of 10 seconds

    while True:
        st = time.time()
        cam.capture_single_image()

        sleep_time = dt - (time.time() - st)
        if sleep_time <= 0:
            print("[cam_capture] WARNING: sleep time smaller than 0, interval time is too low ")
        else:
            time.sleep(sleep_time)


if __name__ == '__main__':
    main(sys.argv)

