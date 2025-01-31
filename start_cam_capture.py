import sys
from Camera import Camera


SAVE_FOLDER = 'C:/Users/local.la/Documents/Simen/ColdRoom/working_folder/'
cam_settings = Camera.Settings(aperture='6.3', shutter_speed='1/50', iso=160)
cam_control_cmd_path = 'C:/Program Files (x86)/digiCamControl/CameraControlCmd.exe'
cam = Camera(cam_control_cmd_path, save_folder=SAVE_FOLDER, image_type='NEF')
cam.setup(cam_settings)


def main(args):
    if len(args) > 1:
        dt = float(args[1])
    else:
        dt = 10  # default time interval of 10 seconds

    cam.capture_multiple_images(10000, frequency=1/dt)


if __name__ == '__main__':
    main(sys.argv)

