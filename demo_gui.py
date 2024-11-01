import flet as ft
import os
from test_calibration import load_calibration
import numpy as np


calib = None


def main(page: ft.Page):
    # CALIBRATION
    def pick_folder_result(e: ft.FilePickerResultEvent):
        calibration_save_folder.value = e.path
        calibration_save_folder.update()

    def calibrate(e):
        global calib
        calib = load_calibration()
        calib['keep_sides'] = [True, True, True, True]
        check_calibration_gui(calib)
        gd.content = ft.Image(src=os.getcwd() + "/_calibration_result.png", width=640, height=400,
                              fit=ft.ImageFit.CONTAIN)
        gd.update()

    def calibration_image_tap(e: ft.TapEvent):
        if calib is not None:
            vb = calib['view_box']
            fac1, fac2 = calib['screen_size'][0]/400, calib['screen_size'][1]/640
            x, y = e.local_x * fac1, e.local_y * fac2

            box_corners = [np.array([vb[0], vb[2]]), np.array([vb[0], vb[3]]),
                           np.array([vb[1], vb[2]]), np.array([vb[1], vb[3]])]  # top left, bottom left, top right, bottom right
            ind = np.argsort([np.sum([(np.array([x, y] - bc)) ** 2]) for bc in
                              box_corners])  # indices of box corners sorted by distance to clicked point
            edge_ind = [None, 0, 1, None, 2, 3][
                ind[0] + ind[1]]  # 0 and 3 are impossible, as 3 would be opposing corners
            calib['keep_sides'][edge_ind] = not calib['keep_sides'][edge_ind]
            check_calibration_gui(calib)

            edges = [
                (int(box_corners[0][0] / fac1), int(box_corners[0][1] / fac2), int(box_corners[1][0] / fac1), int(box_corners[1][1] / fac2), 0),  # left
                (int(box_corners[0][0] / fac1), int(box_corners[0][1] / fac2), int(box_corners[2][0] / fac1), int(box_corners[2][1] / fac2), 1),  # top
                (int(box_corners[1][0] / fac1), int(box_corners[1][1] / fac2), int(box_corners[3][0] / fac1), int(box_corners[3][1] / fac2), 2),  # bottom
                (int(box_corners[2][0] / fac1), int(box_corners[2][1] / fac2), int(box_corners[3][0] / fac1), int(box_corners[3][1] / fac2), 3)  # right
            ]

            colors = [ft.colors.GREEN if ks else ft.colors.RED for ks in calib['keep_sides']]
            paints = [ft.Paint(stroke_width=4, style=ft.PaintingStyle.FILL, color=c) for c in colors]

            lines = [
                ft.canvas.line.Line(x1, y1, x2, y2, paints[i]) for x1, y1, x2, y2, i in edges
            ]

            gd.content = ft.Stack([
                ft.Image(src=os.getcwd() + "/_calibration_result.png", width=640, height=400, fit=ft.ImageFit.CONTAIN),
                ft.canvas.Canvas(lines, width=640, height=400)])
            gd.update()

    calibration_save_folder = ft.Text(os.getcwd() + "/calibration_files/")
    calibration_save_folder_picker = ft.FilePicker(on_result=pick_folder_result)

    page.overlay.append(calibration_save_folder_picker)

    gd = ft.GestureDetector(mouse_cursor=ft.MouseCursor.MOVE, on_tap_down=calibration_image_tap,
                            content=ft.Container(width=640, height=400, bgcolor=ft.colors.GREY))

    # LAYOUT

    t = ft.Tabs(
        selected_index=1,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text="Dashboard",
                icon=ft.icons.DASHBOARD,
                content=ft.Container(
                    content=ft.Text("This is Tab 1"), alignment=ft.alignment.center
                ),
            ),
            ft.Tab(
                text='Calibration',
                icon=ft.icons.CONTROL_CAMERA,
                content=ft.Container(
                    ft.SafeArea(ft.Column([
                        ft.Text("Calibration", size=30),
                        ft.Row([
                            ft.ElevatedButton(text='Choose save folder', icon=ft.icons.FOLDER,
                                              on_click=lambda _: calibration_save_folder_picker.get_directory_path()),
                            calibration_save_folder
                        ]),
                        ft.TextField(label="Save filename", hint_text='save_name', suffix_text='.txt', width=200),
                        ft.Row([
                            ft.ElevatedButton(text='Calibrate', color=ft.colors.GREEN, on_click=calibrate),
                            ft.Checkbox(label='Crop camera view', value=False)
                        ]),
                        gd
                    ], spacing=10))
                ),
            ),
            ft.Tab(
                text="Run",
                icon=ft.icons.PLAY_ARROW,
                content=ft.Text("This is Tab 3"),
            ),
        ],
        expand=1,
    )

    page.add(t)
    page.window.height = 800
    page.update()


def check_calibration_gui(cal):
    import cv2 as cv

    screen = cv.imread('polar_bear_mosaic.png')
    cam_img = cv.imread('calibration_test_photos/test_photo2.jpg')

    # draw a bounding box around the detected result and display the image
    screen_intensity = np.mean(screen)

    view_box, mask_box = cal['view_box'], cal['mask_box']
    cam_img = cam_img[mask_box[2]:mask_box[3], mask_box[0]:mask_box[1]]

    cam_img = cam_img.astype(np.float64) / np.mean(cam_img) * screen_intensity
    cam_img[cam_img > 255] = 255
    cam_img = cam_img.astype(np.uint8)

    resized_cam = cv.resize(cam_img, (view_box[1] - view_box[0], view_box[3] - view_box[2]))
    if (view_box[0] < 0) or (view_box[1] >= screen.shape[1]) or (view_box[2] < 0) or (view_box[3] >= screen.shape[0]):
        raise ValueError("View box lies outside of screen, use mask on camera image using the 'mask' flag")

    screen[view_box[2]:view_box[3], view_box[0]:view_box[1]] = resized_cam
    img = cv.resize(screen, (screen.shape[1] // 2, screen.shape[0] // 2))
    cv.imwrite("_calibration_result.png", img)


ft.app(main)

