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
                              fit=ft.ImageFit.SCALE_DOWN)
        gd.update()

    def calibration_image_tap(e: ft.TapEvent):
        if calib is not None:
            vb = calib['view_box']
            x, y = e.local_x * calib['size'][0]/640, e.local_y * calib['size'][1]/400
            print(x, y)
            box_corners = [np.array([vb[0], vb[2]]) / 2, np.array([vb[0], vb[3]]) / 2,
                           np.array([vb[1], vb[2]]) / 2, np.array([vb[1], vb[3]]) / 2]
            ind = np.argsort([np.sum([(np.array([x, y] - bc)) ** 2]) for bc in
                              box_corners])  # indices of box corners sorted by distance to clicked point
            edge_ind = [None, 0, 1, None, 2, 3][
                ind[0] + ind[1]]  # 0 and 3 are impossible, as 3 would be opposing corners
            calib['keep_sides'][edge_ind] = not calib['keep_sides'][edge_ind]
            check_calibration_gui(calib)
            gd.content = ft.Container(bgcolor=ft.colors.BLUE, width=50, height=50)
            gd.content = ft.Image(src=os.getcwd() + "/_calibration_result.png", width=640, height=400,
                                  fit=ft.ImageFit.SCALE_DOWN)  # TODO: fix this, it's not updating
            gd.update()

    calibration_save_folder = ft.Text(os.getcwd() + "/calibration_files/")
    calibration_save_folder_picker = ft.FilePicker(on_result=pick_folder_result)

    page.overlay.append(calibration_save_folder_picker)

    gd = ft.GestureDetector(mouse_cursor=ft.MouseCursor.MOVE, on_tap_down=calibration_image_tap,
                            content=None)

    # LAYOUT

    t = ft.Tabs(
        selected_index=1,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text="Dashboard",
                content=ft.Container(
                    content=ft.Text("This is Tab 1"), alignment=ft.alignment.center
                ),
            ),
            ft.Tab(
                tab_content=ft.Icon(ft.icons.CONTROL_CAMERA),
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
                text="Tab 3",
                icon=ft.icons.SETTINGS,
                content=ft.Text("This is Tab 3"),
            ),
        ],
        expand=1,
    )

    page.add(t)


def check_calibration_gui(calib):
    import numpy as np
    import cv2 as cv

    def draw_edges():
        # Redraw edges
        box_corners = [[view_box[0] // 2, view_box[2] // 2], [view_box[0] // 2, view_box[3] // 2],
                       [view_box[1] // 2, view_box[2] // 2], [view_box[1] // 2, view_box[3] // 2]]
        bc_ind = [[0, 1], [0, 2], [1, 3],
                  [2, 3]]  # indices of box corners corresponding to edges: left, top, bottom, right
        for i in range(4):
            color = (0, 255, 0) if calib['keep_sides'][i] else (0, 0, 255)
            p1, p2 = box_corners[bc_ind[i][0]], box_corners[bc_ind[i][1]]
            cv.line(img, p1, p2, color=color, thickness=2)

    screen = cv.imread('polar_bear_mosaic.png')
    cam_img = cv.imread('calibration_test_photos/test_photo2.jpg')

    # draw a bounding box around the detected result and display the image
    screen_intensity = np.mean(screen)

    view_box, mask_box = calib['view_box'], calib['mask_box']
    cam_img = cam_img[mask_box[2]:mask_box[3], mask_box[0]:mask_box[1]]

    cam_img = cam_img.astype(np.float64) / np.mean(cam_img) * screen_intensity
    cam_img[cam_img > 255] = 255
    cam_img = cam_img.astype(np.uint8)

    resized_cam = cv.resize(cam_img, (view_box[1] - view_box[0], view_box[3] - view_box[2]))
    if (view_box[0] < 0) or (view_box[1] >= screen.shape[1]) or (view_box[2] < 0) or (view_box[3] >= screen.shape[0]):
        raise ValueError("View box lies outside of screen, use mask on camera image using the 'mask' flag")

    screen[view_box[2]:view_box[3], view_box[0]:view_box[1]] = resized_cam
    img = cv.resize(screen, (screen.shape[1] // 2, screen.shape[0] // 2))
    draw_edges()
    cv.imwrite("_calibration_result.png", img)


ft.app(main)

