import flet as ft
import os
import numpy as np
import time
from glob import glob
import cv2 as cv
from monitor_mask import find_mask_and_ice, find_edges


# TODO:
#   - Implement writing with GUI
#   - Implement changing thresholds
#   - Graph with parameters
#   - Live blinker

FOLDER = "C:/Users/Simen/OneDrive - University of Twente/VC_coldroom/ColdVC_20241212"  # must contain jpg, updates, commands folders
update_paths = sorted(glob(FOLDER + "/updates/*.txt"))
image_paths = sorted(glob(FOLDER + "/jpg/*.jpg"))
WINDOW_OPEN = True
last_update_time = None
start_time = None


def main(page: ft.Page):
    global update_paths, image_paths, WINDOW_OPEN, FOLDER, last_update_time, start_time
    # FUNCTIONALITY
    params = ['position', 'width', 'height', 'curvature']  # parameters that will be shown

    def quit_program():
        global WINDOW_OPEN
        WINDOW_OPEN = False

    # Dashboard
    def pick_folder_result(e: ft.FilePickerResultEvent):
        global FOLDER, update_paths, image_paths, last_update_time, start_time
        data_folder.value = e.path
        data_folder.update()
        FOLDER = e.path
        update_paths = sorted(glob(FOLDER + "/updates/*.txt"))
        image_paths = sorted(glob(FOLDER + "/jpg/*.jpg"))
        last_update_time, start_time = None, None

    data_folder = ft.Text(FOLDER)
    data_folder_picker = ft.FilePicker(on_result=pick_folder_result)
    page.overlay.append(data_folder_picker)

    live_image = ft.Image(
        src="",
        width=300,
        height=500,
        fit=ft.ImageFit.FIT_HEIGHT,
        repeat=ft.ImageRepeat.NO_REPEAT,
        border_radius=ft.border_radius.all(10),
    )

    def toggle_contour(e):
        if len(image_paths) == 0:
            return
        if live_image.src == '_temp_dashboard_image.jpg':
            live_image.src = image_paths[-1]
        else:
            img = cv.imread(image_paths[-1])
            gray_img = np.mean(img, axis=2).astype(np.uint8)
            contour = find_edges(find_mask_and_ice(gray_img)[1], largest_only=True)
            contour = remove_inner_contour_points(contour)
            img = cv.polylines(img, [contour.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=40)
            cv.imwrite('_temp_dashboard_image.jpg', img)
            live_image.src = '_temp_dashboard_image.jpg'
        live_image.update()

    contour_checkbox = ft.Checkbox(label='Show contour', on_change=toggle_contour)

    # Controls
    cboard = ControlBoard()

    def save_changes(e):
        save_button.disabled = True
        save_button.update()
        cboard.submit(data_folder.value)
        time.sleep(2)
        save_button.disabled = False
        save_button.update()

    save_button = ft.ElevatedButton('Save changes', on_click=save_changes)

    # LAYOUT
    row_h = 25  # row height for control tab
    col_w = 100  # column width for control tab
    last_update_text = ft.Text("Last update: -",)
    last_image_text = ft.Text("Last update: -", )
    run_time_text = ft.Text("Run time     --:--:--", size=30, weight=ft.FontWeight.W_300)
    status_boxes = {k: [ft.Container(width=40, height=40, visible=False, border_radius=3, border=ft.border.all(2, ft.colors.GREEN_50))] + [ft.Container(width=30, height=30, visible=False, border_radius=3) for _ in range(cboard.BUFFER_SIZE-1)] for k in params}
    status_rows = [ft.Row([ft.Container(ft.Text(k, size=20, weight=ft.FontWeight.BOLD), width=100, height=50, alignment=ft.Alignment(-1, 0))] + status_boxes[k], spacing=20) for k in params]
    status_rows.insert(0, ft.Row([ft.Container(width=100, height=50), ft.Icon(ft.icons.ARROW_BACK), ft.Container(ft.Text("time", size=20, weight=ft.FontWeight.BOLD), alignment=ft.Alignment(-1, 0))], spacing=20))

    t = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text="Dashboard",
                icon=ft.icons.DASHBOARD,
                content=ft.Container(
                    ft.SafeArea(ft.Row([
                        ft.Column([
                            ft.Row([
                                ft.ElevatedButton(text='Choose data folder', icon=ft.icons.FOLDER,
                                                  on_click=lambda _: data_folder_picker.get_directory_path()),
                                data_folder
                            ]),
                            run_time_text
                        ] + status_rows, spacing=10),
                        ft.Column([last_image_text, contour_checkbox, live_image]),
                    ], spacing=50)), margin=20
                ),
            ),

            ft.Tab(
                text="Controls",
                icon=ft.icons.CONTROL_CAMERA,
                content=ft.Container(
                    ft.SafeArea(ft.Column([
                        ft.Row([last_update_text]),
                        ft.Row([
                            ft.Column([ft.Text(""), ft.Text("Position", weight=ft.FontWeight.BOLD), ft.Text("Width", weight=ft.FontWeight.BOLD), ft.Text("Height", weight=ft.FontWeight.BOLD), ft.Text("Curvature", weight=ft.FontWeight.BOLD)], spacing=row_h, alignment=ft.alignment.center, width=col_w),
                            ft.Column([ft.Text("Current value", text_align=ft.TextAlign.CENTER, width=int(1.7*col_w), weight=ft.FontWeight.BOLD)] + [ft.Row([cboard.buttons[k + '_minus'], cboard.texts[k+'_current'], cboard.buttons[k + '_plus']], spacing=0, height=row_h-5, alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
                                       for k in params], spacing=row_h, width=int(1.7*col_w)),
                            ft.Column([ft.Text("Target", text_align=ft.TextAlign.CENTER, width=int(1.7 * col_w),
                                               weight=ft.FontWeight.BOLD)] + [ft.Row(
                                [cboard.buttons[k + '_target_minus'], cboard.texts[k + '_target'],
                                 cboard.buttons[k + '_target_plus']], spacing=0, height=row_h - 5,
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
                                                                              for k in params], spacing=row_h,
                                      width=int(1.7 * col_w)),
                            ft.Column([ft.Text("Threshold", text_align=ft.TextAlign.CENTER, width=int(1.7*col_w), weight=ft.FontWeight.BOLD)] + [ft.Row([cboard.buttons[k + '_thresh_minus'], cboard.texts[k+'_threshold'], cboard.buttons[k + '_thresh_plus']], spacing=0, height=row_h-5, alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
                                       for k in params], spacing=row_h, width=int(1.7*col_w)),
                            ft.Column(
                                [ft.Text("Error", text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD)] + [cboard.texts[k + '_abs_error'] for k in
                                                                                 params],
                                spacing=row_h, width=col_w),
                            ft.Column(
                                [ft.Text("Rel. error", text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD)] + [cboard.texts[k + '_rel_error'] for k in
                                                                                 params],
                                spacing=row_h, width=col_w)
                        ], spacing=5),
                        ft.Container(
                            ft.Row([ft.ElevatedButton(text='Reset', on_click=cboard.reset), save_button]), margin=30
                        )])
                    ), margin=20
                ),
            ),
        ],
        expand=1,
    )

    page.add(t)
    page.on_close = quit_program
    page.window.height = 800
    page.window.width = 1200
    page.update()

    while WINDOW_OPEN:
        new_updates = [fn for fn in sorted(glob(FOLDER + "/updates/*.txt")) if fn not in update_paths]
        new_images = [fn for fn in sorted(glob(FOLDER + "/jpg/*.jpg")) if fn not in image_paths]

        if last_update_time is None and len(update_paths) > 0 and len(image_paths) > 0:
            cboard.update(update_paths[-1])
            last_update_time = os.path.getmtime(update_paths[-1])
            start_time = os.path.getmtime(update_paths[0])
            live_image.src = image_paths[-1]
            live_image.update()
        elif last_update_time is None:
            last_update_text.value = 'Last update: --'
            last_update_text.update()
            last_image_text.value = 'Last update: --'
            last_image_text.update()
            run_time_text.value = 'Run time     --:--:--'
            run_time_text.update()
            live_image.src = ''
            live_image.update()
            cboard.reset(None)

        if len(new_updates) > 0:
            cboard.update(new_updates[-1])
            last_update_time = os.path.getmtime(new_updates[-1])

            # Update status boxes
            for k in params:
                for i in range(len(cboard.rel_error_history[k])):
                    err = cboard.rel_error_history[k][i]
                    status_boxes[k][i].visible = True
                    status_boxes[k][i].bgcolor = ft.colors.RED_300 if abs(err) >= 2 else ft.colors.AMBER_300 if abs(
                        err) >= 1 else ft.colors.GREEN_300
                    status_boxes[k][i].update()
            update_paths += new_updates

        if len(new_images) > 0:
            show_contours = live_image.src == '_temp_dashboard_image.jpg'
            live_image.src = new_images[-1]
            image_paths += new_images
            if show_contours:
                toggle_contour(None)
            live_image.update()

        if last_update_time is not None and start_time is not None:
            dt = time.time() - last_update_time
            if dt > 3600:
                t_str = '{:.0f} hours'.format(dt // 3600)
            elif dt > 60:
                t_str = '{:.0f} minutes'.format(dt // 60)
            else:
                t_str = '{:.0f} seconds'.format(dt)
            last_update_text.value = 'Last update: {:s} ago'.format(t_str)
            last_update_text.update()
            last_image_text.value = 'Last update: {:s} ago'.format(t_str)
            last_image_text.update()

            rt = int(time.time() - start_time)
            run_time_text.value = 'Run time     {:02d}:{:02d}:{:02d}'.format(rt//3600, (rt//60) % 60, rt % 60)
            run_time_text.update()

        time.sleep(.1)


class ControlBoard:
    def __init__(self):
        self.parameters = ['position', 'width', 'height', 'curvature', 'blur', 'sensitivity', 'contrast', 'flipped', 'transposed']
        self.current_values = {k: np.nan for k in self.parameters}
        self.display_values = {k: np.nan for k in self.parameters}
        self.display_thresholds = {k: np.nan for k in self.parameters}
        self.thresholds = {k: np.nan for k in self.parameters}
        self.display_targets = {k: np.nan for k in self.parameters}
        self.targets = {k: np.nan for k in self.parameters}
        self.errors = {k: np.nan for k in self.parameters}
        self.is_changed = {k: False for k in self.parameters}
        self.is_threshold_changed = {k: False for k in self.parameters}
        self.is_target_changed = {k: False for k in self.parameters}
        self.buttons = self.init_buttons()
        self.texts = self.init_texts()
        self.BUFFER_SIZE = 10  # number of values to remember
        self.rel_error_history = {k: [] for k in self.parameters}

    def init_buttons(self):
        minus_button_style = {'icon': ft.icons.REMOVE}
        plus_button_style = {'icon': ft.icons.ADD}

        buttons = {'position_minus': ft.TextButton(on_click=self.decrease_position,
                                                   on_long_press=self.decrease_position_by_10, **minus_button_style),
                   'position_plus': ft.TextButton(on_click=self.increase_position,
                                                  on_long_press=self.increase_position_by_10, **plus_button_style),
                   'width_minus': ft.TextButton(on_click=self.decrease_width,
                                                on_long_press=self.decrease_width_by_10, **minus_button_style),
                   'width_plus': ft.TextButton(on_click=self.increase_width,
                                               on_long_press=self.increase_width_by_10, **plus_button_style),
                   'height_minus': ft.TextButton(on_click=self.decrease_height,
                                                 on_long_press=self.decrease_height_by_10, **minus_button_style),
                   'height_plus': ft.TextButton(on_click=self.increase_height,
                                                on_long_press=self.increase_height_by_10, **plus_button_style),
                   'curvature_minus': ft.TextButton(on_click=self.decrease_curvature,
                                                    on_long_press=self.decrease_curvature_by_10, **minus_button_style),
                   'curvature_plus': ft.TextButton(on_click=self.increase_curvature,
                                                   on_long_press=self.increase_curvature_by_10, **plus_button_style),
                   'position_thresh_minus': ft.TextButton(on_click=self.decrease_position_thresh,
                                                   on_long_press=self.decrease_position_thresh_by_10, **minus_button_style),
                   'position_thresh_plus': ft.TextButton(on_click=self.increase_position_thresh,
                                                  on_long_press=self.increase_position_thresh_by_10, **plus_button_style),
                   'width_thresh_minus': ft.TextButton(on_click=self.decrease_width_thresh,
                                                on_long_press=self.decrease_width_thresh_by_10, **minus_button_style),
                   'width_thresh_plus': ft.TextButton(on_click=self.increase_width_thresh,
                                               on_long_press=self.increase_width_thresh_by_10, **plus_button_style),
                   'height_thresh_minus': ft.TextButton(on_click=self.decrease_height_thresh,
                                                 on_long_press=self.decrease_height_thresh_by_10, **minus_button_style),
                   'height_thresh_plus': ft.TextButton(on_click=self.increase_height_thresh,
                                                on_long_press=self.increase_height_thresh_by_10, **plus_button_style),
                   'curvature_thresh_minus': ft.TextButton(on_click=self.decrease_curvature_thresh,
                                                    on_long_press=self.decrease_curvature_thresh_by_10, **minus_button_style),
                   'curvature_thresh_plus': ft.TextButton(on_click=self.increase_curvature_thresh,
                                                   on_long_press=self.increase_curvature_thresh_by_10, **plus_button_style),
                   'position_target_minus': ft.TextButton(on_click=self.decrease_position_target,
                                                          on_long_press=self.decrease_position_target_by_10,
                                                          **minus_button_style),
                   'position_target_plus': ft.TextButton(on_click=self.increase_position_target,
                                                         on_long_press=self.increase_position_target_by_10,
                                                         **plus_button_style),
                   'width_target_minus': ft.TextButton(on_click=self.decrease_width_target,
                                                       on_long_press=self.decrease_width_target_by_10,
                                                       **minus_button_style),
                   'width_target_plus': ft.TextButton(on_click=self.increase_width_target,
                                                      on_long_press=self.increase_width_target_by_10,
                                                      **plus_button_style),
                   'height_target_minus': ft.TextButton(on_click=self.decrease_height_target,
                                                        on_long_press=self.decrease_height_target_by_10,
                                                        **minus_button_style),
                   'height_target_plus': ft.TextButton(on_click=self.increase_height_target,
                                                       on_long_press=self.increase_height_target_by_10,
                                                       **plus_button_style),
                   'curvature_target_minus': ft.TextButton(on_click=self.decrease_curvature_target,
                                                           on_long_press=self.decrease_curvature_target_by_10,
                                                           **minus_button_style),
                   'curvature_target_plus': ft.TextButton(on_click=self.increase_curvature_target,
                                                          on_long_press=self.increase_curvature_target_by_10,
                                                          **plus_button_style),
                   }
        return buttons

    def init_texts(self):
        suffixes = ['_current', '_abs_error', '_rel_error', '_threshold', '_target']
        texts = {k+s: ft.Text('-') for k in self.parameters for s in suffixes}
        return texts

    def update(self, update_file):
        data = [line.replace('\n', '').split(': ') for line in open(update_file, 'r').readlines()]
        data = [val for val in data if len(val) == 2]
        dct = {key: val for key, val in data}
        err_dct = {key: str_to_tuple(val) for key, val in dct.items() if 'err' in key}
        par_dct = {key: val_from_text(val) for key, val in dct.items() if 'err' not in key}

        for k in par_dct:
            k2 = 'position' if k == 'center' else k
            if not self.is_changed[k2] or self.display_values[k2] == self.current_values[k2]:
                self.display_values[k2] = par_dct[k]
            self.current_values[k2] = par_dct[k]

        for k in err_dct:
            k2 = {'err_x': 'position', 'err_w': 'width', 'err_h': 'height', 'err_k': 'curvature'}[k]
            self.errors[k2] = err_dct[k][0]
            self.thresholds[k2] = err_dct[k][1]
            self.targets[k2] = err_dct[k][1]
            if not self.is_threshold_changed[k2] or self.display_thresholds[k2] == self.thresholds[k2]:
                self.display_thresholds[k2] = err_dct[k][1]
            if not self.is_target_changed[k2] or self.display_targets[k2] == self.targets[k2]:
                self.display_targets[k2] = err_dct[k][2]
            self.rel_error_history[k2].insert(0, err_dct[k][0]/err_dct[k][1])
            if len(self.rel_error_history[k2]) > self.BUFFER_SIZE:
                self.rel_error_history[k2].pop()

        self.update_texts()

    def update_texts(self):
        suffixes = ['_current', '_abs_error', '_rel_error', '_threshold', '_target']
        for k in self.parameters:
            style = '{:.2f}' if k == 'curvature' else '{:.0f} px'
            self.texts[k + '_current'].value = '-' if np.isnan(self.display_values[k]) else style.format(self.display_values[k])
            self.texts[k + '_abs_error'].value = '-' if np.isnan(self.errors[k]) else style.format(self.errors[k])
            self.texts[k + '_rel_error'].value = '-' if np.isnan(self.errors[k]) else '{:.2f}'.format(np.abs(self.errors[k]/self.thresholds[k]))
            self.texts[k + '_threshold'].value = '-' if np.isnan(self.display_thresholds[k]) else style.format(self.display_thresholds[k])
            self.texts[k + '_target'].value = '-' if np.isnan(self.errors[k]) else style.format(self.current_values[k] - self.errors[k])

            self.texts[k + '_current'].color = ft.colors.BLUE_300 if (self.is_changed[k] and self.current_values[k] != self.display_values[k]) else ft.colors.WHITE
            self.texts[k + '_threshold'].color = ft.colors.BLUE_300 if (
                        self.is_threshold_changed[k] and self.thresholds[k] != self.display_thresholds[k]) else ft.colors.WHITE
            self.texts[k + '_target'].color = ft.colors.BLUE_300 if (
                        self.is_target_changed[k] and self.targets[k] != self.display_targets[k]) else ft.colors.WHITE
            error_color = ft.colors.RED_300 if (np.abs(self.errors[k]) > 2 * self.thresholds[k]) else (ft.colors.AMBER_300 if (np.abs(self.errors[k]) > self.thresholds[k]) else ft.colors.GREEN_300)
            self.texts[k + '_abs_error'].color = error_color
            self.texts[k + '_rel_error'].color = error_color

            for s in suffixes:
                try:
                    self.texts[k + s].update()
                except AssertionError:
                    pass

    def increase_position(self, e):
        self.display_values['position'] += 1
        self.is_changed['position'] = True
        self.update_texts()

    def increase_position_by_10(self, e):
        self.display_values['position'] += 10
        self.is_changed['position'] = True
        self.update_texts()

    def decrease_position(self, e):
        self.display_values['position'] -= 1
        self.is_changed['position'] = True
        self.update_texts()

    def decrease_position_by_10(self, e):
        self.display_values['position'] -= 10
        self.is_changed['position'] = True
        self.update_texts()

    def increase_width(self, e):
        self.display_values['width'] += 1
        self.is_changed['width'] = True
        self.update_texts()

    def increase_width_by_10(self, e):
        self.display_values['width'] += 10
        self.is_changed['width'] = True
        self.update_texts()

    def decrease_width(self, e):
        self.display_values['width'] -= 1
        self.is_changed['width'] = True
        self.update_texts()

    def decrease_width_by_10(self, e):
        self.display_values['width'] -= 10
        self.is_changed['width'] = True
        self.update_texts()

    def increase_height(self, e):
        self.display_values['height'] += 1
        self.is_changed['height'] = True
        self.update_texts()

    def increase_height_by_10(self, e):
        self.display_values['height'] += 10
        self.is_changed['height'] = True
        self.update_texts()

    def decrease_height(self, e):
        self.display_values['height'] -= 1
        self.is_changed['height'] = True
        self.update_texts()

    def decrease_height_by_10(self, e):
        self.display_values['height'] -= 10
        self.is_changed['height'] = True
        self.update_texts()

    def increase_curvature(self, e):
        self.display_values['curvature'] += .1
        self.is_changed['curvature'] = True
        self.update_texts()

    def increase_curvature_by_10(self, e):
        self.display_values['curvature'] += 1
        self.is_changed['curvature'] = True
        self.update_texts()

    def decrease_curvature(self, e):
        self.display_values['curvature'] -= .1
        self.is_changed['curvature'] = True
        self.update_texts()

    def decrease_curvature_by_10(self, e):
        self.display_values['curvature'] -= 1
        self.is_changed['curvature'] = True
        self.update_texts()

    def increase_position_thresh(self, e):
        self.display_thresholds['position'] += 1
        self.is_threshold_changed['position'] = True
        self.update_texts()

    def increase_position_thresh_by_10(self, e):
        self.display_thresholds['position'] += 10
        self.is_threshold_changed['position'] = True
        self.update_texts()

    def decrease_position_thresh(self, e):
        self.display_thresholds['position'] -= 1
        self.is_threshold_changed['position'] = True
        self.update_texts()

    def decrease_position_thresh_by_10(self, e):
        self.display_thresholds['position'] -= 10
        self.is_threshold_changed['position'] = True
        self.update_texts()

    def increase_width_thresh(self, e):
        self.display_thresholds['width'] += 1
        self.is_threshold_changed['width'] = True
        self.update_texts()

    def increase_width_thresh_by_10(self, e):
        self.display_thresholds['width'] += 10
        self.is_threshold_changed['width'] = True
        self.update_texts()

    def decrease_width_thresh(self, e):
        self.display_thresholds['width'] -= 1
        self.is_threshold_changed['width'] = True
        self.update_texts()

    def decrease_width_thresh_by_10(self, e):
        self.display_thresholds['width'] -= 10
        self.is_threshold_changed['width'] = True
        self.update_texts()

    def increase_height_thresh(self, e):
        self.display_thresholds['height'] += 1
        self.is_threshold_changed['height'] = True
        self.update_texts()

    def increase_height_thresh_by_10(self, e):
        self.display_thresholds['height'] += 10
        self.is_threshold_changed['height'] = True
        self.update_texts()

    def decrease_height_thresh(self, e):
        self.display_thresholds['height'] -= 1
        self.is_threshold_changed['height'] = True
        self.update_texts()

    def decrease_height_thresh_by_10(self, e):
        self.display_thresholds['height'] -= 10
        self.is_threshold_changed['height'] = True
        self.update_texts()

    def increase_curvature_thresh(self, e):
        self.display_thresholds['curvature'] += .1
        self.is_threshold_changed['curvature'] = True
        self.update_texts()

    def increase_curvature_thresh_by_10(self, e):
        self.display_thresholds['curvature'] += 1
        self.is_threshold_changed['curvature'] = True
        self.update_texts()

    def decrease_curvature_thresh(self, e):
        self.display_thresholds['curvature'] -= .1
        self.is_threshold_changed['curvature'] = True
        self.update_texts()

    def decrease_curvature_thresh_by_10(self, e):
        self.display_thresholds['curvature'] -= 1
        self.is_threshold_changed['curvature'] = True
        self.update_texts()

    def increase_position_target(self, e):
        self.display_targets['position'] += 1
        self.is_target_changed['position'] = True
        self.update_texts()

    def increase_position_target_by_10(self, e):
        self.display_targets['position'] += 10
        self.is_target_changed['position'] = True
        self.update_texts()

    def decrease_position_target(self, e):
        self.display_targets['position'] -= 1
        self.is_target_changed['position'] = True
        self.update_texts()

    def decrease_position_target_by_10(self, e):
        self.display_targets['position'] -= 10
        self.is_target_changed['position'] = True
        self.update_texts()

    def increase_width_target(self, e):
        self.display_targets['width'] += 1
        self.is_target_changed['width'] = True
        self.update_texts()

    def increase_width_target_by_10(self, e):
        self.display_targets['width'] += 10
        self.is_target_changed['width'] = True
        self.update_texts()

    def decrease_width_target(self, e):
        self.display_targets['width'] -= 1
        self.is_target_changed['width'] = True
        self.update_texts()

    def decrease_width_target_by_10(self, e):
        self.display_targets['width'] -= 10
        self.is_target_changed['width'] = True
        self.update_texts()

    def increase_height_target(self, e):
        self.display_targets['height'] += 1
        self.is_target_changed['height'] = True
        self.update_texts()

    def increase_height_target_by_10(self, e):
        self.display_targets['height'] += 10
        self.is_target_changed['height'] = True
        self.update_texts()

    def decrease_height_target(self, e):
        self.display_targets['height'] -= 1
        self.is_target_changed['height'] = True
        self.update_texts()

    def decrease_height_target_by_10(self, e):
        self.display_targets['height'] -= 10
        self.is_target_changed['height'] = True
        self.update_texts()

    def increase_curvature_target(self, e):
        self.display_targets['curvature'] += .1
        self.is_target_changed['curvature'] = True
        self.update_texts()

    def increase_curvature_target_by_10(self, e):
        self.display_targets['curvature'] += 1
        self.is_target_changed['curvature'] = True
        self.update_texts()

    def decrease_curvature_target(self, e):
        self.display_targets['curvature'] -= .1
        self.is_target_changed['curvature'] = True
        self.update_texts()

    def decrease_curvature_target_by_10(self, e):
        self.display_targets['curvature'] -= 1
        self.is_target_changed['curvature'] = True
        self.update_texts()

    def reset(self, e):
        for k in self.parameters:
            self.display_values[k] = self.current_values[k]
            self.is_changed[k] = False
            self.display_thresholds[k] = self.thresholds[k]
            self.is_threshold_changed[k] = False
            self.display_targets[k] = self.targets[k]
            self.is_target_changed[k] = False
        self.update_texts()

    def submit(self, folder):
        kmap = {'width': 'w', 'height': 'h', 'position': 'm', 'curvature': 'k', 'blur': 'b', 'sensitivity': 's'}

        n = len(glob(folder + "/commands/*.txt"))
        # command = input("Write command here: ")
        actions = []
        for p in self.parameters:
            if self.is_changed[p]:
                actions.append((kmap[p], self.display_values[p]))
            if self.is_threshold_changed[p]:
                actions.append((kmap[p] + "_threshold", self.display_thresholds[p]))
            if self.is_target_changed[p]:
                actions.append((kmap[p] + "_target", self.display_targets[p]))
        command = "\n".join([str(a) for a in actions])

        with open(folder + "/commands/command_{:04d}.txt".format(n), 'w') as f:
            f.write(command)


def val_from_text(s):
    if s[0] == '(':
        return str_to_tuple(s)
    if s == 'True':
        return True
    if s == 'False':
        return False
    if '.' in s:
        return float(s)
    if any([n in s for n in '0123456789']):
        return int(s)
    return s


def str_to_tuple(s):
    s = s.replace('(', '').replace(')', '')
    vals = s.split(', ')
    vals = [np.nan if v in ['', 'na'] else v for v in vals]
    return tuple(vals)


def remove_inner_contour_points(ice_edges, dy=1):
    ice_bins = [ice_edges[np.abs(ice_edges[:, 1] - j * dy) <= dy / 2, :] for j in range(int(np.max(ice_edges[:, 1]) / dy))]
    left, right = [], []
    xmean = np.mean(ice_edges[:, 0])
    for j, ib in enumerate(ice_bins):
        lb, rb = ib[ib[:, 0] < xmean], ib[ib[:, 0] > xmean]
        if len(lb) > 0:
            left.append([np.min(lb[:, 0]), j*dy])
        if len(rb) > 0:
            right.insert(0, [np.max(rb[:, 0]), j*dy])
    return np.array(left + right)


ft.app(main)


