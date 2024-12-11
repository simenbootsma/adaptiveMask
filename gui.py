import flet as ft
import os
import numpy as np
import time
from glob import glob
import matplotlib.pyplot as plt


# TODO:
#   - Implement writing with GUI
#   - Implement changing thresholds
#   - Live images
#   - Graph with parameters

FOLDER = "C:/Users/Simen/OneDrive - University of Twente/VC_coldroom/ColdVC_20241211/"  # must contain jpg, updates, commands folders
update_paths = glob(FOLDER + "updates/*.txt")
WINDOW_OPEN = True


def main(page: ft.Page):
    global update_paths, WINDOW_OPEN
    # FUNCTIONALITY

    def quit_program():
        global WINDOW_OPEN
        WINDOW_OPEN = False

    # Dashboard
    def pick_folder_result(e: ft.FilePickerResultEvent):
        data_folder.value = e.path
        data_folder.update()

    data_folder = ft.Text(os.getcwd())
    data_folder_picker = ft.FilePicker(on_result=pick_folder_result)
    page.overlay.append(data_folder_picker)

    # Controls
    cboard = ControlBoard()


    # LAYOUT
    row_h = 25  # row height for control tab
    col_w = 100  # column width for control tab
    params = ['position', 'width', 'height', 'curvature']
    last_update_text = ft.Text("Last update: -",)

    plt.imsave('temp_imgs/dashboard_img.jpg', np.zeros((1000, 400, 3), dtype=np.uint8))
    live_image = ft.Image(
        src="temp_imgs/dashboard_img.jpg",
        width=300,
        height=600,
        fit=ft.ImageFit.FIT_HEIGHT,
        repeat=ft.ImageRepeat.NO_REPEAT,
        border_radius=ft.border_radius.all(10),
    )

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
                            ft.TextField(label="Save filename", hint_text='save_name', suffix_text='.txt', width=200)
                        ], spacing=10),
                        ft.Column([last_update_text, live_image]),
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
                            ft.Column([ft.Text("Target", text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD)] + [cboard.texts[k + '_target'] for k in params],
                                      spacing=row_h, width=col_w),
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
                            ft.Row([ft.ElevatedButton(text='Reset', on_click=cboard.reset), ft.ElevatedButton('Save changes', on_click=cboard.submit)]), margin=30
                        )])
                    ), margin=20
                ),
            ),
        ],
        expand=1,
    )

    page.add(t)
    page.window.height = 800
    page.on_close = quit_program
    page.window.height = 600
    page.window.width = 1200
    page.update()

    while WINDOW_OPEN:
        new_files = [fn for fn in glob(FOLDER + "updates/*.txt") if fn not in update_paths]
        if len(new_files) > 0:
            cboard.update(new_files[-1])
            update_paths += new_files
        time.sleep(.1)
    # cboard.update(r'update_example.txt')
    for i in range(12, 100):
        fname = r'/Users/simenbootsma/OneDrive - University of Twente/VC_coldroom/ColdVC_20241128/updates/update_{:04d}.txt'.format(i)
        cboard.update(fname.format(i))
        dt = time.time() - os.path.getmtime(fname)
        if dt > 3600:
            t_str = '{:.0f} hours'.format(dt // 3600)
        elif dt > 60:
            t_str = '{:.0f} minutes'.format(dt // 60)
        else:
            t_str = '{:.0f} seconds'.format(dt)
        last_update_text.value = 'Last update: {:s} ago'.format(t_str)
        last_update_text.update()

        img_name = r'/Users/simenbootsma/OneDrive - University of Twente/VC_coldroom/ColdVC_20241128/jpg/IMG_{:05d}.jpg'.format(i)
        live_image.src = img_name
        live_image.update()

        time.sleep(2)


class ControlBoard:
    def __init__(self):
        self.parameters = ['position', 'width', 'height', 'curvature', 'blur', 'sensitivity', 'contrast', 'flipped', 'transposed']
        self.current_values = {k: np.nan for k in self.parameters}
        self.display_values = {k: np.nan for k in self.parameters}
        self.display_thresholds = {k: np.nan for k in self.parameters}
        self.thresholds = {k: np.nan for k in self.parameters}
        self.errors = {k: np.nan for k in self.parameters}
        self.is_changed = {k: False for k in self.parameters}
        self.is_threshold_changed = {k: False for k in self.parameters}
        self.buttons = self.init_buttons()
        self.texts = self.init_texts()

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
                   }
        return buttons

    def init_texts(self):
        suffixes = ['_current', '_abs_error', '_rel_error', '_threshold', '_target']
        texts = {k+s: ft.Text('-') for k in self.parameters for s in suffixes}
        return texts

    def update(self, update_file):
        data = [line[:-1].split(': ') for line in open(update_file, 'r').readlines()]
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
            self.thresholds[k2] = np.abs(err_dct[k][0]/err_dct[k][1])  # TODO: pass threshold directly
            if not self.is_threshold_changed[k2] or self.display_thresholds[k2] == self.thresholds[k2]:
                self.display_thresholds[k2] = np.abs(err_dct[k][0]/err_dct[k][1])

        self.update_texts()

    def update_texts(self):
        suffixes = ['_current', '_abs_error', '_rel_error', '_threshold', '_target']
        for k in self.parameters:
            self.texts[k + '_current'].value = '-' if np.isnan(self.display_values[k]) else '{:.0f} px'.format(self.display_values[k])
            self.texts[k + '_abs_error'].value = '-' if np.isnan(self.errors[k]) else '{:.0f} px'.format(self.errors[k])
            self.texts[k + '_rel_error'].value = '-' if np.isnan(self.errors[k]) else '{:.2f}'.format(np.abs(self.errors[k]/self.thresholds[k]))
            self.texts[k + '_threshold'].value = '-' if np.isnan(self.display_thresholds[k]) else '{:.0f} px'.format(self.display_thresholds[k])
            self.texts[k + '_target'].value = '-' if np.isnan(self.errors[k]) else '{:.0f} px'.format(self.current_values[k] - self.errors[k])

            self.texts[k + '_current'].color = ft.colors.BLUE_300 if (self.is_changed[k] and self.current_values[k] != self.display_values[k]) else ft.colors.WHITE
            self.texts[k + '_threshold'].color = ft.colors.BLUE_300 if (
                        self.is_threshold_changed[k] and self.thresholds[k] != self.display_thresholds[k]) else ft.colors.WHITE
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

    def reset(self, e):
        for k in self.parameters:
            self.display_values[k] = self.current_values[k]
            self.is_changed[k] = False
            self.display_thresholds[k] = self.thresholds[k]
            self.is_threshold_changed[k] = False
        self.update_texts()

    def submit(self, e):
        pass


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
    v1, v2 = s.split(', ')
    return float(v1[1:]), float(v2[:-1])


ft.app(main)


