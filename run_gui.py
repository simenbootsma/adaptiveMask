import flet as ft
from multiprocessing import Process
import run_semi_mask
import move_mouse
import start_cam_capture


def fibonacci():
    while True:
        n, m = 1., 1.
        for _ in range(1000000):
            n_prev = n
            n = n + 1/m
            m = n_prev
            print("\r{:.0f}".format(n), end='')


mouse_process = Process(target=move_mouse.main)
cam_process = Process(target=start_cam_capture.main)
mask_process = Process(target=run_semi_mask.main)
keep_warm_processes = [Process(target=fibonacci) for _ in range(5)]


def main(page: ft.Page):
    def toggle_move_mouse(e):
        global mouse_process
        if move_mouse_button.value:
            mouse_process = Process(target=move_mouse.main)
            mouse_process.start()
        else:
            mouse_process.terminate()

    def start_mask(e):
        global mask_process
        mask_process = Process(target=run_semi_mask.main)
        mask_process.start()
        run_mask_button.bgcolor = ft.Colors.RED_300
        run_mask_button.color = ft.Colors.RED_800
        run_mask_button.text = "Stop mask"
        run_mask_button.on_click = stop_mask
        run_mask_button.update()

    def stop_mask(e):
        mask_process.terminate()
        run_mask_button.bgcolor = ft.Colors.GREEN_300
        run_mask_button.color = ft.Colors.GREEN_800
        run_mask_button.text = "Run mask"
        run_mask_button.on_click = start_mask
        run_mask_button.update()

    def start_cam(e):
        global cam_process
        cam_process = Process(target=start_cam_capture.main)
        cam_process.start()
        start_cam_button.bgcolor = ft.Colors.RED_300
        start_cam_button.color = ft.Colors.RED_800
        start_cam_button.text = "Stop camera"
        start_cam_button.on_click = stop_cam
        start_cam_button.update()

    def stop_cam(e):
        cam_process.terminate()
        start_cam_button.bgcolor = ft.Colors.GREEN_300
        start_cam_button.color = ft.Colors.GREEN_800
        start_cam_button.text = "Start camera"
        start_cam_button.on_click = start_cam
        start_cam_button.update()

    def update_keep_warm(e):
        global keep_warm_processes
        n = int(e.control.value)
        for i in range(n):
            if not keep_warm_processes[i].is_alive():
                keep_warm_processes[i] = Process(target=fibonacci)
                keep_warm_processes[i].start()
        for i in range(n, len(keep_warm_processes)):
            if keep_warm_processes[i].is_alive():
                keep_warm_processes[i].terminate()


    move_mouse_button = ft.Switch(label='Move cursor', on_change=toggle_move_mouse)
    run_mask_button = ft.ElevatedButton('Run mask', bgcolor=ft.Colors.GREEN_300, color=ft.Colors.GREEN_800, on_click=start_mask)
    start_cam_button = ft.ElevatedButton('Start camera', bgcolor=ft.Colors.GREEN_300, color=ft.Colors.GREEN_800, on_click=start_cam)
    keep_warm_slider = ft.Slider(0, 'Heat: {value}', 0, len(keep_warm_processes), divisions=len(keep_warm_processes), on_change=update_keep_warm)

    page.add(ft.SafeArea(ft.Column([move_mouse_button, keep_warm_slider, run_mask_button, start_cam_button], spacing=20)))
    page.window.width = 200
    page.window.height = 200
    page.update()


if __name__ == '__main__':
    ft.app(main)

