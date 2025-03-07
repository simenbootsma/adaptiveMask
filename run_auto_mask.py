import cv2 as cv
from AutoMask import AutoMask
import time
from glob import glob
import rawpy
import flet as ft
from multiprocessing import Process, Queue

IMG_FOLDER = 'C:/Users/local.la/Documents/Simen/ColdRoom/working_folder/'  # folder where images ares saved


def main():
    ft.app(gui)

    # q = Queue()
    # p1 = Process(target=ft.app, args=(gui, q))
    # p2 = Process(target=show_mask, args=(q,))
    #
    # p1.start()
    # p2.start()
    #
    # p1.join()
    # p2.join()


def gui(page: ft.Page):
    # initialize GUI
    page.add(ft.SafeArea(ft.Column([
        ft.Text('Auto mask controls', size=24),
        ft.Divider(),
        ft.Row([ft.Text('Sensitivity'),
                ft.Slider(50, min=1, max=100, divisions=100, label="{value}%", on_change_end=lambda e: q.put(('sensitivity', e.control.value/100)))],
               width=400, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ft.Row([ft.Text('Mask width (cam pixels)'),
                ft.Slider(100, min=1, max=1000, divisions=1000, label="{value} px", on_change_end=lambda e: q.put(('eta', e.control.value)))],
               width=400, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ft.Row([ft.Text('Number of control points'),
                ft.Slider(200, min=10, max=1000, divisions=100, label="{value}", on_change_end=lambda e: q.put(('ncp', e.control.value)))],
               width=400, alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
    ])))

    page.window.width = 500
    page.window.height = 500
    page.update()

    q = Queue()
    p = Process(target=show_mask, args=(q,))
    p.start()
    p.join()


def show_mask(q: Queue):
    # initialize mask
    mask = AutoMask()
    func_map = {'sensitivity': mask.set_sensitivity, 'eta': mask.set_eta, 'ncp': mask.set_ncp}
    cv_window()
    img_paths = glob(IMG_FOLDER + '*.NEF')

    while True:
        new_images = sorted([fn for fn in glob(IMG_FOLDER + "*.NEF") if fn not in img_paths])
        if len(new_images) > 0:
            time.sleep(.5)
            if 'NEF' in new_images[0]:
                img = rawpy.imread(new_images[0]).postprocess()
            else:
                img = cv.imread(new_images[0])
            img_paths.append(new_images[0])

            mask.update(img)

        # handle key presses
        key = cv.waitKey(10)
        if key == 27:
            break

        # check for commands
        while not q.empty():
            key, val = q.get()
            if key in func_map:
                func_map[key](val)

        # show screen
        cv.imshow("window", mask.get_img())

        print('tick')

        time.sleep(.1)

    cv.destroyWindow("window")


def cv_window():
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    cv.moveWindow("window", 900, 400)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


if __name__ == '__main__':
    main()

