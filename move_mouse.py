import mouse
import time


def main():
    move_left = False
    while True:
        if move_left:
            mouse.move(0, 500, duration=1)
            mouse.click()
        else:
            mouse.move(0, 600, duration=1)
            mouse.click()
        move_left = not move_left
        time.sleep(60)


if __name__ == '__main__':
    main()


