import shutil
from glob import glob
import time
import sys
import select


FOLDER = "/Users/simenbootsma/OneDrive - University of Twente/VC_coldroom/ColdVC_20241127/"  # must contain jpg, updates, commands folders
update_paths = []#glob(FOLDER + "updates/*.txt")


def main():
    while True:
        read_update()
        time.sleep(0.01)


def write_command():
    N = len(glob(FOLDER + "commands/*.txt"))
    command = input("Write command here: ")
    i, o, e = select.select([sys.stdin], [], [], 10)
    if i:
        with open(FOLDER + "commands/command_{:04d}.txt".format(N), 'w') as f:
            f.write(command)


def read_update():
    global update_paths
    updates = sorted(glob(FOLDER + 'updates/*.txt'))
    if len(updates) > 0 and updates[-1] not in update_paths:
        data = [line[:-1].split(': ') for line in open(updates[-1], 'r').readlines()]
        data = [val for val in data if len(val) == 2]
        dct = {key: val for key, val in data}
        err_dct = {key: str_to_tuple(val) for key, val in dct.items() if 'err' in key}
        relative_errors = [err_dct[key][1] for key in err_dct]
        err_str = [" ok " if abs(rel_err) <= 1 else "{:.02f}".format(rel_err) for rel_err in relative_errors]

        # add colours
        for i in range(len(err_str)):
            if err_str[i] == ' ok ':
                err_str[i] = '\033[32m ok \033[0m'
            elif abs(float(err_str[i])) <= 2:
                err_str[i] = "\033[33m" + err_str[i] + "\033[0m"
            else:
                err_str[i] = "\033[31m" + err_str[i] + "\033[0m"
        count_str = "[Update {:d}]".format(len(updates))
        print("\r" + count_str + " Errors  |  x: {:s}  | w: {:s}  | h: {:s}  | k: {:s} ".format(*err_str), end='')

        update_paths = updates


def str_to_tuple(s):
    v1, v2 = s.split(', ')
    return float(v1[1:]), float(v2[:-1])


if __name__ == '__main__':
    main()


