from glob import glob


FOLDER = "/Users/simenbootsma/OneDrive - University of Twente/VC_coldroom/ColdVC_20241127/"  # must contain jpg, updates, commands folders

N = len(glob(FOLDER + "commands/*.txt"))
command = input("Write command here: ")

with open(FOLDER + "commands/command_{:04d}.txt".format(N), 'w') as f:
    f.write(command)

