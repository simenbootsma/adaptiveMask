from glob import glob


FOLDER = "C:/Users/Simen/OneDrive - University of Twente/VC_coldroom/ColdVC_20241129/"  # must contain jpg, updates, commands folders

N = len(glob(FOLDER + "commands/*.txt"))
command = input("Write command here: ")

with open(FOLDER + "commands/command_{:04d}.txt".format(N), 'w') as f:
    f.write(command)

