import os

files = os.listdir('./')

for file in files:
    if 'neat-checkpoint' in file:
        os.rename(file, './checkpoints/' + file)
