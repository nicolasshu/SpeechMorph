import numpy as np
import os
from scipy.io.wavfile import read as wavread
# import IPython


# Set the folders
speakers = ['awb','bdl','clb','jmk','ksp','rms','slt']
root = os.getcwd()
folderpath = os.path.join(root,'datasets',speakers[0],'wav')
files = sorted(os.listdir(folderpath))


# Read the files
for file in files:
    file = os.path.join(folderpath,file)
    fs,audio = wavread(file)
    break
# IPython.display.Audio(file)
