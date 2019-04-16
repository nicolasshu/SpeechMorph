import numpy as np
import os
from scipy.io.wavfile import read as wavread
import python_speech_features as sf

import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic

import pysptk
import librosa
import pyrenn
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


# YAAPT pitches
signal = basic.SignalObj(file)
pitchY = pYAAPT.yaapt(signal, frame_length=25, frame_space=5, f0_min=40, f0_max=300)

plt.plot(pitchY.values_interp, label='YAAPT', color='blue')
plt.xlabel('samples')
plt.ylabel('pitch (Hz)')'

#
