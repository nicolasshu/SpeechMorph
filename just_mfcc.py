################################################################################
# IMPORT LIBRARIES
################################################################################
import numpy as np
import os
from scipy.io.wavfile import read as wavread
import IPython
import matplotlib.pyplot as plt
import python_speech_features as sf
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from tqdm import tqdm
import seaborn as sns
import scipy.signal as sig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import pandas as pd
import datetime
import math
import ipdb
from nnmnkwii import metrics


# Extra Libs

from nnmnkwii.datasets import PaddedFileSourceDataset
from nnmnkwii.datasets.cmu_arctic import CMUArcticWavFileDataSource
from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames, delta_features
from nnmnkwii.util import apply_each2d_trim
from nnmnkwii.metrics import melcd
from nnmnkwii.baseline.gmm import MLPG

from os.path import basename, splitext
import sys
import time

import numpy as np
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import pyworld
import pysptk
from pysptk.synthesis import MLSADF, Synthesizer
import librosa
import librosa.display
import IPython
################################################################################
# GLOBAL VARIABLES
################################################################################
plt.rcParams['figure.dpi'] = 150
# plt.rcParams['figure.figsize'] = 1,1
# Set the folders
speakers = ['awb','bdl','clb','jmk','ksp','rms','slt']
combos = []

for subset in itertools.combinations(speakers, 2):
    combos.append(subset)
start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

YN = 0
# YN = int(input('Do you want to see the spectrograms every 100?'))
learning_rate = 0.01
epsilon = 1e-08
w_decay = 0.00
N_epochs = 300
epoch_size = 50

DEBUG = False

files = os.listdir(os.getcwd())
if 'results' not in files:
    os.makedirs('results')

fs = 16000
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)

################################################################################
# HELPER FUNCTIONS
################################################################################
# ==============================================================================
def get_files(speaker):
    root = os.getcwd()
    folderpath = os.path.join(root,'datasets',speaker,'wav')
    files = sorted(os.listdir(folderpath))
    return folderpath, files
# ==============================================================================
def extract_features(audio,fs=16000,numcep=25,winlen=0.025,winstep=0.01):
    mfcc = sf.mfcc(audio,numcep=numcep,samplerate=fs,winlen=winlen,winstep=winstep)
    freq,time,stft = sig.stft(audio,fs=fs,nperseg=winlen*fs,noverlap=(winlen-winstep)*fs)
    lps = np.abs(stft)
    phase = np.angle(stft)
    diff_time = lps.shape[1] - mfcc.T.shape[1]
    mfcc = np.pad(mfcc, [(0, diff_time), (0, 0)], mode='constant')
    # print(mfcc.T.shape)
    # print(lps.shape)
    feat = np.concatenate((mfcc.T,lps))
    # return feat,{'mfcc':mfcc,'lps':lps,'phase':phase},{'f':freq,'t':time,'stft':stft}

    frame_period = 5
    order = 24
    audio = audio.astype(np.float64)
    f0, timeaxis = pyworld.dio(audio, fs, frame_period=frame_period)
    f0 = pyworld.stonemask(audio, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(audio, f0, timeaxis, fs)
    spectrogram = trim_zeros_frames(spectrogram)
    mc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)

    return feat,{'mfcc':mc,'lps':lps,'phase':phase},{'f':freq,'t':time,'stft':stft}
# ==============================================================================
def align_features(feat0,feat1):
    distance, path = fastdtw(feat0,feat0,dist=euclidean)
    assert distance == 0
    distance, path = fastdtw(feat1,feat1,dist=euclidean)
    assert distance == 0

    distance,path = fastdtw(feat0,feat1,dist=euclidean)
    FEAT0 = []
    FEAT1 = []
    for step in path:
        i,j = step
        FEAT0.append(feat0[i])
        FEAT1.append(feat1[j])
    FEAT0 = np.array(FEAT0)
    FEAT1 = np.array(FEAT1)
    # return FEAT0,FEAT1


    X_aligned, Y_aligned = DTWAligner(verbose=0, dist=melcd).transform((feat0, feat1))
    X_aligned, Y_aligned = X_aligned[:, :, 1:], Y_aligned[:, :, 1:]
    return X_aligned, Y_aligned

# ==============================================================================
def mel_cepstral_distortion(target,estimated):
    target = torch.Tensor(target)
    estimated = torch.Tensor(estimated)
    loss = 10*torch.log(torch.Tensor([10])) * torch.sqrt(2*torch.sum((target-estimated)**2))
    # return loss#.pow(2)
    return metrics.melcd(target,estimated)
# ==============================================================================
def stft_distortion(target,estimated):
    target = torch.Tensor(target)
    estimated = torch.Tensor(estimated)
    loss = (torch.mean((target-estimated)**2))#.pow(2)
    return loss
# ==============================================================================
def save_data(model,epoch_train_loss,epoch_val_loss,start_time = start_time):
    save_dict = {"TrainLoss":epoch_train_loss,'ValLoss':epoch_val_loss}
    #ipdb.set_trace()
    save_df = pd.DataFrame.from_dict(save_dict)
    curr_time = start_time
    save_df.to_csv('./results/losslog_'+start_time+'.csv')
    torch.save(model.state_dict(),'./results/model_'+curr_time+'.pt')
# ==============================================================================
def get_output_components(output,num_feat_stft,num_feat_mfcc):
    out_stft = output[:num_feat_stft]
    out_mfcc = output[-num_feat_mfcc:]
    return out_stft,out_mfcc
# ==============================================================================
def HowsMyMemory():
    import datetime, resource
    print("TIME:    "+str(datetime.datetime.now())+' | Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
################################################################################
# USELESS HELPER FUNCTIONS
################################################################################

# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================

################################################################################
# NICK EXPERIMENTING WITH SHIT
################################################################################
class CustomNet7(nn.Module):                                                    # Define the Neural Network
    def __init__(self):
        super(CustomNet7,self).__init__()
        self.description = 'Large FC'
        self.full = nn.Sequential(nn.Linear(25,50),nn.Tanh(),nn.Linear(50,50),nn.Tanh(),nn.Linear(50,25))
    def forward(self,x):
        x = torch.Tensor(x)
        x = self.full(x)
        return x

def OnlyMFCC(model):
    for combo in combos:                                                            # For each combination
        # print(combo)
        break
    combo = ('bdl','rms')                                                       #     Force a Male-to-Male
    print(combo)
    spk0,spk1 = combo[0],combo[1]                                               #     Set the speakers
    folderpath0, files0 = get_files(spk0)                                       #     Obtain the source locations (i.e. files0)
    folderpath1, files1 = get_files(spk1)                                       #     Obtain the source locations (i.e. files1)

    ind = np.arange(len(files0))
    np.random.seed(100)
    np.random.shuffle(ind)
    ind_train, remaining_data = np.split(ind, [int(0.7 * len(ind))])
    ind_test, ind_val = np.split(remaining_data,[int(0.8 * len(remaining_data))])
    ind_train = sorted(ind_train); ind_val   = sorted(ind_val); ind_test  = sorted(ind_test)

    files0_train = [files0[it] for it in ind_train]
    files0_test  = [files0[it] for it in ind_test]
    files0_val   = [files0[it] for it in ind_val]

    files1_train = [files1[it] for it in ind_train]
    files1_test  = [files1[it] for it in ind_test]
    files1_val   = [files1[it] for it in ind_val]


    # Prepare the Neural Network
    model.zero_grad()                                     #     Create the model, and set the gradients to zero
    optimizer = optim.Adam(model.parameters(),lr=learning_rate,eps=epsilon,weight_decay=w_decay); optimizer.zero_grad()  #     Create an optimizer and set the grads to zero
    #optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.3); optimizer.zero_grad()
    epoch_train_loss = []
    epoch_val_loss = []

    for epoch_num in range(N_epochs):

        print('%s | Epoch [%d/%d]' % (model.description,epoch_num+1,N_epochs))
        train_loss_list = []                                                        #          Set the training loss_list
        val_loss_list = []
        # TRAINING PHASE ===========================================================
        # print('TRAINING')
        for file_n, file0,file1 in tqdm(zip(np.arange(len(files0_train)),files0_train,files1_train)):       #     For each set of files
            if file_n == epoch_size: break
            file0 = os.path.join(folderpath0,file0)                                 #          Set the file0
            file1 = os.path.join(folderpath1,file1)                                 #          Set the file1
            fs0,audio0 = wavread(file0)                                             #          Obtain the audio0
            fs1,audio1 = wavread(file1)                                             #          Obtain the audio1
            assert fs0 == fs1                                                       #          Make sure that the sampling freqs are the same

            feat0_,comps0,stft0 = extract_features(audio0,fs=fs0)                   #          Extract the features for audio0
            feat1_,comps1,stft1 = extract_features(audio1,fs=fs1)                   #          Extract the features for audio1

            mfcc0,mfcc1 = align_features(comps0['mfcc'],comps1['mfcc'])             #          Align the MFCC features

            L,F = mfcc0.shape
            batch_loss = 0
            batch_size = L
            for l in range(L):
                seg = mfcc0[l]
                output = model(seg)
                target = torch.Tensor(mfcc1[l])
                loss = mel_cepstral_distortion(target,output)
                # batch_loss += loss
                # if l % batch_size == 0:
                #     train_loss_list.append(batch_loss)
                #     batch_loss.backward()
                #     optimizer.step()
                if DEBUG and l % 100 == 0: print(np.array([output.data.numpy(),target.data.numpy()]).T)
                train_loss_list.append(loss)
                loss.backward()
                optimizer.step()

        train_loss = sum(train_loss_list).data.numpy() / (file_n + 1)

        epoch_train_loss.append(train_loss)

        # VALIDATION PHASE =========================================================
        # print('VALIDATION')
        for file_n, file0,file1 in tqdm(zip(np.arange(len(files0_val)),files0_val,files1_val)):       #     For each set of files
            # if file_n == 3: break
            # print(".",end='')
            file0 = os.path.join(folderpath0,file0)                                 #          Set the file0
            file1 = os.path.join(folderpath1,file1)                                 #          Set the file1
            fs0,audio0 = wavread(file0)                                             #          Obtain the audio0
            fs1,audio1 = wavread(file1)                                             #          Obtain the audio1
            assert fs0 == fs1                                                       #          Make sure that the sampling freqs are the same

            feat0_,comps0,stft0 = extract_features(audio0,fs=fs0)                   #          Extract the features for audio0
            feat1_,comps1,stft1 = extract_features(audio1,fs=fs1)                   #          Extract the features for audio1

            feat0,feat1 = align_features(feat0_.T,feat1_.T)                         #          Align the STFT+MFCC features
            mfcc0,mfcc1 = align_features(comps0['mfcc'],comps1['mfcc'])             #          Align the MFCC features

            L,F = mfcc0.shape                                                       #          L windows and F features
            for l in range(L):                                                      #          For each window segment on audio
                seg = mfcc0[l]                                                      #              Get the segment
                output = model(seg)                                                 #              Pass the segment through the model
                target = torch.Tensor(mfcc1[l])
                val_loss = mel_cepstral_distortion(target,output)
                val_loss_list.append(val_loss)
                if DEBUG and l % 100 == 0: print(np.array([output.data.numpy(),target.data.numpy()]).T)
        val_loss = sum(val_loss_list).data.numpy() / (file_n + 1)
        epoch_val_loss.append(val_loss)

        comparison = np.array([output.data.numpy(),target.data.numpy()]).T
        print('Training Loss: %.2f E6 | Validation Loss: %.2f E6' %(train_loss/1000000,val_loss/1000000))
        print(str(comparison))
        save_data(model,epoch_train_loss,epoch_val_loss)
    plt.plot(epoch_train_loss,label ='Training Loss')
    plt.plot(epoch_val_loss,label='Validation Loss')

################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
    print('Start of Code.')

    my_model = CustomNet7()
    OnlyMFCC(my_model)
    print('End of Code.')

    plt.show()
