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
from scipy.signal import argrelextrema
import scipy.signal as sig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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
    return feat,{'mfcc':mfcc,'lps':lps,'phase':phase},{'f':freq,'t':time,'stft':stft}
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
    return FEAT0,FEAT1
# ==============================================================================
def mel_cepstral_distortion(target,estimated):
    target = torch.Tensor(target)
    estimated = torch.Tensor(estimated)
    loss = 10*torch.log(torch.Tensor([10])) * torch.sqrt(2*torch.sum((target-estimated)**2-1))
    return loss#.pow(2)
# ==============================================================================
def stft_distortion(target,estimated):
    target = torch.Tensor(target)
    estimated = torch.Tensor(estimated)
    loss = (torch.mean((target-estimated)**2))#.pow(2)
    return loss
# ==============================================================================

################################################################################
# USELESS HELPER FUNCTIONS
################################################################################
def plot_rec_lengths(speakers):
    speaker_rec_length = []                                                     # Initialize a speaker list
    for spkr in speakers:                                                       # For every speaker
        folderpath = os.path.join(root,'datasets',spkr,'wav')                   #     Get the folderpath
        files = sorted(os.listdir(folderpath))                                  #     Get all the files
        recordings_length = []                                                  #     Initialize a list to keep ints
        for file in files:                                                      #     For every file
            filepath = os.path.join(folderpath,file)                            #         Compile the file path
            fs, audio = wavread(filepath)                                       #         Read file
            recordings_length.append(len(audio))                                #         Append the length onto list
        speaker_rec_length.append(recordings_length)                            #     Append list onto speaker list

    for spkr,speaker_recs in zip(speakers,speaker_rec_length):                  # For every speaker
        plt.plot(speaker_recs,'o',label=spkr,alpha=0.15,markersize=3)           #     Plot the lengths
    plt.legend()                                                                # Put in the legend
    plt.ylabel('Length of Signal (N)')
    plt.xlabel('File number')
# plot_rec_lengths()
# ==============================================================================
def extract_segments(audio,stride=400):
    N = len(audio)
    T = N/fs
    win_t = 5e-3                        # Set the window duration 5 msec
    win_n = int(win_t/T*N)              # Compute the window length
    num_segs = int((N-win_n)/stride)    # Compute the number of segments
    seg_list = []
    for k in range(num_segs):
        ind_i = k*stride                # Initial index
        ind_f = (k+1)*stride + win_n    # Final index
        seg = audio[ind_i:ind_f]        # Extract the segment
        seg_list.append(seg)
    return seg_list
# ==============================================================================
def extract_features0(audio,function):
    N = len(audio)
    T = N/fs
    win_t = 5e-3                        # Set the window duration 5 msec
    win_n = int(win_t/T*N)              # Compute the window length
    stride = 16000                      # Set the stride length (type=int)
    num_segs = int((N-win_n)/stride)    # Compute the number of segments
    seg_list = []
    for k in range(num_segs):
        ind_i = k*stride                # Initial index
        ind_f = (k+1)*stride + win_n    # Final index
        seg = audio[ind_i:ind_f]        # Extract the segment
        feat_list.append(function(seg))
    return feat_list
# ==============================================================================
def test(filepath):
    fs,audio = wavread(filepath)
    # IPython.display.Audio(filepath)
    import audiolazy as AL
    from audiolazy import Stream
    filt = AL.lpc.covariance(audio,order=10)
    sig = audio.copy()
    s = Stream(audio.copy())
    sig_ = filt(s).take(len(sig))
    N=-600
    plt.subplot(211)
    plt.plot(sig[N:],linewidth=0.5,label='Original signal $s[n]$')
    plt.plot(sig_[N:],linewidth=0.5,label=u'Filtered $\hat{s}[n]$',alpha=0.5)
    plt.title(filepath)
    plt.legend()
    plt.subplot(212)
    plt.plot(sig[N:]-sig_[N:],linewidth=0.5,label=u'$s[n] - \hat{s}[n]$')
    plt.legend()
# test('UW.wav')
# test('AA.wav')
# test('IY.wav')
# ==============================================================================
def get_fundfreq(fs,audio,order=12,mode='covariance'):
    import audiolazy as AL
    from audiolazy import Stream

    modes = {'covariance','autocorrelation'}
    if mode not in modes:
        raise Exception("The value for mode needs to be either 'covariance' or 'autocorrelation'")
    if mode == 'covariance':
        filt_func = AL.lpc.covariance
    if mode == 'autocorrelation':
        filt_func = AL.lpc.autocorrelation

    inv_filt = filt_func(audio,order)
    sig = audio.copy()
    sig_= inv_filt(Stream(sig)).take(len(sig))
    sub = sig - sig_
    autocorr = np.correlate(sub,sub,'same')
    autocorr = autocorr[int(len(autocorr)/2):]
    ind_locmax = np.squeeze(argrelextrema(autocorr,np.greater))
    locmax = autocorr[ind_locmax]
    pitch_period = ind_locmax[np.argmax(locmax)]
    fundamental_freq = fs/pitch_period

    return fundamental_freq
# ==============================================================================
def test2():
    fs,audio = wavread('AA.wav')
    sig,sig_ = get_fundfreq(fs,audio)
    N = -1000
    # N = -len(sig)
    sub = sig[N:]-sig_[N:]
    autocorr = np.correlate(sub,sub,'same')
    autocorr = autocorr[int(len(autocorr)/2):]
    ind_locmax = np.squeeze(argrelextrema(autocorr,np.greater))
    locmax = autocorr[ind_locmax]

    # ind = argrelextrema(yo,np.greater)
    # fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    # ax1.plot(autocorr)
    # ax1.plot(ind_locmax,autocorr[ind_locmax],'rx')
    # dautocorr = np.diff(autocorr)
    # ax2.plot(dautocorr)
    # ax2.plot(ind_locmax,dautocorr[ind_locmax],'rx')
    # d2autocorr = np.diff(np.diff(autocorr))
    # ax3.plot(d2autocorr)
    # ax3.plot(ind_locmax,d2autocorr[ind_locmax],'rx')

    return ind_locmax, locmax
# ==============================================================================
def get_output_components(output,num_feat_stft,num_feat_mfcc):
    out_stft = output[:num_feat_stft]
    out_mfcc = output[-num_feat_mfcc:]
    return out_stft,out_mfcc
# ==============================================================================
# ==============================================================================
# ==============================================================================

def Main():
    # for combo in combos:                                                            # For each combination
    #     spk0,spk1 = combo[0],combo[1]                                               #     Set the speakers
    #     folderpath0, files0 = get_files(spk0)                                       #     Obtain the source locations (i.e. files0)
    #     folderpath1, files1 = get_files(spk1)                                       #     Obtain the source locations (i.e. files1)
    #
    #     # Prepare the Neural Network
    #     model = CustomNet1(); model.zero_grad()                                     #     Create the model, and set the gradients to zero
    #     optimizer = optim.SGD(model.parameters(),lr=0.0001); optimizer.zero_grad()  #     Create an optimizer and set the grads to zero
    #
    #     for file_n, file0,file1 in zip(np.arange(len(files0)),files0,files1):       #     For each set of files
    #         # if file_n == 0 or file_n==1: continue
    #         print('File %d'%file_n)
    #         file0 = os.path.join(folderpath0,file0)                                 #          Set the file0
    #         file1 = os.path.join(folderpath1,file1)                                 #          Set the file1
    #         fs0,audio0 = wavread(file0)                                             #          Obtain the audio0
    #         fs1,audio1 = wavread(file1)                                             #          Obtain the audio1
    #         assert fs0 == fs1                                                       #          Make sure that the sampling freqs are the same
    #
    #         feat0_,comps0,stft0 = extract_features(audio0,fs=fs0)                   #          Extract the features for audio0
    #         feat1_,comps1,stft1 = extract_features(audio1,fs=fs1)                   #          Extract the features for audio1
    #
    #         feat0,feat1 = align_features(feat0_.T,feat1_.T)                         #          Align the STFT+MFCC features
    #         mfcc0,mfcc1 = align_features(comps0['mfcc'],comps1['mfcc'])             #          Align the MFCC features
    #
    #         num_feat_stft = feat0.shape[1]-mfcc0.shape[1]                           #          Obtain the number of DFT coefficients
    #         num_feat_mfcc = mfcc0.shape[1]                                          #          Obtain the number of MFC coefficients
    #
    #         # plt.figure(); plt.subplot(4,1,1); plt.pcolormesh(np.log(feat0_)); plt.subplot(412); plt.pcolormesh(comps0['mfcc'].T);
    #         # plt.subplot(413); plt.pcolormesh(np.log(feat1_)); plt.subplot(414); plt.pcolormesh(comps1['mfcc'].T); plt.tight_layout(); plt.savefig('./non-aligned.png')
    #
    #         # plt.figure(); plt.subplot(411); plt.pcolormesh(np.log(feat0.T)); plt.subplot(412); plt.pcolormesh(mfcc0.T);
    #         # plt.subplot(413); plt.pcolormesh(np.log(feat1.T)); plt.subplot(414); plt.pcolormesh(mfcc1.T); plt.tight_layout(); plt.savefig('./aligned.png')
    #
    #         L,F = feat0.shape                                                       #          L windows and F features
    #
    #         loss_list = []                                                          #          Set the loss_list
    #         for l in range(L):                                                      #          For each window segment on audio
    #             seg = feat0[l]                                                      #              Get the segment
    #             output = model(seg)                                                 #              Pass the segment through the model
    #
    #             target_stft = torch.Tensor(feat1[l][:num_feat_stft])                #              Get the target STFT
    #             MCD = mel_cepstral_distortion(mfcc1,output_mfcc)
    #             STFT_D = stft_distortion(output_stft,target_stft)
    #             loss = MCD + STFT_D                                                 #              Compute the loss
    #             loss.backward()                                                     #              Set the Backpropagation
    #             optimizer.step()                                                    #              Take a step
    #             loss_list.append(loss)                                              #              Append the loss to a loss list
    #
    #
    #
    #         if file_n == 10: break
    #     break
    # plt.plot(loss_list)
    # break

################################################################################
# MAIN
################################################################################
class CustomNet1(nn.Module):                                                    # Define the Neural Network
    def __init__(self):
        super(CustomNet1,self).__init__()
        self.fc1 = nn.Linear(226,452)
        self.fc2 = nn.Linear(452,226)
    def forward(self,x):
        x = torch.Tensor(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
if __name__ == '__main__':
    Main()






####  delete later
# for combo in combos:
#     spk0,spk1 = combo[0],combo[1]
#     folderpath0, files0 = get_files(spk0)
#     folderpath1, files1 = get_files(spk1)
#
#     source_train, source_test = train_test_split(files0, test_size=0.40)
#
#
#     for file0,file1 in zip(files0,files1):
#         file0 = os.path.join(folderpath0,file0)
#         file1 = os.path.join(folderpath1,file1)
#         fs0,audio0 = wavread(file0)
#         fs1,audio1 = wavread(file1)
#         assert fs0 == fs1
#
#         feat0_ = extract_features(audio0)
#         feat1_ = extract_features(audio1)
#
#         feat0,feat1 = align_features(feat0_,feat1_)
#         break
#
#     break





################################################################################
# NICK EXPERIMENTING WITH SHIT
################################################################################

class CustomNet1(nn.Module):                                                    # Define the Neural Network
    def __init__(self):
        super(CustomNet1,self).__init__()
        self.fc1 = nn.Linear(226,452)
        self.fc2 = nn.Linear(452,226)
    def forward(self,x):
        x = torch.Tensor(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
for combo in combos:                                                            # For each combination
    print(combo)
    break

spk0,spk1 = combo[0],combo[1]                                               #     Set the speakers
folderpath0, files0 = get_files(spk0)                                       #     Obtain the source locations (i.e. files0)
folderpath1, files1 = get_files(spk1)                                       #     Obtain the source locations (i.e. files1)

# Prepare the Neural Network
model = CustomNet1(); model.zero_grad()                                     #     Create the model, and set the gradients to zero
optimizer = optim.SGD(model.parameters(),lr=0.0001); optimizer.zero_grad()  #     Create an optimizer and set the grads to zero

N_epochs = 4
epoch_train_loss = []
for epoch_num in range(N_epochs):
    print('Epoch [%d/%d]' % (epoch_num+1,N_epochs) , end=' ')

    epoch_loss_list = []                                                              #          Set the loss_list
    for file_n, file0,file1 in zip(np.arange(len(files0)),files0,files1):       #     For each set of files
        # if file_n == 0 or file_n==1: continue
        # print('File %d'%file_n)
        if file_n == 5: break
        print(".",end='')
        file0 = os.path.join(folderpath0,file0)                                 #          Set the file0
        file1 = os.path.join(folderpath1,file1)                                 #          Set the file1
        fs0,audio0 = wavread(file0)                                             #          Obtain the audio0
        fs1,audio1 = wavread(file1)                                             #          Obtain the audio1
        assert fs0 == fs1                                                       #          Make sure that the sampling freqs are the same

        feat0_,comps0,stft0 = extract_features(audio0,fs=fs0)                   #          Extract the features for audio0
        feat1_,comps1,stft1 = extract_features(audio1,fs=fs1)                   #          Extract the features for audio1

        feat0,feat1 = align_features(feat0_.T,feat1_.T)                         #          Align the STFT+MFCC features
        mfcc0,mfcc1 = align_features(comps0['mfcc'],comps1['mfcc'])             #          Align the MFCC features

        num_feat_stft = feat0.shape[1]-mfcc0.shape[1]                           #          Obtain the number of DFT coefficients
        num_feat_mfcc = mfcc0.shape[1]                                          #          Obtain the number of MFC coefficients
        # plt.figure(); plt.subplot(4,1,1); plt.pcolormesh(np.log(feat0_)); plt.subplot(4,1,2); plt.pcolormesh(comps0['mfcc'].T); plt.subplot(4,1,3); plt.pcolormesh(np.log(feat1_)); plt.subplot(4,1,4); plt.pcolormesh(comps1['mfcc'].T); plt.tight_layout(); plt.savefig('./non-aligned.png'); plt.figure(); plt.subplot(4,1,1); plt.pcolormesh(np.log(feat0.T)); plt.subplot(4,1,2); plt.pcolormesh(mfcc0.T); plt.subplot(4,1,3); plt.pcolormesh(np.log(feat1.T)); plt.subplot(4,1,4); plt.pcolormesh(mfcc1.T); plt.tight_layout(); plt.savefig('./aligned.png')
        L,F = feat0.shape                                                       #          L windows and F features

        for l in range(L):                                                      #          For each window segment on audio
            seg = feat0[l]                                                      #              Get the segment
            output = model(seg)                                                 #              Pass the segment through the model
            output_stft, output_mfcc = get_output_components(output,            #              Get the output STFT and MFCC
                                                        num_feat_stft,
                                                        num_feat_mfcc)
            target_stft = torch.Tensor(feat1[l][:num_feat_stft])                #              Get the target STFT
            MCD = mel_cepstral_distortion(mfcc1,output_mfcc)
            STFT_D = stft_distortion(output_stft,target_stft)
            loss = MCD + STFT_D                                                 #              Compute the hybrid loss
            epoch_loss_list.append(loss)
            loss.backward()                                                     #              Set the Backpropagation
            optimizer.step()                                                    #              Take a step

    epoch_loss = sum(epoch_loss_list).data.numpy()
    epoch_train_loss.append(epoch_loss)
    print('')
plt.plot(epoch_train_loss)
# target_stft
# output_stft
###############################################################################












# TRASH


# DO NOT USE ANYTHING BELOW


#
#
# # ==============================================================================
# # TOY PROBLEM
# folderpath, files = get_files(speakers[0])
# for file in files:
#     file = os.path.join(folderpath,file)
#     fs,audio = wavread(file)
#     break
#
# # ==============================================================================
# # TOY PROBLEM 2
#
# folderpath0, files0 = get_files(speakers[0])
# folderpath1, files1 = get_files(speakers[1])
# for file0,file1 in zip(files0,files1):
#     file0 = os.path.join(folderpath0,file0)
#     file1 = os.path.join(folderpath1,file1)
#     fs0,audio0 = wavread(file0)
#     fs1,audio1 = wavread(file1)
#     assert fs0 == fs1
#     break
#
# mfcc0 = sf.mfcc(audio0,numcep=25)
# mfcc1 = sf.mfcc(audio1,numcep=25)
# distance, path = fastdtw(mfcc0,mfcc0,dist=euclidean)
# assert distance == 0
# distance, path = fastdtw(mfcc1,mfcc1,dist=euclidean)
# assert distance == 0
#
# distance,path = fastdtw(mfcc0,mfcc1,dist=euclidean)
#
# MFCC0 = []
# MFCC1 = []
# for step in path:
#     i,j = step
#     MFCC0.append(mfcc0[i])
#     MFCC1.append(mfcc1[j])
#
# MFCC0 = np.array(MFCC0)
# MFCC1 = np.array(MFCC1)
#
# plt.subplot(411)
# plt.imshow(mfcc0.T)
# plt.subplot(412)
# plt.imshow(mfcc1.T)
# plt.subplot(413)
# plt.imshow(MFCC0.T)
# plt.subplot(414)
# plt.imshow(MFCC1.T)
