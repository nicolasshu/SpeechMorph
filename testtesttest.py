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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
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
def plot_rec_lengths(speakers=speakers):
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
plot_rec_lengths()

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
test('UW.wav')
test('AA.wav')
test('IY.wav')
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
def extract_features(audio,numcep=25):
  	result = sf.mfcc(audio,numcep=numcep)
    # ----------------------------------
    # Rafa, fill in here
    # ----------------------------------





    #-----------------------------------
    # ----------------------------------
    return result
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
################################################################################
# GLOBAL VARIABLES
################################################################################
plt.rcParams['figure.dpi'] = 150
# plt.rcParams['figure.figsize'] = 1,1

# Set the folders
speakers = ['awb','bdl','clb','jmk','ksp','rms','slt']
combos = []
for subset in itertools.combinations(stuff, 2):
  	combos.append(subset)


def Main():
	for combo in combos:
    	spk0,spk1 = combo[0],combo[1]
        folderpath0, files0 = get_files(spk0)
		folderpath1, files1 = get_files(spk1)
    	for file0,file1 in zip(files0,files1):
            file0 = os.path.join(folderpath0,file0)
            file1 = os.path.join(folderpath1,file1)
            fs0,audio0 = wavread(file0)
            fs1,audio1 = wavread(file1)
            assert fs0 == fs1

            mcep0_ = extract_features(audio0)
            mcep1_ = extract_features(audio1)

            mcep0,mcep1 = align_features(mcep0_,mcep1_)

            0



if __name__ == '__main__':
  	Main()
























# TRASH


# DO NOT USE ANYTHING BELOW




# ==============================================================================
# TOY PROBLEM
folderpath, files = get_files(speakers[0])
for file in files:
    file = os.path.join(folderpath,file)
    fs,audio = wavread(file)
    break

# ==============================================================================
# TOY PROBLEM 2

folderpath0, files0 = get_files(speakers[0])
folderpath1, files1 = get_files(speakers[1])
for file0,file1 in zip(files0,files1):
    file0 = os.path.join(folderpath0,file0)
    file1 = os.path.join(folderpath1,file1)
    fs0,audio0 = wavread(file0)
    fs1,audio1 = wavread(file1)
    assert fs0 == fs1
    break

mfcc0 = sf.mfcc(audio0,numcep=25)
mfcc1 = sf.mfcc(audio1,numcep=25)
distance, path = fastdtw(mfcc0,mfcc0,dist=euclidean)
assert distance == 0
distance, path = fastdtw(mfcc1,mfcc1,dist=euclidean)
assert distance == 0

distance,path = fastdtw(mfcc0,mfcc1,dist=euclidean)

MFCC0 = []
MFCC1 = []
for step in path:
    i,j = step
    MFCC0.append(mfcc0[i])
    MFCC1.append(mfcc1[j])

MFCC0 = np.array(MFCC0)
MFCC1 = np.array(MFCC1)

plt.subplot(411)
plt.imshow(mfcc0.T)
plt.subplot(412)
plt.imshow(mfcc1.T)
plt.subplot(413)
plt.imshow(MFCC0.T)
plt.subplot(414)
plt.imshow(MFCC1.T)


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(25,75)
        self.fc2 = nn.Linear(75,25)
    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


np.random.seed(100)
a = torch.Tensor(np.random.randint(10,size=(1,25)))
b = torch.Tensor(np.ones((1,25)))
model = Net()
model.zero_grad()
optimizer = optim.SGD(model.parameters(),lr=0.0001)
optimizer.zero_grad()

loss_list = []
output_list = [b.detach().numpy()]
N=100
for k in range(N):
    output = model(a)
    criterion = nn.MSELoss()
    loss = criterion(output,b)
    loss.backward()
    optimizer.step()
    # print(loss)
    loss_list.append(loss)
    output_list.append(output.detach().numpy())
    if k==N-1:
        plt.plot(loss_list);
        plt.figure();
        img = np.squeeze(np.array(output_list))
        plt.imshow(img,cmap='jet')

def mel_cepstral_distortion(target,estimated):
    return 10*np.log(10) * torch.sqrt(2*torch.sum((b-output)**2))

for n in range(N):
    output = model
