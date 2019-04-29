import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import NicksToolbox as NS
#os.chdir('utils')

NS.ActivateLatex()

data60  = pd.read_csv('../results/FC_25_50_50_25_epochsize60.csv')
data200 = pd.read_csv('../results/FC_25_50_50_25_epochsize200.csv')

x = data200['TrainLoss']
y = data200['ValLoss']
X = []; Y = []
for item in x:
    X.append(float(item[1:-1]))
for item in y:
    Y.append(float(item[1:-1]))
data200['TrainLoss'] = X
data200['ValLoss'] = Y

x = data60['TrainLoss']
y = data60['ValLoss']
X = []; Y = []
for item in x:
    X.append(float(item[1:-1]))
for item in y:
    Y.append(float(item[1:-1]))
data60['TrainLoss'] = X
data60['ValLoss'] = Y

plt.figure(dpi=140)
plt.plot(data60['TrainLoss'],'--',label = 'Training Loss',linewidth=0.75)
plt.plot(data60['ValLoss'],'-',label = 'Validation Loss',linewidth=0.75)
plt.legend(fontsize=16)
plt.ylim([0,5500000])
plt.xlabel('Epochs',fontsize=18)
plt.ylabel('Loss',fontsize=18)
NS.SetTicks(14)
plt.tight_layout()
plt.savefig('./epochsize_60.png')

plt.figure(dpi=140)
plt.plot(data200['TrainLoss'],'--',label = 'Training Loss',linewidth=0.75)
plt.plot(data200['ValLoss'],'-',label = 'Validation Loss',linewidth=0.75)
plt.legend(fontsize=16)
plt.ylim([0,7500000])
plt.xlabel('Epochs',fontsize=18)
plt.ylabel('Loss',fontsize=18)
NS.SetTicks(14)
plt.tight_layout()
plt.savefig('./epochsize_200.png')
