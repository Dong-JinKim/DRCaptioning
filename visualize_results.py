import json
import numpy as np
import pdb

import matplotlib.pyplot as plt

data=json.load(open('res/checkpoint_VG_subjobjcoor_MTL_1e6.t7.json'))


# results_history, loss_history, iter

results_list = data['results_history'].keys()
loss_list = data['loss_history'].keys()

results_list2 = [int(float(i)) for i in results_list]
results_idx = sorted(range(len(results_list2)),key=lambda k:results_list2[k])
results_list = [results_list[i] for i in results_idx]

loss_list2 = [int(float(i)) for i in loss_list]
loss_idx = sorted(range(len(loss_list2)),key=lambda k:loss_list2[k])
loss_list = [loss_list[i] for i in loss_idx]

#data['results_history'][results_list[1]]['loss_results']
#data['loss_history'][loss_list[1]]['total_loss']
#data['loss_history'][loss_list[1]]['captioning_loss']


total_loss_train = []
captioning_loss_train = []
total_loss_val = []
captioning_loss_val = []
mAP = []
meteor = []

for iid in results_list:  #----val
  
  total_loss_val.append(data['results_history'][iid]['loss_results']['total_loss'])
  captioning_loss_val.append(data['results_history'][iid]['loss_results']['captioning_loss'])
  mAP.append(data['results_history'][iid]['ap_results']['map']*100)
  if 'meteor' in data['results_history'][iid]['ap_results'].keys():
    meteor.append(data['results_history'][iid]['ap_results']['meteor']*100)
  



for iid in loss_list: #-----train
  if data['loss_history'][iid]['captioning_loss']>0:
    total_loss_train.append(data['loss_history'][iid]['total_loss'])
    captioning_loss_train.append(data['loss_history'][iid]['captioning_loss'])



N = len(captioning_loss_train)
x_train = np.linspace(1,N,N)
N = len(captioning_loss_val)
x_val = np.linspace(1,N,N)

_,axarr = plt.subplots(1,4)
axarr[0].plot(x_train,captioning_loss_train)
axarr[0].set_title('train_loss')
axarr[1].plot(x_val,total_loss_val,x_val,captioning_loss_val)
axarr[1].set_title('val_loss')
axarr[2].plot(x_val,mAP)
axarr[2].set_title('mAP')
if 'meteor' in data['results_history'][iid]['ap_results'].keys():
  axarr[3].plot(x_val,meteor)
axarr[3].set_title('METEOR')
plt.show()

print("minimum loss is total=%.3f / captioning = %.3f"%(min(total_loss_val),min(captioning_loss_val)))
print("max mAP=%.3f"%(max(mAP)))
if 'meteor' in data['results_history'][iid]['ap_results'].keys():
  print("max METEOR=%.3f"%(max(meteor)))






