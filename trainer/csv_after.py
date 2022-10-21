import models_model_lib as mml
import model_utils
from file_utils import ls
import numpy as np
from skimage.io import imread, imsave
import os
import im_utils

import torch
import pandas as pd


def sum_error(o_model_name,n):

    model_dir=syncdir+project+'/models_models/models3'+n
    
    path = model_utils.get_latest_model_paths(model_dir, k=1)[0]
    model = mml.load_model(path)

    omodel = model_utils.load_model(syncdir+project+'/models/' +o_model_name)

    fnames = ls(syncdir+project+'/models_models/labels/test/')

    ys = np.zeros([len(fnames),4])

    coreted_sum=[]
    predicted_sum=[]
    unsertensy_sum=[]
    unsertensy_cl=[]
    totel_pix=[]
    file_names=[]

    for fname in fnames:
        file_names.append(fname)
        image = imread(syncdir+project+'/models_models/labels/test/'+fname)
        coreted_sum.append(np.sum((image[:,:,0]>0).astype(int)))
        totel_pix.append(image.shape[0]*image.shape[1])
        #persent_coreted =coreted_sum/totel_pix
        #print(coreted_sum,totel_pix,persent_coreted)

    
        image = imread(syncdir+project+'/models_models/data/'+fname)
        predicted = mml.unet_segment(model, image, bs, in_w,
                         out_w, threshold=0.5)
        predicted_sum.append(np.sum(predicted))
        #persent_predicted =predicted_sum/totel_pix
        #print(predicted_sum,persent_predicted)

        image = im_utils.load_image(syncdir+datasets+'/'+os.path.splitext(fname)[0] + '.jpg')

        o_predicted = model_utils.unet_segment(omodel, image, bs, in_w,
                 out_w, threshold=None)

        unsertensy=mml.entorpy(o_predicted)
        unsertensy_predicted = unsertensy > 0.5

        unsertensy_predicted = unsertensy_predicted.astype(int)

        unsertensy_cl.append(np.sum(unsertensy))
        unsertensy_sum.append(np.sum(unsertensy_predicted))

    dict = {'file_names':file_names,'coreted_sum':coreted_sum,'predicted_sum':predicted_sum,'unsertensy_sum':unsertensy_sum,'unsertensy_cl': unsertensy_cl,'totel_pix':totel_pix}

    df = pd.DataFrame(dict)
    df.to_csv(syncdir+project+'/models_models/after3'+ n +'.csv')
    print('don')

global in_w
global out_w
global mem_per_item
global total_mem
global bs

in_w = 572
out_w = 500
mem_per_item = 3800000000
total_mem = 0
print('GPU Available', torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    total_mem += torch.cuda.get_device_properties(i).total_memory
bs = total_mem // mem_per_item
bs = min(12, bs)
print('Batch size', bs)

syncdir='drive_rp_sync'

'''
om='000032_1578339309.pkl'
project = '/projects/biopores_a_corrective'
datasets = '/datasets/biopores_750_training'
for x in range(1,6):
    sum_error(om,str(x))

om='000022_1578319359.pkl'
project = '/projects/biopores_b_corrective'
for x in range(1,6):
    sum_error(om,str(x))

om='000031_1581174019.pkl'
project = '/projects/nodules_a_corrective'
datasets = '/datasets/nodules_750_training'
for x in range(1,6):
    sum_error(om,str(x))

om='000023_1581690809.pkl'
project = '/projects/nodules_b_corrective'
for x in range(1,6):
    sum_error(om,str(x))

om='000046_1578155544.pkl'
project = '/projects/towers_a_corrective'
datasets = '/datasets/towers_750_training'
for x in range(1,6):
    sum_error(om,str(x))
'''
datasets = '/datasets/towers_750_training'
om='000032_1578167455.pkl'
project = '/projects/towers_b_corrective'
for x in range(1,6):
    sum_error(om,str(x))