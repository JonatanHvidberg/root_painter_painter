import json
from skimage.io import imread
import numpy as np
import os
from os.path import exists
import pandas as pd
'''
inex using cration
dataset
imige name
pix sum
pix sek
pix error
'''



projects = '/home/jonatan/'+'drive_rp_sync/projects/rg_2017_ags/'
segmentations = projects + 'segmentations/'

val = projects + 'models_models/labels/val'
train = projects + '/models_models/labels/train'

with open(projects+'rg_2017_ags.seg_proj') as user_file:
    file_contents = user_file.read()

parsed_json = json.loads(file_contents)

#print(parsed_json['file_names'])
file_names = []
dataset = []
pixel_sum = []
pixel_segmentations_sum = []
pixel_error_sum=[]


for fname in parsed_json['file_names']:
    fname=os.path.splitext(fname)[0] + '.png'
    if (exists(segmentations+fname)):
        file_names.append(fname)
        seg_img=imread(segmentations+fname)
        pixel_sum.append(seg_img.shape[0]*seg_img.shape[1])
        pixel_segmentations_sum.append(np.sum(seg_img[:,:,1]))

        if (exists(val+fname)):
            dataset.append('val')
            seg_img=imread(val+fname)

        elif (exists(train+fname)):
            dataset.append('train')
            seg_img=imread(train+fname)
        else:
            pixel_error_sum.append(0)
            dataset.append('nan')

dict = {'file_names':file_names,'dataset':dataset,'pixel_sum':pixel_sum,'pixel_segmentations_sum':pixel_segmentations_sum,'pixel_error_sum':pixel_error_sum}

df = pd.DataFrame(dict)

df.to_csv(projects+'models_models/befor.csv')