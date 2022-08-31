

import models_model_lib  as mml
import model_utils
from datasets2 import TrainDataset as TrainDataset2
import im_utils
import torch

#for test
from skimage.io import imread, imsave
from file_utils import ls
import time
import os


import numpy as np


def setop():
    pass

def train_type2(model_path, train_annot_dir, dataset_dir):
    train_set = TrainDataset2(train_annot_dir,dataset_dir,in_w,out_w)

    model = mml.load_model(model_path)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, nesterov=True)

    mml.train_epoch(train_set, model, optimizer)    
    pass

def segment_gradian(model_paths, image, bs, in_w, out_w):
    """ Average predictions from each model specified in model_paths """
    pred_sum = None
    pred_count = 0
    # then add predictions from the previous models to form an ensemble
    for model_path in model_paths:
        cnn = model_utils.load_model(model_path)
        preds = model_utils.unet_segment(cnn, image,
                             bs, in_w, out_w, threshold=None)
        if pred_sum is not None:
            pred_sum += preds
        else:
            pred_sum = preds
        pred_count += 1
        # get flipped version too (test time augmentation)
        flipped_im = np.fliplr(image)
        flipped_pred = model_utils.unet_segment(cnn, flipped_im, bs, in_w,
                                    out_w, threshold=None)
        pred_sum += np.fliplr(flipped_pred)
        pred_count += 1
    foreground_probs = (pred_sum / pred_count)*255
    foreground_probs = foreground_probs.astype(int)
    #predicted = foreground_probs > threshold
    #predicted = predicted.astype(int)
    return foreground_probs

def gradian_data_setop(model_paths,setup_dir):
    
    fnames = ls(setup_dir)
    fnames = [a for a in fnames if im_utils.is_photo(a)]
    for fname in fnames:

        image = im_utils.load_image(syncdir+datasets+'/'+ os.path.splitext(fname)[0] + '.jpg')
        segmented = segment_gradian(model_paths, image, bs, in_w, out_w)

        #seg_alpha = np.zeros((segmented.shape[0], segmented.shape[1], 3))
        #seg_alpha[segmented > 0] = [0, 1.0, 1.0]

        segmented.shape=(segmented.shape[0],segmented.shape[1],1)

        #seg_alpha  = (seg_alpha * 255).astype(np.uint8)

        image = np.concatenate((image,segmented), axis=2)

        im_utils.save_then_move(syncdir+project+'/models_models/data2/'+fname, image)

def test_data():
    image=imread(syncdir+project+'/models_models/data/B85-1_000.png')
    image_RGB = np.array(image[:,:,:3])
    seg_alpha = np.ones((image_RGB.shape[0], image_RGB.shape[1], 1))*255


    image = (np.concatenate((image_RGB,seg_alpha), axis=2))

    imsave(syncdir+project+'/test2-B85-1_000.png',image)
'''
Data
'''
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
'''
'''


syncdir = '/content/drive/MyDrive/drive_rp_sync'
datasets = '/datasets/biopores_750_training'
project = '/projects/biopores_a_corrective'



segmentations = '/segmentations'
val = '/annotations/val'
train = '/annotations/train'

#print(syncdir+datasets+'/B85-1_000.png')
print(syncdir+project+'/models_models/000015_1578333385.pkl')

#test_data()
#gradian_data_setop([syncdir+project+'/models/000015_1578333385.pkl'], syncdir+project+'/segmentations')
#gradian_data_setop([syncdir+project+'/models/000001_1578331363.pkl'])

#train_type2(model_path, train_annot_dir, dataset_dir)
'''
train_type2(syncdir+project+'/models_models/models/000003_1661779227.pkl'
    , syncdir+project+train
    , syncdir+project+'/models_models/data')
'''
train_type2(syncdir+project+'/models_models/models/000001_1661772775.pkl'
    , syncdir+project+train
    , syncdir+project+'/models_models/data2')

#model=load_model(syncdir+project+'/models_models/models/000001_1661772775.pkl')

#validation(model)

#setup(syncdir+project)
#setup_date(syncdir+project)
#create_first_model_with_random_weights(syncdir+project+'/models_models')






#train_type2(model_path, train_annot_dir, dataset_dir)

'''
train_type2(syncdir+project+'/models_models/000001_2_1660065013.pkl',
    syncdir+project+'/models_models/'+val,
    syncdir+project+'/models_models/data')

'''


#def dif_new_ann(imageSegDir, imageAnnDir):
    #shut be models model segrigation /home/jonatan/Documents/diku/BA/testbil/sek/B85-1_000.png
#A=dif_new_ann('/home/jonatan/Documents/diku/BA/testbil/sek/B85-1_000.png', '/home/jonatan/Documents/diku/BA/testbil/tranORval/B85-1_000.png')
#im_utils.save_then_move('/home/jonatan/Documents/diku/BA/testbil/B85-1_000.png', A)



#dif_seg_ann(imageSegDir, imageAnnDir, imageSaveDir)
#dif_seg_ann(syncdir+project+'/models_models/B1-1_000.png', syncdir+project+val+'/B1-1_000.png', syncdir+project+'/models_models/annotations/val/B1-1_000.png')

#image_and_segmentation(imagedir,imageSegDir)
#sed=image_and_segmentation(syncdir+datasets+'/B1-1_000.jpg',syncdir+project+segmentations+'/B1-1_000.png')
#im_utils.save_then_move(syncdir+project+'/models_models/data/B1-1_000.png', sed)


#setup(syncdir+project)

#test_new_model(syncdir+project+'/models_models/B1-1_000.png')


#create_first_model_with_random_weights(syncdir+project+'/models_models')

'''
image_and_segmentation('/home/jonatan/Documents/diku/BA/testbil/org/B85-1_000.jpg' 
    ,'/home/jonatan/Documents/diku/BA/testbil/sek/B85-1_000.png'
    ,'/home/jonatan/Documents/diku/BA/testbil/sek/test.png')
'''


#dif_seg_aaa()

'''
create_first_model_with_random_weights

'''

'''
create_first_model_with_random_weights
rat json

    for in 
'''