

import models_model_lib as mml
import model_utils
from datasets2 import TrainDataset as TrainDataset2
from datasets3 import TrainDataset as TrainDataset3
import im_utils
import torch

#for traning
from torch.nn.functional import softmax
from loss import combined_loss as criterion
from functools import partial
from model_utils import save_if_better
from torch.utils.data import DataLoader
import multiprocessing
import copy

#for test
from skimage.io import imread, imsave
from file_utils import ls
import time
import os




import numpy as np


def setop():
    pass
def DataLoader_type3(model, im_tile, annot_tile):
        '''
        segmented=mml.simbel_segment(model, im_tile)
        segmented.shape=(segmented.shape[0],segmented.shape[1],1)
        '''
        outputs = model(im_tile)

        print('shape')
        print(im_tile.shape)
        print(outputs.shape)
        print(annot_tile.shape)



        assert 1==1
        assert 1!=1

        im_tile = image_and_segmentation(im_tile, segmented)
        annot_tile = new_ann(im_tile ,annot_tile)

        foreground = np.array(annot_tile)[:, :, 0]
        background = np.array(annot_tile)[:, :, 1]

        # Annotion is cropped post augmentation to ensure
        # elastic grid doesn't remove the edges.
        foreground = foreground[tile_pad:-tile_pad, tile_pad:-tile_pad]
        background = background[tile_pad:-tile_pad, tile_pad:-tile_pad]
        # mask specified pixels of annotation which are defined
        mask = foreground + background
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        foreground = foreground.astype(np.int64)
        foreground = torch.from_numpy(foreground)
        im_tile = im_tile.astype(np.float32)
        im_tile = np.moveaxis(im_tile, -1, 0)
        im_tile = torch.from_numpy(im_tile)
        return im_tile, foreground, mask

def train_epoch(train_set,model, optimizer, dataset_dir, type=2, fmodel='nan'):
    
    model.train()


    train_loader = DataLoader(train_set, bs, shuffle=True,
                              # 12 workers is good for performance
                              # on 2 RTX2080 Tis
                              # 0 workers is good for debugging
                              # don't go above 12 workers and don't go above the number of cpus
                              num_workers=min(multiprocessing.cpu_count(), 12),
                              drop_last=False, pin_memory=True)

    num_of_traning_no_better=0

    while num_of_traning_no_better<10:

        for step, (photo_tiles,
               foreground_tiles,
               defined_tiles) in enumerate(train_loader):
            if type == 3:
                #assert fmodel == 'nan', 'foren model not defint'
                photo_tiles,foreground_tiles, defined_tiles =DataLoader_type3(fmodel,photo_tiles,foreground_tiles)


            photo_tiles = photo_tiles.cuda()
            foreground_tiles = foreground_tiles.cuda()
            defined_tiles = defined_tiles.cuda()

            optimizer.zero_grad()

            outputs = model(photo_tiles)
            softmaxed = softmax(outputs, 1)

            foreground_probs = softmaxed[:, 1, :]

            outputs[:, 0] *= defined_tiles
            outputs[:, 1] *= defined_tiles

            loss = criterion(outputs, foreground_tiles)
            loss.backward()
            optimizer.step()


        if validation(model, dataset_dir):
            num_of_traning_no_better = 0
        else:
            num_of_traning_no_better = num_of_traning_no_better+1


def validation(model,dataset_dir):

    get_val_metrics = partial(mml.get_val_metrics,
                          val_annot_dir=syncdir+project+'/models_models'+val,
                          dataset_dir=dataset_dir,
                          in_w=in_w, out_w=out_w, bs=bs)


    model_dir=syncdir+project+'/models_models/models3'
    prev_path = model_utils.get_latest_model_paths(model_dir, k=1)[0]
    prev_model =mml.load_model(prev_path)

    cur_metrics = get_val_metrics(copy.deepcopy(model))
    prev_metrics = get_val_metrics(prev_model)

    print('cur_metrics')
    print(cur_metrics)

    print('prev_metrics')
    print(prev_metrics)

    
    was_saved = save_if_better(model_dir, model, prev_path,
                           cur_metrics['f1'], prev_metrics['f1'])

    return was_saved

def train_type2(model_path, train_annot_dir, dataset_dir):
    train_set = TrainDataset2(train_annot_dir,dataset_dir,in_w,out_w)

    model = mml.load_model(model_path)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, nesterov=True)

    train_epoch(train_set, model, optimizer, dataset_dir)
    pass

def train_type3(model_path, fmodel_path, train_annot_dir, dataset_dir, dataset_dir2):
    
    fmodel = model_utils.load_model(fmodel_path)


    train_set = TrainDataset3(train_annot_dir,dataset_dir,in_w,out_w)

    model = mml.load_model(model_path)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, nesterov=True)

    train_epoch(train_set, model, optimizer, dataset_dir2,3 ,fmodel)    
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

def get_seg_from_model():
    
    image=imread(syncdir+project+'/models_models/data/B85-1_000.png')
    segmented = ensemble_segment([syncdir+project+'/models_models/models/000004_1661980651.pkl'], image, bs, in_w, out_w)

    seg_alpha = np.zeros((segmented.shape[0], segmented.shape[1], 4))
    seg_alpha[segmented > 0] = [0, 1.0, 1.0, 0.7]

    seg_alpha  = (seg_alpha * 255).astype(np.uint8)

    im_utils.imsave(syncdir+project+'/m1B85-1_000.png', seg_alpha)
    pass

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


get_seg_from_model()

#print(syncdir+datasets+'/B85-1_000.png')
#print(syncdir+project+'/models_models/000015_1578333385.pkl')

#test_data()
#gradian_data_setop([syncdir+project+'/models/000015_1578333385.pkl'], syncdir+project+'/segmentations')
#gradian_data_setop([syncdir+project+'/models/000001_1578331363.pkl'])

#train_type2(model_path, train_annot_dir, dataset_dir)
'''
train_type2(syncdir+project+'/models_models/models/000001_1661772775.pkl'
    , syncdir+project+'/models_models'+train
    , syncdir+project+'/models_models/data')

for x in range(10):
    train_type2(syncdir+project+'/models_models/models2/000001_1661772775.pkl'
        , syncdir+project+'/models_models'+train
        , syncdir+project+'/models_models/data2')
'''

'''
train_type3(syncdir+project+'/models_models/models3/000001_1661772775.pkl'
    , syncdir+project+'/models/000015_1578333385.pkl'
    , syncdir+project+train
    , syncdir+datasets
    , syncdir+project+'/models_models/data2'
    )
'''

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