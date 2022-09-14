
import model_utils
import im_utils
from unet2 import UNetGNRes

import multiprocessing

import time
import os
import numpy as np
from skimage.io import imread, imsave
from skimage import color, img_as_float32
import torch
from torch.nn.functional import softmax
from loss import combined_loss as criterion
from torch.utils.data import DataLoader
from file_utils import ls
from model_utils import save_if_better
from functools import partial
import copy

import glob
import shutil
from math import ceil
import random
import skimage.util as skim_util
from skimage import color
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave

import shutil
import pandas

from metrics import get_metrics


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
from model_utils thens to mage my model
====================================================================================================
'''
def create_first_model_with_random_weights(model_dir):
    # used when no model was specified on project creation.
    
    model_num = 1
    model_name = str(model_num).zfill(6)
    model_name += '_' + str(int(round(time.time()))) + '.pkl'
    model = UNetGNRes()
    model = torch.nn.DataParallel(model)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    model.cuda()
    return model

def load_model(model_path):
    model = UNetGNRes()
    try:
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model)
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
    model.cuda()
    return model

def ensemble_segment(model_paths, image, bs, in_w, out_w,
                     threshold=0.5):
    """ Average predictions from each model specified in model_paths """
    pred_sum = None
    pred_count = 0
    # then add predictions from the previous models to form an ensemble
    for model_path in model_paths:
        cnn = load_model(model_path)

        preds = unet_segment(cnn, image, bs, in_w, out_w, threshold=None)
        if pred_sum is not None:
            pred_sum += preds
        else:
            pred_sum = preds
        pred_count += 1
        # get flipped version too (test time augmentation)
        flipped_im = np.fliplr(image)
        flipped_pred = unet_segment(cnn, flipped_im, bs, in_w,
                                    out_w, threshold=None)
        pred_sum += np.fliplr(flipped_pred)
        pred_count += 1
    foreground_probs = pred_sum / pred_count
    predicted = foreground_probs > threshold
    predicted = predicted.astype(int)
    return predicted

def simbel_segment(cnn, image):
    """ Average predictions from each model specified in model_paths """
    pred_sum = None
    pred_count = 0
    # then add predictions from the previous models to form an ensemble


    preds = unet_segment(cnn, image, bs, in_w, out_w, threshold=None)
    if pred_sum is not None:
        pred_sum += preds
    else:
        pred_sum = preds
    pred_count += 1
    # get flipped version too (test time augmentation)
    flipped_im = np.fliplr(image)
    flipped_pred = unet_segment(cnn, flipped_im, bs, in_w,
                                out_w, threshold=None)
    pred_sum += np.fliplr(flipped_pred)
    pred_count += 1
    foreground_probs = pred_sum / pred_count

    return foreground_probs

def simbel_segment_unsertensu(cnn, image):
    """ Average predictions from each model specified in model_paths """
    pred_sum = None
    pred_count = 0
    # then add predictions from the previous models to form an ensemble


    preds = model_utils.unet_segment(cnn, image, bs, in_w, out_w, threshold=None)
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
    foreground_probs = pred_sum / pred_count

    unsertensy = entorpy(foreground_probs)

    return unsertensy


def unet_segment(cnn, image, bs, in_w, out_w, threshold=0.5):
    """
    Threshold set to None means probabilities returned without thresholding.
    """
    assert image.shape[0] >= in_w, str(image.shape[0])
    assert image.shape[1] >= in_w, str(image.shape[1])

    tiles, coords = im_utils.get_tiles(image,
                                       in_tile_shape=(in_w, in_w, 4),
                                       out_tile_shape=(out_w, out_w))
    tile_idx = 0
    batches = []
    while tile_idx < len(tiles):
        tiles_to_process = []
        for _ in range(bs):
            if tile_idx < len(tiles):
                tile = tiles[tile_idx]

                tile = as_float32(tile)
                tile = im_utils.normalize_tile(tile)
                #print(tile)
                tile = np.moveaxis(tile, -1, 0)
                tile_idx += 1
                tiles_to_process.append(tile)
        tiles_for_gpu = torch.from_numpy(np.array(tiles_to_process))
        tiles_for_gpu.cuda()
        tiles_for_gpu = tiles_for_gpu.float()
        batches.append(tiles_for_gpu)

    output_tiles = []
    for gpu_tiles in batches:
        outputs = cnn(gpu_tiles)
        softmaxed = softmax(outputs, 1)
        foreground_probs = softmaxed[:, 1, :]  # just the foreground probability.
        if threshold is not None:
            predicted = foreground_probs > threshold
            predicted = predicted.view(-1).int()
        else:
            predicted = foreground_probs

        pred_np = predicted.data.cpu().numpy()
        out_tiles = pred_np.reshape((len(gpu_tiles), out_w, out_w))
        for out_tile in out_tiles:
            output_tiles.append(out_tile)

    assert len(output_tiles) == len(coords), (
        f'{len(output_tiles)} {len(coords)}')

    reconstructed = im_utils.reconstruct_from_tiles(output_tiles, coords,
                                                    image.shape[:-1])
    return reconstructed

def entorpy(pf):
    pb = 1-pf

    pfr=-np.log2(pf)
    pbr=-np.log2(pb)

    entorpy=np.nan_to_num(pf*pfr+pb*pbr)

    return entorpy

def get_val_metrics(cnn, val_annot_dir, dataset_dir, in_w, out_w, bs):
    """
    Return the TP, FP, TN, FN, defined_sum, duration
    for the {cnn} on the validation set

    TODO - This is too similar to the train loop. Merge both and use flags.
    """
    start = time.time()
    fnames = ls(val_annot_dir)
    fnames = [a for a in fnames if im_utils.is_photo(a)]
    # TODO: In order to speed things up, be a bit smarter here
    # by only segmenting the parts of the image where we have
    # some annotation defined.
    # implement a 'partial segment' which exlcudes tiles with no
    # annotation defined.
    tps = 0
    fps = 0
    tns = 0
    fns = 0
    defined_sum = 0
    for fname in fnames:
        annot_path = os.path.join(val_annot_dir,
                                  os.path.splitext(fname)[0] + '.png')
        # reading the image may throw an exception.
        # I suspect this is due to it being only partially written to disk
        # simply retry if this happens.
        try:
            annot = imread(annot_path)
        except Exception as ex:
            print(f'Exception reading annotation {annot_path} inside validation method.'
                  'Will retry in 0.1 seconds')
            print(fname, ex)
            time.sleep(0.1)
            annot = imread(annot_path)

        annot = np.array(annot)
        foreground = annot[:, :, 0].astype(bool).astype(int)
        background = annot[:, :, 1].astype(bool).astype(int)
        image_path_part = os.path.join(dataset_dir, os.path.splitext(fname)[0])
        image_path = glob.glob(image_path_part + '.*')[0]

        image = im_utils.imread(image_path)
        predicted = unet_segment(cnn, image, bs, in_w,
                                 out_w, threshold=0.5)

        print('arial',np.sum(predicted),np.sum(foreground))
        
        # mask defines which pixels are defined in the annotation.
        mask = foreground + background
        mask = mask.astype(bool).astype(int)
        predicted *= mask
        predicted = predicted.astype(bool).astype(int)
        y_defined = mask.reshape(-1)
        y_pred = predicted.reshape(-1)[y_defined > 0]
        y_true = foreground.reshape(-1)[y_defined > 0]
        tps += np.sum(np.logical_and(y_pred == 1, y_true == 1))
        tns += np.sum(np.logical_and(y_pred == 0, y_true == 0))
        fps += np.sum(np.logical_and(y_pred == 1, y_true == 0))
        fns += np.sum(np.logical_and(y_pred == 0, y_true == 1))
        defined_sum += np.sum(y_defined > 0)
    duration = round(time.time() - start, 3)
    metrics = get_metrics(tps, fps, tns, fns, defined_sum, duration)
    return metrics

def get_val_old_metrics(cnn, val_annot_dir, dataset_dir, in_w, out_w, bs):

    start = time.time()
    fnames = ls(val_annot_dir)
    fnames = [a for a in fnames if im_utils.is_photo(a)]

    tps = 0
    fps = 0
    tns = 0
    fns = 0
    defined_sum = 0
    for fname in fnames:
        annot_path = os.path.join(val_annot_dir,
                                  os.path.splitext(fname)[0] + '.png')
        # reading the image may throw an exception.
        # I suspect this is due to it being only partially written to disk
        # simply retry if this happens.
        try:
            annot = imread(annot_path)
        except Exception as ex:
            print(f'Exception reading annotation {annot_path} inside validation method.'
                  'Will retry in 0.1 seconds')
            print(fname, ex)
            time.sleep(0.1)
            annot = imread(annot_path)

        annot = np.array(annot)
        foreground = annot[:, :, 0].astype(bool).astype(int)
        background = annot[:, :, 1].astype(bool).astype(int)
        image_path_part = os.path.join(dataset_dir, os.path.splitext(fname)[0])
        image_path = glob.glob(image_path_part + '.*')[0]
        image = im_utils.load_image(image_path)
        predicted = model_utils.unet_segment(cnn, image, bs, in_w,
                                 out_w, threshold=None)
        unsertensy=entorpy(predicted)
        predicted = unsertensy > 0.5
        predicted = predicted.astype(int)
        # mask defines which pixels are defined in the annotation.
        mask = foreground + background
        mask = mask.astype(bool).astype(int)
        predicted *= mask
        predicted = predicted.astype(bool).astype(int)
        y_defined = mask.reshape(-1)
        y_pred = predicted.reshape(-1)[y_defined > 0]
        y_true = foreground.reshape(-1)[y_defined > 0]
        tps += np.sum(np.logical_and(y_pred == 1, y_true == 1))
        tns += np.sum(np.logical_and(y_pred == 0, y_true == 0))
        fps += np.sum(np.logical_and(y_pred == 1, y_true == 0))
        fns += np.sum(np.logical_and(y_pred == 0, y_true == 1))
        defined_sum += np.sum(y_defined > 0)
    duration = round(time.time() - start, 3)
    metrics = get_metrics(tps, fps, tns, fns, defined_sum, duration)
    return metrics

def get_val_dopel_metrics(cnno, cnn , val_annot_dir, dataset_dir, in_w, out_w, bs):

    start = time.time()
    fnames = ls(val_annot_dir)
    fnames = [a for a in fnames if im_utils.is_photo(a)]

    tps = 0
    fps = 0
    tns = 0
    fns = 0
    defined_sum = 0
    for fname in fnames:
        annot_path = os.path.join(val_annot_dir,
                                  os.path.splitext(fname)[0] + '.png')
        # reading the image may throw an exception.
        # I suspect this is due to it being only partially written to disk
        # simply retry if this happens.
        try:
            annot = imread(annot_path)
        except Exception as ex:
            print(f'Exception reading annotation {annot_path} inside validation method.'
                  'Will retry in 0.1 seconds')
            print(fname, ex)
            time.sleep(0.1)
            annot = imread(annot_path)

        annot = np.array(annot)
        foreground = annot[:, :, 0].astype(bool).astype(int)
        background = annot[:, :, 1].astype(bool).astype(int)
        image_path_part = os.path.join(dataset_dir, os.path.splitext(fname)[0])
        image_path = glob.glob(image_path_part + '.*')[0]

        image = im_utils.load_image(image_path)
        predicted = model_utils.unet_segment(cnno, image, bs, in_w,
                                 out_w, threshold=None)
        unsertensy=entorpy(predicted)
        predicted = unsertensy > 0.5
        predicted = predicted.astype(int)

        image = im_utils.imread(image_path) 
        predicted2 = unet_segment(cnn, image, bs, in_w,
                                 out_w, threshold=0.5)
        predicted2=predicted2.astype(bool).astype(int)
        predicted *= predicted2


        # mask defines which pixels are defined in the annotation.
        mask = foreground + background
        mask = mask.astype(bool).astype(int)
        predicted *= mask
        predicted = predicted.astype(bool).astype(int)
        y_defined = mask.reshape(-1)
        y_pred = predicted.reshape(-1)[y_defined > 0]
        y_true = foreground.reshape(-1)[y_defined > 0]
        tps += np.sum(np.logical_and(y_pred == 1, y_true == 1))
        tns += np.sum(np.logical_and(y_pred == 0, y_true == 0))
        fps += np.sum(np.logical_and(y_pred == 1, y_true == 0))
        fns += np.sum(np.logical_and(y_pred == 0, y_true == 1))
        defined_sum += np.sum(y_defined > 0)
    duration = round(time.time() - start, 3)
    metrics = get_metrics(tps, fps, tns, fns, defined_sum, duration)
    return metrics

'''
====================================================================================================
'''



'''
from im_utils
====================================================================================================
'''

def load_train_image_and_annot(dataset_dir, train_annot_dir):
    max_attempts = 60
    attempts = 0
    # used for logging which file caused the problem.
    latest_annot_path = None
    latest_im_path = None
    latest_error = None
    while attempts < max_attempts:
        attempts += 1
        # file systems are unpredictable.
        # We may have problems reading the file.
        # try-catch to avoid this.
        # (just try again)
        try:
            # set to None each time.
            latest_annot_path = None
            latest_im_path = None

            # This might take ages, profile and optimize
            fnames = ls(train_annot_dir)
            fnames = [a for a in fnames if im_utils.is_photo(a)]
            fname = random.sample(fnames, 1)[0]
            annot_path = os.path.join(train_annot_dir, fname)
            image_path_part = os.path.join(dataset_dir,
                                           os.path.splitext(fname)[0])
            # it's possible the image has a different extenstion
            # so use glob to get it
            image_path = glob.glob(image_path_part + '.*')[0]
            latest_im_path = image_path
            image = imread(image_path)
            latest_annot_path = annot_path
            annot = imread(annot_path).astype(bool)
            assert np.sum(annot) > 0
            
            assert image.shape[2] == 4 # should be RGB + segmentation
            # also return fname for debugging purposes.
            return image, annot, fname
        except Exception as e:
            latest_error = e
            # This could be due to an empty annotation saved by the user.
            # Which happens rarely due to deleting all labels in an 
            # existing annotation and is not a problem.
            # give it some time and try again.
            time.sleep(0.1)

    if attempts == max_attempts:
        if latest_annot_path is None: # if annot path still None we know it failed on the photo
            raise Exception(f'Could not load photo {latest_im_path}, {latest_error}')
        else:
            # otherwise it must have failed on the annotation
            raise Exception(f'Could not load annotation {latest_annot_path}, {e}')


'''
====================================================================================================
'''

'''
my funtion
'''
def as_float32(image):#img_as_float32 for imige with seg

        image_seg = np.array(image[:,:,3:])/255
        image_RGB = np.array(image[:,:,:3])

        image_RGB = img_as_float32(image_RGB)
        image = np.concatenate((image_RGB,image_seg), axis=2)

        return image

def image_and_segmentation_dir(imageDir, imageSegDir):
    image = im_utils.load_image(imageDir)
    imageSeg = imread(imageSegDir)
    return image_and_segmentation(image,imageSeg)

def image_and_segmentation(image, imageSeg):

    imageSegBul = np.array(imageSeg[:,:,2:3]) #take the segmen imiget and set it to one layer 
    imageVitSeg = np.concatenate((image,imageSegBul), axis=2) # concat imig and segmentaysen to 4 layer
    
    image = img_as_float32(image)
    imageSegBul = np.array(imageSeg[:,:,2:3], dtype=bool)
    imageVitSeg = np.concatenate((image,imageSegBul), axis=2)
    
    return imageVitSeg



def dif_seg_ann(imageSegDir, imageAnnDir, imageSaveDir):
    imageSeg = imread(imageSegDir) #shut be models model segrigation /home/jonatan/Documents/diku/BA/testbil/sek/B85-1_000.png
    imageAnn = imread(imageAnnDir)

    imageAnnAnn=np.zeros([(imageAnn).shape[0],(imageAnn).shape[1],4])

    for x in range(imageAnn.shape[0]): 
        for y in range(imageAnn.shape[1]):

            if (imageSeg[x][y][3] and not (imageAnn[x][y][3])): #false posetiv
                imageAnnAnn[x][y][1] = 255
                imageAnnAnn[x][y][3] = 180

            if (not (imageSeg[x][y][3]) and imageAnn[x][y][3]): #False negativ
                imageAnnAnn[x][y][0] = 255
                imageAnnAnn[x][y][3] = 180

    im_utils.save_then_move(imageSaveDir, imageAnnAnn)
    pass

def dif_new_ann(imageSegDir, imageAnnDir):
    imageSeg = imread(imageSegDir) #shut be models model segrigation /home/jonatan/Documents/diku/BA/testbil/sek/B85-1_000.png
    imageAnn = imread(imageAnnDir)
    return new_ann(imageSeg,imageAnn)

def new_ann(imageSeg, imageAnn):

    imageAnnAnn=np.zeros([(imageAnn).shape[0],(imageAnn).shape[1],4], np.uint8)

    for x in range(imageAnn.shape[0]): 
        for y in range(imageAnn.shape[1]):

            if ((imageSeg[x][y][3] and imageAnn[x][y][0]) or (not (imageSeg[x][y][3]) and imageAnn[x][y][1])):
                imageAnnAnn[x][y][1] = 255
                imageAnnAnn[x][y][3] = 180

            elif (imageAnn[x][y][3]):
                imageAnnAnn[x][y][0] = 255
                imageAnnAnn[x][y][3] = 180

            else:
                imageAnnAnn[x][y][1] = 255
                imageAnnAnn[x][y][3] = 180
    return imageAnnAnn.astype('uint8')



def test_new_model(out_path):
    #make model 
    #models=load_model(syncdir+project+'/models_models/000001_1659961126.pkl')

    imagedir = syncdir+datasets+'/B1-1_000.jpg'
    imageSegDir = syncdir+project+segmentations+ '/B1-1_000.png'

    image = image_and_segmentation(imagedir,imageSegDir)
    #image = im_utils.load_image(imagedir)

    #load imig
    #B9-1_002.png
    #B1-1_000.png god

    tiles, coords = im_utils.get_tiles(image, in_tile_shape=(in_w, in_w, 4), out_tile_shape=(out_w, out_w))


    #segmented = ensemble_segment([syncdir+project+'/models_models/000001_1659961126.pkl'], image, bs, in_w, out_w)
    segmented = ensemble_segment([syncdir+project+'/models_models/000001_2_1660065013.pkl'], image, bs, in_w, out_w)


    seg_alpha = np.zeros((segmented.shape[0], segmented.shape[1], 4))
    seg_alpha[segmented > 0] = [0, 1.0, 1.0, 0.7]

    seg_alpha  = (seg_alpha * 255).astype(np.uint8)

    im_utils.save_then_move(out_path, seg_alpha)
    pass



def setup(setup_dir):
    os.mkdir(setup_dir +'/models_models')
    os.mkdir(setup_dir +'/models_models' + '/data')
    os.mkdir(setup_dir +'/models_models' + '/models')
    os.mkdir(setup_dir +'/models_models' + '/labels')
    os.mkdir(setup_dir +'/models_models' + '/labels/train')
    os.mkdir(setup_dir +'/models_models' + '/labels/val')
    os.mkdir(setup_dir +'/models_models' + '/labels/test')
    pass

def reat_cfv_seg(project,dataset):
    #dirr='/home/jonatan/Downloads/projects/biopores_a_corrective'

    csvData = pandas.read_csv(project+'/annot_created_times6.csv')
    length=csvData.index.stop-6
    
    csvData.sort_values(["created_datetime"], axis=0, ascending=[False], inplace=True)
    

    c=0
    for x in csvData.index:
        dataset = csvData['dataset'][x]
        file_name = csvData['file_name'][x]

        label=dif_new_ann(project+'/segmentations/'+file_name, project+'/annotations/'+dataset+'/'+file_name)
        print(csvData)

        data_imig=image_and_segmentation_dir(dataset+'/'+os.path.splitext(file_name)[0] + '.jpg', project+'/segmentations/'+file_name)
        
        imsave(project+'/models_models/labels/data/'+file_name, data_imig)

        if c<20:
            imsave(project+'/models_models/labels/test/'+file_name, label)
        elif c>length:
            imsave(project+'/models_models/labels/'+dataset+'/'+file_name,label)
            
        else:
            imsave(project+'/models_models/labels/'+file_name,label)
            
        c=c+1

def setup_date(setup_dir):
    fnames = ls(setup_dir + segmentations)
    fnames = [a for a in fnames if im_utils.is_photo(a)]
    '''
    for fname in fnames:
        Dataimig = image_and_segmentation(syncdir+datasets+'/' + os.path.splitext(fname)[0] + '.jpg' ,setup_dir +'/segmentations/' + fname)
        imsave(setup_dir + '/models_models/data/'+fname, Dataimig)
    '''
    
    val = '/annotations2/val'
    train = '/annotations2/train'
    

    fnames = ls(setup_dir + val)
    fnames = [a for a in fnames if im_utils.is_photo(a)]

    for fname in fnames:
        #dif_new_ann(imageSegDir, imageAnnDir)
        DataImig = dif_new_ann(setup_dir +'/segmentations/' + fname ,setup_dir +'/annotations/val/' + fname)
        imsave(setup_dir + '/models_models/annotations2/val/'+fname, DataImig)


    fnames = ls(setup_dir + train)
    fnames = [a for a in fnames if im_utils.is_photo(a)]

    for fname in fnames:
        #dif_new_ann(imageSegDir, imageAnnDir)
        DataImig = dif_new_ann(setup_dir +'/segmentations/' + fname ,setup_dir +'/annotations/train/' + fname)
        imsave(setup_dir + '/models_models/annotations2/train/'+fname, DataImig)

    pass
