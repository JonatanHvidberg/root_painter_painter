

import model_utils
import im_utils
from unet2 import UNetGNRes
from datasets2 import TrainDataset as TrainDataset2
import models_model_lib 

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

from model_utils import save_if_better

from file_utils import ls
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
                print('shape(tile)')
                print(np.shape(tile))
                #tile = img_as_float32(tile)
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



'''
====================================================================================================
'''

def image_and_segmentation(imageDir, imageSegDir):
    image = im_utils.load_image(imageDir)
    imageSeg = imread(imageSegDir)

    '''

    imageVitSeg=np.zeros([(image).shape[0],(image).shape[1],4])

    for x in range(image.shape[0]): 
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                imageVitSeg[x][y][z]=image[x][y][z]

            imageVitSeg[x][y][3]=imageSeg[x][y][3]

            if (imageSeg[x][y][3] != 0):
                imageVitSeg[x][y][3]=1
    
    '''
    imageSegBul = np.array(imageSeg[:,:,2:3]) #take the segmen imiget and set it to one layer 
    imageVitSeg = np.concatenate((image,imageSegBul), axis=2) # concat imig and segmentaysen to 4 layer

    #im_utils.save_then_move(saveDir, imageVitSeg)
    
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

    imageAnnAnn=np.zeros([(imageAnn).shape[0],(imageAnn).shape[1],4])

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
    return imageAnnAnn



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
    os.mkdir(setup_dir +'/models_models' + '/seg')
    os.mkdir(setup_dir +'/models_models' + '/annotations')
    os.mkdir(setup_dir +'/models_models' + '/annotations/train')
    os.mkdir(setup_dir +'/models_models' + '/annotations/val')
    pass

def setup_date(setup_dir):
    fnames = ls(setup_dir + segmentations)
    fnames = [a for a in fnames if im_utils.is_photo(a)]

    for fname in fnames:
        Dataimig = image_and_segmentation(syncdir+datasets+fname ,setup_dir +'/segmentations/' + fname)
        im_utils.save_then_move(setup_dir '/models_models/data/'+fname, Dataimig)

    '''
    val = '/annotations/val'
    train = '/annotations/train'
    '''

    fnames = ls(setup_dir + val)
    fnames = [a for a in fnames if im_utils.is_photo(a)]

    for fname in fnames:
        #dif_new_ann(imageSegDir, imageAnnDir)
        DataImig = dif_new_ann(setup_dir +'/segmentations/' + fname ,setup_dir +'/annotations/val/' + fname)
        im_utils.save_then_move(setup_dir '/models_models/annotations/val'+fname, Dataimig)


    fnames = ls(setup_dir + train)
    fnames = [a for a in fnames if im_utils.is_photo(a)]

    for fname in fnames:
        #dif_new_ann(imageSegDir, imageAnnDir)
        DataImig = dif_new_ann(setup_dir +'/segmentations/' + fname ,setup_dir +'/annotations/train/' + fname)
        im_utils.save_then_move(setup_dir '/models_models/annotations/train'+fname, Dataimig)

    pass



def train_one_epoch(train_set,model, optimizer):
    
    model.train()


    train_loader = DataLoader(train_set, bs, shuffle=True,
                              # 12 workers is good for performance
                              # on 2 RTX2080 Tis
                              # 0 workers is good for debugging
                              # don't go above 12 workers and don't go above the number of cpus
                              num_workers=min(multiprocessing.cpu_count(), 12),
                              drop_last=False, pin_memory=True)

    for step, (photo_tiles,
           foreground_tiles,
           defined_tiles) in enumerate(train_loader):


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

        if (step>9):
            pass

    validation(model)


def validation(model):

    get_val_metrics = partial(models_model_lib.get_val_metrics,
                          val_annot_dir=' ',
                          dataset_dir=' ',
                          in_w=in_w, out_w=out_w, bs=bs)


    model_dir=' '
    prev_model, prev_path = model_utils.get_prev_model(model_dir)

    cur_metrics = get_val_metrics(copy.deepcopy(model))
    prev_metrics = get_val_metrics(prev_model)

    was_saved = save_if_better(model_dir, model, prev_path,
                           cur_metrics['f1'], prev_metrics['f1'])


def train_type2(model_path, train_annot_dir, dataset_dir):
    train_set = TrainDataset2(train_annot_dir,dataset_dir,in_w,out_w)

    model = load_model(model_path)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, nesterov=True)

    train_one_epoch(train_set, model, optimizer)    
    pass

'''
Data
'''
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
print(syncdir+project+'/models_models')

setup(syncdir+project)
setup_date(syncdir+project)

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