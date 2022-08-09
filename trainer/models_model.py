

import model_utils
import im_utils
from unet2 import UNetGNRes

import time
import os
import numpy as np
from skimage.io import imread, imsave
from skimage import color, img_as_float32
import torch

'''
from model_utils thens to mage my model
====================================================================================================
'''
def create_first_model_with_random_weights(model_dir):
    # used when no model was specified on project creation.
    #os.mkdir(model_dir)
    model_num = 1
    model_name = str(model_num).zfill(6)
    model_name += '_2_' + str(int(round(time.time()))) + '.pkl'
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
    image = img_as_float32(im_utils.load_image(imageDir))
    imageSeg = imread(imageSegDir)


    imageVitSeg=np.zeros([(image).shape[0],(image).shape[1],4])

    for x in range(image.shape[0]): 
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                imageVitSeg[x][y][z]=image[x][y][z]

            imageVitSeg[x][y][3]=imageSeg[x][y][3]

            if (imageSeg[x][y][3] != 0):
                imageVitSeg[x][y][3]=1
                
    
    return imageVitSeg

def dif_seg_aaa(imageSegDir, imageValDir, imageSaveDir):
    imageSeg = imread(imageSegDir) #shut be models model segrigation /home/jonatan/Documents/diku/BA/testbil/sek/B85-1_000.png
    imageVal = imread(imageValDir)

    imageValVal=np.zeros([(imageVal).shape[0],(imageVal).shape[1],4])

    for x in range(imageVal.shape[0]): 
        for y in range(imageVal.shape[1]):

            if (imageSeg[x][y][3] and not (imageVal[x][y][3])):
                imageValVal[x][y][1] = 255
                imageValVal[x][y][3] = 180

            if (not (imageSeg[x][y][3]) and imageVal[x][y][3]):
                imageValVal[x][y][0] = 255
                imageValVal[x][y][3] = 180

    im_utils.save_then_move(imageSaveDir, imageValVal)
    pass

def test_ney_model(out_path):
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
    segmented = ensemble_segment([syncdir+project+'/models_models/000001_test_1660051004.pkl'], image, bs, in_w, out_w)


    seg_alpha = np.zeros((segmented.shape[0], segmented.shape[1], 4))
    seg_alpha[segmented > 0] = [0, 1.0, 1.0, 0.7]

    seg_alpha  = (seg_alpha * 255).astype(np.uint8)

    im_utils.save_then_move(out_path, seg_alpha)
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
project = '/projects/test_01'

segmentations ='/segmentations'
val = '/annotations/val'

#print(syncdir+datasets+'/B85-1_000.png')
print(syncdir+project+'/models_models')

#test_ney_model(syncdir+project+'/models_models/B1-1_000.png')

create_first_model_with_random_weights(syncdir+project+'/models_models')

#image_and_segmentation('/home/jonatan/Documents/diku/BA/testbil/org/B85-1_000.jpg' ,'/home/jonatan/Documents/diku/BA/testbil/sek/B85-1_000.png')
#dif_seg_aaa()

'''
create_first_model_with_random_weights

'''

'''
create_first_model_with_random_weights
rat json

    for in 
'''