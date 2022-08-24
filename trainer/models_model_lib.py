
'''
import torch
from torch import nn
from torch.nn import Parameter

def test_opt():

#    x = torch.tensor([1,2,3,4,5,6,7,8,9,10])
#    y = torch.tensor([10,11,12,13,14,15,16,17,18,19], dtype=torch.float)

    x = torch.tensor([1,2,3,4,5])
    y = torch.tensor([11,12,13,14,15], dtype=torch.float)

    # model

    a = torch.randn(1, requires_grad=True, dtype=torch.float)
    b = torch.randn(1, requires_grad=True, dtype=torch.float)


    print(a,b)
    model = [Parameter(a),Parameter(b)]

    lr=0.05
    criterion = nn.MSELoss()
    optimaser = torch.optim.SGD(model, lr=lr)


    for epoch in range(20):
        optimaser.zero_grad()

        y_p=model[0]+model[1]*x

        loss = criterion(y_p, y)

        loss.backward()

        optimaser.step()

        print(loss)

    print(x,y,y_p)
    print(model[0],model[1])

test_opt()
'''


import model_utils
import im_utils
from unet2 import UNetGNRes
from datasets2 import TrainDataset

import time
import os
import numpy as np
from skimage.io import imread, imsave
from skimage import color, img_as_float32
import torch
from torch.nn.functional import softmax


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
            fnames = [a for a in fnames if is_photo(a)]
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
            assert image.shape[2] == 3 # should be RGB
            # also return fname for debugging purposes.
            return image, annot, fname
        except Elatest_im_pathxception as e:
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