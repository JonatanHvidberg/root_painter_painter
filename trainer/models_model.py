

import model_utils
import im_utils
from unet2 import UNetGNRes

import time
import os
import numpy as np
from skimage.io import imread, imsave
from skimage import color
import torch

'''
from model_utils
====================================================================================================
'''
def create_first_model_with_random_weights(model_dir):
    # used when no model was specified on project creation.
    os.mkdir(model_dir)
    model_num = 1
    model_name = str(model_num).zfill(6)
    model_name += '_' + str(int(round(time.time()))) + '.pkl'
    model = UNetGNRes()
    model = torch.nn.DataParallel(model)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    model.cuda()
    return model
'''
====================================================================================================
'''

def image_and_segmentation(imageDir, imageSegDir):
    image = im_utils.load_image(imageDir)
    imageSeg = imread(imageSegDir)
    print(np.shape(image))

    print(np.shape(imageSeg))

    imageVitSeg=np.zeros([(image).shape[0],(image).shape[1],4])

    for x in range(image.shape[0]): 
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                imageVitSeg[x][y][z]=image[x][y][z]

            imageVitSeg[x][y][3]=imageSeg[x][y][3]

            if (imageSeg[x][y][3] != 0):
                imageVitSeg[x][y][3]=255
                
    print(np.shape(imageVitSeg))
    pass

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

def test_ney_model():
    #make model 
    models=create_first_model_with_random_weights()

    #get_tiles()

    pass

syncdir = '/content/drive/MyDrive/drive_rp_sync'
datasets = '/datasets/biopores_750_training'
project = '/projects/test_01'

segmentations ='/segmentations'
val = '/annotations/val'

#print(syncdir+datasets+'/B85-1_000.png')
print(syncdir+project+'/models_models')

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