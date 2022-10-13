import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import json
import pandas
from file_utils import ls
from os.path import exists
from skimage.io import imread, imsave
import im_utils
from skimage import color, img_as_float32
import os



def new_leb(imageSeg, imageAnn):

    imageAnnAnn=np.zeros([(imageAnn).shape[0],(imageAnn).shape[1],4], np.uint8)

    for x in range(imageAnn.shape[0]): 
        for y in range(imageAnn.shape[1]):

            if ((imageSeg[x][y][3] and imageAnn[x][y][0]) or (not (imageSeg[x][y][3]) and imageAnn[x][y][1])):
                imageAnnAnn[x][y][1] = 255
                imageAnnAnn[x][y][3] = 180

            elif (imageAnn[x][y][3]):
                c=c+1
                imageAnnAnn[x][y][0] = 255
                imageAnnAnn[x][y][3] = 180

            else:
                imageAnnAnn[x][y][1] = 255
                imageAnnAnn[x][y][3] = 180
    return imageAnnAnn.astype('uint8')


def image_and_segmentation(image, imageSeg):

    imageSegBul = np.array(imageSeg[:,:,2:3]) #take the segmen imiget and set it to one layer 
    imageVitSeg = np.concatenate((image,imageSegBul), axis=2) # concat imig and segmentaysen to 4 layer
    
    image = img_as_float32(image)
    imageSegBul = np.array(imageSeg[:,:,2:3], dtype=bool)
    imageVitSeg = np.concatenate((image,imageSegBul), axis=2)
    
    return imageVitSeg



segmentations = 'drive_rp_sync/projects/rg_2017_ags/segmentations/'

train = 'drive_rp_sync/projects/rg_2017_ags/annotations/train'
val = 'drive_rp_sync/projects/rg_2017_ags/annotations/val'

datasets = 'drive_rp_sync/datasets/rg_2017_training_size_900_count_4000/'

trainsave = 'drive_rp_sync/projects/rg_2017_ags/models_models/labels/train'
valsave   = 'drive_rp_sync/projects/rg_2017_ags/models_models/labels/val'
datasave  = 'drive_rp_sync/projects/rg_2017_ags/models_models/data/'

fnames = ls(segmentations)

for fname in fnames:
    #image = im_utils.load_image(datasets+os.path.splitext(fname)[0] + '.jpg')
    imageSeg = imread(segmentations+fname)

    #imsave(datasave+fname,image_and_segmentation(image,imageSeg))

    if (exists(train+fname)):

        imageAnn = imread(train+fname)
        imsave(trainsave+fname,new_leb(imageSeg, imageAnn))

    elif (exists(val+fname)):

        imageAnn = imread(val+fname)
        imsave(valsave+fname,new_leb(imageSeg, imageAnn))

#    file_exists = exists(path+fname)