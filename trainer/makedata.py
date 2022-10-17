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
import shutil


def new_leb(imageSeg, imageAnn):

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

def green_leb(imageSeg):
    imageLeb=np.zeros([(imageSeg).shape[0],(imageSeg).shape[1],4], np.uint8)
    imageLeb[:,:,1]=255
    imageLeb[:,:,3]=180
    return imageLeb.astype('uint8')


def image_and_segmentation(image, imageSeg):

    imageSegBul = np.array(imageSeg[:,:,2:3]) #take the segmen imiget and set it to one layer 
    imageVitSeg = np.concatenate((image,imageSegBul), axis=2) # concat imig and segmentaysen to 4 layer
    
    image = img_as_float32(image)
    imageSegBul = np.array(imageSeg[:,:,2:3], dtype=bool)
    imageVitSeg = np.concatenate((image,imageSegBul), axis=2)
    
    return imageVitSeg



'''
train = 'drive_rp_sync/projects/rg_2017_ags/annotations/train/'
val = 'drive_rp_sync/projects/rg_2017_ags/annotations/val/'

datasets = 'drive_rp_sync/datasets/rg_2017_training_size_900_count_4000/'

trainsave = 'drive_rp_sync/projects/rg_2017_ags/models_models/labels/train/'
valsave   = 'drive_rp_sync/projects/rg_2017_ags/models_models/labels/val/'
datasave  = 'drive_rp_sync/projects/rg_2017_ags/models_models/data/'


fnames = ls(segmentations)
'''



def reat_cfv_seg(project_name):

    dirr='drive_rp_sync/projects/'+ project_name +'/models_models/'
    labels='drive_rp_sync/projects/'+ project_name +'/models_models/labels/'

    segmentations = 'drive_rp_sync/projects/'+ project_name +'/segmentations/'

    trainsave = 'drive_rp_sync/projects/' + project_name + '/models_models/labels/train/'
    valsave   = 'drive_rp_sync/projects/' + project_name + '/models_models/labels/val/'
    testsave  = 'drive_rp_sync/projects/' + project_name + '/models_models/labels/test/'

    csvData = pandas.read_csv(dirr+'befor.csv')

    c=0
    for x in csvData.index:
        dataset=str(csvData['dataset'][x])
        file_names=csvData['file_names'][x]
        if x<7:
            pass
        elif x>1407:
            print('x>1407' ,x)
            if dataset=='nan':
                imsave(testsave+file_names,
                    green_leb(imread(segmentations+file_names)))

            else:
                shutil.move(labels+ dataset+'/'+ file_names
                    ,testsave+file_names)
        elif dataset=='nan':
            if c==6:
                imsave(trainsave+file_names,
                    green_leb(imread(segmentations+file_names)))
                c=0
            else:
                c=c+1
                imsave(valsave+file_names,
                    green_leb(imread(segmentations+file_names)))
        else:
            print(file_names)

reat_cfv_seg('rg_2017_ags')
'''
for nex 

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

'''

'''
            print('x<7' ,x)
            if dataset =='nan':
                pass

            else:
                shutil.move(labels+ dataset+'/'+ file_names
                    ,labels+file_names)
'''
