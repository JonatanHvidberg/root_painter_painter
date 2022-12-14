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
                c=0
                imsave(trainsave+file_names,
                    green_leb(imread(segmentations+file_names)))
                
            else:
                c=c+1
                imsave(valsave+file_names,
                    green_leb(imread(segmentations+file_names)))
        else:
            print(file_names)

def f40(project_name):
    
    dirr='drive_rp_sync/projects/'+ project_name +'/models_models/'
    labels='drive_rp_sync/projects/'+ project_name +'/models_models/labels/'
    labels2='drive_rp_sync/projects/'+ project_name +'/models_models/labels2/'

    segmentations = 'drive_rp_sync/projects/'+ project_name +'/segmentations/'

    trainsave = 'drive_rp_sync/projects/' + project_name + '/models_models/labels/train/'
    valsave   = 'drive_rp_sync/projects/' + project_name + '/models_models/labels/val/'
    testsave  = 'drive_rp_sync/projects/' + project_name + '/models_models/labels/test/'


    trainsave2 = 'drive_rp_sync/projects/' + project_name + '/models_models/labels2/train/'
    valsave2   = 'drive_rp_sync/projects/' + project_name + '/models_models/labels2/val/'
    testsave2  = 'drive_rp_sync/projects/' + project_name + '/models_models/labels2/test/'

    csvData = pandas.read_csv(dirr+'befor.csv')
    c=0
    for x in csvData.index:
        dataset=str(csvData['dataset'][x])
        file_names=csvData['file_names'][x]
        if dataset!='nan':
            if x<6:
                pass
            elif x>1407:
                shutil.copyfile(testsave2+ file_names
                    ,testsave+file_names)
                print(testsave+file_names)
            else:
                shutil.copyfile(labels2+ dataset +'/'+ file_names
                    ,labels+ dataset +'/'+file_names)


def lib2(project_name):
    dirr='drive_rp_sync/projects/'+ project_name +'/models_models/'
    labels='drive_rp_sync/projects/'+ project_name +'/models_models/labels/'
    labels2='drive_rp_sync/projects/'+ project_name +'/models_models/labels2/'

    segmentations = 'drive_rp_sync/projects/'+ project_name +'/segmentations/'


    trainsave2 = 'drive_rp_sync/projects/' + project_name + '/models_models/labels2/train/'
    valsave2   = 'drive_rp_sync/projects/' + project_name + '/models_models/labels2/val/'
    testsave2  = 'drive_rp_sync/projects/' + project_name + '/models_models/labels2/test/'

    csvData = pandas.read_csv(dirr+'befor.csv')

    c=0
    test_bool=0
    for x in csvData.index:
        dataset=str(csvData['dataset'][x])
        file_names=csvData['file_names'][x]
        if dataset=='test':
            test_bool=1
        if x<6:
            pass
        elif test_bool:
            if dataset=='nan':
                imsave(testsave2+file_names,
                    green_leb(imread(segmentations+file_names)))

            else:
                shutil.copyfile(labels+ dataset+'/'+ file_names
                    ,labels2+ dataset+'/'+file_names)
        elif dataset=='nan':
            if c==6:
                c=0
                imsave(valsave2+file_names,
                    green_leb(imread(segmentations+file_names)))
                
            else:
                c=c+1
                imsave(trainsave2+file_names,
                    green_leb(imread(segmentations+file_names)))
        else:
            shutil.copyfile(labels+ dataset+'/'+ file_names
                ,labels2+ dataset+'/'+file_names)

def saveimg(fra,til,dataset,file_names,segmentations):
    if dataset=='nan':
        imsave(til, green_leb(imread(segmentations+file_names)))
    else:
        shutil.copyfile(fra,til)

def lib3(project_name):
    
    dirr='drive_rp_sync/projects/'+ project_name +'/models_models/'
    labels='drive_rp_sync/projects/'+ project_name +'/models_models/labels/'
    labels3='drive_rp_sync/projects/'+ project_name +'/models_models/labels3/'

    segmentations = 'drive_rp_sync/projects/'+ project_name +'/segmentations/'



    trainsave3 = 'drive_rp_sync/projects/' + project_name + '/models_models/labels3/train/'
    valsave3   = 'drive_rp_sync/projects/' + project_name + '/models_models/labels3/val/'
    testsave3  = 'drive_rp_sync/projects/' + project_name + '/models_models/labels3/test/'

    os.mkdir(labels3)

    os.mkdir(trainsave3)
    os.mkdir(valsave3)
    os.mkdir(testsave3)


    csvData = pandas.read_csv(dirr+'befor.csv')

    testnum=len(csvData)-30
    c=0
    for x in csvData.index:
        dataset=str(csvData['dataset'][x])
        file_names=csvData['file_names'][x]

        if x<6:
            pass
        elif x>=testnum:
            saveimg(labels+dataset+'/'+file_names, testsave3+file_names,dataset,file_names,segmentations)
        else:
            if c==6:
                c=0
                saveimg(labels+dataset+'/'+file_names, valsave3+file_names,dataset,file_names,segmentations)
            else:
                c=c+1
                saveimg(labels+dataset+'/'+file_names, trainsave3+file_names,dataset,file_names,segmentations)

def moredata(project_name, datasets):

    dirr='drive_rp_sync/projects/'+ project_name +'/models_models/'
    segmentations = 'drive_rp_sync/projects/'+ project_name +'/segmentations/'
    datasave  = dirr+'data/'

    csvData = pandas.read_csv(dirr+'befor.csv')
    for x in csvData.index:
        file_names=csvData['file_names'][x]
        dataset=str(csvData['dataset'][x])
        if dataset=='nan':
            image = im_utils.load_image(datasets+os.path.splitext(file_names)[0] + '.jpg')
            imageSeg = imread(segmentations+file_names)
            imsave(datasave+file_names,image_and_segmentation(image,imageSeg))



datasets1 = 'drive_rp_sync/datasets/biopores_750_training/'
datasets2 = 'drive_rp_sync/datasets/nodules_750_training/'
datasets3 = 'drive_rp_sync/datasets/towers_750_training/'
moredata('biopores_b_corrective',datasets1)
moredata('nodules_a_corrective',datasets2)
moredata('nodules_b_corrective',datasets2)
moredata('towers_a_corrective',datasets3)
moredata('towers_b_corrective',datasets3)

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
