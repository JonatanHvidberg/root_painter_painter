import models_model_lib as mml
import model_utils
from file_utils import ls
import numpy as np
from skimage.io import imread, imsave
import os
import im_utils

def sum_error(o_model_name):
    return 0

    
    model_dir=syncdir+project+'/models_models/models'
    path = model_utils.get_latest_model_paths(model_dir, k=1)[0]
    model = mml.load_model(path)

    omodel = model_utils.load_model(syncdir+project+'/models/' +o_model_name)

    fnames = ls(syncdir+project+'/models_models/labels/test/')

    ys = np.zeros([len(fnames),4])

    c=0
    for fname in fnames:

        image = imread(syncdir+project+'/models_models/labels/test/'+fname)
        coreted_sum = np.sum((image[:,:,0]>0).astype(int))
        totel_pix = image.shape[0]*image.shape[1]
        #persent_coreted =coreted_sum/totel_pix
        #print(coreted_sum,totel_pix,persent_coreted)

    
        image = imread(syncdir+project+'/models_models/data/'+fname)
        predicted = mml.unet_segment(model, image, bs, in_w,
                         out_w, threshold=0.5)
        predicted_sum = np.sum(predicted)
        #persent_predicted =predicted_sum/totel_pix
        #print(predicted_sum,persent_predicted)

        image = im_utils.load_image(syncdir+datasets+'/'+os.path.splitext(fname)[0] + '.jpg')

        o_predicted = model_utils.unet_segment(omodel, image, bs, in_w,
                 out_w, threshold=None)

        unsertensy=mml.entorpy(o_predicted)
        unsertensy_predicted = unsertensy

        unsertensy_predicted = unsertensy_predicted.astype(int)


        unsertensy_sum = np.sum(unsertensy_predicted)

        ys[c,0]=coreted_sum

        ys[c,1]=predicted_sum

        ys[c,2]=unsertensy_sum

        ys[c,3]=totel_pix

        c = c+1
    
    ys=ys.astype(int)
    ys=ys[np.argsort(ys[:,0])]
    print(ys)