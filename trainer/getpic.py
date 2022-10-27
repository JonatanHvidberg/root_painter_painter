import models_model_lib as mml
import model_utils
import im_utils
from skimage.io import imread, imsave
import numpy as np


def coler_gradian(segmented):

    seg_alpha = np.zeros((segmented.shape[0], segmented.shape[1], 4))

    

    for x in range(segmented.shape[0]):
        for y in range(segmented.shape[1]):
            seg_alpha[x][y] = [0, segmented[x][y], 1-segmented[x][y], 0.5]

            
    seg_alpha = (seg_alpha * 255).astype(np.uint8)

    return seg_alpha

def result(fname,n):
    model_dir=syncdir+project+'/models_models/models'+n
    path = model_utils.get_latest_model_paths(model_dir, k=1)[0]
    model = mml.load_model(path)


    image = imread(syncdir+project+'/models_models/data/'+fname)
    seg = mml.simbel_segment(model,image)
    gradian = coler_gradian(seg)
    imsave(syncdir+project+'/models_models/'+ fname, gradian)

def result_unsertensu(fname,om):
    model = model_utils.load_model(syncdir+project+ '/models/'+om)


    image = im_utils.load_image(syncdir+project+'/models_models/data/'+fname)
    seg = mml.simbel_segment_unsertensu(model,image)
    print('unsertensu', np.max(seg))
    print('unsertensu', seg)
    gradian = coler_gradian(seg)
    imsave(syncdir+project+'/models_models/u_'+ fname, gradian)

def both(fname,n,om):
    result(fname,n)
    result_unsertensu(fname,om)
       
syncdir='drive_rp_sync/projects/'
'''
n='5'
om='000032_1578339309.pkl'
project='biopores_a_corrective'
both('B58-1_002.png',n,om)
both('B100-1_002.png',n,om)
both('B1-1_000.png',n,om)


n='3'
om='000022_1578319359.pkl'
project='biopores_b_corrective'
both('B13-1_003.png',n,om)
both('B38-2_002.png',n,om)

n='5'
om='000028_1581172999.pkl'
project='nodules_a_corrective'
both('081_001.png',n,om)
both('053_003.png',n,om)
'''
n='5'
om='000023_1581690809.pkl'#last model
project='nodules_b_corrective'
both('075_000.png',n,om)
both('074_001.png',n,om)
'''
n='1'
om='000046_1578155544.pkl'
project='towers_a_corrective'
both('16_07_18_12E2d_P7181771_000.png',n,om)
both('16_07_04_10E5b_P7041084_000.png',n,om)
both('16_07_04_10E5b_P7041084_000.png',n,om)

n='2'
om='000031_1578167288.pkl'
project='towers_b_corrective'
both('16_06_21_12E4c_P6210519_000.png',n,om)
both('16_07_18_11E19b_P7181761_000.png',n,om)

n='4'
om='000047_1635794623.pkl'
project='rg_2017_ags'
both('Radimax_Cam4_PipeID_524_Position_2747_TS_2017-07-06 09.47.48.866F.bdiv.rgb_000.png',n,om)
both('Radimax_Cam4_PipeID_524_Position_2747_TS_2017-07-06 09.47.48.866F.bdiv.rgb_000.png',n,om)
'''