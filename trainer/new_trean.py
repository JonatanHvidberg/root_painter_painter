import models_model_lib as mml
import model_utils
import torch

from datasets2 import TrainDataset

from torch.nn.functional import softmax
from loss import combined_loss as criterion
from functools import partial
from model_utils import save_if_better
from torch.utils.data import DataLoader
import multiprocessing
import copy
from file_utils import ls
import shutil
import os

def train_type2(model_path, train_annot_dir, dataset_dir):
    train_set = TrainDataset(train_annot_dir,dataset_dir,in_w,out_w)

    path = model_utils.get_latest_model_paths(model_path, k=1)[0]
    model = mml.load_model(path)

    #model = mml.load_model(model_path)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, nesterov=True)

    train_epoch(train_set, model, optimizer, dataset_dir)


def train_epoch(train_set,model, optimizer, dataset_dir, type=2, fmodel='nan'):
    
    model.train()
    train_loader = DataLoader(train_set, bs, shuffle=True,
                              # 12 workers is good for performance
                              # on 2 RTX2080 Tis
                              # 0 workers is good for debugging
                              # don't go above 12 workers and don't go above the number of cpus
                              num_workers=min(multiprocessing.cpu_count(), 12),
                              drop_last=False, pin_memory=True)

    num_of_traning_no_better=0

    while num_of_traning_no_better<15:

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


        if validation(model, dataset_dir):
            num_of_traning_no_better = 0
        else:
            num_of_traning_no_better = num_of_traning_no_better+1


def validation(model,dataset_dir):

    get_val_metrics = partial(mml.get_val_metrics,
                          val_annot_dir=syncdir+project+'/models_models'+val,
                          dataset_dir=dataset_dir,
                          in_w=in_w, out_w=out_w, bs=bs)


    model_dir=syncdir+project+modelsDir
    prev_path = model_utils.get_latest_model_paths(model_dir, k=1)[0]
    prev_model =mml.load_model(prev_path)

    cur_metrics = get_val_metrics(copy.deepcopy(model))
    prev_metrics = get_val_metrics(prev_model)

    print('cur_metrics')
    print(cur_metrics)

    print('prev_metrics')
    print(prev_metrics)

    
    was_saved = save_if_better(model_dir, model, prev_path,
                           cur_metrics['f1'], prev_metrics['f1'])

    return was_saved



global in_w
global out_w
global mem_per_item
global total_mem
global bs

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



syncdir = 'drive_rp_sync'
datasets=['biopores_b_corrective'
    ,'nodules_a_corrective'
    ,'nodules_b_corrective'
    ,'towers_a_corrective'
    ,'towers_b_corrective']



for dataset in datasets:
    project = '/projects/'+dataset

    segmentations = '/segmentations'
    val = '/labels3/val'
    train = '/labels3/train'
    test = '/labels3/test'

    for x in range(1,6):
        modelsDir='/models_models/models3'+str(x)+'/'
        mml.create_first_model_with_random_weights(syncdir+project+modelsDir)
        train_type2(syncdir+project+modelsDir
            , syncdir+project+'/models_models'+train
            , syncdir+project+'/models_models/data')


'''


for dataset in datasets:
    project = '/projects/'+dataset
    for x in range(1,6):
        modelsDir='/models_models/models3'+str(x)+'/'
        os.mkdir(syncdir+project+modelsDir)



segmentations = '/segmentations'
val = '/labels2/val'
train = '/labels2/train'
test = '/labels2/test'

for x in range(1,6):
    modelsOld='/models_models/models'+str(x)+'/'

    modelsDir='/models_models/models2'+str(x)+'/'
    fnames = ls(syncdir+ project +modelsOld)  
    fnames = sorted(fnames)[-1:][0]
    shutil.copyfile(syncdir+project+modelsOld+fnames, syncdir+project+modelsDir+fnames)
    train_type2(syncdir+project+modelsDir
        , syncdir+project+'/models_models'+train
        , syncdir+project+'/models_models/data')

'''


