from functools import partial
import models_model_lib as mml
import model_utils
import copy

import torch


def val_info(dataset_dir,omodel, model_dir):
    val = '/labels3/val'
    train = '/labels3/train'
    test = '/labels3/test'


    get_val_metrics = partial(mml.get_val_metrics,
                          val_annot_dir=syncdir+project+'/models_models'+val,
                          dataset_dir=dataset_dir,
                          in_w=in_w, out_w=out_w, bs=bs)

    get_test_metrics = partial(mml.get_val_metrics,
                      val_annot_dir=syncdir+project+'/models_models'+test,
                      dataset_dir=dataset_dir,
                      in_w=in_w, out_w=out_w, bs=bs)
'''
    get_old_metrics = partial(mml.get_val_old_metrics,
                  val_annot_dir=syncdir+project+'/models_models'+test,
                  dataset_dir=dataset_dir,
                  in_w=in_w, out_w=out_w, bs=bs)
'''
    print(model_dir)
    print(model_utils.get_latest_model_paths(model_dir, k=1))
    path = model_utils.get_latest_model_paths(model_dir, k=1)[0]
    model =mml.load_model(path)

#    oldmodel = model_utils.load_model(syncdir+project+ '/models/' + omodel)

    val_metrics = get_val_metrics(copy.deepcopy(model))
    test_metrics = get_test_metrics(copy.deepcopy(model))
#    old_metrics = get_old_metrics(copy.deepcopy(oldmodel))

    print('val_metrics')
    print(val_metrics)

    print('test_metrics')
    print(test_metrics)

#    print('old_metrics')
#    print(old_metrics)


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

syncdir='drive_rp_sync'



print('biopores_a_corrective')
datasets = '/datasets/biopores_750_training'
project = '/projects/biopores_a_corrective'
om='000023_1578320581.pkl'
for x in range(1,6):
    model_dir=syncdir+project+'/models_models/models3'+str(x)
    val_info(syncdir+project+'/models_models/data',om, model_dir)

print('biopores_b_corrective')
project = '/projects/biopores_b_corrective'
om='000023_1578320581.pkl'
for x in range(1,6):
    model_dir=syncdir+project+'/models_models/models3'+str(x)
    val_info(syncdir+project+'/models_models/data',om, model_dir)

print('nodules_a_corrective')
datasets = '/datasets/nodules_750_training'
project = '/projects/nodules_a_corrective'
om='000028_1581172999.pkl'
for x in range(1,6):
    model_dir=syncdir+project+'/models_models/models3'+str(x)
    val_info(syncdir+project+'/models_models/data',om, model_dir)

print('nodules_b_corrective')
project = '/projects/nodules_b_corrective'
om='000023_1581690809.pkl'
for x in range(1,6):
    model_dir=syncdir+project+'/models_models/models3'+str(x)
    val_info(syncdir+project+'/models_models/data',om, model_dir)


print('towers_a_corrective')
datasets = '/datasets/towers_750_training'
project = '/projects/towers_a_corrective'
om='000046_1578155544.pkl'
for x in range(1,6):
    model_dir=syncdir+project+'/models_models/models3'+str(x)
    val_info(syncdir+project+'/models_models/data',om, model_dir)

print('towers_b_corrective')
project = '/projects/towers_b_corrective'
om='000040_1578171692.pkl'
for x in range(1,6):
    model_dir=syncdir+project+'/models_models/models3'+str(x)
    val_info(syncdir+project+'/models_models/data',om, model_dir)