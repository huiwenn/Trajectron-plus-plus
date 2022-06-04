
from torch import nn, optim, utils
import numpy as np
import os
import time
import dill
import json
import random
import pathlib
import warnings
from tqdm import tqdm
import visualization
import evaluation
import matplotlib.pyplot as plt
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from model.model_utils import cyclical_lr
from model.dataset import EnvironmentDataset, collate
from tensorboardX import SummaryWriter
from tqdm import tqdm
from argument_parser import args
import torch
from evaluation.evaluation import compute_ade, compute_fde
from utils import prediction_output_to_trajectories
import pickle
import itertools


model_dir = '../experiments/argo2/models/pittsburgh-full' #austin-full'

data_dir =  '/data/jedrzej/argo2city/processed' #
eval_data_dict = 'val/pittsburgh_valenv.pkl' #val/miami_valenv.pkl' #'val/austin_full.pkl'

iter_num = 30
device = 'cuda:0'

conf = os.path.join(model_dir,'config.json')#'../config/argo2.json'
with open(conf, 'r', encoding='utf-8') as conf_json:
    hyperparams = json.load(conf_json)

hyperparams['dynamic_edges'] = args.dynamic_edges
hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
hyperparams['edge_addition_filter'] = args.edge_addition_filter
hyperparams['edge_removal_filter'] = args.edge_removal_filter
hyperparams['batch_size'] = args.batch_size
hyperparams['k_eval'] = args.k_eval
hyperparams['offline_scene_graph'] = args.offline_scene_graph
hyperparams['incl_robot_node'] = args.incl_robot_node
hyperparams['node_freq_mult_train'] = args.node_freq_mult_train
hyperparams['node_freq_mult_eval'] = args.node_freq_mult_eval
hyperparams['scene_freq_mult_train'] = args.scene_freq_mult_train
hyperparams['scene_freq_mult_eval'] = args.scene_freq_mult_eval
hyperparams['scene_freq_mult_viz'] = args.scene_freq_mult_viz
hyperparams['edge_encoding'] = not args.no_edge_encoding
hyperparams['use_map_encoding'] = args.map_encoding
hyperparams['augment'] = args.augment
hyperparams['override_attention_radius'] = args.override_attention_radius


eval_data_path = os.path.join(data_dir, eval_data_dict)
with open(eval_data_path, 'rb') as f:
    eval_env = dill.load(f, encoding='latin1')

eval_scenes = eval_env.scenes[:100]
#eval_env.scenes = list(itertools.chain.from_iterable(eval_scenes))
#eval_scenes = eval_env.scenes

eval_scenes_sample_probs = None

eval_dataset = EnvironmentDataset(eval_env,
                                  hyperparams['state'],
                                  hyperparams['pred_state'],
                                  scene_freq_mult=False,
                                  node_freq_mult=False,
                                  hyperparams=hyperparams,
                                  min_history_timesteps=hyperparams['minimum_history_length'],
                                  min_future_timesteps=hyperparams['prediction_horizon'],
                                  return_robot=True)
eval_data_loader = dict()

for node_type_data_set in eval_dataset:
    print(node_type_data_set.node_type,len(node_type_data_set))

    if len(node_type_data_set) == 0:
        continue

    node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                 collate_fn=collate,
                                                 pin_memory=True,
                                                 batch_size=1,
                                                 shuffle=True,
                                                 num_workers=10)
    eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

print(f"Loaded evaluation data")

log_writer = SummaryWriter(log_dir=model_dir)

model_registrar = ModelRegistrar(model_dir, device) 
model_registrar.load_models(iter_num=iter_num)

eval_trajectron = Trajectron(model_registrar,
                                 hyperparams,
                                 log_writer,
                                 device)
eval_trajectron.set_environment(eval_env)

def batch_statistics(prediction_output_dict,
                             dt,
                             max_hl,
                             ph,
                             node_type_enum,
                             kde=True,
                             obs=False,
                             map=None,
                             prune_ph_to_future=False,
                             best_of=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)
    
    '''
    global pkl_count 

    with open('/home/user/sophiasun/pred/miami_pred_%d.pkl'%pkl_count, 'wb') as f:
        pickle.dump(prediction_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('/home/user/sophiasun/pred/miami_gt_%d.pkl'%pkl_count, 'wb') as f:
        pickle.dump(futures_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    pkl_count += 1
    '''

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] =  {'ade': list(), 'fde': list(), 'kde': list(), 'obs_viols': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            if node.type == 'VEHICLE':
                ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node])
                fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])

                #get best of 
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                
                batch_error_dict[node.type]['ade'].extend(list(ade_errors))
                batch_error_dict[node.type]['fde'].extend(list(fde_errors))

    return batch_error_dict

def get_trajectory(prediction_output_dict,
                             dt,
                             max_hl,
                             ph):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=False)
    pred = []
    gt = []

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            if node.type == 'VEHICLE':
                pred.append(prediction_dict[t][node])
                gt.append(futures_dict[t][node])

    return pred, gt

max_hl = hyperparams['maximum_history_length']
ph = hyperparams['prediction_horizon']
model_registrar.to(device)

errors = []

with torch.no_grad():
    # Predict batch timesteps for evaluation dataset evaluation
    eval_batch_errors = []
    for scene in tqdm(eval_scenes, desc='Sample Evaluation', ncols=80):
        timesteps = scene.sample_timesteps(4) #args.eval_batch_size
        for horizon in [10, 30, 60]:
            ph = horizon
            predictions = eval_trajectron.predict(scene,
                                                timesteps,
                                                ph,
                                                num_samples=6,
                                                min_future_timesteps=ph,
                                                full_dist=False)
            
            samples, gt = get_trajectory(predictions,scene.dt,
                                     max_hl=max_hl, ph=ph)

            stats = batch_statistics(predictions,
                                        scene.dt,
                                        max_hl=max_hl,
                                        ph=ph,
                                        node_type_enum=eval_env.NodeType,
                                        map=scene.map)

            ml_pred = eval_trajectron.predict(scene,
                                    timesteps,
                                    ph,
                                    num_samples=1,
                                    min_future_timesteps=ph,
                                    z_mode=True,
                                    gmm_mode=True,
                                    full_dist=False)

            ml, _ = get_trajectory(ml_pred, scene.dt,
                            max_hl=max_hl, ph=ph)
            
            #overwites above
            stats = batch_statistics(ml_pred,
                                    scene.dt,
                                    max_hl=max_hl,
                                    ph=ph,
                                    node_type_enum=eval_env.NodeType,
                                    map=scene.map)
        
            if len(stats['VEHICLE']['ade']) > 0:
                d = {'id':scene.name, 'horizon': horizon,
                    'ade': np.mean(stats['VEHICLE']['ade']), 'fde': np.mean(stats['VEHICLE']['fde']),
                    'samples': samples, 'ml': ml}
                errors.append(d)

import csv
fieldnames = list(errors[0].keys())

'''
with open(os.path.join(model_dir,'ml_stats.pkl'), 'wb') as f:
    pickle.dump(errors, f, protocol=pickle.HIGHEST_PROTOCOL)
'''

ade = np.mean([a['ade'] for a in errors])
fde = np.mean([a['fde'] for a in errors])
print('ade', ade)
print('fde', fde)



# for node_type in eval_batch_errors[0].keys():
#     for metric in eval_batch_errors[0][node_type].keys():
#         print(metric)
#         metric_batch_error = []
#         for batch_errors in eval_batch_errors:
#             metric_batch_error.extend(batch_errors[node_type][metric])

#         if len(metric_batch_error) > 0:
#             print(len(metric_batch_error))

#             #log_writer.add_histogram(f"{node_type}/{namespace}/{metric}", metric_batch_error, curr_iter)
#             #log_writer.add_scalar(f"{node_type}/{namespace}/{metric}_mean", np.mean(metric_batch_error), curr_iter)
#             #log_writer.add_scalar(f"{node_type}/{namespace}/{metric}_median", np.median(metric_batch_error), curr_iter)
