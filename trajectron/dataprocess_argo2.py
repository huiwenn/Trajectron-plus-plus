from glob import glob
import pickle
import os
import numpy as np
from typing import Any, Dict, List, Tuple, Union
import torch
from torch import nn, optim, utils
from torch.utils.data import IterableDataset, DataLoader, Dataset
from tqdm import tqdm
import sys
import pandas as pd
import dill
import cv2

import io
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as img
from PIL.Image import Image

from shapely.geometry import Point, LineString, MultiLineString
from shapely import affinity
from pyquaternion import Quaternion
from sklearn.model_selection import train_test_split

from environment import Environment, Scene, Node, GeometricMap, derivative_of
from pathlib import Path
from collections import defaultdict
import multiprocessing


sys.path.append("/data/jedrzej/argo2city")
sys.path.append("/data/jedrzej/argo2city/av2_prerelease_sample_code")

from lib.data.scenario_serialization import load_argoverse_scenario_parquet, load_static_map_json
from lib.data.data_schema import (
    ArgoverseScenario,
    ObjectType,
    Polyline,
    ScenarioStaticMap,
    TrackCategory,
)

FREQUENCY = 10
dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}

def process_scene(p):
    scenario_id = p.split("/")[-1]

    map_path = os.path.join(p, f"log_map_archive_{scenario_id}.json")
    parquet_path = os.path.join(p, f"scenario_{scenario_id}.parquet")

    scenario = load_argoverse_scenario_parquet(parquet_path)

    scene_id = scenario.scenario_id
    city = scenario.city
    
    data = pd.DataFrame(columns=['scene_id',
                                'frame_id',
                                 'type',
                                 'node_id',
                                 'robot',
                                 'x', 'y',
                                 'heading'])

    data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°')]))
    
    for track in scenario.tracks:
        track_id = track.track_id
        category = track.object_type.value
        isagent = track.category.value == 2 # if ==2 then is agent

        #only look at pedestrian and vehicles 
        if category=='pedestrian':
            our_category = 'PEDESTRIAN'
        elif category == 'vehicle' or category == 'motorcyclist':
            our_category = 'VEHICLE' #env.NodeType.VEHICLE
        else:
            continue

        trackstates = track.object_states
        xy = [s.position for s in trackstates]
        heading = [s.heading for s in trackstates]
        velocity =  [s.velocity for s in trackstates]

        for state in trackstates:
            data_point = pd.Series({'scene_id': scene_id,
                                    'frame_id': state.timestep,
                                    'type': our_category,
                                    'node_id': 'ego' if isagent else track_id,
                                    'robot': isagent,
                                    'x': state.position[0],
                                    'y': state.position[1],
                                    'vx': state.velocity[0],
                                    'vy': state.velocity[1],
                                    'heading': state.heading})
            data = data.append(data_point, ignore_index=True)

    data.sort_values('frame_id', inplace=True)

    x_min = np.round(data['x'].min())
    x_max = np.round(data['x'].max())
    y_min = np.round(data['y'].min())
    y_max = np.round(data['y'].max())

    data['x'] = data['x'] - x_min
    data['y'] = data['y'] - y_min

    max_timesteps = data['frame_id'].max()
    scene = Scene(timesteps=max_timesteps + 1, dt=0.1, name=str(scene_id))


    # -- process map --

    static_map = load_static_map_json(map_path)
    centerlane = [b['centerline'] for _,b in static_map['lane_segments'].items()]
    xs = [[a['x'] for a in c] for c in centerlane]
    ys = [[a['y'] for a in c] for c in centerlane]
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    coords = [((c[0]['x'],c[0]['y']),(c[-1]['x'],c[-1]['y'])) for c in centerlane]

    lines = MultiLineString(coords)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    patch_h = max_y - min_y
    patch_w = max_x - min_x
    patch_x = min_x + patch_w/2.0
    patch_y = min_y + patch_h/2.0

    canvas_size = (np.round(3 * patch_h).astype(int), np.round(3 * patch_w).astype(int))

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h/patch_h
    scale_width = canvas_w/patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for line in lines:
        #new_line = line.intersection(patch)
        new_line = affinity.affine_transform(line,
                                             [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
        new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))

        coords = np.asarray(list(new_line.coords), np.int32)
        coords = coords.reshape((-1, 2))
        msk = cv2.polylines(map_mask, [coords], False, 1, 2)


    homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
    geo_map = GeometricMap(data=msk, homography=homography, description='map')
    type_map = {'PEDESTRIAN':geo_map, 'VEHICLE':geo_map, 'VISUALIZATION':geo_map}
    scene.map = type_map

    # -- end of map procecing -- 

    for node_id in pd.unique(data['node_id']):

        node_df = data[data['node_id'] == node_id]
        vx = node_df['vx'].values
        vy = node_df['vy'].values

        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)


        data_dict = {('position', 'x'): node_df['x'].values,
                     ('position', 'y'): node_df['y'].values,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay,
                     ('heading', '°'): node_df['heading'].values}

        node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
        node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id, data=node_data)
        scene.nodes.append(node)

    return scene, city

def process_worker(path_list):
    scenes = defaultdict(list)
    for p in tqdm(path_list):
        try:
            scene, city = process_scene(p)
        except Exception as e: 
            print(e)
            continue
        if scene is not None:
            scenes[city].append(scene)
    return scenes

def main():
    #for data_class in ['val', 'train']:
    data_class = 'train'
    print('dataclass ', data_class)
    data_path = '/data/jedrzej/argo2city/'
    output_path = '/data/jedrzej/argo2city/processed/' 

    env = Environment(node_type_list=['PEDESTRIAN', 'VEHICLE'], standardization=standardization) #, 'PEDESTRIAN'
    attention_radius = dict()
    attention_radius[('PEDESTRIAN', 'PEDESTRIAN')] = 10.0
    attention_radius[('PEDESTRIAN', 'VEHICLE')] = 20.0
    attention_radius[('VEHICLE', 'PEDESTRIAN')] = 20.0
    attention_radius[('VEHICLE', 'VEHICLE')] = 30.0

    env.attention_radius = attention_radius
    env.robot_type = env.NodeType.VEHICLE
    scenes = defaultdict(list)

    input_path = os.path.join(data_path, data_class, '*')
    path_list = glob(input_path)
    
    count = 0
    pickle_count = 0

    # 200,000 total files -> 20 pickles. 10,000 per pickle
    # 20 workers. 500 files per worker
    if data_class == 'train':
        pool = multiprocessing.Pool(20)
        pickle_path_lists = [path_list[10000*i:10000*(i+1)] for i in range(20)]
        
        for p in tqdm(pickle_path_lists):
            p_list = [p[500*i:500*(i+1)] for i in range(20)]
            results = pool.map(process_worker, p_list)
            scenes = defaultdict(list)
            print('finished processing batch')
            for r in results:
                for k,v in r.items():
                    scenes[k] = scenes[k] + v

            data_dict_path = os.path.join(output_path, data_class,'argo2_scenes_' + str(pickle_count) + '.pkl')
            with open(data_dict_path, 'wb') as f:
                dill.dump(scenes, f, protocol=dill.HIGHEST_PROTOCOL)
            
            pickle_count += 1
            count += len(p)
            print(f'Processed {count:.2f} scenes')

    
    if data_class == 'val':
        for p in tqdm(path_list):
            try:
                scene, city = process_scene(p)
            except Exception as e: 
                print(e)
                continue

            if scene is not None:
                '''
                if data_class == 'train':
                    scene.augmented = list()
                    angles = np.arange(0, 360, 15)
                    for angle in angles:
                        scene.augmented.append(augment_scene(scene, angle))
                '''
                scenes[city].append(scene)
            count += 1

        for c in scenes.keys():
            city_scenes = scenes[c]
        
            env.scenes = city_scenes

            data_dict_path = os.path.join(output_path, data_class, 'argo2city_' + c + '_full.pkl')
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
            print('Saved Environment! ' + c)


def consiolidate_train():
    path_list = glob('/data/jedrzej/argo2city/processed/train/*')
    cities = ['palo-alto'] #'washington-dc', 'palo-alto', 'miami', 'austin', 'dearborn', 'pittsburgh'
    all_files = []
    for filepath in tqdm(path_list):
        if filepath.split('_')[-2] != 'scenes':
            print('skipped ' + filepath)
            continue
        with open(filepath, 'rb') as f:
            this_scene = pickle.load(f)
            all_files.append(this_scene)

    for c in tqdm(cities):
        print(c)
        env = Environment(node_type_list=['PEDESTRIAN', 'VEHICLE'], standardization=standardization) #, 'PEDESTRIAN'
        attention_radius = dict()
        attention_radius[('PEDESTRIAN', 'PEDESTRIAN')] = 10.0
        attention_radius[('PEDESTRIAN', 'VEHICLE')] = 20.0
        attention_radius[('VEHICLE', 'PEDESTRIAN')] = 20.0
        attention_radius[('VEHICLE', 'VEHICLE')] = 30.0

        env.attention_radius = attention_radius
        env.robot_type = env.NodeType.VEHICLE
        
        scenes = []

        for this_scene in all_files:
            for citi, sce in this_scene.items():
                    if citi!=c:
                        scenes += sce
        
        env.scenes = scenes
        print('made scene, saving data')
        #data_dict_path = os.path.join('/data/jedrzej/argo2city/processed/train/',  c + '_trainingfull.pkl')
        data_dict_path = os.path.join('/home/user/sophiasun/',  c + '_trainingfull.pkl')

        with open(data_dict_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        print('Saved Environment! ' + c)
        del scenes
        del env
        

def consiolidate_val():
    path_list = glob('/data/jedrzej/argo2city/processed/val/*')
    cities = ['austin', 'miami', 'dearborn', 'pittsburgh', 'washington-dc', 'palo-alto']
    scenes = defaultdict(list)
    for c in tqdm(cities):
        for i in range(1,6):
            filepath = '/data/jedrzej/argo2city/processed/val/argo2city_{}_{}.pkl'.format(c,str(i))
            with open(filepath, 'rb') as f:
                try:
                    envs = pickle.load(f)
                except Exception as e: 
                    print(e, filepath)
            scenes[c] = scenes[c] + envs.scenes

    print(len(scenes))

    for c in scenes.keys():
        print(c)
        env = Environment(node_type_list=['PEDESTRIAN', 'VEHICLE'], standardization=standardization) #, 'PEDESTRIAN'
        attention_radius = dict()
        attention_radius[('PEDESTRIAN', 'PEDESTRIAN')] = 10.0
        attention_radius[('PEDESTRIAN', 'VEHICLE')] = 20.0
        attention_radius[('VEHICLE', 'PEDESTRIAN')] = 20.0
        attention_radius[('VEHICLE', 'VEHICLE')] = 30.0

        env.attention_radius = attention_radius
        env.robot_type = env.NodeType.VEHICLE
        

        env.scenes = scenes[c]
        data_dict_path = os.path.join('/data/jedrzej/argo2city/processed/val/',  c + '_full.pkl')
        with open(data_dict_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

        this_scenes = []

        for citi, sce in scenes.items():
            if citi!=c:
                this_scenes += sce
    
        env.scenes = this_scenes

        data_dict_path = os.path.join('/data/jedrzej/argo2city/processed/val/',  c + '_valenv.pkl')
        with open(data_dict_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        print('Saved Environment! ' + c)



def consiolidate_city():
    path_list = glob('/data/jedrzej/argo2city/processed/train/*') #+glob('/data/jedrzej/argo2city/processed/val/*')
    cities = ['palo-alto', 'washington-dc', 'miami', 'austin', 'dearborn', 'pittsburgh']
    all_files = []
    for filepath in tqdm(path_list):
        if filepath.split('_')[-2] != 'scenes':
            print('skipped ' + filepath)
            continue
        with open(filepath, 'rb') as f:
            this_scene = pickle.load(f)
            all_files.append(this_scene)


    city_scenes = {c:[] for c in cities}
    for this_scene in all_files:
        for citi, sce in this_scene.items():
            city_scenes[citi].append(sce)
        

    for c in tqdm(cities):
        print(c)
        env = Environment(node_type_list=['PEDESTRIAN', 'VEHICLE'], standardization=standardization) #, 'PEDESTRIAN'
        attention_radius = dict()
        attention_radius[('PEDESTRIAN', 'PEDESTRIAN')] = 10.0
        attention_radius[('PEDESTRIAN', 'VEHICLE')] = 20.0
        attention_radius[('VEHICLE', 'PEDESTRIAN')] = 20.0
        attention_radius[('VEHICLE', 'VEHICLE')] = 30.0

        env.attention_radius = attention_radius
        env.robot_type = env.NodeType.VEHICLE
        
        env.scenes = city_scenes[c]
        print('made scene, saving data')
        #data_dict_path = os.path.join('/data/jedrzej/argo2city/processed/train/',  c + '_trainingfull.pkl')
        data_dict_path = os.path.join('/home/user/sophiasun/',  c + '_citydata.pkl')

        with open(data_dict_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        print('Saved Environment! ' + c)
        del env

if __name__ == "__main__":
    #main()
    #consiolidate_val()
    #consiolidate_train()
    consiolidate_city()
    
