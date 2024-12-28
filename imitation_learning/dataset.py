import json
import random
import numpy as np
import os
def load_data(task):
    filepath = f"meta-world/2M/{task}.npz"
    data = np.load(filepath)
    observations = data['observations']
    next_observations = data['next_observations']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']
    
    observations = observations.reshape(-1, observations.shape[-1])
    actions = actions.reshape(-1, actions.shape[-1])
    
    return observations, actions

def write_seeds(task, all_files):
    
    seeds = []
    timestamps = []
    for file in all_files:
        timestamp = file.split("/")[-1].split(".")[0]
        timestamps.append(timestamp)
    
    for file in os.listdir(f"raw_data/{task}"):
        if not file.endswith('.json'):
            continue
        with open(f"raw_data/{task}/{file}", 'r') as f:
            data = json.load(f)
        seed = data["seed"]
        seeds.append(seed)
    
    json.dump(seeds, open(f"processed_data/{task}-{len(all_files)}-seeds.json", 'w'))

def load_relabel_data(directory, task, train_count=None, eval_count:int=0, data_path=None, seed_path=None):
    
    all_files = os.listdir(directory)
    if train_count is not None:
        assert train_count + eval_count < len(all_files), "Invalid data count"
    
    eval_files = []
    random.shuffle(all_files)
    if train_count is not None:
        all_files = all_files[:train_count]
        eval_files = all_files[:-eval_count]
    else:
        eval_files = all_files[:eval_count]
        all_files = all_files[eval_count:]
    
    if data_path is not None:
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
                all_files = data["train"]
                eval_files = data["eval"]
        else:
            data = {
                "train": all_files,
                "eval": eval_files
            }
            with open(data_path, 'w') as f:
                json.dump(data, f)
                
    if seed_path is not None:
        if os.path.exists(seed_path):
            with open(seed_path, 'r') as f:
                seeds = json.load(f)
        else:
            write_seeds(task, all_files)
    
    empty = True
    
    observations = None
    actions = None
    eval_observations = None
    eval_actions = None
    
    for file in all_files:
        if file.endswith('.npz'):
            filepath = os.path.join(directory, file)
            data = np.load(filepath)
            load_observations = data['observations']
            load_actions = data['actions']
            
            load_observations = load_observations.reshape(-1, load_observations.shape[-1])
            load_actions = load_actions.reshape(-1, load_actions.shape[-1])
            
            n = min(load_observations.shape[0], load_actions.shape[0])
            load_observations = load_observations[:n]
            load_actions = load_actions[:n]

            if empty:
                observations = load_observations
                actions = load_actions
                empty = False
            else:
                observations = np.concatenate((observations, load_observations), axis=0)
                actions = np.concatenate((actions, load_actions), axis=0)
                
    empty = True
    for file in eval_files:
        if file.endswith('.npz'):
            filepath = os.path.join(directory, file)
            data = np.load(filepath)
            load_observations = data['observations']
            load_actions = data['actions']
            
            load_observations = load_observations.reshape(-1, load_observations.shape[-1])
            load_actions = load_actions.reshape(-1, load_actions.shape[-1])
            
            n = min(load_observations.shape[0], load_actions.shape[0])
            load_observations = load_observations[:n]
            load_actions = load_actions[:n]

            if empty:
                eval_observations = load_observations
                eval_actions = load_actions
                empty = False
            else:
                eval_observations = np.concatenate((eval_observations, load_observations), axis=0)
                eval_actions = np.concatenate((eval_actions, load_actions), axis=0)
    
    return observations, actions, eval_observations, eval_actions