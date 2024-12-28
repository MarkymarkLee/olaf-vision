import json
import os
import random

from tqdm import tqdm
from olaf_metaworld.main import olaf_generate_relabeled_action_data
from olaf_metaworld.vlm_main import vlm_generate_relabeled_action_data

def get_hf_files(task):
    hf_data = os.listdir(f"raw_data/{task}")
    hf_data = [f"raw_data/{task}/{file}" for file in hf_data if file.endswith(".json")]
    return hf_data

def write_env_seeds(task, hf_data):
    
    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")
    
    seeds = []
    
    for file in hf_data[:30]:
        with open(file, 'r') as f:
            data = json.load(f)
        seed = data["seed"]
        seeds.append(seed)
    
    json.dump(seeds, open(f"processed_data/{task}-30-seeds.json", 'w'))
        
    for file in hf_data[30:]:
        with open(file, 'r') as f:
            data = json.load(f)
        seed = data["seed"]
        seeds.append(seed)
    
    json.dump(seeds, open(f"processed_data/{task}-60-seeds.json", 'w'))

if __name__ == '__main__':
    
    tasks = ["button-press-topdown-v2", "button-press-v2"]
    
    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")
    
    for task in tasks:
        
        hf_data = get_hf_files(task)
        
        print("OLAF Processing", task)
        for file in tqdm(hf_data):
            if not os.path.exists(f"processed_data/{task}/olaf"):
                os.makedirs(f"processed_data/{task}/olaf")
            with open(file, 'r') as f:
                data = json.load(f)
            timestamp = data["timestamp"]
            if os.path.exists(f"processed_data/{task}/olaf/{timestamp}.npz"):
                continue
            olaf_generate_relabeled_action_data(file, f"processed_data/{task}/olaf")
        
        print("VLM Processing", task)
        for file in tqdm(hf_data):
            if not os.path.exists(f"processed_data/{task}/vlm"):
                os.makedirs(f"processed_data/{task}/vlm")
            with open(file, 'r') as f:
                data = json.load(f)
            timestamp = data["timestamp"]
            if os.path.exists(f"processed_data/{task}/vlm/{timestamp}.npz"):
                continue
            vlm_generate_relabeled_action_data(file, f"processed_data/{task}/vlm")

