import os
from shutil import copyfile
import json
import numpy as np
from Constants import NUMPY_ACTIONS

ACTIONS = list(NUMPY_ACTIONS*0.2)

def get_choice(method, task, data_time):
    data = np.load(f"processed_data/{task}/{method}/{data_time}.npz")
    actions = data["actions"]
    action = actions[-1]
    for act in ACTIONS:
        if np.array_equal(act, action):
            return list(action)
    return "No change"

def generate_data(task):
                    
    data_paths = []

    for data_file in os.listdir(f"raw_data/{task}"):
        if data_file.endswith(".json"):
            data_paths.append(f"raw_data/{task}/{data_file}")
    
    root_dir = f"static/{task}"
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(f"{root_dir}/videos", exist_ok=True)
    os.makedirs(f"{root_dir}/images", exist_ok=True)
    os.makedirs(f"{root_dir}/responses", exist_ok=True)

    new_data = []
    for index, data_path in enumerate(data_paths):
        with open(data_path, "r") as f:
            data = json.load(f)
            if data['feedback'] == 'good':
                continue
            timestamp = data["timestamp"]
            olaf_choice = get_choice("olaf", task, timestamp)
            vlm_choice = get_choice("vlm", task, timestamp)
            copyfile(data["trajectory_video"], f"{root_dir}/videos/{index}_traj.mp4")
            copyfile(data["traj_image"], f"{root_dir}/images/{index}_traj.png")
            
            next_action_images = []
            for i, next_image in enumerate(data["next_action_images"]):
                copyfile(next_image, f"{root_dir}/images/{index}_next_{i}.png")
                next_action_images.append(f"{root_dir}/images/{index}_next_{i}.png")
            
            responses = {}
            with open(f"processed_data/{task}/olaf/{timestamp}_chat.txt", "r") as f:
                responses["olaf"] = f.read()
            with open(f"processed_data/{task}/vlm/{timestamp}_chat.txt", "r") as f:
                responses["vlm"] = f.read()
            
            with open(f"{root_dir}/responses/{index}_responses.json", "w") as f:
                json.dump(responses, f)
            
            new_data.append({
                "timestamp": timestamp,
                "olaf_choice": olaf_choice,
                "vlm_choice": vlm_choice,
                "trajectory_video": f"{root_dir}/videos/{index}_traj.mp4",
                "traj_image": f"{root_dir}/images/{index}_traj.png",
                "next_action_images": next_action_images,
                "feedback": data["feedback"],
                "responses": f"{root_dir}/responses/{index}_responses.json"
            })
            
    with open(f"{root_dir}/data.json", "w") as f:
        json.dump(new_data, f)
        
generate_data("button-press-v2")
generate_data("button-press-topdown-v2")
os.system("zip -r static.zip static/")
os.system("rm -rf static/")