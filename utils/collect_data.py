import json
import os
import time
from typing import List
import cv2
import numpy as np
import random

from imitation_learning.model import ImitationLearningModel
from imitation_learning.eval import make_env

from Constants import ACTIONS, ACTION_PROMPTS
actions = [(np.array(action), prompt) for action, prompt in zip(ACTIONS, ACTION_PROMPTS)]

def save_env_image(env, filename):
    img = env.render()
    img = cv2.flip(img, -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
    
def get_current_images_count(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 0
    return len(os.listdir(directory))

def step_action(env, action, count) -> bool:
    should_stop = False
    truncate = False
    for _ in range(count):
        obs, reward, done, truncate, info = env.step(action)
        should_stop = info['success'] > 0.5 or done
        if truncate:
            break
    return should_stop, truncate


def collect_data_once(task_name: str, model: ImitationLearningModel, save_dir: str, epsilon: float = 0.1):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if not os.path.exists(f"{save_dir}/images"):
        os.makedirs(f"{save_dir}/images")
    image_count = get_current_images_count(f"{save_dir}/images")
    
    data = {
        "original_image": [],
        "edit_prompt": [],
        "edited_image": []
    }
    if os.path.exists(f"{save_dir}/data.json"):
        with open(f"{save_dir}/data.json", "r") as f:
            data = json.load(f)
    
    env = make_env(task_name)
    obs, info = env.reset()
    done = False
    truncate = False
    success = False
    data_count = 0
    while not done and not truncate:
        if np.random.rand() > epsilon:
            cur_action = model.predict(obs)
            use_action = 0
            obs, reward, done, truncate, info = env.step(cur_action)
            success = info['success']
            success = (success > 0.5)
            if success:
                print("Success!")
                break
            continue
        
        else:
            save_env_image(env, f"{save_dir}/images/{image_count}.png")
            data["original_image"].append(f"{save_dir}/images/{image_count}.png")
            image_count += 1
            
            chosen_action, action_prompt = random.choice(actions)
            data["edit_prompt"].append(action_prompt)
            should_stop, truncate = step_action(env, chosen_action, 10)
            
            save_env_image(env, f"{save_dir}/images/{image_count}.png")
            data["edited_image"].append(f"{save_dir}/images/{image_count}.png")
            image_count += 1
            
            if truncate:
                os.remove(data["original_image"][-1])
                os.remove(data["edited_image"][-1])
                image_count -= 2
                data["edit_prompt"].pop()
                data["edited_image"].pop()
                data["original_image"].pop()
                break
            
            data_count += 1
            
            if should_stop:
                break
            
    env.close()
    
    with open(f"{save_dir}/data.json", "w") as f:
        json.dump(data, f)
    
    print(f"Data collected: {data_count}")
    print(f"Total data collected: {len(data['original_image'])}")
    
    return data_count


def collect_data(task_name: str, model: ImitationLearningModel, count: int):
    total_data = 0
    while total_data < count:
        data_count = collect_data_once(task_name, model, "diffusion_data", epsilon=0.4)
        total_data += data_count
    print("Data collection complete")