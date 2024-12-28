import os
import json
import shutil  # Add this import

import numpy as np

from utils.get_image_prompt import get_edit_prompt


def process_image_data(main_dir='image_data'):
    save_dir = f"diffusion_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dirs = os.listdir(f'{main_dir}/')
    all_dirs = []
    for dir in dirs:
        cur_dirs = os.listdir(f'{main_dir}/{dir}')
        all_dirs += [f"{dir}/{cur_dir}" for cur_dir in cur_dirs]
    print(all_dirs)
    dirs = all_dirs

    processed_data = {
        'original_image': [],
        'edit_prompt': [],
        'edited_image': [],
    }
    
    if os.path.exists(f"{save_dir}/data.json"):
        with open(f"{save_dir}/data.json") as f:
            processed_data = json.load(f)

    image_rename = {}
    image_count = max(len(os.listdir(save_dir)) - 1, 0)

    def update_image_rename(image_path):
        
        nonlocal image_count, image_rename
        
        if image_path not in image_rename:
            image_rename[image_path] = f"{save_dir}/{image_count}.png"
            image_count += 1

            # Copy the image to the new location with the new name
            old_image_path = f"{image_path}"
            new_image_path = image_rename[image_path]
            print(old_image_path, new_image_path)
            shutil.copyfile(old_image_path, new_image_path)

            return image_rename[image_path]
        else:
            return image_rename[image_path]

    for dir in dirs:
        dir_path = f'{main_dir}/{dir}'
        data_file = f"{dir_path}/data.json"
        data = json.load(open(data_file))

        for i, use_action in enumerate(data['use_action']):
            if use_action:
                original_image = data['observations_images'][i]
                original_image = update_image_rename(original_image)

                action = data['actions'][i]
                edit_prompt = get_edit_prompt(action)

                edited_image = data['observations_images'][i + 1]
                edited_image = update_image_rename(edited_image)

                processed_data['original_image'].append(original_image)
                processed_data['edit_prompt'].append(edit_prompt)
                processed_data['edited_image'].append(edited_image)

    with open(f"{save_dir}/data.json", "w") as f:
        json.dump(processed_data, f)
        
    shutil.rmtree(main_dir)
    print(f"Removed directory: {main_dir}")
