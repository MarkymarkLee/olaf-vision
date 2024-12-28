## TODO: process traj video into single traj image
## TODO: process next state images by diffusion model using traj image

import argparse
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from diffusion.inference import DiffusionModel
from Constants import ACTION_PROMPTS


def read_obs(path):    
    with open(path, 'r') as file:
        obs_data: dict = json.load(file)
    return obs_data


def read_video(video_path: str):
    import cv2

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    images = []
    # Loop through frames
    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break  # Break the loop if no more frames

        # Convert the frame (NumPy array) from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a Pillow image
        pil_image = Image.fromarray(frame_rgb)

        # Process the frame with Pillow (e.g., show or save the image)
        # pil_image.show()  # Display the image (closes when you press any key)
        # pil_image.save('frame.png')  # Save the image if needed
        images.append(pil_image)

    # Release the video capture object
    cap.release()
    return images

def combine_traj(images: list):
    # Last image: 0.6
    # Second image: 0.4 * 0.6
    coeff_rem = 1
    img_cnt = 0
    combined_image_np = None
    while img_cnt < min(30, len(images)):
    # while img_cnt < 1:
        coeff = coeff_rem * 0.6
        cur_image = images[-img_cnt-1]
        image_np = np.array(cur_image).astype(np.float32)
        if combined_image_np is None:
            combined_image_np = image_np * coeff
        else:
            combined_image_np += image_np * coeff
        coeff_rem -= coeff
        img_cnt += 1
    
    combined_image_np = combined_image_np.astype(np.uint8)
    combined_image = Image.fromarray(combined_image_np)
    return combined_image

def generate_diffusion_img():
    pass

def process_video(obs_path, model: DiffusionModel):
    obs_data = read_obs(obs_path)
    
    print("Read observation data successfully")
    
    #### Generate trajectory image
    traj_video_path = obs_data.get("trajectory_video", "")
    print(traj_video_path)
    
    traj_images = read_video(traj_video_path)
    traj_image = combine_traj(traj_images)
    print("Traj image generated successfully")
    
    #### Save traj image
    
    input_dir = os.path.dirname(obs_path) + '/images/' + os.path.basename(obs_path).replace('.json', '')
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    
    if len(os.listdir(input_dir)) >= 9:
        print("Already processed")
        return
    
    print(input_dir)
    traj_image.save(input_dir + '/traj.png')
    obs_data["traj_image"] = input_dir + '/traj.png'
    
    
    #### Generate next state images
    next_images = list()
    
    traj_image_np = np.array(traj_image).astype(np.float32)
    for prompt in ACTION_PROMPTS:
        next_image = model.predict_next_image(obs_data["traj_image"], prompt)
        # Weighted addition of images
        
        # Convert images to numpy arrays
        next_image_np = np.array(next_image).astype(np.float32)
        
        combined_image_np = (1 * traj_image_np + 4 * next_image_np) / 5
        combined_image_np = combined_image_np.astype(np.uint8)

        # Convert back to PIL image
        combined_image = Image.fromarray(combined_image_np)
        next_images.append(combined_image)
        # combined_image.save("0_edited.png")

    print("Next state images generated successfully")

    #### Save next state images
    for i, next_image in enumerate(next_images):
        next_image.save(f"{input_dir}/action_{i}.png")
    
    obs_data["next_action_images"] = [
        f"{input_dir}/action_{i}.png" for i in range(len(next_images))
    ]
    
    with open(obs_path, 'w') as file:
        json.dump(obs_data, file)

    print("Updated traj img path & next state img paths into JSON file")
    
if __name__ == '__main__':
    raw_data_dir = 'raw_data'
    filenames = []
    for root, _, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith('.json'):
                filenames.append(os.path.join(root, file))
    filenames = sorted(filenames)
    
    print(filenames)
    
    model = DiffusionModel("models/diffusion")
    
    for filename in tqdm(filenames):
        print(filename)
        process_video(filename, model)
    print("Video processed successfully")
