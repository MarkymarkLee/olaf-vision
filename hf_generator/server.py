# server.py
import os
import random
from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import json
from datetime import datetime
import threading
import queue
import time

import torch
import markdown  # Add this import

from imitation_learning.model import ImitationLearningModel
from imitation_learning.eval import make_env
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import os

from Constants import ACTIONS

def write_video(images, filename, fps=30.0):
    """Write images to a video file."""
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(filename)

app = Flask(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Global variables
frame_queue = queue.Queue(maxsize=100)
is_running = False
current_session = None
env = None
observations = []
actions = []
last_step = 0
seed = 0
trajectory = []

task = None
model_path = None

loading_frame = cv2.imread(os.path.join(current_dir, 'assets', 'loading.png'))
loading_frame = cv2.imencode('.jpg', loading_frame)[1].tobytes()
stop_frame = cv2.imread(os.path.join(current_dir, 'assets', 'stop.png'))
success_frame = cv2.imread(os.path.join(current_dir, 'assets', 'success.png'))

def init():
    global frame_queue, is_running, current_session, env, observations, actions, last_step, task, trajectory
    frame_queue = queue.Queue(maxsize=100)
    is_running = False
    current_session = {}
    env = None
    observations = []
    actions = []
    trajectory = []
    last_step = 0


def reset_session():
    global current_session
    current_session = {
        'timestamp': datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f'),
        'observations': [],
        'actions': [],
        'stop_time': None,
        'feedback': None
    }

def add_stop_frame():
    global frame_queue
    try:
        frame_queue.put_nowait((stop_frame, -1))
    except queue.Full:
        frame_queue.get()  # Remove oldest frame
        frame_queue.put((stop_frame, -1))


def add_success_frame():
    global frame_queue
    try:
        frame_queue.put_nowait((success_frame, -1))
    except queue.Full:
        frame_queue.get()  # Remove oldest frame
        frame_queue.put((success_frame, -1))

def env_loop():
    print('Session started')
    
    global is_running, env, current_session, seed
    seed = random.randint(0, 1000000)
    current_session['seed'] = seed
    
    model = ImitationLearningModel(39, 4)
    model.load_state_dict(torch.load(model_path,weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    
    env = make_env(task, seed=seed)  # Example task
    obs, _ = env.reset()
    current_session['observations'].append(obs.tolist())
    
    count = 0
    success = False
    
    print('Session started')
    
    while is_running:
        # Random actions for now - replace with your model's predictions
        action = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        # Save data
        current_session['observations'].append(obs.tolist())
        current_session['actions'].append(action.tolist())
        
        # Get and process frame
        frame = env.render()
        frame = cv2.flip(frame, -1)
        frame = cv2.putText(frame, f'Step: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Put frame in queue
        try:
            frame_queue.put_nowait((frame, count))
        except queue.Full:
            frame_queue.get()  # Remove oldest frame
            frame_queue.put((frame, count))
        
        success = info['success'] > 0.5
        if done or truncated or success:
            obs, _ = env.reset()
            count = 0
            is_running = False
        
        count += 1
            
        time.sleep(1/30)  # 30 FPS
    
    print('Session stopped')
    if success:
        add_success_frame()
        add_success_frame()
        add_success_frame()
    elif count >= 498:
        add_stop_frame()
        add_stop_frame()
        add_stop_frame()
    
    current_session['stop_time'] = datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f')

def generate_frames():
    global last_step, trajectory
    while frame_queue.empty():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + loading_frame + b'\r\n')
        time.sleep(1/30)
    while True:
        frame, last_step = frame_queue.get()
        trajectory.append(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/hf-generator')
def hf_generator():
    return render_template('hf_generator.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global is_running
    init()
    if not is_running:
        is_running = True
        reset_session()
        threading.Thread(target=env_loop, daemon=True).start()
    return jsonify({'status': 'started'})

@app.route('/stop')
def stop():
    global is_running, current_session
    is_running = False
    current_session['stop_time'] = datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f')
    return jsonify({'status': 'stopped'})

@app.route('/feedback', methods=['POST'])
def feedback():
    global current_session
    data = request.get_json()
    current_session['feedback'] = data['feedback']
    current_session['feedback_step'] = last_step
    current_session['trajectory_video'] = f'raw_data/{task}/videos/{current_session["timestamp"]}.mp4'
    current_session['observations'] = current_session['observations']
    current_session['actions'] = current_session['actions']
    
    write_video(trajectory, current_session['trajectory_video'])
    
    # Save session to file
    filepath = f'raw_data/{task}/{current_session["timestamp"]}.json'
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(filepath, 'w') as f:
        json.dump(current_session, f, indent=4)
    
    return jsonify({'status': 'feedback_saved'})

@app.route('/test')
def test():
    return {'status': 'ok'}

@app.route('/compare/<task>/<index>')
def compare(task: str, index: int):
    index = int(index)
    print(task, index)
    # Load the session data
    data_file = os.path.join(current_dir,f'static/{task}/data.json')
    if not os.path.exists(data_file):
        return jsonify({'error': 'Session not found'}), 404
    
    with open(data_file, 'r') as f:
        data = json.load(f)

    data_len = len(data)
    
    data = data[index]
    
    response_file = data['responses']
    with open(os.path.join(current_dir, response_file), 'r') as f:
        responses = json.load(f)
    
    vlm_choice = [v * 5 for v in data['vlm_choice']]
    chosen_image = ACTIONS.index(vlm_choice)
    
    def cycle(index):
        return (index + data_len) % data_len
    
    session_data = {
        'video': data['trajectory_video'],
        'traj_image': data['traj_image'],
        'feedback': markdown.markdown(data['feedback']),  # Convert Markdown to HTML
        'olaf': markdown.markdown(responses['olaf']),  # Convert Markdown to HTML
        'olaf_action': data['olaf_choice'],  # Convert Markdown to HTML
        'vlm': markdown.markdown(responses['vlm']),  # Convert Markdown to HTML
        'vlm_action': data['vlm_choice'],  # Convert Markdown to HTML
        'chosen_image': chosen_image,
        'next_state_images': data['next_action_images'],
        'task': task,
        'index': index,
        'prev_index': str(cycle(index - 1)),
        'next_index': str(cycle(index + 1)),
    }
    
    return render_template('compare.html', data=session_data)


def parse_arg():
    import argparse
    parser = argparse.ArgumentParser(description='Run the server')
    parser.add_argument('--task', type=str, default='button-press-v2', help='Task name')
    parser.add_argument('--model_path', type=str, default='suboptimal', help='Model path')
    args = parser.parse_args()
    global task
    task = args.task
    global model_path
    model_path = args.model_path
    if model_path == 'suboptimal':
        model_path = f'models/suboptimal/{task}_model.pth'

if __name__ == '__main__':
    parse_arg()
    app.run(debug=True, port=6275)