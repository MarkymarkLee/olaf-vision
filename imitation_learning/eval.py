import metaworld
import random
import cv2
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import os
from stable_baselines3.common.vec_env import DummyVecEnv

def write_video(images, filename, fps=30.0):
    """Write images to a video file."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(filename)

def make_env(env_name, seed=None):
    render_mode = 'rgb_array'  # set a render mode
    camera_name = 'corner'  # one of: ['corner', 'corner2', 'corner3', 'topview', 'behindGripper', 'gripperPOV']
    ml1 = metaworld.ML1(env_name, seed=seed)  # Construct the benchmark, sampling tasks
    env = ml1.train_classes[env_name](
        render_mode=render_mode, camera_name=camera_name,
    )  # Construct the environment
    task = random.choice(ml1.train_tasks)
    env.set_task(task)  # Set task
    return env

def make_env_fn(env_name, seed=None):
    def _init():
        render_mode = 'rgb_array'  # set a render mode
        camera_name = 'corner'  # one of: ['corner', 'corner2', 'corner3', 'topview', 'behindGripper', 'gripperPOV']
        ml1 = metaworld.ML1(env_name, seed=seed)  # Construct the benchmark, sampling tasks
        env = ml1.train_classes[env_name](
            render_mode=render_mode, camera_name=camera_name,
        )  # Construct the environment
        task = random.choice(ml1.train_tasks)
        env.set_task(task)  # Set task
        return env
    return _init

def eval_env(model, task_name: str, trajectory_path: str|None = None, seed: None | int = None) -> float:
    env = make_env(task_name, seed=seed)
    obs, info = env.reset()
    done = False
    truncate = False
    total_reward = 0
    trajectory = []
    success = False
    while not done and not truncate:
        action = model.predict(obs)
        obs, reward, done, truncate, info = env.step(action)
        total_reward += reward
        success = info['success']
        success = (success > 0.5)
        if success:
            print("Success!")
            break
        img = env.render()
        img = cv2.flip(img, -1)
        trajectory.append(img)
    env.close()
    if trajectory_path:
        write_video(trajectory, trajectory_path)
    return success

def eval_multiple_envs(model, task_name: str, count: int, seeds: list[int] | None = None) -> list[float]:
    if seeds is None:
        seeds = [None] * count
    assert len(seeds) == count
    
    env_fns = [make_env_fn(task_name, seed) for seed in seeds]
    envs = DummyVecEnv(env_fns)
    
    obs = envs.reset()
    dones = [False] * count
    total_rewards = [0] * count
    trajectories = [[] for _ in range(count)]
    successes = [False] * count
    success_count = 0
    succeeded = []
    
    while not all(dones):
        actions = model.predict(obs)
        obs, rewards, dones, infos = envs.step(actions)
        for i in range(count):
            total_rewards[i] += rewards[i]
            successes[i] = infos[i]['success'] > 0.5
            if successes[i] and i not in succeeded:
                # print(f"Success in environment {i}!")
                success_count += 1
                succeeded.append(i)     
    
    envs.close()
    
    return success_count / count


