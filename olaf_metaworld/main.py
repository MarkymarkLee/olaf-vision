from olaf_metaworld.lib.LLM import LLMCritic
from olaf_metaworld.lib.args import parse_args
from dotenv import load_dotenv
import os
import json
import numpy as np

def test_env():
    import metaworld
    import random

    task_name = "button-press-v2"
    ml1 = metaworld.ML1(task_name) # Construct the benchmark, sampling tasks

    env = ml1.train_classes[task_name]()  # Create an environment with task `pick_place`
    task = random.choice(ml1.train_tasks)
    env.set_task(task)  # Set task

    obs, _ = env.reset()  # Reset environment
    a = env.action_space.sample()  # Sample an action
    obs, _, _, _, _ = env.step(a)  # Step
    
    # a = env.action_space.sample()  # Sample an action
    # obs_1, _, _, _, _ = env.step(a)  # Step
    
    
    # a = env.action_space.sample()  # Sample an action
    # obs_2, _, _, _, _ = env.step(a)  # Step
    # return obs, obs_1, obs_2
    return obs

def read_obs(path):
    ## TODO: Read state, action and return the last stopped state
    
    with open(path, 'r') as file:
        obs_data: dict = json.load(file)
        
    return obs_data
    
def relabel_action(trajectory, original_actions, new_action, path):
    ## TODO
    # Relabel the past 20-30 states with the new action 
    # Save the whole trajectory into .npz file
    
    refined_actions = []
    if len(original_actions) > 30:
        refined_actions = original_actions[:-30] + [new_action] * 30
    else:
        refined_actions = [new_action] * len(original_actions)
    
    np.savez_compressed(path, observations=trajectory, actions=refined_actions)
    pass
    

def olaf_generate_relabeled_action_data(obs_path, processed_dirname):
    load_dotenv()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    llm = LLMCritic(api_key=api_key)
    
    # obs_path = '../raw_data/button-press-v2/2024-12-03T13:38:58.202986.json'
    obs_data = read_obs(obs_path)
    print("Read observation data successfully")
    
    # Get all information from JSON data
    traj = obs_data.get("observations", [])
    obs = np.array(traj[-1])
    
    feedback: str = obs_data.get("feedback", "")
    timestamp = obs_data.get("timestamp", "")
    
    processed_path = processed_dirname + '/' + timestamp + '.npz'
    
    if feedback.strip() == "good":
        print("Feedback is good. No need to relabel.")
        np.savez_compressed(processed_path, observations=traj, actions=obs_data.get("actions", []))
        return
    
    from Constants import NUMPY_ACTIONS
    action_candidates = NUMPY_ACTIONS * 0.2
    
    # raise
    print("Generating LLM response...")
    result_path = processed_dirname + '/' + timestamp + '_chat.txt'
    result_action = llm.generate_action(action_candidates, obs, feedback, result_path=result_path)
    print("Generated Action from LLM: {}".format(result_action))
    
    # raise
    original_action = obs_data.get("actions", [])
    # processed_path = '../processed_data/button-press-v2/{}.npz'.format(timestamp)
    processed_path = processed_dirname + '/' + timestamp + '.npz'
    dirname = os.path.dirname(processed_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    relabel_action(traj, original_action, result_action, processed_path)
    print("Successfully saved refined action labels into {}".format(processed_path))
    