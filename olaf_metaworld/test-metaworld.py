import metaworld
import random

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

task_name = "button-press-v2"
ml1 = metaworld.ML1(task_name) # Construct the benchmark, sampling tasks

env = ml1.train_classes[task_name]()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
print(env.action_space)
done = False
while not done:
    a = env.action_space.sample()  # Sample an action
    obs, reward, done, truncated, info = env.step(a)  # Step the environment with the sampled random action
    print(obs)



