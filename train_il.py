from imitation_learning.config import TrainingConfig
from imitation_learning.eval import eval_env
from imitation_learning.train import train_il_agent

import torch
import metaworld

all_tasks = metaworld.ML1.ENV_NAMES
button_tasks = [task for task in all_tasks if "button" == task[:6]]
print(button_tasks)
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(task_name):
    print(f"Training task {task_name} on device:{device}")

    config = TrainingConfig(
        task_name=task_name,
        save_dir=f"outputs_2/{task_name}",
        dataset_percentage=1,
        
        batch_size=64,
        training_size_per_epoch=0.05,
        epochs=200,
        lr=0.001,
        lr_step_size=50,
        lr_gamma=0.3,
        
        env_eval_freq=10,
        
        device=device,
        log_freq=10,
    )

    model = train_il_agent(config)

    eval_env(model, task_name, f"outputs/{task_name}/trajectory.mp4")


for task_name in all_tasks:
    train(task_name)

print("All tasks trained!")
    