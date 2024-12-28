import os
from threading import Thread
from imitation_learning.config import UpdatingConfig
from imitation_learning.eval import eval_env
from imitation_learning.update import update_il_agent

import torch
import metaworld

# all_tasks = metaworld.ML1.ENV_NAMES
# button_tasks = [task for task in all_tasks if "button" == task[:6]]
# print(button_tasks)
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(task_name, data_count, relabel_method="olaf"):
    
    assert relabel_method in ["olaf", "vlm"], "Invalid relabel method"
    
    print(f"Training task {task_name}-{relabel_method}-{data_count} on device:{device}")
    
    relabel_dir = f"processed_data/{task_name}/{relabel_method}"
    
    config = UpdatingConfig(
        task_name=task_name,
        save_dir=f"improved_result/{task_name}/{relabel_method}-{data_count}",
        subopt_path=f"models/suboptimal/{task_name}_model.pth",
        relabel_dir=relabel_dir,
        dataset_percentage=1,
        
        batch_size=64,
        training_size_per_epoch=1,
        epochs=1000,
        lr=0.0001,
        train_count=data_count,
        eval_count=10,
        seed_path=f"processed_data/{task_name}-{data_count}-seeds.json",
        data_files_path=f"processed_data/{task_name}-{data_count}-files.json",
        
        env_eval_freq=20,
        eval_episodes=20,
        
        device=device,
        log_freq=10,
        use_wandb=True,
        wandb_id=f"{task_name}-{relabel_method}-{data_count}-good"
    )

    model = update_il_agent(config)

    return model

if __name__ == '__main__':
    tasks = ["button-press-v2", "button-press-topdown-v2"]
    
    for task in tasks:
        for data_count in [30,60,100]:
            for relabel_method in ["olaf", "vlm"]:
                train(task, data_count, relabel_method)