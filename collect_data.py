import numpy as np
from utils.collect_data import collect_data
from imitation_learning.model import ImitationLearningModel
import torch
import metaworld

all_tasks = metaworld.ML1.ENV_NAMES
for task in all_tasks:
    print(task)
    model_path = f"outputs/{task}/model.pth"
    model = ImitationLearningModel(39,4)
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()
    collect_data(
        task_name=task,
        model=model,
        count=100
    )


