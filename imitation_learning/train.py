import os
from imitation_learning.dataset import load_data
from imitation_learning.eval import eval_env
from imitation_learning.model import ImitationLearningModel
from imitation_learning.config import TrainingConfig

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
import metaworld
import json
import numpy as np
from torch.optim.lr_scheduler import StepLR

class Logger:
    def __init__(self, file_path):
        self.file_path = file_path
        file = open(file_path, "w")
        file.write("Training Log\n")
        file.close()
    
    def info(self, message):
        with open(self.file_path, "a") as file:
            file.write(message + "\n")

def train_il_agent(config: TrainingConfig):
    save_dir = config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    task_name = config.task_name
    task_names = metaworld.ML1.ENV_NAMES
    assert task_name in task_names, f"{task_name} not in Dataset"
    
    # save config in directory
    config_json = config.model_dump()
    json.dump(config_json, open(os.path.join(save_dir, "config.json"), "w"), indent=4)
    
    observations, actions = load_data(task_name)
    # Convert to PyTorch tensors
    observations = torch.tensor(observations, dtype=torch.float32).to(config.device)
    actions = torch.tensor(actions, dtype=torch.float32).to(config.device)

    # Create a dataset
    dataset = TensorDataset(observations, actions)
    dataset_size = int(len(dataset) * config.dataset_percentage)
    remain_size = len(dataset) - dataset_size
    dataset, _ = random_split(dataset, [dataset_size, remain_size])
    
    
    training_size = config.training_size_per_epoch
    if config.training_size_per_epoch < 1:
        training_size = int(config.training_size_per_epoch * dataset_size)
    
    # Split the dataset into training and evaluation sets (80% train, 20% eval)
    train_size = int(0.995 * dataset_size)
    eval_size = dataset_size - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    
    # Create data loaders
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    
    model = ImitationLearningModel(observations.shape[1], actions.shape[1])
    model.to(config.device)
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
    
    logger = Logger(os.path.join(save_dir, "train.log"))
    
    print(f"Training model on task {task_name}")
    logger.info(f"Training model on task {task_name}")
    
    # Train the model
    count = 0
    for epoch in tqdm(range(config.epochs)):
        model.train()
        train_loss = 0
        
        subset_indices = np.random.choice(train_size, size=training_size, replace=False)
        train_sampler = SubsetRandomSampler(subset_indices)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
        
        for obs_batch, act_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(obs_batch)
            loss = criterion(predictions, act_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Step the scheduler
        scheduler.step()
        
        # Evaluate the model
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for obs_batch, act_batch in eval_loader:
                predictions = model(obs_batch)
                loss = criterion(predictions, act_batch)
                eval_loss += loss.item()
        eval_loss /= len(eval_loader)
        logger.info(f"Epoch {epoch+1}/{config.epochs},\tTrain Loss: {train_loss:.6f},\tEval Loss: {eval_loss:.6f}")
        
        if (epoch+1) % config.env_eval_freq == 0:
            if not os.path.exists(os.path.join(save_dir, "trajectories")):
                os.makedirs(os.path.join(save_dir, "trajectories"))
            if not os.path.exists(os.path.join(save_dir, "models")):
                os.makedirs(os.path.join(save_dir, "models"))
            trajectory_path = os.path.join(save_dir, "trajectories/", f"trajectory_{epoch+1}.mp4")
            eval_env(model, task_name, trajectory_path)
            torch.save(model.state_dict(), os.path.join(save_dir, "models", f"model_epoch{epoch+1}.pth"))
    
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    
    return model