import os
from imitation_learning.dataset import load_relabel_data
from imitation_learning.eval import eval_env, eval_multiple_envs
from imitation_learning.model import ImitationLearningModel
from imitation_learning.config import UpdatingConfig

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
import metaworld
import json
import numpy as np
  # Add this line

class Logger:
    def __init__(self, file_path):
        self.file_path = file_path
        file = open(file_path, "w")
        file.write("")
        file.close()
    
    def info(self, message):
        with open(self.file_path, "a") as file:
            file.write(message + "\n")

class Plotter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.x = []
        self.y = []
    
    def add_point(self, x, y):
        self.x.append(x)
        self.y.append(y)
    
    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.x, self.y)
        plt.savefig(self.file_path)
        plt.close()

def update_il_agent(config: UpdatingConfig) -> ImitationLearningModel:
    torch.manual_seed(0)
    save_dir = config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    task_name = config.task_name
    task_names = metaworld.ML1.ENV_NAMES
    assert task_name in task_names, f"{task_name} not in Dataset"
    
    # save config in directory
    config_json = config.model_dump()
    json.dump(config_json, open(os.path.join(save_dir, "config.json"), "w"), indent=4)
    
    if config.use_wandb:
        import wandb
        wandb.init(project="RL_Final", id=config.wandb_id, config=config_json)
    
    observations, actions = load_relabel_data(config.relabel_dir)
    print(observations.shape)
    print(actions.shape)
    # Convert to PyTorch tensors
    observations = torch.tensor(observations, dtype=torch.float32).to(config.device)
    actions = torch.tensor(actions, dtype=torch.float32).to(config.device)

    # Create a dataset
    dataset = TensorDataset(observations, actions)
    dataset_size = int(len(dataset) * config.dataset_percentage)
    remain_size = len(dataset) - dataset_size
    # print(remain_size)
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
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    
    model = ImitationLearningModel(observations.shape[1], actions.shape[1])
    model.load_state_dict(torch.load(config.subopt_path, weights_only=True))
    model.to(config.device)
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    logger = Logger(os.path.join(save_dir, "train.log"))
    train_loss_plotter = Plotter(os.path.join(save_dir, "train_plot.png"))
    eval_loss_plotter = Plotter(os.path.join(save_dir, "eval_plot.png"))
    eval_logger = Logger(os.path.join(save_dir, "eval.log"))
    eval_plotter = Plotter(os.path.join(save_dir, "eval_plot.png"))
    # score = eval_multiple_envs(model, task_name, 10)
    # eval_logger.info(f"Epoch 0/{config.epochs},\tScore: {score}")
    # eval_plotter.add_point(0, score)
    
    print(f"Training model on task {task_name}")
    logger.info(f"Training model on task {task_name}")
    
    # Train the model
    count = 0
    for epoch in tqdm(range(config.epochs)):
        model.train()
        train_loss = 0
        
        for obs_batch, act_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(obs_batch)
            loss = criterion(predictions, act_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
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
        train_loss_plotter.add_point(epoch, train_loss)
        eval_loss_plotter.add_point(epoch, eval_loss)
        
        if config.use_wandb:
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "eval_loss": eval_loss})
        
        if (epoch+1) % config.env_eval_freq == 0:
            if not os.path.exists(os.path.join(save_dir, "trajectories")):
                os.makedirs(os.path.join(save_dir, "trajectories"))
            if not os.path.exists(os.path.join(save_dir, "models")):
                os.makedirs(os.path.join(save_dir, "models"))
            
            score = eval_multiple_envs(model, task_name, config.eval_episodes)
            eval_logger.info(f"Epoch {epoch+1}/{config.epochs},\tScore: {score}")
            eval_plotter.add_point(epoch, score)
            if config.use_wandb:
                wandb.log({"epoch": epoch+1, "score": score})
            
            torch.save(model.state_dict(), os.path.join(save_dir, "models", f"model_epoch{epoch+1}.pth"))
    
    if config.use_wandb:
        wandb.finish()
    
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    
    return model