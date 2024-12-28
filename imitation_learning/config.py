from pydantic import BaseModel

class TrainingConfig(BaseModel):
    task_name: str
    save_dir: str

    dataset_percentage: float
    
    # Training parameters
    batch_size: int
    training_size_per_epoch: int | float
    epochs: int
    lr: float
    lr_step_size: int
    lr_gamma: float
    
    # Evaluation parameters
    env_eval_freq: int

    # Device parameters
    device: str
    
    # Logging parameters
    log_freq: int

class UpdatingConfig(BaseModel):
    task_name: str
    save_dir: str
    subopt_path: str
    relabel_dir: str

    dataset_percentage: float
    
    # Training parameters
    batch_size: int
    training_size_per_epoch: int | float
    epochs: int
    lr: float
    
    # Evaluation parameters
    env_eval_freq: int
    eval_episodes: int

    # Device parameters
    device: str
    
    # Logging parameters
    log_freq: int
    use_wandb: bool  # Add this line
    wandb_id: str  # Add this line
