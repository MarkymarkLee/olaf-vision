from torch import nn
import torch

class ImitationLearningModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImitationLearningModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        device = next(self.parameters()).device
        x = torch.tensor(x, dtype=torch.float32).to(device)
        if "cuda" in str(device):
            return self.forward(x).cpu().detach().numpy()
        return self.forward(x).detach().numpy()


class NewImitationLearningModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NewImitationLearningModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        device = next(self.parameters()).device
        x = torch.tensor(x, dtype=torch.float32).to(device)
        if "cuda" in str(device):
            return self.forward(x).cpu().detach().numpy()
        return self.forward(x).detach().numpy()