import numpy as np

data = np.load("../processed_data/button-press-v2/2024-12-03T13:38:58.202986.npz")
# data = np.load("../processed_data/button-press-v2/2024-12-12T00:15:07.793517.npz")
# data = np.load("../raw_data/button-press-v2/2024-12-17T15:37:19.154298.json")

obs = data["observations"]
relabeled_actions = data["actions"]
print(obs.shape, relabeled_actions.shape)


