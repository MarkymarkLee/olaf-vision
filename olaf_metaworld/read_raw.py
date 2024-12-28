import json
import numpy as np


# filepath = "../raw_data/button-press-v2/2024-12-03T13:38:58.202986.json"
# filepath = "../raw_data/button-press-v2/2024-12-12T00:15:07.793517.json"
filepath = "../raw_data/button-press-v2/2024-12-17T15:39:08.068669.json"

with open(filepath, 'r') as file:
    res = json.load(file)
    
obs = np.array(res["observations"])
actions = np.array(res["actions"])
print(obs.shape, actions.shape)
    