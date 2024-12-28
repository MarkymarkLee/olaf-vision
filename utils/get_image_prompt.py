from Constants import ACTIONS, ACTION_PROMPTS
import numpy as np

def get_edit_prompt(action):
    action = np.array(action)
    for i, a in enumerate(ACTIONS):
        a = np.array(a)
        if (a == action).all():
            return ACTION_PROMPTS[i]
    