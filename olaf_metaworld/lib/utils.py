import numpy as np
import base64

# encoding image
def encode_image_url(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    return f"data:image/jpeg;base64,{base64_image}"


# formatting
def format_value(arr, angle=False, convert_int=False):            
    if angle:
        #arr = arr[-1]
        arr = np.where(np.sign(arr) == 0, 0.0, arr)
        arr = np.int8(arr)
        return str(arr)
        
    arr = arr.round(2)
    arr = np.where(np.sign(arr) == 0, 0.0, arr)
    if convert_int:
        arr = np.int8(arr * 100)
    return str(arr)

def format_obs_square(one_step, template_obs):
    robot_pos = one_step['robot_pos']
    robot_angle = one_step['robot_angle']
    
    handle_pos = one_step['handle_pos']
    handle_angle = one_step['handle_angle']
    
    nut_pos = one_step['nut_pos']
    nut_angle = one_step['nut_angle']
    
    peg_pos = one_step['peg_pos']
    peg_angle = one_step['peg_angle']
    
    gripper_state = one_step['gripper_state']
    
    
    template = template_obs.format(
        format_value(robot_pos),
        format_value(robot_angle, angle=True),
        
        format_value(handle_pos),
        format_value(handle_angle, angle=True),        

        format_value(gripper_state),
        
        
        format_value(nut_pos),
        format_value(nut_angle, angle=True),
        
        format_value(peg_pos),
        format_value(peg_angle, angle=True),
        
        format_value(gripper_state),
    )

    return template

def format_obs_press(one_step, template_obs):
    hand_pos = one_step["hand_pos"]
    hand_closed = one_step["hand_closed"]
    button_pos = one_step["button_pos"]    
    
    template = template_obs.format(
        format_value(hand_pos),
        format_value(hand_closed),
        
        format_value(button_pos),
    )

    return template

def format_obs(one_step, template_obs, task):
    if task == "square":
        return format_obs_square(one_step, template_obs)
    return format_obs_press(one_step, template_obs)
    
def format_action(action_candidates):
    template_action = """
    Action Choices:         
    """
    num_actions = len(action_candidates)
    for i in range(num_actions):
        a = """
    Action {}: {}
    """.format(
            i,
            format_value(action_candidates[i])
        )
        template_action += a

    return template_action