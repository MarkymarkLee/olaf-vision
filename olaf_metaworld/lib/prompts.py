# the "initial" system prompt to simulate chatGPT behavior.
system_prompt_behavior = """
You are a helpful assistant who is good at employing math and computer science tools to arrive at the solution. 
You analyze numerical values carefully and think step by step.
"""

system_prompt_behavior_human = """
You are a helpful assistant who is good at employing math and computer science tools to arrive at the solution. 
You analyze numerical values carefully and think step by step. 
You will also pay close attention to the human language correction, interpret the human intention, and use it to arrive at the solution.
Please describe in detail how you apply your mathematical and computational abilities, to arrive at solutions.
"""

system_prompt_visual_behavior = """
You are a helpful assistant who is good at analysing images related to a real world scene and computer science tools to arrive at the solution.
You analyse the relative distance of the scene and objects carefully and think step by step.
"""

system_prompt_visual_behavior_human = """
You are a helpful assistant who is good at analysing images related to a real world scene and computer science tools to arrive at the solution.
You analyse the relative distance of the scene and objects carefully and think step by step.
You will also pay close attention to the human language correction, interpret the human intention, and use it to arrive at the solution.
Please describe in detail how you apply your visual and computational abilities, to arrive at solutions.
"""

# 4 DOF (x, y, z, hand)
prompt_robot = """
You have a robot arm which is the Sawyer, a single robot arm with 7 degrees of freedom.
The robot a parallel gripper equipped with two small finger pads, that comes shipped with the robot arm.
The robot comes with a controller that takes in actions. 

Due to the Metaworld environment restriction, its action space only supports 4 degrees of freedom.
The expected action space of the controller is (dx, dy, dz, dgrip).
The manual reads like the following: 
( dx,  0,  0,     0)     <-- Translation in x-direction (left/right)         
(  0, dy,  0,     0)     <-- Translation in y-direction (forward/backward) 
(  0,  0, dz,     0)     <-- Translation in z-direction (up/down)
(  0,  0,  0, dgrip)     <-- Normalized torque change for the gripper fingers

The value of the dx, dy, dz, dgrip are all continuous between -1 and 1.
Below is the list of values that will influence the robot behavior:

| Action     | Value Change |
| ---------- | ------------ |
| left       | dx < 0       |
| right      | dx > 0       |
| forward    | dy > 0       |
| backward   | dy < 0       |
| up         | dz > 0       |
| down       | dz < 0       |
| grip close | dgrip > 0    |
| grip open  | dgrip < 0    |

"""

prompt_visual_instruction = """
You are given a metaworld scene, where there is a robot arm called Sawyer. It has 4 degrees of freedom (x, y, z, grip).

The images will be given as the following:

1. There will be a single image that shows past trajectory of the robot. 
Please analyze this image and decide a task and a goal for this scene.
2. There will be 8 other images, specifying the next state prediction images.
The robot will be moving or controlling its grippers in these images.
Please analyze each image thoroughly, and specify them in your reasoning.
The lightened arm is current state, and the colored arm is predicted next state.

At last, you will need to determine which next state images is the most appropriate according to past trajectories and 
user feedback (if exists).
Please output the order of image (eg. Image 2) and output the desired direction with a 4-degree action tuple (dx, dy, dz, dgrip).
(Normally, the robot moves 0.1-0.3 in a frame/30fps. You must strictly output the action values between them).
"""

prompt_task_square = """
In this task, the robot must pick a square nut and place it on a rod. The nut has a handle to be grasped.

The task has the following stages:

1. Grasping the Handle: Approach the square nut's handle. The robot will move closer to the square nut handle and the distance between robot position
and handle position will be closer. The robot will grasp the nut by its handle, according to the angles of the handle (roll, pitch, yaw). The robot will need to move to the correct angles (roll, pitch, yaw) and perform the grasp action.

2. Peg Insertion: Lift the nut and get closer to the peg and aligning with its angles (roll, pitch, yaw) at the same time. The distance between robot position will be closer. It then inserts the nut in the peg by moving the nut down the peg.

Here are some example input, and the stage they correspond to:

    Example 1:
    
        Input:

        Information relevant to grasping the handle:
        Robot Position: [-3 16 91]
        Robot Angles: [ 3 -3 44]
        Handle Position: [-8 17 83]
        Handle Angles: [ 0  0 54]
        Gripper State: [-100]

        Information relevant to peg insertion:
        Nut Position: [-11  13  83]
        Nut Angles: [ 0  0 54]
        Peg Position: [23 10 85]
        Peg Angles: [0 0 0]
        Gripper State: [-100]

        Stage: Grasping the Handle
    
    Example 2:
    
        Input:

        Information relevant to grasping the handle:
        Robot Position: [24  2 97]
        Robot Angles: [  4   0 -89]
        Handle Position: [24  4 97]
        Handle Angles: [ -5  -5 -90]
        Gripper State: [100]

        Information relevant to peg insertion:
        Nut Position: [24  9 96]
        Nut Angles: [17 16 12]
        Peg Position: [23 10 85]
        Peg Angles: [0 0 0]
        Gripper State: [100]

        Stage: Peg Insertion

You are given the state information, which include:
1. robot end effector position
2. robot end effector angle in roll, pitch, yaw axis 
3. handle position 
4. handle angle in roll, pitch, yaw axis
5. nut position 
6. nut angle in roll, pitch, yaw axis 
7. peg position 
8. peg angle in roll, pitch, yaw axis 
9. gripper status (100 for closed, -100 for open)

"""

prompt_task_press = """
This is a button pressing task. The robot will find the position of the button, and then press the button.
The button will be placed along y-axis. The goal of the robot is to press the button as hard as it could.
Each episode, the button positions are randomly decided.

You are given the state information, which include:
1. robot end effector 3D Cartesian position
2. Normalized measurement of how open the robot gripper is
3. First object (button) 3D position
"""

prompt_task = {
    "square": prompt_task_square,
    "button-press-v2": prompt_task_press
}

prompt_instruction = """
Your task is that, given a few choices of actions to perform at the current state, you will choose the correct action for the robot to perform.

Note on the position:
You should consider the position of the robot end effector and object, and how they are related to each other.
For example, if the robot end effector is on the left of the object, you should consider moving the robot end effector to the right.

Note on the gripper:
The robot's gripper should be closed if it is beginning to grasp the object, or when it is holding the object. 
When it is approaching the object, the gripper is open.
If the robot gripper needs to be closed, you should continue to close the gripper, even if it is closed.
Similarly, if the robot gripper needs to be open, you should continue to open the gripper, even if it is already open.
"""

prompt_instruction_cot = """

Given the robot and object position, first explain what stage is the task currently in, and what is the relationship between the robot and object. Explain what a good action is supposed to do.
Then based on your result, look at the given actions, and return which of the following actions is the correct action to take.

Let's think step by step.
Explaining your reasoning before arriving at the solution. 

You always produce a single Action value in the end, which is a single number. You must follow this format!
If there are multiple actions, you must only return one of them.
"""

prompt_instruction_cot_return_action = """
Given the robot and object position, first explain what stage is the task currently in, and what is the relationship between the robot and object. Explain what a good action is supposed to do.
Then based on your result, return a correct action to take on the current state in the format of [dx, dy, dz, dgrip] as mentioned above. The action value should be in the appropriate action scale (between -1 and 1).

Let's think step by step.
Explaining your reasoning before arriving at the solution. 

You always produce an action being in a list of length 4. You must follow this format! You must follow this format!

"""

prompt_instruction_cot_edit_action = """
Given the robot and object position, first explain what stage is the task currently in, and what is the relationship between the robot and object. 

Explain what a good action is supposed to do.

Based on your result, identify the action dimension indices that requires modification. 

Then modify the original action in these action dimension indices in the appropriate action scale (between -1 and 1).

Finally, return a correct action to take on the current state in the format of [dx, dy, dz, dgrip] as mentioned above.

Let's think step by step.
Explaining your reasoning before arriving at the solution. 

You always produce an action being in a list of length 4. You must follow this format! You must follow this format!

"""

prompt_instruction_cot_combine_onedim = """

Given the robot and object position, first explain what stage is the task currently in, and what is the relationship between the robot and object. 
Explain what a good action is supposed to do. 
You are also given 8 actions, that can move on different axis. 
You can combine the 8 actions together to generate a new action. 
You will output the final action to take given these actions.

Let's think step by step.
Explaining your reasoning before arriving at the solution. 
"""

prompt_instruction_cot_backup = """

Follow the instructions below to complete the task:

# 1. Given the robot and object position, first identify which stage of the task the robot is in, based on the information above.

# 2. Then explain what is the relationship between the robot and object. Explain what a good action is supposed to do here.

# 3. Then based your result, look at the given actions, and return which of the following two actions is the correct action to take.

Then based your result, look at the given actions, and return which of the following two actions is the correct action to take.

"""

prompt_language_correction = """
You also receive the following human language correction at the current state. Pay close attention to the human language correction,
 interpret the human intention, and use it to arrive at the solution.

Some pointers for human language correction interpretation:

Move backward: decrease the y position
Move forward: increase the y position
Move left: decrease the x position
Move right: increase the x position
Move up: increase the z position
Move down: decrease the z position
Close the gripper: increase the grip value
Open the gripper: decrease the grip value

Human language correction:

"""

prompt_language_correction_vlm = """
You also receive the following human language correction at the current state. Pay close attention to the human language correction,
 interpret the human intention, and use it to arrive at the solution.

Human language correction:

"""


template_obs_square = """
    Input: 
    
    Information relevant to grasping the handle:
    Robot Position: {}
    Robot Angles: {}
    Handle Position: {}
    Handle Angles: {}
    Gripper State: {}
    
    Information relevant to peg insertion:
    Nut Position: {}
    Nut Angles: {}
    Peg Position: {}
    Peg Angles: {}
    Gripper State: {}
    """
    
template_obs_press = """
    Input:
    
    Information relevant to robot:
    Hand Position: {}
    Gripper Value: {}
    
    Information relevant to first object (button):
    Button Position: {}
    
    """

template_obs = {
    "square": template_obs_square,
    "button-press-v2": template_obs_press
}