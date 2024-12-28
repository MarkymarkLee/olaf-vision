summary_prompt = """
Now based on the previous response, summarize what is the final action choice. 
Return the answer as a JSON object, with a single key 'action', and a single value which is a number. 
Do not return any other string besides the json object. For example, if the action is 3, return {'action': 3}
If the text have multiple results for the correct action, you must only return one of them. Do not return multiple answers!
"""

summary_correction = """
This is incorrect format. You should return the answer as single JSON object, with a single key 'action', and the value should be a single number! 
If the text have multiple results for the correct action, you must only return one of them. Do not return multiple answers! Please try again.
"""

summary_prompt_return_action = """
Now based on the previous response, summarize what is the final action choice. 
Return the answer as a JSON object, with a single key 'action', and a single list. The value of JSON object must be a list of 4 numbers.
Do not return any other string besides the json object. 

For example, if the action is [0.2, 0, 0, 0], please return {'action': [0.2, 0, 0, 0]}.
If the action is [0, 0.2, 0, 0], please return {'action': [0, 0.2, 0, 0]}.
If the action is [0, 0, 0.2, 0], please return {'action': [0, 0, 0.2, 0]}.
If the action is [0, 0, 0, 0.2], please return {'action': [0, 0, 0, 0.2]}.
If the action is [-0.1, 0.22, 0.005, -0.1], please return {'action': [-0.1, 0.22, 0.005, -0.1]}.
"""

summary_prompt_return_image_and_action = """
Now based on the previous response, summarize what is the final image and action choice. 
Return the answer as a JSON object, with a single key 'action', and a single list. The value of JSON object must be a list of 4 numbers.
Do not return any other string besides the json object. 

For example, 
if the selected image is Image 3 and the action is [0.2, 0, 0, 0], 
please return {'image': 3, 'action': [0.2, 0, 0, 0]}.
"""

summary_correction_return_action = """
This is incorrect format. You should return the answer as single JSON object, with a single key 'action', and the value should be a single list! Please try again.
"""

summary_prompt_vision_return_action = """
Now based on the previous response, summarize what is the final action choice. 
Return the answer as a JSON object, with a single key 'action', and a single list. The value of JSON object must be a list of 4 numbers.
Do not return any other string besides the json object.

Please follow the instructions below. 

For example, if the action is [0.2, 0, 0, 0], please return {'action': [0.2, 0, 0, 0]}.
If the action is [0, 0.2, 0, 0], please return {'action': [0, 0.2, 0, 0]}.
If the action is [0, 0, 0.2, 0], please return {'action': [0, 0, 0.2, 0]}.
If the action is [0, 0, 0, 0.2], please return {'action': [0, 0, 0, 0.2]}.
If the action is [-0.1, 0.22, 0.005, -0.1], please return {'action': [-0.1, 0.22, 0.005, -0.1]}.
"""
