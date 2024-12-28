import numpy as np
import requests
import json
from olaf_metaworld.lib.utils import encode_image_url
import olaf_metaworld.lib.prompts as system_prompts
import olaf_metaworld.lib.summary as summary_prompts
from Constants import NUMPY_ACTIONS

class VLMCritic():
    def __init__(self, *args, **kwargs):
        # LLM settings
        self.__api_key = kwargs.get('api_key', '')
        # Options: gpt-4o, llava
        self.model = kwargs.get('model_name', 'gpt-4o')
        self.temperature = kwargs.get('temperature', 0)
        
        self.__headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key}"
        }
        
        # Other settings
        self._task_name = "button-press-v2"
    
    def _process_obs(self, obs_path):
        return encode_image_url(obs_path)
    
    def _process_language(self, language_correction):
        return system_prompts.prompt_language_correction_vlm + language_correction
        
    def _generate_prompt(self, action_candidates: list, obs_path: str, language_correction=None):
        system_prompt = system_prompts.system_prompt_visual_behavior

        if language_correction is not None:
            system_prompt = system_prompts.system_prompt_visual_behavior_human
        
        # User prompt Contains:
        # 1. Instruction: General instruction (depending on the traj image and generate some useful words)
        # 2. Traj image
        # 3. Human Feedback (or LLM feedback)
        # 4. next state prediction (multiple images)
        
        # process state obs
        obs_processed = self._process_obs(obs_path)
        next_state_prompt = [
            {
                "type": "image_url",
                "image_url": {
                    "url":  self._process_obs(candidate),
                    "detail": "low"
                },
            } for candidate in action_candidates]
        
        # process user feedback
        if language_correction is not None:
            language_prompt = self._process_language(language_correction)
        
        user_prompt = [
            {
                "type": "text",
                "text": system_prompts.prompt_visual_instruction
            },
            {
                "type": "image_url",
                "image_url": {
                    "url":  obs_processed,
                    "detail": "low"
                },
            },
            {
                "type": "text",
                "text": language_prompt
            },
            *next_state_prompt
        ]

        return system_prompt, user_prompt

    def user_prompt_string(self, user_prompt):
        prompt_string = ""
        for prompt in user_prompt:
            if prompt["type"] == "text":
                prompt_string += prompt["text"] + "\n"
            elif prompt["type"] == "image_url":
                prompt_string += f"\nImage\n"
        return prompt_string
    
    def generate_action(self, next_state_prediction, obs, feedback=None, result_path=None):
        system_prompt, user_prompt = self._generate_prompt(next_state_prediction, obs, feedback)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # First generate CoT response
        response_cot = self.generate(messages)
        
        summary_prompt = summary_prompts.summary_prompt_return_image_and_action
        summary_prompt_correction = summary_prompts.summary_correction_return_action
        
        messages.extend([
            {"role": "assistant", "content": response_cot},
            {"role": "user", "content": summary_prompt}
        ])
        
        # Then summarize and extract the final action
        response = self.generate(messages, response_json=True)
        print(response)
        response = json.loads(response)
        response_image = response["image"]
        response_action = response["action"]
        
        response_action = list(NUMPY_ACTIONS[response_image-1] * 0.2)
        
        user_prompt_string = self.user_prompt_string(user_prompt)
        
        if result_path is not None:
            with open(result_path, 'w') as file:
                file.write("SYSTEM PROMPT\n{}\n\nUSER PROMPT\n{}\n\n".format(system_prompt, user_prompt_string))
                file.write("RESPONSE\n{}\n\nFINAL IMAGE\n{}\n\nFINAL ACTION\n{}\n".format(response_cot, str(response_image), str(response_action)))
        
        return response_action
    
    def generate(self, messages, response_json=False):
        data = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        
        if response_json:
            data["response_format"] = {
                "type": "json_object"
            }
        
        response_text = ""
        with requests.post("https://api.openai.com/v1/chat/completions", headers=self.__headers, json=data) as res:
            if res.status_code == 200:
                response_text = res.json()['choices'][0]['message']['content']
            else:
                raise Exception(f"Status Code {res.status_code}: {res.reason}")
        
        return response_text
        