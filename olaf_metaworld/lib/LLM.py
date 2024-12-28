import requests
import json
import olaf_metaworld.lib.prompts as system_prompts
import olaf_metaworld.lib.summary as summary_prompts
from olaf_metaworld.lib.utils import *

class LLMCritic():
    def __init__(self, *args, **kwargs):
        # LLM settings
        self.__api_key = kwargs.get('api_key', '')
        self.model = kwargs.get('model_name', 'gpt-4o')
        self.temperature = kwargs.get('temperature', 0)
        
        self.__headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key}"
        }
        
        # Other settings
        self._task_name = "button-press-v2"
    
    def _process_obs(self, obs_i):
        return {
            "hand_pos": obs_i[:3],
            "hand_closed": obs_i[3],
            "button_pos": obs_i[4:7],
            "unused_info": obs_i[7:],
        }
    
    def _process_language(self, language_correction):
        return system_prompts.prompt_language_correction + language_correction
        
    def _generate_prompt(self, action_candidates, obs, language_correction=None):
        system_prompt = system_prompts.system_prompt_behavior

        if language_correction is not None:
            system_prompt = system_prompts.system_prompt_behavior_human
        
        user_prompt = system_prompts.prompt_robot \
                    + system_prompts.prompt_task[self._task_name] \
                    + system_prompts.prompt_instruction \
                    + system_prompts.prompt_instruction_cot
                    # TODO: + system_prompts.prompt_instruction_cot_combine_onedim

        # process state obs
        obs_processed = self._process_obs(obs)
        obs_string = format_obs(obs_processed, system_prompts.template_obs[self._task_name], task=self._task_name)
        
        # process action candidates
        action_string = format_action(action_candidates)

        user_prompt = user_prompt + obs_string + action_string

        # process user feedback
        if language_correction is not None:
            language_prompt = self._process_language(language_correction)
            user_prompt = user_prompt + language_prompt

        return system_prompt, user_prompt
    
    def generate_action(self, action_candidates, obs, feedback=None, result_path=None):
        system_prompt, user_prompt = self._generate_prompt(action_candidates, obs, feedback)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # First generate CoT response
        response_cot = self.generate(messages)
        
        summary_prompt = summary_prompts.summary_prompt_return_action
        summary_prompt_correction = summary_prompts.summary_correction_return_action
        
        messages.extend([
            {"role": "assistant", "content": response_cot},
            {"role": "user", "content": summary_prompt}
        ])
        
        # Then summarize and extract the final action
        response_action = self.generate(messages, response_json=True)
        response_action = json.loads(response_action)["action"]
        
        if result_path is not None:
            with open(result_path, 'w') as file:
                file.write("SYSTEM PROMPT\n{}\n\nUSER PROMPT\n{}\n\n".format(system_prompt, user_prompt))
                file.write("RESPONSE\n{}\n\nFINAL ACTION\n{}".format(response_cot, str(response_action)))
        
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

class LLMAgent():
    def __init__(self, *args, **kwargs):
        # LLM settings
        self.__api_key = kwargs.get('api_key', '')
        self.model = kwargs.get('model_name', 'gpt-4o-mini')
        self.temperature = kwargs.get('temperature', 0)
        
        self.__headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key}"
        }
        
    def generate(self, messages):
        data = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        response_text = ""
        with requests.post("https://api.openai.com/v1/chat/completions", headers=self.__headers, json=data) as res:
            if res.status_code == 200:
                response_text = res.json()['choices'][0]['message']['content']
            else:
                raise Exception(f"Status Code {res.status_code}: {res.reason}")
            
        return response_text
    
        