import base64
from openai import OpenAI
import dotenv
import os

class GPT_agent:
    def __init__(self) -> None:
        dotenv.load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def get_image_content(self, image_path):
        base64_image = self.encode_image(image_path)
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                },
            },
        ]
    
    def inference(self, system_prompt, user_prompt, image_paths):
        
        content = [
            {
                "type": "text",
                "text": user_prompt,
            },
        ]
        
        for image_path in image_paths:
            content += self.get_image_content(image_path)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": content }
            ],
        )
        
        return response.choices[0].message.content
