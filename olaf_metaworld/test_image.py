import base64
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Path to your image
image_path1 = "img/0.png"
image_path2 = "img/pikachu.png"

# Getting the base64 string
base64_image1 = encode_image(image_path1)
base64_image2 = encode_image(image_path2)

image_prompts = [
    {
        "type": "image_url",
        "image_url": {
            "url":  f"data:image/jpeg;base64,{base64_image}",
            "detail": "low"
        },
    }
    for base64_image in [base64_image1, base64_image2]
]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is in first and second image?",
            },
            *image_prompts
        ],
    }
]
# messages = [
#     {"role": "user", "content": "Hi!"}
# ]

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

data = {
    "model": 'gpt-4o-mini',
    "temperature": 0,
    "messages": messages,
}

print("Generating response from an input image...")
with requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data) as res:
    response = res.json()
    
with open("test_image.json", "w") as file:
    json.dump(response, file)
    print("Generated response dumped into test_image.json!")
    
    