import json
import os

tasks = os.listdir("raw_data")

for task in tasks:
    data = os.listdir(f"raw_data/{task}")
    data = [file for file in data if file.endswith(".json")]
    print(f"{task}: {len(data)}")
    
    good = []
    for file in data:
        with open(f"raw_data/{task}/{file}") as f:
            d = json.load(f)
        if d['feedback'] == "good":
            good.append(file)
    print(f"Good: {len(good)}")
        
