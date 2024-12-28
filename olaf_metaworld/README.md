# OLAF - Metaworld version

## Usage

Remember to place OPENAI KEY in `.env` file

### LLM

```bash
python main.py -i <your raw_data_path> -o <your process data directory name>

# Example
# python main.py -i ../raw_data/button-press-v2/2024-12-03T13:38:58.202986.json -o ../processed_data/button-press-v2
# Output file will be stored as '../processed_data/button-press-v2/2024-12-03T13:38:58.202986.npz'

# After execution, there will be a prompt message log saved as 'prompt_message.txt'
```

### VLM version

```bash
python vlm_main.py -i <your raw_data_path> -o <your process data directory name>

# Example
# python main.py -i ../raw_data/button-press-v2/2024-12-03T13:38:58.202986.json -o ../processed_data/button-press-v2
# Output file will be stored as '../processed_data/button-press-v2/2024-12-03T13:38:58.202986.npz'

# After execution, there will be a prompt message log saved as 'prompt_message.txt'
```
