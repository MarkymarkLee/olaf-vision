# rl_final_project

## Environment Setup

To setup the environment, we recommend using `conda`.
Please have `conda` installed and run the following code.

```
conda create -n olaf-vision python=3.10
conda activate olaf-vision
pip install -r requirements.txt
```


## Dataset Acquisition

install dataset with `huggingface-cli download ml-jku/meta-world --local-dir=./meta-world --repo-type dataset`

## Run Visualization Webapp

Run

```
bash run_server.sh
```

or run

```
python -m hf_generator.server \
        --task="button-press-v2" \
        --model_path="models/suboptimal/button-press-v2_model.pth"
```

The 2 flags are optional, but can be used to generate data from different tasks or models.

## Run VLM Relabel

Generate traj image (from video) and next state images

```bash
python -m video_preprocess \
        -i raw_data/button-press-v2/2024-12-12T00:15:07.793517.json

# -i: Your raw data path
```

Relabeling

```bash
python -m olaf_metaworld.vlm_main \
        -i raw_data/button-press-v2/2024-12-12T00:15:07.793517.json \
        -o processed_data/button-press-v2

# -i: Your raw data path (must generate traj first)
# -o: Your processed "directory"
```

After running `vlm_main.py`, a GPT log message `prompt_message.txt` is generated.

## Acknowledgements

Metaworld Environment
