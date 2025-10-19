# Olaf Vision

This is the codebase for the project Olaf Vision. The project aims to improve the performance of imitation learning models by leveraging human feedback data.

[Link to our paper](https://github.com/MarkymarkLee/olaf-vision/blob/main/report.pdf)

[Link to our presentation](https://docs.google.com/presentation/d/1Is2y5TNdOjudLpDG65aTcfut5XUKBFXZ4WjHuHnMHRU/edit?usp=sharing)

## Environment Setup

To setup the environment, we recommend using `conda`.
Please have `conda` installed and run the following code.

```
conda create -n olaf-vision python=3.10
conda activate olaf-vision
pip install -r requirements.txt
```

## Dataset Acquisition

### IL Dataset
Install the 2M Metaworld trajectory dataset with 
```
huggingface-cli download ml-jku/meta-world --local-dir=./meta-world --repo-type dataset
```

### Experiments Data and Results
You should have `gdown` installed and install the data and results of our experiments with 
```
bash scripts/download_data.sh
```
Files are organized as follows:

`raw_data/` : raw human feedback data

`processed_data/` : relabeled human feedback data we used in the report
`processed_data_presentation_v1/` : first attempt of relabeled human feedback data for presentation (includes prompting error for VLM)
`processed_data_presentation_v2/` : second attempt of relabeled human feedback data for presentation (includes prompting error for VLM)

`improved_result_report_v1/` : first attempt of improved results for presentation using `processed_data_presentation_v1/`
`improved_result_report_v2/` : second attempt of improved results for presentation using `processed_data_presentation_v2/`

`improved_result_wandb_v1/`: improved results for presentation using `processed_data/` and added logging to wandb
`improved_result_wandb_v2/`: Final Version with every thing corrected. 

### Diffusion Model
You should have `gdown` installed. Download the diffusion model with 
```
bash scripts/download_models.sh
```

### Other Imitation Learning Models (Optional)
You can download the other imitation learning models with 
```
bash scripts/download_il_models.sh
```
The models will be put in the directory `outputs/`.
These models are not used in any of our experiments.

## Run Visualization Webapp

First, download the data needed by running
```
cd hf_generator
bash download_data.sh
cd ..
```

Then, run the following command to start the webapp
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


## Workflow of the Project

### Step 1: Training a suboptimal model
To train a suboptimal model, run the following command
```
python -m train_il.py
```
This defaults to training a suboptimal model for all the tasks in Metaworld.

Or you can use one of the pretrained models in the `models/suboptimal/` directory.


### Step 2: Generating Human Feedback Data
To generate human feedback data, first put a trained suboptimal model in the `models/suboptimal/` directory.
Then, start the webapp by running, note that you should specify the task name and model path.
```
python -m hf_generator.server \
        --task=[taskname] \
        --model_path=[modelpath]
```
After the webapp is started, you can start generating human feedback data by clicking the "Human Feedback Generator" button and following the instructions.

The generated human feedback data will be saved in the `raw_data/` directory.

### Step 3: Generate images used for relabeling
To generate images used for relabeling, you can collect image data and train your own diffusion model, by first running
```
python collect_data.py
```
Then, train the diffusion model by running
```
bash train_diffusion.sh
```
Or you can use the pretrained model by downloading it with
```
bash scripts/download_models.sh
```

After the diffusion model is trained, you can generate images needed for relabeling the human feedback trajectory by running
```
python video_preprocess.py
```
The images will be saved in the `raw_data/` directory.

### Step 4: Relabeling Human Feedback Data
To relabel human feedback data, run the following command
```
python run_relabel.py
```
The results will be saved in the `processed_data/` directory.


### Step 5: Training an Improved Model
To train an improved model, you can adjust the updating config in `retrain_il.py`, and then run the following command
```
python retrain_il.py
```
The results will be saved in the `improved_result/` directory.


## Other File Structures

`diffusion/` : code for training/inference the diffusion model

`hf_generator/` : code for generating human feedback data

`imitation_learning/` : code for training/inference the imitation learning models

`olaf_metaworld/` : code for relabeling human feedback data

`scripts/` : scripts for downloading data and models

`utils/` : utility functions

`Constants.py` : Contains predefined actions and action labels

`count.py` : Count the number of human feedback data for each task

`create_server_data.py` : Transform processed data into a format that can be used by the webapp


