export CUDA_VISIBLE_DEVICES=2
MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR="diffusion/output/"
DATA_DIR="diffusion_data/"

accelerate launch --mixed_precision="fp16" diffusion/train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir="$DATA_DIR" \
    --original_image_column="original_image" \
    --edit_prompt_column="edit_prompt" \
    --edited_image_column="edited_image" \
    --output_dir=$OUTPUT_DIR \
    --resolution=480 \
    --random_flip \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --max_train_steps=20000 \
    --checkpointing_steps=2000 --checkpoints_total_limit=5 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42