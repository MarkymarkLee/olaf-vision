export CUDA_VISIBLE_DEVICES=""

python -m hf_generator.server \
        --task="button-press-v2" \
        --model_path="models/suboptimal/button-press-v2_model.pth"