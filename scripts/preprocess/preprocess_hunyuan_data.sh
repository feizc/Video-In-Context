# export WANDB_MODE="offline"
GPU_NUM=4 # 2,4,8
MODEL_PATH="ckpts"
MODEL_TYPE="hunyuan"
DATA_MERGE_PATH="data/people/merge.txt"
OUTPUT_DIR="data/people-cache"
VALIDATION_PATH="assets/prompt.txt"

ADD_PATH="$(pwd)"
PATH="${ADD_PATH}:${PATH}"
export PATH

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/data_preprocess/preprocess_vae_latents.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --train_batch_size=1 \
    --max_height=480 \
    --max_width=848 \
    --num_frames=93 \
    --dataloader_num_workers 1 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 24 

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/data_preprocess/preprocess_text_embeddings.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR 

torchrun --nproc_per_node=1 \
    fastvideo/data_preprocess/preprocess_validation_text_embeddings.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR \
    --validation_prompt_txt $VALIDATION_PATH