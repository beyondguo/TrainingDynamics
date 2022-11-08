# ====================== Basic GLUE tasks training and evaluation:
# the following is the standard training for GLUE tasks

# export TASK_NAME=sst2
# export MODEL=distilbert-base-cased
# python -m torch.distributed.launch --nproc_per_node 8 --use_env run_glue.py \
#   --model_name_or_path $MODEL \
#   --task_name $TASK_NAME \
#   --max_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 5 \


# ————> Suggested models: 
# prajjwal1/bert-tiny
# distilbert-base-cased
# bert-base-cased
# roberta-large

# =========== Train GLUE tasks =============
# https://huggingface.co/datasets/glue

export TASK_NAME=rte
export MODEL=bert-base-cased
# python -m torch.distributed.launch --nproc_per_node 8 --use_env run_glue_hct.py \
CUDA_VISIBLE_DEVICES=5 python run_glue_hct.py \
  --seed 5 \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 4e-5 \
  --num_train_epochs 10 \
  --temperature 1\
  --mu 0.5 \
  --more_ambiguous
  # --hard_with_ls \
  # --ls_weight 0.1 \
  # --hard_inference
  
  # --with_data_selection \
  # --data_selection_region ambiguous \
  # --output_dir tmp/$TASK_NAME/

