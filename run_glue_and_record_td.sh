
# ====================== Recording Training Dynamics: ===========
# `--nproc_per_node N` means the number of GPUs to use
# Suggested models: 
# prajjwal1/bert-tiny
# distilbert-base-cased
# bert-base-cased
# roberta-large


export TASK_NAME=sst2
export MODEL=prajjwal1/bert-tiny
python -m torch.distributed.launch --nproc_per_node 8 --use_env run_glue.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --do_recording \
