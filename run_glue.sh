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


# ====================== Training with Data Selection: =============
# after you run `data_selection.py` and obtain the `three_regions_data_indices.json` file
# you can train a GLUE classifier again with your specified data selection
# set `--with_data_selection` to turn on data selection
# set `--data_selection_region [region]` to specify the region, choices are "easy", "hard", and "ambiguous"


# ====================== Suggested models: ======================
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
  --with_data_selection \
  --data_selection_region easy \
  # --output_dir tmp/$TASK_NAME/