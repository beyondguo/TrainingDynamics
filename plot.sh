# task_name, model 这俩参数，只是用于展示在plot上
# model_dir 是放 training_dynamics 文件夹的那个目录
# plots_dir 是绘制好的 plot 存放的目录
# burn_out 计算多少轮的 dynamics

# bert-tiny
# distilbert-base-cased
# roberta-large

export TASK_NAME=sst2
export MODEL=bert-tiny
python -m dy_filtering \
    --plot \
    --task_name sst2 \
    --model_dir dy_log/$TASK_NAME/$MODEL \
    --plots_dir dy_log/$TASK_NAME/$MODEL \
    --model $MODEL \
    --burn_out 4

