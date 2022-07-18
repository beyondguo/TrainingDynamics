# Only applied to training set
# python data_selection.py --task_name qnli --model_name bert-base-cased --proportion 0.5 --burn_out 4
import json
import random
random.seed(1)
import argparse
from dy_filtering import read_training_dynamics, compute_train_dy_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--proportion", type=float, default=0.33)
parser.add_argument("--burn_out", type=int)
args = parser.parse_args()

TASK_NAME = args.task_name
MODEL = args.model_name
PROPORTION = args.proportion

# 读取并合并到一个文件
td = read_training_dynamics(f'dy_log/{TASK_NAME}/{MODEL}/')
# 计算 metrics，转化成一个 dataframe
td_df, _ = compute_train_dy_metrics(td, burn_out=args.burn_out)


def consider_ascending_order(filtering_metric: str) -> bool:
    """
    Determine if the metric values' sorting order to get the most `valuable` examples for training.
    """
    if filtering_metric == "variability":
        return False
    elif filtering_metric == "confidence":
        return True
    elif filtering_metric == "threshold_closeness":
        return False
    elif filtering_metric == "forgetfulness":
        return False
    elif filtering_metric == "correctness":
        return True
    else:
        raise NotImplementedError(f"Filtering based on {filtering_metric} not implemented!")



def data_selection(metric, select_worst, proportion, shuffle=True):
    ascending = consider_ascending_order(metric)
    if select_worst:
        ascending = not consider_ascending_order(metric)
    sorted_df = td_df.sort_values(by=metric, ascending=ascending)
    selected_df = sorted_df.head(n=int(proportion * len(sorted_df)))
    indices = list(selected_df['guid'])
    if shuffle:
        random.shuffle(indices)
    return {'indices':indices, 'df':selected_df}


"""
hard-to-learn: METRIC = 'confidence'
easy-to-learn: METRIC = 'confidence', SELECT_WORST = True
ambiguoug: METRIC = 'variability'
"""

three_regions_data_indices = {'hard':data_selection('confidence', False, PROPORTION)['indices'],
                              'easy':data_selection('confidence', True, PROPORTION)['indices'],
                              'ambiguous':data_selection('variability', False, PROPORTION)['indices']}

with open(f'dy_log/{TASK_NAME}/{MODEL}/three_regions_data_indices.json','w') as f:
    f.write(json.dumps(three_regions_data_indices))

# 然后可以直接跑glue任务，在选择训练集的时候，使用select函数来指定对应样本即可：
""" e.g.
from datasets import load_dataset
raw_datasets = load_dataset('glue','sst2')
easy_train_set = raw_datasets['train'].select(three_regions_data_indices['easy'])
"""
