from pathlib import Path
import json
import os
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from tqdm.notebook import tqdm

TGT_COLS = ['task_name', 'from_pretrained', 'model_cfg', 'model_cls', 'model_type', 'lr', 'batch_size', 'HVD_SIZE', 'lr_scheduler', 'input_seq_len', 'model_path', 'num_steps']
TGT_COLS += ['drop_neighborhood', 'drop_description', 'cpt_path']
TGT_COLS += ['task_name', 'train_path', 'valid_path', 'test_path', 'index_path', 'inference_entities_path']

SILENT = True
def parse_tensorboard(path, scalars, silent=SILENT):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    # assert all(
    #     s in ea.Tags()["scalars"] for s in scalars
    # ),"" if silent else "some scalars were not found in the event accumulator"
    found_scalars = [s for s in scalars if s in ea.Tags()['scalars']]
    return {k: pd.DataFrame(ea.Scalars(k)) for k in found_scalars}


def parse_to_csv(path, out_path, target_cols, metric_names, silent=SILENT):
    path = Path(path)

    logs = list(path.glob('**/*tfevents*'))

    experiments = []
    for p in tqdm(logs):
        config_path = os.path.join(p.parent,  'config.json')
        if not os.path.exists(config_path):
            continue

        expr = json.load(open(config_path, 'r'))
        metrics = {}
        try:
            metrics = parse_tensorboard(str(p), [f'{m}/iterations/valid' for m in metric_names])
        except Exception as e:
            if not silent:
                print(f'error: {e}\n\tskip: {p}')
        try:
            metrics_test = parse_tensorboard(str(p), [f'{m}/iterations/test' for m in metric_names])
        except Exception as e:
            metrics_test = {}
            if not silent:
                print(f'error: {e}\n\t no test metrics in: {p}')
        metrics.update(metrics_test)

        if len(metrics) == 0:
            continue
        for m in metric_names:
            if f'{m}/iterations/test' in metrics:
                expr[m] = metrics[f'{m}/iterations/test']['value'].item()
            
            if f'{m}/iterations/valid' in metrics:
                expr[f'best_valid_{m}'] = metrics[f'{m}/iterations/valid']['value'].max()
            else:
                if not silent:
                    print(f"Warning: no validation metrics in {p.parent}")
                continue

        # print(parse_tensorboard(str(p), ['loss/iterations/train'])['loss/iterations/train'].step)
        # try:
        #     expr['num_steps'] = parse_tensorboard(str(p), ['loss/iterations/train'])['loss/iterations/train'].step.max()
        # except AssertionError:
        #     continue
        experiments += [expr]

    experiments = pd.DataFrame(experiments)
    # print('\n\ncolumns: ', experiments.columns)
    
    not_found_cols = [col for col in target_cols if col not in experiments.columns]
    if not_found_cols:
        if not silent:
            print(f'{not_found_cols} not found in columns!!\ncolumns:{experiments.columns}')
    
    found_cols = [col for col in target_cols if col in experiments.columns]
    experiments = experiments[found_cols]
    # print('\n\ncolumns: ', experiments.columns)
    # raise(StopIteration)

    experiments.to_csv(out_path, index=False)

    

path = Path('/home/bulatov/bulatov/KGLM/runs/')
metric_names = ['exact_match', 'exact_match_entity']
metric_names += ['Hits@1', 'Hits@3', 'Hits@5', 'Hits@10', 'Hits@1_pipeline', 'Hits@3_pipeline', 'Hits@5_pipeline', 'Hits@10_pipeline']
target_cols = TGT_COLS + metric_names
out_path = 'results/kglm.csv'

parse_to_csv(path, out_path, target_cols, metric_names)