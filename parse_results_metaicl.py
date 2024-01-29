import os
import glob
import json
from collections import defaultdict
import argparse
from path_config import SAVEPATH

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default="all", help="training dataset name")
parser.add_argument('-f',
                    '--folder',
                    type=str,
                    default='',
                    help="subfolder of dataset folder (e.g. '' or 'finetune')")
parser.add_argument('-p', '--pattern', type=str, help="pattern to match in adapter name")
args = parser.parse_args()

base_path = os.path.join(SAVEPATH, args.dataset)
if args.folder != '':
    base_path += f'/{args.folder}'
files = sorted(os.listdir(base_path))

# Parse results from trainer_state
filename = 'trainer_state.json'
results_dict_state = {}
for file in files:
    res = defaultdict(list)
    if args.pattern is not None:
        if args.pattern not in file:
            continue
    if file == 'test' or file == 'debug':
        continue

    if not os.path.isfile(os.path.join(base_path, file, filename)):
        continue

    with open(os.path.join(base_path, file, filename), 'r') as f:
        states = json.load(f)
        max_step = states["max_steps"]
        eval_results = states["log_history"]

    for d in eval_results:
        if d["step"] != max_step:
            continue

        eval_dict = False
        for key in d.keys():
            if "eval" in key:
                eval_dict = True
            else:
                if "train_loss" in key:
                    res['train_loss'].append(d["train_loss"])

        if not eval_dict:
            continue

        for key in d.keys():
            if key.endswith('rougeL'):
                res['rougeL'].append(d[key])
            if key.endswith('loss'):
                res['test_loss'].append(d[key])

    results = {k: sum(v) / len(v) for k, v in res.items()}
    results_dict_state[file] = results

# Parse results from eval_results
filename = 'eval_results.json'
results_dict = {}
for file in files:
    res = defaultdict(list)
    if args.pattern is not None:
        if args.pattern not in file:
            continue
    if file == 'test' or file == 'debug':
        continue

    if not os.path.isfile(os.path.join(base_path, file, filename)):
        continue

    with open(os.path.join(base_path, file, filename), 'r') as f:
        eval_results = json.load(f)

    for key in eval_results.keys():
        if key.endswith('acc'):
            task = key[:-len('acc')]

            try:
                res['train_loss'].append(results_dict_state[file]['train_loss'])
            except:
                res['train_loss'].append(0)

            try:
                res['rougeL'].append(eval_results[task + 'rougeL'])
            except:
                res['rougeL'].append(results_dict_state[file]['rougeL'])

            try:
                res['test_loss'].append(eval_results[task + 'loss'])
            except:
                res['test_loss'].append(results_dict_state[file]['test_loss'])

            if task.startswith("poem_sentiment"):
                # Classification with P(class|Input) / P(class)
                res['acc'].append(eval_results[task + 'acc_base'])
            elif task.startswith("glue-mrpc"):
                # This dataset has too high evaluation variances to be used for evaluation
                continue
            else:
                # Classification by token averaged loss
                res['acc'].append(eval_results[task + 'acc_norm'])

    n = len(res['rougeL'])
    if n == 0:
        continue

    results = {}
    results['acc'] = sum(res['acc']) / len(res['acc'])
    results['rougeL'] = sum(res['rougeL']) / n
    results['train_loss'] = sum(res['train_loss']) / len(res['train_loss'])
    results['test_loss'] = sum(res['test_loss']) / n
    results_dict[file] = results

# Print results
model = results_dict.keys()
result_keys = ['acc', 'rougeL', 'train_loss', 'test_loss']
sep = ''
for name in sorted(list(model)):
    print(f"Model: {name}")
    for key in result_keys:
        print(f"\t{key:20s}:", end=' ')
        try:
            print(f"{results_dict[name][key]:.2f}{sep}", end=' ')
        except:
            print(f"N/A{sep}", end=' ')
        print()
    print()
