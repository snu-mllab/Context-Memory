import argparse
import os
from path_config import SAVEPATH, CACHEDIR


def tf_str(tf):
    s = "true" if tf else "false"
    return s


def run(args):

    base_cmd = f"python -B -m src.inference"

    # Config file
    model = args.model
    base_cmd = f"{base_cmd} +{args.dataset}={model}"
    base_cmd = f"{base_cmd} training.do_train=false wandb.log=false"

    # Compression type
    attn_type = None
    if "-merge-" in args.eval_path:
        attn_type = "merge"
    elif "-merge_recur-" in args.eval_path:
        attn_type = "merge_recur"
    elif "-concat-" in args.eval_path:
        attn_type = "concat"
    elif "-concat_recur-" in args.eval_path:
        attn_type = "concat_recur"
    else:
        AssertionError("Invalid compression type in eval_path")

    base_cmd = f"{base_cmd} training.comp.comp_type=online"
    base_cmd = f"{base_cmd} training.comp.attn_type={attn_type}"

    for i in range(1, 17):
        if f"-ntok{i}" in args.eval_path:
            base_cmd = f"{base_cmd} training.comp.num_comp_tokens={i}"

    # Evaluation path
    wandb_group = args.dataset
    eval_path = f"{SAVEPATH}/{wandb_group}/{args.eval_path}"
    base_cmd = f"{base_cmd} training.eval_path={eval_path}"

    if args.load_path != '':
        load_path = f"{SAVEPATH}/{wandb_group}/{args.load_path}"
        base_cmd = f"{base_cmd} training.load_path={load_path}"

    if args.interactive:
        base_cmd = f"{base_cmd} training.interactive=true"

    base_cmd = f"{base_cmd} model.cache_dir={CACHEDIR}"

    cmd = base_cmd.split()
    print(cmd)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        "-m",
                        type=str,
                        default="llama-7b",
                        choices=["llama-7b", "llama-2-7b-chat"])
    parser.add_argument("--dataset",
                        "-d",
                        type=str,
                        default='all',
                        choices=['all', 'metaicl', 'dialog'])
    parser.add_argument("--eval_path", "-e", type=str, help="Compression adapter path")
    parser.add_argument(
        "--eval_name",
        type=str,
        default='',
        help="Shortcut for eval_path. Set --eval_path and --dataset to test other models.")
    parser.add_argument("--interactive",
                        "-i",
                        action="store_true",
                        help="Interactive mode. Turning off will run some sanity check test codes.")

    args = parser.parse_args()

    args.load_path = ''
    if args.model == "llama-7b":
        args.load_path = "llama-7b-no"

    # Shortcut commands
    if args.eval_name != '':
        args.dataset == "all"

        if args.model == "llama-7b":
            if args.eval_name == "merge_recur":
                args.eval_path = "finetune/llama-7b-no-online-merge_recur-ntok8"
            if args.eval_name == "concat_recur":
                args.eval_path = "finetune/llama-7b-no-online-concat_recur-ntok2"

        elif args.model == "llama-2-7b-chat":
            if args.eval_name == "merge_recur":
                args.eval_path = "llama-2-7b-chat-online-merge_recur-ntok8"
            if args.eval_name == "concat_recur":
                args.eval_path = "llama-2-7b-chat-online-concat_recur-ntok2"

    run(args)
