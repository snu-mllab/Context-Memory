import argparse
import os
from path_config import SAVEPATH, CACHEDIR, model_path, map_config
from run import parse_path


def run(args):
    if args.interactive:
        base_cmd = f"python -B -m src.test_interact"
    elif args.stream:
        base_cmd = f"python -B -m src.test_stream"
    else:
        base_cmd = f"python -B -m src.test_case"

    # Config file
    model = args.model
    base_cmd = f"{base_cmd} +{args.dataset}={map_config(model)} model.model_name_or_path={model_path(model)}"
    base_cmd = f"{base_cmd} training.do_train=false wandb.log=false"

    # Compression type
    attn_type, n_tok = parse_path(args.eval_path)
    base_cmd = f"{base_cmd} training.comp.comp_type=online"
    base_cmd = f"{base_cmd} training.comp.attn_type={attn_type}"
    base_cmd = f"{base_cmd} training.comp.num_comp_tokens={n_tok}"

    # Evaluation path
    wandb_group = args.dataset
    eval_path = f"{SAVEPATH}/{wandb_group}/{args.eval_path}"
    base_cmd = f"{base_cmd} training.eval_path={eval_path}"

    load_path = None
    if args.load_path != '':
        load_path = f"{SAVEPATH}/{wandb_group}/{args.load_path}"
        base_cmd = f"{base_cmd} training.load_path={load_path}"

    if args.interactive:
        base_cmd = f"{base_cmd} training.interactive=true"
    if args.stream:
        base_cmd = f"{base_cmd} model.stream=true"

    base_cmd = f"{base_cmd} model.cache_dir={CACHEDIR}"

    cmd = base_cmd.split()
    print(cmd)
    if load_path is not None:
        print(f"\nFinetuned base model adapter path : {load_path}")
    print(f"Compression adapter path          : {eval_path}\n")
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
                        default="unified",
                        choices=["unified", "pretrain", "metaicl", "dialog"])
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
    parser.add_argument("--stream", "-s", action="store_true", help="Evaluate streaming setting.")

    args = parser.parse_args()

    args.load_path = ''
    if args.model == "llama-7b":
        args.load_path = "llama-7b-no"

    # Shortcut commands
    if args.eval_name != '':
        if args.model == "llama-7b":
            if args.eval_name == "merge_recur":
                args.eval_path = "finetune/llama-7b-no-online-merge_recur-ntok8"
            if args.eval_name == "concat_recur":
                args.eval_path = "finetune/llama-7b-no-online-concat_recur-ntok2"

        elif args.model == "llama-2-7b-chat":
            args.dataset = "unified"
            if args.eval_name == "merge_recur":
                args.eval_path = "llama-2-7b-chat-online-merge_recur-ntok8"
            if args.eval_name == "concat_recur":
                args.eval_path = "llama-2-7b-chat-online-concat_recur-ntok2"

    # Streaming model
    if args.stream:
        args.model = "llama-7b"
        args.dataset = "pretrain"
        args.eval_path = "finetune/llama-7b-no-online-concat_recur-ntok2-sink"

    run(args)
