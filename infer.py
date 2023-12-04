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

    # Model type
    if "-cond" in args.eval_path or "-gistlora" in args.eval_path:
        base_cmd = f"{base_cmd} training.comp.cond_lora=true"
    else:
        base_cmd = f"{base_cmd} training.comp.cond_lora=false"
    if "-sepembed" in args.eval_path:
        base_cmd = f"{base_cmd} training.comp.separate_embed=true"
    else:
        base_cmd = f"{base_cmd} training.comp.separate_embed=false"

    pos_id_type = "base"
    if "-skip" in args.eval_path or "-norel" in args.eval_path:
        pos_id_type = "skip"
    base_cmd = f"{base_cmd} training.comp.relative_embedding={pos_id_type}"

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

    if attn_type is not None:
        base_cmd = f"{base_cmd} training.comp.comp_type=online"
        base_cmd = f"{base_cmd} training.comp.attn_type={attn_type}"
    else:
        base_cmd = f"{base_cmd} training.comp.add_comp_token=false"

    for i in range(1, 17):
        if (f"-ntok{i}-" in args.eval_path) or (f"-ntok{i}_" in args.eval_path):
            base_cmd = f"{base_cmd} training.comp.num_comp_tokens={i}"

    if args.generation_max_length > 0:
        base_cmd = f"{base_cmd} training.generation_max_length={args.generation_max_length}"

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
    parser.add_argument("--model", "-M", default="llama-7b")
    parser.add_argument("--dataset", default='soda', choices=['metaicl', 'dialog', 'soda'])
    parser.add_argument("--eval_path", "-e", type=str, help="Evaluation model path")
    parser.add_argument("--load_path", "-l", type=str, default='', help="Pretrained model path")
    parser.add_argument("--generation_max_length",
                        type=int,
                        default=-1,
                        help="Generation max length")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    run(args)
