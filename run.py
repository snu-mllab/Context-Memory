import argparse
import os
from path_config import SAVEPATH, CACHEDIR, model_path, map_config


def tf_str(tf):
    s = "true" if tf else "false"
    return s


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_path(eval_path):
    attn_type = "concat_recur"
    n_tok = 1

    max_k = 32
    for i in range(1, max_k + 1):
        if (f"-ntok{i}" in eval_path):
            n_tok = i
            print(f"Update n_tok as {n_tok}")

    if "concat_recur" in eval_path:
        attn_type = "concat_recur"
    elif "merge_recur" in eval_path:
        attn_type = "merge_recur"
    elif "concat" in eval_path:
        attn_type = "concat"
    elif "merge" in eval_path:
        attn_type = "merge"
    else:
        print("No attention type matched!")

    return attn_type, n_tok


def run(args):
    base_cmd = f"python -B -m src.train"

    if "debug" not in args.model:
        if args.load_path != '':
            assert args.model in args.load_path, "Check model and load_path"
        if args.eval_path != '':
            args.train = False
            assert args.model in args.eval_path, "Check model and eval_path"
    else:
        args.no_wandb = True

    ## Config file
    model = args.model
    base_cmd = f"{base_cmd} +{args.dataset}={map_config(model)} model.model_name_or_path={model_path(model)}"
    base_cmd = f"{base_cmd} training.do_train={tf_str(args.train)}"

    ## Data config
    if args.dataset == 'metaicl':
        base_cmd = f"{base_cmd} data.k={args.k}"
    if args.dataset == 'lamp':
        base_cmd = f"{base_cmd} data.dataset_name=lamp2"
        base_cmd = f"{base_cmd} data.k={args.k}"

    ## Compression config
    if args.comp_type in ["no", "neg_control"]:
        base_cmd = f"{base_cmd} training.comp.add_comp_token=false"
        args.cond_lora = args.sepembed = False
        base_cmd = f"{base_cmd} training.comp.relative_embedding=base"
    if args.attn_type == "gist":
        args.comp_type = "fixed"

    base_cmd = f"{base_cmd} training.comp.comp_type={args.comp_type}"
    method_tag = args.comp_type

    if args.eval_path != '':
        args.attn_type, args.n_tok = parse_path(args.eval_path)
        args.sink = True if "-sink" in args.eval_path else False

    if args.comp_type not in ["no", "neg_control"]:
        base_cmd = f"{base_cmd} training.comp.attn_type={args.attn_type}"
        method_tag += f"-{args.attn_type}"

    if args.n_tok > 1:
        base_cmd = f"{base_cmd} training.comp.num_comp_tokens={args.n_tok}"
        method_tag += f"-ntok{args.n_tok}"

    if args.sink:
        base_cmd = f"{base_cmd} training.comp.sink=true"
        method_tag += f"-sink"

    ## Model config
    model_tag = ''
    if model.startswith("llama"):
        if args.lora_r > 0:
            base_cmd = f"{base_cmd} training.lora_r={args.lora_r}"
            method_tag = f"r{args.lora_r}-{method_tag}"
        if args.per_device_train_batch_size >= 1:
            bs = args.per_device_train_batch_size
            base_cmd = f"{base_cmd} training.per_device_train_batch_size={bs}"
            base_cmd = f"{base_cmd} training.gradient_accumulation_steps={128//bs}"
            args.tag += f"_batch{bs}"
    if args.max_steps > 0:
        st = args.max_steps
        args.tag += f"_{st}"
        base_cmd = f"{base_cmd} training.max_steps={st}"
        base_cmd = f"{base_cmd} training.save_steps={st//4} training.eval_steps={st//2}"

    ## Training config
    if args.lr > 0:
        args.tag += f"_lr{args.lr}"
        base_cmd = f"{base_cmd} training.learning_rate={args.lr}"
    if args.seed != 42:
        base_cmd = f"{base_cmd} training.seed={args.seed}"
        args.tag += f"_seed{args.seed}"

    ## Conditional Lora
    if not args.cond_lora:
        base_cmd = f"{base_cmd} training.comp.cond_lora=false"
    if not args.sepembed:
        base_cmd = f"{base_cmd} training.comp.separate_embed=false"

    ## Logging path
    if args.train_dataset is None:
        wandb_group = wandb_group_out = args.dataset
    else:
        wandb_group = wandb_group_out = args.train_dataset
        args.tag += f"_{args.dataset}"

    # Set different folder for longer max context length
    if model.startswith('llama') and args.max_length > 1024:
        base_cmd = f"{base_cmd} data.max_length={args.max_length}"
        wandb_group_out = f"{args.dataset}-len{args.max_length}"

        generation_max_length = args.max_length + args.n_tok * args.k + 128
        base_cmd = f"{base_cmd} training.generation_max_length={generation_max_length}"

    ## Evaluation path
    if args.dataset in ["metaicl", "lamp"] and (args.k != 16 or args.eval_path != ''):
        method_tag += f"-k{args.k}"
    if args.tag != '':
        method_tag += f"{args.tag}"

    # Load a compression adapter/model for evaluation
    if args.eval_path != '':
        subfolder = 'test'

        training_output_dir = f"{SAVEPATH}/{wandb_group_out}/{subfolder}/{args.eval_path}"
        training_output_dir += f"-{method_tag}"

        wandb_name = f"{subfolder}-{os.path.basename(training_output_dir)}"

        # Load finetuned adapter/model
        if args.eval_path.startswith("finetune"):
            assert args.load_path != '', "Check args.load_path! (e.g., llama-7b-no)"
        if args.load_path != '':
            load_path = f"{SAVEPATH}/{wandb_group}/{args.load_path}"
            base_cmd = f"{base_cmd} training.load_path={load_path}"

        eval_path = f"{SAVEPATH}/{wandb_group}/{args.eval_path}"
        base_cmd = f"{base_cmd} training.eval_path={eval_path}"

    # Load finetuned adapter/model before training a compression adapter/model
    elif args.load_path != '':
        subfolder = 'finetune'

        training_output_dir = f"{SAVEPATH}/{wandb_group_out}/{subfolder}/{args.load_path}"
        training_output_dir += f"{model_tag}-{method_tag}"

        wandb_name = f"{subfolder}-{os.path.basename(training_output_dir)}"

        load_path = f"{SAVEPATH}/{wandb_group}/{args.load_path}"
        base_cmd = f"{base_cmd} training.load_path={load_path}"

    else:
        wandb_name = f"{model}{model_tag}-{method_tag}"
        training_output_dir = f"{SAVEPATH}/{wandb_group_out}/{wandb_name}"

    if "debug" in model:
        wandb_name = "debug"
        training_output_dir = f"{SAVEPATH}/{wandb_group_out}/{wandb_name}"

    base_cmd = f"{base_cmd} wandb.group={wandb_group}"
    base_cmd = f"{base_cmd} wandb.name={wandb_name}"
    base_cmd = f"{base_cmd} training.output_dir={training_output_dir}"
    base_cmd = f"{base_cmd} training.eval_rouge={args.eval_rouge}"
    base_cmd = f"{base_cmd} model.cache_dir={CACHEDIR}"

    # No wandb for testing
    if (args.no_wandb) or not args.train:
        base_cmd = f"{base_cmd} wandb.log=false"

    cmd = base_cmd.split()
    print(cmd)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model",
                        "-m",
                        default="llama-7b",
                        choices=[
                            "llama-7b", "llama-13b", "llama-2-7b", "llama-2-7b-chat", "llama-2-13b",
                            "llama-2-13b-chat", "mistral-7b", "mistral-7b-inst", "flan-t5-base",
                            "flan-t5-large", "llama-debug", "mistral-debug"
                        ])
    parser.add_argument("--sepembed",
                        type=str2bool,
                        default=True,
                        help="Train only embeddings for COMP tokens")
    parser.add_argument("--cond_lora", type=str2bool, default=True, help="Use conditional LoRA")
    parser.add_argument("--lora_r",
                        "-r",
                        type=int,
                        default=-1,
                        help="LoRA rank size (default settings are provided in src/config)")
    ## Notes on compression types ##
    ## 1. 'no' refers to the full attention model without compression.
    ## 2. 'neg_control' refers to the full attention model attending only the input without context.
    ## 3. 'fixed' refers to the fixed compression method.
    ## 4. 'online' refers to the online compression method (ours).
    parser.add_argument("--comp_type",
                        default="online",
                        choices=["no", "pos_control", "neg_control", "fixed", "online"],
                        help="Compression type")
    ## Notes on attention types ##
    ## 1. 'concat_recur/merge_recur' refers to CCM-concat/-merge
    ## 2. 'concat/merge' are the variants of CCM-concat/-merge.
    ##    During compression, they do not attend to the previous memory state, attending only to the current context.
    ##    When generating outputs, they attend to the concatenated/merged memory states.
    ##    This strategy shows slightly better performance in MetaICL or LaMP.
    ## 3. 'gist' refers to the Gisting compression method.
    parser.add_argument("--attn_type",
                        default="concat_recur",
                        choices=["concat_recur", "merge_recur", "concat", "merge", "gist"],
                        help="Attention type")
    parser.add_argument("--n_tok",
                        type=int,
                        default=1,
                        help="Number of COMP tokens for each time step")
    ## Notes on datasets ##
    ## 'unified' refers to the mixture of MetaICL and SODA (Table 15 of our arXiv paper).
    ## 'pretrain' refers to the mixture of RedPajama-V2, LmSys-Chat, MetaICL, SODA (Table 4 of our arXiv paper).
    parser.add_argument("--dataset",
                        "-d",
                        default='metaicl',
                        choices=['unified', 'metaicl', 'dialog', 'soda', 'lamp', 'pretrain'],
                        help="Training/evaluation dataset.")
    parser.add_argument(
        "--train_dataset",
        type=str,
        default=None,
        help=
        "Training dataset of the evaluation model. Use when the training dataset is differ from the evaluation dataset."
    )
    parser.add_argument("--k",
                        type=int,
                        default=16,
                        help="Max number of context time steps (metaicl, LaMP)")
    parser.add_argument("--max_length",
                        type=int,
                        default=1024,
                        help="Max token length for each data sample")
    # Training
    parser.add_argument("--train", action="store_true", help="Do finetuning")
    parser.add_argument("--sink",
                        action="store_true",
                        help="Keep attention sink (BOS token) during compression training")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max finetuning steps")
    parser.add_argument("--lr", type=float, default=-1, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", '-b', type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Eval
    parser.add_argument(
        "--load_path",
        type=str,
        default='',
        help=
        "Path for the full-context finetuned adapter (not required for when using the already finetuned models, e.g., LLaMA-2-chat)"
    )
    parser.add_argument("--eval_path", type=str, default='', help="Compression adapter path")
    parser.add_argument("--generation_max_length", type=int, default=-1)
    parser.add_argument("--eval_rouge", action="store_true", help="Evaluate generation performance")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--tag", type=str, default='')

    args = parser.parse_args()

    run(args)
