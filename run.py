import argparse
import os
from path_config import SAVEPATH, CACHEDIR


def tf_str(tf):
    s = "true" if tf else "false"
    return s


def run(args):
    base_cmd = f"python -B -m src.train"

    if "debug" not in args.model:
        if args.load_path != '':
            assert args.model in args.load_path, "Check model and load_path"
        if args.eval_path != '':
            assert args.train is False
            assert args.model in args.eval_path, "Check model and eval_path"
    else:
        args.no_wandb = True

    # Config file
    model = args.model
    base_cmd = f"{base_cmd} +{args.dataset}={model}"

    base_cmd = f"{base_cmd} training.do_train={tf_str(args.train)}"
    if args.dataset == 'metaicl':
        base_cmd = f"{base_cmd} data.k={args.k}"

    if args.dataset == 'lamp':
        base_cmd = f"{base_cmd} data.dataset_name=lamp2"
        base_cmd = f"{base_cmd} data.k={args.k}"
        max_length = 1024
        max_length += 256
        if args.comp_type != "no":
            max_length += args.n_tok * args.k
        base_cmd = f"{base_cmd} training.generation_max_length={max_length}"

    # Compression config
    if args.attn_type == "gist":
        args.comp_type = "fixed"

    if args.comp_type in ["no", "neg_control"]:
        base_cmd = f"{base_cmd} training.comp.add_comp_token=false"
        args.cond_lora = args.sepembed = False
        base_cmd = f"{base_cmd} training.comp.relative_embedding=base"

    base_cmd = f"{base_cmd} training.comp.comp_type={args.comp_type}"

    method_tag = args.comp_type
    if args.comp_type not in ["no", "neg_control"]:
        base_cmd = f"{base_cmd} training.comp.attn_type={args.attn_type}"
        method_tag += f"-{args.attn_type}"

    if args.eval_path != '':
        for i in range(1, 17):
            if (f"-ntok{i}" in args.eval_path):
                args.n_tok = i
                print(f"Update n_tok as {args.n_tok}")

    if args.n_tok > 1:
        base_cmd = f"{base_cmd} training.comp.num_comp_tokens={args.n_tok}"
        method_tag += f"-ntok{args.n_tok}"

    # Model config
    model_tag = ''
    if model.startswith("llama"):
        if args.lora_r > 0:
            base_cmd = f"{base_cmd} training.lora_r={args.lora_r}"
            method_tag = f"r{args.lora_r}-{method_tag}"

        if args.max_steps > 0:
            st = args.max_steps
            args.tag += f"_{st}"
            base_cmd = f"{base_cmd} training.max_steps={st}"
            base_cmd = f"{base_cmd} training.save_steps={st//4} training.eval_steps={st//2}"
        if args.embed:
            base_cmd = f"{base_cmd} training.finetune_embed=true"
            method_tag = f"embed-{method_tag}"
        if args.per_device_train_batch_size >= 1:
            bs = args.per_device_train_batch_size
            base_cmd = f"{base_cmd} training.per_device_train_batch_size={bs}"
            base_cmd = f"{base_cmd} training.gradient_accumulation_steps={128//bs}"
            args.tag += f"_batch{bs}"

    if model.startswith('flan'):
        if args.max_steps > 0:
            st = args.max_steps
            args.tag += f"_{st}"
            base_cmd = f"{base_cmd} training.max_steps={st}"
            base_cmd = f"{base_cmd} training.save_steps={st//2} training.eval_steps={st//2}"

    if args.lr > 0:
        args.tag += f"_lr{args.lr}"
        base_cmd = f"{base_cmd} training.learning_rate={args.lr}"

    # Conditional Lora
    if not args.cond_lora: base_cmd = f"{base_cmd} training.comp.cond_lora=false"
    if not args.sepembed: base_cmd = f"{base_cmd} training.comp.separate_embed=false"

    if args.seed != 42:
        base_cmd = f"{base_cmd} training.seed={args.seed}"
        args.tag += f"_seed{args.seed}"

    # Logging path
    if args.pretrain_dataset is None:
        wandb_group = wandb_group_out = args.dataset
    else:
        wandb_group = wandb_group_out = args.pretrain_dataset
        args.tag += f"_{args.dataset}"

    if model.startswith('llama') and args.max_length > 1024:
        base_cmd = f"{base_cmd} data.max_length={args.max_length}"
        wandb_group_out = f"{args.dataset}-len{args.max_length}"

        generation_max_length = args.max_length + args.n_tok * args.k + 128
        base_cmd = f"{base_cmd} training.generation_max_length={generation_max_length}"

    if args.dataset in ["metaicl", "lamp"] and (args.k != 16 or args.eval_path != ''):
        method_tag += f"-k{args.k}"
    if args.tag != '':
        method_tag += f"{args.tag}"

    # Evaluation path
    if args.eval_path != '':
        subfolder = 'test'

        training_output_dir = f"{SAVEPATH}/{wandb_group_out}/{subfolder}/{args.eval_path}"
        training_output_dir += f"-{method_tag}"

        wandb_name = f"{subfolder}-{os.path.basename(training_output_dir)}"

        # Load pretrained model for evaluation
        if args.eval_path.startswith("finetune"):
            assert args.load_path != '', "Check args.load_path! (e.g., llama-7b-no)"
        if args.load_path != '':
            load_path = f"{SAVEPATH}/{wandb_group}/{args.load_path}"
            base_cmd = f"{base_cmd} training.load_path={load_path}"

        eval_path = f"{SAVEPATH}/{wandb_group}/{args.eval_path}"
        base_cmd = f"{base_cmd} training.eval_path={eval_path}"

    # Load pretrained model for finetuning
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
    base_cmd = f"{base_cmd} model.cache_dir={CACHEDIR}"

    # No wandb for testing
    if (args.no_wandb) or not args.train:
        base_cmd = f"{base_cmd} wandb.log=false"

    if args.override is not None:
        override = " ".join(args.override.split(","))
        base_cmd = f"{base_cmd} {override}"

    cmd = base_cmd.split()
    print(cmd)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    user = os.environ.get("USER", "janghyun")

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="", help="See path_config.py")
    # Model
    parser.add_argument("--model", "-m", default="llama-7b")
    parser.add_argument("--sepembed",
                        type=str2bool,
                        default=True,
                        help="Train only embeddings for COMP tokens")
    parser.add_argument("--embed",
                        action="store_true",
                        help="Train full embedding vectors during LoRA finetuning")
    parser.add_argument("--cond_lora", type=str2bool, default=True, help="Conditional LoRA")
    parser.add_argument("--lora_r",
                        "-r",
                        type=int,
                        default=-1,
                        help="LoRA rank size (default settings are in src/config)")
    # Compression
    parser.add_argument("--comp_type",
                        default="online",
                        choices=["no", "pos_control", "neg_control", "fixed", "online"],
                        help="Compression type")
    parser.add_argument("--attn_type",
                        default="concat_recur",
                        choices=["concat_recur", "merge_recur", "concat", "merge", "gist"],
                        help="Attention type")
    parser.add_argument("--n_tok",
                        type=int,
                        default=1,
                        help="Number of COMP tokens for each context")
    # Data
    parser.add_argument("--dataset",
                        "-d",
                        default='metaicl',
                        choices=['all', 'metaicl', 'dialog', 'soda', 'lamp'],
                        help="The 'all' dataset refers to the mixture of MetaICL and SODA.")
    parser.add_argument("--pretrain_dataset",
                        type=str,
                        default=None,
                        help="Train dataset. Use when it is differ from the evaluation dataset")
    parser.add_argument("--k",
                        type=int,
                        default=16,
                        help="Max number of context time steps (metaicl, LaMP)")
    parser.add_argument("--max_length", type=int, default=1024, help="Max length for data sample")
    parser.add_argument("--generation_max_length",
                        type=int,
                        default=-1,
                        help="Generation max length")
    # Training
    parser.add_argument("--train", action="store_true", help="Conduct finetuning")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max finetuning steps")
    parser.add_argument("--lr", type=float, default=-1, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", '-b', type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Eval
    parser.add_argument("--load_path",
                        type=str,
                        default='',
                        help="Full-context finetuned adapter path (not required for LLaMA-2-chat)")
    parser.add_argument("--eval_path", type=str, default='', help="Compression adapter path")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--override")
    parser.add_argument("--tag", type=str, default='')

    args = parser.parse_args()

    run(args)
