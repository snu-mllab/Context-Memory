import hydra
import torch
from omegaconf.dictconfig import DictConfig
import numpy as np
import os
import time
from .arguments import Arguments, global_setup
from .model import load_model, load_pretrained
from path_config import DATAPATH


def evaluate_perp_stream(model,
                         tokenizer,
                         save_path,
                         data_path=os.path.join(DATAPATH, "pg19/validation.bin"),
                         window_size=128 + 32,
                         comp_size=64,
                         comp_cache_size=8,
                         sink_size=1,
                         max_num_eval_tokens=16000):

    do_comparison = False  # Set to True to enable sliding window without compression

    print("\n==== Streaming evaluation ====")
    print(f"Evaluating {data_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.eval()

    device = "cuda"
    nlls = []
    ppls = []
    past_key_values = position_ids = None
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    try:
        full_text = torch.tensor(
            np.memmap(data_path, dtype=np.uint16, mode='r').reshape([1, -1]).astype("int64"))
    except:
        FileNotFoundError("Data file not found. Run download.py --type data --name pg19")

    init = 9000  # Truncate the beginning of the text book data
    full_text = full_text[:, init:init + max_num_eval_tokens].to(device)
    full_text[:, 0] = tokenizer.bos_token_id

    seq_len = full_text.size(1)
    print(f"seq_len: {seq_len}")
    pbar = range(0, seq_len - 1)

    n_comp = 0
    comp_token = None
    merge = False

    comp_token = torch.tensor(model.comp_token, device=device).reshape([1, -1])
    tag = f"comp_w{window_size}_c{comp_size}_cache{comp_cache_size}"
    if sink_size > 0:
        tag += f"_s{sink_size}"
    if model.sum_token is not None:
        merge = True
        print("Use merge!")

    if do_comparison:
        tag = f"{tag}_comparison"
    print("tag:", tag)

    s = time.time()
    for idx in pbar:
        input_ids = full_text[:, idx:idx + 1]
        assert input_ids[0, 0].item() < 32000

        with torch.no_grad():
            if past_key_values is not None:
                kv_len = past_key_values[0][0].shape[2]
            else:
                kv_len = 0

            position_ids = torch.tensor([kv_len], device=device).reshape([1, -1])
            outputs = model(input_ids,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            use_cache=True)
            logits = outputs.logits
            logits = logits.reshape(-1, logits.size(-1))
            past_key_values = outputs.past_key_values
            label = full_text[:, idx + 1:idx + 2].reshape(-1)

            neg_log_likelihood = loss_fn(logits, label)

            comp_condition = kv_len >= window_size
            if comp_condition:
                past_key_values = compression(model,
                                              past_key_values,
                                              comp_token=comp_token,
                                              n_comp=n_comp,
                                              sink_size=sink_size,
                                              comp_size=comp_size,
                                              comp_cache_size=comp_cache_size,
                                              pos_offset=position_ids,
                                              merge=merge,
                                              do_comparison=do_comparison)
                n_comp += 1

        nlls.append(neg_log_likelihood)

        if (idx + 1) % 100 == 0:
            ppl = torch.exp(torch.stack(nlls).mean()).item()
            ppls.append(ppl)
            eta = (time.time() - s) / (idx + 1) * (seq_len - idx - 1)
            print(
                f"idx: {(idx + 1)}, ppl: {ppl:.2f} (kv {kv_len}, eta: {time.strftime('%H:%M:%S', time.gmtime(eta))}s)"
            )

        if (idx + 1) % 1000 == 0:
            torch.save(ppls, os.path.join(save_path, f"seq{max_num_eval_tokens}_ppl_{tag}.pt"))

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity ({tag} window {window_size}, start {sink_size}): {ppl}")
    return ppl


def compression(model,
                past_key_values,
                comp_token,
                n_comp,
                sink_size=0,
                comp_size=64,
                comp_cache_size=8,
                pos_offset=None,
                merge=False,
                do_comparison=False):
    kv_len = past_key_values[0][0].shape[2]
    comp_token_len = comp_token.shape[1]

    comp_cache_size = comp_cache_size + sink_size  # add sink token size

    if merge:
        prev_memory_size = sink_size + comp_token_len
        if n_comp == 0:
            prev_memory_size = sink_size
    else:
        prev_memory_size = sink_size + n_comp * comp_token_len
    prev_memory_size = min(prev_memory_size, comp_cache_size)

    total_comp_size = prev_memory_size + comp_size
    past_key_values_source = [
        [k[:, :, :total_comp_size], v[:, :, :total_comp_size]] for k, v in past_key_values
    ]

    # Compress KV pairs
    input_ids = comp_token
    position_ids = torch.arange(input_ids.shape[-1], device="cuda").reshape([1, -1])
    if pos_offset is not None:
        position_ids = position_ids + pos_offset - (kv_len - total_comp_size - 1)
    else:
        kv_len = past_key_values_source[0][0].shape[2]
        position_ids = position_ids + kv_len

    outputs = model(
        input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values_source,
        use_cache=True,
    )

    # Update past_key_values
    n_new = input_ids.shape[-1]
    past_key_values_comp = []
    for n_layer in range(len(past_key_values_source)):
        # CCM-merge
        if merge and n_comp >= 1:
            key_past = outputs.past_key_values[n_layer][0][:, :, sink_size:prev_memory_size]
            value_past = outputs.past_key_values[n_layer][1][:, :, sink_size:prev_memory_size]
            key_cur = outputs.past_key_values[n_layer][0][:, :, -n_new:]
            value_cur = outputs.past_key_values[n_layer][0][:, :, -n_new:]

            key_merge = n_comp / (n_comp + 1) * key_past + 1 / (n_comp + 1) * key_cur
            value_merge = n_comp / (n_comp + 1) * value_past + 1 / (n_comp + 1) * value_cur

            key = torch.cat([
                outputs.past_key_values[n_layer][0][:, :, :sink_size],
                key_merge,
                past_key_values[n_layer][0][:, :, total_comp_size:],
            ],
                            dim=2)
            value = torch.cat([
                outputs.past_key_values[n_layer][1][:, :, :sink_size],
                value_merge,
                past_key_values[n_layer][1][:, :, total_comp_size:],
            ],
                              dim=2)
        # CCM-concat
        else:
            cache_oversize = prev_memory_size + n_new - comp_cache_size
            offset = max(0, cache_oversize)

            key = torch.cat([
                outputs.past_key_values[n_layer][0][:, :, :sink_size],
                outputs.past_key_values[n_layer][0][:, :, sink_size + offset:prev_memory_size],
                outputs.past_key_values[n_layer][0][:, :, -n_new:],
                past_key_values[n_layer][0][:, :, total_comp_size:],
            ],
                            dim=2)
            value = torch.cat([
                outputs.past_key_values[n_layer][1][:, :, :sink_size],
                outputs.past_key_values[n_layer][1][:, :, sink_size + offset:prev_memory_size],
                outputs.past_key_values[n_layer][1][:, :, -n_new:],
                past_key_values[n_layer][1][:, :, total_comp_size:],
            ],
                              dim=2)
        past_key_values_comp.append([key, value])

    # For comparison, only use recent KV pairs in the sliding window without compression
    if do_comparison:
        total_size = past_key_values_comp[0][0].shape[2]
        recent_size = total_size - sink_size

        past_key_values_comp = [[
            torch.cat([k[:, :, :sink_size], k[:, :, -recent_size:]], dim=2),
            torch.cat([v[:, :, :sink_size], v[:, :, -recent_size:]], dim=2),
        ] for k, v in past_key_values]

    return past_key_values_comp


@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)

    # Load model and tokenizer
    model, tokenizer = load_model(args)
    if args.training.eval_path != '':
        lora = args.training.peft
        load_pretrained(args.training.eval_path, model, lora=lora)

    evaluate_perp_stream(model, tokenizer, save_path=args.training.output_dir)


if __name__ == "__main__":
    main()
