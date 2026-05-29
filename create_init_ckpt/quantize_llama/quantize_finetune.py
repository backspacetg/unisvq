import os
import glog
import argparse
glog.setLevel(glog.INFO)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from lib import codebook, utils
from lib.algo import finetune


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
parser.add_argument('--hessian_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--incoh_mode', default='had', type=str, choices=['had', 'kron'])
parser.add_argument('--lora_rank',default=0, type=int, help='if <=0 then turned off')
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--scale_search_iters', default=1, type=int)
parser.add_argument('--resid_scale_override', default=-1, type=float)
parser.add_argument('--codebook', type=str)
parser.add_argument('--quip_tune_iters', default=10, type=int)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--full_svd', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--rescale_WH', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=5e-5, type=float)
parser.add_argument('--ft_susv_lr', default=5e-4, type=float)
parser.add_argument('--ft_bs', default=4, type=int)
parser.add_argument('--ft_update_freq', default=2, type=int)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=3, type=int)
parser.add_argument('--ft_train_mode', action='store_true')
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--dataset_path', default="/data", type=str)
parser.add_argument('--skip_mix_hadk', action='store_true')


def check_exist(idx, args):
    suffix = ['qkv', 'o', 'up', 'down', 'layernorm']
    for _ in suffix:
        test = f'{args.save_path}/{idx}_{_}.pt'
        if not os.path.exists(test):
            return False
    return True


def quantize_llama_layer(layer, idx, cb, args, device, pre_orig_emb, orig_emb, position_embeddings):
    if check_exist(idx, args):
        print(f"{idx} exits")
        return
    
    quant_order = [
        ['self_attn.v_proj', 'qkv'],
        ['self_attn.q_proj', 'qkv'],
        ['self_attn.k_proj', 'qkv'],
        ['self_attn.o_proj', 'o'],
        ['mlp.up_proj', 'upgate'],
        ['mlp.gate_proj', 'upgate'],
        ['mlp.down_proj', 'down']
    ]

    finetune.quantize_finetune_decoder_layer(mixed_layer=layer, quant_order=quant_order, idx=idx, cb=cb, args=args, device=device, pre_orig_emb=pre_orig_emb, orig_emb=orig_emb, position_embeddings=position_embeddings)

    torch.save({
            'input_layernorm': layer.input_layernorm.weight,
            'post_attention_layernorm': layer.post_attention_layernorm.weight,
        }, f'{args.save_path}/{idx}_layernorm.pt'
    )
    del layer


def main(args):
    # glog.info("into main")
    cb = codebook.get_codebook(args.codebook)
    # glog.info("codebook get")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype='auto', low_cpu_mem_usage=True, trust_remote_code=True)
    # glog.info("model loaded")
    # save configs
    all_config = {'quant_args': args, 'model_config': model.config}
    quip_params = {
        'lora_rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'codebook': args.codebook,
        'codebook_version': cb.version,
        'codesz': cb.codesz,
        'idx_dtype': str(cb.idx_dtype),
        'packsz': cb.packsz,
        'resid_scale_override': args.resid_scale_override,
        'codebook_scale': getattr(cb, "codebook_scale", 1.0),
        'skip_mix_hadk': args.skip_mix_hadk
    }
    all_config['model_config'].update({'quip_params': quip_params})
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))
    # glog.info("config saved")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # glog.info('loaded model')

    if "pajama" in args.dataset_path:
        devset = utils.sample_rp1t_concat(tokenizer, args.devset_size, args.ctx_size, nproc=args.sample_proc)
    elif "jsonl" in args.dataset_path:
        devset = utils.sample_jsonl_concat(args.dataset_path, tokenizer, args.devset_size, args.ctx_size, nproc=args.sample_proc)
    else:
        not NotImplementedError(args.dataset_path)
    # glog.info('loaded dataset and devset')

    first_input = model.model.embed_tokens(devset) # [dataset_shape, seq_len, hidden_size]

    embedding_cache = {
        "input": first_input,
        "output": torch.zeros(first_input.shape, dtype=first_input.dtype, device=first_input.device)
    }

    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int32)
    position_embeddings_cos, position_embeddings_sin = model.model.rotary_emb(first_input[0], position_ids)
    # glog.info("{} {}".format(position_embeddings_cos.shape, position_embeddings_sin.shape))
    # exit(0)
    attention_mask = _prepare_4d_causal_attention_mask(None, (args.batch_size, args.ctx_size), first_input[:args.batch_size], 0)

    device = torch.device("cuda:0")
    for i in tqdm.tqdm(range(len(model.model.layers))):
        utils.clean()
        if args.ft_epochs > 0:
            position_ids = position_ids.to(device)
            position_embeddings = (position_embeddings_cos.cuda(), position_embeddings_sin.cuda())
            attention_mask = attention_mask.to(device)
            model.model.layers[i].to(device)
            for j in range(args.devset_size // args.batch_size):
                result = model.model.layers[i](
                    embedding_cache["input"][args.batch_size*j:args.batch_size*(j + 1)].to(device),
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False
                )
                embedding_cache["output"][args.batch_size*j:args.batch_size*(j + 1)] = result[0].cpu()
            model.model.layers[i].cpu()
            position_ids = position_ids.cpu()
            attention_mask = attention_mask.cpu()
            utils.clean()

        quantize_llama_layer(
            layer=model.model.layers[i],
            idx=i,
            cb=cb,
            args=args,
            device=device,
            pre_orig_emb=embedding_cache["input"],
            orig_emb=embedding_cache["output"],
            position_embeddings=position_embeddings
        )

        embedding_cache['input'] = embedding_cache['output'].detach().clone()


if __name__ == '__main__':
    glog.info("into function")
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
