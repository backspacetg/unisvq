import os
import argparse
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import glog
import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import  _prepare_4d_causal_attention_mask

from lib import codebook, utils
from lib.algo import finetune

glog.setLevel(glog.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
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
parser.add_argument('--qep_factor', default=0.5, type=float)


def check_exist(idx, args):
    suffix = ['qkv', 'o', 'up', 'down', 'layernorm']
    for _ in suffix:
        test = f'{args.save_path}/{idx}_{_}.pt'
        if not os.path.exists(test):
            return False
    return True


def quantize_llama_layer(
        layer, idx, cb, args, 
        device, pre_orig_emb, 
        orig_emb, hessian_data, 
        position_embeding
    ):
    if check_exist(idx, args):
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

    finetune.quantize_finetune_decoder_layer(
        mixed_layer=layer, 
        quant_order=quant_order, 
        idx=idx, 
        cb=cb, 
        args=args, 
        device=device, 
        pre_orig_emb=pre_orig_emb, 
        orig_emb=orig_emb, 
        hessian_data=hessian_data, 
        position_embeddings=position_embeding
    )

    torch.save({
            'input_layernorm': layer.input_layernorm.weight,
            'post_attention_layernorm': layer.post_attention_layernorm.weight,
        }, f'{args.save_path}/{idx}_layernorm.pt'
    )
    del layer


def modify_hessian_result(activation_results):
    hessian_data = {}
    for linear_name, linear_result in activation_results.items():
        mu = linear_result["mu"]
        H = linear_result["H"]
        mu_fp = linear_result["mu_fp"]
        H_fp_quantize = linear_result["H_fp_quantize"]
        ct = linear_result["ct"]
        mu.div_(ct)
        H.div_(ct)
        H.addmm_(-mu.unsqueeze(-1), mu.unsqueeze(0))
        mu_fp.div_(ct)
        H_fp_quantize.div_(ct)
        H_fp_quantize.addmm_(-mu_fp.unsqueeze(-1), mu.unsqueeze(0))
        hessian_data[linear_name] = {
            'H': H,
            'mu': mu.to(torch.float32),
            'n': H.shape[0],
            'ct': ct,
            'H_fp_quantize': H_fp_quantize
        }
        # glog.info(H)
        # glog.info(H_qep)
    return hessian_data


def qep_modify_weights(layer, layer_outputs, args):
    quant_order = [
        ('self_attn', 'q_proj', 'qkv'),
        ('self_attn', 'k_proj', 'qkv'),
        ('self_attn', 'v_proj', 'qkv'),
        ('self_attn', 'o_proj', 'o'),
        ('mlp', 'up_proj', 'upgate'),
        ('mlp', 'gate_proj', 'upgate'),
        ('mlp', 'down_proj', 'down'),
    ]
    for parent_name, linear_name, group_name in quant_order:
        linear_module = getattr(getattr(layer, parent_name), linear_name)
        factory_kwargs = {
            "dtype": linear_module.weight.dtype,
            "device": linear_module.weight.device
        }
        hessian_fp_quantize = layer_outputs[group_name]['H_fp_quantize'].to(**factory_kwargs)
        hessian = layer_outputs[group_name]['H']
        hessian = torch.linalg.inv(utils.math_utils.regularize_H(hessian, layer_outputs[group_name]['n'], args.sigma_reg)).to(**factory_kwargs)
        H_factor = torch.mm(hessian_fp_quantize, hessian)
        linear_module.weight.data = (1-args.qep_factor) * linear_module.weight.data + args.qep_factor * torch.mm(linear_module.weight.data, H_factor)
    return 


def main(args):
    cb = codebook.get_codebook(args.codebook)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
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
    }
    all_config['model_config'].update({'quip_params': quip_params})
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    glog.info('loaded model')

    if "pajama" in args.dataset_path:
        devset = utils.sample_rp1t_concat(tokenizer, args.devset_size, args.ctx_size, nproc=args.sample_proc)
    elif "jsonl" in args.dataset_path:
        devset = utils.sample_jsonl_concat(args.dataset_path, tokenizer, args.devset_size, args.ctx_size, nproc=args.sample_proc)
    else:
        not NotImplementedError(args.dataset_path)
    glog.info('loaded dataset and devset')

    first_input = model.model.embed_tokens(devset).to(torch.float16)

    embedding_cache = {
        "input": first_input,
        "input_after_quantize": first_input.clone(),
        "output": torch.zeros(first_input.shape, dtype=first_input.dtype, device=first_input.device),
        "output_after_quantize": torch.zeros(first_input.shape, dtype=first_input.dtype, device=first_input.device)
    }

    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int32)
    position_embeddings_cos, position_embeddings_sin = model.model.rotary_emb(first_input[0], position_ids)
    attention_mask = _prepare_4d_causal_attention_mask(None, (args.batch_size, args.ctx_size), first_input[:args.batch_size], 0)
    device = torch.device("cuda:0")

    position_embeddings = (position_embeddings_cos.cuda(), position_embeddings_sin.cuda())
    attention_mask = attention_mask.to(device)

    for i in tqdm.tqdm(range(len(model.model.layers))):
        utils.clean()

        layer_activations = {}
        def hook_generator(layer_name, device):
            def get_hessian_hook(module, inputs):
                layer_hessian = layer_activations.get(layer_name, {})
                inputs_before_quantize = layer_hessian.get("inputs_before_quantize", None)
                n = module.in_features
                if inputs_before_quantize is None:
                    layer_hessian['inputs_before_quantize'] = inputs[0].reshape(-1, n).to(torch.float32)
                    layer_activations[layer_name] = layer_hessian
                else:
                    H = layer_hessian.get("H", torch.zeros(n, n, dtype=torch.float32, device=device))
                    H_fp_quantize = layer_hessian.get("H_fp_quantize", torch.zeros(n, n, dtype=torch.float32, device=device))
                    mu = layer_hessian.get("mu", torch.zeros(n, dtype=torch.float32, device=device))
                    mu_fp = layer_hessian.get("mu_fp", torch.zeros(n, dtype=torch.float32, device=device))
                    ct = layer_hessian.get("ct", 0)
                    
                    input_after_quantize = inputs[0].reshape(-1, n).to(torch.float32)
                    H.addmm_(input_after_quantize.T, input_after_quantize)
                    H_fp_quantize.addmm_(inputs_before_quantize.T, input_after_quantize)
                    mu.add_(input_after_quantize.sum(dim=0))
                    mu_fp.add_(inputs_before_quantize.sum(dim=0))
                    ct += len(input_after_quantize)
                    layer_activations[layer_name] = {
                        "H": H,
                        "mu": mu,
                        "ct": ct,
                        'H_fp_quantize': H_fp_quantize,
                        "mu_fp": mu_fp,
                        'inputs_before_quantize': None
                    }
            return get_hessian_hook

        hook_qkv = model.model.layers[i].self_attn.q_proj.register_forward_pre_hook(hook_generator("qkv", device))
        hook_o = model.model.layers[i].self_attn.o_proj.register_forward_pre_hook(hook_generator("o", device))
        hook_upgate = model.model.layers[i].mlp.up_proj.register_forward_pre_hook(hook_generator("upgate", device))
        hook_down = model.model.layers[i].mlp.down_proj.register_forward_pre_hook(hook_generator("down", device))
        model.model.layers[i].to(device)
        pbar = tqdm.tqdm(range(args.devset_size // args.batch_size), desc="getting hessian info")
        with torch.no_grad():
            for j in pbar:
                for key in ['input', 'input_after_quantize']: 
                    result = model.model.layers[i](
                        embedding_cache[key][args.batch_size*j:args.batch_size*(j + 1)].to(device),
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=False
                    )
                    if key == "input":
                        embedding_cache["output"][args.batch_size*j:args.batch_size*(j + 1)] = result[0].cpu()
                    else:
                        embedding_cache["output_after_quantize"][args.batch_size*j:args.batch_size*(j + 1)] = result[0].cpu()
        hook_qkv.remove()
        hook_o.remove()
        hook_upgate.remove()
        hook_down.remove()

        hessian_data = modify_hessian_result(layer_activations)
        del layer_activations
        # 调整权重
        # if i > 0:
        qep_modify_weights(model.model.layers[i], hessian_data, args=args)
        quantize_llama_layer(
            layer=model.model.layers[i],
            idx=i,
            cb=cb,
            args=args,
            device=device,
            pre_orig_emb=embedding_cache["input_after_quantize"],
            orig_emb=embedding_cache["output_after_quantize"],
            hessian_data=hessian_data,
            position_embeding=position_embeddings
        )
        # 获取量化后的输出
        model.model.layers[i].to(device, dtype=torch.float16)
        pbar = tqdm.tqdm(range(args.devset_size // args.batch_size), desc="getting output after quantization")
        with torch.no_grad():
            for j in pbar:
                result = model.model.layers[i](
                    embedding_cache["input_after_quantize"][args.batch_size*j:args.batch_size*(j + 1)].to(device),
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False
                )
                embedding_cache["output_after_quantize"][args.batch_size*j:args.batch_size*(j + 1)] = result[0].cpu()

        embedding_cache['input_after_quantize'] = embedding_cache['output_after_quantize'].detach().clone()
        embedding_cache['input'] = embedding_cache['output'].detach().clone()

        model.model.layers[i].cpu()
        utils.clean()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
