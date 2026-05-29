import os
import argparse

import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from lib import codebook, utils
from lib.utils.model_version import MODEL_VERSION
from model.general_model import format_quantize_model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--hf_output_path', type=str)


def main(args):
    assert os.path.exists(args.quantized_path)
    base_config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'))
    model_config = saved_config['model_config']
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype='auto', low_cpu_mem_usage=True, config=model_config, trust_remote_code=True).half()
    codebook_id = codebook.get_id(model_config.quip_params['codebook'])
    codesz = model_config.quip_params['codesz']
    # print("id", codebook_id)
    # print(saved_config)
    # exit(0)
    model_config.quip_params['model_version'] = MODEL_VERSION
   
    model = format_quantize_model(model, model_config)

    cpu = torch.device('cpu')
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt', map_location=cpu)
        model.lm_head.weight.copy_(lmhead_data['lm_head'])
        model.model.norm.weight.copy_(lmhead_data['norm'])

    pbar = tqdm.tqdm(range(len(model.model.layers)))
    for layer_index in pbar:
        layer = model.model.layers[layer_index]

        if os.path.exists(f'{args.quantized_path}/{layer_index}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{layer_index}_layernorm.pt',map_location=cpu)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'])
            layer.post_attention_layernorm.weight.copy_(ln_data['post_attention_layernorm'])

        for attn_linear in ['q', 'k', 'v', 'o']:
            try: 
                saved_layer = torch.load(f'{args.quantized_path}/{layer_index}_self_attn.{attn_linear}_proj.pt',map_location=cpu)
                utils.unpack_quip(getattr(layer.self_attn, f"{attn_linear}_proj"), saved_layer, codebook_id, codesz)
            except:
                print("exception occured")
                print(f'{args.quantized_path}/{layer_index}_self_attn.{attn_linear}_proj.pt')
                print(saved_layer.keys())
                exit(0)

        for mlp_linear in ['up', 'gate', 'down']:
            saved_layer = torch.load(f'{args.quantized_path}/{layer_index}_mlp.{mlp_linear}_proj.pt',map_location=cpu)
            utils.unpack_quip(getattr(layer.mlp, f"{mlp_linear}_proj"), saved_layer, codebook_id, codesz, load_tuneable=("tuneable" in model_config.quip_params["codebook"]))

        pbar.write(f'loaded layer {layer_index} done')

    pbar.write(f'saving model...')
    try: 
        model.save_pretrained(args.hf_output_path, safe_serialization=True)
    except:
        model.save_pretrained(args.hf_output_path, safe_serialization=False)

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.hf_output_path)



if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
