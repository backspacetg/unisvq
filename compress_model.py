import os
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from model import MegaTVQInferForCausalLM

input_model = "/data/groups/QY_LLM_Other/wanghaoyu/exp/9g_7b_2bit_tvq"
output_path = "/data/groups/QY_LLM_Other/wanghaoyu/exp/9g_7b_2bit_tvq_packed/"

model = MegaTVQInferForCausalLM.from_pretrained(
    input_model, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True)
model.save_pretrained(output_path)

embedding_path = "/data/groups/QY_LLM_Other/wanghaoyu/pretrained_models/9g_7b"
new_dict = {}
ori_shapes = {}
with safe_open(os.path.join(output_path, "model.safetensors"), framework="pt") as f:
    for key in f.keys():
        if "lm_head.weight" in key or "embed_tokens.weight" in key:
            print("save embeddings")
            com_tensors = torch.load(os.path.join(embedding_path, f"{key}.pt"), weights_only=False)
            for k, v in com_tensors.items():
                new_dict[k] = v
            ori_shapes[key] = f.get_tensor(key).shape
        else:
            new_dict[key] = f.get_tensor(key)
    meta = f.metadata()

save_file(
    new_dict,
    os.path.join(output_path, "model.safetensors"),
    metadata=meta
)

with open(os.path.join(output_path, "config.json")) as f:
    config = json.load(f)
for target_key, ori_shape in ori_shapes.items():
    config["original_shapes"][target_key] = ori_shape
with open(os.path.join(output_path, "config.json"), 'w') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

tokenizer = AutoTokenizer.from_pretrained(input_model, trust_remote_code=True)
tokenizer.save_pretrained(output_path)