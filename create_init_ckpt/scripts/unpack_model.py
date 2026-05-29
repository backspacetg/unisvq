import os
import sys
import tqdm
import argparse
recur_level=2
path = __file__
for _ in range(recur_level):
    path = os.path.dirname(path)
print(path)
sys.path.append(path)

import torch
from transformers import AutoTokenizer

from lib import codebook
from lib.utils import matmul_hadU_cuda
from model.general_model import load_quantized_model, find_layers, set_op_by_name, QuantizedLinear

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str)
parser.add_argument("--output_path", type=str)

torch.set_grad_enabled(False)


def unpack_linear(linear: QuantizedLinear):
    in_features = linear.in_features
    out_features = linear.out_features
    new_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=linear.has_bias)
    dev = linear.Qidxs.device
    new_linear.to(dev)
    cb = codebook.get_quantized_class(linear.codebook_id.item())(dev)
    split_qidxs = cb.maybe_unpack_idxs(linear.Qidxs.cpu())
    Qidxs_list = []
    for i in range(len(split_qidxs)):
        Qidxs_list.append(split_qidxs[i].to(dev))
    W_decompressed = cb.codebook.by_idxs(Qidxs_list[0]).float() / cb.scale
    W_decompressed = matmul_hadU_cuda(W_decompressed, linear.had_left_T.T.contiguous(), linear.K_left)
    W_decompressed = matmul_hadU_cuda(W_decompressed.T, linear.had_right, linear.K_right).T
    W_decompressed = linear.Wscale * linear.SV.unsqueeze(1) * W_decompressed * linear.SU.unsqueeze(0)
    
    if linear.scaleWH:
        W_decompressed = W_decompressed / linear.scaleWH
    W_decompressed = W_decompressed * cb.scale

    new_linear.weight.data = W_decompressed.to(device=new_linear.weight.device, dtype=new_linear.weight.dtype)
    if linear.has_bias:
        new_linear.bias.data = linear.bias.to(device=new_linear.bias.device, dtype=new_linear.bias.dtype)
    return new_linear


if __name__ == "__main__":
    args = parser.parse_args()
    model = load_quantized_model(args.input_path)
    tokenizer = AutoTokenizer.from_pretrained(args.input_path, trust_remote_code=True)

    pbar = tqdm.tqdm(model.model.layers)
    
    for layer_id, layer in enumerate(pbar):
        pbar.set_description_str(f"unpacking {layer_id}")
        linears = find_layers(layer, layers=[QuantizedLinear])
        
        for name, q_linear in linears.items():
            x = torch.randn(2, 128, q_linear.in_features).to(device="cuda")
            new_linear = unpack_linear(q_linear)
            y_q = q_linear(x)
            y_unp = new_linear(x)
            pbar.write("name: {}, err: {:.4f}".format(name, (torch.norm(y_q-y_unp)/torch.norm(y_q).item())))
            set_op_by_name(layer, name, new_linear)
    
    model.to(dtype=torch.bfloat16)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)